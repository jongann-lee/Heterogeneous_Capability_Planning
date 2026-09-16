"""Autoregressive constrained assignment decoder."""

from dataclasses import dataclass

import torch
from torch.distributions import Categorical


@dataclass
class DecoderOutput:
    assignments: list[list[tuple[int, int]]]
    selected_group_indices: list[list[int]]
    log_probabilities: torch.Tensor
    entropies: torch.Tensor
    action_indices: torch.Tensor | None = None

    @property
    def selected_pair_indices(self):
        """Compatibility alias for pre-grouped rollout code."""
        return self.selected_group_indices


class AssignmentDecoder:
    """Select agent/physical-location groups from semantic action logits."""

    @staticmethod
    def _resolve_group_metadata(
            pair_logits, capacities, candidate_physical_group,
            physical_group_capacity, physical_group_representative,
            physical_group_mask):
        batch, _num_agents, num_actions = pair_logits.shape
        device = pair_logits.device
        if candidate_physical_group is None:
            if capacities is None:
                raise ValueError(
                    "action capacities or physical-group metadata are required")
            groups = torch.arange(
                num_actions, dtype=torch.long, device=device)[None].expand(
                    batch, -1)
            representatives = groups
            group_capacity = capacities
            group_mask = torch.ones_like(groups, dtype=torch.bool)
            return groups, group_capacity, representatives, group_mask

        if (physical_group_capacity is None
                or physical_group_representative is None):
            raise ValueError(
                "physical group capacity and representative metadata are required")
        groups = candidate_physical_group
        group_capacity = physical_group_capacity
        representatives = physical_group_representative
        group_mask = (
            torch.ones_like(group_capacity, dtype=torch.bool)
            if physical_group_mask is None else physical_group_mask)
        if groups.shape != (batch, num_actions):
            raise ValueError(
                "candidate_physical_group must have shape [batch, actions]")
        if group_capacity.shape != representatives.shape:
            raise ValueError(
                "physical group capacities and representatives must align")
        if group_capacity.shape[0] != batch or group_mask.shape != group_capacity.shape:
            raise ValueError("physical group metadata must be batch-aligned")
        return groups, group_capacity, representatives, group_mask

    @staticmethod
    def grouped_logits(pair_logits, valid, candidate_physical_group,
                       physical_group_mask):
        """Aggregate semantic aliases with a masked differentiable logsumexp."""
        num_groups = physical_group_mask.shape[-1]
        group_index = candidate_physical_group.unsqueeze(-2).expand_as(pair_logits)
        semantic_logits = pair_logits.masked_fill(~valid, -torch.inf)
        maxima = pair_logits.new_full(
            (*pair_logits.shape[:-1], num_groups), -torch.inf)
        maxima.scatter_reduce_(
            -1, group_index, semantic_logits, reduce="amax",
            include_self=True)
        # The maximum is only a numerical-stability offset. Detaching it keeps
        # the exact logsumexp gradient while avoiding scatter max tie details.
        gathered_maxima = maxima.detach().gather(-1, group_index)
        usable = valid & torch.isfinite(gathered_maxima)
        safe_semantic_logits = torch.where(
            usable, semantic_logits, torch.zeros_like(semantic_logits))
        safe_gathered_maxima = torch.where(
            usable, gathered_maxima, torch.zeros_like(gathered_maxima))
        offsets = safe_semantic_logits - safe_gathered_maxima
        exponentials = torch.exp(offsets) * usable
        sums = torch.zeros_like(maxima)
        sums.scatter_add_(-1, group_index, exponentials)
        has_valid = sums > 0
        result = torch.where(
            has_valid,
            maxima.detach() + torch.log(sums.clamp_min(
                torch.finfo(pair_logits.dtype).tiny)),
            pair_logits.new_full((), -torch.inf))
        return result.masked_fill(~physical_group_mask.unsqueeze(-2), -torch.inf)

    @staticmethod
    def _distribution(group_logits, available_agents, remaining):
        valid = (torch.isfinite(group_logits) & available_agents[..., None]
                 & (remaining > 0)[:, None])
        active = valid.flatten(1).any(dim=1)
        logits = group_logits.masked_fill(~valid, -torch.inf).flatten(1)
        # Finished/padded episodes get a harmless dummy distribution. Mask its
        # contribution below; no GPU scalar read is needed to stop those rows.
        logits = torch.where(active[:, None], logits, torch.zeros_like(logits))
        return Categorical(logits=logits, validate_args=False), active

    def __call__(self, pair_logits, feasible_mask, capacities,
                 training: bool = False, generator=None,
                 candidate_physical_group=None,
                 physical_group_capacity=None,
                 physical_group_representative=None,
                 physical_group_mask=None) -> DecoderOutput:
        batch, num_agents, _num_actions = pair_logits.shape
        (candidate_groups, group_capacities, group_representatives,
         group_masks) = self._resolve_group_metadata(
             pair_logits, capacities, candidate_physical_group,
             physical_group_capacity, physical_group_representative,
             physical_group_mask)
        action_indices = torch.full(
            (batch, num_agents), -1, dtype=torch.long, device=pair_logits.device)
        logp = pair_logits.new_zeros(batch)
        entropy = pair_logits.new_zeros(batch)
        num_groups = group_capacities.shape[1]
        if not num_agents or not num_groups:
            return DecoderOutput([[] for _ in range(batch)],
                                 [[] for _ in range(batch)], logp, entropy,
                                 action_indices)
        # Only whole agents and physical groups are removed during assignment,
        # so semantic marginalization is invariant across autoregressive steps.
        group_logits = self.grouped_logits(
            pair_logits, feasible_mask, candidate_groups, group_masks)
        available = torch.ones((batch, num_agents), dtype=torch.bool,
                               device=pair_logits.device)
        remaining = group_capacities.clone()
        rows = torch.arange(batch, device=pair_logits.device)
        records = []
        for _ in range(num_agents):
            distribution, active = self._distribution(
                group_logits, available, remaining)
            selected = (torch.multinomial(distribution.probs, 1,
                                          generator=generator).squeeze(1)
                        if training else distribution.logits.argmax(dim=1))
            agent = selected // num_groups
            group = selected % num_groups
            action = group_representatives[rows, group]
            records.append(torch.stack((
                agent, action, selected.masked_fill(~active, -1)), dim=1))
            action_indices[rows, agent] = torch.where(
                active, action, action_indices[rows, agent])
            logp = logp + distribution.log_prob(selected) * active
            entropy = entropy + distribution.entropy() * active
            available[rows, agent] &= ~active
            remaining[rows, group] -= (
                active & (remaining[rows, group] < torch.iinfo(remaining.dtype).max)
            ).to(remaining.dtype)
        # One bulk transfer preserves the legacy Python API. Tensor simulation
        # uses action_indices directly instead of copying each action back.
        host_records = torch.stack(records, dim=1).detach().cpu().tolist()
        assignments = [[(agent, action) for agent, action, index in row
                        if index >= 0] for row in host_records]
        flat_indices = [[index for _agent, _action, index in row if index >= 0]
                        for row in host_records]
        return DecoderOutput(assignments, flat_indices, logp, entropy,
                             action_indices)

    def evaluate_selected(self, pair_logits, feasible_mask, capacities,
                          selected_pair_indices,
                          candidate_physical_group=None,
                          physical_group_capacity=None,
                          physical_group_representative=None,
                          physical_group_mask=None):
        """Evaluate recorded autoregressive choices without resampling."""
        batch, num_agents, _num_actions = pair_logits.shape
        (candidate_groups, group_capacities, _group_representatives,
         group_masks) = self._resolve_group_metadata(
             pair_logits, capacities, candidate_physical_group,
             physical_group_capacity, physical_group_representative,
             physical_group_mask)
        logp = pair_logits.new_zeros(batch)
        entropy = pair_logits.new_zeros(batch)
        steps = max(map(len, selected_pair_indices), default=0)
        if not steps:
            return logp, entropy
        selected_indices = torch.tensor(
            [list(indices) + [-1] * (steps - len(indices))
             for indices in selected_pair_indices],
            dtype=torch.long, device=pair_logits.device)
        group_logits = self.grouped_logits(
            pair_logits, feasible_mask, candidate_groups, group_masks)
        available = torch.ones((batch, num_agents), dtype=torch.bool,
                               device=pair_logits.device)
        remaining = group_capacities.clone()
        num_groups = remaining.shape[1]
        rows = torch.arange(batch, device=pair_logits.device)
        for step in range(steps):
            active = selected_indices[:, step] >= 0
            selected = selected_indices[:, step].clamp_min(0)
            # Rows with shorter traces may still have feasible actions. Give
            # those rows dummy distributions too, avoiding masked 0 * -inf.
            distribution, _ = self._distribution(
                group_logits, available & active[:, None], remaining)
            logp = logp + distribution.log_prob(selected) * active
            entropy = entropy + distribution.entropy() * active
            agent = selected // num_groups
            group = selected % num_groups
            available[rows, agent] &= ~active
            remaining[rows, group] -= (
                active & (remaining[rows, group] < torch.iinfo(remaining.dtype).max)
            ).to(remaining.dtype)
        return logp, entropy
