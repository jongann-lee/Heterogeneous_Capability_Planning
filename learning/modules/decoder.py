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
        num_agents = pair_logits.shape[0]
        num_groups = physical_group_mask.numel()
        group_index = candidate_physical_group[None].expand(num_agents, -1)
        semantic_logits = pair_logits.masked_fill(~valid, -torch.inf)
        maxima = pair_logits.new_full((num_agents, num_groups), -torch.inf)
        maxima.scatter_reduce_(
            1, group_index, semantic_logits, reduce="amax",
            include_self=True)
        # The maximum is only a numerical-stability offset. Detaching it keeps
        # the exact logsumexp gradient while avoiding scatter max tie details.
        gathered_maxima = maxima.detach().gather(1, group_index)
        usable = valid & torch.isfinite(gathered_maxima)
        safe_semantic_logits = torch.where(
            usable, semantic_logits, torch.zeros_like(semantic_logits))
        safe_gathered_maxima = torch.where(
            usable, gathered_maxima, torch.zeros_like(gathered_maxima))
        offsets = safe_semantic_logits - safe_gathered_maxima
        exponentials = torch.exp(offsets) * usable
        sums = pair_logits.new_zeros((num_agents, num_groups))
        sums.scatter_add_(1, group_index, exponentials)
        has_valid = sums > 0
        result = torch.where(
            has_valid,
            maxima.detach() + torch.log(sums.clamp_min(
                torch.finfo(pair_logits.dtype).tiny)),
            pair_logits.new_full((), -torch.inf))
        return result.masked_fill(~physical_group_mask[None], -torch.inf)

    def __call__(self, pair_logits, feasible_mask, capacities,
                 training: bool = False, generator=None,
                 candidate_physical_group=None,
                 physical_group_capacity=None,
                 physical_group_representative=None,
                 physical_group_mask=None) -> DecoderOutput:
        batch, _num_agents, _num_actions = pair_logits.shape
        (candidate_groups, group_capacities, group_representatives,
         group_masks) = self._resolve_group_metadata(
             pair_logits, capacities, candidate_physical_group,
             physical_group_capacity, physical_group_representative,
             physical_group_mask)
        assignments, flat_indices = [], []
        joint_logps, joint_entropies = [], []
        for b in range(batch):
            valid = feasible_mask[b].clone()
            remaining = group_capacities[b].clone()
            num_groups = remaining.numel()
            chosen, chosen_flat = [], []
            logp = pair_logits.new_zeros(())
            entropy = pair_logits.new_zeros(())
            while valid.any():
                step_logits = self.grouped_logits(
                    pair_logits[b], valid, candidate_groups[b],
                    group_masks[b])
                step_valid = (
                    torch.isfinite(step_logits)
                    & (remaining > 0).unsqueeze(0)
                    & group_masks[b].unsqueeze(0))
                if not step_valid.any():
                    break
                flat_logits = step_logits.flatten().masked_fill(
                    ~step_valid.flatten(), -torch.inf)
                distribution = Categorical(logits=flat_logits)
                if training:
                    # Categorical.sample has no generator argument in PyTorch.
                    selected = distribution.sample()
                else:
                    selected = flat_logits.argmax()
                agent = int(selected.item()) // num_groups
                group = int(selected.item()) % num_groups
                action = int(group_representatives[b, group].item())
                chosen.append((agent, action))
                chosen_flat.append(int(selected.item()))
                logp = logp + distribution.log_prob(selected)
                entropy = entropy + distribution.entropy()
                valid[agent, :] = False
                remaining[group] -= (
                    remaining[group] < torch.iinfo(remaining.dtype).max).to(
                        remaining.dtype)
            assignments.append(chosen)
            flat_indices.append(chosen_flat)
            joint_logps.append(logp)
            joint_entropies.append(entropy)
        return DecoderOutput(assignments, flat_indices,
                             torch.stack(joint_logps),
                             torch.stack(joint_entropies))

    def evaluate_selected(self, pair_logits, feasible_mask, capacities,
                          selected_pair_indices,
                          candidate_physical_group=None,
                          physical_group_capacity=None,
                          physical_group_representative=None,
                          physical_group_mask=None):
        """Evaluate recorded autoregressive choices without resampling."""
        batch, _num_agents, _num_actions = pair_logits.shape
        (candidate_groups, group_capacities, _group_representatives,
         group_masks) = self._resolve_group_metadata(
             pair_logits, capacities, candidate_physical_group,
             physical_group_capacity, physical_group_representative,
             physical_group_mask)
        joint_logps, joint_entropies = [], []
        for b in range(batch):
            valid = feasible_mask[b].clone()
            remaining = group_capacities[b].clone()
            num_groups = remaining.numel()
            logp = pair_logits.new_zeros(())
            entropy = pair_logits.new_zeros(())
            for flat_index in selected_pair_indices[b]:
                step_logits = self.grouped_logits(
                    pair_logits[b], valid, candidate_groups[b],
                    group_masks[b])
                step_valid = (
                    torch.isfinite(step_logits)
                    & (remaining > 0).unsqueeze(0)
                    & group_masks[b].unsqueeze(0))
                flat_logits = step_logits.flatten().masked_fill(
                    ~step_valid.flatten(), -torch.inf)
                distribution = Categorical(logits=flat_logits)
                selected = torch.as_tensor(
                    flat_index, device=pair_logits.device)
                logp = logp + distribution.log_prob(selected)
                entropy = entropy + distribution.entropy()
                agent = int(flat_index) // num_groups
                group = int(flat_index) % num_groups
                valid[agent, :] = False
                remaining[group] -= (
                    remaining[group] < torch.iinfo(remaining.dtype).max).to(
                        remaining.dtype)
            joint_logps.append(logp)
            joint_entropies.append(entropy)
        return torch.stack(joint_logps), torch.stack(joint_entropies)
