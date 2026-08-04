"""Autoregressive constrained centralized assignment decoder."""

from dataclasses import dataclass

import torch
from torch.distributions import Categorical


@dataclass
class DecoderOutput:
    assignments: list[list[tuple[int, int]]]
    selected_pair_indices: list[list[int]]
    log_probabilities: torch.Tensor
    entropies: torch.Tensor


class AssignmentDecoder:
    """Sequentially select pairs while updating agent/action availability."""

    def __call__(self, pair_logits, feasible_mask, capacities,
                 training: bool = False, generator=None) -> DecoderOutput:
        batch, num_agents, num_actions = pair_logits.shape
        assignments, flat_indices = [], []
        joint_logps, joint_entropies = [], []
        for b in range(batch):
            valid = feasible_mask[b].clone()
            remaining = capacities[b].clone()
            chosen, chosen_flat = [], []
            logp = pair_logits.new_zeros(())
            entropy = pair_logits.new_zeros(())
            while valid.any():
                capacity_ok = (remaining > 0).unsqueeze(0)
                step_valid = valid & capacity_ok
                if not step_valid.any():
                    break
                flat_logits = pair_logits[b].flatten().masked_fill(
                    ~step_valid.flatten(), -torch.inf)
                distribution = Categorical(logits=flat_logits)
                if training:
                    # Categorical.sample has no generator argument in PyTorch.
                    selected = distribution.sample()
                else:
                    selected = flat_logits.argmax()
                agent = int(selected.item()) // num_actions
                action = int(selected.item()) % num_actions
                chosen.append((agent, action))
                chosen_flat.append(int(selected.item()))
                logp = logp + distribution.log_prob(selected)
                entropy = entropy + distribution.entropy()
                valid[agent, :] = False
                if remaining[action] < torch.iinfo(remaining.dtype).max:
                    remaining[action] -= 1
                if remaining[action] <= 0:
                    valid[:, action] = False
            assignments.append(chosen)
            flat_indices.append(chosen_flat)
            joint_logps.append(logp)
            joint_entropies.append(entropy)
        return DecoderOutput(assignments, flat_indices,
                             torch.stack(joint_logps),
                             torch.stack(joint_entropies))
