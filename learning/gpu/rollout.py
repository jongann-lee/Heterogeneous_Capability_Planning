"""Batched tensorized episodic rollout on CUDA."""

from dataclasses import dataclass

import torch


@dataclass
class TensorRollout:
    returns: torch.Tensor
    makespans: torch.Tensor
    completed: torch.Tensor
    deaths: torch.Tensor
    remaining_targets: torch.Tensor
    log_probabilities: torch.Tensor
    entropies: torch.Tensor
    events: int


def collect_tensor_episodes(model, state, observation_builder,
                            death_penalty=100.0, incomplete_penalty=1000.0,
                            max_events=100000, training=True):
    """Roll out a batch while retaining the differentiable policy trace."""
    log_probabilities = []
    entropies = []
    events = 0
    while events < max_events:
        unfinished = ~state.completed()
        if not unfinished.any():
            break
        (observation, distances, predecessors, target_distances,
         target_entries, candidate_entries) = observation_builder.build(state)
        model.train(training)
        with torch.set_grad_enabled(training):
            decoded = model.decode(observation, training=training)
        log_probabilities.append(decoded.log_probabilities)
        entropies.append(decoded.entropies)

        action_indices = torch.full_like(state.positions, -1)
        for b, assignments in enumerate(decoded.assignments):
            for agent, action in assignments:
                action_indices[b, agent] = action
        assigned = action_indices >= 0
        safe_actions = action_indices.clamp_min(0)
        physical = observation_builder.world.candidate_nodes[safe_actions] >= 0
        at_node = assigned & ~state.moving & state.alive & unfinished[:, None]
        state.goal_nodes = torch.where(
            at_node & physical,
            observation_builder.world.candidate_nodes[safe_actions],
            state.goal_nodes)
        next_hops, reachable = observation_builder.next_hops(
            state, safe_actions, distances, predecessors,
            target_distances, target_entries, candidate_entries)
        state.dispatch_next_hops(next_hops, at_node & reachable)
        active, _arriving = state.advance()
        events += 1
        if not active.any():
            break

    remaining = state.target_live.sum(dim=1)
    returns = (-state.clock
               - death_penalty * state.deaths.float()
               - incomplete_penalty * remaining.float())
    parameter = next(model.parameters())
    zero = parameter.sum() * 0.0
    joint_logp = (torch.stack(log_probabilities).sum(dim=0)
                  if log_probabilities else zero.expand(state.batch_size))
    joint_entropy = (torch.stack(entropies).sum(dim=0)
                     if entropies else zero.expand(state.batch_size))
    return TensorRollout(
        returns, state.clock, state.completed(), state.deaths, remaining,
        joint_logp, joint_entropy, events)
