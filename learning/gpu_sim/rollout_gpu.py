"""Batched tensorized GPU episodic rollout."""

from dataclasses import dataclass

import torch


@dataclass
class DecisionTrace:
    observation: object
    selected_pair_indices: list[list[int]]


@dataclass
class TensorRollout:
    returns: torch.Tensor
    makespans: torch.Tensor
    completed: torch.Tensor
    deaths: torch.Tensor
    remaining_targets: torch.Tensor
    stalled: torch.Tensor
    all_agents_dead: torch.Tensor
    oracle_makespans: torch.Tensor
    normalized_regrets: torch.Tensor
    decision_traces: list[DecisionTrace]
    events: int


def collect_tensor_episodes(model, state, observation_builder,
                            death_penalty=100.0, incomplete_penalty=1000.0,
                            max_events=100000, training=True,
                            state_callback=None, oracle_makespans=None):
    """Roll out a batch while retaining the differentiable policy trace."""
    decision_traces = []
    events = 0
    while events < max_events:
        unfinished = ~state.completed() & ~state.stalled
        if not unfinished.any():
            break

        planning_agents = (state.needs_replan & state.alive
                           & unfinished[:, None])
        planning_episodes = planning_agents.any(dim=1)
        if planning_episodes.any():
            (observation, distances, predecessors, target_distances,
             target_entries, candidate_entries) = observation_builder.build(
                 state, planning_episodes)
            observation.feasible_action_mask &= planning_agents[..., None]
            model.train(training)
            # Sampling does not need gradients. Observations and selected
            # actions are retained on CPU and replayed after returns are known.
            with torch.no_grad():
                decoded = model.decode(observation, training=training)
            if training:
                decision_traces.append(DecisionTrace(
                    observation.to("cpu"), decoded.selected_pair_indices))

            action_indices = torch.full_like(state.positions, -1)
            for b, assignments in enumerate(decoded.assignments):
                for agent, action in assignments:
                    action_indices[b, agent] = action
            assigned = action_indices >= 0
            safe_actions = action_indices.clamp_min(0)
            physical = (
                observation_builder.world.candidate_nodes[safe_actions] >= 0)
            (next_hops, reachable, route_next,
             destinations) = observation_builder.next_hops(
                 state, safe_actions, distances, predecessors,
                 target_distances, target_entries, candidate_entries)
            committed = planning_agents & assigned & physical & reachable
            cleared_routes = torch.full_like(state.route_next, -1)
            state.route_next = torch.where(
                planning_agents[..., None], cleared_routes, state.route_next)
            state.route_next = torch.where(
                committed[..., None], route_next, state.route_next)
            state.goal_nodes = torch.where(
                committed, destinations, state.goal_nodes)
            planning_positions = torch.where(
                state.moving, state.transit_to, state.positions)
            has_next = route_next.gather(
                2, planning_positions[..., None]).squeeze(2) >= 0
            state.route_active = torch.where(
                planning_agents, committed & has_next, state.route_active)
            state.needs_replan &= ~planning_agents
            del observation, distances, predecessors, target_distances
            del target_entries, candidate_entries, route_next

        rows = torch.arange(state.batch_size,
                            device=state.positions.device)[:, None]
        agents = torch.arange(state.positions.shape[1],
                              device=state.positions.device)[None, :]
        committed_next = state.route_next[
            rows, agents, state.positions].clamp_min(0)
        dispatch = (state.route_active & ~state.moving & state.alive
                    & unfinished[:, None])
        state.dispatch_next_hops(committed_next, dispatch)
        if state_callback is not None:
            state_callback(state)

        # With no moving agent and no outstanding decision, an episode has
        # genuinely stalled (for example, every agent selected wait).
        state.stalled |= (unfinished & ~state.moving.any(dim=1)
                          & ~state.needs_replan.any(dim=1))
        if (~state.completed() & ~state.stalled).logical_not().all():
            break

        active, _arriving, arrived, information_changed = state.advance()
        events += 1
        route_continues = state.route_next[
            rows, agents, state.positions] >= 0
        destination_reached = arrived & ~route_continues & state.alive
        state.route_active &= ~arrived | route_continues

        # Reveals and encounters change the shared belief/task state. Agents
        # finish an already-started edge, but their replacement routes are
        # planned immediately from those committed arrival nodes. Reaching any
        # assigned destination also triggers a new joint team assignment.
        joint_replan = information_changed | destination_reached.any(dim=1)
        state.route_active &= ~joint_replan[:, None]
        state.needs_replan |= joint_replan[:, None] & state.alive
        completed = state.completed()
        state.route_active &= ~completed[:, None] & state.alive
        state.needs_replan &= ~completed[:, None] & state.alive
        if not active.any():
            # The next loop either plans newly-idle agents or marks them stalled.
            continue

    if state_callback is not None:
        state_callback(state)

    remaining = state.target_live.sum(dim=1)
    completed = state.completed()
    all_agents_dead = ~state.alive.any(dim=1)
    stalled = state.stalled | (~completed & ~state.moving.any(dim=1)
                               & ~state.needs_replan.any(dim=1))
    if oracle_makespans is None:
        oracle_makespans = torch.ones_like(state.clock)
        normalized_regrets = state.clock
        returns = (-state.clock
                   - death_penalty * state.deaths.float()
                   - incomplete_penalty * remaining.float())
    else:
        oracle_makespans = torch.as_tensor(
            oracle_makespans, dtype=state.clock.dtype,
            device=state.clock.device).flatten()
        if oracle_makespans.numel() == 1:
            oracle_makespans = oracle_makespans.expand_as(state.clock)
        if oracle_makespans.shape != state.clock.shape:
            raise ValueError("oracle_makespans must be scalar or batch-sized")
        if (oracle_makespans <= 0).any():
            raise ValueError("oracle_makespans must be positive")
        normalized_regrets = state.clock / oracle_makespans - 1.0
        returns = (-normalized_regrets
                   - death_penalty * state.deaths.float()
                   - incomplete_penalty * remaining.float())
    return TensorRollout(
        returns, state.clock, completed, state.deaths, remaining,
        stalled, all_agents_dead, oracle_makespans, normalized_regrets,
        decision_traces, events)


def replay_tensor_gradients(model, rollout, advantages, entropy_coefficient,
                            update_size, device):
    """Replay decisions using normalized, full-update advantages.

    Each episode is averaged over the number of times the policy was queried.
    Makespan therefore affects the return, but a long trajectory does not also
    receive an incidental multiplier merely because it contains more events.
    """
    advantages = advantages.detach().to(device)
    decision_counts = torch.zeros_like(advantages)
    for trace in rollout.decision_traces:
        decision_counts += torch.tensor(
            [bool(selected) for selected in trace.selected_pair_indices],
            dtype=advantages.dtype, device=device)
    loss_divisors = decision_counts.clamp_min(1.0)
    totals = torch.zeros_like(advantages)
    for trace in rollout.decision_traces:
        observation = trace.observation.to(device)
        logits = model(observation)
        logp, entropy = model.decoder.evaluate_selected(
            logits, observation.feasible_action_mask,
            observation.action_capacities, trace.selected_pair_indices)
        losses = (-(advantages * logp) - entropy_coefficient * entropy) \
            / loss_divisors
        (losses.sum() / update_size).backward()
        totals += losses.detach()
        del observation, logits, logp, entropy, losses
    return totals, decision_counts
