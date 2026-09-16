"""Batched tensorized GPU episodic rollout."""

from dataclasses import dataclass, fields
from contextlib import nullcontext
from time import perf_counter

import torch


@dataclass
class DecisionTrace:
    observation: object
    selected_group_indices: list[list[int]]

    @property
    def selected_pair_indices(self):
        """Compatibility alias for checkpoints/tests using the old name."""
        return self.selected_group_indices


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
    target_counts: torch.Tensor
    decision_traces: list[DecisionTrace]
    events: int
    planning_times_seconds: list[float] | None = None
    planning_call_counts: list[int] | None = None


@torch.no_grad()
def collect_tensor_episodes(model, state, observation_builder,
                            death_penalty=100.0, incomplete_penalty=1000.0,
                            max_events=100000, training=True,
                            state_callback=None, oracle_makespans=None,
                            diagnostics=None, measure_planning_time=False):
    """Roll out without autograd; retain only CPU observations for replay."""
    decision_traces = []
    events = 0
    planning_times = ([0.0] * state.batch_size
                      if measure_planning_time else None)
    planning_counts = ([0] * state.batch_size
                       if measure_planning_time else None)
    phase = diagnostics.phase if diagnostics is not None else lambda _: nullcontext()
    if model is not None:
        model.train(training)
    while events < max_events:
        unfinished = ~state.completed() & ~state.stalled
        if not unfinished.any():
            break

        planning_agents = (state.needs_replan & state.alive
                           & unfinished[:, None])
        planning_episodes = planning_agents.any(dim=1)
        if planning_episodes.any():
            if measure_planning_time:
                # A batched decision has one shared wall-clock latency. Charge
                # it to each active episode, then average per episode later.
                planning_rows = planning_episodes.tolist()
                if state.positions.device.type == "cuda":
                    torch.cuda.synchronize(state.positions.device)
                planning_start = perf_counter()
            with phase("observation_and_routing"):
                (observation, distances, predecessors, target_distances,
                 target_entries, candidate_entries) = observation_builder.build(
                     state, planning_episodes)
            observation.feasible_action_mask &= planning_agents[..., None]
            # Sampling does not need gradients. Observations and selected
            # actions are retained on CPU and replayed after returns are known.
            with phase("policy_decode"):
                decoded = model.decode(observation, training=training)
            if training:
                with phase("trace_to_cpu"):
                    decision_traces.append(DecisionTrace(
                        observation.to("cpu"), decoded.selected_group_indices))

            action_indices = getattr(decoded, "action_indices", None)
            if action_indices is None:
                # Compatibility with custom/testing policies.
                action_indices = torch.full_like(state.positions, -1)
                for b, assignments in enumerate(decoded.assignments):
                    for agent, action in assignments:
                        action_indices[b, agent] = action
            assigned = action_indices >= 0
            safe_actions = action_indices.clamp_min(0)
            physical = (
                observation_builder.world.candidate_nodes[safe_actions] >= 0)
            with phase("route_reconstruction"):
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
            if measure_planning_time:
                if state.positions.device.type == "cuda":
                    torch.cuda.synchronize(state.positions.device)
                elapsed = perf_counter() - planning_start
                for row, active in enumerate(planning_rows):
                    if active:
                        planning_times[row] += elapsed
                        planning_counts[row] += 1

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
        if (oracle_makespans < 0).any():
            raise ValueError("oracle_makespans must be nonnegative")
        if state.target_live.shape[1] > 0 and (oracle_makespans == 0).any():
            raise ValueError("zero oracle makespans require targetless episodes")
        safe_oracles = torch.where(
            oracle_makespans > 0, oracle_makespans,
            torch.ones_like(oracle_makespans))
        normalized_regrets = torch.where(
            oracle_makespans > 0,
            state.clock / safe_oracles - 1.0,
            torch.zeros_like(state.clock))
        returns = (-normalized_regrets
                   - death_penalty * state.deaths.float()
                   - incomplete_penalty * remaining.float())
    return TensorRollout(
        returns, state.clock, completed, state.deaths, remaining,
        stalled, all_agents_dead, oracle_makespans, normalized_regrets,
        torch.full_like(remaining, state.target_live.shape[1]),
        decision_traces, events, planning_times, planning_counts)


def _replay_observation_batch(traces, batch_size, start, end):
    """Concatenate bounded slices of CPU traces without padding new entities."""
    parts, selected = [], []
    while start < end:
        trace = traces[start // batch_size]
        row = start % batch_size
        count = min(end - start, batch_size - row)
        selection = slice(row, row + count)
        parts.append((trace.observation, selection))
        selected.extend(trace.selected_group_indices[selection])
        start += count
    first = parts[0][0]
    values = {}
    for field in fields(first):
        value = getattr(first, field.name)
        if torch.is_tensor(value):
            tensors = [getattr(observation, field.name)[selection]
                       for observation, selection in parts]
            value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        elif isinstance(value, list):
            value = [item for observation, selection in parts
                     for item in getattr(observation, field.name)[selection]]
        values[field.name] = value
    return type(first)(**values), selected


def replay_tensor_gradients(model, rollout, episode_signals,
                            entropy_coefficient, update_size, device,
                            critic_coefficient=0.5, replay_batch_size=64):
    """Replay decisions using EMA advantages or graph-critic residuals.

    Each episode is averaged over the number of times the policy was queried.
    Makespan therefore affects the return, but a long trajectory does not also
    receive an incidental multiplier merely because it contains more events.
    """
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    episode_signals = episode_signals.detach().to(device)
    batch_size = episode_signals.numel()
    decision_counts = torch.tensor([
        sum(bool(trace.selected_group_indices[row])
            for trace in rollout.decision_traces)
        for row in range(batch_size)],
        dtype=episode_signals.dtype, device=device)
    loss_divisors = decision_counts.clamp_min(1.0)
    totals = torch.zeros_like(episode_signals)
    critic_totals = torch.zeros_like(episode_signals)
    entropy_totals = torch.zeros_like(episode_signals)
    value_totals = torch.zeros_like(episode_signals)
    examples = batch_size * len(rollout.decision_traces)
    for start in range(0, examples, replay_batch_size):
        end = min(start + replay_batch_size, examples)
        observation, selected = _replay_observation_batch(
            rollout.decision_traces, batch_size, start, end)
        if not any(selected):
            continue
        observation = observation.to(device)
        rows = torch.arange(start, end, device=device) % batch_size
        signals = episode_signals[rows]
        divisors = loss_divisors[rows]
        logits, values = model.actor_critic(observation)
        logp, entropy = model.decoder.evaluate_selected(
            logits, observation.feasible_action_mask,
            observation.action_capacities, selected,
            candidate_physical_group=(
                observation.candidate_physical_group),
            physical_group_capacity=observation.physical_group_capacity,
            physical_group_representative=(
                observation.physical_group_representative),
            physical_group_mask=observation.physical_group_mask)
        active = torch.tensor(
            [bool(indices) for indices in selected],
            dtype=episode_signals.dtype, device=device)
        if values is None:
            advantages = signals
            critic_losses = torch.zeros_like(signals)
        else:
            advantages = signals - values.detach()
            critic_losses = (values - signals).square() * active
            value_totals.index_add_(0, rows, values.detach() * active / divisors)
        actor_losses = (-(advantages * logp)
                        - entropy_coefficient * entropy) * active
        losses = (actor_losses + critic_coefficient * critic_losses) \
            / divisors
        (losses.sum() / update_size).backward()
        totals.index_add_(0, rows, losses.detach())
        critic_totals.index_add_(0, rows, critic_losses.detach() / divisors)
        entropy_totals.index_add_(0, rows, entropy.detach() * active / divisors)
        del observation, logits, logp, entropy, losses
    return (totals, decision_counts, critic_totals,
            entropy_totals, value_totals)
