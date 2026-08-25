"""GPU-native observation construction from batched tensor episode state."""

import torch

from learning.gpu_sim.observation_cpu import (
    PlannerObservation,
    attach_task_graph_fields,
    feature_dimensions,
)


class TensorObservationBuilder:
    def __init__(self, world, num_target_types, task_graph=True):
        self.world = world
        self.num_target_types = int(num_target_types)
        self.task_graph = bool(task_graph)

    def candidate_roles(self, state):
        live = state.target_live
        unknown = live & ~state.target_known
        target_links = self.world.target_candidate_mask[None] & live[:, None]
        observed_links = self.world.candidate_observed_mask[None] & unknown[:, None]
        staging_links = self.world.candidate_staging_mask[None] & unknown[:, None]
        is_target = target_links.any(dim=2)
        is_observation = observed_links.any(dim=2)
        is_staging = staging_links.any(dim=2)
        active = (is_target | is_observation | is_staging |
                  self.world.candidate_is_wait[None])
        associated = target_links | observed_links | staging_links
        return (is_target, is_observation, is_staging, active, associated,
                observed_links, staging_links)

    def _route_banks(self, state, episode_mask=None):
        """One masked SSSP per source plus exact final-edge target recovery."""
        batch, agents = state.positions.shape
        targets = state.target_live.shape[1]
        distances = torch.full(
            (batch, agents, len(self.world.nodes)), torch.inf,
            device=state.positions.device)
        predecessors = torch.full_like(distances, -1, dtype=torch.long)
        if episode_mask is None:
            episode_mask = torch.ones(batch, dtype=torch.bool,
                                      device=state.positions.device)
        for b in range(batch):
            if not bool(episode_mask[b].item()):
                continue
            blocked_nodes = self.world.target_nodes[
                state.target_live[b]].tolist()
            # A moving agent cannot change its current edge, but a shared
            # replan chooses its route beginning at that edge's arrival node.
            sources = torch.where(
                state.moving[b], state.transit_to[b], state.positions[b])
            standard = self.world.router.sssp(
                sources, blocked_nodes, self.world.required_route_nodes)
            distances[b] = standard.distances
            predecessors[b] = standard.predecessors
        incoming = self.world.target_incoming_nodes.clamp_min(0)
        alternatives = (distances[..., incoming]
                        + self.world.target_incoming_costs[None, None])
        target_distances, entry_slots = alternatives.min(dim=3)
        entry_nodes = incoming[None, None].expand(batch, agents, -1, -1).gather(
            3, entry_slots[..., None]).squeeze(3)
        direct = distances[..., self.world.target_nodes]
        target_distances = torch.where(state.target_live[:, None],
                                       target_distances, direct)
        entry_nodes = torch.where(state.target_live[:, None], entry_nodes,
                                  self.world.target_nodes[None, None])
        planning_positions = torch.where(
            state.moving, state.transit_to, state.positions)
        at_goal = planning_positions[..., None] == self.world.target_nodes
        target_distances = torch.where(at_goal, torch.zeros_like(target_distances),
                                       target_distances)
        entry_nodes = torch.where(
            at_goal, planning_positions[..., None], entry_nodes)
        return distances, predecessors, target_distances, entry_nodes

    def _action_target_distances(self, state, episode_mask):
        """Safe directed region-to-target distances for graph relations.

        All other live targets are blocked. The destination target is recovered
        through its incoming edges, matching the agent-target routing contract.
        This is intentionally planner-visible and never reads target truth.
        """
        world = self.world
        device = state.positions.device
        batch = state.batch_size
        actions = len(world.candidates)
        targets = len(world.targets)
        output = torch.full(
            (batch, actions, targets), torch.inf, device=device)
        if episode_mask is None:
            episode_mask = torch.ones(
                batch, dtype=torch.bool, device=device)
        if not world.candidate_region_mask.any():
            return output

        for episode in range(batch):
            if not bool(episode_mask[episode].item()):
                continue
            blocked = world.target_nodes[state.target_live[episode]].tolist()
            for action in range(actions):
                region = world.candidate_region_nodes[action][
                    world.candidate_region_mask[action]]
                if region.numel() == 0:
                    continue
                # A live target used as the action origin has been serviced by
                # the time this relation applies, so it is no longer a blocker.
                region_set = set(region.tolist())
                effective_blocked = [node for node in blocked
                                     if node not in region_set]
                # SSSP from every target on the reversed graph yields directed
                # original-graph distances from this action region to targets.
                routes = world.reverse_router.sssp(
                    world.target_nodes, effective_blocked, region).distances
                output[episode, action] = routes[:, region].amin(dim=1)
        return output

    def build(self, state, planning_episode_mask=None):
        world = self.world
        device = state.positions.device
        batch, agents = state.positions.shape
        targets = len(world.targets)
        actions = len(world.candidates)
        fa, ft, fc, _fat, _fac, _fct = feature_dimensions(self.num_target_types)
        (is_target, is_observation, is_staging, active, associated,
         observed_links, staging_links) = self.candidate_roles(state)
        route_distances, predecessors, target_route_distances, target_entries = (
            self._route_banks(state, planning_episode_mask))

        agent_x = torch.zeros((batch, agents, fa), device=device)
        agent_x[..., :2] = world.positions[state.positions]
        agent_x[..., 2] = state.alive
        agent_x[..., 3] = state.capabilities[..., 0]
        agent_x[..., 4] = ~state.moving
        agent_x[..., 5] = state.moving
        destination = torch.where(state.moving, state.transit_to, state.goal_nodes)
        agent_x[..., 6:8] = world.positions[destination]
        remaining = (state.arrival_time - state.clock[:, None]).clamp_min(0)
        remaining = torch.where(state.moving, remaining, torch.zeros_like(remaining))
        agent_x[..., 8] = remaining / world.distance_scale
        agent_x[..., 9:] = state.capabilities[..., 1:].float()

        target_x = torch.zeros((batch, targets, ft), device=device)
        target_x[..., :2] = world.positions[world.target_nodes][None]
        target_x[..., 2] = state.target_live
        target_x[..., 3] = ~state.target_live
        target_x[..., 4] = state.target_known
        target_x[..., 5] = ~state.target_known
        capable_counts = state.capabilities[..., 1:].sum(dim=1).float() / agents
        known_type_index = (state.target_types - 1).clamp_min(0)
        target_x[..., 7] = capable_counts.gather(1, known_type_index) * state.target_known
        onehot = torch.nn.functional.one_hot(
            known_type_index, self.num_target_types).float()
        target_x[..., 8:] = onehot * state.target_known[..., None]

        action_x = torch.zeros((batch, actions, fc), device=device)
        physical = world.candidate_nodes >= 0
        region_nodes = world.candidate_region_nodes.clamp_min(0)
        region_mask = world.candidate_region_mask
        region_count = region_mask.sum(dim=1).clamp_min(1)
        region_positions = (world.positions[region_nodes] *
                            region_mask[..., None]).sum(dim=1) / region_count[:, None]
        action_x[..., :2] = region_positions[None]
        action_x[..., 2] = is_target
        action_x[..., 3] = is_observation
        action_x[..., 4] = is_staging
        action_x[..., 5] = world.candidate_is_wait
        action_x[..., 6] = associated.sum(dim=2) / max(targets, 1)
        action_x[..., 7] = observed_links.sum(dim=2) / max(targets, 1)
        region_heights = (world.heights[region_nodes] * region_mask).sum(
            dim=1) / region_count
        action_x[..., 8] = region_heights[None]
        action_x[..., 9] = world.candidate_capacity < 0
        action_x[..., 10] = world.candidate_capacity.clamp_min(0)

        # Agent-target paths unblock the goal target while avoiding every other
        # live target, exactly like observation_cpu._safe_path.
        target_nodes = world.target_nodes
        at_distance = target_route_distances
        reachable_at = torch.isfinite(at_distance) & (at_distance < 1e30)
        at_rel = torch.zeros((batch, agents, targets, 6), device=device)
        normalized_at = torch.where(reachable_at, at_distance / world.distance_scale,
                                    torch.zeros_like(at_distance))
        at_rel[..., 0] = normalized_at
        at_rel[..., 1] = normalized_at
        at_rel[..., 2] = state.target_known[:, None]
        at_rel[..., 3] = ~state.target_known[:, None]
        target_capability = state.capabilities.gather(
            2, state.target_types[:, None, :].expand(-1, agents, -1))
        at_rel[..., 4] = target_capability & state.target_known[:, None]

        # Standard routes block all live targets. Target-action columns select
        # their corresponding goal-unblocked bank.
        safe_nodes = world.candidate_nodes.clamp_min(0)
        region_options = route_distances[..., region_nodes]
        region_options = region_options.masked_fill(
            ~region_mask[None, None], torch.inf)
        ac_distance, candidate_entry_slots = region_options.min(dim=3)
        candidate_entry_nodes = region_nodes[None, None].expand(
            batch, agents, -1, -1).gather(
                3, candidate_entry_slots[..., None]).squeeze(3)
        for target in range(targets):
            columns = world.target_candidate_mask[:, target]
            ac_distance[..., columns] = target_route_distances[
                :, :, target, None]
            candidate_entry_nodes[..., columns] = target_entries[
                :, :, target, None]
        ac_distance[..., ~physical] = 0.0
        reachable = torch.isfinite(ac_distance) & (ac_distance < 1e30)
        ac_rel = torch.zeros((batch, agents, actions, 6), device=device)
        normalized_ac = torch.where(reachable, ac_distance / world.distance_scale,
                                    torch.zeros_like(ac_distance))
        ac_rel[..., 0] = normalized_ac
        ac_rel[..., 1] = normalized_ac
        ac_rel[..., 2] = reachable
        category_ok = ~(is_observation[:, None] & ~is_target[:, None] &
                        ~is_staging[:, None] &
                        ~state.capabilities[..., 0, None])
        compatible = torch.ones((batch, agents, actions), dtype=torch.bool,
                                device=device)
        for target in range(targets):
            columns = world.target_candidate_mask[:, target]
            type_index = state.target_types[:, target, None].expand(
                -1, agents).unsqueeze(2)
            has_capability = state.capabilities.gather(
                2, type_index).squeeze(2)
            compatible[..., columns] = (
                ~state.target_known[:, None, target, None] |
                has_capability[..., None])
        ac_rel[..., 4] = category_ok
        ac_rel[..., 5] = compatible
        feasible = state.alive[..., None] & active[:, None] & reachable & category_ok & compatible

        ct_rel = torch.zeros((batch, actions, targets, 7), device=device)
        ct_rel[..., 0] = world.target_candidate_mask
        ct_rel[..., 1] = observed_links
        ct_rel[..., 2] = staging_links
        if self.task_graph:
            base_distance = self._action_target_distances(
                state, planning_episode_mask)
        else:
            # Preserve the original Transformer's static relation tensor and
            # avoid constructing reverse cuGraph route banks it never consumes.
            target_region_options = world.target_distances[
                :, region_nodes].permute(1, 0, 2)
            target_region_options = target_region_options.masked_fill(
                ~region_mask[:, None], torch.inf)
            static_distance = target_region_options.amin(dim=2)
            base_distance = static_distance[None].expand(batch, -1, -1)
        ct_rel[..., 3] = torch.where(
            torch.isfinite(base_distance),
            base_distance / world.distance_scale,
            torch.zeros_like(base_distance))
        ct_rel[..., 4] = state.target_live[:, None]
        ct_rel[..., 5] = state.target_known[:, None]
        ct_rel[..., 6] = ~state.target_known[:, None]

        observation = PlannerObservation(
            agent_x, target_x, action_x,
            torch.ones((batch, agents), dtype=torch.bool, device=device),
            torch.ones((batch, targets), dtype=torch.bool, device=device),
            active, at_rel, ac_rel, ct_rel, feasible,
            [world.candidates for _ in range(batch)], action_capacities=(
                torch.where(
                    world.candidate_capacity < 0,
                    torch.full_like(world.candidate_capacity,
                                    torch.iinfo(torch.long).max),
                    world.candidate_capacity)[None].expand(batch, -1)))
        ct_reachable = (
            torch.isfinite(base_distance) & (base_distance < 1e30)
            & physical[None, :, None]
        )
        attach_task_graph_fields(observation, reachable_at, ct_reachable)
        return (observation, route_distances, predecessors,
                target_route_distances, target_entries,
                candidate_entry_nodes)

    def next_hops(self, state, action_indices, route_distances, predecessors,
                  target_route_distances, target_entries,
                  candidate_entry_nodes):
        """Resolve and materialize selected routes once on CUDA."""
        world = self.world
        actions = torch.as_tensor(action_indices, dtype=torch.long,
                                  device=state.positions.device)
        batch, agents = actions.shape
        rows = torch.arange(batch, device=actions.device)[:, None]
        agent_rows = torch.arange(agents, device=actions.device)[None, :]
        goals = world.candidate_nodes[actions]
        physical = goals >= 0
        target_links = world.target_candidate_mask[actions]
        is_target = target_links.any(dim=2)
        target_index = target_links.long().argmax(dim=2)
        cursor = candidate_entry_nodes[rows, agent_rows, actions]
        ordinary_distance = route_distances[
            rows, agent_rows, cursor.clamp_min(0)]
        selected_target_distance = target_route_distances[
            rows, agent_rows, target_index]
        selected_distance = torch.where(is_target, selected_target_distance,
                                        ordinary_distance)
        reachable = physical & (selected_distance < 1e30)
        destination = torch.where(is_target, goals, cursor)

        route_next = torch.full(
            (batch, agents, len(world.nodes)), -1, dtype=torch.long,
            device=actions.device)
        # Target routes stop at an incoming node in the standard SSSP bank;
        # append the deliberately unblocked final target edge.
        final_edge = reachable & is_target & (cursor != goals)
        final_rows, final_agents = torch.where(final_edge)
        if final_rows.numel():
            route_next[final_rows, final_agents,
                       cursor[final_rows, final_agents]] = goals[
                           final_rows, final_agents]

        planning_positions = torch.where(
            state.moving, state.transit_to, state.positions)
        active = reachable & (cursor != planning_positions)
        for _ in range(len(world.nodes)):
            previous = predecessors.gather(2, cursor[..., None]).squeeze(2)
            valid = active & (previous >= 0)
            valid_rows, valid_agents = torch.where(valid)
            if valid_rows.numel():
                route_next[valid_rows, valid_agents,
                           previous[valid_rows, valid_agents]] = cursor[
                               valid_rows, valid_agents]
            active = valid & (previous != planning_positions)
            cursor = torch.where(active, previous, cursor)
            if not active.any():
                break
        next_hop = route_next[rows, agent_rows, planning_positions]
        at_destination = reachable & (destination == planning_positions)
        return (next_hop.clamp_min(0), reachable | at_destination,
                route_next, destination)
