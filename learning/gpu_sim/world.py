"""Immutable device terrain plus target-dependent episode overlays."""

from dataclasses import dataclass

import torch

from learning.policy.candidates import (
    CandidateScenarioCache,
    CandidateTerrainCache,
    generate_candidates,
    physical_group_metadata,
)
from learning.gpu_sim.cugraph_router import CuGraphRouter


@dataclass
class TensorTerrain:
    """Target-independent, immutable CUDA representation of one terrain."""

    graph: object
    nodes: list
    node_index: dict
    positions: torch.Tensor
    heights: torch.Tensor
    visibility: torch.Tensor
    edge_cost: torch.Tensor
    neighbors: torch.Tensor
    distance_scale: float
    router: CuGraphRouter
    reverse_router: CuGraphRouter
    candidate_cache: CandidateTerrainCache

    @classmethod
    def from_networkx(cls, graph, device="cuda"):
        nodes = sorted(graph.nodes, key=repr)
        node_index = {node: index for index, node in enumerate(nodes)}
        raw_positions = torch.tensor(
            [graph.nodes[node].get("pos", (0.0, 0.0)) for node in nodes],
            dtype=torch.float32, device=device)
        origin = raw_positions.amin(dim=0)
        scale = torch.maximum(
            (raw_positions.amax(dim=0) - origin).amax(),
            torch.tensor(1.0, device=device))
        positions = (raw_positions - origin) / scale
        raw_heights = torch.tensor(
            [graph.nodes[node].get("height", 0.0) for node in nodes],
            dtype=torch.float32, device=device)
        heights = raw_heights / raw_heights.abs().amax().clamp_min(1.0)

        visibility = torch.zeros(
            (len(nodes), len(nodes)), dtype=torch.bool, device=device)
        for node in nodes:
            visible = {node}
            explicit = graph.nodes[node].get("visible_nodes")
            if explicit is not None:
                visible.update(explicit)
            else:
                for u, v in graph.nodes[node].get("visible_edges", []):
                    visible.update((u, v))
            indices = [node_index[item] for item in visible if item in node_index]
            visibility[node_index[node], indices] = True

        max_degree = max(dict(graph.out_degree()).values())
        edge_cost = torch.full(
            (len(nodes), max_degree), torch.inf, dtype=torch.float32,
            device=device)
        neighbors = torch.full(
            (len(nodes), max_degree), -1, dtype=torch.long, device=device)
        slots = {node: 0 for node in nodes}
        for u, v, weight in graph.edges(data="distance"):
            row, slot = node_index[u], slots[u]
            neighbors[row, slot] = node_index[v]
            edge_cost[row, slot] = float(weight)
            slots[u] += 1
        edges = list(graph.edges(data="distance"))
        router = CuGraphRouter(
            len(nodes), [node_index[u] for u, _v, _w in edges],
            [node_index[v] for _u, v, _w in edges],
            [float(w) for _u, _v, w in edges])
        reverse_router = CuGraphRouter(
            len(nodes), [node_index[v] for _u, v, _w in edges],
            [node_index[u] for u, _v, _w in edges],
            [float(w) for _u, _v, w in edges])
        distance_scale = max(sum(
            float(data.get("distance", 1.0))
            for _u, _v, data in graph.edges(data=True)), 1.0)
        return cls(graph, nodes, node_index, positions, heights, visibility,
                   edge_cost, neighbors, distance_scale, router, reverse_router,
                   CandidateTerrainCache(graph))


@dataclass
class TensorWorld:
    terrain: TensorTerrain
    graph: object
    nodes: list
    node_index: dict
    targets: list
    candidates: list
    positions: torch.Tensor
    heights: torch.Tensor
    target_nodes: torch.Tensor
    candidate_nodes: torch.Tensor
    candidate_region_nodes: torch.Tensor
    candidate_region_mask: torch.Tensor
    candidate_target_mask: torch.Tensor
    candidate_observed_mask: torch.Tensor
    candidate_staging_mask: torch.Tensor
    candidate_staging_arity: torch.Tensor
    candidate_pair_distance: torch.Tensor
    candidate_pair_order: torch.Tensor
    visible_targets: torch.Tensor
    edge_cost: torch.Tensor
    candidate_is_target: torch.Tensor
    candidate_is_observation: torch.Tensor
    candidate_is_staging: torch.Tensor
    candidate_is_wait: torch.Tensor
    candidate_capacity: torch.Tensor
    candidate_physical_group: torch.Tensor
    physical_group_capacity: torch.Tensor
    physical_group_representative: torch.Tensor
    physical_group_mask: torch.Tensor
    target_candidate_mask: torch.Tensor
    allow_unknown_target_actions: bool
    distance_scale: float
    target_distances: torch.Tensor
    target_incoming_nodes: torch.Tensor
    target_incoming_costs: torch.Tensor
    router: CuGraphRouter
    reverse_router: CuGraphRouter

    @classmethod
    def from_networkx(cls, graph, candidate_config, device="cuda",
                      terrain=None):
        terrain = terrain or TensorTerrain.from_networkx(graph, device=device)
        nodes = terrain.nodes
        node_index = terrain.node_index
        targets = sorted(
            (node for node, data in graph.nodes(data=True)
             if data.get("type") in ("target_unreached", "target_reached")),
            key=repr)
        scenario_cache = CandidateScenarioCache(
            graph, terrain_cache=terrain.candidate_cache,
            include_pair_staging=candidate_config.include_pair_staging)
        candidates = generate_candidates(
            graph, candidate_config, terrain.candidate_cache,
            include_all_pairs=True, scenario_cache=scenario_cache)
        target_index = {node: index for index, node in enumerate(targets)}

        positions = terrain.positions
        heights = terrain.heights

        candidate_nodes = torch.tensor([
            -1 if item.node is None else node_index[item.node]
            for item in candidates], dtype=torch.long, device=device)
        regions = [item.region_nodes or (
            frozenset({item.node}) if item.node is not None else frozenset())
            for item in candidates]
        max_region_size = max(map(len, regions), default=0)
        candidate_region_nodes = torch.full(
            (len(candidates), max_region_size), -1, dtype=torch.long,
            device=device)
        candidate_region_mask = torch.zeros_like(candidate_region_nodes,
                                                 dtype=torch.bool)
        for candidate_index, region in enumerate(regions):
            indices = [node_index[node] for node in sorted(region, key=repr)]
            if indices:
                candidate_region_nodes[candidate_index, :len(indices)] = torch.tensor(
                    indices, dtype=torch.long, device=device)
                candidate_region_mask[candidate_index, :len(indices)] = True
        shape = (len(candidates), len(targets))
        associated = torch.zeros(shape, dtype=torch.bool, device=device)
        observed = torch.zeros_like(associated)
        staging = torch.zeros_like(associated)
        for c, item in enumerate(candidates):
            for node in item.associated_targets:
                associated[c, target_index[node]] = True
            for node in item.observed_targets:
                observed[c, target_index[node]] = True
            for node in item.staging_targets:
                staging[c, target_index[node]] = True

        staging_arity = torch.tensor(
            [item.staging_arity for item in candidates], dtype=torch.long,
            device=device)
        pair_distance = torch.tensor([
            torch.inf if item.pair_distance is None else item.pair_distance
            for item in candidates
        ], dtype=torch.float32, device=device)
        pair_indices = [
            index for index, item in enumerate(candidates)
            if item.staging_arity == 2]
        pair_indices.sort(key=lambda index: (
            candidates[index].pair_distance,
            tuple(repr(target) for target in sorted(
                candidates[index].staging_targets, key=repr)),
        ))
        pair_order = torch.tensor(
            pair_indices, dtype=torch.long, device=device)

        (candidate_groups, group_capacities,
         group_representatives) = physical_group_metadata(candidates)
        candidate_physical_group = torch.tensor(
            candidate_groups, dtype=torch.long, device=device)
        physical_group_capacity = torch.tensor(
            group_capacities, dtype=torch.long, device=device)
        physical_group_representative = torch.tensor(
            group_representatives, dtype=torch.long, device=device)
        physical_group_mask = torch.ones(
            len(group_capacities), dtype=torch.bool, device=device)

        target_nodes_tensor = torch.tensor(
            [node_index[node] for node in targets], device=device)
        visible_targets = terrain.visibility[:, target_nodes_tensor]
        edge_cost = terrain.edge_cost
        neighbors = terrain.neighbors
        router = terrain.router
        reverse_router = terrain.reverse_router
        target_candidate_mask = (
            candidate_nodes[:, None] == target_nodes_tensor[None, :])
        distance_scale = terrain.distance_scale
        # Reverse-graph SSSP yields directed Candidate -> Target distances.
        target_sssp = reverse_router.sssp(
            [node_index[node] for node in targets], ())
        max_incoming = max(graph.in_degree(node) for node in targets)
        target_incoming_nodes = torch.full(
            (len(targets), max_incoming), -1, dtype=torch.long, device=device)
        target_incoming_costs = torch.full(
            (len(targets), max_incoming), torch.inf, device=device)
        for target_index_value, target in enumerate(targets):
            for slot, (source, _target, data) in enumerate(
                    graph.in_edges(target, data=True)):
                target_incoming_nodes[target_index_value, slot] = node_index[source]
                target_incoming_costs[target_index_value, slot] = float(
                    data.get("distance", 1.0))
        world = cls(
            terrain=terrain, graph=graph, nodes=nodes, node_index=node_index,
            targets=targets, candidates=candidates, positions=positions,
            heights=heights, target_nodes=target_nodes_tensor,
            candidate_nodes=candidate_nodes,
            candidate_region_nodes=candidate_region_nodes,
            candidate_region_mask=candidate_region_mask,
            candidate_target_mask=associated,
            candidate_observed_mask=observed,
            candidate_staging_mask=staging,
            candidate_staging_arity=staging_arity,
            candidate_pair_distance=pair_distance,
            candidate_pair_order=pair_order,
            visible_targets=visible_targets, edge_cost=edge_cost,
            candidate_is_target=torch.tensor(
                [item.is_target for item in candidates],
                dtype=torch.bool, device=device),
            candidate_is_observation=torch.tensor(
                [item.is_observation for item in candidates],
                dtype=torch.bool, device=device),
            candidate_is_staging=torch.tensor(
                [item.is_staging for item in candidates],
                dtype=torch.bool, device=device),
            candidate_is_wait=torch.tensor(
                [item.is_wait for item in candidates],
                dtype=torch.bool, device=device),
            candidate_capacity=torch.tensor(
                [-1 if item.capacity is None else item.capacity
                 for item in candidates], dtype=torch.long, device=device),
            candidate_physical_group=candidate_physical_group,
            physical_group_capacity=physical_group_capacity,
            physical_group_representative=physical_group_representative,
            physical_group_mask=physical_group_mask,
            target_candidate_mask=target_candidate_mask,
            allow_unknown_target_actions=(
                candidate_config.allow_unknown_target_actions),
            distance_scale=distance_scale,
            target_distances=target_sssp.distances,
            target_incoming_nodes=target_incoming_nodes,
            target_incoming_costs=target_incoming_costs,
            router=router, reverse_router=reverse_router)
        world.neighbors = neighbors
        world.required_route_nodes = torch.unique(torch.cat((
            candidate_region_nodes[candidate_region_mask],
            target_incoming_nodes[target_incoming_nodes >= 0],
            target_nodes_tensor,
        )))
        return world
