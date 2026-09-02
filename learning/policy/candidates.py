"""Deterministic planner-visible candidate generation."""

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any

import networkx as nx

from learning.policy.configuration import CandidateConfig
from simulation.domain import UNKNOWN_TYPE


@dataclass
class Candidate:
    """One semantic action intention and its physical destination."""

    node: Any | None
    is_target: bool = False
    is_observation: bool = False
    is_staging: bool = False
    is_wait: bool = False
    associated_targets: set = field(default_factory=set)
    observed_targets: set = field(default_factory=set)
    staging_targets: set = field(default_factory=set)
    staging_arity: int = 0
    region_nodes: frozenset = field(default_factory=frozenset)
    capacity: int | None = 1  # None means unlimited.
    pair_distance: float | None = None

    def __post_init__(self):
        if self.is_staging and self.staging_arity == 0:
            inferred_arity = len(self.staging_targets)
            if inferred_arity not in (1, 2):
                raise ValueError(
                    "a staging candidate must identify one or two targets")
            self.staging_arity = inferred_arity
        if self.staging_arity not in (0, 1, 2):
            raise ValueError("staging_arity must be 0, 1, or 2")
        if self.staging_arity:
            self.is_staging = True
            if len(self.staging_targets) != self.staging_arity:
                raise ValueError(
                    "staging_arity must match the number of staging_targets")
            self.associated_targets.update(self.staging_targets)

    @property
    def semantic_key(self):
        if self.is_wait:
            return ("wait",)
        if self.is_target:
            return ("target", self.node)
        if self.staging_arity == 1:
            target = min(self.staging_targets, key=repr)
            return ("single_stage", self.node, target)
        if self.staging_arity == 2:
            first, second = sorted(self.staging_targets, key=repr)
            return ("pair_stage", self.node, first, second)
        if self.is_observation:
            signature = tuple(sorted(self.observed_targets, key=repr))
            return ("observation", signature)
        return ("node", self.node)

    @property
    def physical_key(self):
        if self.is_wait:
            return ("special", "wait")
        if self.is_observation:
            region = self.region_nodes or frozenset({self.node})
            return ("observation_region",
                    tuple(sorted(region, key=repr)))
        return ("node", self.node)

    @property
    def key(self):
        """Backward-compatible name for deterministic semantic identity."""
        return self.semantic_key


def physical_group_metadata(candidates):
    """Return candidate-to-group indices, capacities, and representatives."""
    unlimited = 2**63 - 1
    physical_keys = [candidate.physical_key for candidate in candidates]
    ordered_keys = sorted(set(physical_keys), key=repr)
    group_by_key = {key: index for index, key in enumerate(ordered_keys)}
    candidate_groups = []
    capacities = [None] * len(ordered_keys)
    representatives = [None] * len(ordered_keys)
    for candidate_index, (candidate, key) in enumerate(
            zip(candidates, physical_keys)):
        group = group_by_key[key]
        capacity = unlimited if candidate.capacity is None else int(
            candidate.capacity)
        if capacities[group] is None:
            capacities[group] = capacity
            representatives[group] = candidate_index
        elif capacities[group] != capacity:
            raise ValueError(
                f"candidate aliases for physical group {key!r} have "
                "conflicting capacities")
        if key[0] == "node" and candidate.node != key[1]:
            raise ValueError(
                f"candidate alias {candidate.semantic_key!r} does not resolve "
                f"to physical node {key[1]!r}")
        candidate_groups.append(group)
    return candidate_groups, capacities, representatives


class CandidateTerrainCache:
    """Target-independent visibility and lazy distance rankings."""

    def __init__(self, graph):
        self.nodes = tuple(graph.nodes)
        self.visible = {
            node: frozenset(_visible_nodes(graph, node))
            for node in self.nodes
        }
        self._staging_rankings = {}

    def staging_ranking(self, graph, target):
        ranking = self._staging_rankings.get(target)
        if ranking is None:
            distances = nx.single_source_dijkstra_path_length(
                graph.reverse(copy=False) if graph.is_directed() else graph,
                target, weight="distance")
            ranking = tuple(sorted(
                distances, key=lambda node: (distances[node], repr(node))))
            self._staging_rankings[target] = ranking
        return ranking


def _visible_nodes(graph, node):
    explicit = graph.nodes[node].get("visible_nodes")
    if explicit is not None:
        return set(explicit) | {node}
    visible = {node}
    for u, v in graph.nodes[node].get("visible_edges", []):
        visible.update((u, v))
    return visible


def _safe_distances_to_target(graph, target, target_nodes):
    """Directed distances from every safe non-blocked node to ``target``."""
    blocked = set(target_nodes) - {target}
    view = nx.subgraph_view(
        graph, filter_node=lambda node: node not in blocked)
    reverse = view.reverse(copy=False) if graph.is_directed() else view
    try:
        return nx.single_source_dijkstra_path_length(
            reverse, target, weight="distance")
    except nx.NodeNotFound:
        return {}


def _blocked_source_distance(graph, source, distances_to_target):
    """Recover a safe distance when ``source`` is itself a blocked target."""
    best = math.inf
    if source not in graph:
        return best
    for neighbor, edge in graph[source].items():
        remaining = distances_to_target.get(neighbor, math.inf)
        if math.isfinite(remaining):
            best = min(
                best, float(edge.get("distance", 1.0)) + remaining)
    return best


def _pair_staging_definitions(graph, pair_targets, all_target_nodes,
                              distance_maps):
    """Return finite pair definitions ordered by conservative separation."""
    eligible_nodes = [
        node for node in graph if node not in all_target_nodes]
    definitions = []
    ordered_targets = sorted(pair_targets, key=repr)
    for first, second in combinations(ordered_targets, 2):
        first_to_second = _blocked_source_distance(
            graph, first, distance_maps[second])
        second_to_first = _blocked_source_distance(
            graph, second, distance_maps[first])
        if not (math.isfinite(first_to_second)
                and math.isfinite(second_to_first)):
            continue
        location_keys = []
        for node in eligible_nodes:
            first_distance = distance_maps[first].get(node, math.inf)
            second_distance = distance_maps[second].get(node, math.inf)
            if (math.isfinite(first_distance)
                    and math.isfinite(second_distance)):
                location_keys.append((
                    max(first_distance, second_distance),
                    first_distance + second_distance,
                    repr(node), node,
                ))
        if not location_keys:
            continue
        location = min(location_keys)[-1]
        separation = max(first_to_second, second_to_first)
        definitions.append((
            separation, repr(first), repr(second),
            first, second, location,
        ))
    definitions.sort(key=lambda item: item[:3])
    return definitions


class CandidateScenarioCache:
    """Episode-static staging geometry computed once from the belief graph."""

    def __init__(self, graph, terrain_cache=None, include_pair_staging=True):
        self.graph = graph
        self.target_nodes = tuple(sorted(
            (node for node, data in graph.nodes(data=True)
             if data.get("type") in ("target_unreached", "target_reached")),
            key=repr))
        self.target_set = frozenset(self.target_nodes)
        self.terrain_cache = terrain_cache
        all_nodes = (terrain_cache.nodes if terrain_cache is not None
                     else tuple(graph.nodes))
        self.non_target_nodes = tuple(
            node for node in all_nodes if node not in self.target_set)
        self.distance_maps = {
            target: _safe_distances_to_target(
                graph, target, self.target_set)
            for target in self.target_nodes
        }
        self.single_rankings = {
            target: tuple(sorted(
                (node for node in self.non_target_nodes
                 if node in self.distance_maps[target]),
                key=lambda node: (
                    self.distance_maps[target][node], repr(node))))
            for target in self.target_nodes
        }
        self.include_pair_staging = bool(include_pair_staging)
        self.pair_definitions = tuple(
            _pair_staging_definitions(
                graph, self.target_nodes, self.target_set,
                self.distance_maps)
            if self.include_pair_staging else ())

    def validate(self, graph, include_pair_staging):
        if graph is not self.graph:
            raise ValueError(
                "candidate scenario cache belongs to a different graph")
        current_targets = frozenset(
            node for node, data in graph.nodes(data=True)
            if data.get("type") in ("target_unreached", "target_reached"))
        if current_targets != self.target_set:
            raise ValueError(
                "candidate scenario cache does not match the graph targets")
        if include_pair_staging and not self.include_pair_staging:
            raise ValueError(
                "candidate scenario cache was built without pair staging")


def generate_candidates(graph: nx.Graph,
                        config: CandidateConfig,
                        terrain_cache: CandidateTerrainCache | None = None,
                        include_all_pairs: bool = False,
                        scenario_cache: CandidateScenarioCache | None = None,
                        ) -> list[Candidate]:
    """Generate candidates without consulting a ground-truth graph.

    Each observation action represents every node (possibly disconnected)
    that reveals exactly the same set of currently unknown targets. Its
    selected destination is agent-dependent. Staging intentions remain
    semantically separate even when they share one physical terrain node.
    """
    if scenario_cache is None:
        scenario_cache = CandidateScenarioCache(
            graph, terrain_cache=terrain_cache,
            include_pair_staging=config.include_pair_staging)
    else:
        scenario_cache.validate(graph, config.include_pair_staging)
    terrain_cache = terrain_cache or scenario_cache.terrain_cache
    all_targets = list(scenario_cache.target_nodes)
    live = [node for node in all_targets
            if graph.nodes[node].get("type") == "target_unreached"]
    unknown = sorted(
        (node for node in live
         if graph.nodes[node].get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE),
        key=repr)
    result = [Candidate(
        node=target, is_target=True, associated_targets={target})
        for target in live]

    non_targets = scenario_cache.non_target_nodes
    signature_nodes = {}
    unknown_set = set(unknown)
    for node in non_targets:
        visible = (terrain_cache.visible[node] if terrain_cache is not None
                   else _visible_nodes(graph, node))
        signature = frozenset(visible & unknown_set)
        if signature:
            signature_nodes.setdefault(signature, set()).add(node)

    for signature in sorted(signature_nodes, key=lambda value:
                            tuple(map(repr, sorted(value, key=repr)))):
        region = frozenset(signature_nodes[signature])
        representative = min(region, key=repr)
        result.append(Candidate(
            node=representative, is_observation=True,
            associated_targets=set(signature),
            observed_targets=set(signature), region_nodes=region))

    for target in unknown:
        ranked = scenario_cache.single_rankings[target][
            :config.staging_per_target]
        for node in ranked:
            result.append(Candidate(
                node=node, is_staging=True, staging_arity=1,
                staging_targets={target}, associated_targets={target},
                capacity=config.staging_capacity))

    pair_targets = set(all_targets if include_all_pairs else unknown)
    if config.include_pair_staging and len(pair_targets) >= 2:
        pair_definitions = [
            definition for definition in scenario_cache.pair_definitions
            if definition[3] in pair_targets and definition[4] in pair_targets
        ]
        if not include_all_pairs:
            pair_definitions = pair_definitions[:len(unknown)]
        for separation, _first_repr, _second_repr, first, second, node in (
                pair_definitions):
            result.append(Candidate(
                node=node, is_staging=True, staging_arity=2,
                staging_targets={first, second},
                associated_targets={first, second},
                capacity=config.staging_capacity,
                pair_distance=separation))

    if config.include_wait:
        result.append(Candidate(None, is_wait=True, capacity=None))
    return sorted(result, key=lambda item: repr(item.semantic_key))
