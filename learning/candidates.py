"""Deterministic planner-visible candidate generation."""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from learning.configuration import CandidateConfig
from simulation.domain import UNKNOWN_TYPE


@dataclass
class Candidate:
    """One deduplicated action location (or a special action)."""

    node: Any | None
    is_target: bool = False
    is_observation: bool = False
    is_staging: bool = False
    is_wait: bool = False
    associated_targets: set = field(default_factory=set)
    observed_targets: set = field(default_factory=set)
    staging_targets: set = field(default_factory=set)
    region_nodes: frozenset = field(default_factory=frozenset)
    capacity: int | None = 1  # None means unlimited.

    @property
    def key(self):
        if self.is_wait:
            return ("special", "wait")
        if self.is_observation and self.region_nodes:
            return ("observation_region", tuple(sorted(self.region_nodes,
                                                        key=repr)))
        return ("node", self.node)


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
                graph, target, weight="distance")
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


def generate_candidates(graph: nx.Graph,
                        config: CandidateConfig,
                        terrain_cache: CandidateTerrainCache | None = None,
                        ) -> list[Candidate]:
    """Generate candidates without consulting a ground-truth graph.

    Each observation action represents every node (possibly disconnected)
    that reveals exactly the same set of currently unknown targets. Its
    selected destination is agent-dependent. Physical target/staging nodes
    remain deduplicated.
    """
    live = [n for n, d in graph.nodes(data=True)
            if d.get("type") == "target_unreached"]
    unknown = [n for n in live
               if graph.nodes[n].get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE]
    by_node: dict[Any, Candidate] = {}

    def at(node):
        return by_node.setdefault(node, Candidate(node=node))

    for target in live:
        item = at(target)
        item.is_target = True
        item.associated_targets.add(target)

    live_set = set(live)
    all_nodes = terrain_cache.nodes if terrain_cache is not None else graph.nodes
    non_targets = [n for n in all_nodes if n not in live_set]
    signature_nodes = {}
    unknown_set = set(unknown)
    for node in non_targets:
        visible = (terrain_cache.visible[node] if terrain_cache is not None
                   else _visible_nodes(graph, node))
        signature = frozenset(visible & unknown_set)
        if signature:
            signature_nodes.setdefault(signature, set()).add(node)

    observation_regions = []
    for signature in sorted(signature_nodes, key=lambda value:
                            tuple(map(repr, sorted(value, key=repr)))):
        region = frozenset(signature_nodes[signature])
        representative = min(region, key=repr)
        observation_regions.append(Candidate(
            node=representative, is_observation=True,
            associated_targets=set(signature),
            observed_targets=set(signature), region_nodes=region))

    for target in unknown:
        if terrain_cache is None:
            distances = nx.single_source_dijkstra_path_length(
                graph, target, weight="distance")
            ranked = sorted(
                (n for n in non_targets if n in distances),
                key=lambda n: (distances[n], repr(n)),
            )[:config.staging_per_target]
        else:
            ranked = [node for node in terrain_cache.staging_ranking(
                graph, target) if node not in live_set][
                    :config.staging_per_target]
        for node in ranked:
            item = at(node)
            item.is_staging = True
            item.staging_targets.add(target)
            item.associated_targets.add(target)
            item.capacity = config.staging_capacity

    result = ([by_node[node] for node in sorted(by_node, key=repr)]
              + sorted(observation_regions, key=lambda item: item.key))
    if config.include_wait:
        result.append(Candidate(None, is_wait=True, capacity=None))
    return result
