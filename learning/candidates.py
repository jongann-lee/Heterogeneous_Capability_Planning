"""Deterministic planner-visible candidate generation."""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from simulation.domain import UNKNOWN_TYPE


@dataclass
class Candidate:
    """One deduplicated action location (or a special action)."""

    node: Any | None
    is_target: bool = False
    is_observation: bool = False
    is_staging: bool = False
    is_wait: bool = False
    is_continue: bool = False
    associated_targets: set = field(default_factory=set)
    observed_targets: set = field(default_factory=set)
    staging_targets: set = field(default_factory=set)
    capacity: int | None = 1  # None means unlimited.

    @property
    def key(self):
        if self.is_wait:
            return ("special", "wait")
        if self.is_continue:
            return ("special", "continue")
        return ("node", self.node)


def _visible_nodes(graph, node):
    explicit = graph.nodes[node].get("visible_nodes")
    if explicit is not None:
        return set(explicit) | {node}
    visible = {node}
    for u, v in graph.nodes[node].get("visible_edges", []):
        visible.update((u, v))
    return visible


def generate_candidates(graph: nx.Graph, staging_per_target: int = 2,
                        staging_capacity: int = 1, include_wait: bool = True,
                        include_continue: bool = True) -> list[Candidate]:
    """Generate candidates without consulting a ground-truth graph.

    Physical nodes are deduplicated and receive multi-role flags. Staging
    nodes are the nearest non-target nodes to each unknown target.
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

    unknown_set = set(unknown)
    for node in sorted(graph.nodes, key=repr):
        seen = _visible_nodes(graph, node) & unknown_set
        if seen and node not in unknown_set:
            item = at(node)
            item.is_observation = True
            item.observed_targets.update(seen)
            item.associated_targets.update(seen)

    non_targets = [n for n in graph.nodes if n not in set(live)]
    for target in unknown:
        distances = nx.single_source_dijkstra_path_length(
            graph, target, weight="distance")
        ranked = sorted(
            (n for n in non_targets if n in distances),
            key=lambda n: (distances[n], repr(n)),
        )[:max(0, staging_per_target)]
        for node in ranked:
            item = at(node)
            item.is_staging = True
            item.staging_targets.add(target)
            item.associated_targets.add(target)
            item.capacity = staging_capacity

    result = [by_node[node] for node in sorted(by_node, key=repr)]
    if include_wait:
        result.append(Candidate(None, is_wait=True, capacity=None))
    if include_continue:
        result.append(Candidate(None, is_continue=True, capacity=None))
    return result
