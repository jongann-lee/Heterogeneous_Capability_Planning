"""Read node and edge visibility from prepared or legacy graph attributes."""

from __future__ import annotations

import numpy as np


def visible_nodes(graph, observer):
    """Return nodes visible to ``observer``, including the observer itself."""
    explicit = graph.nodes[observer].get("visible_nodes")
    if explicit is not None:
        return set(explicit) | {observer}

    packed = graph.graph.get("packed_visibility")
    if packed is not None:
        nodes = graph.graph["visibility_nodes"]
        index = graph.graph["visibility_index"][observer]
        bits = np.unpackbits(packed[index], bitorder="little")[:len(nodes)]
        return {nodes[int(i)] for i in np.flatnonzero(bits)}

    visible = {observer}
    for u, v in graph.nodes[observer].get("visible_edges", ()):
        visible.update((u, v))
    return visible


def visible_edges(graph, observer):
    """Return visible directed edges, respecting edges removed from ``graph``."""
    explicit = graph.nodes[observer].get("visible_edges")
    if explicit is not None:
        return [edge for edge in explicit if graph.has_edge(*edge)]

    visible = visible_nodes(graph, observer)
    return [(u, v) for u in visible for v in graph.successors(u)
            if v in visible]


def visible_targets(graph, observer, targets):
    """Intersect an observer's viewshed with a small set of target nodes."""
    packed = graph.graph.get("packed_visibility")
    if packed is not None:
        indices = graph.graph["visibility_index"]
        row = packed[indices[observer]]
        return {
            target for target in targets
            if target in indices and row[indices[target] // 8]
            & (1 << (indices[target] % 8))
        }
    return visible_nodes(graph, observer) & set(targets)
