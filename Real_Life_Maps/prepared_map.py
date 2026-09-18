"""Serialization contract for offline-built simulator terrain maps."""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
import pickle
import tempfile

import networkx as nx
import numpy as np


PREPARED_MAP_SCHEMA = 1
DEFAULT_PREPARED_MAP_PATH = (
    Path(__file__).resolve().parent / "WV_tobler_viewshed_64.pkl.gz"
)
REQUIRED_NODE_ATTRIBUTES = frozenset({
    "pos", "height", "elevation_m", "type", "visible_nodes",
    "visible_edges",
})
REQUIRED_EDGE_ATTRIBUTES = frozenset({
    "distance", "is_road", "observed_edge", "num_used",
})


def _validate_graph(graph: nx.DiGraph) -> None:
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("prepared map graph must be a networkx.DiGraph")
    if not graph:
        raise ValueError("prepared map graph must contain at least one node")
    graph_nodes = set(graph)
    packed = graph.graph.get("packed_visibility")
    if packed is not None:
        ordered = graph.graph.get("visibility_nodes")
        index = graph.graph.get("visibility_index")
        count = len(graph)
        if (not isinstance(ordered, (tuple, list)) or len(ordered) != count
                or set(ordered) != graph_nodes or not isinstance(index, dict)
                or len(index) != count
                or any(index.get(node) != i for i, node in enumerate(ordered))):
            raise ValueError("prepared map packed visibility node order is invalid")
        if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8
                or packed.shape != (count, (count + 7) // 8)):
            raise ValueError("prepared map packed visibility has invalid shape")
        rows = np.arange(count)
        if np.any((packed[rows, rows // 8] & (1 << (rows % 8))) == 0):
            raise ValueError("prepared map visibility must include each observer")
    graph_edges = set(graph.edges)
    for node, data in graph.nodes(data=True):
        required = REQUIRED_NODE_ATTRIBUTES
        if packed is not None:
            required = required - {"visible_nodes", "visible_edges"}
        missing = required - data.keys()
        if missing:
            raise ValueError(
                f"prepared map node {node!r} is missing {sorted(missing)}"
            )
        if packed is None:
            visible_nodes = set(data["visible_nodes"])
            if node not in visible_nodes or not visible_nodes <= graph_nodes:
                raise ValueError(
                    f"prepared map node {node!r} has invalid visible_nodes"
                )
            visible_edges = set(data["visible_edges"])
            induced = {
                (u, v) for u in visible_nodes
                for v in graph.successors(u) if v in visible_nodes
            }
            if visible_edges != induced or not visible_edges <= graph_edges:
                raise ValueError(
                    f"prepared map node {node!r} has visible_edges inconsistent "
                    "with visible_nodes"
                )
    for u, v, data in graph.edges(data=True):
        missing = REQUIRED_EDGE_ATTRIBUTES - data.keys()
        if missing:
            raise ValueError(
                f"prepared map edge {(u, v)!r} is missing {sorted(missing)}"
            )
        distance = data.get("distance")
        if (not isinstance(distance, (int, float))
                or not math.isfinite(float(distance)) or distance <= 0):
            raise ValueError(
                f"prepared map edge {(u, v)!r} has invalid distance"
            )


def resolve_prepared_map_path(path=DEFAULT_PREPARED_MAP_PATH) -> Path:
    """Return the canonical artifact path used for loading and cache keys."""
    return Path(path).expanduser().resolve()


def prepared_map_sha256(path=DEFAULT_PREPARED_MAP_PATH) -> str:
    """Hash the serialized prepared-map content without modifying it."""
    resolved = resolve_prepared_map_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_map_identity(path, graph: nx.DiGraph, metadata: dict,
                          configured_path=None) -> dict:
    """Return portable identity and descriptive metadata for one artifact."""
    resolved = resolve_prepared_map_path(path)
    coarse_shape = metadata.get("coarse_shape")
    coarse_size = metadata.get("coarse_size")
    dimensions = (list(coarse_shape) if coarse_shape is not None else
                  [coarse_size, coarse_size] if coarse_size is not None else None)
    return {
        "configured_path": str(
            path if configured_path is None else configured_path),
        "resolved_path": str(resolved),
        "schema_version": PREPARED_MAP_SCHEMA,
        "sha256": prepared_map_sha256(resolved),
        "dimensions": dimensions,
        "coarse_size": coarse_size,
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "metadata": dict(metadata),
    }


def validate_prepared_map(payload: dict) -> tuple[nx.DiGraph, dict]:
    """Validate and return the graph and immutable build metadata."""
    if not isinstance(payload, dict):
        raise ValueError("prepared map must contain a dictionary payload")
    if payload.get("schema_version") != PREPARED_MAP_SCHEMA:
        raise ValueError(
            "unsupported prepared map schema: "
            f"{payload.get('schema_version')!r}"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("prepared map metadata must be a dictionary")
    coarse_size = metadata.get("coarse_size")
    if coarse_size is not None and (
            isinstance(coarse_size, bool) or not isinstance(coarse_size, int)
            or coarse_size < 1):
        raise ValueError("prepared map coarse_size must be a positive integer")
    coarse_shape = metadata.get("coarse_shape")
    if coarse_shape is not None and (
            not isinstance(coarse_shape, (list, tuple)) or len(coarse_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in coarse_shape)):
        raise ValueError("prepared map coarse_shape must contain two positive integers")
    if coarse_shape is not None and coarse_size is not None \
            and tuple(coarse_shape) != (coarse_size, coarse_size):
        raise ValueError("prepared map coarse_size and coarse_shape disagree")
    graph = payload.get("graph")
    _validate_graph(graph)
    return graph, dict(metadata)


def load_prepared_map(path=DEFAULT_PREPARED_MAP_PATH) -> tuple[nx.DiGraph, dict]:
    """Load a map produced by ``Real_Life_Maps.build_map``."""
    path = resolve_prepared_map_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared terrain map not found: {path}. Build it with "
            "`python -m Real_Life_Maps.build_map`."
        )
    try:
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
    except (OSError, EOFError, pickle.PickleError) as error:
        raise ValueError(f"could not read prepared terrain map {path}: {error}") \
            from error
    return validate_prepared_map(payload)


def save_prepared_map(graph: nx.DiGraph, metadata: dict, path) -> Path:
    """Atomically write one validated, simulator-ready terrain artifact."""
    payload = {
        "schema_version": PREPARED_MAP_SCHEMA,
        "metadata": dict(metadata),
        "graph": graph,
    }
    validate_prepared_map(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                delete=False) as raw_stream:
            temporary = Path(raw_stream.name)
            with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "DEFAULT_PREPARED_MAP_PATH",
    "PREPARED_MAP_SCHEMA",
    "REQUIRED_EDGE_ATTRIBUTES",
    "REQUIRED_NODE_ATTRIBUTES",
    "load_prepared_map",
    "prepared_map_identity",
    "prepared_map_sha256",
    "resolve_prepared_map_path",
    "save_prepared_map",
    "validate_prepared_map",
]
