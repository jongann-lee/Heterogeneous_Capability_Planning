"""Compile a blueprint occupancy raster into a sparse indoor planning graph.

The 5x5 default preserves every four-connected fine-pixel component inside a
patch as its own node. An edge exists only across an actual shared fine-pixel
border; its cost is the number of unit-cost fine steps along the stored route.
Visibility is exact supercover line of sight on the original occupancy raster.
The prepared map embeds packed visibility rather than replicating millions of
visible edges in NetworkX node dictionaries.

Run from the repository root:
    python -m Real_Life_Maps.indoor_map_generation

The output includes diagnostics and a versioned, simulator-ready map. Use
``--prepare-existing`` to convert a previously generated diagnostic bundle
without recomputing line of sight. Use ``--prune-existing`` to keep one node
per occupied patch and add exact path-backed shortcuts through discarded
split-patch pieces.
"""

from __future__ import annotations

import argparse
from collections import deque
import ctypes
import gzip
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import networkx as nx
import numpy as np
from scipy import ndimage

from Real_Life_Maps.prepared_map import save_prepared_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Real_Life_Maps" / "50052749"
DEFAULT_OUTPUT = ROOT / "outputs" / "map_diagnostics" / "indoor_50052749_patch5"
FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)


def load_walkable(input_dir: Path) -> tuple[np.ndarray, dict]:
    """Keep the main interior plus explicitly labelled interior islands.

    The source uses 254 for free and 0 for wall. ``valid_space`` is a zone-ID
    raster, not a binary free-space mask: some passable corridors have ID 0.
    The largest four-connected free component is the primary interior. Positive
    zone IDs retain labelled rooms disconnected by closed/missing door pixels.
    Unlabelled exterior/background free components are discarded.
    """
    occupancy = np.load(input_dir / "occ_map.npy", allow_pickle=False)
    zones = np.load(input_dir / "valid_space.npy", allow_pickle=False)
    if occupancy.ndim != 2 or occupancy.shape != zones.shape:
        raise ValueError("occupancy and zone rasters must be aligned 2D arrays")
    values = set(map(int, np.unique(occupancy)))
    if values != {0, 254}:
        raise ValueError(f"expected occupancy values {{0, 254}}, got {values}")
    free = occupancy == 254
    labels, _ = ndimage.label(free, FOUR_CONNECTED)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    main_component = int(component_sizes.argmax())
    walkable = (labels == main_component) | (free & (zones > 0))
    metadata = {
        "fine_shape": list(map(int, occupancy.shape)),
        "occupancy_free_value": 254,
        "occupancy_wall_value": 0,
        "main_free_component_pixels": int(component_sizes[main_component]),
        "walkable_pixels": int(walkable.sum()),
        "positive_zone_ids": [int(x) for x in np.unique(zones) if x > 0],
        "selection": "main 4-connected free component plus labelled free islands",
    }
    return np.ascontiguousarray(walkable, dtype=np.uint8), metadata


def _anchor_and_routes(pixels: list[tuple[int, int]], center):
    """Choose a reachable medoid and its shortest in-component pixel routes."""
    anchor = min(
        pixels,
        key=lambda pixel: (
            sum(abs(pixel[0] - other[0]) + abs(pixel[1] - other[1])
                for other in pixels),
            abs(pixel[0] - center[0]) + abs(pixel[1] - center[1]),
            pixel,
        ),
    )
    allowed = set(pixels)
    distance = {anchor: 0}
    predecessor = {anchor: None}
    queue = deque([anchor])
    while queue:
        row, col = queue.popleft()
        for neighbor in ((row - 1, col), (row, col + 1),
                         (row + 1, col), (row, col - 1)):
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[row, col] + 1
                predecessor[neighbor] = (row, col)
                queue.append(neighbor)
    assert len(distance) == len(pixels)
    return anchor, distance, predecessor


def _path_from_anchor(predecessor, pixel):
    path = [pixel]
    while predecessor[path[-1]] is not None:
        path.append(predecessor[path[-1]])
    return tuple(reversed(path))


def build_graph(walkable: np.ndarray, patch_size: int = 5):
    """Return directed graph, fine-pixel owners and deterministic node order."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    height, width = walkable.shape
    if height % patch_size or width % patch_size:
        raise ValueError("fine raster dimensions must be divisible by patch_size")
    if walkable.dtype != np.uint8:
        walkable = np.asarray(walkable, dtype=np.uint8)
    fine_labels, _ = ndimage.label(walkable, FOUR_CONNECTED)
    owners = np.full((height, width), -1, dtype=np.int32)
    graph = nx.DiGraph()
    nodes = []
    distances = []
    predecessors = []
    anchors = []
    components = []
    split_patches = 0

    for patch_row in range(height // patch_size):
        y0 = patch_row * patch_size
        for patch_col in range(width // patch_size):
            x0 = patch_col * patch_size
            local, count = ndimage.label(
                walkable[y0:y0 + patch_size, x0:x0 + patch_size],
                FOUR_CONNECTED,
            )
            split_patches += count > 1
            for part in range(1, count + 1):
                ys, xs = np.where(local == part)
                pixels = [(int(y0 + y), int(x0 + x))
                          for y, x in zip(ys, xs)]
                anchor, local_dist, predecessor = _anchor_and_routes(
                    pixels, (y0 + (patch_size - 1) / 2,
                             x0 + (patch_size - 1) / 2))
                index = len(nodes)
                node = (patch_row, patch_col, part - 1)
                nodes.append(node)
                anchors.append(anchor)
                components.append(int(fine_labels[anchor]))
                distances.append(local_dist)
                predecessors.append(predecessor)
                for pixel in pixels:
                    owners[pixel] = index
                graph.add_node(
                    node,
                    pos=(float(anchor[1]), float(anchor[0])),
                    dense_anchor=anchor,
                    coarse_cell=(patch_row, patch_col),
                    fine_pixel_count=len(pixels),
                    fine_component=int(fine_labels[anchor]),
                    type="intermediate",
                    height=0.0,
                    elevation_m=0.0,
                    visibility_index=index,
                )

    # Only shared orthogonal pixel borders create edges. A patch containing a
    # wall-divided pair of regions therefore cannot teleport between them.
    crossings = {}

    def consider(first, second):
        i, j = int(owners[first]), int(owners[second])
        if i < 0 or j < 0 or i == j:
            return
        if i > j:
            i, j, first, second = j, i, second, first
        cost = distances[i][first] + 1 + distances[j][second]
        old = crossings.get((i, j))
        if old is None or (cost, first, second) < old:
            crossings[i, j] = (cost, first, second)

    for x in range(patch_size, width, patch_size):
        rows = np.flatnonzero((owners[:, x - 1] >= 0) & (owners[:, x] >= 0))
        for y in rows:
            consider((int(y), x - 1), (int(y), x))
    for y in range(patch_size, height, patch_size):
        cols = np.flatnonzero((owners[y - 1, :] >= 0) & (owners[y, :] >= 0))
        for x in cols:
            consider((y - 1, int(x)), (y, int(x)))

    for (i, j), (cost, first, second) in crossings.items():
        left = _path_from_anchor(predecessors[i], first)
        right = _path_from_anchor(predecessors[j], second)
        fine_path = left + tuple(reversed(right))
        assert len(fine_path) - 1 == cost
        base = dict(
            distance=float(cost),
            is_road=False,
            observed_edge=False,
            num_used=1.0,
        )
        graph.add_edge(nodes[i], nodes[j], fine_path=fine_path, **base)
        graph.add_edge(nodes[j], nodes[i],
                       fine_path=tuple(reversed(fine_path)), **base)

    info = {
        "patch_size": patch_size,
        "coarse_shape": [height // patch_size, width // patch_size],
        "node_count": len(nodes),
        "directed_edge_count": graph.number_of_edges(),
        "split_patches": int(split_patches),
        "fine_components_represented": len(set(components)),
        "distance_units": "fine-pixel orthogonal steps",
        "connectivity": "4-neighbor; edges require a shared walkable fine-pixel face",
    }
    return graph, owners, nodes, np.asarray(anchors, dtype=np.int32), \
        np.asarray(components, dtype=np.int32), info


def _python_los(open_pixels, first, second):
    """Portable reference implementation of strict supercover line of sight."""
    row0, col0 = map(int, first)
    row1, col1 = map(int, second)
    dx, dy = abs(col1 - col0), abs(row1 - row0)
    sx = 1 if col1 > col0 else -1
    sy = 1 if row1 > row0 else -1
    x, y, ix, iy = col0, row0, 0, 0

    def clear(row, col):
        return (0 <= row < open_pixels.shape[0]
                and 0 <= col < open_pixels.shape[1]
                and bool(open_pixels[row, col]))

    if not clear(row0, col0) or not clear(row1, col1):
        return False
    while ix < dx or iy < dy:
        comparison = (1 + 2 * ix) * dy - (1 + 2 * iy) * dx
        if comparison == 0:
            if not clear(y, x + sx) or not clear(y + sy, x):
                return False
            x += sx; y += sy; ix += 1; iy += 1
        elif comparison < 0:
            x += sx; ix += 1
        else:
            y += sy; iy += 1
        if not clear(y, x):
            return False
    return True


def _compile_los(directory: Path):
    source = Path(__file__).with_name("indoor_los.c")
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise RuntimeError("full visibility needs clang/gcc/cc for the C ray caster")
    library = directory / ("indoor_los.dylib" if sys.platform == "darwin"
                           else "indoor_los.so")
    command = [compiler, "-O3", "-std=c99", "-fPIC", "-shared",
               str(source), "-o", str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    loaded.indoor_los.argtypes = [byte_pointer, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int]
    loaded.indoor_los.restype = ctypes.c_int
    loaded.indoor_visibility_rows.argtypes = [
        byte_pointer, ctypes.c_int, ctypes.c_int, int_pointer, int_pointer,
        ctypes.c_int, byte_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    loaded.indoor_visibility_rows.restype = ctypes.c_uint64
    return loaded


def compute_visibility(walkable, anchors, components, chunk_size=128):
    """Build symmetric packed node LOS at the original raster resolution."""
    walkable = np.ascontiguousarray(walkable, dtype=np.uint8)
    anchors = np.ascontiguousarray(anchors, dtype=np.int32)
    components = np.ascontiguousarray(components, dtype=np.int32)
    count = len(anchors)
    row_bytes = (count + 7) // 8
    packed = np.zeros((count, row_bytes), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="indoor-los-") as temp:
        library = _compile_los(Path(temp))
        open_ptr = walkable.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        anchor_ptr = anchors.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        component_ptr = components.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        packed_ptr = packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        visible_pairs = 0
        started = time.monotonic()
        for first in range(0, count, chunk_size):
            last = min(first + chunk_size, count)
            visible_pairs += library.indoor_visibility_rows(
                open_ptr, *walkable.shape, anchor_ptr, component_ptr,
                count, packed_ptr, row_bytes, first, last)
            if last == count or last % 1024 < chunk_size:
                elapsed = time.monotonic() - started
                print(f"  visibility: {last}/{count} observers; "
                      f"{elapsed:.1f}s elapsed", flush=True)
    return packed, int(visible_pairs)


def visible_node_indices(packed: np.ndarray, observer_index: int,
                         node_count: int) -> np.ndarray:
    """Decode one visibility row, in the same node order as graph.pkl.gz."""
    bits = np.unpackbits(packed[observer_index], bitorder="little")
    return np.flatnonzero(bits[:node_count])


def visible_edges_for(graph: nx.DiGraph, nodes: list, packed: np.ndarray,
                      observer_index: int) -> tuple:
    """Induce the usual visible-edge set on demand from visible nodes."""
    visible = {nodes[i] for i in visible_node_indices(
        packed, observer_index, len(nodes))}
    return tuple((u, v) for u in visible for v in graph.successors(u)
                 if v in visible)


def write_prepared_map(graph: nx.DiGraph, nodes: list, packed: np.ndarray,
                       metadata: dict, path: Path) -> Path:
    """Write one portable map with compact visibility in the graph payload."""
    if path.exists():
        raise FileExistsError(f"prepared map already exists: {path}")
    graph.graph["packed_visibility"] = np.ascontiguousarray(packed)
    graph.graph["visibility_nodes"] = tuple(nodes)
    graph.graph["visibility_index"] = {
        node: index for index, node in enumerate(nodes)
    }
    return save_prepared_map(
        graph, {key: value for key, value in metadata.items()
                if key not in {"artifact_note", "prepared_map"}}, path)


def _combined_fine_path(graph: nx.DiGraph, route: list) -> tuple:
    """Concatenate edge routes without skipping or duplicating fine pixels."""
    fine_path = []
    for first, second in zip(route, route[1:]):
        segment = graph.edges[first, second]["fine_path"]
        if fine_path:
            assert fine_path[-1] == segment[0]
            fine_path.extend(segment[1:])
        else:
            fine_path.extend(segment)
    return tuple(fine_path)


def prune_split_patches(graph: nx.DiGraph, nodes: list,
                        packed: np.ndarray):
    """Keep one node per occupied patch and route through removed pieces.

    Every shortcut is an actual path in the original graph with the exact
    fine-pixel route and cost. No shortcut is inferred from geometric proximity.
    """
    by_patch = {}
    for node in nodes:
        by_patch.setdefault(node[:2], []).append(node)
    keep = {
        max(parts, key=lambda node: (
            graph.nodes[node]["fine_pixel_count"], graph.degree(node),
            -node[2]))
        for parts in by_patch.values()
    }
    retained_nodes = [node for node in nodes if node in keep]
    removed = set(graph) - keep
    pruned = graph.subgraph(keep).copy()
    pruned.graph.clear()
    shortcuts_added = 0
    shortcuts_improved = 0
    removed_components = list(nx.weakly_connected_components(
        graph.subgraph(removed)))
    for component in removed_components:
        boundary = sorted({
            neighbor for node in component
            for neighbor in (*graph.predecessors(node), *graph.successors(node))
            if neighbor in keep
        })
        if len(boundary) < 2:
            continue
        boundary_set = set(boundary)
        for source in boundary:
            distances, routes = nx.single_source_dijkstra(
                graph.subgraph(component | {source}), source,
                weight="distance")
            for interior in component:
                if interior not in distances:
                    continue
                for destination in graph.successors(interior):
                    if destination == source or destination not in boundary_set:
                        continue
                    route = routes[interior] + [destination]
                    distance = distances[interior] + graph.edges[
                        interior, destination]["distance"]
                    previous = pruned.get_edge_data(source, destination)
                    if previous is not None and previous["distance"] <= distance:
                        continue
                    fine_path = _combined_fine_path(graph, route)
                    assert len(fine_path) - 1 == distance
                    pruned.add_edge(
                        source, destination, distance=float(distance),
                        fine_path=fine_path, is_road=False,
                        observed_edge=False, num_used=1.0,
                        shortcut_via=tuple(route[1:-1]))
                    if previous is None:
                        shortcuts_added += 1
                    else:
                        shortcuts_improved += 1

    # Check that shortcuts preserve exactly the original connectivity among
    # surviving nodes, without linking separate fine-map components.
    original_components = {}
    original_component_sets = list(nx.weakly_connected_components(graph))
    for index, component in enumerate(original_component_sets):
        for node in component & keep:
            original_components[node] = index
    result_components = {}
    for index, component in enumerate(nx.weakly_connected_components(pruned)):
        for node in component:
            result_components[node] = index
    original_to_result = {}
    result_to_original = {}
    for node in keep:
        original = original_components[node]
        result = result_components[node]
        assert original_to_result.setdefault(original, result) == result
        assert result_to_original.setdefault(result, original) == original

    relabel = {node: node[:2] for node in retained_nodes}
    assert len(set(relabel.values())) == len(relabel)
    pruned = nx.relabel_nodes(pruned, relabel, copy=True)
    kept_indices = np.asarray(
        [index for index, node in enumerate(nodes) if node in keep],
        dtype=np.int32)
    new_count = len(retained_nodes)
    new_packed = np.empty((new_count, (new_count + 7) // 8), dtype=np.uint8)
    for new_index, old_index in enumerate(kept_indices):
        visible = np.unpackbits(packed[old_index], bitorder="little")
        new_packed[new_index] = np.packbits(
            visible[kept_indices], bitorder="little")
        pruned.nodes[relabel[retained_nodes[new_index]]][
            "visibility_index"] = new_index
        pruned.nodes[relabel[retained_nodes[new_index]]][
            "original_node"] = retained_nodes[new_index]

    report = {
        "original_nodes": len(nodes),
        "retained_nodes": new_count,
        "removed_split_nodes": len(removed),
        "removed_piece_components": len(removed_components),
        "shortcuts_added": shortcuts_added,
        "shortcuts_improved": shortcuts_improved,
        "original_weak_components": len(original_component_sets),
        "retained_weak_components": len(set(result_components.values())),
        "removed_pixels": sum(graph.nodes[node]["fine_pixel_count"]
                              for node in removed),
    }
    return pruned, [relabel[node] for node in retained_nodes], new_packed, report


def prepare_existing_bundle(output_dir: Path) -> Path:
    """Convert existing graph and LOS files without repeating map compilation."""
    prepared_path = output_dir / "prepared_map.pkl.gz"
    if prepared_path.exists():
        raise FileExistsError(f"prepared map already exists: {prepared_path}")
    with gzip.open(output_dir / "graph.pkl.gz", "rb") as stream:
        payload = pickle.load(stream)
    with np.load(output_dir / "visibility.npz") as visibility:
        packed = visibility["packed"]
    metadata = json.loads((output_dir / "report.json").read_text())
    return write_prepared_map(
        payload["graph"], payload["nodes"], packed, metadata,
        prepared_path)


def prune_existing_bundle(output_dir: Path) -> Path:
    """Build a two-integer-node variant from an existing diagnostic bundle."""
    prepared_path = output_dir / "prepared_map_pruned.pkl.gz"
    if prepared_path.exists():
        raise FileExistsError(f"prepared map already exists: {prepared_path}")
    with gzip.open(output_dir / "graph.pkl.gz", "rb") as stream:
        payload = pickle.load(stream)
    with np.load(output_dir / "visibility.npz") as visibility:
        packed = visibility["packed"]
    graph, nodes, new_packed, pruning = prune_split_patches(
        payload["graph"], payload["nodes"], packed)
    metadata = json.loads((output_dir / "report.json").read_text())
    metadata.update({
        "node_count": len(nodes),
        "directed_edge_count": graph.number_of_edges(),
        "pruning": pruning,
    })
    write_prepared_map(graph, nodes, new_packed, metadata, prepared_path)
    (output_dir / "pruning_report.json").write_text(
        json.dumps(pruning, indent=2) + "\n")
    render_pruned_graph(
        graph, np.load(output_dir / "walkable.npy", allow_pickle=False),
        output_dir / "pruned_graph.png")
    return prepared_path


def render_pruned_graph(graph: nx.DiGraph, walkable: np.ndarray,
                        output_path: Path) -> Path:
    """Show retained nodes and exact fine-pixel routes for shortcut edges."""
    from PIL import Image, ImageDraw

    if output_path.exists():
        raise FileExistsError(f"pruned graph image already exists: {output_path}")
    base = np.where(walkable, 244, 32).astype(np.uint8)
    image = Image.fromarray(np.repeat(base[:, :, None], 3, axis=2))
    image = image.resize((walkable.shape[1] * 2, walkable.shape[0] * 2),
                         resample=Image.Resampling.NEAREST)
    drawing = ImageDraw.Draw(image)
    for first, second, data in graph.edges(data=True):
        if first >= second:
            continue
        coordinates = [(2 * col, 2 * row) for row, col in data["fine_path"]]
        if "shortcut_via" in data:
            drawing.line(coordinates, fill=(236, 93, 34), width=2)
        else:
            drawing.line(coordinates, fill=(113, 169, 190), width=1)
    for _, data in graph.nodes(data=True):
        row, col = data["dense_anchor"]
        x, y = 2 * col, 2 * row
        drawing.ellipse((x - 1, y - 1, x + 1, y + 1),
                        fill=(13, 69, 100))
    image.save(output_path)
    return output_path


def _nearest_observer(anchors: np.ndarray, xy):
    col, row = xy
    squared = ((anchors[:, 0] - row) ** 2 + (anchors[:, 1] - col) ** 2)
    return int(squared.argmin())


def render_diagnostics(walkable, graph, nodes, anchors, packed, output_dir,
                       patch_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    height, width = walkable.shape
    base = np.where(walkable, 255, 35).astype(np.uint8)
    coordinates = anchors[:, [1, 0]].astype(float)
    undirected = [(u, v, data) for u, v, data in graph.edges(data=True)
                  if graph.nodes[u]["visibility_index"]
                  < graph.nodes[v]["visibility_index"]]
    edge_segments = [np.asarray([(col, row) for row, col in data["fine_path"]],
                                dtype=float) for _, _, data in undirected]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    for axis in axes:
        axis.imshow(base, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_xlim(0, width); axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_xlabel("Fine pixel column"); axis.set_ylabel("Fine pixel row")
    axes[0].set_title("Fine occupancy used for path cost and visibility")
    axes[1].add_collection(LineCollection(
        edge_segments, colors="#e34a33", linewidths=0.38, alpha=0.7,
        rasterized=True))
    axes[1].scatter(coordinates[:, 0], coordinates[:, 1],
                    s=0.35, c="#168aad", alpha=0.75, rasterized=True)
    axes[1].set_title(f"{len(nodes):,} component-aware coarse nodes; "
                      f"{len(undirected):,} physical connections")
    fig.savefig(output_dir / "coarse_graph.png", dpi=190)
    plt.close(fig)

    # Crops reveal whether edges respect the many narrow rooms and doorways.
    regions = ((150, 350, 105, 270), (650, 855, 220, 430))
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for axis, (xmin, xmax, ymin, ymax) in zip(axes, regions):
        axis.imshow(base, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.add_collection(LineCollection(
            edge_segments, colors="#e34a33", linewidths=0.8,
            alpha=0.8, rasterized=True))
        axis.scatter(coordinates[:, 0], coordinates[:, 1],
                     s=2.0, c="#168aad", rasterized=True)
        axis.set_xlim(xmin, xmax); axis.set_ylim(ymax, ymin)
        axis.set_aspect("equal")
        axis.set_title("Coarse routes through fine-pixel openings")
    fig.savefig(output_dir / "coarse_graph_detail.png", dpi=180)
    plt.close(fig)

    examples = ((430, 120), (270, 195), (400, 550), (710, 440))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for axis, point in zip(axes.flat, examples):
        index = _nearest_observer(anchors, point)
        seen = visible_node_indices(packed, index, len(nodes))
        axis.imshow(base, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.scatter(coordinates[seen, 0], coordinates[seen, 1],
                     s=1.5, c="#1f9aca", alpha=0.75, rasterized=True)
        axis.scatter(*coordinates[index], s=65, c="#ec202a", marker="x",
                     linewidths=2.4)
        axis.set_xlim(0, width); axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_title(f"Observer {tuple(map(int, anchors[index]))}: "
                       f"{len(seen):,} visible coarse nodes")
    fig.savefig(output_dir / "visibility_examples.png", dpi=190)
    plt.close(fig)

    counts = np.bitwise_count(packed).sum(axis=1)
    heat = np.full((height // patch_size, width // patch_size), np.nan)
    for node, count in zip(nodes, counts):
        row, col, _ = node
        if math.isnan(heat[row, col]) or count > heat[row, col]:
            heat[row, col] = int(count)
    fig, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(heat, cmap="viridis", interpolation="nearest")
    axis.set_title("Coarse nodes visible from each location")
    fig.colorbar(image, ax=axis, label="Visible nodes")
    fig.savefig(output_dir / "visibility_counts.png", dpi=190)
    plt.close(fig)
    return {
        "visible_nodes_min": int(counts.min()),
        "visible_nodes_median": float(np.median(counts)),
        "visible_nodes_max": int(counts.max()),
    }


def generate(input_dir: Path, output_dir: Path, patch_size: int = 5):
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; select a new path")
    walkable, input_info = load_walkable(input_dir)
    print("Building component-aware coarse graph", flush=True)
    graph, owners, nodes, anchors, components, graph_info = build_graph(
        walkable, patch_size)
    print(f"  {len(nodes):,} nodes, {graph.number_of_edges():,} directed edges; "
          f"{graph_info['split_patches']} split patches", flush=True)
    print("Computing full-resolution line of sight", flush=True)
    packed, visible_pairs = compute_visibility(walkable, anchors, components)
    output_dir.mkdir(parents=True)
    with gzip.open(output_dir / "graph.pkl.gz", "wb") as stream:
        pickle.dump({"graph": graph, "nodes": nodes}, stream,
                    protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(output_dir / "visibility.npz", packed=packed,
                        anchors=anchors, components=components)
    np.save(output_dir / "walkable.npy", walkable)
    print("Rendering diagnostics", flush=True)
    visual_info = render_diagnostics(
        walkable, graph, nodes, anchors, packed, output_dir, patch_size)
    report = {
        "source": str(input_dir.resolve()),
        **input_info, **graph_info,
        "visibility_backend": "strict supercover line of sight on fine pixels",
        "visible_unordered_pairs": visible_pairs,
        **visual_info,
    }
    prepared_path = write_prepared_map(
        graph, nodes, packed, report, output_dir / "prepared_map.pkl.gz")
    report["prepared_map"] = str(prepared_path)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=5)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--prepare-existing", action="store_true",
                          help="convert an existing output bundle without recalculating LOS")
    existing.add_argument("--prune-existing", action="store_true",
                          help="write a one-node-per-patch map with path-backed shortcuts")
    args = parser.parse_args(argv)
    if args.prepare_existing:
        print(prepare_existing_bundle(args.output_dir))
    elif args.prune_existing:
        print(prune_existing_bundle(args.output_dir))
    else:
        generate(args.input_dir, args.output_dir, args.patch_size)


if __name__ == "__main__":
    main()
