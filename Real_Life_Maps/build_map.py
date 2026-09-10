"""Offline compiler for simulator-ready real-terrain graphs.

The compiler keeps the simulator's 64x64 planning topology, but derives its
directed travel times from a denser physical raster and its visibility from
GRASS ``r.viewshed``.  It is deliberately an explicit build step: simulation,
training, and evaluation only load the resulting prepared-map artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
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
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin, xy
from rasterio.warp import reproject

from Real_Life_Maps.prepared_map import (
    DEFAULT_PREPARED_MAP_PATH,
    PREPARED_MAP_SCHEMA,
    save_prepared_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_MAP_ROOT = Path(__file__).resolve().parent
DEFAULT_DEM_PATH = REAL_MAP_ROOT / "WV_DEM.tif"
DEFAULT_ROAD_PATH = REAL_MAP_ROOT / "WV_roads.pkl"
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT / "outputs" / "map_diagnostics" / "wv_64"

TOBLER_BASE_KPH = 6.0
TOBLER_SLOPE_DECAY = 3.5
TOBLER_SLOPE_OFFSET = 0.05


@dataclass(frozen=True)
class DenseTerrain:
    elevation_m: np.ndarray
    transform: object
    crs: object
    source_bounds: tuple[float, float, float, float]

    @property
    def height(self) -> int:
        return int(self.elevation_m.shape[0])

    @property
    def width(self) -> int:
        return int(self.elevation_m.shape[1])

    @property
    def cell_width_m(self) -> float:
        return abs(float(self.transform.a))

    @property
    def cell_height_m(self) -> float:
        return abs(float(self.transform.e))


def tobler_speed_mps(grade: float, surface_factor: float = 1.0) -> float:
    """Return Tobler walking speed for directed grade, in meters/second."""
    if not math.isfinite(grade):
        raise ValueError("grade must be finite")
    if not math.isfinite(surface_factor) or surface_factor <= 0:
        raise ValueError("surface_factor must be finite and positive")
    speed_kph = (
        surface_factor * TOBLER_BASE_KPH
        * math.exp(-TOBLER_SLOPE_DECAY * abs(grade + TOBLER_SLOPE_OFFSET))
    )
    return speed_kph / 3.6


def read_dense_terrain(dem_path, dense_cell_size_m: float) -> DenseTerrain:
    """Read a projected DEM into a north-up dense physical analysis raster."""
    if not math.isfinite(dense_cell_size_m) or dense_cell_size_m <= 0:
        raise ValueError("dense_cell_size_m must be finite and positive")
    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError("the terrain DEM must use a projected CRS")
        if dataset.crs.linear_units.lower() not in ("metre", "meter"):
            raise ValueError("the terrain DEM projected units must be meters")
        bounds = dataset.bounds
        width = max(2, int(math.ceil((bounds.right - bounds.left)
                                     / dense_cell_size_m)))
        height = max(2, int(math.ceil((bounds.top - bounds.bottom)
                                      / dense_cell_size_m)))
        source = dataset.read(1, masked=True).filled(np.nan)
        elevation = np.full((height, width), np.nan, dtype=np.float64)
        # GRASS r.viewshed assumes square cells.  Reprojecting onto this exact
        # metric lattice avoids the subtly rectangular pixels produced by an
        # out_shape read when the source bounds are not exact size multiples.
        transform = from_origin(
            bounds.left, bounds.top, dense_cell_size_m, dense_cell_size_m
        )
        reproject(
            source=np.asarray(source, dtype=np.float64),
            destination=elevation,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=dataset.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        crs = dataset.crs
    if not np.isfinite(elevation).any():
        raise ValueError("the terrain DEM contains no finite elevations")
    return DenseTerrain(
        elevation_m=elevation,
        transform=transform,
        crs=crs,
        source_bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
    )


def load_road_data(path, coarse_size: int) -> tuple[set, set]:
    """Load the existing coarse road-node and explicit road-edge sets."""
    if path is None:
        return set(), set()
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("n_size") != coarse_size:
        raise ValueError(
            f"road data was built for {payload.get('n_size')!r}, not the "
            f"requested {coarse_size}x{coarse_size} map"
        )
    road_nodes = {tuple(node) for node in payload.get("road_nodes", ())}
    road_edges = {
        (tuple(edge[0]), tuple(edge[1]))
        for edge in payload.get("road_edges", ())
    }
    return road_nodes, road_edges


def _raw_coarse_cell(node, coarse_size: int) -> tuple[int, int]:
    """Undo the historical clockwise rotation of coarse graph coordinates."""
    row, col = node
    return coarse_size - 1 - col, row


def _dense_axis_bounds(index: int, coarse_size: int, length: int) -> tuple[int, int]:
    start = int(math.floor(index * length / coarse_size))
    stop = int(math.ceil((index + 1) * length / coarse_size))
    return max(0, start), min(length, max(start + 1, stop))


def dense_anchor(node, coarse_size: int, terrain: DenseTerrain) -> tuple[int, int]:
    raw_row, raw_col = _raw_coarse_cell(node, coarse_size)
    row_start, row_stop = _dense_axis_bounds(
        raw_row, coarse_size, terrain.height)
    col_start, col_stop = _dense_axis_bounds(
        raw_col, coarse_size, terrain.width)
    return (row_start + row_stop - 1) // 2, (col_start + col_stop - 1) // 2


def _corridor_bounds(first, second, coarse_size, terrain, halo):
    raw_cells = [
        _raw_coarse_cell(first, coarse_size),
        _raw_coarse_cell(second, coarse_size),
    ]
    row_ranges = [
        _dense_axis_bounds(row, coarse_size, terrain.height)
        for row, _col in raw_cells
    ]
    col_ranges = [
        _dense_axis_bounds(col, coarse_size, terrain.width)
        for _row, col in raw_cells
    ]
    row_start = max(0, min(value[0] for value in row_ranges) - halo)
    row_stop = min(terrain.height, max(value[1] for value in row_ranges) + halo)
    col_start = max(0, min(value[0] for value in col_ranges) - halo)
    col_stop = min(terrain.width, max(value[1] for value in col_ranges) + halo)
    return row_start, row_stop, col_start, col_stop


def _step_metrics(terrain, first, second, surface_factor):
    first_row, first_col = first
    second_row, second_col = second
    horizontal = math.hypot(
        (second_col - first_col) * terrain.cell_width_m,
        (second_row - first_row) * terrain.cell_height_m,
    )
    dz = float(
        terrain.elevation_m[second_row, second_col]
        - terrain.elevation_m[first_row, first_col]
    )
    grade = dz / horizontal
    surface = math.hypot(horizontal, dz)
    seconds = surface / tobler_speed_mps(grade, surface_factor)
    return seconds, horizontal, surface


def _least_time_corridor(
        terrain, start, goal, bounds, surface_factor
        ) -> tuple[float, float, float]:
    """Find a local 8-connected dense route and return time/length metrics."""
    if start == goal:
        raise ValueError("dense analysis resolution is too coarse for this map")
    row_start, row_stop, col_start, col_stop = bounds
    rows, cols = row_stop - row_start, col_stop - col_start
    distances = np.full((rows, cols), np.inf, dtype=np.float64)
    predecessor = {}
    local_start = start[0] - row_start, start[1] - col_start
    distances[local_start] = 0.0
    queue = [(0.0, start)]
    offsets = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    while queue:
        cost, current = heapq.heappop(queue)
        local = current[0] - row_start, current[1] - col_start
        if cost != distances[local]:
            continue
        if current == goal:
            break
        for dr, dc in offsets:
            neighbor = current[0] + dr, current[1] + dc
            if not (row_start <= neighbor[0] < row_stop
                    and col_start <= neighbor[1] < col_stop):
                continue
            if not (np.isfinite(terrain.elevation_m[current])
                    and np.isfinite(terrain.elevation_m[neighbor])):
                continue
            step_time, _horizontal, _surface = _step_metrics(
                terrain, current, neighbor, surface_factor)
            candidate = cost + step_time
            neighbor_local = (
                neighbor[0] - row_start, neighbor[1] - col_start
            )
            if candidate < distances[neighbor_local]:
                distances[neighbor_local] = candidate
                predecessor[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    goal_local = goal[0] - row_start, goal[1] - col_start
    travel_time = float(distances[goal_local])
    if not math.isfinite(travel_time):
        return math.inf, math.inf, math.inf
    horizontal_total = 0.0
    surface_total = 0.0
    current = goal
    while current != start:
        previous = predecessor[current]
        _seconds, horizontal, surface = _step_metrics(
            terrain, previous, current, surface_factor)
        horizontal_total += horizontal
        surface_total += surface
        current = previous
    return travel_time, horizontal_total, surface_total


def build_travel_graph(
        terrain: DenseTerrain, coarse_size: int, road_nodes, road_edges,
        rough_terrain_factor: float = 0.6, road_factor: float = 1.0,
        corridor_halo: int = 1, progress: bool = False) -> nx.DiGraph:
    """Aggregate dense Tobler routes into a coarse directed planning graph."""
    if terrain.height < coarse_size or terrain.width < coarse_size:
        raise ValueError(
            "dense raster must have at least one distinct cell per coarse row "
            "and column"
        )
    if corridor_halo < 0:
        raise ValueError("corridor_halo must be nonnegative")
    for name, value in (
        ("rough_terrain_factor", rough_terrain_factor),
        ("road_factor", road_factor),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    graph = nx.grid_2d_graph(
        coarse_size, coarse_size, create_using=nx.DiGraph)
    finite = terrain.elevation_m[np.isfinite(terrain.elevation_m)]
    minimum, maximum = float(finite.min()), float(finite.max())
    relief = maximum - minimum
    anchors = {}
    for node in graph:
        anchor = dense_anchor(node, coarse_size, terrain)
        anchors[node] = anchor
        elevation = float(terrain.elevation_m[anchor])
        map_x, map_y = xy(terrain.transform, *anchor, offset="center")
        normalized = 0.0 if relief == 0 else (elevation - minimum) / relief * 10.0
        graph.nodes[node].update(
            pos=node,
            type="intermediate",
            height=float(normalized),
            elevation_m=elevation,
            map_x_m=float(map_x),
            map_y_m=float(map_y),
            is_road=node in road_nodes,
            dense_anchor=anchor,
        )

    edges = list(graph.edges)
    for index, (u, v) in enumerate(edges, start=1):
        is_road = (u, v) in road_edges
        factor = road_factor if is_road else rough_terrain_factor
        bounds = _corridor_bounds(
            u, v, coarse_size, terrain, corridor_halo)
        seconds, horizontal, surface = _least_time_corridor(
            terrain, anchors[u], anchors[v], bounds, factor)
        if not math.isfinite(seconds):
            graph.remove_edge(u, v)
            continue
        elevation_change = (
            graph.nodes[v]["elevation_m"] - graph.nodes[u]["elevation_m"]
        )
        graph.edges[u, v].update(
            distance=float(seconds),
            travel_time_s=float(seconds),
            horizontal_distance_m=float(horizontal),
            surface_distance_m=float(surface),
            elevation_change_m=float(elevation_change),
            mean_grade=float(elevation_change / horizontal),
            surface_factor=float(factor),
            is_road=bool(is_road),
            observed_edge=False,
            num_used=1.0,
        )
        if progress and index % 1024 == 0:
            print(f"  travel edges: {index}/{len(edges)}", flush=True)
    return graph


def _write_dense_geotiff(terrain: DenseTerrain, path: Path) -> None:
    values = np.where(
        np.isfinite(terrain.elevation_m), terrain.elevation_m, -9999.0
    ).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=terrain.height,
        width=terrain.width, count=1, dtype="float32", crs=terrain.crs,
        transform=terrain.transform, nodata=-9999.0, compress="deflate",
    ) as dataset:
        dataset.write(values, 1)


def _resolve_grass_command(grass_command: str) -> str:
    """Resolve GRASS from PATH or a standard macOS application bundle."""
    resolved = shutil.which(grass_command)
    if resolved is not None:
        return resolved
    command_path = Path(grass_command).expanduser()
    if command_path.is_file():
        return str(command_path.resolve())
    if grass_command == "grass":
        applications = Path("/Applications")
        if applications.is_dir():
            candidates = sorted(
                applications.glob(
                    "GRASS-*.app/Contents/Resources/bin/grass"
                ),
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
    raise FileNotFoundError(
        f"GRASS executable {grass_command!r} was not found. Install GRASS "
        "GIS or pass --grass-command to the map builder."
    )


def compute_node_visibility_grass(
        terrain: DenseTerrain, graph: nx.DiGraph, coarse_size: int,
        max_distance_m: float, observer_height_m: float,
        target_height_m: float, grass_command: str = "grass",
        grass_memory_mb: int = 500) -> np.ndarray:
    """Run GRASS once as a session and return coarse node visibility rows."""
    if not math.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("max_distance_m must be finite and positive")
    for name, value in (
        ("observer_height_m", observer_height_m),
        ("target_height_m", target_height_m),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if grass_memory_mb < 1:
        raise ValueError("grass_memory_mb must be positive")
    resolved_grass = _resolve_grass_command(grass_command)
    nodes = list(graph)
    anchors = np.asarray(
        [dense_anchor(node, coarse_size, terrain) for node in nodes],
        dtype=np.int32,
    )
    coordinates = np.asarray([
        xy(terrain.transform, int(row), int(col), offset="center")
        for row, col in anchors
    ], dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="terrain-viewshed-") as directory:
        directory = Path(directory)
        dense_path = directory / "dense_dem.tif"
        anchor_path = directory / "anchors.npz"
        output_path = directory / "visibility.npy"
        _write_dense_geotiff(terrain, dense_path)
        np.savez(
            anchor_path, rows=anchors[:, 0], cols=anchors[:, 1],
            coordinates=coordinates,
        )
        command = [
            resolved_grass,
            "--tmp-project", str(dense_path),
            "--exec", sys.executable, "-m",
            "Real_Life_Maps.grass_viewshed_worker",
            "--dem", str(dense_path),
            "--anchors", str(anchor_path),
            "--output", str(output_path),
            "--max-distance", str(float(max_distance_m)),
            "--observer-height", str(float(observer_height_m)),
            "--target-height", str(float(target_height_m)),
            "--memory", str(int(grass_memory_mb)),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        visibility = np.load(output_path, allow_pickle=False)
    expected = (len(nodes), len(nodes))
    if visibility.shape != expected:
        raise ValueError(
            f"GRASS visibility output has shape {visibility.shape}, expected "
            f"{expected}"
        )
    visibility = np.asarray(visibility, dtype=bool)
    np.fill_diagonal(visibility, True)
    return visibility


def attach_node_first_visibility(
        graph: nx.DiGraph, visibility: np.ndarray) -> None:
    """Store visible nodes first, then induce every visible directed edge."""
    nodes = list(graph)
    if visibility.shape != (len(nodes), len(nodes)):
        raise ValueError("visibility matrix does not align with graph nodes")
    for index, observer in enumerate(nodes):
        row = np.asarray(visibility[index], dtype=bool).copy()
        row[index] = True
        visible_nodes = tuple(
            node for node, visible in zip(nodes, row) if visible
        )
        visible_set = set(visible_nodes)
        visible_edges = tuple(
            (u, v) for u in visible_nodes
            for v in graph.successors(u) if v in visible_set
        )
        graph.nodes[observer]["visible_nodes"] = visible_nodes
        graph.nodes[observer]["visible_edges"] = visible_edges


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_diagnostics(
        graph: nx.DiGraph, terrain: DenseTerrain, metadata: dict,
        output_dir) -> None:
    """Write visual and numeric QA artifacts for one explicit map build."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_size = int(metadata["coarse_size"])
    elevation = np.asarray([
        graph.nodes[row, col]["elevation_m"]
        for row in range(coarse_size) for col in range(coarse_size)
    ]).reshape(coarse_size, coarse_size)
    outgoing = np.asarray([
        np.mean([graph.edges[node, other]["distance"]
                 for other in graph.successors(node)])
        for node in graph
    ]).reshape(coarse_size, coarse_size)
    road = np.zeros((coarse_size, coarse_size), dtype=float)
    for u, v, data in graph.edges(data=True):
        if data["is_road"]:
            road[u] += 0.5
            road[v] += 0.5
    gy, gx = np.gradient(
        terrain.elevation_m,
        terrain.cell_height_m,
        terrain.cell_width_m,
    )
    slope_degrees = np.degrees(np.arctan(np.hypot(gx, gy)))
    # Coarse graph coordinates retain the historical clockwise DEM rotation.
    # Rotate dense QA panels the same way so terrain and graph products can be
    # compared visually without an implicit coordinate transform.
    dense_elevation = np.rot90(terrain.elevation_m, k=-1)
    dense_slope = np.rot90(slope_degrees, k=-1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    panels = (
        (dense_elevation, "Dense elevation (simulator orientation)",
         "terrain", "Elevation (m)"),
        (dense_slope, "Dense slope (simulator orientation)",
         "magma", "Slope (degrees)"),
        (outgoing, "Mean outgoing coarse travel time", "viridis", "Seconds"),
        (road, "Explicit coarse road connections", "Greys", "Road degree"),
    )
    for axis, (values, title, cmap, label) in zip(axes.flat, panels):
        image = axis.imshow(values, cmap=cmap, origin="upper")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.8, label=label)
    fig.savefig(output_dir / "terrain_diagnostics.png", dpi=170)
    plt.close(fig)

    center = (coarse_size // 2, coarse_size // 2)
    lowest = min(graph, key=lambda node: graph.nodes[node]["elevation_m"])
    highest = max(graph, key=lambda node: graph.nodes[node]["elevation_m"])
    observers = tuple(dict.fromkeys((lowest, center, highest)))
    fig, axes = plt.subplots(
        1, len(observers), figsize=(6 * len(observers), 5),
        constrained_layout=True, squeeze=False)
    for axis, observer in zip(axes[0], observers):
        mask = np.zeros((coarse_size, coarse_size), dtype=np.uint8)
        for node in graph.nodes[observer]["visible_nodes"]:
            mask[node] = 1
        axis.imshow(elevation, cmap="terrain", origin="upper", alpha=0.75)
        axis.imshow(
            np.ma.masked_where(mask == 0, mask), cmap="Blues", origin="upper",
            alpha=0.55, vmin=0, vmax=1)
        axis.scatter(observer[1], observer[0], c="red", marker="x", s=80)
        axis.set_title(
            f"Observer {observer}\n{int(mask.sum())} visible nodes")
    fig.savefig(output_dir / "visibility_diagnostics.png", dpi=170)
    plt.close(fig)

    travel_times = np.asarray([
        float(data["distance"]) for _u, _v, data in graph.edges(data=True)
    ])
    road_times = np.asarray([
        float(data["distance"]) for _u, _v, data in graph.edges(data=True)
        if data["is_road"]
    ])
    rough_times = np.asarray([
        float(data["distance"]) for _u, _v, data in graph.edges(data=True)
        if not data["is_road"]
    ])
    visibility_counts = np.asarray([
        len(graph.nodes[node]["visible_nodes"]) for node in graph
    ])

    def describe(values):
        if len(values) == 0:
            return {"count": 0}
        return {
            "count": int(len(values)),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
        }

    report = {
        "metadata": metadata,
        "travel_time_s": describe(travel_times),
        "road_travel_time_s": describe(road_times),
        "rough_travel_time_s": describe(rough_times),
        "visible_node_count": describe(visibility_counts),
    }
    (output_dir / "map_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_map(args) -> tuple[nx.DiGraph, dict]:
    started = time.perf_counter()
    print(f"Loading dense terrain at approximately {args.dense_cell_size:g} m")
    terrain = read_dense_terrain(args.dem, args.dense_cell_size)
    road_nodes, road_edges = load_road_data(args.roads, args.coarse_size)
    print(
        f"Building {args.coarse_size}x{args.coarse_size} directed travel graph "
        f"from {terrain.height}x{terrain.width} dense cells"
    )
    graph = build_travel_graph(
        terrain, args.coarse_size, road_nodes, road_edges,
        rough_terrain_factor=args.rough_terrain_factor,
        road_factor=args.road_factor,
        corridor_halo=args.corridor_halo,
        progress=True,
    )
    print("Computing GRASS node viewsheds")
    visibility = compute_node_visibility_grass(
        terrain, graph, args.coarse_size,
        max_distance_m=args.visibility_distance,
        observer_height_m=args.observer_height,
        target_height_m=args.target_height,
        grass_command=args.grass_command,
        grass_memory_mb=args.grass_memory,
    )
    print("Deriving visible edges from visible nodes")
    attach_node_first_visibility(graph, visibility)
    metadata = {
        "schema_version": PREPARED_MAP_SCHEMA,
        "builder": "Real_Life_Maps.build_map",
        "coarse_size": int(args.coarse_size),
        "dense_shape": [terrain.height, terrain.width],
        "dense_cell_width_m": terrain.cell_width_m,
        "dense_cell_height_m": terrain.cell_height_m,
        "crs": terrain.crs.to_string(),
        "bounds": list(terrain.source_bounds),
        "height_units": "meters",
        "distance_units": "seconds",
        "dem_sha256": _file_sha256(args.dem),
        "roads_sha256": _file_sha256(args.roads),
        "tobler_base_kph": TOBLER_BASE_KPH,
        "tobler_slope_decay": TOBLER_SLOPE_DECAY,
        "tobler_slope_offset": TOBLER_SLOPE_OFFSET,
        "rough_terrain_factor": float(args.rough_terrain_factor),
        "road_factor": float(args.road_factor),
        "dense_movement_neighbors": 8,
        "corridor_halo_dense_cells": int(args.corridor_halo),
        "visibility_backend": "GRASS r.viewshed",
        "visibility_node_first": True,
        "visibility_edge_rule": "both endpoints visible",
        "visibility_distance_m": float(args.visibility_distance),
        "observer_height_m": float(args.observer_height),
        "target_height_m": float(args.target_height),
        "earth_curvature": False,
        "atmospheric_refraction": False,
        "build_seconds": float(time.perf_counter() - started),
    }
    return graph, metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a simulator-ready map using dense Tobler costs and "
                    "GRASS node viewsheds."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH)
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROAD_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED_MAP_PATH)
    parser.add_argument("--coarse-size", type=int, default=64)
    parser.add_argument("--dense-cell-size", type=float, default=10.0,
                        help="approximate dense analysis cell size in meters")
    parser.add_argument("--rough-terrain-factor", type=float, default=0.6)
    parser.add_argument("--road-factor", type=float, default=1.0)
    parser.add_argument("--corridor-halo", type=int, default=1,
                        help="dense cells added around each two-cell corridor")
    parser.add_argument("--visibility-distance", type=float, default=1500.0)
    parser.add_argument("--observer-height", type=float, default=1.7)
    parser.add_argument("--target-height", type=float, default=0.0)
    parser.add_argument("--grass-command", default="grass")
    parser.add_argument("--grass-memory", type=int, default=500)
    parser.add_argument("--diagnostics-dir", type=Path,
                        default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.coarse_size < 2:
        parser.error("--coarse-size must be at least 2")
    for option, value in (
        ("--dense-cell-size", args.dense_cell_size),
        ("--rough-terrain-factor", args.rough_terrain_factor),
        ("--road-factor", args.road_factor),
        ("--visibility-distance", args.visibility_distance),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{option} must be finite and positive")
    for option, value in (
        ("--observer-height", args.observer_height),
        ("--target-height", args.target_height),
    ):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{option} must be finite and nonnegative")
    if args.corridor_halo < 0:
        parser.error("--corridor-halo must be nonnegative")
    if args.grass_memory < 1:
        parser.error("--grass-memory must be positive")
    for path, label in ((args.dem, "DEM"), (args.roads, "road data")):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    graph, metadata = build_map(args)
    output = save_prepared_map(graph, metadata, args.output)
    if not args.no_diagnostics:
        print(f"Writing diagnostics to {args.diagnostics_dir}")
        build_diagnostics(graph, read_dense_terrain(
            args.dem, args.dense_cell_size), metadata, args.diagnostics_dir)
    print(f"Prepared map: {output}")
    print(f"Build time: {metadata['build_seconds']:.1f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DenseTerrain",
    "attach_node_first_visibility",
    "build_diagnostics",
    "build_travel_graph",
    "compute_node_visibility_grass",
    "dense_anchor",
    "load_road_data",
    "read_dense_terrain",
    "tobler_speed_mps",
]
