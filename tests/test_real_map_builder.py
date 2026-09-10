"""Focused checks for offline real-map compilation."""

import math
from pathlib import Path
import tempfile

import networkx as nx
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from Real_Life_Maps.build_map import (
    DenseTerrain,
    attach_node_first_visibility,
    build_travel_graph,
    dense_anchor,
    tobler_speed_mps,
)
from Real_Life_Maps.prepared_map import load_prepared_map, save_prepared_map
from simulation.agent import Agent
from simulation.domain import UNKNOWN_TYPE
from simulation.engine import sensed_nodes_truth
from simulation.real_map_benchmark import build_graphs


def _flat_terrain(size=12):
    return DenseTerrain(
        elevation_m=np.zeros((size, size), dtype=np.float64),
        transform=from_origin(0.0, float(size * 10), 10.0, 10.0),
        crs=CRS.from_epsg(26917),
        source_bounds=(0.0, 0.0, float(size * 10), float(size * 10)),
    )


def _prepared_graph():
    terrain = _flat_terrain()
    graph = build_travel_graph(
        terrain, coarse_size=3,
        road_nodes={(0, 0), (1, 0)},
        road_edges={((0, 0), (1, 0))},
    )
    visibility = np.eye(len(graph), dtype=bool)
    visibility[0, 1] = True
    attach_node_first_visibility(graph, visibility)
    return graph


def test_tobler_surface_factors_and_directed_grade():
    flat = tobler_speed_mps(0.0)
    assert math.isclose(flat * 3.6, 6.0 * math.exp(-0.175))
    assert math.isclose(tobler_speed_mps(0.0, 0.6), flat * 0.6)
    assert tobler_speed_mps(-0.05) > flat
    assert tobler_speed_mps(0.2) < tobler_speed_mps(-0.2)


def test_dense_anchor_preserves_clockwise_coarse_orientation():
    terrain = _flat_terrain()
    assert dense_anchor((0, 0), 3, terrain) == (9, 1)
    assert dense_anchor((2, 0), 3, terrain) == (9, 9)
    assert dense_anchor((0, 2), 3, terrain) == (1, 1)


def test_explicit_road_edge_uses_one_and_rough_edge_uses_point_six():
    graph = _prepared_graph()
    road = graph.edges[(0, 0), (1, 0)]
    reverse = graph.edges[(1, 0), (0, 0)]
    assert road["is_road"]
    assert road["surface_factor"] == 1.0
    assert not reverse["is_road"]
    assert reverse["surface_factor"] == 0.6
    assert math.isclose(reverse["distance"], road["distance"] / 0.6)


def test_visible_edges_are_induced_after_visible_nodes():
    graph = _prepared_graph()
    nodes = list(graph)
    observer = nodes[0]
    visible_nodes = set(graph.nodes[observer]["visible_nodes"])
    assert visible_nodes == {nodes[0], nodes[1]}
    assert set(graph.nodes[observer]["visible_edges"]) == {
        edge for edge in graph.edges
        if edge[0] in visible_nodes and edge[1] in visible_nodes
    }


def test_simulator_prefers_explicit_node_visibility():
    graph = _prepared_graph()
    observer = list(graph)[0]
    graph.nodes[observer]["visible_edges"] = ()
    visible = sensed_nodes_truth(graph, Agent(observer, capabilities={0}))
    assert visible == set(graph.nodes[observer]["visible_nodes"])


def test_prepared_map_round_trip_and_validation():
    graph = _prepared_graph()
    metadata = {"coarse_size": 3, "distance_units": "seconds"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "map.pkl.gz"
        save_prepared_map(graph, metadata, path)
        loaded, loaded_metadata = load_prepared_map(path)
    assert nx.utils.graphs_equal(graph, loaded)
    assert loaded_metadata == metadata


def test_real_map_benchmark_consumes_prepared_artifact():
    graph = _prepared_graph()
    metadata = {"coarse_size": 3, "distance_units": "seconds"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "map.pkl.gz"
        save_prepared_map(graph, metadata, path)
        env, truth, target_types = build_graphs(
            map_path=path,
            source=(0, 0),
            targets=[(2, 2)],
            obstacle_specs=[],
            target_num_neighbors=1,
            target_recursion=1,
            target_num_obstacles=0,
            num_target_types=1,
        )
    assert env.nodes[(0, 0)]["type"] == "source"
    assert env.nodes[(2, 2)]["rps_type"] == UNKNOWN_TYPE
    assert truth.nodes[(2, 2)]["rps_type"] == 1
    assert target_types == {(2, 2): 1}
    assert env.edges[(0, 0), (1, 0)]["distance"] > 0.0


def _main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(_main())
