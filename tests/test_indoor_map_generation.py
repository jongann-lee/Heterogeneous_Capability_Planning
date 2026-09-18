"""Synthetic regressions for component-aware indoor map compilation."""

import json
from pathlib import Path
import tempfile

import numpy as np
import networkx as nx

from learning.gpu_sim.instances import inspect_prepared_map, make_prepared_map_instance
from learning.policy.candidates import CandidateTerrainCache
from learning.test import evaluate
from planning.policies.scout_wrp import _compile
from Real_Life_Maps.indoor_map_generation import (
    _python_los,
    build_graph,
    compute_visibility,
    prune_split_patches,
    visible_node_indices,
    write_prepared_map,
)
from simulation.engine import sensed_edges_truth, sensed_nodes_truth
from simulation.agent import Agent
from simulation.indoor_rendering import IndoorRenderer, interpolate_edge
from simulation.visibility import visible_targets


def test_wall_split_patch_has_no_teleport_edge():
    walkable = np.ones((5, 10), dtype=np.uint8)
    walkable[:, 2] = 0
    graph, owners, nodes, anchors, components, info = build_graph(walkable, 5)
    assert info["split_patches"] == 1
    assert len(nodes) == 3
    left = nodes[int(owners[2, 1])]
    right = nodes[int(owners[2, 3])]
    neighbor = nodes[int(owners[2, 5])]
    assert left != right
    assert not graph.has_edge(left, right)
    assert not graph.has_edge(left, neighbor)
    assert graph.has_edge(right, neighbor)
    path = graph.edges[right, neighbor]["fine_path"]
    assert path[0] == graph.nodes[right]["dense_anchor"]
    assert path[-1] == graph.nodes[neighbor]["dense_anchor"]
    assert graph.edges[right, neighbor]["distance"] == len(path) - 1
    assert all(walkable[pixel] for pixel in path)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               for a, b in zip(path, path[1:]))


def test_supercover_blocks_wall_and_corner_peeking():
    mask = np.ones((5, 5), dtype=np.uint8)
    mask[:, 2] = 0
    assert not _python_los(mask, (2, 1), (2, 3))
    mask[2, 2] = 1
    assert _python_los(mask, (2, 1), (2, 3))
    corner = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert not _python_los(corner, (0, 0), (1, 1))


def test_compiled_visibility_is_symmetric_and_wall_occluded():
    walkable = np.ones((5, 10), dtype=np.uint8)
    walkable[:, 2] = 0
    _, owners, nodes, anchors, components, _ = build_graph(walkable, 5)
    packed, _ = compute_visibility(walkable, anchors, components, 16)
    left = int(owners[2, 1])
    right = int(owners[2, 3])
    neighbor = int(owners[2, 5])
    assert set(visible_node_indices(packed, left, len(nodes))) == {left}
    assert neighbor in visible_node_indices(packed, right, len(nodes))
    assert right in visible_node_indices(packed, neighbor, len(nodes))


def test_sparse_rectangular_prepared_map_preserves_visibility():
    walkable = np.ones((5, 10), dtype=np.uint8)
    walkable[:, 2] = 0
    graph, owners, nodes, anchors, components, _ = build_graph(walkable, 5)
    packed, _ = compute_visibility(walkable, anchors, components, 16)
    source = nodes[int(owners[2, 3])]
    target = nodes[int(owners[2, 5])]
    hidden = nodes[int(owners[2, 1])]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "indoor.pkl.gz"
        write_prepared_map(graph, nodes, packed, {"coarse_shape": [1, 2]}, path)
        _, loaded, _, identity = inspect_prepared_map(path)
        assert identity["dimensions"] == [1, 2]
        assert set(loaded) == set(nodes)
        assert visible_targets(loaded, source, {target, hidden}) == {target}
        assert target in CandidateTerrainCache(loaded).visible[source]
        env, truth, agents = make_prepared_map_instance(
            map_path=path, source_position=source, target_positions=[target],
            target_types=[1], agent_capabilities=[{0, 1}],
            num_target_types=1, num_agents=1)
        assert target in sensed_nodes_truth(truth, agents[0])
        assert hidden not in sensed_nodes_truth(truth, agents[0])
        assert (source, target) in sensed_edges_truth(truth, agents[0])
        safe_nodes, safe_index, _, _, masks = _compile(
            env, {target}, [target])
        assert source in safe_nodes and masks[safe_index[source]] == 1
        assert env.nodes[target]["rps_type"] != truth.nodes[target]["rps_type"]


def test_pruning_keeps_real_corridor_and_runs_test_evaluator():
    walkable = np.zeros((5, 15), dtype=np.uint8)
    walkable[0, :] = 1
    walkable[2:, 5:10] = 1
    graph, _, nodes, anchors, components, _ = build_graph(walkable, 5)
    packed, _ = compute_visibility(walkable, anchors, components, 16)
    pruned, two_part_nodes, pruned_visibility, report = prune_split_patches(
        graph, nodes, packed)
    source, target, isolated = (0, 0), (0, 2), (0, 1)
    assert set(pruned) == {source, target, isolated}
    assert report["removed_split_nodes"] == 1
    assert report["shortcuts_added"] == 2
    assert pruned.has_edge(source, target)
    assert not pruned.has_edge(source, isolated)
    assert not pruned.has_edge(isolated, target)
    route = pruned.edges[source, target]["fine_path"]
    assert pruned.edges[source, target]["distance"] == len(route) - 1
    assert all(walkable[pixel] for pixel in route)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               for a, b in zip(route, route[1:]))

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        path = directory / "pruned.pkl.gz"
        write_prepared_map(
            pruned, two_part_nodes, pruned_visibility,
            {"coarse_shape": [1, 3], "fine_shape": [5, 15]}, path)
        suite = directory / "suite.json"
        suite.write_text(json.dumps({
            "schema_version": 1,
            "suite_id": "indoor_shortcut",
            "terrain_id": "indoor_synthetic",
            "num_target_types": 1,
            "source_position": list(source),
            "agent_configurations": [{
                "id": "hybrid", "agent_count": 1, "profile": "hybrid",
                "capabilities": [[0, 1]],
            }],
            "target_configurations": [{
                "id": "single", "target_count": 1, "profile": "single",
                "targets": [{"position": list(target), "type": 1}],
            }],
        }))
        records, *_ = evaluate(
            suite=suite, map_path=path, policy="scout-then-execute",
            device="cpu", render=True, render_dir=directory / "frames",
            output_mp4=directory / "result.mp4", render_dt=2.0)
        assert list((directory / "frames").glob("frame_*.png"))
    assert len(records) == 1
    assert records[0]["completed"]
    assert records[0]["deaths"] == 0


def test_indoor_rendering_follows_fine_route_around_wall():
    graph = nx.DiGraph()
    first, second = "start", "goal"
    graph.add_node(first, pos=(1.0, 1.0), dense_anchor=(1, 1), type="source")
    graph.add_node(second, pos=(5.0, 1.0), dense_anchor=(1, 5),
                   type="target_unreached", rps_type=1)
    route = ((1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5))
    graph.add_edge(first, second, fine_path=route, distance=6.0)
    walkable = np.zeros((4, 7), dtype=np.uint8)
    for row, col in route:
        walkable[row, col] = 1
    midpoint = interpolate_edge(graph, first, second, 0.5)
    assert midpoint == (3.0, 2.0)
    assert walkable[int(midpoint[1]), int(midpoint[0])]
    assert not walkable[1, 3]  # Straight anchor interpolation crosses the wall.

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raster = directory / "walkable.npy"
        np.save(raster, walkable)
        renderer = IndoorRenderer(
            graph, metadata={"fine_shape": [4, 7]}, raster_path=raster)
        assert renderer.interpolate_edge(first, second, 0.5) == midpoint
        agent = Agent(first, capabilities={1})
        agent.planned_path = [first, second]
        frame = directory / "frame_0000.png"
        renderer(graph, graph, [agent], 0, frame, agent_xy=[midpoint])
        assert frame.is_file() and frame.stat().st_size > 0


def _main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _main()
