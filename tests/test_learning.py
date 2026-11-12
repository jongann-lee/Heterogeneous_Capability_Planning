"""Fast checks for the centralized learned planner."""

import copy
from collections import Counter, OrderedDict
from dataclasses import asdict, replace
from datetime import datetime
from itertools import combinations, permutations, product
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch
import yaml

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Real_Life_Maps.prepared_map import save_prepared_map
from learning.analyze import analyze_evaluation, format_analysis
from learning.evaluation_suite import (
    AgentConfiguration,
    EvaluationCase,
    TargetConfiguration,
    TargetDefinition,
    load_evaluation_suite,
    select_evaluation_cases,
    validate_evaluation_suite_locations,
)
from learning.policy.candidates import (
    Candidate,
    CandidateScenarioCache,
    CandidateTerrainCache,
    generate_candidates,
    physical_group_metadata,
)
import learning.policy.candidates as candidate_module
from learning.policy.configuration import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    InstanceConfig,
    LearningConfig,
    MAX_EDGE_NORMALIZATION,
    PER_DECISION_EDGE_NORMALIZATION,
    RAW_DISTANCE_FEATURE_SCHEMA_VERSION,
    RAW_EDGE_NORMALIZATION,
    load_config,
)
from learning.gpu_sim.routing import GridRouter
from learning.gpu_sim.instances import (
    _prepared_terrain_template_cached,
    inspect_prepared_map,
    make_prepared_map_instance,
)
from learning.gpu_sim.cugraph_router import CuGraphRouter
from learning.gpu_sim.state import TensorEpisodeState
from learning.gpu_sim.world import TensorWorld
from learning.policy.model import (
    HeterogeneousGraphPolicy,
    VanillaTransformerPolicy,
    build_policy,
)
from learning.modules import (
    AssignmentDecoder,
    BeliefWeightedCapabilityRelations,
    DecoderOutput,
)
from learning.gpu_sim.observation_cpu import (
    batch_observations,
    build_observation,
    normalize_task_graph_travel_times,
)
from learning.gpu_sim.observation_gpu import TensorObservationBuilder
from learning.policy.oracle import parallel_tsp
from planning.full_information import (
    FullInformationInfeasibleError,
    full_information_makespan,
    solve_full_information,
)
from planning.policies.scout_then_execute import (
    ScoutThenExecutePolicy,
    solve_cooperative_scouting,
)
from planning.policies.scout_wrp import solve_cover_walk
from learning.policy.adapter import LearnedPolicyAdapter
from learning.gpu_sim.rollout_cpu import calculate_episode_return, collect_episode
from learning.gpu_sim.rollout_gpu import (
    DecisionTrace,
    collect_tensor_episodes,
    replay_tensor_gradients,
)
from learning.train import _episode_agent_count, _serialized_run_config, train
from learning.test import (
    _cuda_batch_plan,
    _evaluation_output_path,
    _planning_metrics,
    _select_prepared_map_path,
    evaluate as evaluate_suite,
    summarize as summarize_evaluation,
)
import learning.test as evaluation_module
from learning.task_progress import TaskProgressRecorder
from learning.test_single_detailed import evaluate as evaluate_single_detailed
from simulation.agent import Agent
from simulation.domain import UNKNOWN_TYPE, init_target_types
from simulation.engine import run_simulation
from simulation.rendering import _agent_color, _stacked_agent_label_offsets


def _line(length=5):
    graph = nx.DiGraph()
    for node in range(length):
        graph.add_node(node, pos=(node, 0), type="intermediate",
                       height=float(node), visible_edges=[])
    for node in range(length - 1):
        for u, v in ((node, node + 1), (node + 1, node)):
            graph.add_edge(u, v, distance=1.0, observed_edge=False)
    graph.nodes[0]["type"] = "source"
    return graph


class _ReferenceRouter:
    """CPU exact routes for exercising tensor simulation without CUDA."""

    def __init__(self, num_nodes, sources, destinations, costs):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(num_nodes))
        self.graph.add_weighted_edges_from(zip(sources, destinations, costs))

    def sssp(self, sources, blocked_nodes=()):
        sources = torch.as_tensor(sources).flatten()
        blocked = torch.zeros((len(sources), len(self.graph)), dtype=torch.bool)
        blocked[:, list(blocked_nodes)] = True
        return self.sssp_batch(sources, blocked)

    def sssp_batch(self, sources, blocked):
        distances = torch.full((len(sources), len(self.graph)), torch.inf)
        predecessors = torch.full(distances.shape, -1, dtype=torch.long)
        for row, source in enumerate(sources.tolist()):
            excluded = set(torch.where(blocked[row])[0].tolist()) - {source}
            graph = nx.subgraph_view(
                self.graph, filter_node=lambda node: node not in excluded)
            lengths, paths = nx.single_source_dijkstra(graph, source)
            for node, length in lengths.items():
                distances[row, node] = length
                if len(paths[node]) > 1:
                    predecessors[row, node] = paths[node][-2]
        return SimpleNamespace(distances=distances, predecessors=predecessors)


def _prepared_grid(size, height_offset=0.0):
    graph = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
    for node in graph:
        height = float(height_offset + node[0] + node[1])
        graph.nodes[node].update(
            pos=node, height=height, elevation_m=height,
            type="intermediate", visible_nodes=(node,), visible_edges=(),
            diagnostic_label=f"node-{node[0]}-{node[1]}")
    for u, v in graph.edges:
        graph.edges[u, v].update(
            distance=1.0, is_road=False, observed_edge=False,
            num_used=1.0, diagnostic_label=f"edge-{u}-{v}")
    return graph


def _save_prepared_grid(directory, name, size, height_offset=0.0):
    path = Path(directory) / name
    graph = _prepared_grid(size, height_offset=height_offset)
    save_prepared_map(
        graph,
        {"coarse_size": size, "distance_units": "seconds",
         "terrain_label": name},
        path)
    return path


def _write_single_case_suite(directory, source=(0, 0), target=(1, 1),
                             terrain_id="temporary_terrain"):
    path = Path(directory) / "suite.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "suite_id": "temporary_suite",
        "terrain_id": terrain_id,
        "num_target_types": 1,
        "source_position": list(source),
        "agent_configurations": [{
            "id": "agents", "agent_count": 1, "profile": "hybrid",
            "capabilities": [[0, 1]],
        }],
        "target_configurations": [{
            "id": "targets", "target_count": 1, "profile": "single",
            "targets": [{"position": list(target), "type": 1}],
        }],
    }))
    return path


def _instance(two_agents=True):
    graph = _line()
    graph.nodes[4].update(type="target_unreached", rps_type=1)
    agents = [Agent(0, capabilities={0, 1})]
    if two_agents:
        agents.append(Agent(0, capabilities={2}))
    return graph, agents


def _pair_graph(targets=("a", "b"), pair_weights=None):
    """Directed target graph with one safe staging hub for every pair."""
    graph = nx.DiGraph()
    targets = tuple(targets)
    for index, target in enumerate(targets):
        graph.add_node(
            target, pos=(index, 1), height=0.0,
            type="target_unreached", rps_type=UNKNOWN_TYPE,
            visible_edges=[])
    pair_weights = pair_weights or {}
    for pair_index, (first, second) in enumerate(combinations(targets, 2)):
        default = float(pair_index + 1)
        forward, reverse = pair_weights.get(
            (first, second), (default, default))
        if forward is not None:
            graph.add_edge(first, second, distance=float(forward))
        if reverse is not None:
            graph.add_edge(second, first, distance=float(reverse))
        hub = f"hub_{first}_{second}"
        graph.add_node(
            hub, pos=(pair_index, 0), height=0.0,
            type="intermediate", visible_edges=[])
        graph.add_edge(hub, first, distance=1.0)
        graph.add_edge(hub, second, distance=1.0)
    return graph


def _candidate_config(staging_per_target=0):
    return replace(
        load_config().candidates,
        staging_per_target=staging_per_target,
        include_pair_staging=True,
        include_wait=False,
    )


def _model():
    torch.manual_seed(7)
    config = replace(
        load_config().model,
        num_target_types=2,
        model_dim=32,
        num_heads=4,
        num_world_blocks=1,
    )
    model = VanillaTransformerPolicy(config)
    model.eval()
    return model


def test_real_terrain_visibility_cache_is_persistent_and_dem_keyed():
    first_heights = np.zeros((3, 3), dtype=np.float32)
    changed_heights = first_heights.copy()
    changed_heights[1, 1] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        first = RealTerrainGrid(first_heights, source=(0, 0), targets=[])
        assert not first.compute_all_visibilities(
            max_radius=2, angular_res=8, cache_dir=directory)
        expected = {
            node: first.G.nodes[node]["visible_edges"]
            for node in first.G
        }

        second = RealTerrainGrid(first_heights, source=(0, 0), targets=[])
        second._get_polytope_visibility = lambda *_args: (_ for _ in ()).throw(
            AssertionError("visibility was recalculated instead of loaded"))
        assert second.compute_all_visibilities(
            max_radius=2, angular_res=8, cache_dir=directory)
        assert {
            node: second.G.nodes[node]["visible_edges"]
            for node in second.G
        } == expected

        changed = RealTerrainGrid(changed_heights, source=(0, 0), targets=[])
        assert not changed.compute_all_visibilities(
            max_radius=2, angular_res=8, cache_dir=directory)
        assert len(list(Path(directory).glob("*.pkl"))) == 2


def test_development_evaluation_suite_preserves_original_rps_case():
    suite, path = load_evaluation_suite("development")
    assert path.name == "wv_rps_fixed_v1.json"
    assert suite.suite_id == "wv_rps_fixed_v1"
    assert suite.source_position == (0, 0)
    assert len(suite.agent_configurations) == 1
    assert len(suite.target_configurations) == 1
    case = select_evaluation_cases(suite)[0]
    assert case.scenario_id == "rps_agents__rps_targets"
    assert case.agent.capabilities == tuple(map(frozenset, (
        {0}, {1}, {2}, {3})))
    assert [target.position for target in case.target.targets] == [
        (14, 54), (1, 29), (33, 17), (34, 35),
        (63, 37), (37, 5), (49, 58),
    ]
    assert [target.target_type for target in case.target.targets] == [
        1, 2, 2, 1, 2, 3, 3]


def test_evaluation_output_defaults_to_timestamp_and_honors_override():
    timestamp = datetime(2026, 9, 10, 16, 42, 3, 123456)
    default = _evaluation_output_path(now=timestamp)
    assert default.parent.name == "evaluation"
    assert default.name == "2026-09-10_16-42-03_123456.json"
    explicit = _evaluation_output_path("named-result.json")
    assert explicit == Path("named-result.json").resolve()


def test_prepared_map_selection_factory_and_cache_are_map_aware():
    with tempfile.TemporaryDirectory() as directory:
        first_path = _save_prepared_grid(directory, "first.pkl.gz", 2)
        second_path = _save_prepared_grid(directory, "second.pkl.gz", 3, 10.0)
        _prepared_terrain_template_cached.cache_clear()
        first_resolved, first_graph, _metadata, first_info = (
            inspect_prepared_map(first_path))
        second_resolved, second_graph, _metadata, second_info = (
            inspect_prepared_map(second_path))
        assert len(first_graph) == 4
        assert len(second_graph) == 9
        assert first_info["dimensions"] == [2, 2]
        assert second_info["dimensions"] == [3, 3]
        assert first_info["sha256"] != second_info["sha256"]
        assert second_graph.nodes[(2, 2)]["diagnostic_label"] == "node-2-2"
        assert second_graph.edges[(2, 2), (2, 1)]["diagnostic_label"].startswith(
            "edge-")

        base = load_config()
        config_payload = asdict(base)
        config_payload["instances"]["map_path"] = str(second_path)
        config_path = Path(directory) / "learning.yaml"
        config_path.write_text(yaml.safe_dump(config_payload))
        selected_config = load_config(config_path)
        assert selected_config.instances.map_path == str(second_path)
        checkpoint_config = replace(
            base, instances=replace(base.instances, map_path=str(first_path)))
        explicit_config = replace(
            base, instances=replace(base.instances, map_path=str(second_path)))
        assert _select_prepared_map_path(
            None, explicit_config, checkpoint_config) == str(second_path)
        assert _select_prepared_map_path(
            str(first_path), explicit_config, checkpoint_config) == str(first_path)

        env, truth, agents = make_prepared_map_instance(
            seed=0, num_target_types=1, num_agents=1,
            source_position=(2, 2), target_positions=[(2, 1)],
            target_types=[1], agent_capabilities=[{0, 1}],
            map_path=selected_config.instances.map_path)
        assert len(env) == len(truth) == 9
        assert agents[0].position == (2, 2)
        assert truth.nodes[(2, 1)]["rps_type"] == 1
        assert str(first_resolved) != str(second_resolved)

        try:
            make_prepared_map_instance(
                seed=0, num_target_types=1, num_agents=1,
                source_position=(2, 2), target_positions=[(3, 0)],
                target_types=[1], agent_capabilities=[{0, 1}],
                map_path=second_path)
        except ValueError as error:
            assert "not present in the prepared map" in str(error)
        else:
            raise AssertionError("an absent prepared-map target was accepted")


def test_suite_coordinates_use_selected_graph_membership_not_wv_bounds():
    with tempfile.TemporaryDirectory() as directory:
        suite_path = _write_single_case_suite(
            directory, source=(64, 64), target=(64, 63),
            terrain_id="not_wv_and_not_whitelisted")
        suite, _path = load_evaluation_suite(suite_path)
        assert suite.terrain_id == "not_wv_and_not_whitelisted"
        graph = nx.DiGraph()
        graph.add_nodes_from(((64, 64), (64, 63)))
        validate_evaluation_suite_locations(suite, graph)
        graph.remove_node((64, 63))
        try:
            validate_evaluation_suite_locations(suite, graph)
        except ValueError as error:
            assert "not present in the selected prepared map" in str(error)
        else:
            raise AssertionError("a suite target absent from the graph was accepted")


def test_factorial_evaluation_suite_has_180_filterable_cases():
    suite, path = load_evaluation_suite("test")
    explicit, explicit_path = load_evaluation_suite(path)
    assert explicit == suite
    assert explicit_path == path
    assert path.name == "wv_factorial_test_v1.json"
    assert len(suite.agent_configurations) == 12
    assert len(suite.target_configurations) == 15
    assert len(select_evaluation_cases(suite)) == 180
    for count in range(3, 7):
        assert sum(item.agent_count == count
                   for item in suite.agent_configurations) == 3
    for count in range(5, 10):
        assert sum(item.target_count == count
                   for item in suite.target_configurations) == 3
    filtered = select_evaluation_cases(
        suite, agent_count=4, target_count=8)
    assert len(filtered) == 9
    assert len(select_evaluation_cases(
        suite, agent_config="agents_04_b",
        target_config="targets_08_c")) == 1
    assert select_evaluation_cases(suite, limit=1)[0].scenario_id == (
        "agents_03_a__targets_05_a")

    plan = _cuda_batch_plan(select_evaluation_cases(suite))
    batches = [batch for _target_id, target_batches in plan
               for batch in target_batches]
    assert len(plan) == 15
    assert len(batches) == 60
    assert {len(batch) for batch in batches} == {3}
    for batch in batches:
        assert len({case.target.id for _index, case in batch}) == 1
        assert len({case.agent.agent_count for _index, case in batch}) == 1


def test_washington_suite_uses_the_opposite_corner_with_the_same_factorial_cases():
    suite, _path = load_evaluation_suite("learning/evaluation_suites/wa_factorial_test_v1.json")
    original, _original_path = load_evaluation_suite("test")
    assert suite.suite_id == "wa_factorial_test_v1"
    assert suite.source_position == (63, 63)
    assert suite.agent_configurations == original.agent_configurations
    assert suite.target_configurations == original.target_configurations
    cases = select_evaluation_cases(suite)
    assert len(cases) == 180
    assert all(case.source_position == (63, 63) for case in cases)
    _path, graph, _metadata, _identity = inspect_prepared_map(
        "Real_Life_Maps/WA_tobler_viewshed_64.pkl.gz")
    validate_evaluation_suite_locations(suite, graph)


def test_kth_big_suite_crosses_three_eight_agent_profiles_with_three_fourteen_target_layouts():
    suite, _path = load_evaluation_suite("learning/evaluation_suites/KTH_indoor_big.json")
    assert suite.suite_id == "kth_indoor_big_factorial_v2"
    assert suite.source_position == (43, 77)
    assert suite.num_target_types == 5
    assert len(suite.agent_configurations) == len(suite.target_configurations) == 3
    assert {agent.agent_count for agent in suite.agent_configurations} == {8}
    assert {agent.id for agent in suite.agent_configurations} == {
        "agents_08_a", "agents_08_b", "agents_08_c"}
    agents = {agent.profile: agent for agent in suite.agent_configurations}
    assert set(agents) == {"specialists", "balanced_hybrid", "scout_redundant"}
    for profile, scout_count in (("specialists", 2), ("balanced_hybrid", 3),
                                 ("scout_redundant", 4)):
        capabilities = agents[profile].capabilities
        assert len(capabilities) == 8
        assert sum(0 in values for values in capabilities) == scout_count
        assert sum(map(len, capabilities)) == 16  # Mean 2, including scouting.
        assert all(1 <= len(values) <= 3 and values - {0} for values in capabilities)
        assert set.union(*(set(values) for values in capabilities)) == set(range(6))
        coverage = Counter(value for values in capabilities for value in values if value > 0)
        assert min(coverage.values()) >= 2
        assert max(coverage.values()) - min(coverage.values()) <= 1
    layouts = {layout.profile: layout for layout in suite.target_configurations}
    assert set(layouts) == {"balanced", "types_1_2_heavy", "multi_cluster_balanced"}
    for layout in layouts.values():
        assert layout.target_count == 14
        assert len({target.position for target in layout.targets}) == 14
    for profile in ("balanced", "multi_cluster_balanced"):
        counts = Counter(t.target_type for t in layouts[profile].targets)
        assert set(counts) == set(range(1, 6))
        assert max(counts.values()) - min(counts.values()) <= 1
    heavy = Counter(t.target_type for t in layouts["types_1_2_heavy"].targets)
    assert heavy == {1: 5, 2: 5, 3: 2, 4: 1, 5: 1}
    cases = select_evaluation_cases(suite)
    assert len(cases) == 9
    assert len({case.scenario_id for case in cases}) == 9
    batches = _cuda_batch_plan(cases)
    assert len(batches) == 3
    assert all(len(agent_batches) == 1 and len(agent_batches[0]) == 3
               for _target_id, agent_batches in batches)


def test_kth_big_targets_respect_room_walls_visibility_and_safe_reachability():
    from scipy import ndimage
    from Real_Life_Maps.indoor_map_generation import load_walkable
    from simulation.visibility import visible_nodes, visible_targets

    suite, suite_path = load_evaluation_suite("learning/evaluation_suites/KTH_indoor_big.json")
    payload = json.loads(suite_path.read_text())
    _path, graph, _metadata, identity = inspect_prepared_map("Real_Life_Maps/KTH_Indoor_map.pkl.gz")
    validate_evaluation_suite_locations(suite, graph)
    assert payload["placement_design"]["prepared_map_sha256"] == identity["sha256"]
    rasters = Path(__file__).parents[1] / "Real_Life_Maps" / "50052749"
    walkable, _raster_metadata = load_walkable(rasters)
    zones = np.load(rasters / "valid_space.npy", allow_pickle=False)
    wall_clearance = ndimage.distance_transform_edt(walkable)
    initially_visible = visible_nodes(graph, suite.source_position)
    custom, _custom_path = load_evaluation_suite(suite_path.with_name("KTH_indoor_big_single.json"))
    custom_positions = {target.position for layout in custom.target_configurations
                        for target in layout.targets}
    all_positions = set()
    for layout, definition in zip(suite.target_configurations, payload["target_configurations"]):
        positions = {target.position for target in layout.targets}
        assert not positions & all_positions
        assert not positions & initially_visible
        assert not positions & custom_positions
        all_positions.update(positions)
        safe_graph = nx.subgraph_view(graph, filter_node=lambda n: n not in positions)
        reachable = nx.descendants(safe_graph, suite.source_position) | {suite.source_position}
        room_counts = Counter()
        for target, record in zip(layout.targets, definition["targets"]):
            anchor = graph.nodes[target.position]["dense_anchor"]
            assert walkable[anchor]
            assert 2 <= wall_clearance[anchor] <= 10
            room = int(zones[anchor])
            assert room > 0 and record["room_zone_id"] == room
            room_counts[room] += 1
            assert all(max(abs(target.position[i] - old[i]) for i in (0, 1)) >= 3
                       for old in custom_positions)
            assert any(node in reachable for node in graph.predecessors(target.position))
            assert any(node in reachable for node in graph.successors(target.position))
            witnesses = visible_nodes(graph, target.position) & reachable
            assert witnesses
            assert target.position in visible_targets(graph, min(witnesses), {target.position})
        if layout.profile == "multi_cluster_balanced":
            assert set(room_counts) == {7, 16, 29, 34}
            assert sorted(room_counts.values()) == [3, 3, 4, 4]
            assert len(definition["clusters"]) == 4
            cluster_positions = set()
            for cluster in definition["clusters"]:
                nodes = [tuple(position) for position in cluster["positions"]]
                assert len(nodes) in (3, 4)
                assert all(int(zones[graph.nodes[node]["dense_anchor"]]) == cluster["room_zone_id"]
                           for node in nodes)
                first = np.array(graph.nodes[nodes[0]]["dense_anchor"])
                assert all(np.linalg.norm(np.array(graph.nodes[node]["dense_anchor"]) - first) <= 25
                           for node in nodes)
                cluster_positions.update(nodes)
            assert cluster_positions == positions
        else:
            assert len(room_counts) == 14
    assert len(all_positions) == 42


def test_factorial_targets_have_one_alternating_heavy_and_multi_clusters():
    suite, _path = load_evaluation_suite("test")
    expected_heavy_type = {5: 1, 6: 2, 7: 1, 8: 2, 9: 1}
    expected_cluster_sizes = {
        5: [2, 3],
        6: [3, 3],
        7: [2, 2, 3],
        8: [2, 3, 3],
        9: [3, 3, 3],
    }
    for target_count, heavy_type in expected_heavy_type.items():
        configurations = [
            item for item in suite.target_configurations
            if item.target_count == target_count
        ]
        assert {item.profile for item in configurations} == {
            "balanced", f"type_{heavy_type}_heavy",
            "multi_cluster_balanced"}
        heavy = next(item for item in configurations
                     if item.profile.endswith("_heavy"))
        type_counts = {
            target_type: sum(target.target_type == target_type
                             for target in heavy.targets)
            for target_type in range(1, 4)
        }
        assert type_counts[heavy_type] > max(
            count for target_type, count in type_counts.items()
            if target_type != heavy_type)
        clustered = next(
            item for item in configurations
            if item.profile == "multi_cluster_balanced")
        positions = [target.position for target in clustered.targets]
        unseen = set(range(len(positions)))
        components = []
        while unseen:
            component = {unseen.pop()}
            frontier = list(component)
            while frontier:
                current = frontier.pop()
                row, column = positions[current]
                neighbors = {
                    index for index in unseen
                    if max(abs(positions[index][0] - row),
                           abs(positions[index][1] - column)) <= 4
                }
                unseen -= neighbors
                component |= neighbors
                frontier.extend(neighbors)
            components.append(component)
        assert sorted(map(len, components)) == expected_cluster_sizes[
            target_count]
        cluster_type_counts = [
            sum(target.target_type == target_type
                for target in clustered.targets)
            for target_type in range(1, 4)
        ]
        assert max(cluster_type_counts) - min(cluster_type_counts) <= 1


def test_cuda_evaluation_executes_compatible_cases_in_one_batch():
    if not torch.cuda.is_available():
        return
    env = nx.DiGraph()
    for node, position, node_type in (
        (0, (0, 0), "source"),
        (1, (1, 0), "intermediate"),
        (2, (2, 1), "target_unreached"),
        (3, (2, -1), "target_unreached"),
    ):
        env.add_node(
            node, pos=position, height=0.0, type=node_type,
            visible_edges=[])
    for source, target in (
        (0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1),
    ):
        env.add_edge(
            source, target, distance=1.0, observed_edge=False)
    # Both target types are known at time zero. This regression tests CUDA
    # suite batching, not exploration by a randomly initialized policy.
    env.nodes[0]["visible_edges"] = [(1, 2), (1, 3)]
    truth = env.copy()
    init_target_types(env, truth, {2: 1, 3: 2})

    target_configuration = TargetConfiguration(
        "targets", 2, "synthetic",
        (TargetDefinition((2, 1), 1), TargetDefinition((2, -1), 2)))
    agent_configurations = (
        AgentConfiguration(
            "agents_a", 2, "synthetic",
            (frozenset({0, 1}), frozenset({2}))),
        AgentConfiguration(
            "agents_b", 2, "synthetic",
            (frozenset({0, 2}), frozenset({1}))),
    )
    cases = [
        EvaluationCase(
            "synthetic", f"{agents.id}__targets", 0, 2,
            agents, target_configuration)
        for agents in agent_configurations
    ]
    config = load_config()
    config = replace(config, model=replace(
        config.model, num_target_types=2, model_dim=32, num_heads=4,
        message_passing_blocks=1, distance_embedding_dim=8,
        critic_hidden_dim=16))
    class TargetsFirst(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decoder = AssignmentDecoder()

        def decode(self, observation, training=False):
            target_actions = observation.serves_mask.any(dim=2)
            logits = observation.task_agent_features.new_full(
                observation.feasible_action_mask.shape, -20.0
            )
            logits = torch.where(
                target_actions[:, None], logits.new_tensor(20.0), logits
            ) + self.anchor * 0.0
            return self.decoder(
                logits,
                observation.feasible_action_mask,
                observation.action_capacities,
                training=training,
                candidate_physical_group=observation.candidate_physical_group,
                physical_group_capacity=observation.physical_group_capacity,
                physical_group_representative=(
                    observation.physical_group_representative
                ),
                physical_group_mask=observation.physical_group_mask,
            )

    model = TargetsFirst().to("cuda").eval()
    with patch.object(
            evaluation_module, "_case_factory",
            return_value=(env, truth, [])):
        records = evaluation_module._gpu_suite_records(
            model, config, cases, "cuda")
    assert len(records) == 2
    assert {record["evaluation_batch_id"] for record in records} == {0}
    assert {record["evaluation_batch_size"] for record in records} == {2}


def test_detailed_evaluation_requires_one_scenario_and_one_episode():
    missing = Path("/definitely/missing/checkpoint.pt")
    for options in ({}, {"limit": 1, "episodes": 2}):
        try:
            evaluate_single_detailed(missing, suite="test", **options)
        except ValueError as error:
            assert "exactly one scenario and --episodes 1" in str(error)
        else:
            raise AssertionError("multiple detailed episodes were accepted")
    try:
        evaluate_single_detailed(missing, suite="test", limit=1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a single detailed case did not reach checkpoint loading")


def test_detailed_kth_defaults_allow_explicit_overrides():
    import learning.test_single_detailed as detailed

    suite, path = load_evaluation_suite(detailed.DEFAULT_SUITE)
    assert path.name == "KTH_indoor_big.json"
    cases = select_evaluation_cases(
        suite, agent_config=detailed.DEFAULT_AGENT_CONFIG,
        target_config=detailed.DEFAULT_TARGET_CONFIG)
    assert len(cases) == 1
    assert cases[0].agent.profile == "balanced_hybrid"
    assert cases[0].agent.agent_count == 8
    assert cases[0].target.profile == "balanced"
    assert cases[0].target.target_count == 14
    assert detailed.DEFAULT_MAP_PATH.name == "KTH_Indoor_map.pkl.gz"
    assert detailed.DEFAULT_OUTPUT_DIR.parts[-2:] == ("evaluation", "KTH-detailed")
    with patch.object(detailed, "_evaluate", return_value="recorded") as evaluate:
        assert detailed.evaluate(policy="scout-then-execute") == "recorded"
        assert evaluate.call_args.kwargs["suite"] == detailed.DEFAULT_SUITE
        assert evaluate.call_args.kwargs["map_path"] == detailed.DEFAULT_MAP_PATH
        assert evaluate.call_args.kwargs["agent_config"] == "agents_08_b"
        assert evaluate.call_args.kwargs["target_config"] == "targets_14_a"
        detailed.evaluate(agent_config="agents_08_c", target_config="targets_14_c")
        assert evaluate.call_args.kwargs["agent_config"] == "agents_08_c"
        assert evaluate.call_args.kwargs["target_config"] == "targets_14_c"
        detailed.evaluate(suite="development", map_path="other-map.pkl.gz")
        assert evaluate.call_args.kwargs["suite"] == "development"
        assert evaluate.call_args.kwargs["map_path"] == "other-map.pkl.gz"
        assert "agent_config" not in evaluate.call_args.kwargs
        assert "target_config" not in evaluate.call_args.kwargs
    for flags, agent_id, target_id in (
        ([], "agents_08_b", "targets_14_a"),
        (["--agent-config", "agents_08_c", "--target-config", "targets_14_c"],
         "agents_08_c", "targets_14_c"),
    ):
        with patch.object(evaluation_module, "evaluate", side_effect=ValueError("checked")) as run, \
                patch("sys.stderr"):
            try:
                detailed.main(["--policy", "fi-opt", *flags])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("the evaluation stub did not run")
        assert run.call_args.kwargs["agent_config"] == [agent_id]
        assert run.call_args.kwargs["target_config"] == [target_id]


def test_detailed_task_progress_matches_cpu_and_tensor_transitions():
    truth = _line(5)
    truth.nodes[0]["visible_nodes"] = [0, 2]
    truth.nodes[1]["visible_nodes"] = [1, 4]
    for node in (2, 4):
        truth.nodes[node].update(type="target_unreached", rps_type=1)
    env = truth.copy()
    for node in (2, 4):
        env.nodes[node]["rps_type"] = UNKNOWN_TYPE
    # Suite order intentionally differs from the tensor world's target order.
    targets = (4, 2)
    cpu = TaskProgressRecorder(targets, truth)
    cpu.initialize(env)
    agent = Agent(0, capabilities={0, 1})
    agent.planned_path = [0, 1, 2, 3, 4]
    result = run_simulation(
        env, truth, [agent], policy=evaluation_module._PreserveFullInformationRoutes())
    cpu.consume_cpu_events(result["events"])
    expected = cpu.finish(result["makespan"])
    assert [(e["time"], e["event"], e["target_id"]) for e in expected["events"]] == [
        (0.0, "task_discovered", 1), (1.0, "task_discovered", 0),
        (2.0, "task_completed", 1), (4.0, "task_completed", 0),
    ]
    assert expected["timeline"][0]["scouted_count"] == 1
    assert expected["timeline"][-1]["completed_count"] == 2

    config = replace(load_config().candidates, staging_per_target=0,
                     include_pair_staging=False)
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        world = TensorWorld.from_networkx(env, config, device="cpu")
    state = TensorEpisodeState.create(world, [world.node_index[0]],
                                     [[[True, True]]], [[1, 1]])
    tensor = TaskProgressRecorder(targets, truth)
    tensor.initialize(env)
    tensor.capture_tensor_state(state)
    for node in (1, 2, 3, 4):
        state.dispatch_next_hops(
            [[world.node_index[node]]], torch.tensor([[True]]))
        state.advance()
        tensor.capture_tensor_state(state)
        tensor.capture_tensor_state(state)  # Repeated callbacks must not duplicate facts.
    assert tensor.finish(float(state.clock[0])) == expected
    json.dumps(expected)  # The complete progress object is JSON serializable.


def test_detailed_progress_handles_contact_death_simultaneity_and_incompletion():
    truth = _line(3)
    for node in (1, 2):
        truth.nodes[node].update(type="target_unreached", rps_type=1)
    env = truth.copy()
    for node in (1, 2):
        env.nodes[node]["rps_type"] = UNKNOWN_TYPE
    progress = TaskProgressRecorder((1, 2), truth)
    progress.initialize(env)
    progress.consume_cpu_events([
        {"time": 1.0, "event": "agent_dies", "node": 1},
        {"time": 1.0, "event": "reveal", "node": 2},
        {"time": 1.0, "event": "agent_wins", "node": 2},
    ])
    result = progress.finish(5.0)
    assert [p["time"] for p in result["timeline"]] == [0.0, 1.0, 5.0]
    last = result["timeline"][-1]
    assert last["discovered_count"] == 2 and last["scouted_count"] == 1
    assert last["completed_count"] == 1 and last["remaining_count"] == 1
    assert result["targets"][0]["discovery_source"] == "contact"
    assert result["targets"][0]["completed_at"] is None
    assert result["targets"][1]["completed_at"] == 1.0


def test_detailed_episode_hooks_preserve_learned_cpu_tensor_and_classical_results():
    truth = _line(3)
    truth.nodes[0]["visible_nodes"] = [0, 2]
    truth.nodes[2].update(type="target_unreached", rps_type=1)
    env = truth.copy()
    env.nodes[2]["rps_type"] = UNKNOWN_TYPE
    model = _graph_model(num_target_types=1, use_critic=False)
    config = replace(
        load_config(), model=model.config,
        candidates=replace(load_config().candidates, staging_per_target=0,
                           include_pair_staging=False, include_wait=False))

    def recorder():
        return TaskProgressRecorder([2], truth)

    with torch.no_grad():
        cpu = evaluation_module._cpu_episode(
            model, config, env, truth, [Agent(0, capabilities={0, 1})],
            "cpu", task_progress=recorder())
        with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
            tensor, _terrain, _world = evaluation_module._gpu_episode(
                model, config, env, truth, [Agent(0, capabilities={0, 1})],
                "cpu", task_progress=recorder())
        classical = evaluation_module._classical_episode(
            "scout-then-execute", config, env, truth,
            [Agent(0, capabilities={0, 1})], task_progress=recorder())
    for record in (cpu, tensor, classical):
        assert record["completed"] and record["makespan"] == 2.0
        assert record["task_progress"] == cpu["task_progress"]
        assert record["task_progress"]["targets"][0]["discovered_at"] == 0.0
        assert record["task_progress"]["targets"][0]["completed_at"] == 2.0


def test_detailed_cli_exports_classical_progress_and_refuses_overwrite():
    from learning.test_single_detailed import main as detailed_main

    directory = Path(tempfile.mkdtemp(prefix="detailed-evaluation-"))
    map_path = _save_prepared_grid(directory, "terrain.pkl.gz", 2)
    suite_path = _write_single_case_suite(directory)
    output = directory / "detailed.json"
    detailed_main([
        "--policy", "fi-opt", "--suite", str(suite_path),
        "--map-path", str(map_path), "--output", str(output), "--no-progress",
    ])
    original = output.read_bytes()
    payload = json.loads(original)
    assert payload["detailed"]
    scenario = payload["scenarios"][0]
    progress = scenario["task_progress"]
    assert progress["targets"][0]["discovery_source"] == "full_information"
    assert progress["timeline"][0]["discovered_count"] == 1
    assert progress["timeline"][0]["scouted_count"] == 0
    assert progress["timeline"][-1]["completed_count"] == 1
    assert progress["targets"][0]["completed_at"] == scenario["makespan"]
    # Defaults choose the output directory, while explicit suite/map flags win.
    with patch("learning.test_single_detailed.DEFAULT_OUTPUT_DIR", directory / "default"):
        detailed_main([
            "--policy", "fi-opt", "--suite", str(suite_path),
            "--map-path", str(map_path), "--no-progress",
        ])
    default_outputs = list((directory / "default").glob("*_detailed.json"))
    assert len(default_outputs) == 1
    assert json.loads(default_outputs[0].read_text())["suite_id"] == "temporary_suite"
    with patch.object(evaluation_module, "evaluate", side_effect=AssertionError), \
            patch("sys.stderr"):
        try:
            detailed_main(["--policy", "fi-opt", "--output", str(output)])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("existing detailed results were overwritten")
    assert output.read_bytes() == original


def test_evaluation_render_requires_one_filtered_scenario():
    missing_checkpoint = Path("/definitely/missing/checkpoint.pt")
    for filters in ({}, {"agent_count": 4, "target_count": 8}):
        try:
            evaluate_suite(
                missing_checkpoint, suite="test", render=True, **filters)
        except ValueError as error:
            assert str(error) == (
                "--render requires exactly one selected evaluation scenario")
        else:
            raise AssertionError("multi-scenario rendering was accepted")

    for filters in (
        {"agent_config": "agents_04_b", "target_config": "targets_08_c"},
        {"limit": 1},
    ):
        try:
            evaluate_suite(
                missing_checkpoint, suite="test", render=True, **filters)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(
                "single-scenario rendering did not reach checkpoint")


def test_evaluation_analysis_aggregates_size_and_failure_metrics():
    def record(identifier, agents, targets, makespan, completed, deaths,
               remaining, regret, stalled=False, all_dead=False):
        return {
            "suite_id": "synthetic_suite",
            "scenario_id": identifier,
            "agent_count": agents,
            "target_count": targets,
            "makespan": makespan,
            "normalized_regret": regret,
            "completed": completed,
            "deaths": deaths,
            "remaining_targets": remaining,
            "stalled": stalled,
            "all_agents_dead": all_dead,
        }

    payload = {
        "suite_id": "synthetic_suite",
        "scenarios": [
            record("a", 3, 5, 10.0, True, 0, 0, 0.1),
            record("b", 3, 5, 14.0, False, 1, 2, 0.4, stalled=True),
            record("c", 4, 5, 8.0, True, 2, 0, -0.1),
        ],
    }
    analysis = analyze_evaluation(payload)
    overall = analysis["overall"]
    assert overall["scenario_count"] == 3
    assert overall["completion_count"] == 2
    assert abs(overall["failure_rate"] - 1 / 3) < 1e-12
    assert overall["total_deaths"] == 3
    assert abs(overall["death_scenario_rate"] - 2 / 3) < 1e-12
    assert abs(overall["agent_mortality_rate"] - 0.3) < 1e-12
    assert overall["completed_makespan"] == {
        "count": 2, "mean": 9.0, "std": 1.0}

    by_size = {
        (item["agent_count"], item["target_count"]): item
        for item in analysis["by_agent_and_target_count"]
    }
    assert by_size[(3, 5)]["makespan"] == {
        "count": 2, "mean": 12.0, "std": 2.0}
    assert by_size[(3, 5)]["completion_rate"] == 0.5
    assert by_size[(4, 5)]["makespan"] == {
        "count": 1, "mean": 8.0, "std": 0.0}
    rendered = format_analysis(analysis)
    assert "Makespan by problem size" in rendered
    assert "Completion by problem size" in rendered
    assert "Scenarios with a death: 2/3 (66.67%)" in rendered
    assert "Agent deaths: 3/10 deployed (30.00%)" in rendered
    assert "mean_planning_time_seconds" not in overall
    assert "Average planning time per replan" not in rendered

    timed_payload = copy.deepcopy(payload)
    timed_payload["scenarios"][0].update(
        mean_planning_time_seconds=0.2, planning_call_count=2)
    timed_payload["scenarios"][1].update(
        mean_planning_time_seconds=0.4, planning_call_count=8)
    timed_payload["scenarios"][2]["mean_planning_time_seconds"] = None
    timed = analyze_evaluation(timed_payload)
    assert abs(timed["overall"]["mean_planning_time_seconds"] - 0.3) < 1e-12
    assert timed["overall"]["planning_timed_rollouts"] == 2
    assert "Average planning time per replan: 300.000 ms" in format_analysis(timed)

    timed_payload["scenarios"][1]["mean_planning_time_seconds"] = -1.0
    try:
        analyze_evaluation(timed_payload)
    except ValueError as error:
        assert "invalid mean_planning_time_seconds" in str(error)
    else:
        raise AssertionError("negative planning time was accepted")


def test_evaluation_planning_time_averages_scenario_means_equally():
    def record(total, calls):
        return {
            "return": 0.0,
            "makespan": 1.0,
            "oracle_makespan": 1.0,
            "fi_opt_makespan": 1.0,
            "normalized_regret": 0.0,
            "completed": True,
            "deaths": 0,
            "remaining_targets": 0,
            "stalled": False,
            "all_agents_dead": False,
            **_planning_metrics(total, calls),
        }

    summary = summarize_evaluation([
        record(0.4, 2), record(1.2, 4),
    ])
    assert abs(summary["mean_planning_time_seconds"] - 0.25) < 1e-12
    assert summary["planning_timed_rollouts"] == 2
    assert _planning_metrics(0.0, 0)["mean_planning_time_seconds"] is None


def test_rendering_uses_distinct_agent_colors_and_stacks_shared_labels():
    colors = [_agent_color(index) for index in range(6)]
    assert len(set(colors)) == 6
    assert _stacked_agent_label_offsets(
        [(1, 2), (3, 4), (1.0, 2.0), (1, 2)]) == [
            (0, 16), (0, 16), (0, 36), (0, 56)]


def _graph_model(num_target_types=2, use_critic=True, decoder_mode="mlp",
                 use_position=False, use_agent_alive=True, use_task_completed=True):
    torch.manual_seed(11)
    config = replace(
        load_config().model,
        architecture="task_graph",
        num_target_types=num_target_types,
        model_dim=32,
        num_heads=4,
        message_passing_blocks=2,
        distance_embedding_dim=8,
        critic_hidden_dim=16,
        use_critic=use_critic,
        decoder_mode=decoder_mode,
        use_position=use_position,
        use_agent_alive=use_agent_alive,
        use_task_completed=use_task_completed,
    )
    model = HeterogeneousGraphPolicy(config)
    model.eval()
    return model


def _valid_task_distances(observation, batch_index=0):
    values = []
    for distance_name, mask_name in (
            ("agent_target_distances", "agent_target_distance_mask"),
            ("agent_action_distances", "agent_action_distance_mask"),
            ("action_target_distances", "action_target_distance_mask")):
        distances = getattr(observation, distance_name)[batch_index].squeeze(-1)
        mask = getattr(observation, mask_name)[batch_index]
        values.append(distances[mask])
    return torch.cat(values)


def test_tensor_router_matches_masked_networkx_shortest_paths():
    graph = _line(6)
    edges = list(graph.edges(data="distance"))
    router = GridRouter.from_edges(
        len(graph), [u for u, _v, _w in edges],
        [v for _u, v, _w in edges], [w for _u, _v, w in edges])
    blocked = torch.zeros((2, len(graph)), dtype=torch.bool)
    blocked[1, 3] = True
    result = router.shortest_paths(torch.tensor([0, 0]),
                                   torch.tensor([5, 5]), blocked)
    assert result.distances[0] == 5
    assert result.goals_reached.tolist() == [True, False]
    paths, lengths = router.reconstruct_paths(
        result, torch.tensor([0, 0]), torch.tensor([5, 5]))
    assert paths[0, :lengths[0]].tolist() == [0, 1, 2, 3, 4, 5]


def test_cugraph_router_caches_only_unmodified_base_rows():
    router = object.__new__(CuGraphRouter)
    router.num_nodes = 4
    router.target_nodes = ()
    router.max_cached_routes = 8
    router._base_sssp_cache = OrderedDict()
    calls = []
    base = (
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        torch.tensor([-1, 0, 1, 2]),
    )
    blocked = (
        torch.tensor([0.0, 1.0, torch.inf, torch.inf]),
        torch.tensor([-1, 0, -1, -1]),
    )

    def fake_run(source, blocked_nodes, graph=None):
        key = tuple(blocked_nodes)
        calls.append((int(source), key))
        return base if not key else blocked

    router._run_sssp = fake_run
    router.graph = lambda blocked_nodes=(): tuple(blocked_nodes)

    unmodified = router.sssp([0])
    assert torch.equal(unmodified.distances[0], base[0])
    assert calls == [(0, ())]
    router.sssp([0])
    assert calls == [(0, ())]

    rerouted = router.sssp([0], blocked_nodes=[2])
    assert torch.equal(rerouted.distances[0], blocked[0])
    assert calls == [(0, ()), (0, (2,))]
    assert list(router._base_sssp_cache) == [0]

    # Blocked results and their graph variants do not survive this call.
    router.sssp([0], blocked_nodes=[2])
    assert calls == [(0, ()), (0, (2,)), (0, (2,))]


def test_cugraph_batched_sssp_is_exact_for_independent_blockers_when_cuda():
    if not torch.cuda.is_available():
        return
    router = CuGraphRouter(
        5,
        [0, 1, 2, 0, 4, 2, 3],
        [1, 2, 3, 4, 3, 1, 4],
        [1.0, 1.0, 1.0, 10.0, 1.0, 4.0, 2.0],
    )
    sources = torch.tensor([0, 0, 2], device="cuda")
    blocked = torch.zeros((3, 5), dtype=torch.bool, device="cuda")
    blocked[1, 2] = True
    blocked[2, 2] = True  # A query source is always made traversable.
    result = router.sssp_batch(sources, blocked)
    assert torch.allclose(result.distances[:, 3], torch.tensor(
        [3.0, 11.0, 1.0], device="cuda"))
    assert result.predecessors[:, 3].tolist() == [2, 4, 2]

    # Force separate chunks; row ordering, source unblocking, asymmetric costs
    # and unreachable nodes must agree with the single-traversal route bank.
    router.max_batch_vertices = 5
    chunked = router.sssp_batch(sources, blocked)
    assert torch.equal(chunked.distances, result.distances)
    assert torch.equal(chunked.predecessors, result.predecessors)
    assert not router._base_sssp_cache


def test_cugraph_query_chunks_bound_both_edges_and_vertices():
    router = object.__new__(CuGraphRouter)
    router.num_nodes = 4096
    router._edges = range(32_768)
    router.max_batch_edges = 1_000_000
    router.max_batch_vertices = 262_144
    assert router._query_batch_size() == 30
    router.max_batch_vertices = 8192
    assert router._query_batch_size() == 2
    router.max_batch_vertices = 1
    assert router._query_batch_size() == 1


def test_cugraph_temporary_allocations_do_not_accumulate_when_cuda():
    if not torch.cuda.is_available():
        return
    from rmm.statistics import enable_statistics, get_statistics

    enable_statistics()
    nodes = 64
    sources = list(range(nodes - 1)) + list(range(1, nodes))
    destinations = list(range(1, nodes)) + list(range(nodes - 1))
    router = CuGraphRouter(
        nodes, sources, destinations, [1.0] * len(sources),
        max_batch_vertices=8 * nodes)

    def exercise():
        for count in (1, 7, 16, 3, 11):
            query_sources = torch.arange(count, device="cuda")
            blocked = torch.zeros((count, nodes), dtype=torch.bool, device="cuda")
            blocked[:, count + 1] = True
            result = router.sssp_batch(query_sources, blocked)
            assert (result.distances[
                torch.arange(count, device="cuda"), query_sources] == 0).all()

    exercise()  # Warm up CUDA/library state before comparing live allocations.
    torch.cuda.synchronize()
    torch_before = torch.cuda.memory_allocated()
    rmm_before = get_statistics().current_bytes
    rmm_count_before = get_statistics().current_count
    for _ in range(4):
        exercise()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= torch_before + 1024 ** 2
    # Even a one-byte native membership result leaked per call is a regression.
    assert get_statistics().current_bytes == rmm_before
    assert get_statistics().current_count == rmm_count_before
    assert not router._base_sssp_cache


def test_training_diagnostics_are_optional_and_keep_only_scalars():
    from learning.gpu_sim.diagnostics import TrainingDiagnostics

    disabled = TrainingDiagnostics("cuda", enabled=False)
    # Disabled diagnostics work without a driver and do not synchronize CUDA.
    with patch("torch.cuda.synchronize", side_effect=AssertionError):
        disabled.reset()
        with disabled.phase("work"):
            pass
        assert disabled.report(1) == {}
    enabled = TrainingDiagnostics("cpu", enabled=True)
    with enabled.phase("work"):
        pass
    assert enabled.metrics["time/work_seconds"] >= 0
    assert all(isinstance(value, float) for value in enabled.metrics.values())
    enabled.reset()
    assert not enabled.metrics


def test_cuda_relation_cache_matches_direct_queries_and_ignores_reveals():
    if not torch.cuda.is_available():
        return
    graph = _line(5)
    graph.edges[3, 2]["distance"] = 4.0
    for node in (2, 4):
        graph.nodes[node].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    world = TensorWorld.from_networkx(graph, load_config().candidates, device="cuda")
    builder = TensorObservationBuilder(world, 2)
    state = TensorEpisodeState.create(
        world, torch.tensor([0, 0], device="cuda"),
        torch.ones((2, 1, 3), dtype=torch.bool, device="cuda"),
        torch.tensor([[1, 2], [1, 2]], device="cuda"))
    planning = torch.ones(2, dtype=torch.bool, device="cuda")
    for live in ((True, True), (True, False), (False, True), (False, False)):
        state.target_live[:] = torch.tensor(live, device="cuda")
        expected = builder._compute_action_target_distances(state, planning)
        with patch.object(world.reverse_router, "sssp_batch",
                          wraps=world.reverse_router.sssp_batch) as routes:
            actual = builder._action_target_distances(state, planning)
            assert torch.equal(actual, expected)
            assert routes.call_count == 1
            # Revelation and agent positions affect other features, not these
            # directed region minima. No second reverse SSSP bank is needed.
            state.target_known.logical_not_()
            state.positions[:] = 1
            again = builder._action_target_distances(state, planning)
            assert torch.equal(again, expected)
            assert routes.call_count == 1
    assert builder._relation_distances.numel() * 4 <= 8 * 1024 ** 2


def test_legacy_batch_size_config_is_split():
    payload = """
model:
  num_target_types: 1
  model_dim: 8
  num_heads: 1
  num_world_blocks: 1
  dropout: 0.0
  relation_hidden_dim: 4
candidates:
  staging_per_target: 1
  staging_capacity: 1
  include_wait: true
  include_continue: true
reinforce:
  learning_rate: 0.001
  entropy_coefficient: 0.0
  baseline_decay: 0.9
  death_penalty: 1.0
  incomplete_penalty: 1.0
  gradient_clip_norm: 1.0
training:
  episodes: 2
  batch_size: 3
  num_agents: 1
  seed: 0
  device: cpu
  checkpoint: checkpoints
  wandb: false
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.yaml"
        path.write_text(payload)
        config = load_config(path)
    assert config.training.simulation_batch_size == 3
    assert config.training.reinforce_batch_size == 3
    assert config.instances.min_targets == 7
    assert config.instances.max_targets == 7
    assert config.instances.min_agents is None
    assert config.instances.max_agents is None
    assert config.candidates.include_pair_staging


def test_episode_agent_count_uses_seeded_range_and_fixed_overrides():
    instances = InstanceConfig(
        min_targets=5, max_targets=9, min_agents=3, max_agents=6)
    counts = [
        _episode_agent_count(seed, 4, instances)
        for seed in range(32)
    ]
    assert counts == [
        _episode_agent_count(seed, 4, instances)
        for seed in range(32)
    ]
    assert set(counts) == {3, 4, 5, 6}
    assert _episode_agent_count(
        0, 4, instances, requested_num_agents=5) == 5
    assert _episode_agent_count(
        0, 4, instances, agent_capabilities=[{0}, {1}, {2}]) == 3


def test_parallel_tsp_partitions_targets_to_minimize_makespan():
    graph = nx.DiGraph()
    for node in ("s", "a", "b"):
        graph.add_node(node, type="intermediate")
    graph.nodes["a"].update(type="target_unreached", rps_type=1)
    graph.nodes["b"].update(type="target_unreached", rps_type=1)
    for u, v, cost in (("s", "a", 2), ("a", "s", 2),
                       ("s", "b", 3), ("b", "s", 3),
                       ("a", "b", 5), ("b", "a", 5)):
        graph.add_edge(u, v, distance=cost)
    agents = [Agent("s", capabilities={1}), Agent("s", capabilities={1})]
    assert parallel_tsp(graph, agents) == 3.0


def _full_information_route_graph():
    graph = nx.DiGraph()
    for node, node_type in (
        ("s", "source"), ("x", "intermediate"),
        ("a", "target_unreached"), ("b", "target_unreached"),
    ):
        graph.add_node(node, type=node_type, pos=(0, 0), visible_edges=[])
    graph.nodes["a"]["rps_type"] = 1
    graph.nodes["b"]["rps_type"] = 2
    for source, target, distance in (
        ("s", "x", 1.0), ("x", "a", 1.0), ("x", "b", 3.0),
        ("a", "x", 7.0), ("b", "x", 2.0),
    ):
        graph.add_edge(source, target, distance=distance, observed_edge=False)
    return graph


def test_full_information_plan_is_complete_disjoint_and_executable():
    truth = _full_information_route_graph()
    agents = [
        Agent("s", capabilities={1}),
        Agent("s", capabilities={2}),
        Agent("s", capabilities=set()),
    ]
    graph_before = copy.deepcopy(nx.node_link_data(truth))
    agent_before = [copy.deepcopy(agent.__dict__) for agent in agents]
    plan = solve_full_information(truth, agents)
    assert plan.feasible and plan.exact
    assigned = [target for values in plan.assignments for target in values]
    assert sorted(assigned) == ["a", "b"]
    assert len(set(assigned)) == len(assigned)
    assert plan.assignments[2] == () and plan.paths[2] == ("s",)
    assert plan.makespan == 4.0
    assert full_information_makespan(truth, agents) == plan.makespan
    assert nx.node_link_data(truth) == graph_before
    assert [agent.__dict__ for agent in agents] == agent_before

    env = truth.copy()
    init_target_types(env, truth, {"a": 1, "b": 2})
    for agent, route in zip(agents, plan.paths):
        agent.planned_path = list(route)
    result = run_simulation(
        env, truth, agents,
        policy=lambda _env, _agents, **_kwargs: None,
    )
    assert result["completed"] and not result["deaths"]
    assert result["makespan"] == plan.makespan


def test_full_information_routes_record_incidental_supported_targets():
    graph = nx.DiGraph()
    for node, node_type, target_type in (
        ("s", "source", None), ("a", "target_unreached", 1),
        ("b", "target_unreached", 1),
    ):
        graph.add_node(node, type=node_type, rps_type=target_type)
    graph.add_edge("s", "a", distance=1.0)
    graph.add_edge("a", "b", distance=1.0)
    plan = solve_full_information(graph, [Agent("s", capabilities={1})])
    assert plan.feasible
    assert plan.assignments == (("a", "b"),)
    assert plan.target_orders == (("a", "b"),)
    assert plan.paths == (("s", "a", "b"),)


def test_full_information_can_revisit_an_already_serviced_target():
    graph = nx.DiGraph()
    for node, node_type in (
        ("s", "source"), ("a", "target_unreached"),
        ("b", "target_unreached"), ("c", "target_unreached"),
    ):
        graph.add_node(node, type=node_type, rps_type=1)
    for source, target in (
        ("s", "a"), ("a", "b"), ("b", "a"), ("a", "c"),
    ):
        graph.add_edge(source, target, distance=1.0)
    plan = solve_full_information(graph, [Agent("s", capabilities={1})])
    assert plan.feasible and plan.makespan == 4.0
    assert plan.paths == (("s", "a", "b", "a", "c"),)
    assert plan.target_orders == (("a", "b", "c"),)


def test_full_information_never_relies_on_another_agent_clearing_a_target():
    graph = nx.DiGraph()
    for node, node_type, target_type in (
        ("s", "source", None), ("a", "target_unreached", 1),
        ("b", "target_unreached", 2),
    ):
        graph.add_node(node, type=node_type, rps_type=target_type)
    graph.add_edge("s", "a", distance=1.0)
    graph.add_edge("a", "b", distance=1.0)
    agents = [Agent("s", capabilities={1}), Agent("s", capabilities={2})]
    plan = solve_full_information(graph, agents)
    assert not plan.feasible
    assert "executable" in plan.diagnostic["message"]
    try:
        full_information_makespan(graph, agents)
    except FullInformationInfeasibleError as error:
        assert error.diagnostic == plan.diagnostic
    else:
        raise AssertionError("scalar FI-OPT did not report infeasibility")

    unsupported = graph.copy()
    unsupported.nodes["b"]["rps_type"] = 3
    unsupported_plan = solve_full_information(unsupported, agents)
    assert not unsupported_plan.feasible
    assert unsupported_plan.diagnostic["targets"] == ["b"]


def test_full_information_handles_overlap_asymmetry_release_and_permutations():
    graph = _full_information_route_graph()
    agents = [
        Agent("s", capabilities={1, 2}),
        Agent("s", capabilities={1}),
    ]
    first = full_information_makespan(
        graph, agents, release_times=[2.0, 0.0])
    second = full_information_makespan(
        graph, list(reversed(agents)), release_times=[0.0, 2.0])
    assert first == second == 6.0
    assert graph.edges["a", "x"]["distance"] == 7.0

    reordered = nx.DiGraph()
    for node in reversed(list(graph.nodes)):
        reordered.add_node(node, **graph.nodes[node])
    for source, target, data in reversed(list(graph.edges(data=True))):
        reordered.add_edge(source, target, **data)
    assert full_information_makespan(
        reordered, agents, release_times=[2.0, 0.0]) == first


def test_full_information_matches_brute_force_on_a_small_complete_graph():
    graph = nx.DiGraph()
    targets = ("a", "b", "c")
    types = {"a": 1, "b": 1, "c": 2}
    for node in ("s", *targets):
        graph.add_node(
            node,
            type="source" if node == "s" else "target_unreached",
            rps_type=types.get(node),
        )
    distances = {
        ("s", "a"): 2, ("s", "b"): 4, ("s", "c"): 3,
        ("a", "b"): 1, ("a", "c"): 5,
        ("b", "a"): 2, ("b", "c"): 1,
        ("c", "a"): 1, ("c", "b"): 3,
    }
    for edge, distance in distances.items():
        graph.add_edge(*edge, distance=float(distance))
    agents = [
        Agent("s", capabilities={0, 1, 2}),
        Agent("s", capabilities={1}),
    ]

    brute = float("inf")
    for owners in product(range(len(agents)), repeat=len(targets)):
        if any(
            not agents[owner].can_service(types[target])
            for target, owner in zip(targets, owners)
        ):
            continue
        finish = []
        for agent_index in range(len(agents)):
            jobs = [
                target for target, owner in zip(targets, owners)
                if owner == agent_index
            ]
            if not jobs:
                finish.append(0.0)
                continue
            finish.append(min(
                sum(
                    distances[("s" if index == 0 else order[index - 1], target)]
                    for index, target in enumerate(order)
                )
                for order in permutations(jobs)
            ))
        brute = min(brute, max(finish))
    assert full_information_makespan(graph, agents) == brute


def test_full_information_targetless_normalization_has_no_divide_by_zero():
    graph = _line(2)
    agents = [Agent(0, capabilities={1})]
    assert full_information_makespan(graph, agents) == 0.0
    result = {
        "makespan": 0.0, "num_deaths": 0, "remaining_targets": [],
    }
    assert calculate_episode_return(result, oracle_makespan=0.0) == 0.0


def _cooperative_visibility_graph():
    graph = nx.DiGraph()
    positions = {
        "s": (0, 0), "l1": (-1, 0), "l2": (-2, 0),
        "r1": (1, 0), "r2": (2, 0),
        "tl": (-2, 1), "tr": (2, 1),
    }
    for node, position in positions.items():
        graph.add_node(
            node, pos=position, height=0.0,
            type=("source" if node == "s" else
                  "target_unreached" if node in {"tl", "tr"} else
                  "intermediate"),
            visible_edges=[],
        )
    for source, target in (
        ("s", "l1"), ("l1", "l2"), ("s", "r1"), ("r1", "r2"),
        ("l2", "tl"), ("r2", "tr"),
    ):
        graph.add_edge(source, target, distance=1.0, observed_edge=False)
        graph.add_edge(target, source, distance=1.0, observed_edge=False)
    graph.nodes["l2"]["visible_edges"] = [("l2", "tl")]
    graph.nodes["r2"]["visible_edges"] = [("r2", "tr")]
    graph.nodes["tl"]["rps_type"] = UNKNOWN_TYPE
    graph.nodes["tr"]["rps_type"] = UNKNOWN_TYPE
    return graph


def test_cooperative_scouting_uses_all_scouts_and_minimizes_last_reveal():
    env = _cooperative_visibility_graph()
    beliefs_before = {
        target: env.nodes[target]["rps_type"] for target in ("tl", "tr")
    }
    scouts = [Agent("s", capabilities={0}), Agent("s", capabilities={0, 1})]
    plan = solve_cooperative_scouting(env, scouts)
    assert plan.feasible and plan.exact
    assert plan.makespan == 2.0
    assert all(plan.responsibilities)
    assert {target for values in plan.responsibilities for target in values} == {
        "tl", "tr"
    }
    assert all(path[0] == "s" for path in plan.paths)
    assert {
        target: env.nodes[target]["rps_type"] for target in ("tl", "tr")
    } == beliefs_before

    pure_plan = solve_cooperative_scouting(
        env,
        [Agent("s", capabilities={0}), Agent("s", capabilities={0})],
    )
    assert pure_plan.makespan == plan.makespan
    assert pure_plan.paths == plan.paths

    single = solve_cooperative_scouting(env, scouts[:1])
    wrp_path, wrp_schedule, missing, _stats = solve_cover_walk(
        env, "s", ["tl", "tr"], {"tl", "tr"}, weight=1.0)
    assert not missing and wrp_path is not None
    assert single.makespan == max(wrp_schedule.values()) == 6.0


def test_strict_scout_then_execute_waits_and_matches_phase_prediction():
    env = _cooperative_visibility_graph()
    truth = env.copy()
    init_target_types(env, truth, {"tl": 1, "tr": 1})
    agents = [
        Agent("s", capabilities={0}), Agent("s", capabilities={0}),
        Agent("s", capabilities={1}),
    ]
    policy = ScoutThenExecutePolicy()
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    assert policy.diagnostics["scouting_completion_time"] == 2.0
    assert result["makespan"] == policy.diagnostics["predicted_makespan"]
    json.dumps(policy.diagnostics)
    first_service = min(
        event["time"] for event in result["events"]
        if event["event"] == "agent_wins"
    )
    assert first_service > policy.diagnostics["scouting_completion_time"]
    assert agents[2].trajectory[0] == "s"


def test_hybrid_scouts_join_overlapping_full_information_execution():
    env = _cooperative_visibility_graph()
    truth = env.copy()
    init_target_types(env, truth, {"tl": 1, "tr": 2})
    agents = [
        Agent("s", capabilities={0, 1}),
        Agent("s", capabilities={0, 2}),
        Agent("s", capabilities={1, 2}),
    ]
    policy = ScoutThenExecutePolicy()
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    assert "tl" in policy.diagnostics["fi_assignments"][0]
    assert "tr" in policy.diagnostics["fi_assignments"][1]
    assert agents[0].trajectory[-1] == "tl"
    assert agents[1].trajectory[-1] == "tr"


def test_scout_then_execute_reports_unscoutable_and_handles_transit_boundary():
    env = _cooperative_visibility_graph()
    env.nodes["r2"]["visible_edges"] = []
    scouts = [Agent("s", capabilities={0}), Agent("s", capabilities={0, 1})]
    plan = solve_cooperative_scouting(env, scouts)
    assert not plan.feasible and plan.unscoutable == ("tr",)

    graph = nx.DiGraph()
    for node, node_type in (
        ("s", "source"), ("x", "intermediate"),
        ("t", "target_unreached"),
    ):
        graph.add_node(node, type=node_type, rps_type=1, visible_edges=[])
    graph.add_edge("s", "x", distance=3.0)
    graph.add_edge("x", "t", distance=4.0)
    agent = Agent("s", capabilities={0, 1})
    policy = ScoutThenExecutePolicy()
    policy.set_runtime_state([agent], [("s", "x", 0.0, 3.0)], 1.0)
    policy(graph, [agent])
    assert policy.phase == "execution"
    assert agent.planned_path == ["x", "t"]
    assert policy.diagnostics["execution_makespan_estimate"] == 6.0
    assert policy.diagnostics["predicted_makespan"] == 7.0


def test_classical_evaluation_needs_no_checkpoint_or_model_load():
    truth = _line(3)
    truth.nodes[2].update(type="target_unreached", rps_type=1)
    env = truth.copy()
    env.nodes[2]["rps_type"] = UNKNOWN_TYPE

    def factory(_case, map_path=None):
        return env.copy(), truth.copy(), [Agent(0, capabilities={1})]

    with patch.object(evaluation_module, "_case_factory", side_effect=factory), \
            patch.object(torch, "load", side_effect=AssertionError("loaded weights")):
        records, summary, _config, weights, _suite, _path = evaluate_suite(
            None, suite="development", limit=1, policy="fi-opt", device="cuda")
    assert weights is None
    assert records[0]["completed"]
    assert abs(records[0]["normalized_regret"]) < 1e-12
    assert records[0]["simulation_backend"] == "cpu_classical"
    assert records[0]["planning_call_count"] == 1
    assert records[0]["mean_planning_time_seconds"] >= 0.0
    assert summary["mean_planning_time_seconds"] == (
        records[0]["mean_planning_time_seconds"])

    scout_env = _cooperative_visibility_graph()
    scout_truth = scout_env.copy()
    init_target_types(scout_env, scout_truth, {"tl": 1, "tr": 1})

    def scout_factory(_case, map_path=None):
        return (
            scout_env.copy(), scout_truth.copy(),
            [Agent("s", capabilities={0}), Agent("s", capabilities={1})],
        )

    with patch.object(evaluation_module, "_case_factory", side_effect=scout_factory), \
            patch.object(torch, "load", side_effect=AssertionError("loaded weights")):
        scout_records, *_rest = evaluate_suite(
            None, suite="development", limit=1,
            policy="scout-then-execute", episodes=2)
    assert len(scout_records) == 2
    assert all(record["completed"] for record in scout_records)
    assert all(record["planning_call_count"] >= 1 for record in scout_records)

    try:
        evaluate_suite(None, suite="development", limit=1, policy="learned")
    except ValueError as error:
        assert "requires a checkpoint" in str(error)
    else:
        raise AssertionError("learned evaluation accepted a missing checkpoint")


def test_oracle_normalized_return_applies_dimensionless_failure_penalties():
    result = {
        "makespan": 120.0, "num_deaths": 1,
        "remaining_targets": ["target"],
    }
    value = calculate_episode_return(
        result, death_penalty=20.0, incomplete_penalty=60.0,
        oracle_makespan=100.0)
    assert abs(value - -80.2) < 1e-9


def test_oracle_reward_makes_immediate_stall_strictly_bad():
    stalled = {
        "makespan": 0.0, "num_deaths": 0,
        "remaining_targets": list(range(7)),
    }
    value = calculate_episode_return(
        stalled, death_penalty=1.0, incomplete_penalty=10.0,
        oracle_makespan=146.475743)
    assert abs(value - -69.0) < 1e-9


def test_cpu_and_tensor_fi_normalized_rewards_match():
    class FinishedTensorState:
        clock = torch.tensor([120.0])
        target_live = torch.tensor([[True, True]])
        deaths = torch.tensor([1])
        alive = torch.tensor([[True]])
        moving = torch.tensor([[False]])
        needs_replan = torch.tensor([[False]])
        stalled = torch.tensor([True])

        def completed(self):
            return ~self.target_live.any(dim=1)

    cpu = calculate_episode_return(
        {"makespan": 120.0, "num_deaths": 1,
         "remaining_targets": ["a", "b"]},
        death_penalty=20.0, incomplete_penalty=60.0,
        oracle_makespan=100.0,
    )
    tensor = collect_tensor_episodes(
        None, FinishedTensorState(), None,
        death_penalty=20.0, incomplete_penalty=60.0,
        training=False, oracle_makespans=100.0,
    )
    assert abs(float(tensor.returns[0]) - cpu) < 1e-4
    assert abs(float(tensor.normalized_regrets[0]) - 0.2) < 1e-6


def test_tensor_episode_transition_matches_cpu_line_episode():
    graph, agents = _instance(False)
    agents[0].capabilities = frozenset({0, 1})

    def straight_line(env, active_agents, **_kwargs):
        for agent in active_agents:
            agent.planned_path = list(range(agent.position, 5))

    cpu = run_simulation(graph, graph.copy(), agents, policy=straight_line)
    neighbors = torch.full((5, 2), -1, dtype=torch.long)
    costs = torch.full((5, 2), torch.inf)
    for node in range(5):
        adjacent = list(graph.successors(node))
        neighbors[node, :len(adjacent)] = torch.tensor(adjacent)
        costs[node, :len(adjacent)] = 1.0
    visible = torch.zeros((5, 1), dtype=torch.bool)
    visible[3:, 0] = True
    world = SimpleNamespace(
        positions=torch.zeros((5, 2)), targets=[4],
        target_nodes=torch.tensor([4]), visible_targets=visible,
        neighbors=neighbors, edge_cost=costs)
    state = TensorEpisodeState.create(
        world, [0], [[[True, True]]], [[1]])
    for next_node in range(1, 5):
        state.dispatch_next_hops(torch.tensor([[next_node]]),
                                 torch.tensor([[True]]))
        state.advance()
    assert state.completed().item() == cpu["completed"]
    assert state.clock.item() == cpu["makespan"]
    assert state.traversal_cost.sum().item() == cpu["total_cost"]
    assert state.deaths.item() == cpu["num_deaths"]
    assert state.target_known.item()


def test_yaml_configuration_loads_and_validates():
    config = load_config()
    assert config.model.model_dim % config.model.num_heads == 0
    assert config.model.feature_schema_version == CURRENT_FEATURE_SCHEMA_VERSION
    assert config.model.edge_normalization in {
        RAW_EDGE_NORMALIZATION, PER_DECISION_EDGE_NORMALIZATION, MAX_EDGE_NORMALIZATION}
    assert isinstance(config.model.use_position, bool)
    assert config.model.edge_normalization_epsilon > 0
    assert config.candidates.include_wait
    assert isinstance(config.candidates.include_pair_staging, bool)
    assert not config.candidates.allow_unknown_target_actions
    assert config.instances.min_targets == 5
    assert config.instances.max_targets == 9
    assert config.instances.min_agents == 3
    assert config.instances.max_agents == 6
    assert config.training.simulation_batch_size >= 1
    assert config.training.reinforce_batch_size >= 1
    assert config.training.device in {"auto", "cpu", "cuda"}


def test_checkpoint_directories_prefer_best_then_final_then_latest():
    from learning.policy.evaluation import load_policy

    model = _graph_model()
    config = replace(load_config(), model=model.config)
    names = ("best_weights.pt", "trained_weights.pt", "latest_weights.pt")
    states = {}
    marker = next(iter(model.state_dict()))
    for index, name in enumerate(names):
        states[name] = {key: value.clone() for key, value in model.state_dict().items()}
        states[name][marker].fill_(index + 1)
    with tempfile.TemporaryDirectory() as directory:
        for index, present in enumerate((names, names[:1] + names[2:], names[1:], names[2:])):
            run = Path(directory) / str(index)
            run.mkdir()
            config_path = run / "config.yaml"
            config_path.write_text(yaml.safe_dump(_serialized_run_config(config)))
            for name in present:
                torch.save(states[name], run / name)
            weights_path, saved_config_path = evaluation_module._resolve_checkpoint(run)
            assert weights_path == run / present[0]
            assert saved_config_path == config_path
            loaded, _adapter = load_policy(run, device="cpu")
            assert torch.equal(loaded.state_dict()[marker], states[present[0]][marker])
            # Explicit paths override the directory's preference in both APIs.
            for name in present:
                explicit, _saved_config = evaluation_module._resolve_checkpoint(run / name)
                assert explicit == run / name
                loaded, _adapter = load_policy(run / name, device="cpu")
                assert torch.equal(loaded.state_dict()[marker], states[name][marker])
        empty = Path(directory) / "empty"
        empty.mkdir()
        try:
            evaluation_module._resolve_checkpoint(empty)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("an empty checkpoint directory was accepted")


def test_training_writes_latest_best_and_final_weights():
    config = load_config()
    model_config = replace(
        config.model,
        num_target_types=1,
        model_dim=16,
        num_heads=4,
        message_passing_blocks=1,
        distance_embedding_dim=4,
    )
    candidate_config = replace(
        config.candidates, staging_per_target=0, include_wait=False,
        allow_unknown_target_actions=True)

    def instance_factory(_episode):
        truth = _line(3)
        truth.nodes[2].update(type="target_unreached", rps_type=1)
        env = truth.copy()
        env.nodes[2]["rps_type"] = UNKNOWN_TYPE
        return env, truth, [Agent(0, capabilities={1})]

    with tempfile.TemporaryDirectory() as directory:
        training_config = replace(
            config.training, episodes=2, device="cpu", wandb=False,
            checkpoint=directory)
        run_config = LearningConfig(
            model_config, candidate_config, config.reinforce,
            training_config,
            replace(config.instances, min_targets=1, max_targets=1))
        model, history = train(
            instance_factory, 1, episodes=2,
            model_config=model_config,
            candidate_config=candidate_config,
            reinforce_config=config.reinforce,
            device="cpu", checkpoint=directory, run_config=run_config)
        run_directory = train.last_run_directory
        assert run_directory is not None
        expected_files = {
            "config.yaml", "checkpoint_state.yaml", "latest_weights.pt",
            "best_weights.pt", "trained_weights.pt",
        }
        assert expected_files <= {
            path.name for path in run_directory.iterdir()}
        checkpoint_state = yaml.safe_load(
            (run_directory / "checkpoint_state.yaml").read_text())
        assert checkpoint_state["latest_episodes_seen"] == 2
        assert checkpoint_state["best_episodes_seen"] in {1, 2}
        assert checkpoint_state["best_mean_return"] == max(
            record["return"] for record in history)
        latest = torch.load(
            run_directory / "latest_weights.pt", map_location="cpu",
            weights_only=True)
        assert all(torch.equal(value, latest[name])
                   for name, value in model.state_dict().items())
        from learning.policy.evaluation import load_policy
        best_model, _policy = load_policy(
            run_directory / "best_weights.pt", device="cpu")
        assert isinstance(best_model, HeterogeneousGraphPolicy)


def test_task_graph_checkpoint_evaluates_with_suite_type_count():
    base = load_config()
    model_config = replace(
        base.model, num_target_types=3, model_dim=16, num_heads=4,
        message_passing_blocks=2, distance_embedding_dim=4,
        critic_hidden_dim=8)
    config = replace(
        base, model=model_config,
        candidates=replace(
            base.candidates, staging_per_target=0,
            include_pair_staging=False, allow_unknown_target_actions=True),
        training=replace(base.training, device="cpu", wandb=False))
    model = build_policy(model_config)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.yaml").write_text(
            yaml.safe_dump(
                _serialized_run_config(config), sort_keys=False),
            encoding="utf-8")
        torch.save(model.state_dict(), checkpoint / "trained_weights.pt")
        map_path = _save_prepared_grid(directory, "runtime_map.pkl.gz", 2)
        suite_path = _write_single_case_suite(directory)
        records, _summary, runtime_config, *_rest = evaluate_suite(
            checkpoint, suite=suite_path, map_path=map_path,
            device="cpu", limit=1)
    assert len(records) == 1
    assert runtime_config.model.architecture == "task_graph"
    assert runtime_config.model.num_target_types == 1


def test_tensor_training_accumulates_microbatches_and_handles_partial_update():
    from learning.train import train_gpu

    config = load_config()
    config = replace(
        config,
        model=replace(config.model, num_target_types=1, model_dim=16,
                      num_heads=4, message_passing_blocks=1,
                      distance_embedding_dim=4, use_critic=True),
        candidates=replace(config.candidates, staging_per_target=0,
                           include_pair_staging=False, include_wait=False,
                           allow_unknown_target_actions=True),
        training=replace(config.training, wandb=False, device="cpu"),
        instances=replace(config.instances, map_path=None))

    # All overlays share one physical terrain; vary the target to exercise
    # rebuilding a world while reusing its immutable terrain.
    def shared_terrain_factory(episode):
        truth = _line(4)
        for node, pos in enumerate(((-10., 20.), (0., 20.), (10., 25.), (30., 40.))):
            truth.nodes[node]["pos"] = pos
        target = 2 + episode % 2
        truth.nodes[target].update(type="target_unreached", rps_type=1)
        env = truth.copy()
        env.nodes[target]["rps_type"] = UNKNOWN_TYPE
        return env, truth, [Agent(0, capabilities={1})]

    torch.manual_seed(73)
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        model, history, _world, run_directory = train_gpu(
            *shared_terrain_factory(0), episodes=5, simulation_batch_size=2,
            reinforce_batch_size=3, model_config=config.model,
            candidate_config=config.candidates, reinforce_config=config.reinforce,
            run_config=config, device="cpu", instance_factory=shared_terrain_factory)
    assert run_directory is None
    assert [record["episode"] for record in history] == list(range(5))
    assert all(record["completed"] for record in history)
    assert all(record["decisions"] == 1 for record in history)
    assert all(np.isfinite(record["loss"]) for record in history)
    assert all(not torch.is_tensor(value) for record in history for value in record.values())
    assert any(parameter.grad is not None for parameter in model.critic.parameters())
    # Exercise the tensor observation path on CPU too, including non-square
    # geometry, so normalization parity is checked without CUDA hardware.
    source = _world.nodes[1]
    cpu_observation = build_observation(
        _world.graph, [Agent(source, capabilities={1})], 1,
        candidate_config=config.candidates)
    state = TensorEpisodeState.create(_world, [1], [[[False, True]]], [[1]])
    tensor_observation = TensorObservationBuilder(
        _world, 1, edge_normalization=config.model.edge_normalization).build(state)[0]
    assert torch.allclose(cpu_observation.agent_positions,
                          torch.tensor([[[0.25, 0.0]]]))
    assert torch.allclose(cpu_observation.agent_positions,
                          tensor_observation.agent_positions)
    assert torch.allclose(cpu_observation.target_positions,
                          tensor_observation.target_positions)
    model.eval()
    assert torch.allclose(model(cpu_observation), model(tensor_observation), atol=1e-5)


def test_checkpoint_map_identity_relocation_and_mismatch_override():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        original_map = _save_prepared_grid(
            directory, "original.pkl.gz", 2)
        mismatched_map = _save_prepared_grid(
            directory, "mismatched.pkl.gz", 2, height_offset=5.0)
        relocated_map = directory / "relocated.pkl.gz"
        shutil.copyfile(original_map, relocated_map)
        suite_path = _write_single_case_suite(directory)
        resolved_map, _graph, _metadata, map_info = inspect_prepared_map(
            original_map)

        base = load_config()
        model_config = replace(
            base.model, num_target_types=1, model_dim=16, num_heads=4,
            num_world_blocks=1, message_passing_blocks=1,
            distance_embedding_dim=4)
        candidate_config = replace(
            base.candidates, staging_per_target=0,
            include_pair_staging=False, include_wait=False,
            allow_unknown_target_actions=True)
        training_config = replace(
            base.training, episodes=1, simulation_batch_size=1,
            reinforce_batch_size=1, num_agents=1, seed=0, device="cpu",
            checkpoint=str(directory / "checkpoints"), wandb=False)
        instance_config = replace(
            base.instances, min_targets=1, max_targets=1,
            min_agents=None, max_agents=None, map_path=str(resolved_map))
        run_config = LearningConfig(
            model_config, candidate_config, base.reinforce,
            training_config, instance_config)

        def instance_factory(_episode):
            return make_prepared_map_instance(
                seed=0, num_target_types=1, num_agents=1,
                source_position=(0, 0), target_positions=[(1, 1)],
                target_types=[1], agent_capabilities=[{0, 1}],
                map_path=resolved_map)

        _model, history = train(
            instance_factory, 1, episodes=1,
            model_config=model_config, candidate_config=candidate_config,
            reinforce_config=base.reinforce, device="cpu",
            checkpoint=training_config.checkpoint, run_config=run_config,
            prepared_map=map_info)
        assert len(history) == 1
        run_directory = train.last_run_directory
        saved = yaml.safe_load(
            (run_directory / "config.yaml").read_text(encoding="utf-8"))
        assert saved["instances"]["map_path"] == str(resolved_map)
        assert saved["prepared_map"]["sha256"] == map_info["sha256"]
        assert saved["prepared_map"]["schema_version"] == 1
        assert saved["prepared_map"]["dimensions"] == [2, 2]
        assert saved["prepared_map"]["metadata"]["terrain_label"] == (
            "original.pkl.gz")
        assert saved["model"]["feature_schema_version"] == (
            CURRENT_FEATURE_SCHEMA_VERSION)
        assert saved["model"]["edge_normalization"] == (
            model_config.edge_normalization)
        assert saved["feature_schema"]["version"] == (
            CURRENT_FEATURE_SCHEMA_VERSION)
        assert saved["model"]["use_position"] == model_config.use_position
        assert saved["feature_schema"]["use_position"] == model_config.use_position
        assert saved["candidates"]["allow_unknown_target_actions"]

        explicit_config_path = directory / "explicit-config.yaml"
        explicit_payload = dict(saved)
        explicit_payload["instances"] = dict(saved["instances"])
        explicit_payload["instances"]["map_path"] = str(mismatched_map)
        explicit_config_path.write_text(yaml.safe_dump(explicit_payload))
        try:
            evaluate_suite(
                run_directory, config_path=explicit_config_path,
                suite=suite_path, device="cpu")
        except ValueError as error:
            assert "does not match the checkpoint" in str(error)
        else:
            raise AssertionError(
                "checkpoint map took precedence over the explicit config")

        records, *_rest = evaluate_suite(
            run_directory, config_path=explicit_config_path,
            suite=suite_path, device="cpu",
            map_path=relocated_map)
        assert len(records) == 1
        assert evaluation_module.evaluate.last_prepared_map[
            "hash_matches_checkpoint"] is True
        assert evaluation_module.evaluate.last_prepared_map[
            "resolved_path"] == str(relocated_map.resolve())

        try:
            evaluate_suite(
                run_directory, suite=suite_path, device="cpu",
                map_path=mismatched_map)
        except ValueError as error:
            assert "does not match the checkpoint" in str(error)
            assert "--allow-map-mismatch" in str(error)
        else:
            raise AssertionError("a mismatched checkpoint map was accepted")

        mismatch_records, *_rest = evaluate_suite(
            run_directory, suite=suite_path, device="cpu",
            map_path=mismatched_map, allow_map_mismatch=True)
        assert len(mismatch_records) == 1
        evaluation_map = evaluation_module.evaluate.last_prepared_map
        assert evaluation_map["hash_matches_checkpoint"] is False
        assert evaluation_map["allow_map_mismatch"] is True
        assert evaluation_map["mismatch_override_used"] is True


def test_graph_and_transformer_configs_select_separate_policies():
    graph_config = load_config()
    transformer_path = Path(__file__).parents[1] / "learning" / "config_transformer.yaml"
    transformer_config = load_config(transformer_path)
    assert graph_config.model.architecture == "task_graph"
    assert not graph_config.model.use_critic
    assert transformer_config.model.architecture == "transformer"
    graph_policy = build_policy(graph_config.model)
    assert isinstance(graph_policy, HeterogeneousGraphPolicy)
    assert not graph_policy.has_critic
    assert isinstance(build_policy(transformer_config.model),
                      VanillaTransformerPolicy)


def test_task_graph_decoder_config_validates_and_preserves_legacy_checkpoints():
    config = load_config()
    assert config.model.decoder_mode == "mlp"
    assert isinstance(config.model.use_position, bool)
    for invalid in ({"decoder_mode": "cosine"}, {"decoder_mode": "unknown"},
                    {"use_position": "true"}, {"use_position": 1}):
        try:
            replace(config.model, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid decoder configuration accepted: {invalid}")

    payload = asdict(config)
    payload["model"].pop("decoder_mode")
    payload["model"].pop("use_position")
    payload["model"].pop("use_agent_alive")
    payload["model"].pop("use_task_completed")
    with tempfile.TemporaryDirectory() as directory:
        legacy_path = Path(directory) / "legacy.yaml"
        legacy_path.write_text(yaml.safe_dump(payload))
        legacy = load_config(legacy_path)
        assert legacy.model.decoder_mode == "mlp"
        assert not legacy.model.use_position
        assert legacy.model.use_agent_alive and legacy.model.use_task_completed
        model = _graph_model()
        restored = build_policy(replace(
            legacy.model, model_dim=32, message_passing_blocks=2,
            distance_embedding_dim=8, critic_hidden_dim=16))
        restored.load_state_dict(model.state_dict(), strict=True)

        payload["model"].update(decoder_mode="dot_product")
        dot_product_path = Path(directory) / "dot_product.yaml"
        dot_product_path.write_text(yaml.safe_dump(payload))
        dot_product_config = load_config(dot_product_path)
        assert dot_product_config.model.decoder_mode == "dot_product"
        saved = _serialized_run_config(dot_product_config)
        assert saved["model"]["decoder_mode"] == "dot_product"


def test_status_feature_toggles_control_encoder_columns_and_gradients():
    graph, agents = _instance()
    graph.nodes[3].update(type="target_unreached", rps_type=2)
    for alive, completed, position, mode in product(
            (False, True), (False, True), (False, True), ("mlp", "dot_product")):
        observation = build_observation(
            graph, agents, 2, use_agent_alive=alive, use_task_completed=completed)
        model = _graph_model(use_agent_alive=alive, use_task_completed=completed,
                             use_position=position, decoder_mode=mode)
        inputs = {}
        hooks = [
            model.agent_encoder.register_forward_pre_hook(
                lambda _module, args: inputs.update(agent=args[0])),
            model.target_encoder.register_forward_pre_hook(
                lambda _module, args: inputs.update(target=args[0])),
        ]
        try:
            logits, value = model.actor_critic(observation)
        finally:
            for hook in hooks:
                hook.remove()
        expected_agent = observation.task_agent_features[..., 0 if alive else 1:]
        expected_target = observation.task_target_features[..., 0 if completed else 1:]
        if position:
            expected_agent = torch.cat((expected_agent, observation.agent_positions), dim=-1)
            expected_target = torch.cat((expected_target, observation.target_positions), dim=-1)
        assert torch.equal(inputs["agent"], expected_agent)
        assert torch.equal(inputs["target"], expected_target)
        assert inputs["agent"].shape[-1] == 2 + int(alive) + 2 * int(position)
        assert inputs["target"].shape[-1] == 1 + int(completed) + 2 * int(position)
        (logits[observation.feasible_action_mask].sum() + value.square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_status_toggles_mask_inactive_graph_relations_before_cpu_tensor_normalization():
    graph = _line(5)
    graph.nodes[2].update(type="target_reached", rps_type=1)
    graph.nodes[4].update(type="target_unreached", rps_type=2)
    agents = [Agent(0, capabilities={1, 2}), Agent(1, capabilities={0, 1})]
    agents[1].alive = False
    candidates = replace(load_config().candidates, staging_per_target=0,
                         include_pair_staging=False)
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        world = TensorWorld.from_networkx(graph, candidates, device="cpu")
    state = TensorEpisodeState.create(
        world, [world.node_index[0]],
        [[[False, True, True], [True, True, False]]], [[1, 2]])
    state.positions[0, 1] = world.node_index[1]
    state.alive[0, 1] = False
    state.target_live[0, 0] = False
    state.target_known[:] = True
    raw = build_observation(
        graph, agents, 2, candidate_config=candidates,
        edge_normalization=RAW_EDGE_NORMALIZATION,
        use_agent_alive=False, use_task_completed=False)
    raw_values = _valid_task_distances(raw)
    for mode in (RAW_EDGE_NORMALIZATION, PER_DECISION_EDGE_NORMALIZATION, MAX_EDGE_NORMALIZATION):
        cpu = build_observation(
            graph, agents, 2, candidate_config=candidates, edge_normalization=mode,
            use_agent_alive=False, use_task_completed=False)
        tensor = TensorObservationBuilder(
            world, 2, edge_normalization=mode,
            use_agent_alive=False, use_task_completed=False).build(state)[0]
        for observation in (cpu, tensor):
            assert observation.agent_mask.tolist() == [[True, False]]
            assert observation.target_mask.tolist() == [[False, True]]
            assert not observation.agent_target_distance_mask[0, 1].any()
            assert not observation.agent_target_distance_mask[0, :, 0].any()
            assert not observation.agent_action_distance_mask[0, 1].any()
            assert not observation.action_target_distance_mask[0, :, 0].any()
            assert not observation.serves_mask[0, :, 0].any()
            assert not observation.reveals_mask[0, :, 0].any()
            assert not observation.stages_for_mask[0, :, 0].any()
            assert not observation.feasible_action_mask[0, 1].any()
        assert [c.key for c in cpu.candidates[0]] == [c.key for c in tensor.candidates[0]]
        assert all(c.node != 2 for c in cpu.candidates[0] if c.is_target)
        for name, mask_name in (
                ("agent_target_distances", "agent_target_distance_mask"),
                ("agent_action_distances", "agent_action_distance_mask"),
                ("action_target_distances", "action_target_distance_mask")):
            mask = getattr(cpu, mask_name)
            assert torch.equal(mask, getattr(tensor, mask_name))
            assert torch.allclose(getattr(cpu, name)[mask], getattr(tensor, name)[mask], atol=1e-6)
        expected = raw_values
        if mode == PER_DECISION_EDGE_NORMALIZATION:
            expected = (expected - expected.mean()) / expected.std(unbiased=False).clamp_min(1e-6)
        elif mode == MAX_EDGE_NORMALIZATION:
            expected = expected / expected.max()
        assert torch.allclose(_valid_task_distances(cpu), expected)

        model = _graph_model(use_agent_alive=False, use_task_completed=False, use_position=True)
        logits, value = model.actor_critic(cpu)
        perturbed = copy.deepcopy(cpu)
        perturbed.task_agent_features[0, 1] = 17.0
        perturbed.task_target_features[0, 0] = 31.0
        perturbed.agent_positions[0, 1] = 11.0
        perturbed.target_positions[0, 0] = 13.0
        perturbed.agent_capability_membership[0, 1] = True
        perturbed.target_capability_beliefs[0, 0] = 27.0
        changed_logits, changed_value = model.actor_critic(perturbed)
        assert torch.allclose(logits, changed_logits, atol=1e-6)
        assert torch.allclose(value, changed_value, atol=1e-6)


def test_completed_tasks_have_no_target_observation_or_staging_action_links():
    graph = _line(7)
    for node in (2, 4, 6):
        graph.nodes[node].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    for node, visible in {0: [2, 4], 1: [2], 3: [4, 6], 5: [6]}.items():
        graph.nodes[node]["visible_nodes"] = visible
    config = replace(load_config().candidates, staging_per_target=2,
                     include_pair_staging=True, include_wait=True)
    cache = CandidateScenarioCache(graph, include_pair_staging=True)
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        world = TensorWorld.from_networkx(graph, config, device="cpu")
    state = TensorEpisodeState.create(
        world, [world.node_index[0]], [[[True, True]]], [[1, 1, 1]])
    done = world.targets.index(2)
    state.target_live[0, done] = False
    state.target_known[0, done] = True
    current = graph
    current.nodes[2].update(type="target_reached", rps_type=1)
    candidates = generate_candidates(current, config, scenario_cache=cache)
    assert any(c.is_staging and c.staging_arity == 2 for c in candidates)
    assert any(c.is_observation and c.observed_targets == {4} for c in candidates)
    for candidate in candidates:
        assert not (candidate.is_target and candidate.node == 2)
        assert 2 not in candidate.associated_targets
        assert 2 not in candidate.observed_targets
        assert 2 not in candidate.staging_targets
    builder = TensorObservationBuilder(
        world, 1, use_agent_alive=False, use_task_completed=False)
    (_target, _observation, _staging, active, associated,
     observed_links, staging_links) = builder.candidate_roles(state)
    assert not associated[..., done].any()
    assert not observed_links[..., done].any()
    assert not staging_links[..., done].any()
    assert not active[0, world.target_candidate_mask[:, done]].any()
    assert not active[0, world.candidate_staging_mask[:, done]].any()
    observation = builder.build(state)[0]
    assert not observation.target_mask[0, done]
    assert not observation.agent_target_distance_mask[..., done].any()
    assert not observation.action_target_distance_mask[..., done].any()
    for relation in (observation.serves_mask, observation.reveals_mask,
                     observation.stages_for_mask):
        assert not relation[..., done].any()


def test_status_toggles_checkpoint_roundtrip_and_old_defaults():
    from learning.policy.evaluation import load_policy

    root = Path(tempfile.mkdtemp(prefix="status-feature-checkpoints-"))
    for alive, completed in product((False, True), repeat=2):
        model = _graph_model(use_agent_alive=alive, use_task_completed=completed)
        config = replace(load_config(), model=model.config)
        for name in ("use_agent_alive", "use_task_completed"):
            for invalid in ("false", 0, None):
                try:
                    replace(model.config, **{name: invalid})
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid {name} accepted")
        run = root / f"{alive}-{completed}"
        run.mkdir()
        serialized = _serialized_run_config(config)
        assert serialized["feature_schema"]["use_agent_alive"] == alive
        assert serialized["feature_schema"]["use_task_completed"] == completed
        (run / "config.yaml").write_text(yaml.safe_dump(serialized))
        torch.save(model.state_dict(), run / "best_weights.pt")
        loaded, adapter = load_policy(run)
        assert loaded.config.use_agent_alive == adapter.use_agent_alive == alive
        assert loaded.config.use_task_completed == adapter.use_task_completed == completed
        assert loaded.agent_encoder.network[0].in_features == 2 + int(alive)
        assert loaded.target_encoder.network[0].in_features == 1 + int(completed)
    legacy = root / "legacy"
    legacy.mkdir()
    model = _graph_model()
    serialized = _serialized_run_config(replace(load_config(), model=model.config))
    for name in ("use_agent_alive", "use_task_completed"):
        serialized["model"].pop(name)
        serialized["feature_schema"].pop(name)
    (legacy / "config.yaml").write_text(yaml.safe_dump(serialized))
    torch.save(model.state_dict(), legacy / "best_weights.pt")
    loaded, adapter = load_policy(legacy)
    assert loaded.config.use_agent_alive and loaded.config.use_task_completed
    assert adapter.use_agent_alive and adapter.use_task_completed
    override = root / "override.yaml"
    config = replace(load_config(), model=replace(
        model.config, use_agent_alive=False, use_task_completed=False))
    override.write_text(yaml.safe_dump(_serialized_run_config(config)))
    try:
        evaluate_suite(legacy, config_path=override, suite="development", device="cpu")
    except ValueError as error:
        assert "feature schema is incompatible" in str(error)
    else:
        raise AssertionError("status feature override was silently accepted")


def test_status_toggles_prune_completed_targets_during_tensor_gradient_replay():
    graph = _line(5)
    for node in (2, 4):
        graph.nodes[node].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    candidates = replace(load_config().candidates, staging_per_target=0,
                         include_pair_staging=False, include_wait=False)
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        world = TensorWorld.from_networkx(graph, candidates, device="cpu")
    state = TensorEpisodeState.create(world, [world.node_index[0]], [[[False, True]]], [[1, 1]])
    state.target_known[:] = True
    model = _graph_model(num_target_types=1, use_agent_alive=False, use_task_completed=False)
    builder = TensorObservationBuilder(world, 1, use_agent_alive=False, use_task_completed=False)
    rollout = collect_tensor_episodes(model, state, builder, training=True, oracle_makespans=4.0)
    assert rollout.completed.all() and len(rollout.decision_traces) == 2
    assert rollout.decision_traces[0].observation.target_mask.tolist() == [[True, True]]
    last = rollout.decision_traces[1].observation
    assert last.target_mask.tolist() == [[False, True]]
    completed_actions = world.target_candidate_mask[:, 0]
    assert not last.action_mask[0, completed_actions].any()
    assert not last.feasible_action_mask[0, :, completed_actions].any()
    replay_tensor_gradients(model, rollout, rollout.returns, 0.01, 1, "cpu")
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_agent_positions_use_full_rectangular_map_bounds_and_handle_flat_axes():
    graph = _line(5)
    coordinates = [(-10., 20.), (0., 20.), (10., 25.), (20., 30.), (30., 40.)]
    for node, pos in enumerate(coordinates):
        graph.nodes[node]["pos"] = pos
    for node in (2, 4):
        graph.nodes[node].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    agents = [Agent(1, capabilities={1}), Agent(3, capabilities={0})]
    observation = build_observation(graph, agents, 2)
    expected = torch.tensor([[[0.25, 0.0], [0.75, 0.5]]])
    assert torch.allclose(observation.agent_positions, expected)
    expected_targets = torch.tensor([[[0.5, 0.25], [1.0, 1.0]]])
    assert torch.allclose(observation.target_positions, expected_targets)
    # Transformer's historical aspect-preserving positions remain separate.
    assert observation.agent_features[0, 1, 1] == 0.25

    transformed = graph.copy()
    for node, (x, y) in enumerate(coordinates):
        transformed.nodes[node]["pos"] = (0.01 * x + 7, 3 * y - 5)
    scaled = build_observation(transformed, agents, 2)
    assert torch.allclose(scaled.agent_positions, expected, atol=1e-6)
    assert torch.allclose(scaled.target_positions, expected_targets, atol=1e-6)
    line = build_observation(_line(5), agents[:1], 2)
    assert torch.allclose(line.agent_positions, torch.tensor([[[0.25, 0.0]]]))
    point = build_observation(_line(1), [Agent(0, capabilities={1})], 2)
    assert torch.equal(point.agent_positions, torch.zeros(1, 1, 2))
    assert point.target_positions.shape == (1, 0, 2)
    padded = batch_observations([line, observation]).to("cpu")
    assert torch.equal(padded.agent_positions[0, 1], torch.zeros(2))
    assert torch.equal(padded.agent_positions[1], expected[0])
    assert not padded.target_positions[0].any()
    assert torch.equal(padded.target_positions[1], expected_targets[0])


def test_task_graph_position_toggle_controls_agent_input_and_gradients():
    graph, agents = _instance()
    agents[0].position = 1
    observation = build_observation(graph, agents, 2)
    changed = copy.copy(observation)
    changed.agent_positions = 1.0 - observation.agent_positions
    for decoder_mode in ("mlp", "dot_product"):
        for use_position in (False, True):
            model = _graph_model(decoder_mode=decoder_mode,
                                 use_position=use_position)
            inputs = []
            hook = model.agent_encoder.register_forward_pre_hook(
                lambda _module, args: inputs.append(args[0]))
            try:
                logits = model(observation)
                changed_logits = model(changed)
            finally:
                hook.remove()
            assert inputs[0].shape[-1] == (5 if use_position else 3)
            if use_position:
                assert torch.equal(inputs[0][..., -2:], observation.agent_positions)
                assert not torch.allclose(logits, changed_logits)
                logits[observation.feasible_action_mask].sum().backward()
                gradient = model.agent_encoder.network[0].weight.grad
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient[:, -2:])
            else:
                assert torch.equal(logits, changed_logits)
            missing = copy.copy(observation)
            missing.agent_positions = None
            if use_position:
                try:
                    model(missing)
                except ValueError as error:
                    assert "agent_positions" in str(error)
                else:
                    raise AssertionError("missing positions were accepted")
            else:
                assert torch.equal(logits, model(missing))


def test_position_enabled_policy_preserves_padding_and_agent_permutations():
    graph, agents = _instance()
    agents[0].position = 1
    large = build_observation(graph, agents, 2)
    small = build_observation(graph, agents[:1], 2)
    batch = batch_observations([small, large])
    model = _graph_model(use_position=True)
    logits, values = model.actor_critic(large)
    alone_logits, alone_value = model.actor_critic(small)
    batch_logits, batch_values = model.actor_critic(batch)
    assert torch.allclose(alone_logits[0, 0], batch_logits[0, 0], atol=1e-5)
    assert torch.allclose(alone_value[0], batch_values[0], atol=1e-5)
    permuted = copy.copy(large)
    for name in ("agent_features", "agent_positions", "agent_mask",
                 "task_agent_features", "agent_capability_membership",
                 "agent_target_relations", "agent_action_relations",
                 "feasible_action_mask", "agent_target_distances",
                 "agent_action_distances", "agent_target_distance_mask",
                 "agent_action_distance_mask"):
        setattr(permuted, name, getattr(large, name)[:, [1, 0]])
    permuted_logits, permuted_values = model.actor_critic(permuted)
    assert torch.allclose(permuted_logits, logits[:, [1, 0]], atol=1e-5)
    assert torch.allclose(permuted_values, values, atol=1e-5)
    restored = build_policy(model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    assert torch.allclose(restored(large), logits, atol=1e-5)


def test_position_toggle_includes_target_coordinates_but_not_candidate_coordinates():
    graph, agents = _instance()
    graph.nodes[3].update(type="target_unreached", rps_type=2)
    observation = build_observation(graph, agents, 2)
    changed = copy.copy(observation)
    changed.target_positions = 1.0 - observation.target_positions
    for mode in ("mlp", "dot_product"):
        for enabled in (False, True):
            model = _graph_model(decoder_mode=mode, use_position=enabled)
            inputs = {"target": [], "action": []}
            hooks = [
                model.target_encoder.register_forward_pre_hook(
                    lambda _module, args: inputs["target"].append(args[0])),
                model.action_encoder.register_forward_pre_hook(
                    lambda _module, args: inputs["action"].append(args[0])),
            ]
            try:
                logits = model(observation)
                changed_logits = model(changed)
            finally:
                for hook in hooks:
                    hook.remove()
            assert inputs["target"][0].shape[-1] == (4 if enabled else 2)
            assert torch.equal(inputs["action"][0], observation.task_action_features)
            assert inputs["action"][0].shape[-1] == 4
            if enabled:
                assert torch.equal(inputs["target"][0][..., -2:], observation.target_positions)
                assert not torch.allclose(logits, changed_logits)
                logits[observation.feasible_action_mask].sum().backward()
                gradients = model.target_encoder.network[0].weight.grad
                assert torch.isfinite(gradients).all()
                assert torch.count_nonzero(gradients[:, -2:])
            else:
                assert torch.equal(logits, changed_logits)
            missing = copy.copy(observation)
            missing.target_positions = None
            if enabled:
                try:
                    model(missing)
                except ValueError as error:
                    assert "target_positions" in str(error)
                else:
                    raise AssertionError("missing target coordinates were accepted")
            else:
                assert torch.equal(logits, model(missing))

    # Target order changes only the corresponding input rows, not the policy.
    model = _graph_model(use_position=True)
    permuted = copy.copy(observation)
    for name in ("target_features", "target_positions", "target_mask",
                 "task_target_features", "target_capability_beliefs"):
        setattr(permuted, name, getattr(observation, name)[:, [1, 0]])
    for name in ("agent_target_relations", "agent_target_distances",
                 "agent_target_distance_mask", "action_target_relations",
                 "action_target_distances", "action_target_distance_mask",
                 "serves_mask", "reveals_mask", "stages_for_mask"):
        setattr(permuted, name, getattr(observation, name)[:, :, [1, 0]])
    assert torch.allclose(model(observation), model(permuted), atol=1e-5)


def test_position_enabled_checkpoints_preserve_agent_and_target_inputs():
    from learning.policy.evaluation import load_policy

    model = _graph_model(use_position=True)
    graph, agents = _instance()
    observation = build_observation(graph, agents, 2)
    payload = _serialized_run_config(replace(load_config(), model=model.config))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "config.yaml").write_text(yaml.safe_dump(payload))
        torch.save(model.state_dict(), directory / "best_weights.pt")
        loaded, _adapter = load_policy(directory, device="cpu")
        assert loaded.config.use_position
        assert loaded.agent_encoder.network[0].in_features == 5
        assert loaded.target_encoder.network[0].in_features == 4
        assert loaded.action_encoder.network[0].in_features == 4
        assert torch.allclose(loaded(observation), model(observation), atol=1e-5)


def test_task_graph_dot_product_logits_preserve_magnitude_and_scale_by_dimension():
    config = replace(load_config().model, model_dim=4, num_heads=1,
                     decoder_mode="dot_product", use_critic=False)
    model = build_policy(config)
    agents = torch.tensor([[[2., 0., 0., 0.], [0., 0., 0., 0.]]],
                          requires_grad=True)
    actions = torch.tensor([[[3., 0., 0., 0.], [-4., 0., 0., 0.],
                             [0., 1., 0., 0.], [0., 0., 0., 0.]]],
                           requires_grad=True)
    feasible = torch.tensor([[[True, True, True, True],
                              [True, False, True, True]]])
    observation = SimpleNamespace(feasible_action_mask=feasible)
    with patch.object(model, "encode", return_value=(
            agents, None, actions, None, None)):
        logits, value = model.actor_critic(observation)
    expected = torch.tensor([[[3., -4., 0., 0.],
                              [0., -torch.inf, 0., 0.]]])
    assert torch.allclose(logits, expected)
    with patch.object(model, "encode", return_value=(
            3 * agents, None, actions, None, None)):
        assert torch.allclose(model(observation), 3 * expected)
    assert value is None
    assert model.actor is None
    logits[feasible].sum().backward()
    assert torch.isfinite(agents.grad).all()
    assert torch.isfinite(actions.grad).all()


def test_task_graph_dot_product_decodes_batches_and_trains_graph_embeddings():
    graph, agents = _instance()
    small = build_observation(graph, agents[:1], 2)
    large = build_observation(graph, agents, 2)
    observation = batch_observations([small, large])
    model = _graph_model(decoder_mode="dot_product")
    alone_logits, alone_value = model.actor_critic(small)
    logits, values = model.actor_critic(observation)
    actions = small.action_features.shape[1]
    assert torch.allclose(alone_logits[0, 0], logits[0, 0, :actions], atol=1e-5)
    assert torch.allclose(alone_value[0], values[0], atol=1e-5)
    assert torch.isneginf(logits[~observation.feasible_action_mask]).all()
    decoded, decoded_values = model.decode_with_value(observation, training=False)
    assert torch.allclose(decoded_values, values)
    for batch, assignments in enumerate(decoded.assignments):
        assert len({agent for agent, _action in assignments}) == len(assignments)
        counts = {}
        for agent, action in assignments:
            assert observation.feasible_action_mask[batch, agent, action]
            group = observation.candidate_physical_group[batch, action].item()
            counts[group] = counts.get(group, 0) + 1
            assert counts[group] <= observation.physical_group_capacity[batch, group]
    loss = -decoded.log_probabilities.mean() + values.square().mean()
    loss.backward()
    for module in (model.agent_encoder, model.action_encoder,
                   model.distance_encoder, model.graph_blocks, model.critic):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
    restored = build_policy(model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    assert torch.allclose(restored(observation), logits, atol=1e-5)


def test_task_graph_schema_uses_beliefs_semantics_and_effective_distances():
    graph = _line()
    graph.nodes[4].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    graph.nodes[2]["visible_edges"] = [(3, 4)]
    agent = Agent(0, capabilities={0, 1})
    observation = build_observation(
        graph, [agent], 2,
        transit=[(0, 1, 0.0, 1.0)], clock=0.25,
        replan_transit=True, edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    raw_observation = build_observation(
        graph, [agent], 2,
        transit=[(0, 1, 0.0, 1.0)], clock=0.25,
        replan_transit=True, edge_normalization=RAW_EDGE_NORMALIZATION)

    assert observation.task_agent_features.shape[-1] == 3
    assert observation.task_target_features.shape[-1] == 2
    assert observation.task_action_features.shape[-1] == 4
    # Unknown types use a uniform planner belief, never truth or all-zero.
    assert torch.allclose(
        observation.target_capability_beliefs[0, 0],
        torch.tensor([0.5, 0.5]))
    assert observation.agent_capability_membership[0, 0].tolist() == [True, False]
    assert observation.capability_mask[0].tolist() == [True, True]

    target_action = next(
        i for i, item in enumerate(observation.candidates[0])
        if item.is_target)
    wait_action = next(
        i for i, item in enumerate(observation.candidates[0])
        if item.is_wait)
    observation_action = next(
        i for i, item in enumerate(observation.candidates[0])
        if item.is_observation)
    target_index = 0
    raw_values = _valid_task_distances(raw_observation)
    mean = raw_values.mean()
    std = raw_values.std(correction=0)
    expected_remaining = (torch.tensor(0.75) - mean) / std.clamp_min(1.0e-6)
    assert torch.allclose(
        observation.task_agent_features[0, 0, 1], expected_remaining)
    assert torch.allclose(
        observation.agent_features[0, 0, 8], expected_remaining)
    raw_target_distance = raw_observation.agent_action_distances[
        0, 0, target_action, 0]
    expected_target_distance = (
        raw_target_distance - mean) / std.clamp_min(1.0e-6)
    assert torch.allclose(
        observation.agent_action_distances[0, 0, target_action, 0],
        expected_target_distance)
    assert torch.allclose(
        observation.agent_action_relations[0, 0, target_action, 0],
        expected_target_distance)
    normalized_values = _valid_task_distances(observation)
    assert abs(float(normalized_values.mean())) < 1e-6
    assert abs(float(normalized_values.std(correction=0)) - 1.0) < 1e-6
    assert observation.serves_mask[0, target_action, target_index]
    assert observation.reveals_mask[0, observation_action, target_index]
    assert not observation.action_target_distance_mask[
        0, wait_action, target_index]


def test_belief_weighted_capability_messages_are_exact_and_unnormalized():
    relation = BeliefWeightedCapabilityRelations(2)
    with torch.no_grad():
        relation.capability_to_target.weight.copy_(torch.eye(2))
        relation.target_to_capability.weight.copy_(torch.eye(2))
    capabilities = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])
    targets = torch.tensor([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
    beliefs = torch.tensor([[[1.0, 0.0, 0.0],
                             [1 / 3, 1 / 3, 1 / 3],
                             [0.25, 0.75, 0.0]]])
    target_messages, capability_messages = relation(
        targets, capabilities, beliefs,
        torch.ones((1, 3), dtype=torch.bool),
        torch.ones((1, 3), dtype=torch.bool))
    assert torch.allclose(target_messages, torch.tensor([[
        [1.0, 0.0], [4 / 3, 2.0], [0.25, 1.5],
    ]]))
    # The reverse direction is a raw sum. In particular, its first row is not
    # divided by the aggregate belief mass 1 + 1/3 + 1/4.
    assert torch.allclose(capability_messages, torch.tensor([[
        [29 / 6, 13 / 4], [35 / 6, 19 / 4], [4 / 3, 1.0],
    ]]))

    masked_targets, masked_capabilities = relation(
        targets, capabilities, torch.ones_like(beliefs),
        torch.tensor([[True, False, False]]),
        torch.tensor([[True, False, False]]))
    assert torch.equal(masked_targets, torch.tensor([[[1.0, 0.0],
                                                       [0.0, 0.0],
                                                       [0.0, 0.0]]]))
    assert torch.equal(masked_capabilities, torch.tensor([[[2.0, 1.0],
                                                            [0.0, 0.0],
                                                            [0.0, 0.0]]]))
    empty_targets, empty_capabilities = relation(
        targets, capabilities, beliefs,
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool))
    assert not empty_targets.any() and not empty_capabilities.any()
    assert torch.isfinite(empty_targets).all()
    assert torch.isfinite(empty_capabilities).all()


def test_dead_agents_and_serviced_targets_send_no_capability_messages():
    graph = _line(3)
    graph.nodes[2].update(type="target_reached", rps_type=1)
    dead_agent = Agent(0, capabilities={1, 2})
    dead_agent.alive = False
    observation = build_observation(graph, [dead_agent], 2)
    model = _graph_model(2)
    block = model.graph_blocks[0]
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    hooks = [
        block.capability_relations["agent_to_capability"].register_forward_hook(
            capture("agent_to_capability")),
        block.capability_relations["capability_to_agent"].register_forward_hook(
            capture("capability_to_agent")),
        block.belief_relations.register_forward_hook(capture("belief")),
    ]
    try:
        encoded = model.encode(observation)
    finally:
        for hook in hooks:
            hook.remove()
    assert not captured["agent_to_capability"].any()
    assert not captured["capability_to_agent"].any()
    assert not captured["belief"][0].any()
    assert not captured["belief"][1].any()
    assert all(torch.isfinite(value).all() for value in encoded)


def test_capability_relabeling_preserves_task_graph_logits():
    graph = _line(5)
    graph.nodes[3].update(type="target_unreached", rps_type=1)
    graph.nodes[4].update(type="target_unreached", rps_type=3)
    agents = [Agent(0, capabilities={0, 1, 2}),
              Agent(0, capabilities={3})]
    original = build_observation(graph, agents, 3)

    relabeled_graph = graph.copy()
    relabeled_graph.nodes[3]["rps_type"] = 3
    relabeled_graph.nodes[4]["rps_type"] = 1
    relabeled_agents = [Agent(0, capabilities={0, 3, 2}),
                        Agent(0, capabilities={1})]
    relabeled = build_observation(relabeled_graph, relabeled_agents, 3)
    model = _graph_model(3)
    with torch.no_grad():
        original_logits = model(original)
        relabeled_logits = model(relabeled)
    assert torch.equal(
        original.feasible_action_mask, relabeled.feasible_action_mask)
    assert torch.allclose(
        original_logits, relabeled_logits, atol=1.0e-6, equal_nan=True)


def test_one_task_graph_model_handles_three_and_thirty_two_capabilities():
    graph_three = _line(3)
    graph_three.nodes[2].update(type="target_unreached", rps_type=3)
    observation_three = build_observation(
        graph_three, [Agent(0, capabilities={0, 3})], 3)
    graph_thirty_two = _line(3)
    graph_thirty_two.nodes[2].update(
        type="target_unreached", rps_type=32)
    observation_thirty_two = build_observation(
        graph_thirty_two, [Agent(0, capabilities={0, 32})], 32)

    model = _graph_model(3)
    parameter_shapes = {
        name: parameter.shape for name, parameter in model.named_parameters()}
    with torch.no_grad():
        logits_three = model(observation_three)
        logits_thirty_two = model(observation_thirty_two)
        batched_logits = model(batch_observations(
            [observation_three, observation_thirty_two]))
    assert torch.isfinite(
        logits_three[observation_three.feasible_action_mask]).all()
    assert torch.isfinite(
        logits_thirty_two[observation_thirty_two.feasible_action_mask]).all()
    assert torch.allclose(
        logits_three[0], batched_logits[0, :, :logits_three.shape[2]],
        atol=1.0e-5, equal_nan=True)
    assert torch.allclose(
        logits_thirty_two[0],
        batched_logits[1, :, :logits_thirty_two.shape[2]],
        atol=1.0e-5, equal_nan=True)
    model_thirty_two = _graph_model(32)
    assert parameter_shapes == {
        name: parameter.shape
        for name, parameter in model_thirty_two.named_parameters()}
    if torch.cuda.is_available():
        model.cuda()
        cuda_observation = observation_thirty_two.to("cuda")
        with torch.no_grad():
            cuda_logits = model(cuda_observation)
        assert torch.allclose(
            logits_thirty_two,
            cuda_logits.cpu(), atol=1.0e-5, equal_nan=True)


def test_target_revelation_updates_only_visible_belief_rows():
    graph = _line(4)
    graph.nodes[2].update(
        type="target_unreached", rps_type=UNKNOWN_TYPE)
    graph.nodes[3].update(
        type="target_unreached", rps_type=UNKNOWN_TYPE)
    agents = [Agent(0, capabilities={0, 1, 2, 3})]
    hidden = build_observation(graph, agents, 3)
    assert torch.allclose(
        hidden.target_capability_beliefs,
        torch.full((1, 2, 3), 1 / 3))

    graph.nodes[2]["rps_type"] = 2
    revealed = build_observation(graph, agents, 3)
    target_row = revealed.targets[0].index(2)
    hidden_row = revealed.targets[0].index(3)
    assert torch.equal(
        revealed.target_capability_beliefs[0, target_row],
        torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(
        revealed.target_capability_beliefs[0, hidden_row],
        torch.full((3,), 1 / 3))


def test_task_graph_distance_normalization_is_per_observation_and_shared():
    first_graph = _line(5)
    first_graph.nodes[4].update(type="target_unreached", rps_type=1)
    second_graph = first_graph.copy()
    second_graph.edges[0, 1]["distance"] = 7.0
    first = build_observation(
        first_graph, [Agent(0, capabilities={1})], 2,
        edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    second = build_observation(
        second_graph,
        [Agent(0, capabilities={1}), Agent(1, capabilities={2})], 2,
        edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    batched = batch_observations([first, second])
    for row in range(2):
        values = _valid_task_distances(batched, row)
        assert values.numel() > 1
        assert abs(float(values.mean())) < 1.0e-6
        assert abs(float(values.std(correction=0)) - 1.0) < 1.0e-6


def test_task_graph_normalization_has_zero_variance_and_no_edge_fallbacks():
    singleton = nx.DiGraph()
    singleton.add_node(
        "t", pos=(0, 0), height=0.0,
        type="target_unreached", rps_type=1, visible_edges=[])
    zero_variance = build_observation(
        singleton, [Agent("t", capabilities={1})], 1,
        edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    values = _valid_task_distances(zero_variance)
    assert values.numel() == 3
    assert torch.equal(values, torch.zeros_like(values))

    targetless = _line(2)
    no_edges = build_observation(
        targetless, [Agent(0, capabilities={1})], 1,
        transit=[(0, 1, 0.0, 1.0)], clock=0.5,
        replan_transit=True, edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    assert _valid_task_distances(no_edges).numel() == 0
    assert not no_edges.agent_action_distance_mask.any()
    assert not no_edges.agent_action_distances.any()
    # With no edge statistics there is no scale for a moving ETA either.
    assert no_edges.task_agent_features[0, 0, 1] == 0


def test_no_edge_normalization_retains_raw_travel_times_in_the_current_schema():
    config = load_config()
    config = replace(config, model=replace(
        config.model, edge_normalization=RAW_EDGE_NORMALIZATION, use_position=False,
        use_agent_alive=True, use_task_completed=True))
    assert config.model.feature_schema_version == CURRENT_FEATURE_SCHEMA_VERSION
    assert config.model.edge_normalization == RAW_EDGE_NORMALIZATION
    graph = _line(6)
    graph.nodes[5].update(type="target_unreached", rps_type=1)
    graph.edges[3, 4]["distance"] = 2.0
    graph.edges[4, 5]["distance"] = 3.0
    observation = build_observation(
        graph, [Agent(3, capabilities={1})], 1,
        transit=[(3, 4, 0.0, 2.0)], clock=0.5, replan_transit=True,
        edge_normalization=RAW_EDGE_NORMALIZATION)
    target_action = next(i for i, candidate in enumerate(observation.candidates[0])
                         if candidate.is_target)
    assert observation.task_agent_features[0, 0, 1] == 1.5
    assert observation.agent_target_distances[0, 0, 0, 0] == 4.5
    assert observation.agent_action_distances[0, 0, target_action, 0] == 4.5
    assert observation.action_target_distances[0, target_action, 0, 0] == 0.0
    with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
        world = TensorWorld.from_networkx(graph, config.candidates, device="cpu")
    state = TensorEpisodeState.create(world, [world.node_index[3]],
                                      [[[False, True]]], [[1]])
    state.target_known[:] = True
    state.dispatch_next_hops(torch.tensor([[world.node_index[4]]]),
                              torch.tensor([[True]]))
    state.clock[:] = 0.5
    tensor_observation = TensorObservationBuilder(
        world, 1, edge_normalization=RAW_EDGE_NORMALIZATION).build(state)[0]
    for name in ("task_agent_features", "agent_target_distances",
                 "agent_action_distances", "action_target_distances"):
        assert torch.allclose(getattr(tensor_observation, name), getattr(observation, name))
    scaled = graph.copy()
    for _u, _v, attributes in scaled.edges(data=True):
        attributes["distance"] *= 10.0
    enlarged = build_observation(
        scaled, [Agent(3, capabilities={1})], 1,
        transit=[(3, 4, 0.0, 20.0)], clock=5.0, replan_transit=True,
        edge_normalization=RAW_EDGE_NORMALIZATION)
    for name in ("agent_target_distances", "agent_action_distances",
                 "action_target_distances"):
        assert torch.allclose(getattr(enlarged, name), 10.0 * getattr(observation, name))
    assert enlarged.task_agent_features[0, 0, 1] == 15.0
    model = build_policy(config.model)
    assert model.agent_encoder.network[0].in_features == 3
    assert model.target_encoder.network[0].in_features == 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "raw.yaml"
        serialized = _serialized_run_config(config)
        path.write_text(yaml.safe_dump(serialized))
        assert load_config(path) == config
        assert serialized["feature_schema"]["edge_normalization"] == "none"
        assert serialized["feature_schema"]["version"] == CURRENT_FEATURE_SCHEMA_VERSION


def test_max_normalization_preserves_zero_and_scales_all_edge_families_per_episode():
    graph = _line(6)
    graph.nodes[5].update(type="target_unreached", rps_type=1)
    agents = [Agent(3, capabilities={1}), Agent(1, capabilities={1})]
    raw = build_observation(
        graph, agents, 1, transit=[(3, 4, 0., 1.), None],
        replan_transit=True, edge_normalization=RAW_EDGE_NORMALIZATION)
    expected_max = _valid_task_distances(raw).max()
    assert expected_max == 4.0
    scaled = copy.deepcopy(raw)
    names = ("agent_target_distances", "agent_action_distances",
             "action_target_distances")
    masks = ("agent_target_distance_mask", "agent_action_distance_mask",
             "action_target_distance_mask")
    for name in names:
        getattr(scaled, name).mul_(1.0e-12)
    batch = batch_observations([raw, scaled])
    # Invalid slots, including wait edges, must never set the scale.
    for name, mask in zip(names, masks):
        getattr(batch, name).masked_fill_(~getattr(batch, mask)[..., None], torch.inf)
    remaining = torch.tensor([[1., 0.], [1.0e-12, 0.]])
    normalize_task_graph_travel_times(batch, remaining, MAX_EDGE_NORMALIZATION, 1e-6)
    for name, mask in zip(names, masks):
        actual = getattr(batch, name)
        valid = getattr(batch, mask)
        assert torch.allclose(actual[0], actual[1], atol=1e-6)
        assert torch.allclose(actual[0][valid[0]],
                              getattr(raw, name)[0][valid[0]] / expected_max)
        assert not actual[~valid].any()
        assert torch.isfinite(actual).all()
    assert _valid_task_distances(batch).max() == 1.0
    assert torch.allclose(batch.task_agent_features[..., 1],
                          torch.tensor([[0.25, 0.], [0.25, 0.]]))
    assert torch.equal(batch.agent_features[..., 8], batch.task_agent_features[..., 1])
    target_action = next(i for i, c in enumerate(raw.candidates[0]) if c.is_target)
    assert batch.action_target_distances[0, target_action, 0, 0] == 0


def test_max_normalization_handles_zero_empty_and_config_checkpoint_metadata():
    config = load_config()
    max_config = replace(config, model=replace(
        config.model, edge_normalization=MAX_EDGE_NORMALIZATION))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "max.yaml"
        path.write_text(yaml.safe_dump(_serialized_run_config(max_config)))
        assert load_config(path).model.edge_normalization == MAX_EDGE_NORMALIZATION
        metadata = _serialized_run_config(max_config)["feature_schema"]
        assert metadata["edge_normalization"] == MAX_EDGE_NORMALIZATION
    try:
        replace(max_config.model, feature_schema_version=RAW_DISTANCE_FEATURE_SCHEMA_VERSION)
    except ValueError:
        pass
    else:
        raise AssertionError("schema 1 accepted max normalization")
    singleton = nx.DiGraph()
    singleton.add_node("t", pos=(0, 0), type="target_unreached", rps_type=1)
    zero = build_observation(singleton, [Agent("t", capabilities={1})], 1,
                             edge_normalization=MAX_EDGE_NORMALIZATION)
    assert not _valid_task_distances(zero).any()
    empty = build_observation(_line(2), [Agent(0, capabilities={1})], 1,
                              edge_normalization=MAX_EDGE_NORMALIZATION,
                              transit=[(0, 1, 0., 1.)], replan_transit=True)
    assert _valid_task_distances(empty).numel() == 0
    assert empty.task_agent_features[0, 0, 1] == 0


def test_task_graph_normalization_excludes_unreachable_wait_and_padding():
    graph = nx.DiGraph()
    for node, kind in (("s", "source"), ("r", "target_unreached"),
                       ("u", "target_unreached")):
        graph.add_node(
            node, pos=(len(graph), 0), height=0.0, type=kind,
            rps_type=1 if kind == "target_unreached" else UNKNOWN_TYPE,
            visible_edges=[])
    graph.add_edge("s", "r", distance=2.0, observed_edge=False)
    graph.add_edge("r", "s", distance=2.0, observed_edge=False)
    observation = build_observation(
        graph, [Agent("s", capabilities={1})], 1)
    unreachable_target = observation.targets[0].index("u")
    unreachable_action = next(
        index for index, candidate in enumerate(observation.candidates[0])
        if candidate.is_target and candidate.node == "u")
    wait_action = next(
        index for index, candidate in enumerate(observation.candidates[0])
        if candidate.is_wait)
    assert not observation.agent_target_distance_mask[
        0, 0, unreachable_target]
    assert observation.agent_target_distances[
        0, 0, unreachable_target, 0] == 0
    assert not observation.agent_action_distance_mask[
        0, 0, unreachable_action]
    assert observation.agent_action_distances[
        0, 0, unreachable_action, 0] == 0
    assert not observation.agent_action_distance_mask[0, 0, wait_action]
    assert observation.agent_action_distances[0, 0, wait_action, 0] == 0

    targetless = build_observation(
        _line(2), [Agent(0, capabilities={1})], 1)
    padded = batch_observations([targetless, observation])
    assert not padded.target_mask[0].any()
    assert not padded.agent_target_distance_mask[0].any()
    assert not padded.action_target_distance_mask[0].any()
    assert not padded.agent_target_distances[0].any()
    assert not padded.action_target_distances[0].any()
    original_actions = targetless.action_mask.shape[1]
    assert not padded.action_mask[0, original_actions:].any()
    assert not padded.agent_action_distance_mask[
        0, :, original_actions:].any()
    assert not padded.agent_action_distances[
        0, :, original_actions:].any()


def test_uniform_travel_time_scaling_preserves_features_and_policy_logits():
    graph = _line(5)
    graph.nodes[4].update(type="target_unreached", rps_type=1)
    scaled = graph.copy()
    for source, target in scaled.edges:
        scaled.edges[source, target]["distance"] *= 80.0
    original_distances = {
        edge: graph.edges[edge]["distance"] for edge in graph.edges}
    agent = Agent(0, capabilities={1})
    scaled_agent = Agent(0, capabilities={1})
    base = build_observation(
        graph, [agent], 2,
        transit=[(0, 1, 0.0, 1.0)], clock=0.25,
        replan_transit=True, edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    enlarged = build_observation(
        scaled, [scaled_agent], 2,
        transit=[(0, 1, 0.0, 80.0)], clock=20.0,
        replan_transit=True, edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    for name in (
            "agent_target_distances", "agent_action_distances",
            "action_target_distances", "task_agent_features",
            "agent_target_relations", "agent_action_relations",
            "action_target_relations"):
        assert torch.allclose(
            getattr(base, name), getattr(enlarged, name), atol=1.0e-6)
    for name in (
            "agent_target_distance_mask", "agent_action_distance_mask",
            "action_target_distance_mask", "feasible_action_mask"):
        assert torch.equal(getattr(base, name), getattr(enlarged, name))
    model = _graph_model()
    assert torch.allclose(model(base), model(enlarged), atol=1.0e-6)
    assert {
        edge: graph.edges[edge]["distance"] for edge in graph.edges
    } == original_distances


def test_unknown_target_action_toggle_and_agent_roles_use_visible_beliefs():
    graph = _line(3)
    graph.nodes[2].update(
        type="target_unreached", rps_type=UNKNOWN_TYPE)
    base = replace(
        load_config().candidates, staging_per_target=0,
        include_pair_staging=False)
    agents = [
        Agent(0, capabilities={0}),
        Agent(0, capabilities={1}),
        Agent(0, capabilities={0, 2}),
    ]

    def target_feasibility(allow_unknown, belief):
        graph.nodes[2]["rps_type"] = belief
        config = replace(
            base, allow_unknown_target_actions=allow_unknown)
        candidates = generate_candidates(graph, config)
        observation = build_observation(
            graph, agents, 2, candidates=candidates,
            candidate_config=config)
        target_action = next(
            index for index, candidate in enumerate(candidates)
            if candidate.is_target)
        return observation.feasible_action_mask[0, :, target_action]

    assert target_feasibility(True, UNKNOWN_TYPE).tolist() == [False, True, True]
    assert target_feasibility(False, UNKNOWN_TYPE).tolist() == [False, False, False]
    assert target_feasibility(True, 1).tolist() == [False, True, False]
    assert target_feasibility(False, 1).tolist() == [False, True, False]
    assert target_feasibility(True, 2).tolist() == [False, False, True]


def test_cpu_cuda_observation_parity_for_normalization_and_unknown_actions():
    if not torch.cuda.is_available():
        return
    graph = _line(3)
    for node, pos in enumerate(((-2., 10.), (3., 11.), (8., 14.))):
        graph.nodes[node]["pos"] = pos
    graph.nodes[2].update(
        type="target_unreached", rps_type=UNKNOWN_TYPE)
    candidate_config = replace(
        load_config().candidates, staging_per_target=0,
        include_pair_staging=False, allow_unknown_target_actions=True)
    cpu = build_observation(
        graph, [Agent(1, capabilities={1, 3})], 3,
        candidate_config=candidate_config)
    world = TensorWorld.from_networkx(
        graph, candidate_config, device="cuda")
    state = TensorEpisodeState.create(
        world, [world.node_index[1]],
        [[[False, True, False, True]]], [[2]])
    config = load_config().model
    gpu = TensorObservationBuilder(
        world, 3, edge_normalization=config.edge_normalization,
        edge_normalization_epsilon=(
            config.edge_normalization_epsilon)).build(state)[0]
    assert [candidate.key for candidate in cpu.candidates[0]] == [
        candidate.key for candidate in gpu.candidates[0]]
    for name in (
            "agent_target_distances", "agent_action_distances",
            "action_target_distances", "task_agent_features",
            "task_target_features", "agent_capability_membership",
            "target_capability_beliefs", "capability_mask", "agent_positions",
            "target_positions"):
        assert torch.allclose(
            getattr(cpu, name), getattr(gpu, name).cpu(), atol=1.0e-5)
    for name in (
            "agent_target_distance_mask", "agent_action_distance_mask",
            "action_target_distance_mask", "feasible_action_mask"):
        assert torch.equal(getattr(cpu, name), getattr(gpu, name).cpu())


def test_checkpoint_feature_schema_requires_explicit_compatibility_override():
    from learning.policy.evaluation import load_policy

    base = load_config()
    legacy_model_config = replace(
        base.model,
        feature_schema_version=RAW_DISTANCE_FEATURE_SCHEMA_VERSION,
        edge_normalization=RAW_EDGE_NORMALIZATION)
    legacy_config = replace(base, model=legacy_model_config)
    serialized = _serialized_run_config(legacy_config)
    assert serialized["feature_schema"]["version"] == (
        RAW_DISTANCE_FEATURE_SCHEMA_VERSION)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "config.yaml").write_text(
            yaml.safe_dump(serialized, sort_keys=False), encoding="utf-8")
        torch.save(
            build_policy(legacy_model_config).state_dict(),
            directory / "latest_weights.pt")
        try:
            load_policy(directory, device="cpu")
        except ValueError as error:
            assert "older raw-distance feature schema" in str(error)
        else:
            raise AssertionError("a legacy feature schema was accepted silently")
        model, adapter = load_policy(
            directory, device="cpu",
            allow_feature_schema_mismatch=True)
        assert model.config.feature_schema_version == (
            RAW_DISTANCE_FEATURE_SCHEMA_VERSION)
        assert adapter.edge_normalization == RAW_EDGE_NORMALIZATION


def test_task_graph_policy_is_permutation_equivariant_and_has_finite_critic():
    graph, agents = _instance()
    graph.nodes[3].update(type="target_unreached", rps_type=2)
    observation = build_observation(graph, agents, 2)
    model = _graph_model()
    base_logits, base_value = model.actor_critic(observation)
    assert torch.isfinite(base_logits[observation.feasible_action_mask]).all()
    assert torch.isfinite(base_value).all()

    swapped_agents = copy.copy(observation)
    order_a = torch.tensor([1, 0])
    for name in ("agent_features", "agent_mask", "task_agent_features",
                 "agent_capability_membership"):
        setattr(swapped_agents, name, getattr(observation, name)[:, order_a])
    for name in ("agent_target_relations", "agent_action_relations",
                 "feasible_action_mask", "agent_target_distances",
                 "agent_action_distances", "agent_target_distance_mask",
                 "agent_action_distance_mask"):
        setattr(swapped_agents, name, getattr(observation, name)[:, order_a])
    swapped_logits, swapped_value = model.actor_critic(swapped_agents)
    assert torch.allclose(swapped_logits, base_logits[:, order_a], atol=1e-5,
                          equal_nan=True)
    assert torch.allclose(swapped_value, base_value, atol=1e-5)

    swapped_actions = copy.copy(observation)
    order_c = torch.arange(observation.action_features.shape[1] - 1, -1, -1)
    for name in ("action_features", "action_mask", "task_action_features"):
        setattr(swapped_actions, name, getattr(observation, name)[:, order_c])
    for name in ("action_target_relations", "action_target_distances",
                 "action_target_distance_mask", "serves_mask", "reveals_mask",
                 "stages_for_mask"):
        setattr(swapped_actions, name, getattr(observation, name)[:, order_c])
    for name in ("agent_action_relations", "feasible_action_mask",
                 "agent_action_distances", "agent_action_distance_mask"):
        setattr(swapped_actions, name, getattr(observation, name)[:, :, order_c])
    swapped_logits, swapped_value = model.actor_critic(swapped_actions)
    assert torch.allclose(swapped_logits, base_logits[:, :, order_c], atol=1e-5,
                          equal_nan=True)
    assert torch.allclose(swapped_value, base_value, atol=1e-5)

    swapped_targets = copy.copy(observation)
    order_t = torch.tensor([1, 0])
    for name in ("target_features", "target_mask", "task_target_features",
                 "target_capability_beliefs"):
        setattr(swapped_targets, name, getattr(observation, name)[:, order_t])
    for name in ("agent_target_relations", "agent_target_distances",
                 "agent_target_distance_mask"):
        setattr(swapped_targets, name,
                getattr(observation, name)[:, :, order_t])
    for name in ("action_target_relations", "action_target_distances",
                 "action_target_distance_mask", "serves_mask", "reveals_mask",
                 "stages_for_mask"):
        setattr(swapped_targets, name,
                getattr(observation, name)[:, :, order_t])
    swapped_logits, swapped_value = model.actor_critic(swapped_targets)
    assert torch.allclose(swapped_logits, base_logits, atol=1e-5,
                          equal_nan=True)
    assert torch.allclose(swapped_value, base_value, atol=1e-5)

    loss = (base_logits[observation.feasible_action_mask].mean()
            + base_value.mean())
    model.train()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()
               if parameter.grad is not None)


def test_task_graph_policy_is_padding_invariant():
    graph, agents = _instance()
    small = build_observation(graph, agents[:1], 2)
    large = build_observation(graph, agents, 2)
    batch = batch_observations([small, large])
    model = _graph_model()
    alone_logits, alone_value = model.actor_critic(small)
    batch_logits, batch_values = model.actor_critic(batch)
    actions = small.action_features.shape[1]
    assert torch.allclose(
        alone_logits[0, 0, :actions], batch_logits[0, 0, :actions], atol=1e-5)
    assert torch.allclose(alone_value[0], batch_values[0], atol=1e-5)


def test_task_graph_tensor_replay_trains_actor_and_critic():
    graph, agents = _instance()
    observation = build_observation(graph, agents, 2)
    unlimited = torch.iinfo(torch.long).max
    observation.action_capacities = torch.tensor([[
        unlimited if item.capacity is None else item.capacity
        for item in observation.candidates[0]
    ]], dtype=torch.long)
    model = _graph_model()
    model.train()
    with torch.no_grad():
        decoded = model.decode(observation, training=False)
    rollout = SimpleNamespace(decision_traces=[DecisionTrace(
        observation, decoded.selected_pair_indices)])
    outputs = replay_tensor_gradients(
        model, rollout, torch.tensor([-2.0]),
        entropy_coefficient=0.01, update_size=1, device="cpu",
        critic_coefficient=0.5)
    losses, counts, critic_losses, entropies, values = outputs
    assert counts.item() == 1
    assert torch.isfinite(losses).all()
    assert torch.isfinite(critic_losses).all()
    assert torch.isfinite(entropies).all()
    assert torch.isfinite(values).all()
    assert any(parameter.grad is not None
               for parameter in model.critic.parameters())


def test_task_graph_without_critic_replays_actor_only():
    graph, agents = _instance()
    observation = build_observation(graph, agents, 2)
    unlimited = torch.iinfo(torch.long).max
    observation.action_capacities = torch.tensor([[
        unlimited if item.capacity is None else item.capacity
        for item in observation.candidates[0]
    ]], dtype=torch.long)
    model = _graph_model(use_critic=False)
    model.train()
    logits, value = model.actor_critic(observation)
    assert value is None
    with torch.no_grad():
        decoded = model.decoder(
            logits, observation.feasible_action_mask,
            observation.action_capacities, training=False)
    rollout = SimpleNamespace(decision_traces=[DecisionTrace(
        observation, decoded.selected_pair_indices)])
    outputs = replay_tensor_gradients(
        model, rollout, torch.tensor([1.0]),
        entropy_coefficient=0.01, update_size=1, device="cpu",
        critic_coefficient=0.5)
    _losses, _counts, critic_losses, _entropies, values = outputs
    assert torch.equal(critic_losses, torch.zeros_like(critic_losses))
    assert torch.equal(values, torch.zeros_like(values))
    assert all(parameter.grad is None
               for parameter in model.critic.parameters())
    assert any(parameter.grad is not None
               for parameter in model.actor.parameters())


def _assert_replay_batching_preserves_episode_weighting_and_gradients(device):
    graph, agents = _instance()
    observation = batch_observations([
        build_observation(graph, agents, 2), build_observation(graph, agents, 2)])
    for use_critic in (False, True):
        model = _graph_model(use_critic=use_critic).to(device)
        with torch.no_grad():
            decoded = model.decode(observation.to(device), training=False)
        traces = [DecisionTrace(observation, [
            decoded.selected_group_indices[0] if step % 3 else [],
            decoded.selected_group_indices[1] if step % 2 else [],
        ]) for step in range(6)]
        rollout = SimpleNamespace(decision_traces=traces)
        expected_outputs = expected_gradients = None
        for replay_batch_size in (1, 2, 5, 64):
            model.zero_grad(set_to_none=True)
            outputs = replay_tensor_gradients(
                model, rollout, torch.tensor([-2.0, 0.5]),
                entropy_coefficient=0.01, update_size=5, device=device,
                replay_batch_size=replay_batch_size)
            gradients = [None if parameter.grad is None else parameter.grad.clone()
                         for parameter in model.parameters()]
            assert outputs[1].tolist() == [4.0, 3.0]
            if expected_outputs is None:
                expected_outputs, expected_gradients = outputs, gradients
                continue
            assert all(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
                       for actual, expected in zip(outputs, expected_outputs))
            for actual, expected in zip(gradients, expected_gradients):
                if expected is None:
                    assert actual is None
                else:
                    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_replay_batching_preserves_episode_weighting_and_gradients():
    _assert_replay_batching_preserves_episode_weighting_and_gradients("cpu")


def test_replay_batching_preserves_episode_weighting_and_gradients_when_cuda():
    if torch.cuda.is_available():
        _assert_replay_batching_preserves_episode_weighting_and_gradients("cuda")


def test_cached_candidate_generation_matches_uncached_generation():
    graph, _agents = _instance(False)
    config = load_config().candidates
    uncached = generate_candidates(graph, config)
    cached = generate_candidates(graph, config, CandidateTerrainCache(graph))
    assert [candidate.key for candidate in cached] == [
        candidate.key for candidate in uncached]
    assert [candidate.staging_targets for candidate in cached] == [
        candidate.staging_targets for candidate in uncached]


def test_scenario_cache_computes_staging_geometry_only_once():
    graph = _pair_graph(("a", "b", "c"))
    config = _candidate_config(staging_per_target=1)
    with (
        patch.object(
            candidate_module, "_safe_distances_to_target",
            wraps=candidate_module._safe_distances_to_target,
        ) as distance_spy,
        patch.object(
            candidate_module, "_pair_staging_definitions",
            wraps=candidate_module._pair_staging_definitions,
        ) as pair_spy,
    ):
        cache = CandidateScenarioCache(
            graph, include_pair_staging=config.include_pair_staging)
        assert distance_spy.call_count == 3
        assert pair_spy.call_count == 1

        first = generate_candidates(
            graph, config, scenario_cache=cache)
        graph.nodes["c"]["rps_type"] = 1
        second = generate_candidates(
            graph, config, scenario_cache=cache)

        assert distance_spy.call_count == 3
        assert pair_spy.call_count == 1
        assert len([item for item in first if item.staging_arity == 2]) == 3
        assert len([item for item in second if item.staging_arity == 2]) == 1


def test_pair_staging_count_selection_and_belief_filtering():
    config = _candidate_config()
    empty = nx.DiGraph()
    assert not [candidate for candidate in generate_candidates(empty, config)
                if candidate.staging_arity == 2]

    singleton = _pair_graph(("a",))
    assert not [candidate for candidate in generate_candidates(singleton, config)
                if candidate.staging_arity == 2]

    two = _pair_graph(("a", "b"))
    pair = [candidate for candidate in generate_candidates(two, config)
            if candidate.staging_arity == 2]
    assert len(pair) == 1
    assert pair[0].staging_targets == {"a", "b"}

    weights = {
        ("a", "b"): (1, 1),
        ("a", "c"): (2, 2),
        ("a", "d"): (3, 3),
        ("b", "c"): (4, 4),
        ("b", "d"): (5, 5),
        ("c", "d"): (6, 6),
    }
    four = _pair_graph(("a", "b", "c", "d"), weights)
    selected = [candidate for candidate in generate_candidates(four, config)
                if candidate.staging_arity == 2]
    assert len(selected) == 4
    assert {frozenset(candidate.staging_targets) for candidate in selected} == {
        frozenset(pair) for pair in (
            ("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"))}

    four.nodes["c"]["rps_type"] = 1
    four.nodes["d"]["type"] = "target_reached"
    filtered = [candidate for candidate in generate_candidates(four, config)
                if candidate.staging_arity == 2]
    assert len(filtered) == 1
    assert filtered[0].staging_targets == {"a", "b"}


def test_pair_staging_excludes_infinite_pairs_and_uses_symmetric_distance():
    config = _candidate_config()
    one_way = _pair_graph(
        ("a", "b"), {("a", "b"): (2.0, None)})
    assert not [candidate for candidate in generate_candidates(one_way, config)
                if candidate.staging_arity == 2]

    asymmetric = _pair_graph(
        ("a", "b"), {("a", "b"): (2.0, 7.0)})
    pair = next(candidate for candidate in generate_candidates(
        asymmetric, config) if candidate.staging_arity == 2)
    assert pair.pair_distance == 7.0


def test_pair_staging_ties_use_stable_target_order():
    targets = ("a", "b", "c", "d")
    weights = {pair: (1.0, 1.0) for pair in combinations(targets, 2)}
    graph = _pair_graph(targets, weights)
    pairs = [candidate for candidate in generate_candidates(
        graph, _candidate_config()) if candidate.staging_arity == 2]
    assert {frozenset(candidate.staging_targets) for candidate in pairs} == {
        frozenset(pair) for pair in (
            ("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"))}


def test_pair_staging_minimax_total_and_repr_tiebreaks():
    config = _candidate_config()

    def graph_with_locations(locations):
        graph = nx.DiGraph()
        for index, target in enumerate(("p", "q")):
            graph.add_node(
                target, pos=(index, 1), height=0.0,
                type="target_unreached", rps_type=UNKNOWN_TYPE,
                visible_edges=[])
        graph.add_edge("p", "q", distance=10.0)
        graph.add_edge("q", "p", distance=10.0)
        for index, (node, to_p, to_q) in enumerate(locations):
            graph.add_node(
                node, pos=(index, 0), height=0.0,
                type="intermediate", visible_edges=[])
            if to_p is not None:
                graph.add_edge(node, "p", distance=float(to_p))
            if to_q is not None:
                graph.add_edge(node, "q", distance=float(to_q))
        return graph

    minimax = graph_with_locations([
        ("x", 5, 1), ("y", 3, 3), ("unreachable", 1, None)])
    pair = next(candidate for candidate in generate_candidates(
        minimax, config) if candidate.staging_arity == 2)
    assert pair.node == "y"

    total_tie = graph_with_locations([("x", 4, 1), ("y", 4, 3)])
    pair = next(candidate for candidate in generate_candidates(
        total_tie, config) if candidate.staging_arity == 2)
    assert pair.node == "x"

    repr_tie = graph_with_locations([("x", 2, 2), ("y", 2, 2)])
    pair = next(candidate for candidate in generate_candidates(
        repr_tie, config) if candidate.staging_arity == 2)
    assert pair.node == "x"
    assert pair.node not in pair.staging_targets


def test_single_staging_ranks_candidate_to_target_on_directed_graph():
    graph = nx.DiGraph()
    graph.add_node(
        "target", pos=(0, 1), height=0.0, type="target_unreached",
        rps_type=UNKNOWN_TYPE, visible_edges=[])
    for index, node in enumerate(("near_from_target", "near_to_target")):
        graph.add_node(
            node, pos=(index, 0), height=0.0,
            type="intermediate", visible_edges=[])
    graph.add_edge("target", "near_from_target", distance=1.0)
    graph.add_edge("near_from_target", "target", distance=100.0)
    graph.add_edge("target", "near_to_target", distance=100.0)
    graph.add_edge("near_to_target", "target", distance=2.0)
    singles = [candidate for candidate in generate_candidates(
        graph, _candidate_config(staging_per_target=1))
        if candidate.staging_arity == 1]
    assert len(singles) == 1
    assert singles[0].node == "near_to_target"


def test_semantic_staging_aliases_share_physical_group_and_encode_arity():
    graph = nx.DiGraph()
    for index, target in enumerate(("p", "q")):
        graph.add_node(
            target, pos=(index, 1), height=0.0,
            type="target_unreached", rps_type=UNKNOWN_TYPE,
            visible_edges=[])
    graph.add_node(
        "v", pos=(0, 0), height=0.0,
        type="source", visible_edges=[])
    graph.add_edge("p", "q", distance=5.0)
    graph.add_edge("q", "p", distance=5.0)
    graph.add_edge("v", "p", distance=1.0)
    graph.add_edge("v", "q", distance=1.0)
    candidates = generate_candidates(
        graph, _candidate_config(staging_per_target=1))
    staging = [candidate for candidate in candidates
               if candidate.node == "v" and candidate.is_staging]
    assert len(staging) == 3
    assert len({candidate.semantic_key for candidate in staging}) == 3
    assert len({candidate.physical_key for candidate in staging}) == 1
    groups, capacities, _representatives = physical_group_metadata(candidates)
    staging_indices = [candidates.index(candidate) for candidate in staging]
    assert len({groups[index] for index in staging_indices}) == 1
    assert capacities[groups[staging_indices[0]]] == 1

    observation = build_observation(
        graph, [Agent("v", capabilities={0, 1, 2})], 2,
        candidates=candidates)
    single_indices = [index for index, candidate in enumerate(candidates)
                      if candidate.staging_arity == 1]
    pair_index = next(index for index, candidate in enumerate(candidates)
                      if candidate.staging_arity == 2)
    assert torch.equal(
        observation.action_features[0, single_indices, 4],
        torch.full((2,), 0.5))
    assert observation.action_features[0, pair_index, 4] == 1.0
    assert observation.stages_for_mask[0, pair_index].sum() == 2


def _check_tensor_observation_region_updates(device):
    graph = nx.DiGraph()
    nodes = ("s", "u", "x", "x2", "y", "z", "a", "b", "c")
    for i, node in enumerate(nodes):
        graph.add_node(node, pos=(i, i % 3), height=float(i),
                       type="target_unreached" if node in "abc" else "intermediate",
                       rps_type=UNKNOWN_TYPE, visible_nodes=[])
    graph.nodes["s"]["type"] = "source"
    for node, forward, backward in (("u", 8., 2.), ("x", 5., 3.),
                                     ("x2", 9., 4.), ("y", 1., 7.),
                                     ("z", 2., 1.), ("a", 11., 13.),
                                     ("b", 12., 14.), ("c", 13., 15.)):
        graph.add_edge("s", node, distance=forward, observed_edge=False)
        graph.add_edge(node, "s", distance=backward, observed_edge=False)
    for node, visible in (("u", ["a"]), ("x", ["a", "b"]),
                           ("x2", ["a", "b"]), ("y", ["b"]),
                           ("z", ["b", "c"])):
        graph.nodes[node]["visible_nodes"] = visible
    candidates = replace(load_config().candidates, staging_per_target=1)
    if device == "cpu":
        with patch("learning.gpu_sim.world.CuGraphRouter", _ReferenceRouter):
            world = TensorWorld.from_networkx(graph, candidates, device=device)
    else:
        world = TensorWorld.from_networkx(graph, candidates, device=device)
    state = TensorEpisodeState.create(
        world, [world.node_index["s"]],
        [[[True, True, False], [False, True, False]]] * 2, [[1, 2, 1]] * 2)
    snapshots = []
    for method in (RAW_EDGE_NORMALIZATION, PER_DECISION_EDGE_NORMALIZATION,
                   MAX_EDGE_NORMALIZATION):
        builder = TensorObservationBuilder(world, 2, edge_normalization=method)
        # Rows reveal different subsets in the same batch. Later remove a
        # target too, then remove every unknown observation action.
        for phase in range(4):
            known = ([[False, False, False]] * 2 if phase == 0 else
                     [[True, False, False], [True, False, True]] if phase == 1 else
                     [[False, False, True], [True, True, False]] if phase == 2 else
                     [[True, True, True]] * 2)
            live = ([[True, False, True], [False, True, True]] if phase == 2
                    else [[True, True, True]] * 2)
            state.target_known[:] = torch.tensor(known, device=device)
            state.target_live[:] = torch.tensor(live, device=device)
            built = builder.build(state)
            tensor_observation = built[0]
            snapshots.append(tensor_observation.to("cpu"))
            representatives = builder.observation_representatives(state).cpu()
            model = _graph_model(use_position=True).to(device)
            tensor_logits, tensor_values = model.actor_critic(tensor_observation)
            for row in range(2):
                belief = graph.copy()
                for t, node in enumerate(world.targets):
                    belief.nodes[node].update(
                        type="target_unreached" if live[row][t] else "target_reached",
                        rps_type=(1, 2, 1)[t] if known[row][t] else UNKNOWN_TYPE)
                cpu = build_observation(
                    belief, [Agent("s", capabilities={0, 1}),
                             Agent("s", capabilities={1})], 2,
                    candidate_config=candidates, edge_normalization=method)
                assert torch.allclose(tensor_observation.target_positions[row].cpu(),
                                      cpu.target_positions[0])
                by_key = {candidate.semantic_key: i
                          for i, candidate in enumerate(cpu.candidates[0])}
                active_slots = torch.where(tensor_observation.action_mask[row])[0].tolist()
                matched = {}
                for slot in active_slots:
                    original = world.candidates[slot]
                    if original.is_observation:
                        revealed = tensor_observation.reveals_mask[row, slot].cpu()
                        signature = tuple(t for i, t in enumerate(world.targets) if revealed[i])
                        key = ("observation", signature)
                        atom_slots = torch.where(representatives[row] == slot)[0].tolist()
                        region = frozenset().union(*(
                            world.candidates[i].region_nodes for i in atom_slots))
                        assert region == cpu.candidates[0][by_key[key]].region_nodes
                    else:
                        key = original.semantic_key
                    assert key not in matched, "duplicate active observation signature"
                    matched[key] = slot
                assert set(matched) == set(by_key)
                order = [matched[c.semantic_key] for c in cpu.candidates[0]]
                for name in ("action_features", "task_action_features",
                             "action_target_relations", "action_target_distances",
                             "action_target_distance_mask", "reveals_mask",
                             "serves_mask", "stages_for_mask"):
                    actual = getattr(tensor_observation, name)[row, order].cpu()
                    assert torch.allclose(actual, getattr(cpu, name)[0], atol=1e-5), (method, phase, row, name)
                for name in ("agent_action_distances", "agent_action_relations",
                             "agent_action_distance_mask", "feasible_action_mask"):
                    actual = getattr(tensor_observation, name)[row][:, order].cpu()
                    assert torch.allclose(actual, getattr(cpu, name)[0], atol=1e-5), (method, phase, row, name)
                cpu_logits, cpu_values = model.actor_critic(cpu.to(device))
                assert torch.allclose(tensor_logits[row][:, order], cpu_logits[0], atol=1e-5)
                assert torch.allclose(tensor_values[row], cpu_values[0], atol=1e-5)
                if phase == 1 and row == 0:
                    slot = matched[("observation", ("b",))]
                    # The representative atom is {x,x2}; the best destination
                    # is y, which belongs to a different atom in the union.
                    assert world.candidates[slot].region_nodes == frozenset({"x", "x2"})
                    assert built[-1][row, 0, slot].item() == world.node_index["y"]
                    actions = torch.full_like(state.positions, slot)
                    next_hop, reachable, routes, destinations = builder.next_hops(state, actions, *built[1:])
                    assert reachable[row, 0]
                    assert destinations[row, 0].item() == world.node_index["y"]
                    assert next_hop[row, 0].item() == world.node_index["y"]
    # Replaying an earlier decision must not observe later region changes.
    model = _graph_model(use_position=True)
    trace = snapshots[1]
    decoded = model.decode(trace, training=False)
    rollout = SimpleNamespace(decision_traces=[DecisionTrace(trace, decoded.selected_group_indices)])
    losses, *_rest = replay_tensor_gradients(
        model, rollout, torch.tensor([1., -1.]), entropy_coefficient=0.01,
        update_size=2, device="cpu")
    assert torch.isfinite(losses).all()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in model.agent_encoder.parameters())


def test_tensor_observation_regions_follow_current_unknown_sets():
    _check_tensor_observation_region_updates("cpu")


def test_cuda_observation_regions_follow_current_unknown_sets_when_cuda():
    if torch.cuda.is_available():
        _check_tensor_observation_region_updates("cuda")


def test_cuda_static_pair_activation_matches_cpu_belief_selection():
    graph = _pair_graph(("a", "b", "c", "d"))
    config = _candidate_config()
    static_candidates = generate_candidates(
        graph, config, include_all_pairs=True)
    targets = sorted(
        (node for node, data in graph.nodes(data=True)
         if data["type"].startswith("target_")), key=repr)
    target_index = {target: index for index, target in enumerate(targets)}
    shape = (len(static_candidates), len(targets))
    target_mask = torch.zeros(shape, dtype=torch.bool)
    observed_mask = torch.zeros_like(target_mask)
    staging_mask = torch.zeros_like(target_mask)
    for candidate_index, candidate in enumerate(static_candidates):
        if candidate.is_target:
            target_mask[candidate_index, target_index[candidate.node]] = True
        for target in candidate.observed_targets:
            observed_mask[candidate_index, target_index[target]] = True
        for target in candidate.staging_targets:
            staging_mask[candidate_index, target_index[target]] = True
    pair_order = sorted(
        (index for index, candidate in enumerate(static_candidates)
         if candidate.staging_arity == 2),
        key=lambda index: (
            static_candidates[index].pair_distance,
            tuple(repr(target) for target in sorted(
                static_candidates[index].staging_targets, key=repr))))
    world = SimpleNamespace(
        target_candidate_mask=target_mask,
        candidate_nodes=torch.arange(len(static_candidates)),
        candidate_observed_mask=observed_mask,
        candidate_staging_mask=staging_mask,
        candidate_staging_arity=torch.tensor([
            candidate.staging_arity for candidate in static_candidates]),
        candidate_pair_order=torch.tensor(pair_order, dtype=torch.long),
        candidate_is_wait=torch.tensor([
            candidate.is_wait for candidate in static_candidates]),
    )
    state = SimpleNamespace(
        target_live=torch.tensor([[True, True, True, False]]),
        target_known=torch.tensor([[False, False, True, False]]),
    )
    roles = TensorObservationBuilder(world, 3).candidate_roles(state)
    is_staging, staging_links = roles[2], roles[6]
    active_pairs = {
        frozenset(static_candidates[index].staging_targets)
        for index in torch.where(
            is_staging[0]
            & (world.candidate_staging_arity == 2))[0].tolist()
    }

    cpu_graph = graph.copy()
    cpu_graph.nodes["c"]["rps_type"] = 1
    cpu_graph.nodes["d"]["type"] = "target_reached"
    cpu_pairs = {
        frozenset(candidate.staging_targets)
        for candidate in generate_candidates(cpu_graph, config)
        if candidate.staging_arity == 2
    }
    assert active_pairs == cpu_pairs == {frozenset(("a", "b"))}
    active_pair_index = next(index for index in range(len(static_candidates))
                             if is_staging[0, index]
                             and static_candidates[index].staging_arity == 2)
    assert staging_links[0, active_pair_index].sum() == 2


def test_hidden_ground_truth_never_enters_observation():
    env, agents = _instance(False)
    env.nodes[4]["rps_type"] = UNKNOWN_TYPE
    first = build_observation(env, agents, 2)
    truth_a, truth_b = env.copy(), env.copy()
    truth_a.nodes[4]["rps_type"] = 1
    truth_b.nodes[4]["rps_type"] = 2
    # The builder accepts no truth graph; changing either truth copy cannot
    # affect a planner observation.
    second = build_observation(env, agents, 2)
    assert torch.equal(first.target_features, second.target_features)
    assert first.target_features[0, 0, -2:].sum() == 0
    assert torch.equal(
        first.target_capability_beliefs,
        torch.tensor([[[0.5, 0.5]]]))


def test_variable_sizes_batch_and_padding_invariance():
    graph, agents = _instance()
    small = build_observation(graph, agents[:1], 2)
    large = build_observation(graph, agents, 2)
    batch = batch_observations([small, large])
    assert batch.agent_features.shape[1] == 2
    assert not batch.agent_mask[0, 1]
    model = _model()
    alone = model(small)[0, :1, :small.action_features.shape[1]]
    padded = model(batch)[0, :1, :small.action_features.shape[1]]
    assert torch.allclose(alone, padded, atol=1e-5)


def test_agent_action_target_permutation_equivariance():
    graph, agents = _instance()
    observation = build_observation(graph, agents, 2)
    model = _model()
    base = model(observation)

    swapped_agents = copy.copy(observation)
    order_a = torch.tensor([1, 0])
    swapped_agents.agent_features = observation.agent_features[:, order_a]
    swapped_agents.agent_mask = observation.agent_mask[:, order_a]
    swapped_agents.agent_target_relations = observation.agent_target_relations[:, order_a]
    swapped_agents.agent_action_relations = observation.agent_action_relations[:, order_a]
    swapped_agents.feasible_action_mask = observation.feasible_action_mask[:, order_a]
    assert torch.allclose(model(swapped_agents), base[:, order_a], atol=1e-5,
                          equal_nan=True)

    swapped_actions = copy.copy(observation)
    order_c = torch.arange(observation.action_features.shape[1] - 1, -1, -1)
    swapped_actions.action_features = observation.action_features[:, order_c]
    swapped_actions.action_mask = observation.action_mask[:, order_c]
    swapped_actions.agent_action_relations = observation.agent_action_relations[:, :, order_c]
    swapped_actions.action_target_relations = observation.action_target_relations[:, order_c]
    swapped_actions.feasible_action_mask = observation.feasible_action_mask[:, :, order_c]
    assert torch.allclose(model(swapped_actions), base[:, :, order_c], atol=1e-5,
                          equal_nan=True)

    # Duplicate the target token, then verify an explicit target permutation.
    graph.nodes[3].update(type="target_unreached", rps_type=2)
    two_targets = build_observation(graph, agents, 2)
    base_two = model(two_targets)
    swapped_targets = copy.copy(two_targets)
    order_t = torch.tensor([1, 0])
    swapped_targets.target_features = two_targets.target_features[:, order_t]
    swapped_targets.target_mask = two_targets.target_mask[:, order_t]
    swapped_targets.agent_target_relations = two_targets.agent_target_relations[:, :, order_t]
    swapped_targets.action_target_relations = two_targets.action_target_relations[:, :, order_t]
    assert torch.allclose(model(swapped_targets), base_two, atol=1e-5)


def test_masks_enforce_dead_transit_scout_and_compatibility_rules():
    graph, agents = _instance()
    agents[1].alive = False
    candidates = generate_candidates(graph, load_config().candidates)
    observation = build_observation(graph, agents, 2, candidates)
    assert not observation.feasible_action_mask[0, 1].any()
    target_index = next(i for i, c in enumerate(candidates) if c.is_target)
    assert observation.feasible_action_mask[0, 0, target_index]
    # Type-2-only agent cannot select known type-1 target.
    assert not observation.feasible_action_mask[0, 1, target_index]

    live_agent = Agent(0, capabilities={2})
    transit = [(0, 1, 0.0, 1.0)]
    moving = build_observation(graph, [live_agent], 2, candidates, transit, 0.5)
    assert not moving.feasible_action_mask.any()

    future = build_observation(
        graph, [Agent(0, capabilities={1})], 2, candidates, transit, 0.5,
        replan_transit=True, edge_normalization=PER_DECISION_EDGE_NORMALIZATION)
    target_index = next(i for i, item in enumerate(candidates)
                        if item.is_target)
    assert future.feasible_action_mask[0, 0, target_index]
    # The future route begins at committed arrival node 1, three edges from
    # target 4. The normalized Transformer relation and task-graph edge share
    # the new per-decision value.
    assert torch.equal(
        future.agent_action_relations[0, 0, target_index, 0],
        future.agent_action_distances[0, 0, target_index, 0])

    graph.nodes[4]["rps_type"] = UNKNOWN_TYPE
    pure_observe = [Candidate(2, is_observation=True, observed_targets={4}),
                    Candidate(None, is_wait=True, capacity=None)]
    blind = build_observation(graph, [Agent(0, capabilities={1})], 2,
                              pure_observe)
    assert not blind.feasible_action_mask[0, 0, 0]


def test_decoder_constraints_and_probabilities():
    logits = torch.tensor([[[10.0, 1.0], [9.0, 1.0], [8.0, 1.0]]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    capacities = torch.tensor([[1, torch.iinfo(torch.long).max]])
    output = AssignmentDecoder()(logits, valid, capacities, training=False)
    assert len({a for a, _ in output.assignments[0]}) == 3
    assert sum(c == 0 for _, c in output.assignments[0]) == 1
    assert sum(c == 1 for _, c in output.assignments[0]) == 2
    replay_logp, replay_entropy = AssignmentDecoder().evaluate_selected(
        logits, valid, capacities, output.selected_pair_indices)
    assert torch.allclose(replay_logp, output.log_probabilities)
    assert torch.allclose(replay_entropy, output.entropies)
    masked = valid.clone()
    masked[:, :, 0] = False
    sampled = AssignmentDecoder()(logits, masked, capacities, training=True)
    assert all(c == 1 for _, c in sampled.assignments[0])
    assert torch.isfinite(sampled.log_probabilities).all()


def test_grouped_decoder_sums_alias_probabilities_and_replays_exactly():
    decoder = AssignmentDecoder()
    logits = torch.tensor([[[0.0, 0.0, 0.5]]], requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    action_capacities = torch.ones((1, 3), dtype=torch.long)
    candidate_groups = torch.tensor([[0, 0, 1]])
    group_capacities = torch.tensor([[1, 1]])
    representatives = torch.tensor([[0, 2]])
    group_mask = torch.ones((1, 2), dtype=torch.bool)

    grouped = decoder.grouped_logits(
        logits[0], valid[0], candidate_groups[0], group_mask[0])
    assert torch.allclose(
        grouped[0],
        torch.stack((torch.logsumexp(logits[0, 0, :2], dim=0),
                     logits[0, 0, 2])))
    output = decoder(
        logits, valid, action_capacities, training=False,
        candidate_physical_group=candidate_groups,
        physical_group_capacity=group_capacities,
        physical_group_representative=representatives,
        physical_group_mask=group_mask)
    # Neither alias is individually largest, but their summed location wins.
    assert output.assignments == [[(0, 0)]]
    expected_distribution = torch.distributions.Categorical(logits=grouped[0])
    assert torch.allclose(
        output.log_probabilities[0],
        expected_distribution.log_prob(torch.tensor(0)))
    assert torch.allclose(
        output.entropies[0], expected_distribution.entropy())

    replay_logp, replay_entropy = decoder.evaluate_selected(
        logits, valid, action_capacities, output.selected_pair_indices,
        candidate_physical_group=candidate_groups,
        physical_group_capacity=group_capacities,
        physical_group_representative=representatives,
        physical_group_mask=group_mask)
    assert torch.allclose(replay_logp, output.log_probabilities)
    assert torch.allclose(replay_entropy, output.entropies)

    output.log_probabilities.sum().backward()
    assert logits.grad[0, 0, 0] != 0
    assert logits.grad[0, 0, 1] != 0


def test_grouped_decoder_consumes_shared_capacity_once():
    decoder = AssignmentDecoder()
    logits = torch.tensor([[
        [10.0, 9.0, 0.0],
        [8.0, 7.0, 0.0],
    ]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    unlimited = torch.iinfo(torch.long).max
    output = decoder(
        logits, valid, torch.ones((1, 3), dtype=torch.long),
        training=False,
        candidate_physical_group=torch.tensor([[0, 0, 1]]),
        physical_group_capacity=torch.tensor([[1, unlimited]]),
        physical_group_representative=torch.tensor([[0, 2]]),
        physical_group_mask=torch.ones((1, 2), dtype=torch.bool))
    assert len(output.assignments[0]) == 2
    assert sum(action == 0 for _agent, action in output.assignments[0]) == 1
    assert sum(action == 2 for _agent, action in output.assignments[0]) == 1


def test_grouped_logits_are_invariant_to_semantic_alias_permutation():
    decoder = AssignmentDecoder()
    logits = torch.tensor([
        [0.2, -0.4, 0.7, 0.1],
        [1.2, 0.3, -0.5, 0.8],
    ])
    valid = torch.ones_like(logits, dtype=torch.bool)
    groups = torch.tensor([0, 1, 0, 1])
    group_mask = torch.ones(2, dtype=torch.bool)
    base = decoder.grouped_logits(logits, valid, groups, group_mask)
    order = torch.tensor([3, 0, 2, 1])
    permuted = decoder.grouped_logits(
        logits[:, order], valid[:, order], groups[order], group_mask)
    assert torch.allclose(base, permuted)


def test_grouped_decoder_cpu_cuda_parity_when_cuda():
    if not torch.cuda.is_available():
        return
    decoder = AssignmentDecoder()
    logits = torch.tensor([[
        [0.0, 0.0, 0.5],
        [1.0, -0.5, 0.25],
    ]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    action_capacities = torch.ones((1, 3), dtype=torch.long)
    candidate_groups = torch.tensor([[0, 0, 1]])
    group_capacities = torch.tensor([[1, torch.iinfo(torch.long).max]])
    representatives = torch.tensor([[0, 2]])
    group_mask = torch.ones((1, 2), dtype=torch.bool)

    def decode(device):
        return decoder(
            logits.to(device), valid.to(device),
            action_capacities.to(device), training=False,
            candidate_physical_group=candidate_groups.to(device),
            physical_group_capacity=group_capacities.to(device),
            physical_group_representative=representatives.to(device),
            physical_group_mask=group_mask.to(device))

    cpu = decode("cpu")
    cuda = decode("cuda")
    assert cpu.assignments == cuda.assignments
    assert cpu.selected_group_indices == cuda.selected_group_indices
    assert torch.allclose(
        cpu.log_probabilities, cuda.log_probabilities.cpu(), atol=1e-6)
    assert torch.allclose(cpu.entropies, cuda.entropies.cpu(), atol=1e-6)


def test_batched_decoder_handles_padding_and_replays_the_same_gradients():
    decoder = AssignmentDecoder()
    torch.manual_seed(71)
    logits = torch.randn(3, 4, 5, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[1, 1:] = False
    valid[2] = False
    metadata = {
        "candidate_physical_group": torch.tensor([[0, 0, 1, 2, 2]]).expand(3, -1),
        "physical_group_capacity": torch.tensor([[1, 2, 0]]).expand(3, -1),
        "physical_group_representative": torch.tensor([[0, 2, 3]]).expand(3, -1),
        "physical_group_mask": torch.tensor([[True, True, False]]).expand(3, -1),
    }
    decoded = decoder(logits, valid, None, **metadata)
    assert list(map(len, decoded.assignments)) == [3, 1, 0]
    assert torch.equal(decoded.action_indices[2], torch.full((4,), -1))
    assert decoded.log_probabilities[2] == decoded.entropies[2] == 0
    for row in range(3):
        single = decoder(
            logits[row:row+1], valid[row:row+1], None,
            **{name: value[row:row+1] for name, value in metadata.items()})
        assert single.assignments[0] == decoded.assignments[row]
        assert torch.allclose(single.log_probabilities[0], decoded.log_probabilities[row])
        assert torch.allclose(single.entropies[0], decoded.entropies[row])
        for agent, action in decoded.assignments[row]:
            assert decoded.action_indices[row, agent] == action
    sampled = decoder(
        logits, valid, None, training=True,
        generator=torch.Generator().manual_seed(17), **metadata)
    logp, entropy = decoder.evaluate_selected(
        logits, valid, None, sampled.selected_group_indices, **metadata)
    assert torch.allclose(logp, sampled.log_probabilities)
    assert torch.allclose(entropy, sampled.entropies)
    replay_gradient = torch.autograd.grad((logp + 0.03 * entropy).sum(), logits)[0]
    sample_gradient = torch.autograd.grad(
        (sampled.log_probabilities + 0.03 * sampled.entropies).sum(), logits)[0]
    assert torch.isfinite(replay_gradient).all()
    assert torch.allclose(replay_gradient, sample_gradient, atol=1e-6)
    assert not replay_gradient[2].any()


def test_physical_group_capacity_conflicts_are_rejected():
    aliases = [
        Candidate(
            "v", is_staging=True, staging_arity=1,
            staging_targets={"a"}, capacity=1),
        Candidate(
            "v", is_staging=True, staging_arity=1,
            staging_targets={"b"}, capacity=2),
    ]
    try:
        physical_group_metadata(aliases)
    except ValueError as error:
        assert "conflicting capacities" in str(error)
    else:
        raise AssertionError("conflicting alias capacities were accepted")


def test_forward_backward_and_tiny_overfit():
    graph, agents = _instance(False)
    observation = build_observation(graph, agents, 2)
    model = _model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    target = next(i for i, c in enumerate(observation.candidates[0]) if c.is_target)
    initial = None
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(observation)[0, 0]
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([target]))
        initial = float(loss.detach()) if initial is None else initial
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                   if p.grad is not None)
        optimizer.step()
    assert float(loss.detach()) < initial * 0.1


class _TargetFirst(torch.nn.Module):
    """Deterministic decoder used only to exercise the complete adapter path."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def decode(self, observation, candidates, training=False):
        pairs = []
        used = set()
        for agent in range(observation.agent_features.shape[1]):
            valid = observation.feasible_action_mask[0, agent]
            choices = [i for i, item in enumerate(candidates[0])
                       if item.is_target and valid[i] and i not in used]
            if not choices:
                choices = [i for i, item in enumerate(candidates[0])
                           if item.is_wait and valid[i]]
            if choices:
                pairs.append((agent, choices[0]))
                if candidates[0][choices[0]].capacity is not None:
                    used.add(choices[0])
        zero = self.anchor.reshape(1) * 0
        return DecoderOutput([pairs], [[]], zero, zero)


class _GroupedAliasesThenTargets(torch.nn.Module):
    """Smoke policy whose first choice only wins after alias aggregation."""

    def __init__(self, staging_node):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.decoder = AssignmentDecoder()
        self.staging_node = staging_node
        self.calls = 0
        self.first_individual_is_target = None
        self.first_selected_node = None

    def decode(self, observation, candidates, training=False):
        logits = observation.action_features.new_full(
            observation.feasible_action_mask.shape, -5.0)
        items = candidates[0]
        if self.calls == 0:
            for action, candidate in enumerate(items):
                if candidate.is_staging and candidate.node == self.staging_node:
                    logits[0, :, action] = 0.0
                elif candidate.is_target:
                    logits[0, :, action] = 0.5
            individual_action = int(logits[0, 0].argmax())
            self.first_individual_is_target = items[
                individual_action].is_target
        else:
            for action, candidate in enumerate(items):
                if candidate.is_target:
                    logits[0, :, action] = 10.0
        logits = logits + self.anchor * 0.0
        output = self.decoder(
            logits, observation.feasible_action_mask,
            observation.action_capacities, training=training,
            candidate_physical_group=observation.candidate_physical_group,
            physical_group_capacity=observation.physical_group_capacity,
            physical_group_representative=(
                observation.physical_group_representative),
            physical_group_mask=observation.physical_group_mask)
        if self.calls == 0:
            selected_action = output.assignments[0][0][1]
            self.first_selected_node = items[selected_action].node
        self.calls += 1
        return output


def test_cpu_rollout_executes_grouped_alias_location_winner():
    truth = nx.DiGraph()
    for node, position, node_type in (
        ("s", (0, 0), "source"),
        ("v", (1, 0), "intermediate"),
        ("p", (2, 1), "target_unreached"),
        ("q", (2, -1), "target_unreached"),
    ):
        truth.add_node(
            node, pos=position, height=0.0, type=node_type,
            visible_edges=[])
    for source, target, distance in (
        ("s", "v", 1), ("v", "s", 1),
        ("v", "p", 1), ("p", "v", 1),
        ("v", "q", 1), ("q", "v", 1),
        ("p", "q", 2), ("q", "p", 2),
    ):
        truth.add_edge(
            source, target, distance=float(distance), observed_edge=False)
    env = truth.copy()
    init_target_types(env, truth, {"p": 1, "q": 2})
    config = replace(
        _candidate_config(staging_per_target=1),
        allow_unknown_target_actions=True)
    model = _GroupedAliasesThenTargets("v")
    policy = LearnedPolicyAdapter(
        model, 2, candidate_config=config, training=False)
    result = run_simulation(
        env, truth, [Agent("s", capabilities={1, 2})], policy=policy)
    # The recorded individual winner was a target, while grouped decoding
    # selected the representative of the three-alias staging location.
    assert model.first_individual_is_target
    assert model.first_selected_node == "v"
    assert result["completed"]


def test_complete_adapter_episode_runs_through_simulator():
    truth = _line(3)
    env = _line(3)
    truth.nodes[2]["type"] = env.nodes[2]["type"] = "target_unreached"
    init_target_types(env, truth, {2: 1})
    # Contact is known in planner view for this integration-only instance.
    env.nodes[2]["rps_type"] = 1
    agents = [Agent(0, capabilities={1})]
    policy = LearnedPolicyAdapter(_TargetFirst(), 1)
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"]


def test_learning_rollout_forwards_rendering():
    truth = _line(3)
    env = _line(3)
    truth.nodes[2]["type"] = env.nodes[2]["type"] = "target_unreached"
    init_target_types(env, truth, {2: 1})
    env.nodes[2]["rps_type"] = 1
    agents = [Agent(0, capabilities={1})]
    policy = LearnedPolicyAdapter(_TargetFirst(), 1)
    with tempfile.TemporaryDirectory() as directory:
        rollout = collect_episode(
            env, truth, agents, policy, render_dir=directory, render_dt=1.0,
            measure_policy_time=True)
        frames = sorted(Path(directory).glob("frame_*.png"))
    assert rollout.result["completed"]
    assert rollout.result["policy_time_seconds"] >= 0.0
    assert len(frames) == 3


def test_cuda_trace_render_reconstructs_observed_nodes():
    env = nx.DiGraph()
    for index, node in enumerate(("source", "hall", "target")):
        env.add_node(node, pos=(float(index), 0.0), type="intermediate")
    env.nodes["source"].update(type="source", visible_nodes=("source", "hall"))
    env.nodes["hall"]["visible_nodes"] = ("hall", "target")
    env.nodes["target"].update(
        type="target_unreached", rps_type=UNKNOWN_TYPE,
        visible_nodes=("target",))
    truth = env.copy()
    truth.nodes["target"]["rps_type"] = 1
    world = SimpleNamespace(nodes=("source", "hall", "target"),
                            targets=("target",))

    def snapshot(node, clock, moving, known):
        return {
            "positions": torch.tensor([node]),
            "alive": torch.tensor([True]),
            "target_live": torch.tensor([True]),
            "target_known": torch.tensor([known]),
            "moving": torch.tensor([moving]),
            "transit_from": torch.tensor([0]),
            "transit_to": torch.tensor([1]),
            "arrival_time": torch.tensor([1.0]),
            "clock": torch.tensor(clock),
            "route_next": torch.full((1, 3), -1, dtype=torch.long),
        }

    rendered = []

    def capture(frame_env, _truth, _agents, _index, _path, **_kwargs):
        rendered.append((
            {node for node, data in frame_env.nodes(data=True)
             if data.get("observed_node")},
            frame_env.nodes["target"]["rps_type"],
        ))

    trace = [snapshot(0, 0.0, True, False),
             snapshot(1, 1.0, False, True)]
    count = evaluation_module._render_gpu_trace(
        trace, env, truth, [Agent("source", capabilities={0, 1})],
        world, "/tmp", 1.0, frame_renderer=capture)
    assert count == 2
    assert rendered == [
        ({"source", "hall"}, UNKNOWN_TYPE),
        ({"source", "hall", "target"}, 1),
    ]


def _main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(_main())
