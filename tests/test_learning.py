"""Fast checks for the centralized learned planner."""

import copy
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import yaml

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from learning.policy.candidates import (Candidate, CandidateTerrainCache,
                                        generate_candidates)
from learning.policy.configuration import (
    InstanceConfig,
    LearningConfig,
    load_config,
)
from learning.gpu_sim.routing import GridRouter
from learning.gpu_sim.cugraph_router import CuGraphRouter
from learning.gpu_sim.state import TensorEpisodeState
from learning.policy.model import (
    HeterogeneousGraphPolicy,
    VanillaTransformerPolicy,
    build_policy,
)
from learning.modules import AssignmentDecoder, DecoderOutput
from learning.gpu_sim.observation_cpu import batch_observations, build_observation
from learning.policy.oracle import parallel_tsp
from learning.policy.adapter import LearnedPolicyAdapter
from learning.gpu_sim.rollout_cpu import calculate_episode_return, collect_episode
from learning.gpu_sim.rollout_gpu import DecisionTrace, replay_tensor_gradients
from learning.train import _episode_agent_count, train
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


def _instance(two_agents=True):
    graph = _line()
    graph.nodes[4].update(type="target_unreached", rps_type=1)
    agents = [Agent(0, capabilities={0, 1})]
    if two_agents:
        agents.append(Agent(0, capabilities={2}))
    return graph, agents


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


def test_rendering_uses_distinct_agent_colors_and_stacks_shared_labels():
    colors = [_agent_color(index) for index in range(6)]
    assert len(set(colors)) == 6
    assert _stacked_agent_label_offsets(
        [(1, 2), (3, 4), (1.0, 2.0), (1, 2)]) == [
            (0, 16), (0, 16), (0, 36), (0, 56)]


def _graph_model(num_target_types=2, use_critic=True):
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
    )
    model = HeterogeneousGraphPolicy(config)
    model.eval()
    return model


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
    assert config.candidates.include_wait
    assert config.instances.min_targets == 5
    assert config.instances.max_targets == 9
    assert config.instances.min_agents == 3
    assert config.instances.max_agents == 6
    assert config.training.simulation_batch_size >= 1
    assert config.training.reinforce_batch_size >= 1
    assert config.training.device in {"auto", "cpu", "cuda"}


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
        config.candidates, staging_per_target=0, include_wait=False)

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


def test_task_graph_schema_uses_beliefs_semantics_and_effective_distances():
    graph = _line()
    graph.nodes[4].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    graph.nodes[2]["visible_edges"] = [(3, 4)]
    agent = Agent(0, capabilities={0, 1})
    observation = build_observation(
        graph, [agent], 2,
        transit=[(0, 1, 0.0, 1.0)], clock=0.25,
        replan_transit=True)

    assert observation.task_agent_features.shape[-1] == 5
    assert observation.task_target_features.shape[-1] == 3
    assert observation.task_action_features.shape[-1] == 4
    # Unknown types use a uniform planner belief, never truth or all-zero.
    assert torch.allclose(
        observation.task_target_features[0, 0, 1:],
        torch.tensor([0.5, 0.5]))

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
    distance_scale = sum(float(data["distance"])
                         for _u, _v, data in graph.edges(data=True))
    assert abs(float(observation.task_agent_features[0, 0, 1]) - 0.75) < 1e-6
    # The Transformer keeps its original normalized remaining-time feature.
    assert abs(float(observation.agent_features[0, 0, 8])
               - 0.75 / distance_scale) < 1e-6
    expected = 0.75 + 3.0
    assert abs(float(observation.agent_action_distances[
        0, 0, target_action, 0]) - expected) < 1e-6
    # The preserved Transformer relation still uses its original normalized
    # route feature; only task-graph edge inputs were rolled back to raw time.
    assert abs(float(observation.agent_action_relations[
        0, 0, target_action, 0]) - 3.0 / distance_scale) < 1e-6
    assert observation.serves_mask[0, target_action, target_index]
    assert observation.reveals_mask[0, observation_action, target_index]
    assert not observation.action_target_distance_mask[
        0, wait_action, target_index]


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
    for name in ("agent_features", "agent_mask", "task_agent_features"):
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
    for name in ("target_features", "target_mask", "task_target_features"):
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


def test_cached_candidate_generation_matches_uncached_generation():
    graph, _agents = _instance(False)
    config = load_config().candidates
    uncached = generate_candidates(graph, config)
    cached = generate_candidates(graph, config, CandidateTerrainCache(graph))
    assert [candidate.key for candidate in cached] == [
        candidate.key for candidate in uncached]
    assert [candidate.staging_targets for candidate in cached] == [
        candidate.staging_targets for candidate in uncached]


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
        replan_transit=True)
    target_index = next(i for i, item in enumerate(candidates)
                        if item.is_target)
    assert future.feasible_action_mask[0, 0, target_index]
    # The future route begins at committed arrival node 1, three edges from
    # target 4, rather than being scored from the current edge's source 0.
    distance_scale = sum(
        float(data["distance"]) for _u, _v, data in graph.edges(data=True))
    assert abs(float(future.agent_action_relations[0, 0, target_index, 0])
               - 3.0 / distance_scale) < 1e-6

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
            env, truth, agents, policy, render_dir=directory, render_dt=1.0)
        frames = sorted(Path(directory).glob("frame_*.png"))
    assert rollout.result["completed"]
    assert len(frames) == 3


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
