"""Fast checks for the centralized learned planner."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace

import networkx as nx
import torch

from learning.candidates import (Candidate, CandidateTerrainCache,
                                 generate_candidates)
from learning.configuration import load_config
from learning.decoder import AssignmentDecoder, DecoderOutput
from learning.gpu.routing import GridRouter
from learning.gpu.state import TensorEpisodeState
from learning.model import CentralizedPolicy
from learning.observation import batch_observations, build_observation
from learning.policy_adapter import LearnedPolicyAdapter
from simulation.agent import Agent
from simulation.domain import UNKNOWN_TYPE, init_target_types
from simulation.engine import run_simulation


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
    model = CentralizedPolicy(config)
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
    assert config.candidates.include_continue
    assert config.training.simulation_batch_size >= 1
    assert config.training.reinforce_batch_size >= 1
    assert config.training.device in {"auto", "cpu", "cuda"}


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
    continue_index = next(i for i, c in enumerate(candidates) if c.is_continue)
    assert moving.feasible_action_mask.sum() == 1
    assert moving.feasible_action_mask[0, 0, continue_index]

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
