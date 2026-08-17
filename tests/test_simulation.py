"""Fast regression tests for the generalized capability-based pipeline.

The graph attribute ``rps_type`` remains a compatibility detail; the behavior
under test is general typed-target service, not rock-scissors-paper.
"""

import random

import networkx as nx

from planning.policies import baseline1, baseline2
from simulation.agent import Agent
from simulation.domain import (
    AGENT_DIES,
    AGENT_WINS,
    UNKNOWN_TYPE,
    assign_agent_capabilities,
    assign_target_types,
    init_target_types,
    resolve_encounter,
    validate_capabilities,
)
from simulation.engine import (
    observe_and_reveal,
    resolve_combat_on_arrival,
    run_simulation,
)


def _line(n, dist=1.0):
    graph = nx.DiGraph()
    for node in range(n):
        graph.add_node(node, type="intermediate", pos=(node, 0))
    for node in range(n - 1):
        for u, v in ((node, node + 1), (node + 1, node)):
            graph.add_edge(
                u, v, distance=float(dist), observed_edge=False, num_used=1.0)
    graph.nodes[0]["type"] = "source"
    return graph


def _grid(m, n, dist=1.0):
    graph = nx.grid_2d_graph(m, n, create_using=nx.DiGraph)
    for node in graph.nodes():
        graph.nodes[node].update(type="intermediate", pos=node)
    for u, v in graph.edges():
        graph.edges[u, v].update(
            distance=float(dist), observed_edge=False, num_used=1.0)
    graph.nodes[(0, 0)]["type"] = "source"
    return graph


def _set_targets(graph, target_types):
    for node in target_types:
        graph.nodes[node]["type"] = "target_unreached"


def _full_visibility(graph):
    edges = list(graph.edges())
    for node in graph.nodes():
        graph.nodes[node]["visible_edges"] = list(edges)


def _local_visibility(graph):
    for node in graph.nodes():
        graph.nodes[node]["visible_edges"] = (
            [(node, neighbor) for neighbor in graph.successors(node)]
            + [(neighbor, node) for neighbor in graph.predecessors(node)]
        )


def _set_heights(graph, heights, default=0.0):
    for node in graph.nodes():
        graph.nodes[node]["height"] = float(heights.get(node, default))


def _encounter(capabilities, target_type):
    env = _line(2)
    truth = _line(2)
    _set_targets(env, {1: target_type})
    _set_targets(truth, {1: target_type})
    init_target_types(env, truth, {1: target_type})
    agent = Agent(1, capabilities=capabilities)
    outcome = resolve_combat_on_arrival(env, truth, agent, 0, 1)
    return outcome, env.nodes[1]["type"], agent.alive, env.nodes[1]["rps_type"]


def test_capability_validation():
    assert validate_capabilities({0, 1, 4}, 4) == frozenset({0, 1, 4})
    try:
        validate_capabilities({5}, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range capability was accepted")


def test_binary_encounter_rule():
    assert resolve_encounter({0, 1, 3}, 1) == AGENT_WINS
    assert resolve_encounter({0, 1, 3}, 2) == AGENT_DIES
    assert resolve_encounter({0}, 1) == AGENT_DIES


def test_supported_contact_services_target():
    outcome, node_type, alive, revealed = _encounter({0, 2, 4}, 4)
    assert outcome == AGENT_WINS
    assert node_type == "target_reached"
    assert alive
    assert revealed == 4


def test_unsupported_contact_kills_agent():
    outcome, node_type, alive, revealed = _encounter({0, 2, 4}, 3)
    assert outcome == AGENT_DIES
    assert node_type == "target_unreached"
    assert not alive
    assert revealed == 3


def test_random_target_types_are_reproducible_and_in_range():
    targets = [(0, 1), (2, 3), (4, 5), (6, 7)]
    first = assign_target_types(targets, num_target_types=7, rng=random.Random(9))
    second = assign_target_types(targets, num_target_types=7, rng=random.Random(9))
    assert first == second
    assert all(1 <= value <= 7 for value in first.values())


def test_random_capabilities_are_reproducible_and_cover_types():
    kwargs = dict(
        num_agents=4,
        num_target_types=7,
        capability_probability=0.2,
        scout_probability=0.0,
        ensure_target_coverage=True,
        ensure_scout=True,
    )
    first = assign_agent_capabilities(rng=random.Random(11), **kwargs)
    second = assign_agent_capabilities(rng=random.Random(11), **kwargs)
    assert first == second
    assert all(any(target_type in values for values in first)
               for target_type in range(1, 8))
    assert any(0 in values for values in first)


def test_capability_zero_grants_scouting():
    truth = _line(5)
    _full_visibility(truth)
    _set_targets(truth, {4: 5})
    env = _line(5)
    _set_targets(env, {4: 5})
    init_target_types(env, truth, {4: 5})

    hybrid = Agent(0, capabilities={0, 2, 5})
    _, revealed = observe_and_reveal(env, truth, [hybrid])
    assert revealed == 1
    assert env.nodes[4]["rps_type"] == 5


def test_agent_without_zero_is_blind():
    truth = _line(5)
    _full_visibility(truth)
    _set_targets(truth, {4: 5})
    env = _line(5)
    _set_targets(env, {4: 5})
    init_target_types(env, truth, {4: 5})

    agent = Agent(3, capabilities={5})
    _, revealed = observe_and_reveal(env, truth, [agent])
    assert revealed == 0
    assert env.nodes[4]["rps_type"] == UNKNOWN_TYPE


def test_placeholder_clears_n_type_instance():
    target_types = {
        (0, 4): 1,
        (4, 0): 2,
        (4, 4): 3,
        (2, 2): 4,
    }
    truth = _grid(5, 5)
    _full_visibility(truth)
    _set_targets(truth, target_types)
    env = _grid(5, 5)
    _set_targets(env, target_types)
    init_target_types(env, truth, target_types)

    agents = [
        Agent((0, 0), capabilities={0}),
        Agent((0, 0), capabilities={1, 4}),
        Agent((0, 0), capabilities={2}),
        Agent((0, 0), capabilities={3}),
    ]
    result = run_simulation(env, truth, agents)
    assert result["completed"], result["remaining_targets"]
    assert len(result["eliminated_targets"]) == 4
    assert all(agent.alive for agent in agents)


def test_committed_path_does_not_replan_at_intermediate_nodes():
    truth = _line(6)
    env = _line(6)
    _set_targets(truth, {5: 1})
    _set_targets(env, {5: 1})
    init_target_types(env, truth, {5: 1})
    calls = []

    def committed_policy(_env, active_agents, **_kwargs):
        calls.append(tuple(agent.position for agent in active_agents))
        for agent in active_agents:
            agent.planned_path = list(range(agent.position, 6))

    agent = Agent(0, capabilities={1})
    result = run_simulation(env, truth, [agent], policy=committed_policy)
    assert result["completed"]
    assert result["policy_calls"] == 1
    assert calls == [(0,)]


def test_placeholder_leaves_unsupported_target_incomplete():
    target_types = {(0, 3): 4}
    truth = _grid(2, 4)
    _full_visibility(truth)
    _set_targets(truth, target_types)
    env = _grid(2, 4)
    _set_targets(env, target_types)
    init_target_types(env, truth, target_types)

    agents = [
        Agent((0, 0), capabilities={0}),
        Agent((0, 0), capabilities={1, 2, 3}),
    ]
    result = run_simulation(env, truth, agents)
    assert not result["completed"]
    assert result["remaining_targets"] == [(0, 3)]
    assert all(agent.alive for agent in agents)


def test_baseline_prefers_supported_target():
    env = _grid(5, 5)
    target_types = {(0, 2): 4, (2, 0): 2, (4, 4): 3}
    _set_targets(env, target_types)
    for target, target_type in target_types.items():
        env.nodes[target]["rps_type"] = target_type
    agent = Agent((0, 0), capabilities={2, 3})
    baseline1.replan(env, [agent])
    assert agent.planned_path[-1] == (2, 0)


def test_baseline_unknown_gamble_can_kill_agent():
    target_types = {(0, 3): 4}
    truth = _grid(2, 4)
    _set_targets(truth, target_types)
    env = _grid(2, 4)
    _set_targets(env, target_types)
    init_target_types(env, truth, target_types)

    agent = Agent((0, 0), capabilities={1, 2})
    result = run_simulation(env, truth, [agent], policy=baseline1.replan)
    assert not result["completed"]
    assert not agent.alive
    assert result["remaining_targets"] == [(0, 3)]


def test_pure_scout_baseline_uses_tallest_safe_node():
    env = _grid(5, 5)
    _set_targets(env, {(4, 4): 1})
    env.nodes[(4, 4)]["rps_type"] = UNKNOWN_TYPE
    _set_heights(env, {(4, 4): 10.0, (1, 1): 8.0})
    scout = Agent((0, 0), capabilities={0})
    baseline1.replan(env, [scout])
    assert scout.planned_path[-1] == (1, 1)


def test_baseline2_selects_least_capable_scout_then_releases_it():
    env = _grid(3, 3)
    _local_visibility(env)
    target = (2, 2)
    _set_targets(env, {target: 3})
    env.nodes[target]["rps_type"] = UNKNOWN_TYPE

    broad_scout = Agent((0, 0), capabilities={0, 1, 2})
    narrow_scout = Agent((0, 0), capabilities={0, 3})
    attacker = Agent((0, 0), capabilities={1})
    policy = baseline2.make_policy()

    policy(env, [broad_scout, narrow_scout, attacker])
    assert policy.state["designated_scout"] is narrow_scout
    assert not policy.state["scouting_done"]
    assert narrow_scout.planned_path
    assert narrow_scout.planned_path[-1] != target

    # Once scouting has revealed the last type, the designated hybrid joins
    # the attacker layer and receives the type-3 target only it can service.
    env.nodes[target]["rps_type"] = 3
    policy(env, [broad_scout, narrow_scout, attacker])
    assert policy.state["scouting_done"]
    assert policy.state["claims"][target] is narrow_scout
    assert narrow_scout.planned_path[-1] == target


def test_baseline2_overlapping_capabilities_create_unique_claims():
    env = _grid(3, 3)
    targets = {(0, 2): 1, (2, 0): 1}
    _set_targets(env, targets)
    for target, target_type in targets.items():
        env.nodes[target]["rps_type"] = target_type

    first = Agent((0, 0), capabilities={1})
    second = Agent((0, 0), capabilities={1})
    policy = baseline2.make_policy()
    policy(env, [first, second])

    claims = policy.state["claims"]
    assert set(claims) == set(targets)
    assert len({id(owner) for owner in claims.values()}) == 2


def test_makespan_equals_traversal_cost():
    costs = [2.0, 3.0, 5.0]
    truth = _line(4)
    env = _line(4)
    for index, cost in enumerate(costs):
        for graph in (truth, env):
            graph.edges[index, index + 1]["distance"] = cost
            graph.edges[index + 1, index]["distance"] = cost
            graph.nodes[3]["type"] = "target_unreached"
    init_target_types(env, truth, {3: 7})
    agent = Agent(0, capabilities={7})
    result = run_simulation(env, truth, [agent], policy=baseline1.replan)
    assert result["completed"]
    assert agent.alive
    assert abs(result["makespan"] - sum(costs)) < 1e-9
    assert abs(agent.total_traversal_cost - sum(costs)) < 1e-9


def _main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
