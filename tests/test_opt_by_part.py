"""Synthetic regressions for concurrent scouting and known-task execution."""

import json
import random
from unittest.mock import patch

import networkx as nx

from learning.test import evaluate
from planning.full_information import solve_full_information
from planning.policies.opt_by_part import (
    OptByPartPolicy,
    _select_hybrid_executors,
    make_policy,
)
from planning.policies.scout_then_execute import ScoutThenExecutePolicy
from simulation.agent import Agent
from simulation.domain import UNKNOWN_TYPE
from simulation.engine import run_simulation


def _star(known=1, unknown=2):
    graph = nx.DiGraph()
    graph.add_node("s", type="source", pos=(0, 0), visible_nodes=[])
    for index in range(known + unknown):
        witness, target = f"w{index}", f"t{index}"
        graph.add_node(witness, type="intermediate", pos=(index, 1),
                       visible_nodes=[target])
        graph.add_node(target, type="target_unreached", pos=(index, 2),
                       rps_type=1 if index < known else UNKNOWN_TYPE,
                       visible_nodes=[])
        graph.add_edge("s", witness, distance=float(index + 1))
        graph.add_edge(witness, "s", distance=float(index + 2))
        graph.add_edge(witness, target, distance=1.0)
        graph.add_edge(target, witness, distance=2.0)
    return graph


def test_hybrid_roles_follow_capability_gap_and_ignore_unknown_count():
    agents = [Agent("s", capabilities=values) for values in (
        {0}, {1}, {0, 1}, set(),
    )]
    agents.append(Agent("s", capabilities={0, 1}))
    agents[-1].alive = False
    policy = make_policy()
    policy(_star(known=1, unknown=2), agents)
    assert policy.diagnostics["agent_roles"] == [
        "scout", "execute", "scout", "wait", "wait",
    ]
    assert set(policy.diagnostics["scout_agent_indices"]) == {0, 2}
    assert agents[1].planned_path[-1] == "t0"

    for known, unknown in ((2, 1), (2, 0), (2, 3)):
        policy(_star(known, unknown), agents)
        assert policy.diagnostics["agent_roles"] == [
            "scout" if unknown else "wait", "execute", "execute", "wait", "wait",
        ]
        assert policy.diagnostics["known_capability_demand"] == {1: 2}
        assert policy.diagnostics["service_only_capability_supply"] == {1: 1}
        assert policy.diagnostics["initial_capability_gap"] == {1: 1}
        assert policy.diagnostics["remaining_capability_gap"] == {}

    policy(_star(known=1, unknown=1), agents)
    assert policy.diagnostics["hybrid_executor_indices"] == []
    assert policy.diagnostics["agent_roles"][2] == "scout"
    json.dumps(policy.diagnostics)


def test_capability_tally_subtracts_service_agents_and_prefers_largest_gap_fill():
    env = _star(known=4, unknown=1)
    for target, target_type in zip(("t0", "t1", "t2", "t3"), (1, 1, 2, 3)):
        env.nodes[target]["rps_type"] = target_type
    agents = [Agent("s", capabilities=values) for values in (
        {1, 3}, {0, 1}, {0, 1, 2}, {0, 2}, {0, 3}, {2},
    )]
    agents[-1].alive = False  # Dead service capacity must not reduce the gap.
    policy = OptByPartPolicy()
    policy(env, agents)
    diagnostics = policy.diagnostics
    assert diagnostics["known_capability_demand"] == {1: 2, 2: 1, 3: 1}
    assert diagnostics["service_only_capability_supply"] == {1: 1, 3: 1}
    assert diagnostics["initial_capability_gap"] == {1: 1, 2: 1}
    assert diagnostics["remaining_capability_gap"] == {}
    assert diagnostics["hybrid_executor_indices"] == [2]
    assert diagnostics["agent_roles"] == [
        "execute", "scout", "execute", "scout", "scout", "wait",
    ]
    assert diagnostics["scouting_feasible"] and diagnostics["execution_feasible"]
    assert all(len(agents[index].planned_path) > 1 for index in (0, 2))


def test_repeated_type_demand_recruits_only_needed_hybrids():
    env = _star(known=3, unknown=1)
    agents = [Agent("s", capabilities={0, 1}) for _ in range(4)]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert policy.diagnostics["initial_capability_gap"] == {1: 3}
    assert len(policy.diagnostics["hybrid_executor_indices"]) == 3
    assert policy.diagnostics["agent_roles"].count("execute") == 3
    assert policy.diagnostics["agent_roles"].count("scout") == 1

    env.nodes["t0"]["type"] = "target_reached"
    policy(env, agents)
    assert policy.diagnostics["known_capability_demand"] == {1: 2}
    assert len(policy.diagnostics["hybrid_executor_indices"]) == 2
    assert policy.diagnostics["agent_roles"].count("execute") == 2
    assert policy.diagnostics["agent_roles"].count("scout") == 2


def test_random_tiebreak_selects_only_hybrids_with_maximum_contribution():
    env = _star(known=1, unknown=1)
    agents = [Agent("s", capabilities=values)
              for values in ({0, 1}, {0, 2}, {0, 1})]
    policy = OptByPartPolicy()
    with patch("planning.policies.opt_by_part.random.choice", return_value=2) as choose:
        policy(env, agents)
    choose.assert_called_once_with([0, 2])
    assert policy.diagnostics["hybrid_executor_indices"] == [2]
    assert policy.diagnostics["agent_roles"] == ["scout", "scout", "execute"]


def test_random_hybrid_selection_is_reproducible_with_evaluation_seed():
    env = _star(known=2, unknown=1)
    agents = [Agent("s", capabilities={0, 1}) for _ in range(4)]
    rng_state = random.getstate()
    try:
        random.seed(42)
        first = _select_hybrid_executors(env, agents, {"t0", "t1"})
        random.seed(42)
        second = _select_hybrid_executors(env, agents, {"t0", "t1"})
    finally:
        random.setstate(rng_state)
    assert first == second
    assert len(first[0]) == 2


def test_unfillable_capability_counts_do_not_block_supported_execution_or_scouts():
    env = _star(known=3, unknown=1)
    env.nodes["t2"]["rps_type"] = 2
    agents = [Agent("s", capabilities={0, 1}), Agent("s", capabilities={0, 3})]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert policy.diagnostics["hybrid_executor_indices"] == [0]
    assert policy.diagnostics["remaining_capability_gap"] == {1: 1, 2: 1}
    assert policy.diagnostics["deferred_known_targets"] == ["t2"]
    assert policy.diagnostics["agent_roles"] == ["execute", "scout"]
    assert policy.diagnostics["execution_feasible"]
    assert len(agents[0].planned_path) > 1 and len(agents[1].planned_path) > 1


def test_hybrid_only_team_keeps_scouting_while_servicing_known_tasks():
    env = _star(known=1, unknown=2)
    truth = env.copy()
    for target in ("t0", "t1", "t2"):
        truth.nodes[target]["rps_type"] = 1
    agents = [Agent("s", capabilities={0, 1}) for _ in range(3)]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert policy.diagnostics["agent_roles"].count("execute") == 1
    assert policy.diagnostics["agent_roles"].count("scout") == 2
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    first_service = min(event["time"] for event in result["events"]
                        if event["event"] == "agent_wins")
    final_reveal = max(event["time"] for event in result["events"]
                       if event["event"] == "reveal")
    assert first_service < final_reveal


def test_services_known_task_before_scouting_finishes():
    def instance():
        env = _star(known=0, unknown=2)
        truth = env.copy()
        for target in ("t0", "t1"):
            truth.nodes[target]["rps_type"] = 1
        return env, truth, [Agent("s", capabilities={0}),
                            Agent("s", capabilities={1})]

    env, truth, agents = instance()
    policy = OptByPartPolicy()
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    first_service = min(event["time"] for event in result["events"]
                        if event["event"] == "agent_wins")
    final_reveal = max(event["time"] for event in result["events"]
                       if event["event"] == "reveal")
    assert first_service < final_reveal

    env, truth, agents = instance()
    strict = ScoutThenExecutePolicy()
    result = run_simulation(env, truth, agents, policy=strict)
    assert result["completed"]
    assert all(event["time"] >= strict.diagnostics["scouting_completion_time"]
               for event in result["events"] if event["event"] == "agent_wins")


def test_execution_blocks_unknown_targets_and_preserves_beliefs():
    env = _star(known=1, unknown=1)
    # An unknown target offers an unsafe shortcut to the known target.
    env.add_edge("s", "t1", distance=0.1)
    env.add_edge("t1", "t0", distance=0.1)
    before = {node: dict(data) for node, data in env.nodes(data=True)}
    agents = [Agent("s", capabilities={0}), Agent("s", capabilities={1})]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert agents[1].planned_path == ["s", "w0", "t0"]
    assert "t1" not in agents[1].planned_path
    assert {node: dict(data) for node, data in env.nodes(data=True)} == before

    truth = env.copy()
    truth.nodes["t1"]["rps_type"] = 1
    result = run_simulation(env, truth, agents, policy=OptByPartPolicy())
    assert result["completed"] and not result["deaths"]


def test_hybrid_fills_missing_service_type_while_unused_hybrid_scouts():
    env = _star(known=1, unknown=2)
    env.nodes["t0"]["rps_type"] = 2
    agents = [Agent("s", capabilities={0, 2}),
              Agent("s", capabilities={1}), Agent("s", capabilities={0, 1})]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert policy.diagnostics["deferred_known_targets"] == []
    assert policy.diagnostics["hybrid_executor_indices"] == [0]
    assert policy.diagnostics["agent_roles"] == ["execute", "execute", "scout"]
    assert policy.diagnostics["scouting_feasible"]
    assert agents[0].planned_path[-1] == "t0"
    assert agents[1].planned_path == ["s"]
    assert len(agents[2].planned_path) > 1

    truth = env.copy()
    for target in ("t1", "t2"):
        truth.nodes[target]["rps_type"] = 1
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]


def test_unscoutable_tasks_do_not_block_known_execution():
    env = _star(known=1, unknown=1)
    env.nodes["w1"]["visible_nodes"] = []
    agents = [Agent("s", capabilities={0}), Agent("s", capabilities={1})]
    truth = env.copy()
    truth.nodes["t1"]["rps_type"] = 1
    policy = OptByPartPolicy()
    result = run_simulation(env, truth, agents, policy=policy)
    assert not result["completed"] and not result["deaths"]
    assert result["remaining_targets"] == ["t1"]
    assert not policy.diagnostics["scouting_feasible"]
    assert "t0" in agents[1].trajectory


def test_execution_can_open_a_previously_infeasible_scout_route():
    env = nx.DiGraph()
    for index, node in enumerate(("s", "known", "witness", "unknown")):
        env.add_node(node, type="intermediate", pos=(index, 0), visible_nodes=[])
    env.nodes["s"]["type"] = "source"
    env.nodes["known"].update(type="target_unreached", rps_type=1)
    env.nodes["unknown"].update(type="target_unreached", rps_type=UNKNOWN_TYPE)
    env.nodes["witness"]["visible_nodes"] = ["unknown"]
    for left, right in (("s", "known"), ("known", "witness"),
                        ("witness", "unknown")):
        env.add_edge(left, right, distance=1.0)
        env.add_edge(right, left, distance=2.0)
    agents = [Agent("s", capabilities={0}), Agent("s", capabilities={1})]
    policy = OptByPartPolicy()
    policy(env, agents)
    assert not policy.diagnostics["scouting_feasible"]
    assert agents[1].planned_path == ["s", "known"]
    truth = env.copy()
    truth.nodes["unknown"]["rps_type"] = 1
    result = run_simulation(env, truth, agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    assert "witness" in agents[0].trajectory


def test_transit_role_changes_keep_destination_and_release_offset():
    env = _star(known=1, unknown=2)
    agents = [Agent("s", capabilities={1}), Agent("s", capabilities={0, 1})]
    policy = OptByPartPolicy()
    policy.set_runtime_state(agents, [None, ("s", "w0", 0.0, 3.0)], 1.0)
    policy(env, agents)
    assert policy.diagnostics["agent_roles"][1] == "scout"
    assert agents[1].planned_path[0] == "w0"

    env.nodes["t1"]["rps_type"] = 1  # Two type-1 tasks minus one service agent.
    policy(env, agents)
    assert policy.diagnostics["agent_roles"][1] == "execute"
    assert agents[1].planned_path[0] == "w0"
    expected = solve_full_information(
        env.subgraph([node for node in env if node != "t2"]), agents,
        start_nodes=["s", "w0"], release_times=[0.0, 2.0],
    )
    assert agents[1].planned_path == list(expected.paths[1])
    assert policy.diagnostics["execution_makespan_estimate"] == expected.makespan

    # Reassigning back to scouting cannot cancel an incoming service encounter.
    env.nodes["t1"]["rps_type"] = UNKNOWN_TYPE
    policy.set_runtime_state(agents, [None, ("w0", "t0", 0.0, 3.0)], 1.0)
    policy(env, agents)
    assert policy.diagnostics["scouts_finishing_committed_encounters"] == [1]
    assert not policy.diagnostics["scouting_feasible"]
    assert agents[1].planned_path == ["t0"]


def test_all_known_matches_fi_opt_and_targetless_waits():
    env = _star(known=2, unknown=0)
    agents = [Agent("s", capabilities={0, 1}), Agent("s", capabilities={1})]
    expected = solve_full_information(env, agents)
    policy = OptByPartPolicy()
    policy(env, agents)
    assert tuple(tuple(agent.planned_path) for agent in agents) == expected.paths
    result = run_simulation(env, env.copy(), agents, policy=policy)
    assert result["completed"] and not result["deaths"]
    assert result["makespan"] == expected.makespan

    OptByPartPolicy()(_star(known=0, unknown=0),
                      agents := [Agent("s", capabilities={0, 1})])
    assert agents[0].planned_path == ["s"]


def test_classical_evaluation_accepts_opt_by_part_without_checkpoint():
    def factory(_case, map_path=None):
        env = _star(known=0, unknown=2)
        truth = env.copy()
        for target in ("t0", "t1"):
            truth.nodes[target]["rps_type"] = 1
        return env, truth, [Agent("s", capabilities={0}), Agent("s", capabilities={1})]

    with patch("learning.test._case_factory", side_effect=factory), \
            patch("torch.load", side_effect=AssertionError("loaded weights")):
        records, summary, _config, weights, _suite, _path = evaluate(
            None, suite="development", limit=1, policy="opt-by-part", episodes=2,
            device="cuda")
    assert weights is None
    assert len(records) == 2 and summary["completion_rate"] == 1.0
    for record in records:
        assert record["simulation_backend"] == "cpu_classical"
        assert record["planning_call_count"] > 1
        assert record["policy_diagnostics"]["hybrid_assignment_rule"] == "capability_gap"


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
