"""
Fast, self-contained unit tests for the Rock-Scissor-Paper mechanics.

Runs on tiny synthetic DiGraphs (matching the real-map schema: directed grid
edges with 'distance'/'observed_edge'/'num_used', nodes with
'type'/'rps_type'/'visible_edges') so there's no DEM, no rendering, no heavy
deps beyond networkx. Run with:

    uv run python Multi_Agent/test_rps.py
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import networkx as nx

from Multi_Agent.finite_horizon_MA import Agent
from Multi_Agent.rps import (
    SCOUT, ROCK, SCISSOR, PAPER, UNKNOWN_TYPE,
    DRAW, AGENT_WINS, AGENT_DIES,
    beats, resolve_encounter, assign_target_types, init_target_types,
)
from Multi_Agent.rps_simulation import (
    observe_and_reveal, resolve_combat_on_arrival, run_rps_simulation,
)


# ---------------------------------------------------------------------------
# Synthetic graph helpers (DiGraph, both directions per adjacency).
# ---------------------------------------------------------------------------

def _line(n, dist=1.0):
    """Path 0-1-...-(n-1) as a DiGraph; node 0 = source, rest intermediate."""
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, type="intermediate", pos=(i, 0))
    for i in range(n - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            G.add_edge(a, b, distance=dist, observed_edge=False, num_used=1.0)
    G.nodes[0]["type"] = "source"
    return G


def _grid(m, n, dist=1.0):
    """m x n grid as a DiGraph with the base schema; (0,0) = source."""
    G = nx.grid_2d_graph(m, n, create_using=nx.DiGraph)
    for node in G.nodes():
        G.nodes[node]["type"] = "intermediate"
        G.nodes[node]["pos"] = node
    for u, v in G.edges():
        G.edges[u, v].update(distance=dist, observed_edge=False, num_used=1.0)
    G.nodes[(0, 0)]["type"] = "source"
    return G


def _set_targets(G, type_map):
    """Mark nodes as targets; type_map = {node: rps_type} (truth)."""
    for node in type_map:
        G.nodes[node]["type"] = "target_unreached"


def _full_visibility(G):
    """God-view visible_edges (every node sees every edge) -- for scouts."""
    all_edges = list(G.edges())
    for node in G.nodes():
        G.nodes[node]["visible_edges"] = list(all_edges)


def _set_heights(G, hmap, default=0.0):
    """Set node 'height' from hmap (node -> height), default elsewhere."""
    for node in G.nodes():
        G.nodes[node]["height"] = float(hmap.get(node, default))


def _local_visibility(G):
    """visible_edges = incident edges only (combat-agent style)."""
    for node in G.nodes():
        G.nodes[node]["visible_edges"] = (
            [(node, v) for v in G.successors(node)]
            + [(u, node) for u in G.predecessors(node)]
        )


# ---------------------------------------------------------------------------
# 1. Pure rules.
# ---------------------------------------------------------------------------

def test_beats_cycle():
    assert beats(ROCK, SCISSOR) and beats(SCISSOR, PAPER) and beats(PAPER, ROCK)
    assert not beats(SCISSOR, ROCK) and not beats(PAPER, SCISSOR) and not beats(ROCK, PAPER)
    for t in (ROCK, SCISSOR, PAPER):
        assert not beats(t, t)  # same type never beats itself


def test_resolve_encounter_table():
    assert resolve_encounter(ROCK, SCISSOR) == AGENT_WINS
    assert resolve_encounter(SCISSOR, PAPER) == AGENT_WINS
    assert resolve_encounter(PAPER, ROCK) == AGENT_WINS
    assert resolve_encounter(ROCK, PAPER) == AGENT_DIES
    assert resolve_encounter(SCISSOR, ROCK) == AGENT_DIES
    assert resolve_encounter(PAPER, SCISSOR) == AGENT_DIES
    for t in (ROCK, SCISSOR, PAPER):
        assert resolve_encounter(t, t) == DRAW
        assert resolve_encounter(SCOUT, t) == AGENT_DIES  # scout always dies


# ---------------------------------------------------------------------------
# 2. Combat resolution mutates state correctly.
# ---------------------------------------------------------------------------

def _combat_case(agent_type, target_type):
    """Run one encounter on a 2-node graph; return (outcome, node_type, alive,
    revealed_type)."""
    env_map = _line(2)
    truth = _line(2)
    _set_targets(env_map, {1: None})
    _set_targets(truth, {1: target_type})
    init_target_types(env_map, truth, {1: target_type})

    agent = Agent(1, agent_type=agent_type)  # already standing on the target
    outcome = resolve_combat_on_arrival(env_map, truth, agent, 0, 1)
    return (outcome, env_map.nodes[1]["type"], agent.alive,
            env_map.nodes[1]["rps_type"])


def test_combat_win_eliminates_target():
    outcome, node_type, alive, revealed = _combat_case(ROCK, SCISSOR)
    assert outcome == AGENT_WINS
    assert node_type == "target_reached"   # eliminated
    assert alive is True                    # agent survives
    assert revealed == SCISSOR              # encounter revealed the type


def test_combat_loss_kills_agent():
    outcome, node_type, alive, revealed = _combat_case(ROCK, PAPER)
    assert outcome == AGENT_DIES
    assert node_type == "target_unreached"  # target untouched
    assert alive is False                   # agent died
    assert revealed == PAPER


def test_combat_draw_changes_nothing():
    outcome, node_type, alive, revealed = _combat_case(ROCK, ROCK)
    assert outcome == DRAW
    assert node_type == "target_unreached"
    assert alive is True
    assert revealed == ROCK


def test_scout_dies_on_target():
    for tt in (ROCK, SCISSOR, PAPER):
        outcome, node_type, alive, _ = _combat_case(SCOUT, tt)
        assert outcome == AGENT_DIES
        assert node_type == "target_unreached"  # scout can't eliminate
        assert alive is False


# ---------------------------------------------------------------------------
# 3. Heterogeneous visibility: scout reveals far, combat reveals only adjacent.
# ---------------------------------------------------------------------------

def test_scout_reveals_distant_type():
    truth = _line(5)
    _full_visibility(truth)               # scout can see the whole line
    _set_targets(truth, {4: SCISSOR})
    env_map = _line(5)
    _set_targets(env_map, {4: None})
    init_target_types(env_map, truth, {4: SCISSOR})

    scout = Agent(0, agent_type=SCOUT)
    _, revealed = observe_and_reveal(env_map, truth, [scout])
    assert revealed == 1
    assert env_map.nodes[4]["rps_type"] == SCISSOR  # revealed from afar


def test_combat_agent_is_blind():
    # Attackers sense nothing -- not even the target right next to them. The
    # only way they learn a type is by stepping onto it (combat), tested above.
    truth = _line(5)
    _full_visibility(truth)               # truth would allow far sight...
    _set_targets(truth, {4: SCISSOR})

    def fresh_env():
        e = _line(5)
        _set_targets(e, {4: None})
        init_target_types(e, truth, {4: SCISSOR})
        return e

    # From far away: nothing revealed.
    env_map = fresh_env()
    rock_far = Agent(0, agent_type=ROCK)
    _, revealed = observe_and_reveal(env_map, truth, [rock_far])
    assert revealed == 0
    assert env_map.nodes[4]["rps_type"] == UNKNOWN_TYPE

    # Even standing adjacent (node 3): still blind, still nothing revealed.
    env_map = fresh_env()
    rock_adj = Agent(3, agent_type=ROCK)
    _, revealed = observe_and_reveal(env_map, truth, [rock_adj])
    assert revealed == 0
    assert env_map.nodes[4]["rps_type"] == UNKNOWN_TYPE


# ---------------------------------------------------------------------------
# 4. End-to-end with the placeholder policy.
# ---------------------------------------------------------------------------

def test_end_to_end_clears_all_targets():
    # 5x5 grid, source at (0,0). Targets spread out (not adjacent) with one of
    # each type plus a second scissor so the single rock agent must clear two
    # in sequence.
    type_map = {
        (0, 4): SCISSOR,   # beaten by rock
        (4, 0): PAPER,     # beaten by scissor
        (4, 4): ROCK,      # beaten by paper
        (2, 2): SCISSOR,   # second scissor -> rock clears two sequentially
    }
    truth = _grid(5, 5)
    _full_visibility(truth)
    _set_targets(truth, type_map)
    env_map = _grid(5, 5)
    _set_targets(env_map, type_map)
    init_target_types(env_map, truth, type_map)

    agents = [
        Agent((0, 0), agent_type=SCOUT),
        Agent((0, 0), agent_type=ROCK),
        Agent((0, 0), agent_type=SCISSOR),
        Agent((0, 0), agent_type=PAPER),
    ]
    result = run_rps_simulation(env_map, truth, agents)

    assert result["completed"], (
        f"not completed; remaining={result['remaining_targets']}")
    assert len(result["eliminated_targets"]) == 4
    assert agents[0].alive, "scout should never die in the placeholder policy"
    assert all(a.alive for a in agents[1:]), "no combat agent should die here"


def test_unbeatable_target_left_incomplete():
    # Only a scout + a rock agent, but the lone target is PAPER (beats rock).
    # The placeholder must NOT throw the rock away: rock idles, scout reveals,
    # nobody can clear it -> run ends incomplete with everyone alive.
    type_map = {(0, 3): PAPER}
    truth = _grid(2, 4)
    _full_visibility(truth)
    _set_targets(truth, type_map)
    env_map = _grid(2, 4)
    _set_targets(env_map, type_map)
    init_target_types(env_map, truth, type_map)

    agents = [Agent((0, 0), agent_type=SCOUT), Agent((0, 0), agent_type=ROCK)]
    result = run_rps_simulation(env_map, truth, agents)

    assert not result["completed"]
    assert result["remaining_targets"] == [(0, 3)]
    assert all(a.alive for a in agents), "nobody should die: rock never engages paper"


def test_assign_target_types_reproducible():
    import random
    random.seed(7)
    a = assign_target_types([(0, 1), (2, 3), (4, 5)])
    random.seed(7)
    b = assign_target_types([(0, 1), (2, 3), (4, 5)])
    assert a == b
    assert all(v in (ROCK, SCISSOR, PAPER) for v in a.values())


# ---------------------------------------------------------------------------
# 5. Baseline 1 policy (attacker: win>draw>unknown>lose; scout: tallest point).
# ---------------------------------------------------------------------------

from Multi_Agent import baseline1_all_key as b1


def _env_with_targets(type_map, heights=None):
    env = _grid(5, 5)
    _set_targets(env, type_map)
    truth = _grid(5, 5)
    _set_targets(truth, type_map)
    init_target_types(env, truth, type_map)
    # reveal the given types directly on env_map so the policy can categorise.
    for t, tt in type_map.items():
        env.nodes[t]["rps_type"] = tt
    if heights:
        _set_heights(env, heights)
    return env


def test_baseline1_attacker_prefers_win():
    # rock attacker: scissor=win, rock=draw, paper=lose, plus an unknown.
    env = _env_with_targets({(0, 2): SCISSOR, (2, 0): ROCK, (4, 4): PAPER})
    env.nodes[(0, 4)]["type"] = "target_unreached"
    env.nodes[(0, 4)]["rps_type"] = UNKNOWN_TYPE
    rock = Agent((0, 0), agent_type=ROCK)
    b1.replan(env, [rock])
    assert rock.planned_path[-1] == (0, 2)  # engaged the win target


def test_baseline1_closest_within_category():
    # two scissor (win) targets; rock must pick the nearer one.
    env = _env_with_targets({(0, 2): SCISSOR, (0, 4): SCISSOR})
    rock = Agent((0, 0), agent_type=ROCK)
    b1.replan(env, [rock])
    assert rock.planned_path[-1] == (0, 2)


def test_baseline1_falls_through_to_draw():
    # no win available; rock has a draw and a lose -> picks draw and engages.
    env = _env_with_targets({(2, 0): ROCK, (4, 4): PAPER})
    rock = Agent((0, 0), agent_type=ROCK)
    b1.replan(env, [rock])
    assert rock.planned_path[-1] == (2, 0)  # draw target, engaged


def test_baseline1_engages_unknown_literally():
    # only an unknown and a lose; rock prefers unknown and walks ALL the way
    # onto it (blind gamble), not stopping short.
    env = _env_with_targets({(4, 4): PAPER})
    env.nodes[(0, 4)]["type"] = "target_unreached"
    env.nodes[(0, 4)]["rps_type"] = UNKNOWN_TYPE
    rock = Agent((0, 0), agent_type=ROCK)
    b1.replan(env, [rock])
    assert rock.planned_path[-1] == (0, 4)  # steps onto the unknown


def test_baseline1_engages_lose_literally():
    # rock with only a paper (lose) target: it still goes and engages (and will
    # die) -- intended weakness, no special-casing.
    env = _env_with_targets({(4, 4): PAPER})
    rock = Agent((0, 0), agent_type=ROCK)
    b1.replan(env, [rock])
    assert rock.planned_path[-1] == (4, 4)


def test_baseline1_blind_attacker_dies_charging_unknown():
    # No scout, so nothing reveals the type. A blind rock charges the unknown
    # target, which is paper -> it dies, the target survives, run incomplete.
    type_map = {(0, 3): PAPER}
    truth = _grid(2, 4)
    _set_targets(truth, type_map)            # no _full_visibility: no observer
    env = _grid(2, 4)
    _set_targets(env, type_map)
    init_target_types(env, truth, type_map)

    rock = Agent((0, 0), agent_type=ROCK)
    result = run_rps_simulation(env, truth, [rock], policy=b1.replan)
    assert not rock.alive                     # died on the blind gamble
    assert not result["completed"]
    assert result["remaining_targets"] == [(0, 3)]


def test_baseline1_scout_goes_to_tallest():
    env = _env_with_targets({(4, 4): ROCK}, heights={(3, 1): 9.0})
    scout = Agent((0, 0), agent_type=SCOUT)
    b1.replan(env, [scout])
    assert scout.planned_path[-1] == (3, 1)  # tallest non-target node


def test_baseline1_scout_avoids_target_as_tallest():
    # tallest cell is a live target -> scout must NOT pick it (would die).
    env = _env_with_targets({(4, 4): ROCK}, heights={(4, 4): 9.0, (1, 1): 5.0})
    scout = Agent((0, 0), agent_type=SCOUT)
    b1.replan(env, [scout])
    assert scout.planned_path[-1] == (1, 1)  # next-tallest, non-target


def test_baseline1_end_to_end():
    type_map = {(0, 4): SCISSOR, (4, 0): PAPER, (4, 4): ROCK, (2, 2): SCISSOR}
    truth = _grid(5, 5)
    _full_visibility(truth)
    _set_targets(truth, type_map)
    env = _grid(5, 5)
    _set_targets(env, type_map)
    _set_heights(env, {(3, 3): 9.0})   # scout vantage (non-target)
    init_target_types(env, truth, type_map)

    agents = [Agent((0, 0), agent_type=t) for t in (SCOUT, ROCK, SCISSOR, PAPER)]
    result = run_rps_simulation(env, truth, agents, policy=b1.replan)
    assert result["completed"], result["remaining_targets"]
    assert agents[0].alive                       # scout survives
    assert all(a.alive for a in agents[1:])      # no attacker dies
    assert result["makespan"] > 0                # continuous-time completion


# ---------------------------------------------------------------------------
# 6. Continuous time: makespan reflects traversal cost.
# ---------------------------------------------------------------------------

def _weighted_line(costs):
    """Line 0..len(costs) with per-edge distances = costs; node 0 = source."""
    G = nx.DiGraph()
    for i in range(len(costs) + 1):
        G.add_node(i, type="intermediate", pos=(i, 0))
    for i, c in enumerate(costs):
        for a, b in ((i, i + 1), (i + 1, i)):
            G.add_edge(a, b, distance=float(c), observed_edge=False, num_used=1.0)
    G.nodes[0]["type"] = "source"
    return G


def test_makespan_equals_traversal_cost():
    # Edge costs 2,3,5; a scissor target at node 3; a rock attacker (rock beats
    # scissor). The rock charges (blind, unknown) and wins at node 3. Makespan
    # must equal the summed edge costs, i.e. time == cost.
    costs = [2.0, 3.0, 5.0]
    truth = _weighted_line(costs)
    env = _weighted_line(costs)
    for g in (truth, env):
        g.nodes[3]["type"] = "target_unreached"
    init_target_types(env, truth, {3: SCISSOR})
    rock = Agent(0, agent_type=ROCK)
    result = run_rps_simulation(env, truth, [rock], policy=b1.replan)
    assert result["completed"]
    assert rock.alive
    assert abs(result["makespan"] - sum(costs)) < 1e-9   # 2+3+5 = 10
    assert abs(rock.total_traversal_cost - sum(costs)) < 1e-9


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------

def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
