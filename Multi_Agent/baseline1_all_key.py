"""
Baseline 1 for the rock_scissor_paper direction.

(The file name ``baseline1_all_key`` is legacy from the fixed_key lineage; this
module no longer has anything to do with keys.)

A simple, uncoordinated, type-aware policy that routes the two agent roles
differently:

* **Attackers** (rock/scissor/paper) choose a target by category preference

      win  >  draw  >  unknown  >  lose

  where the category is the outcome the attacker would get against that
  target's (possibly still-hidden) type. Ties within a category are broken by
  the **closest** target (shortest safe-travel distance). The best non-empty
  category that has a *reachable* target wins; otherwise the next category is
  tried.

* **Scouts** ignore targets entirely and head for the **tallest point on the
  map** (the node with the greatest ``height``) -- the best single vantage,
  since visibility here is height-gated, so a high observer sees (and reveals)
  the most.

Engagement is LITERAL -- attackers really go to their chosen target and engage
it, whatever its type:
  * win     -> eliminate the target.
  * draw    -> harmless no-op (same type).
  * unknown -> a gamble. Attackers are BLIND (they sense nothing), so the only
               way to learn an unknown type is to step on it: the attacker may
               win, draw, or DIE. Dying on a bad unknown is an intended
               weakness of this deliberately weak baseline.
  * lose    -> the attacker dies. (It still goes to the lowest-preference
               target if nothing better is reachable.)
Routes still go *around* all OTHER live targets so the attacker reaches its
chosen target rather than dying on a different one en route. Scouts (which also
die on targets) keep routing around every live target.

Use it as the simulation policy:

    from Multi_Agent import baseline1_all_key
    run_rps_simulation(env_map, ground_truth, agents, policy=baseline1_all_key.replan)
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx

from Multi_Agent.rps import SCOUT, UNKNOWN_TYPE, beats, TYPE_NAMES


# Preference order over encounter categories (most preferred first).
ATTACKER_PREFERENCE = ("win", "draw", "unknown", "lose")


# ---------------------------------------------------------------------------
# Small self-contained graph helpers (the repo's baselines are deliberately
# standalone; these mirror the ones in rps_simulation).
# ---------------------------------------------------------------------------

def _path_distance(graph, path):
    """Sum of edge 'distance' along ``path`` (missing edges skipped)."""
    return sum(
        graph.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
        if graph.has_edge(path[k], path[k + 1])
    )


def _safe_path(graph, src, goal, avoid):
    """Shortest ``src``->``goal`` path that avoids the ``avoid`` nodes.

    Routes on a copy with ``avoid`` removed (never removing ``src``/``goal``)
    so the path never crosses an unwanted target. Returns ``(path, cost)`` or
    ``(None, inf)`` if unreachable.
    """
    if src == goal:
        return [src], 0.0
    g = graph
    rm = [n for n in avoid if n != src and n != goal]
    if rm:
        g = graph.copy()
        g.remove_nodes_from(rm)
    try:
        path = nx.shortest_path(g, src, goal, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")
    return path, _path_distance(g, path)


def _category(attacker_type, target_type):
    """Encounter category for ``attacker_type`` vs a target's ``target_type``."""
    if target_type == UNKNOWN_TYPE:
        return "unknown"
    if target_type == attacker_type:
        return "draw"
    if beats(attacker_type, target_type):
        return "win"
    return "lose"


def _tallest_point(env_map, exclude):
    """Node with the greatest ``height`` that is not in ``exclude`` (the live
    targets). Returns None if no node carries a height."""
    best, best_h = None, float("-inf")
    for n, d in env_map.nodes(data=True):
        if n in exclude:
            continue
        h = d.get("height")
        if h is not None and h > best_h:
            best, best_h = n, h
    return best


# ---------------------------------------------------------------------------
# Per-role routing.
# ---------------------------------------------------------------------------

def _route_attacker(env_map, agent, live_targets, live_set, verbose):
    """Pick a target by category preference, closest within the chosen
    category, and route the attacker to ENGAGE it (step onto it).

    Engagement is literal for every category: a win eliminates, a draw is a
    no-op, an unknown is a blind gamble, and a lose is fatal -- the attacker
    commits regardless (this is a deliberately weak baseline)."""
    buckets = {c: [] for c in ATTACKER_PREFERENCE}
    for t in live_targets:
        tt = env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)
        buckets[_category(agent.agent_type, tt)].append(t)

    for cat in ATTACKER_PREFERENCE:
        best_t, best_path, best_cost = None, None, float("inf")
        for t in buckets[cat]:
            path, cost = _safe_path(env_map, agent.position, t, live_set - {t})
            if path is not None and cost < best_cost:
                best_t, best_path, best_cost = t, path, cost
        if best_t is None:
            continue  # nothing reachable in this category; drop to the next

        agent.planned_path = best_path  # walk all the way onto the target
        if verbose:
            print(f"  [b1] {TYPE_NAMES[agent.agent_type]} @ {agent.position} "
                  f"-> engage {cat} target {best_t} (cost {best_cost:.2f})")
        return
    # no reachable target in any category -> idle


def _route_scout(env_map, scout, tallest, live_set, verbose):
    """Send the scout to the tallest point, routing around all live targets."""
    if tallest is None or scout.position == tallest:
        return  # nowhere to go / already perched -> stay and keep watching
    path, _cost = _safe_path(env_map, scout.position, tallest, live_set)
    if path is not None and len(path) >= 2:
        scout.planned_path = path
        if verbose:
            print(f"  [b1] scout @ {scout.position} -> tallest point {tallest} "
                  f"(h={env_map.nodes[tallest].get('height')})")


# ---------------------------------------------------------------------------
# Policy entry point (matches the run_rps_simulation policy signature).
# ---------------------------------------------------------------------------

def replan(env_map: nx.Graph, agents, reward_ratio=1.0, obs_discount_factor=1.0,
           sample_recursion=0, sample_num_obstacle=0, sample_obstacle_hop=0,
           verbose=False):
    """Assign every living agent a planned_path under the baseline-1 rules.

    The reward / sampling kwargs are accepted for interface compatibility with
    reward-driven policies but ignored -- this baseline scores by distance.
    """
    for a in agents:
        a.planned_path = []

    live_targets = [n for n, d in env_map.nodes(data=True)
                    if d.get("type") == "target_unreached"]
    live_set = set(live_targets)
    tallest = _tallest_point(env_map, exclude=live_set)

    if verbose:
        print("=" * 60)
        print("baseline1 replan (attacker: win>draw>unknown>lose; scout: tallest)")

    for agent in agents:
        if not agent.alive:
            continue
        if agent.agent_type == SCOUT:
            _route_scout(env_map, agent, tallest, live_set, verbose)
        else:
            _route_attacker(env_map, agent, live_targets, live_set, verbose)

    if verbose:
        print("=" * 60)
