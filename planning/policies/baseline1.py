"""Baseline 1: independent nearest-target routing with a static scout vantage.

This deliberately simple baseline makes no joint assignment and does not wait
for information before taking risk.

Service-capable agents
----------------------
Every agent with at least one positive capability ranks live targets as:

    supported > unknown > unsupported

It selects the closest reachable target in the first non-empty category and
walks all the way onto it. Contact is literal: a supported target is serviced,
while an unsupported target kills the agent. An unknown target is therefore a
blind gamble. Routes avoid every *other* live target so an agent cannot trigger
an unintended contact on the way to its selected target.

Pure scouts
-----------
An agent whose only capability is ``0`` routes to the highest non-target node
and stays there. It avoids all live targets because capability ``0`` grants
observation but does not service any target type.

Hybrid agents
-------------
Agents with capability ``0`` plus positive target capabilities follow the
service-agent rule. Their scouting capability still applies automatically at
every node they visit, but baseline 1 does not plan a dedicated scouting tour
for them.

Usage::

    from planning.policies import baseline1
    from simulation.engine import run_simulation
    run_simulation(env_map, ground_truth, agents, policy=baseline1.replan)
"""

import networkx as nx

from simulation.domain import UNKNOWN_TYPE, capability_label


# Preference order over encounter categories (most preferred first).
ATTACKER_PREFERENCE = ("supported", "unknown", "unsupported")


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


def _category(agent, target_type):
    """Encounter category for an agent and a known/unknown target type."""
    if target_type == UNKNOWN_TYPE:
        return "unknown"
    return "supported" if agent.can_service(target_type) else "unsupported"


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

    Engagement is literal for every category: supported contact succeeds,
    unsupported contact is fatal, and unknown contact is a blind gamble."""
    buckets = {c: [] for c in ATTACKER_PREFERENCE}
    for t in live_targets:
        tt = env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)
        buckets[_category(agent, tt)].append(t)

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
            print(f"  [b1] {capability_label(agent.capabilities)} @ {agent.position} "
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
# Policy entry point (matches the simulation engine's policy signature).
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
        print("baseline1 replan "
              "(agent: supported>unknown>unsupported; pure scout: tallest)")

    for agent in agents:
        if not agent.alive:
            continue
        if agent.scout_capable and not agent.can_engage:
            _route_scout(env_map, agent, tallest, live_set, verbose)
        elif agent.can_engage:
            _route_attacker(env_map, agent, live_targets, live_set, verbose)

    if verbose:
        print("=" * 60)
