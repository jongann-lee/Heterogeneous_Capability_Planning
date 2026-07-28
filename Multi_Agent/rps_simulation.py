"""
Map-agnostic Rock-Scissor-Paper simulation core.

This is the continuous-time, discrete-event driver for the `rock_scissor_paper`
direction. It is deliberately decoupled from the real-DEM benchmark so the
mechanics can be exercised on a tiny synthetic graph in a fast unit test (see
test_rps.py). The real-map entry point (rps_real_map.py) just builds the two
graphs and hands them here.

Two-graph partial observability (same contract as the base multi-agent sim):
  * ``env_map``      -- the planner's view. Targets start with
                        ``rps_type = UNKNOWN_TYPE``; the planner only learns a
                        target's type by *observing* it.
  * ``ground_truth`` -- reality: true edge set (obstacles removed), true
                        ``rps_type`` on every target, and ``visible_edges``
                        pruned to the true edges.

What this module adds on top of the base loop:
  1. **Heterogeneous visibility by agent type.** A SCOUT senses its full
     polytope ``visible_edges`` and is the ONLY observer. Combat agents
     (rock/scissor/paper) are **blind** -- they sense nothing, so they neither
     discover blockages nor reveal target types by looking. An attacker learns
     a target's type only by stepping onto it (combat always reveals).
  2. **Combat on contact.** When an agent steps onto a live target the
     encounter is resolved by rps.resolve_encounter: a winning combat agent
     eliminates the target, a losing one dies, a draw leaves both in place,
     and a scout always dies. Every encounter reveals the target's type.
  3. **Deaths.** A dead agent is removed from play for the rest of the run.
  4. **Continuous time (discrete-event simulation).** An edge of cost c takes
     time c to traverse; the loop jumps between agent-arrival *events* via a
     time-ordered heap instead of stepping fixed ticks, so traversal cost is
     reflected directly in elapsed time and *waiting is no longer free*. The
     objective is the **makespan** (clock time of the last elimination) plus a
     death penalty. An agent commits to its current edge and is re-planned when
     it reaches the next node; observation happens at node arrivals.

The assignment policy is pluggable: pass any ``policy(env_map, agents, ...)``
that sets each agent's ``planned_path``. On each event the policy is called
with the living, *at-a-node* agents only (in-transit agents are committed to
their current edge until they arrive). ``naive_type_aware_replan`` below is a
minimal SAFE placeholder so the loop runs; replace it with the real baseline.
"""

import sys
import os
import heapq

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import networkx as nx

from Multi_Agent.rps import (
    SCOUT, UNKNOWN_TYPE, DRAW, AGENT_WINS, AGENT_DIES,
    beats, resolve_encounter, TYPE_NAMES,
)


# ---------------------------------------------------------------------------
# Edge / path helpers (direction-agnostic; the real graph is a DiGraph with
# both orientations present for every grid adjacency).
# ---------------------------------------------------------------------------

def _ekey(u, v):
    """Direction-agnostic key for the edge between ``u`` and ``v``."""
    return frozenset((u, v))


def _path_distance(graph, path):
    """Sum of edge 'distance' along ``path`` (missing edges skipped)."""
    return sum(
        graph.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
        if graph.has_edge(path[k], path[k + 1])
    )


# ---------------------------------------------------------------------------
# Type-aware sensing.
# ---------------------------------------------------------------------------

def sensed_edges_truth(ground_truth, agent):
    """Edges the agent ACTUALLY senses from its position, by type.

    SCOUT  -> full polytope ``visible_edges`` (terrain-limited, long range).
    combat -> nothing: attackers are BLIND. They learn a target's type only by
              engaging it (combat always reveals), never by looking.
    """
    if agent.agent_type != SCOUT:
        return []
    pos = agent.position
    return [e for e in ground_truth.nodes[pos].get("visible_edges", [])
            if ground_truth.has_edge(*e)]


def sensed_edges_assumed(env_map, agent):
    """Edges the planner ASSUMES the agent senses. Scout only -- attackers are
    blind, so they sense nothing and discover no blockages themselves."""
    if agent.agent_type != SCOUT:
        return []
    pos = agent.position
    return list(env_map.nodes[pos].get("visible_edges", []))


def sensed_nodes_truth(ground_truth, agent):
    """Nodes the agent can see (endpoints of its truly-sensed edges + self)."""
    nodes = {agent.position}
    for u, v in sensed_edges_truth(ground_truth, agent):
        nodes.add(u)
        nodes.add(v)
    return nodes


# ---------------------------------------------------------------------------
# Observation: blockage discovery + target-type revelation.
# ---------------------------------------------------------------------------

def observe_and_reveal(env_map, ground_truth, agents, events=None, time=0.0,
                       verbose=False):
    """Each living agent observes from its position.

    Mutates ``env_map``: marks sensed edges observed, removes newly discovered
    blocked edges, and reveals the ``rps_type`` of any target the agent can
    see. Returns ``(newly_blocked_edge_keys, num_types_revealed)`` -- both are
    replan triggers (a freshly revealed type can let an idle combat agent act).
    """
    newly_blocked = set()
    revealed = 0

    for idx, agent in enumerate(agents):
        if not agent.alive:
            continue

        observable = {_ekey(*e) for e in sensed_edges_truth(ground_truth, agent)}
        assumed_edges = sensed_edges_assumed(env_map, agent)
        assumed = {_ekey(*e) for e in assumed_edges}
        newly_blocked |= (assumed - observable)

        # The agent has now looked at every edge in its footprint.
        for e in assumed_edges:
            if env_map.has_edge(*e):
                env_map.edges[e]["observed_edge"] = True

        # Reveal the type of any target within sight.
        for node in sensed_nodes_truth(ground_truth, agent):
            data = env_map.nodes.get(node, {})
            if data.get("type") not in ("target_unreached", "target_reached"):
                continue
            if data.get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE:
                continue
            true_type = ground_truth.nodes[node].get("rps_type", UNKNOWN_TYPE)
            env_map.nodes[node]["rps_type"] = true_type
            revealed += 1
            if events is not None:
                events.append({"time": time, "event": "reveal", "agent": idx,
                               "node": node, "rps_type": true_type})
            if verbose:
                print(f"  [reveal] agent {idx} ({TYPE_NAMES[agent.agent_type]}) "
                      f"saw target {node} -> {TYPE_NAMES[true_type]}")

    # Apply the blockages to the planner's map.
    if newly_blocked:
        for key in newly_blocked:
            u, v = tuple(key)
            for e in ((u, v), (v, u)):
                if env_map.has_edge(*e):
                    env_map.remove_edge(*e)
        for node in env_map.nodes():
            ve = env_map.nodes[node].get("visible_edges")
            if ve is not None:
                env_map.nodes[node]["visible_edges"] = [
                    e for e in ve if _ekey(*e) not in newly_blocked
                ]

    return newly_blocked, revealed


# ---------------------------------------------------------------------------
# Combat resolution (transition-triggered: an encounter happens the instant an
# agent steps onto a live target).
# ---------------------------------------------------------------------------

def resolve_combat_on_arrival(env_map, ground_truth, agent, agent_idx, node,
                              events=None, time=0.0, verbose=False):
    """Resolve an encounter for ``agent`` that just stepped onto live ``node``.

    Always reveals the target's type. Returns the outcome string
    (rps.DRAW / AGENT_WINS / AGENT_DIES) and mutates state:
      * AGENT_WINS -> node becomes 'target_reached' (eliminated).
      * AGENT_DIES -> agent.alive = False.
      * DRAW       -> no state change beyond revealing the type.
    """
    true_type = ground_truth.nodes[node].get("rps_type", UNKNOWN_TYPE)
    env_map.nodes[node]["rps_type"] = true_type  # an encounter always reveals
    outcome = resolve_encounter(agent.agent_type, true_type)

    if outcome == AGENT_WINS:
        env_map.nodes[node]["type"] = "target_reached"
    elif outcome == AGENT_DIES:
        agent.alive = False

    if events is not None:
        events.append({"time": time, "event": outcome, "agent": agent_idx,
                       "node": node, "agent_type": agent.agent_type,
                       "rps_type": true_type})
    if verbose:
        print(f"  [combat] agent {agent_idx} ({TYPE_NAMES[agent.agent_type]}) "
              f"vs target {node} ({TYPE_NAMES[true_type]}) -> {outcome}")
    return outcome


# ---------------------------------------------------------------------------
# Placeholder policy. SAFE and simple, NOT optimized -- replace with the
# real baseline. It never sends a combat agent at a target it can't beat and
# never routes a scout onto a target (which would kill it).
# ---------------------------------------------------------------------------

def _safe_path(env_map, src, goal, avoid):
    """Shortest ``src``->``goal`` path that avoids the ``avoid`` nodes.

    Routes on a copy of ``env_map`` with ``avoid`` removed (never removing
    ``src`` or ``goal``) so the path never crosses an unwanted target node.
    Returns ``(path, cost)`` or ``(None, inf)`` if unreachable.
    """
    if src == goal:
        return [src], 0.0
    graph = env_map
    avoid = [n for n in avoid if n != src and n != goal]
    if avoid:
        graph = env_map.copy()
        graph.remove_nodes_from(avoid)
    try:
        path = nx.shortest_path(graph, src, goal, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")
    return path, _path_distance(graph, path)


def naive_type_aware_replan(env_map, agents, reward_ratio=1.0,
                            obs_discount_factor=1.0, sample_recursion=0,
                            sample_num_obstacle=0, sample_obstacle_hop=0,
                            verbose=False):
    """Minimal type-aware assignment so the loop runs end-to-end.

    * Combat agents: claim the nearest REVEALED target they beat, routing
      AROUND every other live target (never walking through an enemy they
      can't fight). If none beatable, idle and wait for a scout to reveal.
    * Scouts: approach the nearest UNKNOWN-type target to reveal it, also
      routing around every live target and stopping one node short so the
      scout never steps onto a target (which would kill it). If nothing is
      unknown, idle.

    The reward / sampling kwargs are accepted for interface compatibility with
    reward-driven policies but ignored here (this stub scores by distance).
    """
    for a in agents:
        a.planned_path = []

    live_targets = [n for n, d in env_map.nodes(data=True)
                    if d.get("type") == "target_unreached"]
    if not live_targets:
        return
    live_set = set(live_targets)

    # --- combat agents -> nearest beatable revealed target ---
    claimed = set()
    for a in agents:
        if not a.alive or a.agent_type == SCOUT:
            continue
        best_t, best_path, best_cost = None, None, float("inf")
        for t in live_targets:
            if t in claimed:
                continue
            tt = env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)
            if tt == UNKNOWN_TYPE or not beats(a.agent_type, tt):
                continue
            path, cost = _safe_path(env_map, a.position, t, live_set - {t})
            if path is not None and cost < best_cost:
                best_t, best_path, best_cost = t, path, cost
        if best_t is not None:
            a.planned_path = best_path
            claimed.add(best_t)
            if verbose:
                print(f"  [assign] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                      f"-> beat target {best_t} "
                      f"({TYPE_NAMES[env_map.nodes[best_t]['rps_type']]})")

    unknown = [t for t in live_targets
               if env_map.nodes[t].get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE]

    def approach_unknown(a):
        """Route ``a`` to one node short of the nearest unknown target, going
        around all live targets so it reveals the type without stepping on
        (and dying to) any target. Returns True if a path was set."""
        best_path, best_cost = None, float("inf")
        for goal in unknown:
            path, cost = _safe_path(env_map, a.position, goal, live_set - {goal})
            if path is not None and cost < best_cost:
                best_path, best_cost = path, cost
        if best_path is not None and len(best_path) >= 2:
            a.planned_path = best_path[:-1]  # stop before stepping onto target
            if verbose:
                print(f"  [assign] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                      f"-> approach {best_path[-1]} (stop at {best_path[-2]})")
            return True
        return False

    # --- scouts -> approach nearest unknown target to reveal it ---
    for a in agents:
        if a.alive and a.agent_type == SCOUT and unknown:
            approach_unknown(a)

    # NOTE: attackers are blind, so only a scout can reveal a target's type
    # without an encounter. This stub therefore needs a scout in the roster to
    # make progress safely; with none, its combat agents idle on unknown-only
    # maps. (baseline1_all_key instead has blind attackers charge unknowns and
    # accept the gamble.)


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------

def run_rps_simulation(env_map, ground_truth, agents, policy=None,
                       reward_ratio=1.0, obs_discount_factor=1.0,
                       sample_recursion=0, sample_num_obstacle=0,
                       sample_obstacle_hop=0, death_penalty=100.0,
                       max_events=100000, verbose=False,
                       render_dir=None, render_dt=1.0):
    """Run the RPS simulation in continuous time until all targets are cleared
    or the team stalls.

    Discrete-event: traversing an edge of cost c takes time c. The loop pops the
    next agent-arrival from a time-ordered heap, advances the clock to it,
    observes / resolves combat at that node, re-plans the living at-a-node agents
    (in-transit agents are committed to their current edge and re-planned only
    when they arrive), and re-schedules moves. Waiting therefore costs real time
    and shows up in the makespan.

    Args:
        env_map: planner view (copied internally; targets carry UNKNOWN types).
        ground_truth: reality (true edges + edge ``distance``, true target
            ``rps_type``); edge distance doubles as traversal time.
        agents: list of Agent (typed, all positioned at the source).
        policy: ``policy(env_map, at_node_living_agents, **kwargs)`` that sets
            each passed agent's ``planned_path``. Defaults to
            :func:`naive_type_aware_replan`.
        death_penalty: added to the objective per dead agent (a tunable knob, in
            makespan time-units).
        max_events: safety cap on processed arrivals.
        render_dir: if set, write interpolated PNG frames spaced ``render_dt``
            apart in time.

    Returns a result dict including ``makespan`` (clock of the last elimination,
    or of termination if incomplete), ``objective`` (= makespan +
    death_penalty * num_deaths), completion, remaining/eliminated targets,
    per-agent traversal cost, deaths/survivors, and the time-stamped event log.
    """
    policy = policy or naive_type_aware_replan
    env_map = env_map.copy()
    events = []
    n = len(agents)
    transit = [None] * n            # transit[i] = (u, v, depart_t, arrive_t) or None
    heap = []                       # entries: (arrive_t, agent_idx)
    clock = 0.0
    pos = nx.get_node_attributes(ground_truth, "pos")

    def do_replan():
        # Only living, at-a-node agents are (re)planned; in-transit agents are
        # committed to their current edge until they arrive.
        planners = [a for i, a in enumerate(agents)
                    if a.alive and transit[i] is None]
        policy(env_map, planners, reward_ratio=reward_ratio,
               obs_discount_factor=obs_discount_factor,
               sample_recursion=sample_recursion,
               sample_num_obstacle=sample_num_obstacle,
               sample_obstacle_hop=sample_obstacle_hop, verbose=verbose)

    def schedule(i):
        """Commit at-a-node agent i to the next edge of its plan (push an
        arrival event). No-op if it's dead, already moving, has no next step,
        or that edge is blocked in reality (then it idles until re-planned)."""
        a = agents[i]
        if not a.alive or transit[i] is not None or len(a.planned_path) < 2:
            return
        u, v = a.position, a.planned_path[1]
        if a.planned_path[0] != u or not ground_truth.has_edge(u, v):
            return
        arrive = clock + ground_truth.edges[u, v]["distance"]  # cost == time
        transit[i] = (u, v, clock, arrive)
        heapq.heappush(heap, (arrive, i))

    def targets_remain():
        return any(d.get("type") == "target_unreached"
                   for _, d in env_map.nodes(data=True))

    # --- rendering: interpolated fixed-dt sampling of the continuous timeline ---
    render_state = {"frame": 0, "next_t": 0.0}

    def _interp_xy(i, tau):
        tr = transit[i]
        if tr is None:
            return pos[agents[i].position]
        u, v, dep, arr = tr
        frac = 0.0 if arr <= dep else max(0.0, min(1.0, (tau - dep) / (arr - dep)))
        (x0, y0), (x1, y1) = pos[u], pos[v]
        return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))

    def emit_frame(tau):
        if render_dir is None:
            return
        from Multi_Agent.simulation_utils import render_rps_frame  # lazy (matplotlib)
        xys = [_interp_xy(i, tau) for i in range(n)]
        render_rps_frame(
            env_map, ground_truth, agents, render_state["frame"],
            os.path.join(render_dir, f"frame_{render_state['frame']:04d}.png"),
            agent_xy=xys, title=f"t = {tau:.1f}")
        render_state["frame"] += 1

    # --- initialize: observe from the source, plan, schedule, first frame ---
    observe_and_reveal(env_map, ground_truth,
                       [a for a in agents if a.alive], events, clock, verbose)
    do_replan()
    for i in range(n):
        schedule(i)
    emit_frame(0.0)
    render_state["next_t"] = render_dt

    processed = 0
    while heap and targets_remain() and processed < max_events:
        t_next = heap[0][0]
        # Emit interpolated frames for sample times strictly before the event.
        if render_dir is not None:
            while render_state["next_t"] < t_next:
                emit_frame(render_state["next_t"])
                render_state["next_t"] += render_dt

        arrive_t, i = heapq.heappop(heap)
        clock = arrive_t
        processed += 1
        a = agents[i]
        u, v, _dep, _arr = transit[i]
        transit[i] = None
        a.move(u, v, ground_truth.edges[u, v]["distance"])  # -> position=v, cost, trim plan

        # Observe from the new node (scout reveals; attacker is blind), then
        # resolve an encounter if the node is a live target.
        observe_and_reveal(env_map, ground_truth, [a], events, clock, verbose)
        if env_map.nodes[v].get("type") == "target_unreached":
            resolve_combat_on_arrival(env_map, ground_truth, a, i, v,
                                      events, clock, verbose)

        # Re-plan the living at-a-node team and (re)schedule their next edges.
        do_replan()
        for j in range(n):
            schedule(j)

    emit_frame(clock)  # final state

    remaining = [nd for nd, d in env_map.nodes(data=True)
                 if d.get("type") == "target_unreached"]
    eliminated = [nd for nd, d in env_map.nodes(data=True)
                  if d.get("type") == "target_reached"]
    deaths = [i for i, a in enumerate(agents) if not a.alive]
    makespan = clock
    return {
        "env_map": env_map,
        "agents": agents,
        "makespan": makespan,
        "objective": makespan + death_penalty * len(deaths),
        "completed": not remaining,
        "remaining_targets": remaining,
        "eliminated_targets": eliminated,
        "deaths": deaths,
        "num_deaths": len(deaths),
        "survivors": [i for i, a in enumerate(agents) if a.alive],
        "total_cost": sum(a.total_traversal_cost for a in agents),
        "per_agent_cost": [a.total_traversal_cost for a in agents],
        "events": events,
    }
