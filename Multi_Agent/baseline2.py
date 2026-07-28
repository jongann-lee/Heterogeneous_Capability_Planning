"""
Baseline 2 ("P1") for the rock_scissor_paper direction.

A coordinated, belief-aware, receding-horizon policy that is *recomputed from
scratch on every arrival event*. Its whole design principle is:

    REVEAL, then STRIKE -- never gamble a death away.

A death costs +``death_penalty`` (==100) time-units in the objective
(``makespan + death_penalty * num_deaths``), which dwarfs the handful of
time-units it costs to wait for a scout to reveal a target's type. So P1

  1. drives the scout(s) on a greedy *information-gain vantage tour* to reveal
     unknown target types cheaply (info-gain per unit travel),
  2. solves a global, TYPE-GATED min-cost assignment of attackers to the
     REVEALED targets they can beat (Hungarian / ``linear_sum_assignment``),
     routing every mover on a SAFE path that steps on no other live target, and
  3. only *gambles* an attacker onto an UNKNOWN target when there is provably no
     safe way to learn its type first (it is occluded from every reachable
     vantage, or every scout is dead) -- and even then only the cheapest
     expendable attacker, exploiting that a losing/drawing probe still REVEALS
     the type so a surviving beater can follow up.

Contrast with :mod:`baseline1_all_key`, whose blind attackers charge unknown
targets and routinely die on them: P1 is built to keep ``num_deaths`` at zero
whenever the map is scoutable, which is exactly what the objective rewards.

The policy is exposed three ways, all self-contained (only imports from
``Multi_Agent.rps`` + stdlib + networkx/numpy/scipy):

  * module-level DEFAULT_* hyperparameters,
  * a :func:`make_policy` factory returning a ``replan``-signature callable that
    closes over a chosen hyperparameter set (used for ablations), and
  * a default ``replan = make_policy()`` entry point.

    from Multi_Agent import baseline2
    run_rps_simulation(env_map, ground_truth, agents, policy=baseline2.replan)
    # ablate:
    pol = baseline2.make_policy(gamble_mode="never", scout_info_weight=2.0)
    run_rps_simulation(env_map, ground_truth, agents, policy=pol)
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from Multi_Agent.rps import (
    SCOUT, ROCK, SCISSOR, PAPER, UNKNOWN_TYPE, COMBAT_TYPES,
    beats, resolve_encounter, TYPE_NAMES,
)


# ---------------------------------------------------------------------------
# Hyperparameters (module-level DEFAULTS). ``make_policy`` overrides any subset.
# Keep this set SMALL and make every knob actually take effect.
# ---------------------------------------------------------------------------

#: lambda: weight on information gain vs travel distance in the scout's vantage
#: score (higher => the scout is willing to travel further per unknown it will
#: reveal). The ``reward_ratio`` kwarg passed by the sim multiplies this.
DEFAULT_SCOUT_INFO_WEIGHT = 1.0

#: When may an attacker commit to an UNKNOWN-type target (a gamble)?
#:   "never"          -- never gamble; wait indefinitely / leave it incomplete.
#:   "occluded_only"  -- gamble only a target no reachable vantage can reveal
#:                       (or when no scout is alive). [default -- safe + complete]
#:   "ev_positive"    -- additionally gamble whenever the risk-aware expected
#:                       commit utility EV is positive.
DEFAULT_GAMBLE_MODE = "occluded_only"

#: Time-units charged per death in the objective (matches the sim default).
DEFAULT_DEATH_COST = 100.0

#: Value (in objective time-units saved) of eliminating one target. Used only by
#: the EV gate. Large-ish so a *certain* win is clearly worth pursuing.
DEFAULT_V_ELIM = 50.0

#: Weight on traversal distance in the EV gate (per unit path cost).
DEFAULT_W_COST = 1.0

#: If True, the scout mildly prefers vantages that reveal targets whose beating
#: attacker is currently idle (so the strike can start immediately).
DEFAULT_PRIORITIZE_WAITING = True

_GAMBLE_MODES = ("never", "occluded_only", "ev_positive")


# ---------------------------------------------------------------------------
# Small self-contained graph helpers (the repo's baselines are standalone;
# these mirror the ones in rps_simulation / baseline1 but avoid full
# graph.copy() in the hot path -- we remove only the few live-target nodes,
# then restore them, so a 4096-node graph is not copied per (agent, target)).
# ---------------------------------------------------------------------------

def _path_distance(graph, path):
    """Sum of edge 'distance' along ``path`` (missing edges skipped)."""
    return sum(
        graph.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
        if graph.has_edge(path[k], path[k + 1])
    )


class _Blocker:
    """Context manager that temporarily removes a set of nodes from ``graph``
    (restoring them + their incident edges on exit).

    Used to carve out live-target nodes so shortest-path routing physically
    cannot cross an enemy the mover can't fight. Avoids ``graph.copy()`` in the
    hot loop: on a big graph we detach only a few nodes and stitch them back.
    """

    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = [n for n in nodes if graph.has_node(n)]
        self._node_data = {}
        self._edges = []  # (u, v, data) for every incident directed edge

    def __enter__(self):
        g = self.graph
        for n in self.nodes:
            self._node_data[n] = dict(g.nodes[n])
        # Snapshot incident edges (both orientations for a DiGraph).
        for n in self.nodes:
            if g.is_directed():
                for v in list(g.successors(n)):
                    self._edges.append((n, v, dict(g.edges[n, v])))
                for u in list(g.predecessors(n)):
                    if u != n:
                        self._edges.append((u, n, dict(g.edges[u, n])))
            else:
                for v in list(g.neighbors(n)):
                    self._edges.append((n, v, dict(g.edges[n, v])))
        g.remove_nodes_from(self.nodes)
        return self

    def __exit__(self, *exc):
        g = self.graph
        for n in self.nodes:
            g.add_node(n, **self._node_data[n])
        for u, v, data in self._edges:
            if g.has_node(u) and g.has_node(v):
                g.add_edge(u, v, **data)
        return False


def _safe_path(graph, src, goal, avoid):
    """Shortest ``src``->``goal`` path avoiding the ``avoid`` nodes.

    Never removes ``src``/``goal`` themselves (so a mover can start on, or end
    on, a target). Returns ``(path, cost)`` or ``(None, inf)`` if unreachable.
    Temporarily detaches the ``avoid`` nodes rather than copying the graph.
    """
    if src == goal:
        return [src], 0.0
    if not graph.has_node(src) or not graph.has_node(goal):
        return None, float("inf")
    rm = [n for n in avoid if n != src and n != goal]
    if rm:
        with _Blocker(graph, rm):
            try:
                path = nx.shortest_path(graph, src, goal, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None, float("inf")
            return path, _path_distance(graph, path)
    try:
        path = nx.shortest_path(graph, src, goal, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")
    return path, _path_distance(graph, path)


def _multi_source_safe_costs(graph, src, goals, avoid):
    """Costs from ``src`` to every node in ``goals`` via a SINGLE Dijkstra that
    avoids ``avoid`` (minus src/goals). Much cheaper than one shortest_path call
    per goal on a big graph. Returns ``{goal: (path, cost)}`` for reached goals.

    ``goals`` themselves are never in ``avoid`` (a mover is allowed to end on a
    goal), so all goals stay in the graph and Dijkstra can terminate on them.
    """
    out = {}
    if not graph.has_node(src):
        return out
    goal_set = set(g for g in goals if graph.has_node(g))
    if not goal_set:
        return out
    rm = [n for n in avoid if n != src and n not in goal_set]

    def _run():
        # Single-source Dijkstra with predecessor + distance maps.
        dist, paths = nx.single_source_dijkstra(graph, src, weight="distance")
        for g in goal_set:
            if g in dist:
                out[g] = (paths[g], dist[g])

    if rm:
        with _Blocker(graph, rm):
            _run()
    else:
        _run()
    return out


# ---------------------------------------------------------------------------
# Belief / state queries over env_map.
# ---------------------------------------------------------------------------

def _live_targets(env_map):
    return [n for n, d in env_map.nodes(data=True)
            if d.get("type") == "target_unreached"]


def _rps_type(env_map, t):
    return env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)


def _endpoints_of_visible(env_map, v):
    """Set of node ids that are an endpoint of some edge in ``visible_edges[v]``.

    A still-unknown live target ``t`` is revealable by a scout standing at ``v``
    iff ``t`` is in this set -- exactly the sim's ``sensed_nodes_truth`` rule
    (it reveals the type of any target that is an endpoint of a sensed edge).
    """
    seen = set()
    for e in env_map.nodes[v].get("visible_edges", ()):  # e == (u, w)
        # e may be a 2-tuple of node ids.
        seen.add(e[0])
        seen.add(e[1])
    return seen


def _build_vantage_inversion(env_map, unknown_targets):
    """Invert visibility once per call: ``target -> set(vantage nodes)`` and its
    transpose ``vantage -> set(unknown targets revealed there)``.

    Scans every node's ``visible_edges`` ONCE (keyed implicitly by the current
    unknown-target set, since we only keep unknown live targets). This is the
    single expensive sweep; everything downstream indexes into it.

    Returns ``(target_to_vantages, vantage_to_targets)``.
    """
    unknown_set = set(unknown_targets)
    target_to_vantages = {t: set() for t in unknown_set}
    vantage_to_targets = {}
    if not unknown_set:
        return target_to_vantages, vantage_to_targets

    for v, data in env_map.nodes(data=True):
        ve = data.get("visible_edges")
        if not ve:
            continue
        revealed_here = set()
        for e in ve:  # e == (u, w)
            a, b = e[0], e[1]
            if a in unknown_set:
                revealed_here.add(a)
            if b in unknown_set:
                revealed_here.add(b)
        if revealed_here:
            vantage_to_targets[v] = revealed_here
            for t in revealed_here:
                target_to_vantages[t].add(v)
    return target_to_vantages, vantage_to_targets


# ---------------------------------------------------------------------------
# The P1 policy factory.
# ---------------------------------------------------------------------------

def make_policy(scout_info_weight=DEFAULT_SCOUT_INFO_WEIGHT,
                gamble_mode=DEFAULT_GAMBLE_MODE,
                death_cost=DEFAULT_DEATH_COST,
                v_elim=DEFAULT_V_ELIM,
                w_cost=DEFAULT_W_COST,
                prioritize_waiting=DEFAULT_PRIORITIZE_WAITING):
    """Build a ``replan``-signature policy closing over a hyperparameter set.

    See the module docstring / DEFAULT_* constants for the meaning of each knob.
    Raises ValueError on an unknown ``gamble_mode``.
    """
    if gamble_mode not in _GAMBLE_MODES:
        raise ValueError(
            f"gamble_mode must be one of {_GAMBLE_MODES}, got {gamble_mode!r}")

    def replan(env_map, agents, reward_ratio=1.0, obs_discount_factor=1.0,
               sample_recursion=0, sample_num_obstacle=0, sample_obstacle_hop=0,
               verbose=False):
        """Assign every living, at-a-node agent a ``planned_path`` under P1.

        ``agents`` may be a SUBSET of the roster (only the living, at-a-node
        agents the sim hands us). The reward/sampling kwargs are accepted for
        interface compatibility; ``reward_ratio`` scales the scout's info weight.
        """
        # 0. Reset. Default = idle (stay put); we fill in paths below.
        for a in agents:
            a.planned_path = []

        live_targets = _live_targets(env_map)
        if not live_targets:
            return
        live_set = set(live_targets)

        scouts = [a for a in agents if a.alive and a.agent_type == SCOUT]
        attackers = [a for a in agents
                     if a.alive and a.agent_type in COMBAT_TYPES]

        revealed_live = [t for t in live_targets
                         if _rps_type(env_map, t) != UNKNOWN_TYPE]
        unknown_live = [t for t in live_targets
                        if _rps_type(env_map, t) == UNKNOWN_TYPE]

        # A scout is alive somewhere on the roster iff we were handed one OR one
        # is in transit. We can only *see* the ones handed to us, but even an
        # in-transit scout means "reveals are still coming" for the gamble gate.
        # The sim only passes at-a-node agents, so use the ones we have; an
        # in-transit scout will re-plan on arrival. Treat "any scout in `agents`"
        # as the observable signal, and additionally consult whether ANY vantage
        # exists (below) to decide occlusion.
        scout_alive = len(scouts) > 0

        info_weight = float(scout_info_weight) * float(reward_ratio or 1.0)

        # ---- Vantage inversion (single sweep, keyed by unknown-target set). ----
        target_to_vantages, vantage_to_targets = _build_vantage_inversion(
            env_map, unknown_live)

        # =====================================================================
        # 2. ATTACKERS -- type-gated global min-cost assignment onto revealed
        #    targets they beat. (Do attackers first so the scout can know which
        #    attackers are idle, for its optional waiting-priority weight.)
        # =====================================================================
        assigned_target = {}   # attacker index in `attackers` -> target node
        assigned_path = {}     # attacker index -> path

        if attackers and revealed_live:
            # Cost matrix: attacker i -> revealed target j it BEATS; else +inf.
            n_a = len(attackers)
            targets_j = revealed_live
            n_t = len(targets_j)
            BIG = 1e12
            cost = np.full((n_a, n_t), BIG, dtype=float)
            path_cache = {}

            for i, a in enumerate(attackers):
                # Beatable revealed targets for this attacker.
                beatable = [t for t in targets_j
                            if beats(a.agent_type, _rps_type(env_map, t))]
                if not beatable:
                    continue
                # One Dijkstra from this attacker to all beatable targets,
                # routing around every OTHER live target so it cannot die en
                # route (goal targets stay in the graph so it can end on them).
                avoid = live_set - set(beatable)
                costs = _multi_source_safe_costs(env_map, a.position,
                                                 beatable, avoid)
                for t, (p, c) in costs.items():
                    j = targets_j.index(t)
                    cost[i, j] = c
                    path_cache[(i, j)] = p

            # Solve the assignment (minimize total safe travel).
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= BIG:
                    continue  # forbidden (unbeatable / unreachable) pair
                a = attackers[i]
                t = targets_j[j]
                path = path_cache.get((i, j))
                if path and len(path) >= 2:
                    a.planned_path = path
                    assigned_target[i] = t
                    assigned_path[i] = path
                elif path and len(path) == 1 and path[0] == a.position == t:
                    # Already standing on the target (combat resolves this tick).
                    a.planned_path = path
                    assigned_target[i] = t
                    assigned_path[i] = path
                if verbose and i in assigned_target:
                    print(f"  [b2] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                          f"-> STRIKE beat target {t} "
                          f"({TYPE_NAMES[_rps_type(env_map, t)]}) "
                          f"cost {cost[i, j]:.2f}")

        # Attackers with no assignment so far are (for now) idle / waiting.
        idle_attackers = [i for i in range(len(attackers))
                          if i not in assigned_target]

        # =====================================================================
        # 1. SCOUT(S) -- greedy information-gain vantage tour.
        #    Route each scout to the reachable vantage maximizing
        #    (info gain) / (travel cost), revealing the most still-UNKNOWN live
        #    targets per unit distance. Route SAFELY (avoid ALL live targets;
        #    a scout dies on any) and NEVER end on a live target.
        # =====================================================================
        # Attacker types that are currently idle (for the waiting-priority bonus).
        idle_beater_types = set()
        if prioritize_waiting:
            for i in idle_attackers:
                idle_beater_types.add(attackers[i].agent_type)

        # Vantage nodes usable as a scout destination: must NOT be a live target
        # (scout would die) and must reveal >=1 unknown.
        candidate_vantages = [v for v in vantage_to_targets
                              if v not in live_set]

        for scout in scouts:
            if not unknown_live or not candidate_vantages:
                scout.planned_path = []  # nothing to learn -> park safely (idle)
                continue

            # Does the scout ALREADY reveal >=1 unknown from where it stands?
            here = vantage_to_targets.get(scout.position)
            if here and scout.position not in live_set:
                # Standing on a productive, safe vantage: staying is fine (idle).
                scout.planned_path = []
                if verbose:
                    print(f"  [b2] scout @ {scout.position} already reveals "
                          f"{len(here)} unknown(s) -> hold")
                continue

            # Reachable vantages + their safe-travel cost via one Dijkstra,
            # routing around ALL live targets (scout must not touch any).
            reach = _multi_source_safe_costs(
                env_map, scout.position, candidate_vantages, live_set)

            best_v, best_score, best_path = None, float("-inf"), None
            for v, (p, c) in reach.items():
                revealed = vantage_to_targets.get(v, ())
                if not revealed:
                    continue
                gain = 0.0
                for t in revealed:
                    w = 1.0
                    if prioritize_waiting:
                        # Bonus if some idle attacker could beat t once revealed.
                        tt_needed = [at for at in COMBAT_TYPES
                                     if at in idle_beater_types]
                        # We don't know t's type yet, but any idle beater raises
                        # the value of learning it (>=1 of the 3 types beats t).
                        if tt_needed:
                            w += 0.25
                    gain += w
                # Info-gain per unit distance (with a small floor so a
                # zero-distance vantage -- adjacent-hop -- isn't infinite).
                denom = max(c, 1e-6)
                score = info_weight * gain / denom
                if score > best_score:
                    best_v, best_score, best_path = v, score, p

            if best_path is not None and len(best_path) >= 2:
                scout.planned_path = best_path  # end ON the vantage (safe: not a target)
                if verbose:
                    print(f"  [b2] scout @ {scout.position} -> vantage {best_v} "
                          f"(reveals {len(vantage_to_targets[best_v])}, "
                          f"score {best_score:.3f})")
            else:
                scout.planned_path = []  # no safe productive vantage -> idle

        # =====================================================================
        # 3. GAMBLE GATE -- only commit an idle attacker onto an UNKNOWN target
        #    when it cannot be revealed safely first.
        # =====================================================================
        if gamble_mode == "never" or not idle_attackers or not unknown_live:
            return

        # Which unknown targets can still be revealed by SOME reachable vantage
        # given a live scout? If a scout is alive and a target has a vantage that
        # the scout can safely reach, WAIT -- don't gamble it.
        revealable_now = set()
        if scout_alive:
            for scout in scouts:
                reach = _multi_source_safe_costs(
                    env_map, scout.position,
                    [v for v in candidate_vantages], live_set)
                for v in reach:
                    revealable_now |= vantage_to_targets.get(v, set())

        # An unknown target is "occluded" if NO scout can safely reach a vantage
        # that reveals it -- either no vantage exists off the target, or no live
        # scout can get to one. Those are the only gamble candidates in
        # "occluded_only" mode.
        def _is_occluded(t):
            vs = target_to_vantages.get(t, set())
            safe_vs = [v for v in vs if v not in live_set]
            if not safe_vs:
                return True  # only vantage is the target cell itself (deadly)
            if not scout_alive:
                return True  # nobody left to reveal it
            return t not in revealable_now  # scout can't safely reach a vantage

        gamble_candidates = [t for t in unknown_live if _is_occluded(t)]
        if not gamble_candidates:
            return  # everything unknown is still safely revealable -> wait

        # For each remaining idle attacker, find its cheapest safe path onto a
        # gamble-candidate target, then decide per ``gamble_mode``.
        # We assign at most one gamble per idle attacker, cheapest-first, and
        # avoid double-committing two attackers to the same target.
        p_win = p_draw = p_loss = 1.0 / 3.0
        gambled_targets = set()

        # Build (attacker_i, target, path, dist) options.
        options = []
        for i in idle_attackers:
            a = attackers[i]
            avoid = live_set - set(gamble_candidates)
            costs = _multi_source_safe_costs(env_map, a.position,
                                             gamble_candidates, avoid)
            for t, (p, c) in costs.items():
                if p and len(p) >= 2:
                    options.append((c, i, t, p))
                elif p and len(p) == 1 and p[0] == a.position == t:
                    options.append((c, i, t, p))
        options.sort(key=lambda o: o[0])  # cheapest first

        used_attackers = set()
        for c, i, t, p in options:
            if i in used_attackers or t in gambled_targets:
                continue
            commit = False
            if gamble_mode == "occluded_only":
                commit = True  # occluded + we're here => probe it now
            elif gamble_mode == "ev_positive":
                ev = (p_win * v_elim
                      - p_loss * death_cost
                      - w_cost * c)
                commit = ev > 0.0
            if commit:
                a = attackers[i]
                a.planned_path = p
                used_attackers.add(i)
                gambled_targets.add(t)
                if verbose:
                    print(f"  [b2] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                          f"-> GAMBLE occluded target {t} (cost {c:.2f}, "
                          f"mode={gamble_mode})")

        return

    # Stamp a readable name / the hyperparameters for debugging & ablation logs.
    replan.__name__ = "baseline2_replan"
    replan.hyperparameters = {
        "scout_info_weight": scout_info_weight,
        "gamble_mode": gamble_mode,
        "death_cost": death_cost,
        "v_elim": v_elim,
        "w_cost": w_cost,
        "prioritize_waiting": prioritize_waiting,
    }
    return replan


# Default entry point (matches the run_rps_simulation policy signature).
replan = make_policy()
