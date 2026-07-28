"""
GTSP information-tour scout policy for the rock_scissor_paper direction.

The SCOUT half is the experiment; the ATTACKER half is copied verbatim from
baseline1_all_key so that any change in results is attributable to scouting.

Scout: a minimum-travel COVERING TOUR over target witness sets
----------------------------------------------------------------
A still-unknown live target j is revealable from exactly the cells

    W_j = { v : some edge incident to j is in visible_edges[v],
                and v is not itself a live target }        (its "witness set")

(this mirrors the sim's ``sensed_nodes_truth`` rule: a scout at v reveals
every target that is an endpoint of an edge it senses). The scout's job is a
Generalized-TSP / Covering-Salesman tour: starting from where it stands,
visit at least one cell of every W_j with minimum total travel, never
touching a live target (a scout dies on any). Witness sets may overlap and
be disconnected -- both are handled natively below.

Per replan epoch (only when the belief actually changed -- see caching):

  1. INVERT visibility once: target -> witness cells, cell -> unknown
     targets revealed from there.
  2. UNSCOUTABLE detection: a target with no reachable safe witness cell can
     never be revealed by the scout; it is excluded from the tour and
     reported on the agent (``scout.unscoutable``) so a future attacker
     layer can route gambles there deliberately.
  3. MASK-GROUP candidates: witness cells are grouped by their reveal mask
     (the subset of unknown targets seen from the cell); per group keep a
     few spatially spread representatives (nearest-to-scout + farthest-point
     samples). Tens of candidate nodes instead of thousands of cells.
  4. DISTANCE MATRIX: one Dijkstra per representative on the planner graph
     with every live-target node temporarily removed. All terrain ugliness
     (disconnected witness components, asymmetric up/down-hill costs) is
     absorbed here; the tour solver only ever sees the matrix.
  5. SOLVE the covering tour EXACTLY by Held-Karp-style dynamic programming
     over (covered-target bitmask, current node): open path, fixed start at
     the scout's position. Overlapping masks cost nothing extra -- covering
     semantics live in the bitmask union. Above ``dp_target_cap`` scoutable
     targets it falls back to a greedy new-targets-per-cost chain.
  6. STITCH the stop sequence into a full cell path via the cached Dijkstra
     predecessor paths, then SIMULATE the walk cell-by-cell: cells along the
     way reveal targets too (en-route freebies), so the path is truncated at
     the moment every scoutable target has been revealed, and the simulation
     yields the REVEAL SCHEDULE

         scout.reveal_schedule = { target : time-from-now }

     in simulation time units (edge cost == traversal time). This schedule
     is the interface a smarter attacker layer can consume to pre-position
     before a reveal lands.

Caching / adaptivity: the tour is committed and re-used across arrival
events; it is re-solved from the scout's current position only when the
belief changes -- a type reveal (the unknown-target set shrinks) or a
discovered blockage (the planner graph loses edges) -- or if the scout ever
finds itself off its plan. A reveal thus triggers an adaptive re-solve with
the remaining targets, which also removes any stop made redundant by
en-route reveals.

Attackers (copied from baseline1_all_key, deliberately weak)
------------------------------------------------------------
Category preference  win > draw > unknown > lose  over the outcome the
attacker would get against each live target's (possibly hidden) type;
closest reachable target within the best non-empty category; engagement is
LITERAL (walks all the way on, gambling on unknowns and even dying on
"lose" if nothing better is reachable); routes avoid all OTHER live targets.

Usage
-----
    from Multi_Agent import scout_gtsp
    run_rps_simulation(env_map, ground_truth, agents, policy=scout_gtsp.replan)
    # or with knobs:
    pol = scout_gtsp.make_policy(k_reps=3, max_reps=40, dp_target_cap=10)
"""

import sys
import os
import heapq

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx

from Multi_Agent.rps import SCOUT, UNKNOWN_TYPE, beats, TYPE_NAMES


# ---------------------------------------------------------------------------
# Hyperparameters (defaults; ``make_policy`` overrides any subset).
# ---------------------------------------------------------------------------

#: Spatially spread representative cells kept per reveal-mask group.
DEFAULT_K_REPS = 3
#: Cap on total representative cells across all mask groups (keeps the DP and
#: the number of Dijkstras small). Coverage of every scoutable target is
#: preserved when trimming.
DEFAULT_MAX_REPS = 40
#: Max number of scoutable unknown targets for the EXACT bitmask DP; above
#: this the greedy chain fallback is used (2^cap states).
DEFAULT_DP_TARGET_CAP = 10


INF = float("inf")


# ---------------------------------------------------------------------------
# Attacker half -- copied from baseline1_all_key (kept standalone on purpose,
# matching the repo's baseline convention). Behavior is identical.
# ---------------------------------------------------------------------------

ATTACKER_PREFERENCE = ("win", "draw", "unknown", "lose")


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
    ``(None, inf)`` if unreachable. (Verbatim from baseline1_all_key.)
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
        return None, INF
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


def _route_attacker(env_map, agent, live_targets, live_set, verbose):
    """baseline1's attacker rule: best category, closest within it, engage
    literally (walk all the way onto the target, whatever it is)."""
    buckets = {c: [] for c in ATTACKER_PREFERENCE}
    for t in live_targets:
        tt = env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)
        buckets[_category(agent.agent_type, tt)].append(t)

    for cat in ATTACKER_PREFERENCE:
        best_t, best_path, best_cost = None, None, INF
        for t in buckets[cat]:
            path, cost = _safe_path(env_map, agent.position, t, live_set - {t})
            if path is not None and cost < best_cost:
                best_t, best_path, best_cost = t, path, cost
        if best_t is None:
            continue  # nothing reachable in this category; drop to the next

        agent.planned_path = best_path  # walk all the way onto the target
        if verbose:
            print(f"  [gtsp|b1-att] {TYPE_NAMES[agent.agent_type]} @ "
                  f"{agent.position} -> engage {cat} target {best_t} "
                  f"(cost {best_cost:.2f})")
        return
    # no reachable target in any category -> idle


# ---------------------------------------------------------------------------
# Scout half -- graph plumbing.
# ---------------------------------------------------------------------------

class _Blocker:
    """Temporarily remove nodes from ``graph`` (restore on exit). Avoids a
    full graph.copy() in the hot path; same trick as baseline2."""

    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = [n for n in nodes if graph.has_node(n)]
        self._node_data = {}
        self._edges = []

    def __enter__(self):
        g = self.graph
        for n in self.nodes:
            self._node_data[n] = dict(g.nodes[n])
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


def _dijkstra_from(env_map, src, avoid):
    """Single-source Dijkstra from ``src`` with ``avoid`` nodes removed.

    Returns ``(dist, paths)`` dicts over reachable nodes. ``src`` is never
    removed. This is THE primitive that absorbs all terrain/occlusion
    geometry: downstream, only these distances matter.
    """
    rm = [n for n in avoid if n != src]
    if rm:
        with _Blocker(env_map, rm):
            return nx.single_source_dijkstra(env_map, src, weight="distance")
    return nx.single_source_dijkstra(env_map, src, weight="distance")


def _invert_visibility(env_map, unknown_set, live_set):
    """One sweep over all nodes: which unknown targets does each SAFE cell
    reveal (cell -> mask set), and each target's witness cells (target ->
    cells). A cell that is itself a live target is excluded (deadly)."""
    cell_reveals = {} # List of unknown targets from each safe cell (v -> {t1, t2, ...})
    witnesses = {t: set() for t in unknown_set} # List of witness cells for each unknown target (t -> {v1, v2, ...})
    if not unknown_set:
        return cell_reveals, witnesses
    for v, data in env_map.nodes(data=True):
        if v in live_set:
            continue
        ve = data.get("visible_edges")
        if not ve:
            continue
        seen = set()
        for e in ve:  # e == (u, w)
            if e[0] in unknown_set:
                seen.add(e[0])
            if e[1] in unknown_set:
                seen.add(e[1])
        if seen:
            cell_reveals[v] = seen
            for t in seen:
                witnesses[t].add(v)
    return cell_reveals, witnesses


def _node_xy(env_map, n):
    """Best-effort 2D coordinates for spatial spread sampling."""
    p = env_map.nodes[n].get("pos")
    if p is not None:
        return (float(p[0]), float(p[1]))
    if isinstance(n, tuple) and len(n) == 2:
        return (float(n[0]), float(n[1]))
    try:
        return (float(n), 0.0)
    except (TypeError, ValueError):
        return (0.0, 0.0)


def _farthest_point_sample(env_map, cells, k, first):
    """``first`` + up to k-1 more cells chosen by max-min Euclidean spread."""
    chosen = [first]
    cand = [c for c in cells if c != first]
    while len(chosen) < k and cand:
        far = max(cand, key=lambda c: min(
            (_node_xy(env_map, c)[0] - _node_xy(env_map, s)[0]) ** 2
            + (_node_xy(env_map, c)[1] - _node_xy(env_map, s)[1]) ** 2
            for s in chosen))
        chosen.append(far)
        cand.remove(far)
    return chosen


def _select_representatives(env_map, cell_reveals, scoutable, dist_from_scout,
                            k_reps, max_reps):
    """Group reachable witness cells by reveal mask; keep a few spread
    representatives per group; trim under ``max_reps`` without ever dropping
    coverage of a scoutable target.

    Returns ``[(cell, mask_frozenset), ...]``.
    """
    groups = {}
    for v, seen in cell_reveals.items():
        if v not in dist_from_scout:
            continue  # unreachable cell is useless as a stop
        m = frozenset(seen & scoutable) # the visible unknown targets from v that are scoutable
        if m:
            groups.setdefault(m, []).append(v)

    def reps_for(k): # Builds the representative list for a given k (spread per mask group).
        out = []
        for m, cells in groups.items():
            nearest = min(cells, key=lambda c: dist_from_scout[c])
            for c in _farthest_point_sample(env_map, cells, k, nearest):
                out.append((c, m))
        return out

    for k in sorted({k_reps, 2, 1}, reverse=True):
        if k > k_reps:
            continue
        reps = reps_for(k)
        if len(reps) <= max_reps:
            return reps

    # Still too many with k=1: one rep per mask exceeds the budget. Keep a
    # greedy set cover first (coverage guaranteed), then fill with the
    # largest remaining masks until the budget is spent.
    one_per_mask = {m: min(cells, key=lambda c: dist_from_scout[c])
                    for m, cells in groups.items()}
    kept, covered = [], set()
    pool = dict(one_per_mask)
    while covered != scoutable and pool:
        m = max(pool, key=lambda mm: (len(mm - covered),
                                      -dist_from_scout[pool[mm]]))
        if not (m - covered):
            break
        kept.append((pool.pop(m), m))
        covered |= m
    for m in sorted(pool, key=len, reverse=True):
        if len(kept) >= max_reps:
            break
        kept.append((pool[m], m))
    return kept


# ---------------------------------------------------------------------------
# Covering-tour solvers (exact bitmask DP + greedy fallback).
# ---------------------------------------------------------------------------

def _solve_cover_tour_dp(D, rep_masks, full_mask):
    """Exact open-path covering tour via Dijkstra over (mask, node) states.

    ``D[i][j]``: travel cost node i -> node j, where index 0 is the scout's
    position and 1..R are representatives; ``rep_masks[j-1]``: int bitmask of
    targets revealed at representative j. Returns the list of representative
    indices (1-based into D) in visit order, or None if full coverage is
    unreachable.
    """
    R = len(rep_masks)
    start = (0, 0)  # (mask, node)
    best = {start: 0.0}
    parent = {}
    heap = [(0.0, 0, 0)]
    goal_state = None
    while heap:
        c, mask, i = heapq.heappop(heap)
        if c > best.get((mask, i), INF):
            continue
        if mask == full_mask:
            goal_state = (mask, i)
            break
        for j in range(1, R + 1):
            mj = rep_masks[j - 1]
            nm = mask | mj
            if nm == mask:
                continue  # stop adds nothing -> never useful to go there next
            d = D[i][j]
            if d == INF:
                continue
            nc = c + d
            if nc < best.get((nm, j), INF):
                best[(nm, j)] = nc
                parent[(nm, j)] = (mask, i)
                heapq.heappush(heap, (nc, nm, j))
    if goal_state is None:
        return None
    order = []
    state = goal_state
    while state != start:
        mask, i = state
        order.append(i)
        state = parent[state]
    order.reverse()
    return order


def _solve_cover_tour_greedy(D, rep_masks, full_mask):
    """Greedy chain: repeatedly hop to the representative with the best
    (newly covered targets) / (travel cost) ratio. Fallback for large target
    counts; same interface as the DP."""
    R = len(rep_masks)
    mask, i = 0, 0
    order = []
    while mask != full_mask:
        best_j, best_score = None, -1.0
        for j in range(1, R + 1):
            gain = bin((mask | rep_masks[j - 1]) ^ mask).count("1")
            if gain == 0 or D[i][j] == INF:
                continue
            score = gain / max(D[i][j], 1e-9)
            if score > best_score:
                best_j, best_score = j, score
        if best_j is None:
            break  # remaining targets unreachable from here
        order.append(best_j)
        mask |= rep_masks[best_j - 1]
        i = best_j
    return order or None


# ---------------------------------------------------------------------------
# Scout planning (tour -> stitched path -> reveal schedule).
# ---------------------------------------------------------------------------

def _plan_scout(env_map, scout, unknown_live, live_set,
                k_reps, max_reps, dp_target_cap, verbose):
    """(Re)plan the scout's covering tour if the belief changed; otherwise
    keep the committed plan. Sets ``planned_path``, ``reveal_schedule``,
    ``unscoutable`` on the agent."""
    belief_key = (frozenset(unknown_live), env_map.number_of_edges())

    # ---- cache: keep the committed tour while the belief is unchanged ----
    if getattr(scout, "_gtsp_key", None) == belief_key:
        pp = scout.planned_path
        if pp and pp[0] == scout.position:
            return  # mid-tour, nothing new -> keep walking the plan
        if not pp or len(pp) == 1:
            return  # tour finished (or idling); nothing to learn anew
        # plan exists but doesn't start at our position -> fall through, replan

    scout.planned_path = []
    scout.reveal_schedule = {}
    scout.unscoutable = set()
    scout._gtsp_key = belief_key

    if not unknown_live:
        if verbose:
            print("  [gtsp] scout: no unknown targets -> idle")
        return

    unknown_set = set(unknown_live)
    cell_reveals, witnesses = _invert_visibility(env_map, unknown_set, live_set)

    # Reachability from the scout (live targets are not part of its world).
    dist_scout, paths_scout = _dijkstra_from(env_map, scout.position, live_set)

    scoutable = {t for t, cells in witnesses.items()
                 if any(v in dist_scout for v in cells)}
    scout.unscoutable = unknown_set - scoutable
    if verbose and scout.unscoutable:
        print(f"  [gtsp] scout: UNSCOUTABLE targets {sorted(scout.unscoutable)} "
              f"(no reachable safe witness cell)")
    if not scoutable:
        if verbose:
            print("  [gtsp] scout: nothing scoutable -> idle")
        return

    # ---- candidate representatives (mask groups, spread samples) ----
    reps = _select_representatives(env_map, cell_reveals, scoutable,
                                   dist_scout, k_reps, max_reps)
    if not reps:
        return

    # ---- bitmask encoding ----
    t_index = {t: b for b, t in enumerate(sorted(scoutable, key=str))}
    full_mask = (1 << len(t_index)) - 1

    def to_mask(fs):
        m = 0
        for t in fs:
            if t in t_index:
                m |= 1 << t_index[t]
        return m

    rep_nodes = [c for c, _m in reps]
    rep_masks = [to_mask(m) for _c, m in reps]

    # ---- distance matrix + predecessor paths (index 0 = scout position) ----
    nodes = [scout.position] + rep_nodes
    dijk = {scout.position: (dist_scout, paths_scout)}
    for c in rep_nodes:
        if c not in dijk:
            dijk[c] = _dijkstra_from(env_map, c, live_set)
    D = [[dijk[u][0].get(v, INF) for v in nodes] for u in nodes]

    # ---- solve ----
    if len(t_index) <= dp_target_cap:
        order = _solve_cover_tour_dp(D, rep_masks, full_mask)
        solver = "dp"
    else:
        order = _solve_cover_tour_greedy(D, rep_masks, full_mask)
        solver = "greedy"
    if not order:
        if verbose:
            print("  [gtsp] scout: no covering tour found -> idle")
        return

    # ---- stitch stops into a cell path via cached predecessor paths ----
    path = [scout.position]
    cur = scout.position
    for j in order:
        stop = nodes[j]
        leg = dijk[cur][1].get(stop)
        if leg is None or len(leg) < 1:
            break  # matrix said reachable; be defensive anyway
        path.extend(leg[1:])
        cur = stop

    # ---- simulate the walk: en-route reveals, schedule, truncation ----
    remaining = set(scoutable)
    schedule = {}
    clock = 0.0
    out = [path[0]]
    for t in cell_reveals.get(path[0], ()):  # (normally already revealed)
        if t in remaining:
            schedule[t] = 0.0
            remaining.discard(t)
    for k in range(1, len(path)):
        u, v = path[k - 1], path[k]
        if not env_map.has_edge(u, v):
            break  # defensive: plan crosses a since-removed edge
        clock += env_map.edges[u, v]["distance"]
        out.append(v)
        for t in cell_reveals.get(v, ()):
            if t in remaining:
                schedule[t] = clock
                remaining.discard(t)
        if not remaining:
            break  # everything scoutable revealed -> rest of tour is waste

    scout.planned_path = out if len(out) >= 2 else []
    scout.reveal_schedule = schedule
    if verbose:
        stops = " -> ".join(str(nodes[j]) for j in order)
        print(f"  [gtsp] scout tour ({solver}, {len(order)} stops): {stops}")
        for t in sorted(schedule, key=schedule.get):
            print(f"  [gtsp]   reveal ETA  target {t}: t+{schedule[t]:.2f}")


# ---------------------------------------------------------------------------
# Policy factory / entry point (matches the run_rps_simulation signature).
# ---------------------------------------------------------------------------

def make_policy(k_reps=DEFAULT_K_REPS, max_reps=DEFAULT_MAX_REPS,
                dp_target_cap=DEFAULT_DP_TARGET_CAP):
    """Build a ``replan``-signature policy closing over the scout knobs."""

    def replan(env_map, agents, reward_ratio=1.0, obs_discount_factor=1.0,
               sample_recursion=0, sample_num_obstacle=0,
               sample_obstacle_hop=0, verbose=False):
        """GTSP covering-tour scout + baseline1 attackers.

        ``agents`` is the sim's living, at-a-node subset. The reward /
        sampling kwargs are accepted for interface compatibility and ignored
        (the scout minimizes pure travel for full coverage; attackers score
        by distance, as in baseline1).
        """
        live_targets = [n for n, d in env_map.nodes(data=True)
                        if d.get("type") == "target_unreached"]
        live_set = set(live_targets)
        unknown_live = [t for t in live_targets
                        if env_map.nodes[t].get("rps_type",
                                                UNKNOWN_TYPE) == UNKNOWN_TYPE]

        scouts = []
        for a in agents:
            if not a.alive:
                a.planned_path = []
                continue
            if a.agent_type == SCOUT:
                scouts.append(a)  # handled below (cache-aware)
                continue
            a.planned_path = []
            if live_targets:
                _route_attacker(env_map, a, live_targets, live_set, verbose)

        if not scouts:
            return
        if not live_targets:
            for s in scouts:
                s.planned_path = []
            return

        # Single-scout policy: the first scout runs the tour; extras idle.
        _plan_scout(env_map, scouts[0], unknown_live, live_set,
                    k_reps, max_reps, dp_target_cap, verbose)
        for s in scouts[1:]:
            s.planned_path = []

    replan.__name__ = "scout_gtsp_replan"
    replan.hyperparameters = {
        "k_reps": k_reps,
        "max_reps": max_reps,
        "dp_target_cap": dp_target_cap,
    }
    return replan


#: Default entry point.
replan = make_policy()
