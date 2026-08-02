"""Baseline 2: WRP scouting followed by capability-aware target service.

The scout plans the PROVABLY OPTIMAL covering walk by heuristic search over
coverage-augmented states, in the style of the Watchman Route Problem on
grids (Seiref, Jaffey, Lopatin & Felner, "Solving the Watchman Route Problem
on a Grid with Heuristic Search", ICAPS 2020).

Why this replaces the representative-sampling GTSP pipeline (scout_gtsp):
    that pipeline SAMPLES cells per witness group (FPS) and can in principle
    lose optimality through the sampling. Here there is NO selection anywhere:

        state       = (cell, set of targets revealed so far)
        transition  = move to a neighbor cell; pay the edge 'distance';
                      OR the neighbor's reveal mask into the set
        start       = (scout position, {})           goal = any (cell, FULL)

    A shortest path in this product space IS the optimal covering walk over
    ALL cells: visiting order, entry cells, and en-route "freebie" reveals
    are all resolved jointly, and the reveal schedule is read off the path.
    The space is 2^m x N (m = number of scoutable unknown targets) -- the
    exponential is confined to m, which is small (<= #targets) and shrinks
    with every reveal.

Search = A* with two ADMISSIBLE lower bounds (max of both; optimality is
preserved -- heuristics only prune, in the ICAPS-WRP fashion):

  h_single(v, R) = max over unrevealed t of  d(v -> W_t)
        you must at least reach the farthest witness set you still owe.
        The fields d(. -> W_t) are precomputed once per solve by one
        multi-source Dijkstra per target on the REVERSED graph (costs are
        asymmetric: uphill != downhill).

  h_pivot(v, R)  = MST over {v} u {pivot targets not in R}
        pivots = a greedily chosen family of unrevealed targets whose
        witness sets are pairwise DISJOINT (so the bound never double
        counts): any completing walk must enter each pivot region
        separately, hence costs at least the minimum spanning tree over
        v and those regions (edge weights: d(v -> W_t) and min set-to-set
        distances, direction-minimized so the bound stays a lower bound).
        Pivot choice tunes TIGHTNESS (speed) only -- never correctness.

Graceful degradation: above ``exact_target_cap`` unknown targets the search
runs as Weighted A* (f = g + w*h), returning a solution within factor w of
optimal (the SoCS-2021 suboptimal-WRP recipe). The current real-map instance
has seven targets, so it remains below the default exact-search threshold.

Scout selection and phase behavior
----------------------------------
Among all living agents with capability ``0``, the policy selects the one with
the fewest positive target capabilities (stable roster order breaks ties).
That agent performs only the WRP covering walk until every safely scoutable
live target has been revealed. It then joins the attacker layer using its
positive capabilities. Other service-capable agents attack throughout.

Interface, caching, and outputs: policy
``replan(env_map, agents, ...)``; the tour is cached on the agent under a
belief key (unknown-target set, planner edge count) and re-solved only on a
reveal / blockage / plan deviation; the scout publishes
``scout.reveal_schedule`` ({target: time-from-now}) and
``scout.unscoutable`` (targets with no reachable witness cell).

    from planning.policies import baseline2
    from simulation.engine import run_simulation
    run_simulation(env_map, ground_truth, agents, policy=baseline2.replan)
    pol = baseline2.make_policy(
        exact_target_cap=12, suboptimal_weight=1.5)
"""

import heapq

import math

from simulation.domain import UNKNOWN_TYPE, capability_label
from planning.policies.baseline1 import _route_attacker

INF = float("inf")

#: Above this many scoutable unknown targets, fall back to Weighted A*.
DEFAULT_EXACT_TARGET_CAP = 12
#: Weight for the fallback (solution guaranteed within this factor of optimal).
DEFAULT_SUBOPT_WEIGHT = 1.5
#: Time-constant (in cost/time units) of the imminence weighting used by the
#: attackers' hedge point: pending unknown target j gets weight
#: exp(-reveal_eta_j / hedge_tau), so soon-revealing targets pull harder.
DEFAULT_HEDGE_TAU = 30.0


# ---------------------------------------------------------------------------
# Graph compilation (index arrays; live-target cells excluded entirely).
# ---------------------------------------------------------------------------

def _compile(env_map, live_set, unknown_sorted):
    """Index the safe cells and compile adjacency + reveal masks.

    Returns (nodes, idx, fwd, rev, cellmask) where fwd[i] / rev[i] are lists
    of (neighbor_index, edge_cost) in forward / reversed orientation and
    cellmask[i] is the bitmask of unknown targets revealed from cell i.
    """
    nodes = [n for n in env_map.nodes() if n not in live_set]
    idx = {n: i for i, n in enumerate(nodes)}
    tbit = {t: b for b, t in enumerate(unknown_sorted)}

    fwd = [[] for _ in nodes]
    rev = [[] for _ in nodes]
    for u, v, d in env_map.edges(data="distance"):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        fwd[iu].append((iv, d))
        rev[iv].append((iu, d))

    cellmask = [0] * len(nodes)
    unknown_set = set(unknown_sorted)
    for n, data in env_map.nodes(data=True):
        i = idx.get(n)
        if i is None:
            continue
        ve = data.get("visible_edges")
        if not ve:
            continue
        mk = 0
        for e in ve:
            if e[0] in unknown_set:
                mk |= 1 << tbit[e[0]]
            if e[1] in unknown_set:
                mk |= 1 << tbit[e[1]]
        cellmask[i] = mk
    return nodes, idx, fwd, rev, cellmask


def _multi_source_dijkstra(adj, seeds):
    """dist[i] = min cost from any seed to i along ``adj`` (list-of-lists)."""
    dist = [INF] * len(adj)
    heap = []
    for s in seeds:
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j, w in adj[i]:
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(heap, (nd, j))
    return dist


# ---------------------------------------------------------------------------
# Heuristic machinery (ICAPS-WRP style).
# ---------------------------------------------------------------------------

def _build_heuristics(nodes, fwd, rev, cellmask, m):
    """Precompute, per target: the to-witness distance field (reversed
    Dijkstra) and, over a greedy DISJOINT-witness pivot family, the
    direction-minimized set-to-set distance matrix.

    Returns (dtw, pivots, piv_dist):
      dtw[b][i]      = d(cell i -> nearest witness of target-bit b)
      pivots         = list of target bits with pairwise disjoint witnesses
      piv_dist[a][b] = lower bound on travel between witness sets of pivots
    """
    witness = [[] for _ in range(m)]
    for i, mk in enumerate(cellmask):
        for b in range(m):
            if mk & (1 << b):
                witness[b].append(i)

    # d(cell -> W_b): multi-source Dijkstra on the REVERSED graph.
    dtw = [_multi_source_dijkstra(rev, witness[b]) if witness[b]
           else [INF] * len(nodes) for b in range(m)]

    # Greedy pivot family: pairwise-disjoint witness sets (smallest first --
    # small sets are the hardest to serve incidentally, the ICAPS intuition).
    order = sorted(range(m), key=lambda b: len(witness[b]) or 1 << 30)
    pivots, used = [], set()
    for b in order:
        if not witness[b]:
            continue
        wset = set(witness[b])
        if wset & used:
            continue
        pivots.append(b)
        used |= wset

    # Set-to-set lower-bound distances between pivot witness regions:
    # one FORWARD multi-source Dijkstra per pivot, then direction-minimized.
    P = len(pivots)
    piv_dist = [[0.0] * P for _ in range(P)]
    fdist = {b: _multi_source_dijkstra(fwd, witness[b]) for b in pivots}
    for a in range(P):
        for bidx in range(a + 1, P):
            ba, bb = pivots[a], pivots[bidx]
            dab = min((fdist[ba][i] for i in witness[bb]), default=INF)
            dba = min((fdist[bb][i] for i in witness[ba]), default=INF)
            lb = min(dab, dba)
            piv_dist[a][bidx] = piv_dist[bidx][a] = lb
    return dtw, pivots, piv_dist


def _heuristic(i, mask, dtw, pivots, piv_dist, m):
    """max(h_single, h_pivot) -- both admissible, so the max is too."""
    # h_single: farthest still-owed witness set.
    h1 = 0.0
    for b in range(m):
        if not (mask & (1 << b)):
            d = dtw[b][i]
            if d > h1:
                h1 = d

    # h_pivot: Prim MST over {current cell} + unrevealed pivots.
    # rem holds positions into `pivots` of pivots not yet revealed.
    rem = [k for k, b in enumerate(pivots) if not (mask & (1 << b))]
    if not rem:
        return h1
    # nodes: 0 = current cell, 1..len(rem) = pivot regions
    n = len(rem) + 1
    in_tree = [False] * n
    key = [INF] * n
    key[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = min((k for k in range(n) if not in_tree[k]), key=lambda k: key[k])
        if key[u] == INF:
            return h1  # some pivot unreachable; h1 already reflects INF cases
        in_tree[u] = True
        total += key[u]
        for k in range(1, n):
            if in_tree[k]:
                continue
            if u == 0:
                w = dtw[pivots[rem[k - 1]]][i]
            else:
                w = piv_dist[rem[u - 1]][rem[k - 1]]
            if w < key[k]:
                key[k] = w
    return max(h1, total)


# ---------------------------------------------------------------------------
# The A* search over (cell, revealed-mask) states.
# ---------------------------------------------------------------------------

def solve_cover_walk(env_map, start, unknown, live_set, weight=1.0):
    """Optimal (weight=1) covering walk from ``start`` revealing every
    target in ``unknown``, never touching a live target.

    Returns (path_cells, schedule, unscoutable, stats) --
    path ends at the moment of the last reveal (truncation is inherent);
    schedule maps target -> cost-from-now; stats = dict(expanded, touched).
    Unscoutable targets are excluded from the goal mask and reported.
    """
    unknown_sorted = sorted(unknown, key=str)
    nodes, idx, fwd, rev, cellmask = _compile(env_map, live_set, unknown_sorted)
    if start not in idx:
        return None, {}, set(unknown), {"expanded": 0, "touched": 0}
    m = len(unknown_sorted)
    s = idx[start]

    dtw, pivots, piv_dist = _build_heuristics(nodes, fwd, rev, cellmask, m)

    # Scoutability: a target with no finite witness distance from the start
    # can never be revealed by this scout.
    unscoutable = {unknown_sorted[b] for b in range(m) if dtw[b][s] == INF}
    goal_bits = [b for b in range(m) if dtw[b][s] < INF]
    if not goal_bits:
        return None, {}, unscoutable, {"expanded": 0, "touched": 0}
    FULL = 0
    for b in goal_bits:
        FULL |= 1 << b
    reachable_mask = FULL  # bits we actually require

    def h(i, mask):
        return _heuristic(i, mask | ~reachable_mask & ((1 << m) - 1),
                          dtw, pivots, piv_dist, m)

    start_mask = cellmask[s] & reachable_mask  # standing here already reveals
    g_best = {(start_mask, s): 0.0}
    parent = {(start_mask, s): None}
    h0 = h(s, start_mask)
    heap = [(weight * h0, 0.0, start_mask, s)]
    expanded = 0
    goal_state = None

    while heap:
        f, g, mask, i = heapq.heappop(heap)
        if g > g_best.get((mask, i), INF):
            continue
        expanded += 1
        if mask & reachable_mask == reachable_mask:
            goal_state = (mask, i)
            break
        for j, w in fwd[i]:
            nm = (mask | cellmask[j]) & reachable_mask
            ng = g + w
            key = (nm, j)
            if ng < g_best.get(key, INF):
                g_best[key] = ng
                parent[key] = (mask, i)
                heapq.heappush(heap, (ng + weight * h(j, nm), ng, nm, j))

    stats = {"expanded": expanded, "touched": len(g_best)}
    if goal_state is None:
        return None, {}, unscoutable, stats

    # Reconstruct cells; then derive the reveal schedule by walking forward.
    chain = []
    st = goal_state
    while st is not None:
        chain.append(st)
        st = parent[st]
    chain.reverse()
    path = [nodes[i] for _mask, i in chain]

    schedule = {}
    seen = chain[0][0]
    clock = 0.0
    for k in range(1, len(chain)):
        (_pm, pi), (cm, ci) = chain[k - 1], chain[k]
        # edge cost between consecutive cells
        cost = next(w for j, w in fwd[pi] if j == ci)
        clock += cost
        new = cm & ~seen
        if new:
            for b in range(m):
                if new & (1 << b):
                    schedule[unknown_sorted[b]] = clock
            seen = cm
    return path, schedule, unscoutable, stats


# ---------------------------------------------------------------------------
# Scout planning wrapper (cache + agent outputs), then the policy.
# ---------------------------------------------------------------------------

def _plan_scout(env_map, scout, unknown_live, live_set,
                exact_target_cap, suboptimal_weight, verbose):
    belief_key = (frozenset(unknown_live), env_map.number_of_edges())

    if getattr(scout, "_wrp_key", None) == belief_key:
        pp = scout.planned_path
        if pp and pp[0] == scout.position:
            return                       # mid-walk, belief unchanged
        if not pp or len(pp) == 1:
            return                       # walk finished; nothing new to learn
        # plan exists but is stale (position mismatch) -> fall through

    scout.planned_path = []
    scout.reveal_schedule = {}
    scout.unscoutable = set()
    scout._wrp_key = belief_key

    if not unknown_live:
        if verbose:
            print("  [wrp] scout: no unknown targets -> idle")
        return

    weight = 1.0 if len(unknown_live) <= exact_target_cap else suboptimal_weight
    path, schedule, unscoutable, stats = solve_cover_walk(
        env_map, scout.position, unknown_live, live_set, weight=weight)

    scout.unscoutable = unscoutable
    if verbose and unscoutable:
        print(f"  [wrp] scout: UNSCOUTABLE targets {sorted(unscoutable, key=str)}")
    if path is None or len(path) < 2:
        if verbose:
            print("  [wrp] scout: nothing scoutable -> idle")
        return

    scout.planned_path = path
    scout.reveal_schedule = schedule
    if verbose:
        mode = "A*" if weight == 1.0 else f"wA*({weight})"
        cost = max(schedule.values()) if schedule else 0.0
        print(f"  [wrp] scout walk ({mode}): cost {cost:.2f}, "
              f"{len(path)} cells, expanded {stats['expanded']} states")
        for t in sorted(schedule, key=schedule.get):
            print(f"  [wrp]   reveal ETA  target {t}: t+{schedule[t]:.2f}")


# ---------------------------------------------------------------------------
# Attacker layer (PTSP-style): exact visit walks + hedge point + gamble gate.
#
# Revealed targets receive stable, unique claims. This is necessary because
# generalized capability sets may overlap: several agents can service the same
# target type. After claiming, the original WRP/PTSP visit-order machinery is
# unchanged.
#
#   (1) is solved exactly by the same (cell, visited-mask) A* as the scout,
#       with singleton "witness" sets = the job cells themselves.
#   (2) is the a-priori/probabilistic-TSP hedge: walk to the cell minimizing
#       sum_j exp(-eta_j / tau) * d(cell -> j) over pending scoutable unknowns
#       (eta_j from the scout's published reveal_schedule). With no
#       target-specific prior supplied, pending unknowns receive equal weight.
#   (3) fires only for scout.unscoutable targets (or a scout-less roster):
#       with death = 100 time-units, gambling on a scoutable unknown is never
#       worth it -- but an unscoutable target must be probed, and any
#       contact reveals the type (service or agent loss), so the cheapest free
#       attacker probes it and the reveal routes a capable survivor next.
# ---------------------------------------------------------------------------


def solve_visit_walk(env_map, start, points, avoid, weight=1.0):
    """Optimal walk from ``start`` stepping on every cell in ``points``
    (order chosen by the search), never touching an ``avoid`` cell.

    Same (cell, visited-mask) A* as :func:`solve_cover_walk`, with singleton
    witness sets: cellmask has point b's bit only at point b itself, h_single
    is distance-to-farthest-remaining-point and h_pivot is the MST bound over
    the remaining points (singletons are trivially pairwise disjoint).

    Returns (path, arrivals, unreachable, stats); ``arrivals`` maps each
    reached point -> cost-from-now. Reachable points are always all served;
    unreachable ones are reported and skipped.
    """
    pts = sorted(set(points), key=str)
    avoid = set(avoid) - set(pts) - {start}
    m = len(pts)
    if m == 0:
        return [start], {}, set(), {"expanded": 0, "touched": 0}

    nodes = [n for n in env_map.nodes() if n not in avoid]
    idx = {n: i for i, n in enumerate(nodes)}
    if start not in idx:
        return None, {}, set(pts), {"expanded": 0, "touched": 0}
    fwd = [[] for _ in nodes]
    rev = [[] for _ in nodes]
    for u, v, d in env_map.edges(data="distance"):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        fwd[iu].append((iv, d))
        rev[iv].append((iu, d))

    cellmask = [0] * len(nodes)
    for b, ptc in enumerate(pts):
        if ptc in idx:
            cellmask[idx[ptc]] |= 1 << b

    s = idx[start]
    dtw = [_multi_source_dijkstra(rev, [idx[ptc]]) if ptc in idx
           else [INF] * len(nodes) for ptc in pts]
    unreachable = {pts[b] for b in range(m) if dtw[b][s] == INF}
    goal_bits = [b for b in range(m) if dtw[b][s] < INF]
    if not goal_bits:
        return None, {}, unreachable, {"expanded": 0, "touched": 0}
    FULL = 0
    for b in goal_bits:
        FULL |= 1 << b

    pivots = goal_bits  # singletons: every reachable point is a pivot
    P = len(pivots)
    piv_dist = [[0.0] * P for _ in range(P)]
    for a2 in range(P):
        for b2 in range(a2 + 1, P):
            pa, pb = pivots[a2], pivots[b2]
            ia, ib = idx[pts[pa]], idx[pts[pb]]
            lb = min(dtw[pb][ia], dtw[pa][ib])
            piv_dist[a2][b2] = piv_dist[b2][a2] = lb

    unreach_bits = ((1 << m) - 1) & ~FULL

    def h(i, mask):
        return _heuristic(i, mask | unreach_bits, dtw, pivots, piv_dist, m)

    start_mask = cellmask[s] & FULL
    g_best = {(start_mask, s): 0.0}
    parent = {(start_mask, s): None}
    heap = [(weight * h(s, start_mask), 0.0, start_mask, s)]
    expanded = 0
    goal_state = None
    while heap:
        f, g, mask, i = heapq.heappop(heap)
        if g > g_best.get((mask, i), INF):
            continue
        expanded += 1
        if mask & FULL == FULL:
            goal_state = (mask, i)
            break
        for j, w in fwd[i]:
            nm = (mask | cellmask[j]) & FULL
            ng = g + w
            key = (nm, j)
            if ng < g_best.get(key, INF):
                g_best[key] = ng
                parent[key] = (mask, i)
                heapq.heappush(heap, (ng + weight * h(j, nm), ng, nm, j))

    stats = {"expanded": expanded, "touched": len(g_best)}
    if goal_state is None:
        return None, {}, unreachable, stats

    chain = []
    st = goal_state
    while st is not None:
        chain.append(st)
        st = parent[st]
    chain.reverse()
    path = [nodes[i] for _mask, i in chain]

    arrivals = {}
    seen = chain[0][0]
    clock = 0.0
    for k in range(1, len(chain)):
        (_pm, pi), (cm, ci) = chain[k - 1], chain[k]
        clock += next(w for j, w in fwd[pi] if j == ci)
        new = cm & ~seen
        if new:
            for b in range(m):
                if new & (1 << b):
                    arrivals[pts[b]] = clock
            seen = cm
    return path, arrivals, unreachable, stats


def _dist_field_to(env_map, goal, avoid):
    """d(cell -> goal) for every cell, avoiding ``avoid`` (goal kept)."""
    avoid = set(avoid) - {goal}
    nodes = [n for n in env_map.nodes() if n not in avoid]
    idx = {n: i for i, n in enumerate(nodes)}
    rev = [[] for _ in nodes]
    for u, v, d in env_map.edges(data="distance"):
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None:
            continue
        rev[iv].append((iu, d))
    if goal not in idx:
        return {}
    dist = _multi_source_dijkstra(rev, [idx[goal]])
    return {nodes[i]: dist[i] for i in range(len(nodes)) if dist[i] < INF}


def _hedge_cell(env_map, scoutable_unknown, live_set, schedule, tau):
    """The cell minimizing the imminence-weighted expected distance to the
    pending scoutable unknown targets. Ties: cover more targets first."""
    fields = {}
    for t in scoutable_unknown:
        fields[t] = _dist_field_to(env_map, t, live_set)
    weights = {t: math.exp(-max(schedule.get(t, 0.0), 0.0) / tau)
               for t in scoutable_unknown}
    best, best_key = None, None
    for v in env_map.nodes():
        if v in live_set:
            continue
        score, cover = 0.0, 0
        for t in scoutable_unknown:
            d = fields[t].get(v)
            if d is None:
                continue
            cover += 1
            score += weights[t] * d
        if cover == 0:
            continue
        k = (-cover, score)
        if best_key is None or k < best_key:
            best, best_key = v, k
    return best


def _plan_attackers_ptsp(env_map, attackers, live_targets, live_set,
                         unknown_live, state, hedge_tau, verbose):
    """Layers 1-3 for every living, at-a-node attacker (cache-aware)."""
    revealed_live = [t for t in live_targets if t not in set(unknown_live)]
    unknown_set = set(unknown_live)
    n_edges = env_map.number_of_edges()

    # Gamble-eligible targets: unscoutable per the scout, or everything
    # unknown if this roster never had a scout to wait for.
    if state.get("scout_seen"):
        gamble_targets = sorted(unknown_set & state.get("unscoutable", set()),
                                key=str)
    else:
        gamble_targets = sorted(unknown_set, key=str)
    schedule = state.get("schedule", {})

    # Probe deferral: while scoutable reveals are still pending, hold off on
    # gambles -- reveals are imminent and they determine which attacker's
    # death would be cheapest (the informed prober choice). Once every
    # remaining unknown is unscoutable (or the roster never had a scout),
    # probing is the only way forward.
    probes_active = len(unknown_set) > 0 and \
        len(set(gamble_targets)) == len(unknown_set)

    # Maintain stable unique claims across replans, including while a claimed
    # agent is in transit and therefore absent from this policy call.
    claims = state.setdefault("claims", {})
    roster = state.get("roster", attackers)
    eligible = [
        a for a in roster
        if a.alive and a.can_engage
        and not (a is state.get("designated_scout")
                 and not state.get("scouting_done", False))
    ]
    for target, owner in list(claims.items()):
        target_type = env_map.nodes[target].get("rps_type", UNKNOWN_TYPE) \
            if env_map.has_node(target) else UNKNOWN_TYPE
        if (target not in revealed_live or not owner.alive
                or not owner.can_service(target_type) or owner not in eligible):
            del claims[target]

    claimed_counts = {id(a): 0 for a in eligible}
    for owner in claims.values():
        claimed_counts[id(owner)] = claimed_counts.get(id(owner), 0) + 1

    for target in sorted(revealed_live, key=str):
        if target in claims:
            continue
        target_type = env_map.nodes[target]["rps_type"]
        candidates = [a for a in eligible if a.can_service(target_type)]
        if not candidates:
            continue
        field = _dist_field_to(env_map, target, live_set)
        reachable = [a for a in candidates if a.position in field]
        if not reachable:
            continue
        owner = min(
            reachable,
            key=lambda a: (
                field[a.position],
                claimed_counts.get(id(a), 0),
                state["roster_order"].get(id(a), 10**9),
            ),
        )
        claims[target] = owner
        claimed_counts[id(owner)] = claimed_counts.get(id(owner), 0) + 1

    free = []          # attackers with no committed jobs this epoch
    for a in attackers:
        jobs = sorted(
            [target for target, owner in claims.items() if owner is a],
            key=str,
        )
        key = (tuple(jobs), frozenset(unknown_set), n_edges)
        if getattr(a, "_att_key", None) == key:
            pp = a.planned_path
            if pp and pp[0] == a.position:
                continue                     # walking a still-valid plan
            if not pp or len(pp) == 1:
                if not jobs:
                    free.append(a)           # idle at hedge; may re-task below
                continue
        a._att_key = key
        a.planned_path = []
        if jobs:
            path, arrivals, unreach, stats = solve_visit_walk(
                env_map, a.position, jobs, live_set - set(jobs))
            if path and len(path) >= 2:
                a.planned_path = path
            if verbose:
                print(f"  [wrp-att] {capability_label(a.capabilities)} "
                      f"@ {a.position}: "
                      f"route {len(jobs)} job(s), cost "
                      f"{max(arrivals.values()) if arrivals else 0:.2f}, "
                      f"expanded {stats['expanded']}")
        else:
            free.append(a)

    if not free:
        return

    # ---- Layer 3: probes for unscoutable targets (cheapest free attacker).
    assigned = set()
    for t in (gamble_targets if probes_active else []):
        field = _dist_field_to(env_map, t, live_set)
        cands = [a for a in free if id(a) not in assigned
                 and a.position in field]
        if not cands:
            continue
        prober = min(cands, key=lambda a: field[a.position])
        path, _arr, _unr, _st = solve_visit_walk(
            env_map, prober.position, [t], live_set - {t})
        if path and len(path) >= 2:
            prober.planned_path = path
            prober._att_key = None          # force re-plan after the probe
            assigned.add(id(prober))
            if verbose:
                print(f"  [wrp-att] {capability_label(prober.capabilities)} @ "
                      f"{prober.position}: PROBE unscoutable {t} "
                      f"(cost {field[prober.position]:.2f})")

    # ---- Layer 2: hedge the remaining free attackers while reveals pend.
    hedgers = [a for a in free if id(a) not in assigned]
    scoutable_unknown = sorted(unknown_set - set(gamble_targets), key=str)
    if not hedgers or not scoutable_unknown:
        return
    hkey = (frozenset(unknown_set), n_edges)
    if state.get("hedge_key") != hkey:
        state["hedge_key"] = hkey
        state["hedge_cell"] = _hedge_cell(env_map, scoutable_unknown,
                                          live_set, schedule, hedge_tau)
        if verbose:
            print(f"  [wrp-att] hedge point -> {state['hedge_cell']}")
    hc = state.get("hedge_cell")
    if hc is None:
        return
    for a in hedgers:
        if a.position == hc:
            continue
        path, _arr, _unr, _st = solve_visit_walk(
            env_map, a.position, [hc], live_set)
        if path and len(path) >= 2:
            a.planned_path = path


# ---------------------------------------------------------------------------
# Policy factory / entry point.
# ---------------------------------------------------------------------------

def make_policy(exact_target_cap=DEFAULT_EXACT_TARGET_CAP,
                suboptimal_weight=DEFAULT_SUBOPT_WEIGHT,
                attacker_mode="ptsp",
                hedge_tau=DEFAULT_HEDGE_TAU):
    """Build a ``replan``-signature policy.

    attacker_mode:
      "ptsp"      -- route/hedge/gamble attacker layer (default).
      "baseline1" -- baseline1's greedy attackers (ablation reference).
    """
    if attacker_mode not in ("ptsp", "baseline1"):
        raise ValueError(f"unknown attacker_mode {attacker_mode!r}")
    state = {}   # shared belief, roster, claims, and scout phase

    def replan(env_map, agents, reward_ratio=1.0, obs_discount_factor=1.0,
               sample_recursion=0, sample_num_obstacle=0,
               sample_obstacle_hop=0, verbose=False):
        """WRP-A* covering-walk scout + PTSP attacker layer."""
        live_targets = [n for n, d in env_map.nodes(data=True)
                        if d.get("type") == "target_unreached"]
        live_set = set(live_targets)
        unknown_live = [t for t in live_targets
                        if env_map.nodes[t].get("rps_type",
                                                UNKNOWN_TYPE) == UNKNOWN_TYPE]

        # The simulator currently passes only living agents at nodes. Remember
        # object references so selection and target claims remain stable while
        # other agents are in transit.
        roster = state.setdefault("roster", [])
        known_ids = {id(a) for a in roster}
        for a in agents:
            if id(a) not in known_ids:
                roster.append(a)
                known_ids.add(id(a))
        state["roster_order"] = {id(a): i for i, a in enumerate(roster)}

        scout_candidates = [a for a in roster if a.alive and a.scout_capable]
        designated = state.get("designated_scout")
        if designated not in scout_candidates:
            designated = min(
                scout_candidates,
                key=lambda a: (
                    sum(value > 0 for value in a.capabilities),
                    state["roster_order"][id(a)],
                ),
                default=None,
            )
            state["designated_scout"] = designated
            state["scouting_done"] = designated is None

        at_node_ids = {id(a) for a in agents if a.alive}

        # The designated scout follows WRP exclusively during the scout phase.
        if (designated is not None and id(designated) in at_node_ids
                and live_targets and not state.get("scouting_done", False)):
            _plan_scout(env_map, designated, unknown_live, live_set,
                        exact_target_cap, suboptimal_weight, verbose)
            state["scout_seen"] = True
            state["unscoutable"] = set(
                getattr(designated, "unscoutable", ()))
            state["schedule"] = dict(
                getattr(designated, "reveal_schedule", {}))
            scoutable_unknown = (
                set(unknown_live) - state["unscoutable"])
            if not scoutable_unknown and len(designated.planned_path) < 2:
                state["scouting_done"] = True
                designated._att_key = None
                if verbose:
                    print("  [wrp] scouting complete; designated scout joins "
                          "the attacker layer")
        elif designated is not None and not live_targets:
            designated.planned_path = []
            state["scouting_done"] = True

        attackers = [
            a for a in agents
            if a.alive and a.can_engage
            and not (a is designated and not state.get("scouting_done", False))
        ]
        if not attackers or not live_targets:
            return
        if attacker_mode == "baseline1":
            for a in attackers:
                a.planned_path = []
                _route_attacker(env_map, a, live_targets, live_set, verbose)
            return
        _plan_attackers_ptsp(env_map, attackers, live_targets, live_set,
                             unknown_live, state, hedge_tau, verbose)

    replan.__name__ = "baseline2_wrp_replan"
    replan.hyperparameters = {
        "exact_target_cap": exact_target_cap,
        "suboptimal_weight": suboptimal_weight,
        "attacker_mode": attacker_mode,
        "hedge_tau": hedge_tau,
    }
    # Read-only-by-convention hook for tests and experiment diagnostics.
    replan.state = state
    return replan


#: Default entry point.
replan = make_policy()
