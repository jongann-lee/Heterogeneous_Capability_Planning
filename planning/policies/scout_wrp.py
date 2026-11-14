"""Watchman Route Problem covering walks shared by the evaluation baselines.

A* searches physical-node/visibility-mask states, using exact search through
twelve unknown tasks and weighted A* above that threshold.
"""

import heapq
import math

from simulation.visibility import visible_targets

INF = float("inf")
DEFAULT_EXACT_TARGET_CAP = 12
DEFAULT_SUBOPT_WEIGHT = 1.5


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
    for n in env_map:
        i = idx.get(n)
        if i is None:
            continue
        mk = 0
        for target in visible_targets(env_map, n, unknown_set):
            mk |= 1 << tbit[target]
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
