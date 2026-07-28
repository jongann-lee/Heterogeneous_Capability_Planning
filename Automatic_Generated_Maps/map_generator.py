"""
Simplified map generator for benchmarking RepeatedTopK vs Shortest Path agents.

Algorithm:
1. Sample evenly-spread blob centers on the grid.
2. Grow blobs outward via BFS (distance-ordered → roughly convex shapes),
   maintaining a 1-node gap between blobs.
3. Remove all blob↔corridor boundary edges, then re-add 2-3 per blob.
3b. Fix diagonal disconnects: where two corridor nodes are diagonally
    adjacent with blob nodes blocking both grid paths, convert one
    blob node to corridor.
4. Place chokepoints on corridor edges that are adjacent to a blob and
   at least 35 % of the way along the shortest path.
"""

import sys, os, heapq
import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.height_graph_generation import HeightMapGrid

# ── helpers ────────────────────────────────────────────────────────────────

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(r, c, gs):
    return 0 <= r < gs and 0 <= c < gs


def _neighbors4(r, c, gs):
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(nr, nc, gs):
            yield (nr, nc)


# ── 1. sample blob centres ────────────────────────────────────────────────

def _sample_centers(gs, num_blobs, source, target, rng, min_sep_frac=0.30, margin=2):
    """Rejection-sample centres that are spread apart and away from S/T."""
    min_sep = max(3, int(min_sep_frac * gs))

    forbidden = set()
    for base in (source, target):
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                r, c = base[0] + dr, base[1] + dc
                if _in_bounds(r, c, gs):
                    forbidden.add((r, c))

    centers = []
    for _ in range(num_blobs):
        for _try in range(500):
            cand = (rng.randint(2, gs - 2), rng.randint(2, gs - 2))
            if cand in forbidden:
                continue
            if any(abs(cand[0] - c[0]) + abs(cand[1] - c[1]) < min_sep for c in centers):
                continue
            centers.append(cand)
            break
        else:
            raise ValueError(f"Could not place blob centre {len(centers)+1}/{num_blobs}")
    return centers


# ── 2. grow blobs ─────────────────────────────────────────────────────────

def _grow_blobs(gs, centers, coverage, source, target, rng):
    """
    BFS-grow blobs from centres (distance-ordered → roughly convex).
    Keeps a 1-node gap between distinct blobs.  Grid border row/col is
    forbidden so perimeter corridors stay open.
    """
    total_target = int(coverage * gs * gs)
    base = total_target // len(centers)

    # per-blob target sizes (±25 %)
    sizes = [max(5, int(base * rng.uniform(0.75, 1.25))) for _ in centers]

    claimed = {}          # node → blob index
    forbidden = {source, target}

    # reserve grid border
    for i in range(gs):
        for node in [(0, i), (gs - 1, i), (i, 0), (i, gs - 1)]:
            forbidden.add(node)

    blobs = [[] for _ in centers]
    queues = []

    for bi, ctr in enumerate(centers):
        claimed[ctr] = bi
        blobs[bi].append(ctr)
        pq = []
        for nb in _neighbors4(*ctr, gs):
            heapq.heappush(pq, (1, rng.random(), nb))
        queues.append(pq)

    # round-robin: one node per blob per round
    active = set(range(len(centers)))
    stale = 0
    while active and stale < len(centers) * 3:
        progress = False
        for bi in list(active):
            if len(blobs[bi]) >= sizes[bi]:
                active.discard(bi)
                continue
            ctr = centers[bi]
            while queues[bi]:
                _, _, node = heapq.heappop(queues[bi])
                if node in claimed or node in forbidden:
                    continue
                # gap constraint: no 4-neighbour belongs to a *different* blob
                if any(claimed.get(nb, bi) != bi for nb in _neighbors4(*node, gs)
                       if nb in claimed):
                    continue
                claimed[node] = bi
                blobs[bi].append(node)
                for nb in _neighbors4(*node, gs):
                    if nb not in claimed and nb not in forbidden:
                        d = abs(nb[0] - ctr[0]) + abs(nb[1] - ctr[1])
                        heapq.heappush(queues[bi], (d, rng.random(), nb))
                progress = True
                break
            else:
                active.discard(bi)
        stale = 0 if progress else stale + 1

    return [b for b in blobs if len(b) >= 3]


# ── 3. pick spatially-spread entry edges ──────────────────────────────────

def _pick_entries(boundary_edges, blob_set, num):
    """Greedy farthest-point selection among boundary edges."""
    # for each edge, identify the corridor-side endpoint
    items = []
    for u, v in boundary_edges:
        corridor = v if u in blob_set else u
        items.append(((u, v), corridor))

    if len(items) <= num:
        return [e for e, _ in items]

    chosen = [items[0]]
    rest = items[1:]
    while len(chosen) < num and rest:
        best_i, best_d = 0, -1
        for i, (_, c) in enumerate(rest):
            d = min(abs(c[0] - ch[0]) + abs(c[1] - ch[1]) for _, ch in chosen)
            if d > best_d:
                best_d, best_i = d, i
        chosen.append(rest.pop(best_i))
    return [e for e, _ in chosen]


# ── 3b. fix diagonal disconnects ─────────────────────────────────────────

def _fix_diagonal_disconnects(graph, blob_set, blobs, gs):
    """
    Find pairs of corridor nodes that are diagonally adjacent where both
    shared grid neighbours are blob nodes.  Convert one blob node to
    corridor so they can connect through the grid.

    E.g. corridor at (r,c) and (r+1,c+1) share neighbours (r,c+1) and
    (r+1,c).  If both are blob nodes, pick one, convert it to corridor,
    remove its blob edges, and add corridor edges.
    """
    corridor_nodes = {n for n in graph.nodes() if n not in blob_set}

    node_to_blob = {}
    for bi, b in enumerate(blobs):
        for n in b:
            node_to_blob[n] = bi

    _DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    converted = set()

    for (r, c) in list(corridor_nodes):
        for dr, dc in _DIAGS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, gs):
                continue
            diag = (nr, nc)
            if diag not in corridor_nodes:
                continue

            # two shared grid neighbours
            mid1 = (r, nc)
            mid2 = (nr, c)

            if mid1 not in blob_set or mid2 not in blob_set:
                continue  # at least one path exists, no fix needed

            # both blocked by blob — convert one
            # prefer one already converted (free), else pick from larger blob
            pick = None
            if mid1 in converted:
                continue  # already fixed
            if mid2 in converted:
                continue  # already fixed

            b1 = node_to_blob.get(mid1)
            b2 = node_to_blob.get(mid2)
            if b1 is not None and b2 is not None:
                pick = mid1 if len(blobs[b1]) >= len(blobs[b2]) else mid2
            else:
                pick = mid1

            # convert pick: blob → corridor
            converted.add(pick)
            blob_set.discard(pick)
            bi = node_to_blob.pop(pick, None)
            if bi is not None:
                try:
                    blobs[bi].remove(pick)
                except ValueError:
                    pass
            graph.nodes[pick]["height"] = 0.0

            # remove blob-internal edges
            for nb in list(graph.neighbors(pick)):
                if nb in blob_set:
                    graph.remove_edge(pick, nb)

            # add corridor edges to grid neighbours that are corridor
            for nb in _neighbors4(*pick, gs):
                if nb in graph and nb not in blob_set and not graph.has_edge(pick, nb):
                    h_u = graph.nodes[pick]["height"]
                    h_v = graph.nodes[nb]["height"]
                    graph.add_edge(pick, nb,
                                   distance=1.0 + 2.0 * abs(h_u - h_v),
                                   observed_edge=False)

            corridor_nodes.add(pick)


# ── 4. chokepoint placement ───────────────────────────────────────────────

def _find_corridors(graph, blob_set):
    """
    Find corridor chains: maximal connected sequences of corridor nodes
    where EVERY node has exactly 2 corridor neighbours (strict narrow passage).

    Returns a list of corridors, each being an ordered list of nodes.
    """
    corridor_nodes = {n for n in graph.nodes() if n not in blob_set}

    # corridor-corridor degree
    cor_deg = {}
    for n in corridor_nodes:
        cor_deg[n] = sum(1 for nb in graph.neighbors(n) if nb in corridor_nodes)

    # strictly degree-2 corridor nodes only
    chain_nodes = {n for n, d in cor_deg.items() if d == 2}

    # walk connected components of chain_nodes
    visited = set()
    corridors = []

    for start in chain_nodes:
        if start in visited:
            continue

        chain = [start]
        visited.add(start)

        for direction in (0, 1):
            current = start
            while True:
                nxt = None
                for nb in graph.neighbors(current):
                    if nb in chain_nodes and nb not in visited:
                        nxt = nb
                        break
                if nxt is None:
                    break
                visited.add(nxt)
                if direction == 0:
                    chain.insert(0, nxt)
                else:
                    chain.append(nxt)
                current = nxt

        if len(chain) >= 2:
            corridors.append(chain)

    return corridors


def _place_chokepoints(graph, source, target, blob_set, num_cp, rng):
    """
    1. Find corridor chains (sequences of narrow-passage nodes).
    2. From each corridor, take the last 2-3 edges as chokepoints.
    """
    corridors = _find_corridors(graph, blob_set)

    chokepoints = []
    for chain in corridors:
        # edges along this chain
        edges = []
        for i in range(len(chain) - 1):
            u, v = chain[i], chain[i + 1]
            if graph.has_edge(u, v):
                edges.append((u, v))

        if len(edges) == 0:
            continue

        # take last 2-3 edges of the corridor
        n_take = min(3, len(edges))
        chunk = edges[-n_take:]

        for e in chunk:
            chokepoints.append(e)

    # deduplicate
    seen = set()
    unique = []
    for u, v in chokepoints:
        ek = tuple(sorted((u, v)))
        if ek not in seen:
            seen.add(ek)
            unique.append((u, v))

    # cap at num_cp, shuffle to spread if we have too many
    if len(unique) > num_cp:
        rng.shuffle(unique)
        unique = unique[:num_cp]

    return unique


# ── main entry point ──────────────────────────────────────────────────────

def generate_corridor_map(grid_size=12, num_blobs=4, blob_coverage=0.45,
                          num_chokepoints=18, block_prob=0.5, seed=None):
    rng = np.random.RandomState(seed)
    source, target = (0, 0), (grid_size - 1, grid_size - 1)

    # 1 centres → 2 grow → build HeightMapGrid
    centers = _sample_centers(grid_size, num_blobs, source, target, rng)
    blobs   = _grow_blobs(grid_size, centers, blob_coverage, source, target, rng)
    if len(blobs) < 2:
        raise ValueError("Too few blobs grown")

    blob_set = set(n for b in blobs for n in b)

    hmap = HeightMapGrid(m=grid_size, n=grid_size)
    for b in blobs:
        hmap.add_plataeu(b)
    hmap.calculate_distances()
    hmap.calculate_simple_visibility(blobs)

    # 3 remove boundary edges, re-add 2-3 per blob
    removed = {}                       # sorted_edge → attrs
    for u, v in list(hmap.G.edges()):
        if (u in blob_set) != (v in blob_set):
            ek = tuple(sorted((u, v)))
            removed[ek] = dict(hmap.G.edges[u, v])
            hmap.G.remove_edge(u, v)

    per_blob_boundary = {}             # blob_idx → [sorted_edge, …]
    for ek in removed:
        u, v = ek
        blob_node = u if u in blob_set else v
        for bi, b in enumerate(blobs):
            if blob_node in set(b):
                per_blob_boundary.setdefault(bi, []).append(ek)
                break

    for bi, edges in per_blob_boundary.items():
        n_entries = 2 if len(blobs[bi]) < 12 else 3
        n_entries = min(n_entries, len(edges))
        for e in _pick_entries(edges, blob_set, n_entries):
            u, v = e
            if not hmap.G.has_edge(u, v):
                hmap.G.add_edge(u, v, **removed[e])

    if not nx.has_path(hmap.G, source, target):
        raise ValueError("Graph disconnected after blob construction")

    # 3b fix diagonal disconnects
    _fix_diagonal_disconnects(hmap.G, blob_set, blobs, grid_size)

    # refresh visibility (prune edges that no longer exist)
    hmap.calculate_simple_visibility(blobs)
    for node in hmap.G.nodes():
        vis = hmap.G.nodes[node].get("visible_edges", [])
        hmap.G.nodes[node]["visible_edges"] = [
            e for e in vis
            if hmap.G.has_edge(e[0], e[1]) or hmap.G.has_edge(e[1], e[0])
        ]

    env_graph = hmap.get_graph()

    # 4 chokepoints
    chokepoints = _place_chokepoints(env_graph, source, target, blob_set,
                                     num_chokepoints, rng)
    # ensure edge attrs
    for u, v in chokepoints:
        e = env_graph.edges[u, v]
        if "distance" not in e:
            h_u = env_graph.nodes[u]["height"]
            h_v = env_graph.nodes[v]["height"]
            e["distance"] = 1.0 + 2.0 * abs(h_u - h_v)
        if "observed_edge" not in e:
            e["observed_edge"] = False

    return {
        "env_graph":    env_graph,
        "blobs":        blobs,
        "chokepoints":  chokepoints,
        "source":       source,
        "target":       target,
        "block_prob":   block_prob,
        "seed":         seed,
        "grid_size":    grid_size,
    }


# ── validation (lightweight) ──────────────────────────────────────────────

def _validate_map(map_data, min_cp=6, blockage_trials=30, min_cv=0.03):
    g  = map_data["env_graph"]
    s, t = map_data["source"], map_data["target"]
    cps  = map_data["chokepoints"]
    bp   = map_data.get("block_prob", 0.5)

    if not nx.has_path(g, s, t) or len(cps) < min_cp:
        return False

    blob_set = set(n for b in map_data["blobs"] for n in b)

    # SP shouldn't mostly go through blobs
    sp = nx.shortest_path(g, s, t, weight="distance")
    if sum(1 for n in sp if n in blob_set) / len(sp) > 0.3:
        return False

    # blockage impact: we need variance in SP costs
    rng = np.random.RandomState((map_data.get("seed", 0) or 0) + 9999)
    baseline = sum(g.edges[sp[i], sp[i+1]]["distance"] for i in range(len(sp) - 1))
    costs = []
    for _ in range(blockage_trials):
        to_rm = [(u, v) for (u, v), r in zip(cps, rng.rand(len(cps)))
                 if r < bp and g.has_edge(u, v)]
        if not to_rm:
            costs.append(baseline)
            continue
        bg = g.copy()
        bg.remove_edges_from(to_rm)
        if not nx.has_path(bg, s, t):
            continue
        try:
            bsp = nx.shortest_path(bg, s, t, weight="distance")
            costs.append(sum(bg.edges[bsp[i], bsp[i+1]]["distance"]
                             for i in range(len(bsp) - 1)))
        except Exception:
            continue

    if len(costs) < blockage_trials // 2:
        return False
    arr = np.array(costs)
    if arr.mean() <= 0 or arr.std() / arr.mean() < min_cv:
        return False
    return True


# ── suite generation ──────────────────────────────────────────────────────

def generate_map_suite(num_maps=50, seed_start=1000):
    configs = [
        # (grid_size, num_blobs, coverage, num_cp, block_prob, label)
        (12, 4, 0.45, 18, 0.50, "standard_12x12"),
        (12, 4, 0.45, 21, 0.50, "dense_cp_12x12"),
        (14, 4, 0.43, 18, 0.50, "standard_14x14"),
        (14, 4, 0.43, 21, 0.50, "dense_cp_14x14"),
        (16, 5, 0.40, 21, 0.50, "standard_16x16"),
        (16, 5, 0.40, 24, 0.50, "dense_cp_16x16"),
        (12, 4, 0.45, 18, 0.40, "lowblock_12x12"),
        (14, 4, 0.43, 18, 0.40, "lowblock_14x14"),
        (16, 5, 0.40, 21, 0.40, "lowblock_16x16"),
        (14, 6, 0.43, 18, 0.50, "six_blobs_14x14"),
    ]
    per_cfg = max(1, num_maps // len(configs))
    remainder = num_maps - per_cfg * len(configs)

    maps = []
    mid = 0
    for ci, (gs, nb, bc, nc, bp, label) in enumerate(configs):
        need = per_cfg + (1 if ci < remainder else 0)
        got, tries = 0, 0
        while got < need and tries < need * 100:
            seed = seed_start + mid
            tries += 1
            try:
                md = generate_corridor_map(grid_size=gs, num_blobs=nb,
                                           blob_coverage=bc, num_chokepoints=nc,
                                           block_prob=bp, seed=seed)
                md["label"], md["map_id"] = label, mid
                if _validate_map(md):
                    maps.append(md)
                    got += 1
            except Exception:
                pass
            mid += 1
    return maps


# ── quick smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    md = generate_corridor_map(grid_size=12, num_blobs=4, seed=42)
    g = md["env_graph"]
    blob_set = set(n for b in md["blobs"] for n in b)
    gs = md["grid_size"]

    print(f"Grid:  {gs}×{gs}")
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print(f"Blobs: {len(md['blobs'])}, "
          f"sizes: {sorted((len(b) for b in md['blobs']), reverse=True)}")
    print(f"Blob nodes: {len(blob_set)} "
          f"({len(blob_set)/g.number_of_nodes()*100:.0f}%)")
    print(f"Chokepoints: {len(md['chokepoints'])}")

    sp = nx.shortest_path(g, md["source"], md["target"], weight="distance")
    sp_len = sum(g.edges[sp[i], sp[i+1]]["distance"] for i in range(len(sp)-1))
    print(f"SP cost: {sp_len:.1f}, hops: {len(sp)-1}")
    print(f"Valid: {_validate_map(md)}")