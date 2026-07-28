"""Debug specific failing seeds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from map_generator import generate_corridor_map, _validate_map
import numpy as np
import networkx as nx

for seed in [47, 58, 66, 69, 52, 65, 70]:
    md = generate_corridor_map(grid_size=12, seed=seed)
    g = md['env_graph']
    s, t = md['source'], md['target']
    cps = md['chokepoints']
    ab = set()
    for b in md['blobs']:
        ab.update(b)

    sp = nx.shortest_path(g, s, t, weight="distance")
    sp_len = sum(g.edges[sp[i], sp[i+1]]['distance'] for i in range(len(sp)-1))

    # Check each criterion
    # 1. CP adjacency
    bad_cps = 0
    for u, v in cps:
        u_adj = any((u[0]+dr, u[1]+dc) in ab for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
        v_adj = any((v[0]+dr, v[1]+dc) in ab for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
        if not u_adj and not v_adj:
            bad_cps += 1

    # 2. SP blob fraction
    blob_frac = sum(1 for n in sp if n in ab) / len(sp)

    # 3. CP coverage
    cp_set = set(tuple(sorted(e)) for e in cps)
    temp_g = g.copy()
    paths_through = 0
    total_p = 0
    for _ in range(10):
        try:
            p = nx.shortest_path(temp_g, s, t, weight="distance")
            total_p += 1
            pe = set(tuple(sorted((p[i], p[i+1]))) for i in range(len(p)-1))
            if pe & cp_set:
                paths_through += 1
            for i in range(len(p)-1):
                u, v = p[i], p[i+1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except:
            break
    cp_cov = paths_through/total_p if total_p > 0 else 0

    # 4. Blockage CV
    rng_val = np.random.RandomState(seed + 9999)
    sp_costs = []
    baseline_cost = sp_len
    for _ in range(30):
        er = []
        rolls = rng_val.rand(len(cps))
        for i, edge in enumerate(cps):
            if rolls[i] < 0.5:
                u, v = edge
                if g.has_edge(u, v):
                    er.append((u, v))
                elif g.has_edge(v, u):
                    er.append((v, u))
        if not er:
            sp_costs.append(baseline_cost)
            continue
        bg = g.copy()
        bg.remove_edges_from(er)
        if not nx.has_path(bg, s, t):
            continue
        try:
            bsp = nx.shortest_path(bg, s, t, weight='distance')
            cost = sum(bg.edges[bsp[i], bsp[i+1]]['distance'] for i in range(len(bsp)-1))
            sp_costs.append(cost)
        except:
            continue
    cv = np.std(sp_costs)/np.mean(sp_costs) if sp_costs else 0
    detours = sum(1 for c in sp_costs if c > baseline_cost + 0.5)

    fail_reasons = []
    if bad_cps > 0:
        fail_reasons.append(f"bad_cps={bad_cps}")
    if blob_frac > 0.3:
        fail_reasons.append(f"blob_sp={blob_frac:.2f}")
    if cp_cov < 0.5:
        fail_reasons.append(f"cp_cov={cp_cov:.2f}")
    if cv < 0.03:
        fail_reasons.append(f"cv={cv:.4f}")
    if detours < 2:
        fail_reasons.append(f"detours={detours}")
    if len(sp_costs) < 15:
        fail_reasons.append(f"trials={len(sp_costs)}")

    valid = _validate_map(md)
    print(f"seed={seed}: {'VALID' if valid else 'FAIL'} SP={sp_len:.0f} "
          f"reasons=[{', '.join(fail_reasons)}] CPs={len(cps)}")
