"""Test V5 blob-growth generator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from map_generator import generate_corridor_map, _validate_map
import numpy as np
import networkx as nx

# Multi-seed test for 12x12
print("=== 12x12 multi-seed ===")
valid_count = 0
total = 30
for seed in range(42, 42 + total):
    try:
        md = generate_corridor_map(grid_size=12, num_blobs=4, seed=seed)
        g = md['env_graph']
        ab = set()
        for b in md['blobs']:
            ab.update(b)
        v = _validate_map(md)
        if v:
            valid_count += 1
        blob_sizes = sorted([len(b) for b in md['blobs']], reverse=True)
        sp = nx.shortest_path(g, md['source'], md['target'], weight='distance')
        sp_len = sum(g.edges[sp[i], sp[i+1]]['distance'] for i in range(len(sp)-1))

        # Quick blockage check
        cp_set = set(tuple(sorted(e)) for e in md['chokepoints'])
        rng_val = np.random.RandomState(seed + 9999)
        baseline_cost = sp_len
        detours = 0
        for _ in range(30):
            er = []
            rolls = rng_val.rand(len(md['chokepoints']))
            for i, edge in enumerate(md['chokepoints']):
                if rolls[i] < 0.5:
                    u, vv = edge
                    if g.has_edge(u, vv):
                        er.append((u, vv))
            if not er:
                continue
            bg = g.copy()
            bg.remove_edges_from(er)
            if not nx.has_path(bg, md['source'], md['target']):
                continue
            try:
                bsp = nx.shortest_path(bg, md['source'], md['target'], weight='distance')
                cost = sum(bg.edges[bsp[i], bsp[i+1]]['distance'] for i in range(len(bsp)-1))
                if cost > baseline_cost + 0.5:
                    detours += 1
            except:
                pass

        status = "VALID" if v else "fail"
        print(f"  seed={seed}: {status} blobs={blob_sizes} edges={g.number_of_edges()} "
              f"blob%={len(ab)/144*100:.0f} SP={sp_len:.0f} CPs={len(md['chokepoints'])} "
              f"detours={detours}")
    except Exception as e:
        print(f"  seed={seed}: ERROR {e}")

print(f"\n12x12 validation rate: {valid_count}/{total} = {valid_count/total*100:.0f}%")

# 14x14
print("\n=== 14x14 multi-seed ===")
valid14 = 0
for seed in range(42, 72):
    try:
        md = generate_corridor_map(grid_size=14, num_blobs=4, seed=seed)
        v = _validate_map(md)
        if v:
            valid14 += 1
    except:
        pass
print(f"14x14 validation rate: {valid14}/30 = {valid14/30*100:.0f}%")

# 16x16
print("\n=== 16x16 multi-seed ===")
valid16 = 0
for seed in range(42, 72):
    try:
        md = generate_corridor_map(grid_size=16, num_blobs=5, seed=seed)
        v = _validate_map(md)
        if v:
            valid16 += 1
    except:
        pass
print(f"16x16 validation rate: {valid16}/30 = {valid16/30*100:.0f}%")
