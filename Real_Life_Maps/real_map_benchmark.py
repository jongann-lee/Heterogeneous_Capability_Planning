"""
Benchmark runner for the real terrain (DEM) map.

Automatically places obstacle ovals on the map based on shortest path and
diverse sampled paths, then runs both agents (SP replanning vs RepeatedTopK)
across many probabilistic blockage realizations.

Each oval acts as a single chokepoint unit — all its edges block/unblock together.
"""

import sys
import os
import time
import json
import csv
import argparse
import random

import numpy as np
import networkx as nx
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Single_Agent.repeated_topk import RepeatedTopK


# ---------------------------------------------------------------------------
# 1. DEM Loading
# ---------------------------------------------------------------------------

def get_grid_from_local_dem(file_path, n_size):
    """
    Loads a local DEM GeoTIFF and resamples to n_size x n_size.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(file_path) as dataset:
        data = dataset.read(
            1,
            out_shape=(n_size, n_size),
            resampling=Resampling.bilinear
        )
        if dataset.nodata is not None:
            data = np.where(data == dataset.nodata, np.nan, data)
        return data


def load_real_terrain(dem_path, n_size=64):
    """Loads DEM and applies the clockwise 90-degree rotation correction."""
    height_grid = get_grid_from_local_dem(dem_path, n_size)
    corrected = np.rot90(height_grid, k=-1)
    return corrected


# ---------------------------------------------------------------------------
# 2. Oval Placement
# ---------------------------------------------------------------------------

def determine_oval_shape(path, center_idx, lookback=3):
    """
    Examines the preceding `lookback` edges before center_idx in the path
    to decide the oval orientation.

    Returns (rx, ry):
        - All vertical   -> rx=5, ry=3  (wide, blocks vertical corridor)
        - All horizontal  -> rx=3, ry=5  (tall, blocks horizontal corridor)
        - Mixed           -> rx=4, ry=4
    """
    start = max(0, center_idx - lookback)
    edges = [(path[i], path[i + 1]) for i in range(start, center_idx)]

    if not edges:
        return 4, 4

    vertical = 0
    horizontal = 0
    for (r1, c1), (r2, c2) in edges:
        if c1 == c2 and r1 != r2:
            vertical += 1
        elif r1 == r2 and c1 != c2:
            horizontal += 1

    if vertical == len(edges) and len(edges) >= 2:
        return 5, 3
    elif horizontal == len(edges) and len(edges) >= 2:
        return 3, 5
    else:
        return 4, 4


def compute_oval_nodes(graph, cx, cy, rx, ry, source, target):
    """
    Computes the set of nodes inside the ellipse defined by center (cx, cy)
    and radii (rx, ry), matching the equation in RealTerrainGrid.add_obstacle:
        ((c - cx)^2 / rx^2) + ((r - cy)^2 / ry^2) <= 1
    Excludes source and target.
    """
    nodes = set()
    for node in graph.nodes():
        r, c = node
        if ((c - cx) ** 2 / rx ** 2) + ((r - cy) ** 2 / ry ** 2) <= 1:
            if node != source and node != target:
                nodes.add(node)
    return nodes


def select_centers_from_chunks(path, num_ovals, rng, buffer=6):
    """
    Divides path into `num_ovals` equal chunks and randomly samples one
    center from each chunk's interior, with a buffer from chunk boundaries
    to prevent oval overlap.
    """
    n = len(path)
    chunk_size = n // num_ovals
    centers = []

    for i in range(num_ovals):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size if i < num_ovals - 1 else n

        # Apply buffer within the chunk
        safe_start = chunk_start + buffer
        safe_end = chunk_end - buffer

        if safe_start >= safe_end:
            # Fallback: use smaller buffer
            safe_start = chunk_start + 2
            safe_end = chunk_end - 2

        if safe_start >= safe_end:
            # Last resort: use chunk midpoint
            mid = (chunk_start + chunk_end) // 2
            centers.append(path[mid])
        else:
            idx = rng.randint(safe_start, safe_end)
            centers.append(path[idx])

    return centers


def select_center_from_path_interior(path, rng, margin=8):
    """Selects one center from the interior of a path."""
    usable = path[margin:-margin]
    if len(usable) < 1:
        usable = path[3:-3]
    if len(usable) < 1:
        usable = path[1:-1]
    if len(usable) < 1:
        return path[len(path) // 2]
    idx = rng.randint(0, len(usable))
    return usable[idx]


def place_obstacle_ovals(env_graph, target_graph, source, target, seed=42,
                          num_sp_ovals=3, num_diverse_ovals=3):
    """
    Places obstacle ovals:
      - num_sp_ovals on the shortest path (one per chunk)
      - num_diverse_ovals on the next diverse sampled paths from the target graph

    Returns list of dicts: [{'center': (cx,cy), 'rx': int, 'ry': int, 'nodes': set}, ...]
    """
    rng = np.random.RandomState(seed)
    ovals = []

    # --- Shortest path ovals ---
    sp = nx.shortest_path(env_graph, source=source, target=target, weight='distance')
    sp_centers = select_centers_from_chunks(sp, num_ovals=num_sp_ovals, rng=rng, buffer=6)

    for center_node in sp_centers:
        center_idx = sp.index(center_node)
        rx, ry = determine_oval_shape(sp, center_idx)
        cx, cy = center_node[1], center_node[0]  # coordinate swap: (r,c) -> (cx=c, cy=r)
        nodes = compute_oval_nodes(env_graph, cx, cy, rx, ry, source, target)
        ovals.append({'center': (cx, cy), 'rx': rx, 'ry': ry, 'nodes': nodes,
                      'path_type': 'shortest_path'})

    # --- Diverse paths ovals ---
    diverse_paths = []
    if target_graph.has_edge(source, target):
        diverse_paths = target_graph.edges[source, target].get('diverse_paths', [])
    elif target_graph.has_edge(target, source):
        diverse_paths = target_graph.edges[target, source].get('diverse_paths', [])

    # Skip index 0 (shortest path duplicate), take next num_diverse_ovals
    alt_paths = diverse_paths[1:1 + num_diverse_ovals] if len(diverse_paths) > 1 else []

    for alt_path in alt_paths:
        # Ensure path goes source -> target
        if len(alt_path) > 0 and alt_path[0] != source:
            alt_path = alt_path[::-1]

        center_node = select_center_from_path_interior(alt_path, rng, margin=8)
        center_idx = alt_path.index(center_node)
        rx, ry = determine_oval_shape(alt_path, center_idx)
        cx, cy = center_node[1], center_node[0]
        nodes = compute_oval_nodes(env_graph, cx, cy, rx, ry, source, target)
        ovals.append({'center': (cx, cy), 'rx': rx, 'ry': ry, 'nodes': nodes,
                      'path_type': 'diverse_path'})

    return ovals


# ---------------------------------------------------------------------------
# 3. Blocking Model
# ---------------------------------------------------------------------------

def apply_oval_blockages(env_graph, ovals, blocked_mask):
    """
    Creates a blocked_env_graph by removing edges for active (blocked) ovals.

    Args:
        env_graph: clean environment graph (DiGraph)
        ovals: list of oval dicts with 'nodes' key
        blocked_mask: list of bool, True = this oval is blocked

    Returns:
        blocked_env_graph with edges removed and visible_edges updated
    """
    blocked_env_graph = env_graph.copy()

    # Collect all obstacle nodes from blocked ovals
    all_blocked_nodes = set()
    for i, oval in enumerate(ovals):
        if blocked_mask[i]:
            all_blocked_nodes.update(oval['nodes'])

    # Find edges to remove: any edge where u or v is in blocked nodes
    edges_to_remove = []
    for u, v in blocked_env_graph.edges():
        if u in all_blocked_nodes or v in all_blocked_nodes:
            edges_to_remove.append((u, v))

    blocked_env_graph.remove_edges_from(edges_to_remove)
    edges_to_remove_set = set(edges_to_remove)

    # Update visible_edges for all nodes
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            current_visible = blocked_env_graph.nodes[node]["visible_edges"]
            updated_visible = [e for e in current_visible if e not in edges_to_remove_set]
            blocked_env_graph.nodes[node]["visible_edges"] = updated_visible

    return blocked_env_graph


# ---------------------------------------------------------------------------
# 4. Agent Runners
# ---------------------------------------------------------------------------

def run_shortest_path_agent(env_graph, blocked_env_graph, hamiltonian_target_path,
                            source, target):
    """
    Runs the shortest-path replanning agent. Returns total travel distance.
    """
    shortest_path = []
    for i in range(len(hamiltonian_target_path) - 1):
        begin_node = hamiltonian_target_path[i]
        end_node = hamiltonian_target_path[i + 1]
        section_path = nx.shortest_path(env_graph, source=begin_node,
                                        target=end_node, weight="distance")
        if len(shortest_path) > 0 and shortest_path[-1] == section_path[0]:
            shortest_path.extend(section_path[1:])
        else:
            shortest_path.extend(section_path)

    path_1 = shortest_path.copy()
    target_nodes = hamiltonian_target_path.copy()
    env_graph1 = env_graph.copy()

    current_node = path_1[0]
    next_target_index = 1
    index = 0
    total_travel_distance = 0.0

    while index < len(path_1) - 1:
        next_node = path_1[index + 1]

        if current_node == target_nodes[next_target_index]:
            next_target_index += 1

        observable_edges = set(blocked_env_graph.nodes[current_node].get("visible_edges", []))
        assumed_observable_edges = set(env_graph1.nodes[current_node].get("visible_edges", []))
        blocked_edges = assumed_observable_edges - observable_edges

        if len(blocked_edges) > 0:
            for edge in blocked_edges:
                u, v = edge
                if env_graph1.has_edge(u, v):
                    env_graph1.remove_edge(u, v)
            for node in env_graph1.nodes():
                if "visible_edges" in env_graph1.nodes[node]:
                    current_visible = env_graph1.nodes[node]["visible_edges"]
                    updated_visible = set(current_visible) - blocked_edges
                    env_graph1.nodes[node]["visible_edges"] = list(updated_visible)

            path_edges = [(path_1[i], path_1[i + 1]) for i in range(index, len(path_1) - 1)]
            blocked_edges_both_directions = set()
            for u, v in blocked_edges:
                blocked_edges_both_directions.add((u, v))
                blocked_edges_both_directions.add((v, u))

            if any(edge in blocked_edges_both_directions for edge in path_edges):
                remaining_targets = target_nodes[next_target_index:]
                new_path = [current_node]
                for t in remaining_targets:
                    try:
                        segment = nx.shortest_path(env_graph1, source=new_path[-1],
                                                   target=t, weight="distance")
                        new_path.extend(segment[1:])
                    except nx.NetworkXNoPath:
                        return None
                path_1 = path_1[:index + 1] + new_path[1:]

        next_node = path_1[index + 1]
        if not env_graph.has_edge(current_node, next_node):
            return None
        total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
        current_node = next_node
        index += 1

    return total_travel_distance


def run_our_agent(env_graph, blocked_env_graph, target_graph, hamiltonian_target_path,
                  source, target, reward_ratio=10.0, sample_recursion=4,
                  sample_num_obstacle=3, sample_obstacle_hop=4):
    """
    Runs the RepeatedTopK agent. Returns total travel distance.
    """
    env_graph2 = env_graph.copy()
    path2_generator = RepeatedTopK(
        reward_ratio=reward_ratio,
        env_graph=env_graph2,
        target_graph=target_graph,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop
    )

    path_2 = path2_generator.find_best_path()
    target_nodes = hamiltonian_target_path.copy()

    current_node = path_2[0]
    next_target_index = 1
    index = 0
    total_travel_distance = 0.0

    while index < len(path_2) - 1:
        next_node = path_2[index + 1]

        if current_node == target_nodes[next_target_index]:
            next_target_index += 1

        observable_edges = set(blocked_env_graph.nodes[current_node].get("visible_edges", []))
        assumed_observable_edges = set(env_graph2.nodes[current_node].get("visible_edges", []))
        blocked_edges = assumed_observable_edges - observable_edges

        # Mark observed edges as seen
        for edge in assumed_observable_edges:
            if env_graph2.has_edge(*edge):
                env_graph2.edges[edge]["observed_edge"] = True

        if len(blocked_edges) > 0:
            for edge in blocked_edges:
                u, v = edge
                if env_graph2.has_edge(u, v):
                    env_graph2.remove_edge(u, v)
            for node in env_graph2.nodes():
                if "visible_edges" in env_graph2.nodes[node]:
                    current_visible = env_graph2.nodes[node]["visible_edges"]
                    updated_visible = set(current_visible) - blocked_edges
                    env_graph2.nodes[node]["visible_edges"] = list(updated_visible)

            path_edges = [(path_2[i], path_2[i + 1]) for i in range(index, len(path_2) - 1)]
            blocked_edges_both_directions = set()
            for u, v in blocked_edges:
                blocked_edges_both_directions.add((u, v))
                blocked_edges_both_directions.add((v, u))

            if any(edge in blocked_edges_both_directions for edge in path_edges):
                remaining_targets = target_nodes[next_target_index:]
                new_path = [current_node]
                for t in remaining_targets:
                    try:
                        segment = path2_generator.alternate_path_online(new_path[-1], t)
                        if segment is None:
                            return None
                        new_path.extend(segment[1:])
                    except Exception:
                        return None
                path_2 = path_2[:index + 1] + new_path[1:]

        next_node = path_2[index + 1]
        if not env_graph.has_edge(current_node, next_node):
            return None
        total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
        current_node = next_node
        index += 1

    return total_travel_distance


# ---------------------------------------------------------------------------
# 5. Main Benchmark
# ---------------------------------------------------------------------------

def run_real_map_benchmark(dem_path, n_size=64, num_runs=200, block_prob=0.5,
                           reward_ratio=1.0,
                           target_recursion=4, target_num_obstacles=3,
                           target_obstacle_hop=4,
                           sample_recursion=4, sample_num_obstacle=3,
                           sample_obstacle_hop=4,
                           output_csv="real_map_results.csv",
                           output_json="real_map_summary.json",
                           oval_seed=42,
                           num_sp_ovals=3, num_diverse_ovals=0):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(output_dir, output_csv)
    output_json_path = os.path.join(output_dir, output_json)

    # --- Step 1: Load DEM and build clean graph ---
    print("Loading DEM and building terrain graph...")
    height_grid = load_real_terrain(dem_path, n_size)
    terrain = RealTerrainGrid(height_grid, k_up=1.0, k_down=2.0)

    print("Computing visibilities (this may take a while)...")
    terrain.compute_all_visibilities()

    env_graph = terrain.get_graph().copy()
    source = terrain.source   # (n_size-1, n_size-1)
    target = terrain.target   # (0, 0)

    print(f"Graph: {env_graph.number_of_nodes()} nodes, {env_graph.number_of_edges()} edges")
    print(f"Source: {source}, Target: {target}")

    # --- Step 2: Build target graph ---
    print("Building target graph with diverse paths...")
    target_graph = create_fully_connected_target_graph(
        env_graph, recursions=target_recursion,
        num_obstacles=target_num_obstacles,
        obstacle_hop=target_obstacle_hop
    )

    # --- Step 3: Pre-compute Hamiltonian path ---
    path_generator = RepeatedTopK(
        reward_ratio=1.0, env_graph=env_graph.copy(), target_graph=target_graph,
        sample_recursion=sample_recursion, sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop
    )
    hamiltonian_target_path = path_generator.generate_Hamiltonian_path()
    print(f"Hamiltonian target path: {hamiltonian_target_path}")

    # --- Step 4: Place obstacle ovals (ONCE) ---
    print("\nPlacing obstacle ovals...")
    ovals = place_obstacle_ovals(env_graph, target_graph, source, target, seed=oval_seed,
                                  num_sp_ovals=num_sp_ovals, num_diverse_ovals=num_diverse_ovals)
    num_ovals = len(ovals)

    print(f"Placed {num_ovals} obstacle ovals:")
    for i, oval in enumerate(ovals):
        print(f"  Oval {i}: center=({oval['center'][0]},{oval['center'][1]}), "
              f"rx={oval['rx']}, ry={oval['ry']}, "
              f"nodes={len(oval['nodes'])}, type={oval['path_type']}")

    # --- Step 5: Main benchmark loop ---
    print(f"\nRunning benchmark: {num_runs} runs, block_prob={block_prob}, "
          f"reward_ratio={reward_ratio}")

    sp_costs = []
    our_costs = []
    sp_runtimes = []
    our_runtimes = []
    run_details = []  # For CSV

    for run_idx in tqdm(range(num_runs), desc="Benchmarking"):
        np.random.seed(42 + run_idx)
        random.seed(42 + run_idx)

        # Sample which ovals are blocked
        blocked_mask = [np.random.rand() < block_prob for _ in range(num_ovals)]
        num_blocked = sum(blocked_mask)

        # Create blocked environment
        blocked_env_graph = apply_oval_blockages(env_graph, ovals, blocked_mask)

        # Skip if goal unreachable
        if not nx.has_path(blocked_env_graph, source, target):
            continue

        # --- SP Agent ---
        t0 = time.time()
        sp_cost = run_shortest_path_agent(
            env_graph.copy(), blocked_env_graph, hamiltonian_target_path, source, target
        )
        sp_time = time.time() - t0

        # --- Our Agent ---
        t0 = time.time()
        our_cost = run_our_agent(
            env_graph.copy(), blocked_env_graph, target_graph, hamiltonian_target_path,
            source, target, reward_ratio=reward_ratio,
            sample_recursion=sample_recursion,
            sample_num_obstacle=sample_num_obstacle,
            sample_obstacle_hop=sample_obstacle_hop
        )
        our_time = time.time() - t0

        if sp_cost is not None and our_cost is not None:
            sp_costs.append(sp_cost)
            our_costs.append(our_cost)
            sp_runtimes.append(sp_time)
            our_runtimes.append(our_time)
            run_details.append({
                'run_idx': run_idx,
                'num_blocked_ovals': num_blocked,
                'sp_cost': sp_cost,
                'our_cost': our_cost,
                'sp_runtime': sp_time,
                'our_runtime': our_time,
            })

    # --- Step 6: Output ---
    valid_runs = len(sp_costs)

    if valid_runs == 0:
        print("\nNo valid runs completed. All blockage configurations may disconnect the graph.")
        return

    # Console output (matching benchmark.py format)
    sp_mean = np.mean(sp_costs)
    sp_var = np.var(sp_costs)
    sp_std = np.std(sp_costs)
    our_mean = np.mean(our_costs)
    our_var = np.var(our_costs)
    our_std = np.std(our_costs)

    improvement = ((sp_mean - our_mean) / sp_mean) * 100 if sp_mean > 0 else 0
    wins = sum(1 for s, o in zip(sp_costs, our_costs) if o < s)
    ties = sum(1 for s, o in zip(sp_costs, our_costs) if o == s)
    losses = sum(1 for s, o in zip(sp_costs, our_costs) if o > s)
    win_rate = (wins / valid_runs) * 100

    print("\n" + "=" * 50)
    print("REAL MAP BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Map: WV_DEM {n_size}x{n_size}")
    print(f"Ovals: {num_ovals}, Block prob: {block_prob}")
    print(f"Valid runs: {valid_runs}/{num_runs}")
    print("-" * 50)

    print(f"SP Agent:")
    print(f"  Mean Path Cost:  {sp_mean:.2f}")
    print(f"  Variance:        {sp_var:.2f}")
    print(f"  Std Deviation:   {sp_std:.2f}")
    print(f"  Avg Runtime:     {np.mean(sp_runtimes):.4f}s")

    print(f"Our Agent (RepeatedTopK):")
    print(f"  Mean Path Cost:  {our_mean:.2f}")
    print(f"  Variance:        {our_var:.2f}")
    print(f"  Std Deviation:   {our_std:.2f}")
    print(f"  Avg Runtime:     {np.mean(our_runtimes):.4f}s")

    print(f"\nImprovement: {improvement:.2f}%")
    print(f"Win/Tie/Loss: {wins}/{ties}/{losses}")
    print(f"Win rate: {win_rate:.1f}%")
    print("=" * 50)

    # CSV output
    csv_fields = ['run_idx', 'num_blocked_ovals', 'sp_cost', 'our_cost',
                  'sp_runtime', 'our_runtime']
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in run_details:
            writer.writerow({
                'run_idx': row['run_idx'],
                'num_blocked_ovals': row['num_blocked_ovals'],
                'sp_cost': f"{row['sp_cost']:.4f}",
                'our_cost': f"{row['our_cost']:.4f}",
                'sp_runtime': f"{row['sp_runtime']:.4f}",
                'our_runtime': f"{row['our_runtime']:.4f}",
            })
    print(f"\nPer-run results saved to: {output_csv_path}")

    # JSON summary
    summary = {
        'map': f'WV_DEM_{n_size}x{n_size}',
        'num_ovals': num_ovals,
        'block_prob': block_prob,
        'num_runs': num_runs,
        'valid_runs': valid_runs,
        'agent_params': {
            'reward_ratio': reward_ratio,
            'target_recursion': target_recursion,
            'target_num_obstacles': target_num_obstacles,
            'target_obstacle_hop': target_obstacle_hop,
            'sample_recursion': sample_recursion,
            'sample_num_obstacle': sample_num_obstacle,
            'sample_obstacle_hop': sample_obstacle_hop,
        },
        'ovals': [
            {'center': oval['center'], 'rx': oval['rx'], 'ry': oval['ry'],
             'num_nodes': len(oval['nodes']), 'path_type': oval['path_type']}
            for oval in ovals
        ],
        'sp_mean': float(sp_mean),
        'sp_var': float(sp_var),
        'sp_std': float(sp_std),
        'our_mean': float(our_mean),
        'our_var': float(our_var),
        'our_std': float(our_std),
        'improvement_pct': float(improvement),
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'win_rate_pct': float(win_rate),
        'sp_avg_runtime': float(np.mean(sp_runtimes)),
        'our_avg_runtime': float(np.mean(our_runtimes)),
    }

    with open(output_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_json_path}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RepeatedTopK vs SP on real terrain DEM map"
    )
    parser.add_argument("--dem-path", type=str, default=None,
                        help="Path to WV_DEM.tif (defaults to same directory)")
    parser.add_argument("--grid-size", type=int, default=64,
                        help="Grid resampling size (default: 64)")
    parser.add_argument("--num-runs", type=int, default=200,
                        help="Number of blockage realizations (default: 200)")
    parser.add_argument("--block-prob", type=float, default=0.5,
                        help="Per-oval blocking probability (default: 0.5)")
    parser.add_argument("--reward-ratio", type=float, default=1.0,
                        help="RepeatedTopK reward ratio (default: 1.0)")
    parser.add_argument("--output", type=str, default="real_map_results.csv",
                        help="Output CSV filename")
    parser.add_argument("--output-summary", type=str, default="real_map_summary.json",
                        help="Output summary JSON filename")
    parser.add_argument("--oval-seed", type=int, default=42,
                        help="Random seed for oval placement (default: 42)")
    parser.add_argument("--num-sp-ovals", type=int, default=3,
                        help="Number of ovals on shortest path (default: 3)")
    parser.add_argument("--num-diverse-ovals", type=int, default=0,
                        help="Number of ovals on diverse paths (default: 0)")
    args = parser.parse_args()

    if args.dem_path is None:
        args.dem_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "WV_DEM.tif"
        )

    if not os.path.exists(args.dem_path):
        print(f"Error: DEM file not found: {args.dem_path}")
        sys.exit(1)

    run_real_map_benchmark(
        dem_path=args.dem_path,
        n_size=args.grid_size,
        num_runs=args.num_runs,
        block_prob=args.block_prob,
        reward_ratio=args.reward_ratio,
        output_csv=args.output,
        output_json=args.output_summary,
        oval_seed=args.oval_seed,
        num_sp_ovals=args.num_sp_ovals,
        num_diverse_ovals=args.num_diverse_ovals,
    )


if __name__ == "__main__":
    main()
