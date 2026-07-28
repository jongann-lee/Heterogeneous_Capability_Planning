"""
Benchmark runner for automatically generated maps.

Runs the RepeatedTopK agent and Shortest Path agent on each generated map
across multiple random blockage realizations, then saves results to CSV.
"""

import sys
import os
import time
import json
import csv
import argparse

import numpy as np
import networkx as nx
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.target_graph import create_fully_connected_target_graph
from Graph_Generation.edge_block_generation import block_specific_edges
from Single_Agent.repeated_topk import RepeatedTopK

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from map_generator import generate_map_suite
from analytic_prune import prune_maps, score_map


def run_shortest_path_agent(env_graph, blocked_env_graph, hamiltonian_target_path, source, target):
    """
    Runs the shortest-path replanning agent. Returns total travel distance.
    """
    shortest_path = []
    for i in range(len(hamiltonian_target_path) - 1):
        begin_node = hamiltonian_target_path[i]
        end_node = hamiltonian_target_path[i + 1]
        section_path = nx.shortest_path(env_graph, source=begin_node, target=end_node, weight="distance")
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

            path_edges = [(path_1[i], path_1[i+1]) for i in range(index, len(path_1) - 1)]
            blocked_edges_both_directions = set()
            for u, v in blocked_edges:
                blocked_edges_both_directions.add((u, v))
                blocked_edges_both_directions.add((v, u))

            if any(edge in blocked_edges_both_directions for edge in path_edges):
                remaining_targets = target_nodes[next_target_index:]
                new_path = [current_node]
                for t in remaining_targets:
                    try:
                        segment = nx.shortest_path(env_graph1, source=new_path[-1], target=t, weight="distance")
                        new_path.extend(segment[1:])
                    except nx.NetworkXNoPath:
                        return None  # Unreachable
                path_1 = path_1[:index + 1] + new_path[1:]

        next_node = path_1[index + 1]
        if not env_graph.has_edge(current_node, next_node):
            return None  # Edge doesn't exist
        total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
        current_node = next_node
        index += 1

    return total_travel_distance


def run_our_agent(env_graph, blocked_env_graph, target_graph, hamiltonian_target_path,
                  source, target, reward_ratio=3.0, sample_recursion=4,
                  sample_num_obstacle=4, sample_obstacle_hop=1):
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

        # Mark observed edges
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

            path_edges = [(path_2[i], path_2[i+1]) for i in range(index, len(path_2) - 1)]
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


def benchmark_single_map(map_data, num_runs=200, target_recursion=4,
                          target_num_obstacles=4, target_obstacle_hop=1):
    """
    Benchmarks both agents on a single generated map.

    Returns dict with per-run results for both agents.
    """
    env_graph = map_data['env_graph']
    chokepoints = map_data['chokepoints']
    source = map_data['source']
    target = map_data['target']
    block_prob = map_data['block_prob']

    # Create target graph
    # NOTE: create_fully_connected_target_graph modifies env_graph in-place
    # to set 'num_used' on edges. This is the same pattern as benchmark.py.
    try:
        target_graph = create_fully_connected_target_graph(
            env_graph,
            recursions=target_recursion,
            num_obstacles=target_num_obstacles,
            obstacle_hop=target_obstacle_hop
        )
    except Exception as e:
        return {'error': str(e), 'map_id': map_data.get('map_id', -1)}

    # Pre-compute Hamiltonian path
    try:
        path_generator = RepeatedTopK(
            reward_ratio=1.0, env_graph=env_graph.copy(), target_graph=target_graph,
            sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1
        )
        hamiltonian_target_path = path_generator.generate_Hamiltonian_path()
    except Exception as e:
        return {'error': str(e), 'map_id': map_data.get('map_id', -1)}

    sp_costs = []
    our_costs = []
    sp_runtimes = []
    our_runtimes = []
    valid_runs = 0

    for run_idx in range(num_runs):
        np.random.seed(42 + run_idx)
        import random
        random.seed(42 + run_idx)

        # Randomly block chokepoints
        edges_to_remove = []
        if len(chokepoints) > 0:
            RNG = np.random.rand(len(chokepoints))
            for i, edge in enumerate(chokepoints):
                if RNG[i] < block_prob:
                    u, v = edge
                    if env_graph.has_edge(u, v):
                        edges_to_remove.append((u, v))
                    elif env_graph.has_edge(v, u):
                        edges_to_remove.append((v, u))

        # Create blocked environment
        blocked_env_graph = block_specific_edges(env_graph, edges_to_remove)
        blocked_env_graph.remove_edges_from(edges_to_remove)

        # Skip if goal unreachable
        if not nx.has_path(blocked_env_graph, source, target):
            continue

        # Update visible_edges in blocked graph
        for node in blocked_env_graph.nodes():
            if "visible_edges" in blocked_env_graph.nodes[node]:
                current_visible = blocked_env_graph.nodes[node]["visible_edges"]
                edges_to_remove_set = set()
                for e in edges_to_remove:
                    edges_to_remove_set.add(tuple(sorted(e)))
                updated_visible = [e for e in current_visible
                                  if tuple(sorted(e)) not in edges_to_remove_set]
                blocked_env_graph.nodes[node]["visible_edges"] = updated_visible

        # --- Shortest Path Agent ---
        t0 = time.time()
        sp_cost = run_shortest_path_agent(
            env_graph.copy(), blocked_env_graph, hamiltonian_target_path, source, target
        )
        sp_time = time.time() - t0

        # --- Our Agent ---
        t0 = time.time()
        our_cost = run_our_agent(
            env_graph.copy(), blocked_env_graph, target_graph, hamiltonian_target_path,
            source, target
        )
        our_time = time.time() - t0

        if sp_cost is not None and our_cost is not None:
            sp_costs.append(sp_cost)
            our_costs.append(our_cost)
            sp_runtimes.append(sp_time)
            our_runtimes.append(our_time)
            valid_runs += 1

    return {
        'map_id': map_data.get('map_id', -1),
        'label': map_data.get('label', 'unknown'),
        'grid_size': map_data.get('grid_size', -1),
        'block_prob': block_prob,
        'num_blobs': len(map_data.get('blobs', [])),
        'num_chokepoints': len(chokepoints),
        'valid_runs': valid_runs,
        'sp_costs': sp_costs,
        'our_costs': our_costs,
        'sp_runtimes': sp_runtimes,
        'our_runtimes': our_runtimes,
    }


def screen_maps(candidates, screen_runs=50, target_count=None):
    """
    Phase 1: Analytic screening of candidate maps using structural scores.
    No simulation required -- uses graph properties to predict which maps
    will favor the visibility-aware agent.
    """
    print(f"  Scoring {len(candidates)} candidates analytically...")
    scored = prune_maps(candidates, target_count=target_count, verbose=False)
    screened = []
    for map_data, score, components in scored:
        label = map_data.get('label', '?')
        seed = map_data.get('seed', '?')
        print(f"  Map (seed={seed}, {label}): score={score:.2f} "
              f"[vis={components['cp_visibility']:.2f}, "
              f"detour={components['detour_near_blob']:.2f}, "
              f"entry={components['entry_efficiency']:.2f}, "
              f"impact={components['cp_impact']:.2f}, "
              f"diversity={components['path_diversity']:.2f}]")
        screened.append(map_data)
    return screened


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on auto-generated maps")
    parser.add_argument("--num-maps", type=int, default=50,
                        help="Number of maps to benchmark")
    parser.add_argument("--num-runs", type=int, default=200,
                        help="Number of blockage realizations per map")
    parser.add_argument("--seed-start", type=int, default=1000,
                        help="Starting seed for map generation")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV filename")
    parser.add_argument("--output-summary", type=str, default="benchmark_summary.json",
                        help="Output summary JSON filename")
    parser.add_argument("--screen-runs", type=int, default=50,
                        help="(Deprecated) Screening now uses analytic scores")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(output_dir, args.output)
    output_json = os.path.join(output_dir, args.output_summary)

    # Generate candidates in batches and screen until we have enough
    maps = []
    batch_size = args.num_maps * 3
    current_seed = args.seed_start
    batch_num = 0

    while len(maps) < args.num_maps:
        batch_num += 1
        print(f"\n--- Batch {batch_num}: Generating {batch_size} candidates (seed_start={current_seed}) ---")
        candidates = generate_map_suite(num_maps=batch_size, seed_start=current_seed)
        print(f"Generated {len(candidates)} structurally valid maps")

        remaining = args.num_maps - len(maps)
        print(f"Screening (need {remaining} more maps)...")
        # screened = screen_maps(candidates, target_count=remaining)
        screened = candidates
        maps.extend(screened)
        print(f"Batch {batch_num}: {len(screened)} passed, total: {len(maps)}/{args.num_maps}")

        current_seed += batch_size * 10  # Advance seed range

        if batch_num >= 5:
            print("Maximum batches reached, proceeding with available maps.")
            break

    print(f"\nPhase 2: Running full benchmark on {len(maps)} maps with {args.num_runs} runs each...")

    # CSV header
    csv_fields = [
        'map_id', 'label', 'grid_size', 'block_prob', 'num_blobs',
        'num_chokepoints', 'valid_runs',
        'sp_mean', 'sp_std', 'our_mean', 'our_std',
        'improvement_pct', 'our_wins_pct',
        'sp_avg_runtime', 'our_avg_runtime'
    ]

    all_results = []

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i, map_data in enumerate(tqdm(maps, desc="Benchmarking maps")):
            print(f"\n--- Map {i+1}/{len(maps)}: {map_data.get('label', '?')} "
                  f"(id={map_data.get('map_id', '?')}, "
                  f"grid={map_data.get('grid_size', '?')}x{map_data.get('grid_size', '?')}, "
                  f"blobs={len(map_data.get('blobs', []))}, "
                  f"chokepoints={len(map_data.get('chokepoints', []))}) ---")

            result = benchmark_single_map(map_data, num_runs=args.num_runs)

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
                continue

            if result['valid_runs'] == 0:
                print(f"  No valid runs (all blockage configs disconnected the graph)")
                continue

            sp_mean = np.mean(result['sp_costs'])
            sp_std = np.std(result['sp_costs'])
            our_mean = np.mean(result['our_costs'])
            our_std = np.std(result['our_costs'])
            improvement_pct = ((sp_mean - our_mean) / sp_mean) * 100 if sp_mean > 0 else 0

            # Count per-run wins
            wins = sum(1 for s, o in zip(result['sp_costs'], result['our_costs']) if o < s)
            ties = sum(1 for s, o in zip(result['sp_costs'], result['our_costs']) if o == s)
            our_wins_pct = (wins / result['valid_runs']) * 100

            row = {
                'map_id': result['map_id'],
                'label': result['label'],
                'grid_size': result['grid_size'],
                'block_prob': result['block_prob'],
                'num_blobs': result['num_blobs'],
                'num_chokepoints': result['num_chokepoints'],
                'valid_runs': result['valid_runs'],
                'sp_mean': f"{sp_mean:.2f}",
                'sp_std': f"{sp_std:.2f}",
                'our_mean': f"{our_mean:.2f}",
                'our_std': f"{our_std:.2f}",
                'improvement_pct': f"{improvement_pct:.2f}",
                'our_wins_pct': f"{our_wins_pct:.1f}",
                'sp_avg_runtime': f"{np.mean(result['sp_runtimes']):.4f}",
                'our_avg_runtime': f"{np.mean(result['our_runtimes']):.4f}",
            }
            writer.writerow(row)
            csvfile.flush()

            all_results.append(result)

            print(f"  Valid runs: {result['valid_runs']}/{args.num_runs}")
            print(f"  SP  mean: {sp_mean:.2f} +/- {sp_std:.2f}")
            print(f"  Our mean: {our_mean:.2f} +/- {our_std:.2f}")
            print(f"  Improvement: {improvement_pct:.2f}%, Win rate: {our_wins_pct:.1f}%")

    # Aggregate summary
    if all_results:
        all_sp = []
        all_ours = []
        for r in all_results:
            all_sp.extend(r['sp_costs'])
            all_ours.extend(r['our_costs'])

        total_wins = sum(1 for s, o in zip(all_sp, all_ours) if o < s)
        total_ties = sum(1 for s, o in zip(all_sp, all_ours) if o == s)
        total_losses = sum(1 for s, o in zip(all_sp, all_ours) if o > s)
        total_runs = len(all_sp)

        # Per-map improvement stats
        map_improvements = []
        for r in all_results:
            sp_m = np.mean(r['sp_costs'])
            our_m = np.mean(r['our_costs'])
            if sp_m > 0:
                map_improvements.append(((sp_m - our_m) / sp_m) * 100)

        summary = {
            'total_maps': len(all_results),
            'total_runs': total_runs,
            'overall_sp_mean': float(np.mean(all_sp)),
            'overall_our_mean': float(np.mean(all_ours)),
            'overall_improvement_pct': float(((np.mean(all_sp) - np.mean(all_ours)) / np.mean(all_sp)) * 100),
            'total_wins': total_wins,
            'total_ties': total_ties,
            'total_losses': total_losses,
            'win_rate_pct': float(total_wins / total_runs * 100) if total_runs > 0 else 0,
            'maps_with_positive_improvement': sum(1 for x in map_improvements if x > 0),
            'mean_per_map_improvement_pct': float(np.mean(map_improvements)) if map_improvements else 0,
            'median_per_map_improvement_pct': float(np.median(map_improvements)) if map_improvements else 0,
            'std_per_map_improvement_pct': float(np.std(map_improvements)) if map_improvements else 0,
        }

        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Total maps benchmarked: {summary['total_maps']}")
        print(f"Total valid runs:       {summary['total_runs']}")
        print(f"Overall SP mean cost:   {summary['overall_sp_mean']:.2f}")
        print(f"Overall Our mean cost:  {summary['overall_our_mean']:.2f}")
        print(f"Overall improvement:    {summary['overall_improvement_pct']:.2f}%")
        print(f"Win/Tie/Loss:           {total_wins}/{total_ties}/{total_losses}")
        print(f"Win rate:               {summary['win_rate_pct']:.1f}%")
        print(f"Maps w/ improvement:    {summary['maps_with_positive_improvement']}/{summary['total_maps']}")
        print(f"Mean per-map improvement: {summary['mean_per_map_improvement_pct']:.2f}%")
        print(f"Median per-map improvement: {summary['median_per_map_improvement_pct']:.2f}%")
        print("=" * 60)
        print(f"\nResults saved to: {output_csv}")
        print(f"Summary saved to: {output_json}")
    else:
        print("\nNo successful results to summarize.")


if __name__ == "__main__":
    main()
