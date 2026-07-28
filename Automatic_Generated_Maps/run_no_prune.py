"""
Run 50 maps with NO pruning, benchmark all of them, and save top 3 best/worst.

Outputs:
  - no_prune_results.csv: Full benchmark results
  - top_bottom_maps.pkl: Pickled list of (map_data, improvement_pct) for the
    3 best and 3 worst maps by improvement percentage.
"""

import sys
import os
import pickle
import csv
import time

import numpy as np
import networkx as nx
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from map_generator import generate_map_suite
from run_benchmark import benchmark_single_map


def main():
    print("=" * 60)
    print("Generating 50 maps with NO pruning")
    print("=" * 60)

    # Generate maps -- generate_map_suite already does structural validation
    # but we skip analytic scoring / pruning entirely
    maps = generate_map_suite(num_maps=50, seed_start=3000)
    print(f"Generated {len(maps)} structurally valid maps\n")

    # Benchmark all maps
    results = []  # (map_data, improvement_pct, result_dict)

    for i, map_data in enumerate(tqdm(maps, desc="Benchmarking")):
        label = map_data.get('label', '?')
        seed = map_data.get('seed', '?')

        result = benchmark_single_map(map_data, num_runs=200)

        if 'error' in result or result['valid_runs'] == 0:
            print(f"  Map {i} ({label}, seed={seed}): SKIPPED (error or no valid runs)")
            continue

        sp_mean = np.mean(result['sp_costs'])
        our_mean = np.mean(result['our_costs'])
        improvement_pct = ((sp_mean - our_mean) / sp_mean) * 100 if sp_mean > 0 else 0.0

        wins = sum(1 for s, o in zip(result['sp_costs'], result['our_costs']) if o < s)
        win_pct = (wins / result['valid_runs']) * 100

        results.append((map_data, improvement_pct, result))

        print(f"  Map {i:2d} ({label:20s}, seed={seed:5d}): "
              f"improvement={improvement_pct:+6.2f}%, "
              f"win_rate={win_pct:4.1f}%, "
              f"SP={sp_mean:.1f}, Ours={our_mean:.1f}")

    # Sort by improvement
    results.sort(key=lambda x: -x[1])  # Best first

    print("\n" + "=" * 60)
    print(f"Completed {len(results)} maps")
    print("=" * 60)

    if len(results) == 0:
        print("No valid results!")
        return

    improvements = [r[1] for r in results]
    print(f"Mean improvement: {np.mean(improvements):.2f}%")
    print(f"Median improvement: {np.median(improvements):.2f}%")
    print(f"Std improvement: {np.std(improvements):.2f}%")
    print(f"Maps with positive improvement: {sum(1 for x in improvements if x > 0)}/{len(results)}")

    # Top 3 best and worst
    top3 = results[:3]
    bottom3 = results[-3:]

    print("\nTop 3 BEST maps:")
    for i, (md, imp, _) in enumerate(top3):
        print(f"  {i+1}. seed={md.get('seed')}, label={md.get('label')}, "
              f"improvement={imp:+.2f}%")

    print("\nTop 3 WORST maps:")
    for i, (md, imp, _) in enumerate(bottom3):
        print(f"  {i+1}. seed={md.get('seed')}, label={md.get('label')}, "
              f"improvement={imp:+.2f}%")

    # Save the 6 maps for visualization
    save_data = {
        'top3': [(md, imp) for md, imp, _ in top3],
        'bottom3': [(md, imp) for md, imp, _ in bottom3],
        'all_improvements': improvements,
    }

    output_path = os.path.join(script_dir, 'top_bottom_maps.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved top/bottom maps to: {output_path}")

    # Also save full CSV
    csv_path = os.path.join(script_dir, 'no_prune_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'map_id', 'label', 'seed', 'grid_size', 'num_blobs',
            'num_chokepoints', 'valid_runs', 'sp_mean', 'our_mean',
            'improvement_pct', 'win_pct'
        ])
        writer.writeheader()
        for md, imp, res in results:
            sp_mean = np.mean(res['sp_costs'])
            our_mean = np.mean(res['our_costs'])
            wins = sum(1 for s, o in zip(res['sp_costs'], res['our_costs']) if o < s)
            writer.writerow({
                'map_id': md.get('map_id', -1),
                'label': md.get('label', '?'),
                'seed': md.get('seed', '?'),
                'grid_size': md.get('grid_size', '?'),
                'num_blobs': len(md.get('blobs', [])),
                'num_chokepoints': len(md.get('chokepoints', [])),
                'valid_runs': res['valid_runs'],
                'sp_mean': f"{sp_mean:.2f}",
                'our_mean': f"{our_mean:.2f}",
                'improvement_pct': f"{imp:.2f}",
                'win_pct': f"{(wins / res['valid_runs']) * 100:.1f}",
            })
    print(f"Saved full results to: {csv_path}")


if __name__ == "__main__":
    main()
