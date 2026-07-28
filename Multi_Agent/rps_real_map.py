"""
Real-DEM entry point for the Rock-Scissor-Paper model.

Builds the planner graph (clean) and the ground-truth graph (obstacles
applied) from the WV DEM, assigns each target a random RPS type, instantiates
the default roster (one scout + one of each combat type), and runs
Multi_Agent.rps_simulation.run_rps_simulation with a pluggable policy
(defaults to the naive type-aware placeholder -- swap in your baseline).

This mirrors the graph-building half of multi_agent_simulation.py but stays
self-contained so it doesn't disturb the base benchmark. The simulation runs in
continuous time (discrete-event): edge cost == traversal time, and the reported
objective is the makespan plus a death penalty. ``--render`` writes interpolated
per-turn PNGs + an MP4; otherwise it prints and writes a JSON/CSV summary.

    uv run python Multi_Agent/rps_real_map.py --help
    uv run python Multi_Agent/rps_real_map.py --policy baseline1 --seed 0
    uv run python Multi_Agent/rps_real_map.py --policy baseline1 --seed 1 --render
"""

import sys
import os
import copy
import json
import csv
import time
import pickle
import argparse
import subprocess
import random

import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Multi_Agent.finite_horizon_MA import Agent
from Multi_Agent.rps import (
    AGENT_TYPES, TYPE_NAMES, assign_target_types, init_target_types,
)
from Multi_Agent.rps_simulation import run_rps_simulation, naive_type_aware_replan


# Same default obstacles as multi_agent_simulation.DEFAULT_OBSTACLE_SPECS.
# DEFAULT_OBSTACLE_SPECS = [
#     ((47, 43), 5, 3),
#     ((30, 36), 3, 5),
#     ((31, 14), 4, 4),
#     ((36, 55), 5, 3),
#     ((14, 17), 4, 4),
# ]
DEFAULT_OBSTACLE_SPECS = []

DEFAULT_TARGETS = ((14, 54), (1, 29), (33, 17), (34, 35), (63, 37), (37, 5), (49, 58))


# ---------------------------------------------------------------------------
# DEM / road loading (small copies so we don't import the matplotlib-heavy
# multi_agent_simulation driver).
# ---------------------------------------------------------------------------

def _load_real_terrain(dem_path, n_size):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(dem_path) as dataset:
        data = dataset.read(1, out_shape=(n_size, n_size),
                            resampling=Resampling.bilinear)
        if dataset.nodata is not None:
            data = np.where(data == dataset.nodata, np.nan, data)
    return np.rot90(data, k=-1)


def _load_roads(road_pkl):
    if road_pkl is None or not os.path.exists(road_pkl):
        return set(), set()
    with open(road_pkl, "rb") as f:
        data = pickle.load(f)
    return data["road_nodes"], data["road_edges"]


def build_rps_graphs(dem_path, road_pkl=None, n_size=64, source=(0, 0),
                     targets=DEFAULT_TARGETS, obstacle_specs=None,
                     target_num_neighbors=3, target_recursion=2,
                     target_num_obstacles=3, target_obstacle_hop=4,
                     rng=None):
    """Build (env_graph, ground_truth, target_types).

    env_graph is the planner's clean view with targets' rps_type = UNKNOWN;
    ground_truth has obstacles applied and the true rps_type on every target.
    Returns None if any target is unreachable after blocking.
    """
    height_grid = _load_real_terrain(dem_path, n_size)
    road_nodes, road_edges = _load_roads(road_pkl)
    targets = list(targets)

    terrain = RealTerrainGrid(height_grid, source=source, targets=targets,
                              k_up=1.0, k_down=2.0,
                              road_nodes=road_nodes, road_edges=road_edges)
    terrain.compute_all_visibilities()
    env_graph = terrain.get_graph().copy()

    # Populate edge `num_used` (used by reward-driven policies) as a side
    # effect of target-graph construction.
    create_fully_connected_target_graph(
        env_graph, source=source, targets=targets,
        num_neighbors=target_num_neighbors, recursions=target_recursion,
        num_obstacles=target_num_obstacles, obstacle_hop=target_obstacle_hop,
    )

    # Ground truth: apply obstacles, strip obstacle-incident edges + visibility.
    if obstacle_specs is None:
        obstacle_specs = DEFAULT_OBSTACLE_SPECS
    blocked = copy.deepcopy(terrain)
    for center, rx, ry in obstacle_specs:
        blocked.add_obstacle(center=center, rx=rx, ry=ry)
    ground_truth = blocked.get_graph().copy()
    obs_edges = [(u, v) for u, v in ground_truth.edges()
                 if ground_truth.nodes[u].get("type") == "obstacle"
                 or ground_truth.nodes[v].get("type") == "obstacle"]
    ground_truth.remove_edges_from(obs_edges)
    obs_set = set(obs_edges)
    for node in ground_truth.nodes():
        if "visible_edges" in ground_truth.nodes[node]:
            ground_truth.nodes[node]["visible_edges"] = [
                e for e in ground_truth.nodes[node]["visible_edges"] if e not in obs_set
            ]

    for t in targets:
        if not nx.has_path(ground_truth, source, t):
            print(f"Target {t} unreachable from {source} after blocking. Aborting.")
            return None

    target_types = assign_target_types(targets, rng=rng)
    init_target_types(env_graph, ground_truth, target_types)
    return env_graph, ground_truth, target_types


def run(dem_path, road_pkl=None, n_size=64, source=(0, 0), targets=DEFAULT_TARGETS,
        agent_types=AGENT_TYPES, obstacle_specs=None, policy=None,
        reward_ratio=1.0, obs_discount_factor=0.9,
        sample_recursion=2, sample_num_obstacle=3, sample_obstacle_hop=4,
        seed=0, output_json=None, output_csv=None, verbose=False,
        render=False, render_dir=None, output_mp4=None, mp4_fps=4, render_dt=1.0):
    random.seed(seed)
    np.random.seed(seed)

    built = build_rps_graphs(dem_path, road_pkl, n_size, source, targets,
                             obstacle_specs)
    if built is None:
        return None
    env_graph, ground_truth, target_types = built

    print("=" * 56)
    print("ROCK-SCISSOR-PAPER  |  real DEM")
    print("=" * 56)
    print(f"seed={seed}  targets={len(list(targets))}  agents="
          f"{[TYPE_NAMES[t] for t in agent_types]}")
    print("target types (ground truth): "
          + ", ".join(f"{t}:{TYPE_NAMES[tt]}" for t, tt in target_types.items()))

    agents = [Agent(source, agent_type=t) for t in agent_types]

    if render:
        from Multi_Agent.simulation_utils import clear_frame_dir
        clear_frame_dir(render_dir)
        print(f"rendering frames -> {render_dir}")

    t0 = time.perf_counter()
    result = run_rps_simulation(
        env_graph, ground_truth, agents, policy=policy or naive_type_aware_replan,
        reward_ratio=reward_ratio, obs_discount_factor=obs_discount_factor,
        sample_recursion=sample_recursion, sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop, verbose=verbose,
        render_dir=render_dir if render else None, render_dt=render_dt,
    )
    runtime = time.perf_counter() - t0

    if render and output_mp4:
        from Multi_Agent.simulation_utils import make_mp4_from_frames
        try:
            make_mp4_from_frames(render_dir, output_mp4, fps=mp4_fps)
            print(f"mp4 -> {output_mp4}")
        except FileNotFoundError:
            print("ffmpeg not found - frames saved but no MP4 (brew install ffmpeg).")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e}")

    print("-" * 56)
    print(f"completed:   {result['completed']}")
    print(f"makespan:    {result['makespan']:.2f}   objective: {result['objective']:.2f}")
    print(f"eliminated:  {len(result['eliminated_targets'])}/{len(list(targets))} targets")
    if result["remaining_targets"]:
        print(f"remaining:   {result['remaining_targets']}")
    dead = [f"{i}:{TYPE_NAMES[agents[i].agent_type]}" for i in result["deaths"]]
    print(f"deaths:      {dead if dead else 'none'}")
    print(f"per-agent cost: {[round(c, 2) for c in result['per_agent_cost']]}")
    print(f"runtime: {runtime:.3f}s")
    print("=" * 56)

    if output_json:
        summary = {
            "seed": seed,
            "agent_types": [TYPE_NAMES[t] for t in agent_types],
            "target_types": {str(t): TYPE_NAMES[tt] for t, tt in target_types.items()},
            "completed": result["completed"],
            "makespan": result["makespan"],
            "objective": result["objective"],
            "num_deaths": result["num_deaths"],
            "eliminated": len(result["eliminated_targets"]),
            "remaining_targets": [str(t) for t in result["remaining_targets"]],
            "deaths": [TYPE_NAMES[agents[i].agent_type] for i in result["deaths"]],
            "per_agent_cost": [float(c) for c in result["per_agent_cost"]],
            "runtime": runtime,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary -> {output_json}")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent_idx", "type", "alive", "cost", "trajectory_len"])
            for i, a in enumerate(agents):
                w.writerow([i, TYPE_NAMES[a.agent_type], a.alive,
                            f"{a.total_traversal_cost:.4f}", len(a.trajectory)])
        print(f"per-agent -> {output_csv}")

    return result


def main():
    p = argparse.ArgumentParser(description="Rock-Scissor-Paper on the real DEM map")
    here = os.path.dirname(os.path.abspath(__file__))
    real_maps = os.path.join(project_root, "Real_Life_Maps")
    p.add_argument("--dem-path", default=os.path.join(real_maps, "WV_DEM.tif"))
    p.add_argument("--road-pkl", default=os.path.join(real_maps, "WV_roads.pkl"))
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agents", default="0,1,2,3",
                   help="comma-separated agent type ids (0=scout,1=rock,2=scissor,3=paper)")
    p.add_argument("--policy",
                   choices=["baseline1", "scout_gtsp", "scout_wrp", "placeholder"],
                   default="baseline1",
                   help="assignment policy (default: baseline1)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--render", action="store_true",
                   help="write one PNG per turn and compile an MP4")
    p.add_argument("--render-dir",
                   default=os.path.join(here, "my_policy_simulation", "rps_frames"))
    p.add_argument("--output-mp4",
                   default=os.path.join(here, "my_policy_simulation", "rps_simulation.mp4"))
    p.add_argument("--mp4-fps", type=int, default=4)
    p.add_argument("--render-dt", type=float, default=1.0,
                   help="sim-time between rendered frames (continuous time)")
    p.add_argument("--output-json",
                   default=os.path.join(here, "my_policy_simulation", "rps_summary.json"))
    p.add_argument("--output-csv",
                   default=os.path.join(here, "my_policy_simulation", "rps_per_agent.csv"))
    args = p.parse_args()

    if not os.path.exists(args.dem_path):
        print(f"DEM not found: {args.dem_path}")
        sys.exit(1)
    road_pkl = args.road_pkl if os.path.exists(args.road_pkl) else None
    agent_types = [int(x) for x in args.agents.split(",") if x.strip() != ""]
    bad = [t for t in agent_types if t not in AGENT_TYPES]
    if bad:
        print(f"invalid agent type ids {bad}; valid: {list(AGENT_TYPES)}")
        sys.exit(1)

    if args.policy == "baseline1":
        from Multi_Agent.baseline1_all_key import replan as policy
    elif args.policy == "scout_gtsp":
        from Multi_Agent.scout_gtsp import replan as policy
    elif args.policy == "scout_wrp":
        from Multi_Agent.scout_wrp import replan as policy
    else:
        policy = naive_type_aware_replan

    run(args.dem_path, road_pkl=road_pkl, n_size=args.grid_size,
        agent_types=agent_types, policy=policy, seed=args.seed, verbose=args.verbose,
        output_json=args.output_json, output_csv=args.output_csv,
        render=args.render, render_dir=args.render_dir,
        output_mp4=args.output_mp4, mp4_fps=args.mp4_fps, render_dt=args.render_dt)


if __name__ == "__main__":
    main()
