"""
Real-DEM entry point for the generalized capability model.

Builds the planner graph (clean) and the ground-truth graph (obstacles
applied) from the WV DEM, assigns each target a random type in ``1..n``,
generates random agent capability subsets, and runs
``simulation.engine.run_simulation`` with a pluggable policy
(defaults to the naive type-aware placeholder -- swap in your baseline).

This mirrors the graph-building half of multi_agent_simulation.py but stays
self-contained so it doesn't disturb the base benchmark. The simulation runs in
continuous time (discrete-event): edge cost == traversal time, and the reported
objective is the makespan plus a death penalty. ``--render`` writes interpolated
per-turn PNGs + an MP4; otherwise it prints and writes a JSON/CSV summary.

    uv run python -m simulation.real_map_benchmark --help
    uv run python -m simulation.real_map_benchmark --policy baseline1 --seed 0
    uv run python -m simulation.real_map_benchmark --policy baseline2 --seed 1
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from simulation.agent import Agent
from simulation.domain import (
    assign_agent_capabilities, assign_target_types, capability_label,
    init_target_types, target_type_name, validate_capabilities,
)
from simulation.engine import run_simulation, naive_type_aware_replan


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


def build_graphs(dem_path, road_pkl=None, n_size=64, source=(0, 0),
                 targets=DEFAULT_TARGETS, obstacle_specs=None,
                 target_num_neighbors=3, target_recursion=2,
                 target_num_obstacles=3, target_obstacle_hop=4,
                 num_target_types=3, rng=None):
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

    target_types = assign_target_types(
        targets, num_target_types=num_target_types, rng=rng)
    init_target_types(env_graph, ground_truth, target_types)
    return env_graph, ground_truth, target_types


def run(dem_path, road_pkl=None, n_size=64, source=(0, 0), targets=DEFAULT_TARGETS,
        num_target_types=3, num_agents=4, agent_capabilities=None,
        capability_probability=0.5, scout_probability=0.25,
        ensure_target_coverage=True, ensure_scout=True,
        obstacle_specs=None, policy=None,
        reward_ratio=1.0, obs_discount_factor=0.9,
        sample_recursion=2, sample_num_obstacle=3, sample_obstacle_hop=4,
        seed=0, output_json=None, output_csv=None, verbose=False,
        render=False, render_dir=None, output_mp4=None, mp4_fps=4, render_dt=1.0):
    random.seed(seed)
    np.random.seed(seed)

    rng = random.Random(seed)
    built = build_graphs(
        dem_path, road_pkl, n_size, source, targets, obstacle_specs,
        num_target_types=num_target_types, rng=rng)
    if built is None:
        return None
    env_graph, ground_truth, target_types = built

    print("=" * 56)
    print("CAPABILITY-BASED MULTI-AGENT PLANNING  |  real DEM")
    print("=" * 56)
    if agent_capabilities is None:
        agent_capabilities = assign_agent_capabilities(
            num_agents=num_agents,
            num_target_types=num_target_types,
            capability_probability=capability_probability,
            scout_probability=scout_probability,
            ensure_target_coverage=ensure_target_coverage,
            ensure_scout=ensure_scout,
            rng=rng,
        )
    else:
        agent_capabilities = [
            validate_capabilities(values, num_target_types)
            for values in agent_capabilities
        ]
        if not agent_capabilities:
            raise ValueError("at least one agent capability set is required")

    print(f"seed={seed}  target_types=1..{num_target_types}  "
          f"targets={len(list(targets))}  agents={len(agent_capabilities)}")
    print("agent capabilities: "
          + ", ".join(f"{i}:{capability_label(values)}"
                      for i, values in enumerate(agent_capabilities)))
    print("target types (ground truth): "
          + ", ".join(f"{t}:{target_type_name(tt)}"
                      for t, tt in target_types.items()))

    agents = [Agent(source, capabilities=values)
              for values in agent_capabilities]

    if render:
        from simulation.rendering import clear_frame_dir
        clear_frame_dir(render_dir)
        print(f"rendering frames -> {render_dir}")

    t0 = time.perf_counter()
    result = run_simulation(
        env_graph, ground_truth, agents, policy=policy or naive_type_aware_replan,
        reward_ratio=reward_ratio, obs_discount_factor=obs_discount_factor,
        sample_recursion=sample_recursion, sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop, verbose=verbose,
        render_dir=render_dir if render else None, render_dt=render_dt,
    )
    runtime = time.perf_counter() - t0

    if render and output_mp4:
        from simulation.rendering import make_mp4_from_frames
        try:
            make_mp4_from_frames(render_dir, output_mp4, fps=mp4_fps)
            print(f"mp4 -> {output_mp4}")
        except FileNotFoundError:
            print("ffmpeg not found - frames saved but no MP4 (brew install ffmpeg).")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e}")
            if e.stderr:
                print(e.stderr.strip())

    print("-" * 56)
    print(f"completed:   {result['completed']}")
    print(f"makespan:    {result['makespan']:.2f}   objective: {result['objective']:.2f}")
    print(f"eliminated:  {len(result['eliminated_targets'])}/{len(list(targets))} targets")
    if result["remaining_targets"]:
        print(f"remaining:   {result['remaining_targets']}")
    dead = [f"{i}:{capability_label(agents[i].capabilities)}"
            for i in result["deaths"]]
    print(f"deaths:      {dead if dead else 'none'}")
    print("per-agent traversal time:")
    for i, (agent, traversal_time) in enumerate(
            zip(agents, result["per_agent_cost"])):
        types = ",".join(str(value) for value in sorted(agent.capabilities))
        print(f"  agent {i} ({types}): {traversal_time:.2f}")
    print(f"runtime: {runtime:.3f}s")
    print("=" * 56)

    if output_json:
        summary = {
            "seed": seed,
            "num_target_types": num_target_types,
            "agent_capabilities": [sorted(a.capabilities) for a in agents],
            "target_types": {str(t): tt for t, tt in target_types.items()},
            "completed": result["completed"],
            "makespan": result["makespan"],
            "objective": result["objective"],
            "num_deaths": result["num_deaths"],
            "eliminated": len(result["eliminated_targets"]),
            "remaining_targets": [str(t) for t in result["remaining_targets"]],
            "deaths": result["deaths"],
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
            w.writerow(["agent_idx", "capabilities", "scout_capable",
                        "alive", "cost", "trajectory_len"])
            for i, a in enumerate(agents):
                w.writerow([i, ",".join(map(str, sorted(a.capabilities))),
                            a.scout_capable, a.alive,
                            f"{a.total_traversal_cost:.4f}", len(a.trajectory)])
        print(f"per-agent -> {output_csv}")

    return result


def _parse_agent_capabilities(value, num_target_types):
    """Parse ``0,1,3;2,4;0,2`` into one capability set per agent."""
    groups = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        values = set() if not raw_group else {
            int(item.strip()) for item in raw_group.split(",") if item.strip()
        }
        groups.append(validate_capabilities(values, num_target_types))
    if not groups:
        raise ValueError("at least one agent capability group is required")
    return groups


def main():
    p = argparse.ArgumentParser(
        description="Capability-based multi-agent planning on the real DEM map")
    real_maps = os.path.join(PROJECT_ROOT, "Real_Life_Maps")
    output_root = os.path.join(PROJECT_ROOT, "outputs", "my_policy_simulation")
    p.add_argument("--dem-path", default=os.path.join(real_maps, "WV_DEM.tif"))
    p.add_argument("--road-pkl", default=os.path.join(real_maps, "WV_roads.pkl"))
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-target-types", type=int, default=3,
                   help="target types are the integers 1..N")
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--capability-probability", type=float, default=0.5,
                   help="independent probability of each positive capability")
    p.add_argument("--scout-probability", type=float, default=0.25,
                   help="independent probability that an agent receives capability 0")
    p.add_argument("--agent-capabilities",
                   help="explicit semicolon-separated sets, e.g. '0,1,3;2;1,2'")
    p.add_argument("--allow-uncovered-types", action="store_true",
                   help="do not force every target type onto at least one agent")
    p.add_argument("--allow-no-scout", action="store_true",
                   help="do not force at least one agent to receive capability 0")
    p.add_argument("--policy",
                   choices=["baseline1", "baseline2", "placeholder"],
                   default="baseline1",
                   help="baseline1, WRP-based baseline2, or the safe placeholder")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--render", action="store_true",
                   help="write one PNG per turn and compile an MP4")
    p.add_argument("--render-dir",
                   default=os.path.join(output_root, "frames"))
    p.add_argument("--output-mp4",
                   default=os.path.join(output_root, "render_result.mp4"))
    p.add_argument("--mp4-fps", type=int, default=4)
    p.add_argument("--render-dt", type=float, default=1.0,
                   help="sim-time between rendered frames (continuous time)")
    p.add_argument("--output-json",
                   default=os.path.join(output_root, "summary.json"))
    p.add_argument("--output-csv",
                   default=os.path.join(output_root, "trajectory_per_agent.csv"))
    args = p.parse_args()

    if not os.path.exists(args.dem_path):
        print(f"DEM not found: {args.dem_path}")
        sys.exit(1)
    road_pkl = args.road_pkl if os.path.exists(args.road_pkl) else None
    try:
        explicit_capabilities = (
            _parse_agent_capabilities(
                args.agent_capabilities, args.num_target_types)
            if args.agent_capabilities is not None else None
        )
    except ValueError as exc:
        p.error(str(exc))

    if args.policy == "baseline1":
        from planning.policies.baseline1 import replan as policy
    elif args.policy == "baseline2":
        from planning.policies.baseline2 import replan as policy
    else:
        policy = naive_type_aware_replan

    run(args.dem_path, road_pkl=road_pkl, n_size=args.grid_size,
        num_target_types=args.num_target_types, num_agents=args.num_agents,
        agent_capabilities=explicit_capabilities,
        capability_probability=args.capability_probability,
        scout_probability=args.scout_probability,
        ensure_target_coverage=not args.allow_uncovered_types,
        ensure_scout=not args.allow_no_scout,
        policy=policy, seed=args.seed, verbose=args.verbose,
        output_json=args.output_json, output_csv=args.output_csv,
        render=args.render, render_dir=args.render_dir,
        output_mp4=args.output_mp4, mp4_fps=args.mp4_fps, render_dt=args.render_dt)


if __name__ == "__main__":
    main()


# Compatibility for notebooks written before the package cleanup.
build_rps_graphs = build_graphs
