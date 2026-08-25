"""Deterministic checkpoint evaluation on the 64x64 WV terrain."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from learning.policy.configuration import DEFAULT_CONFIG_PATH, load_config
from learning.gpu_sim.instances import make_wv_dem_instance
from learning.policy.model import build_policy
from learning.policy.oracle import parallel_tsp


# Fixed sanity-check setup. Set a value to None to sample that component from
# each episode seed. These intentionally mirror the editable globals in train.py.
SOURCE_POSITION = (0, 0)
TARGET_POSITIONS = [(14,54), (1,29), (33,17), (34,35), (63,37), (37,5), (49,58)]
TARGET_TYPES = [1, 2, 2, 1, 2, 3, 3]
AGENT_CAPABILITIES = [{0}, {1}, {2}, {3}]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "outputs" / "my_policy_simulation"
DEFAULT_RENDER_DIR = DEFAULT_RENDER_ROOT / "frames"
DEFAULT_OUTPUT_MP4 = DEFAULT_RENDER_ROOT / "render_result.mp4"


def _resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        weights = path / "trained_weights.pt"
        config = path / "config.yaml"
    else:
        weights = path
        config = path.with_name("config.yaml")
    if not weights.is_file():
        raise FileNotFoundError(f"trained weights not found: {weights}")
    return weights, config if config.is_file() else None


def _episode_factory(seed, config):
    return make_wv_dem_instance(
        seed, config.model.num_target_types, config.training.num_agents,
        source_position=SOURCE_POSITION,
        target_positions=TARGET_POSITIONS,
        target_types=TARGET_TYPES,
        agent_capabilities=AGENT_CAPABILITIES,
        min_targets=config.instances.min_targets,
        max_targets=config.instances.max_targets)


def _encode_episode(world, truth, agents, num_target_types, device):
    source = world.node_index[agents[0].position]
    capabilities = torch.zeros(
        (1, len(agents), num_target_types + 1),
        dtype=torch.bool, device=device)
    for agent_index, agent in enumerate(agents):
        for capability in agent.capabilities:
            capabilities[0, agent_index, capability] = True
    target_types = torch.tensor([[
        truth.nodes[target]["rps_type"] for target in world.targets
    ]], dtype=torch.long, device=device)
    return source, capabilities, target_types


def _capture_state(trace):
    """Return a callback that stores only one CPU snapshot per event time."""
    def capture(state):
        names = ("positions", "alive", "target_live", "target_known", "moving",
                 "transit_from", "transit_to", "arrival_time", "clock",
                 "route_next")
        snapshot = {name: getattr(state, name)[0].detach().cpu().clone()
                    for name in names}
        if trace and float(trace[-1]["clock"]) == float(snapshot["clock"]):
            trace[-1] = snapshot
        else:
            trace.append(snapshot)
    return capture


def _gpu_episode(model, config, env, truth, agents, device, terrain=None,
                 trace=None):
    from learning.gpu_sim.observation_gpu import TensorObservationBuilder
    from learning.gpu_sim.rollout_gpu import collect_tensor_episodes
    from learning.gpu_sim.state import TensorEpisodeState
    from learning.gpu_sim.world import TensorWorld

    world = TensorWorld.from_networkx(
        env, config.candidates, device=device, terrain=terrain)
    source, capabilities, target_types = _encode_episode(
        world, truth, agents, config.model.num_target_types, device)
    state = TensorEpisodeState.create(
        world, [source], capabilities, target_types)
    oracle_makespan = parallel_tsp(truth, agents)
    rollout = collect_tensor_episodes(
        model, state,
        TensorObservationBuilder(
            world, config.model.num_target_types,
            task_graph=config.model.architecture == "task_graph"),
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty,
        training=False,
        state_callback=None if trace is None else _capture_state(trace),
        oracle_makespans=oracle_makespan)
    record = {
        "return": float(rollout.returns[0]),
        "makespan": float(rollout.makespans[0]),
        "oracle_makespan": float(rollout.oracle_makespans[0]),
        "normalized_regret": float(rollout.normalized_regrets[0]),
        "completed": bool(rollout.completed[0]),
        "deaths": int(rollout.deaths[0]),
        "remaining_targets": int(rollout.remaining_targets[0]),
        "stalled": bool(rollout.stalled[0]),
        "all_agents_dead": bool(rollout.all_agents_dead[0]),
    }
    return record, world.terrain, world


def _route_from_snapshot(snapshot, agent_index, node, node_count):
    route = [node]
    successors = snapshot["route_next"][agent_index]
    seen = {node}
    while len(route) <= node_count:
        next_node = int(successors[node])
        if next_node < 0 or next_node in seen:
            break
        route.append(next_node)
        seen.add(next_node)
        node = next_node
    return route


def _render_gpu_trace(trace, env, truth, source_agents, world, frames_dir,
                      render_dt):
    """Render a device rollout after it completes; no CPU policy replay."""
    from simulation.agent import Agent
    from simulation.domain import UNKNOWN_TYPE
    from simulation.rendering import render_frame

    if not trace:
        return 0
    display_agents = [Agent(source_agents[0].position,
                            capabilities=agent.capabilities)
                      for agent in source_agents]
    trajectories = [[int(trace[0]["positions"][i])]
                    for i in range(len(display_agents))]
    snapshot_trajectories = []
    for snapshot in trace:
        for i in range(len(display_agents)):
            node = int(snapshot["positions"][i])
            if trajectories[i][-1] != node:
                trajectories[i].append(node)
        snapshot_trajectories.append([path.copy() for path in trajectories])

    final_time = float(trace[-1]["clock"])
    frame_times = []
    time_value = 0.0
    while time_value < final_time:
        frame_times.append(time_value)
        time_value += render_dt
    frame_times.append(final_time)
    snapshot_index = 0
    targets = list(world.targets)
    for frame_index, tau in enumerate(frame_times):
        while (snapshot_index + 1 < len(trace)
               and float(trace[snapshot_index + 1]["clock"]) <= tau):
            snapshot_index += 1
        snapshot = trace[snapshot_index]
        visible_env = env.copy()
        for target_index, target in enumerate(targets):
            live = bool(snapshot["target_live"][target_index])
            known = bool(snapshot["target_known"][target_index])
            visible_env.nodes[target]["type"] = (
                "target_unreached" if live else "target_reached")
            visible_env.nodes[target]["rps_type"] = (
                int(truth.nodes[target]["rps_type"]) if known else UNKNOWN_TYPE)
        xys = []
        positions = []
        for i, agent in enumerate(display_agents):
            node_index = int(snapshot["positions"][i])
            positions.append(node_index)
            agent.position = world.nodes[node_index]
            agent.alive = bool(snapshot["alive"][i])
            agent.trajectory = [world.nodes[node]
                                for node in snapshot_trajectories[snapshot_index][i]]
            route_start = node_index
            if bool(snapshot["moving"][i]):
                start = int(snapshot["transit_from"][i])
                end = int(snapshot["transit_to"][i])
                arrival = float(snapshot["arrival_time"][i])
                clock = float(snapshot["clock"])
                fraction = 0.0 if arrival <= clock else max(
                    0.0, min(1.0, (tau - clock) / (arrival - clock)))
                p0 = truth.nodes[world.nodes[start]]["pos"]
                p1 = truth.nodes[world.nodes[end]]["pos"]
                xys.append((p0[0] + fraction * (p1[0] - p0[0]),
                            p0[1] + fraction * (p1[1] - p0[1])))
                route_start = end
                prefix = [start, end]
            else:
                xys.append(truth.nodes[world.nodes[node_index]]["pos"])
                prefix = [node_index]
            route = _route_from_snapshot(
                snapshot, i, route_start, len(world.nodes))
            indices = prefix + route[1:] if prefix[-1] == route[0] else prefix
            agent.planned_path = [world.nodes[node] for node in indices]
        render_frame(
            visible_env, truth, display_agents, frame_index,
            str(Path(frames_dir) / f"frame_{frame_index:04d}.png"),
            title=f"t = {tau:.1f}", agent_xy=xys)
    return len(frame_times)


def _cpu_episode(model, config, env, truth, agents, device,
                 render_dir=None, render_dt=1.0):
    from learning.policy.adapter import LearnedPolicyAdapter
    from learning.gpu_sim.rollout_cpu import collect_episode

    adapter = LearnedPolicyAdapter(
        model, config.model.num_target_types, training=False,
        candidate_config=config.candidates, device=device)
    oracle_makespan = parallel_tsp(truth, agents)
    rollout = collect_episode(
        env, truth, agents, adapter,
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty,
        render_dir=render_dir, render_dt=render_dt,
        oracle_makespan=oracle_makespan)
    result = rollout.result
    return {
        "return": float(rollout.episode_return),
        "makespan": float(result["makespan"]),
        "oracle_makespan": float(oracle_makespan),
        "normalized_regret": float(result["normalized_regret"]),
        "completed": bool(result["completed"]),
        "deaths": int(result["num_deaths"]),
        "remaining_targets": len(result["remaining_targets"]),
        "stalled": bool(not result["completed"] and result["survivors"]),
        "all_agents_dead": bool(not result["survivors"]),
    }


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


def _std(records, key):
    mean = _mean(records, key)
    return (sum((float(record[key]) - mean) ** 2 for record in records)
            / len(records)) ** 0.5


def summarize(records):
    """Return aggregate deterministic-policy statistics."""
    return {
        "episodes": len(records),
        "completion_rate": _mean(records, "completed"),
        "mean_return": _mean(records, "return"),
        "return_std": _std(records, "return"),
        "mean_makespan": _mean(records, "makespan"),
        "makespan_std": _std(records, "makespan"),
        "mean_oracle_makespan": _mean(records, "oracle_makespan"),
        "mean_normalized_regret": _mean(records, "normalized_regret"),
        "normalized_regret_std": _std(records, "normalized_regret"),
        "mean_deaths": _mean(records, "deaths"),
        "death_std": _std(records, "deaths"),
        "mean_remaining_targets": _mean(records, "remaining_targets"),
        "stalled_rate": _mean(records, "stalled"),
        "all_agents_dead_rate": _mean(records, "all_agents_dead"),
    }


def _episode_output_path(path, episode, episodes):
    """Give each rendered episode its own output when evaluating a set."""
    path = Path(path)
    if episodes == 1:
        return path
    return path.with_name(f"{path.stem}_episode_{episode:04d}{path.suffix}")


def evaluate(checkpoint, config_path=None, episodes=1, seed=None,
             device=None, render=False, render_dir=None, output_mp4=None,
             mp4_fps=4, render_dt=1.0):
    weights_path, checkpoint_config = _resolve_checkpoint(checkpoint)
    selected_config = config_path or checkpoint_config or DEFAULT_CONFIG_PATH
    config = load_config(selected_config)
    if seed is None:
        seed = config.training.seed
    torch.manual_seed(seed)
    requested_device = (device or config.training.device).lower()
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if requested_device == "auto" else requested_device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    model = build_policy(config.model).to(resolved_device)
    state_dict = torch.load(weights_path, map_location=resolved_device,
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    records = []
    terrain = None
    if render:
        from simulation.rendering import clear_frame_dir
        render_dir = Path(
            render_dir or weights_path.parent / "render_frames").resolve()
        output_mp4 = Path(
            output_mp4 or weights_path.parent / "render.mp4").resolve()
    with torch.no_grad():
        for episode in range(episodes):
            env, truth, agents = _episode_factory(seed + episode, config)
            if render:
                from simulation.rendering import make_mp4_from_frames
                episode_frames = (render_dir if episodes == 1 else
                                  render_dir / f"episode_{episode:04d}")
                clear_frame_dir(episode_frames)
                if resolved_device == "cuda":
                    trace = []
                    record, terrain, world = _gpu_episode(
                        model, config, env, truth, agents, resolved_device,
                        terrain=terrain, trace=trace)
                    _render_gpu_trace(
                        trace, env, truth, agents, world, episode_frames,
                        render_dt)
                    record["simulation_backend"] = "cuda_tensor"
                else:
                    record = _cpu_episode(
                        model, config, env, truth, agents, resolved_device,
                        render_dir=episode_frames, render_dt=render_dt)
                    record["simulation_backend"] = "cpu_render"
                episode_video = _episode_output_path(
                    output_mp4, episode, episodes)
                try:
                    make_mp4_from_frames(
                        episode_frames, episode_video, fps=mp4_fps)
                    record["video"] = str(episode_video)
                except (FileNotFoundError, subprocess.CalledProcessError) as error:
                    print(f"warning: MP4 creation failed: {error}", file=sys.stderr)
                record["frames"] = str(episode_frames)
            elif resolved_device == "cuda":
                record, terrain, _world = _gpu_episode(
                    model, config, env, truth, agents, resolved_device,
                    terrain=terrain)
            else:
                record = _cpu_episode(
                    model, config, env, truth, agents, resolved_device)
            record["episode"] = episode
            record["seed"] = seed + episode
            records.append(record)
    return records, summarize(records), config, weights_path


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained policy deterministically on the WV DEM")
    parser.add_argument(
        "checkpoint", help="timestamped checkpoint directory or weights file")
    parser.add_argument("--config", help="config override; defaults to checkpoint config")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", help="optional JSON result path")
    parser.add_argument("--render", action="store_true",
                        help="write one PNG per turn and compile an MP4")
    parser.add_argument("--render-dir",
                        default=str(DEFAULT_RENDER_DIR))
    parser.add_argument("--output-mp4",
                        default=str(DEFAULT_OUTPUT_MP4))
    parser.add_argument("--mp4-fps", type=int, default=4)
    parser.add_argument("--render-dt", type=float, default=1.0,
                        help="sim-time between rendered frames (continuous time)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.mp4_fps < 1:
        parser.error("--mp4-fps must be positive")
    if args.render_dt <= 0:
        parser.error("--render-dt must be positive")

    records, summary, config, weights = evaluate(
        args.checkpoint, args.config, args.episodes, args.seed, args.device,
        render=args.render, render_dir=args.render_dir,
        output_mp4=args.output_mp4, mp4_fps=args.mp4_fps,
        render_dt=args.render_dt)
    payload = {
        "checkpoint": str(weights),
        "deterministic": True,
        "instance": {
            "source_position": SOURCE_POSITION,
            "target_positions": TARGET_POSITIONS,
            "target_types": TARGET_TYPES,
            "agent_capabilities": (
                None if AGENT_CAPABILITIES is None
                else [sorted(values) for values in AGENT_CAPABILITIES]),
        },
        "summary": summary,
        "episodes": records,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
