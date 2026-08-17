"""Deterministic checkpoint evaluation on the 64x64 WV terrain."""

import argparse
import json
from pathlib import Path

import torch

from learning.configuration import DEFAULT_CONFIG_PATH, load_config
from learning.instances import make_wv_dem_instance
from learning.model import CentralizedPolicy


# Fixed sanity-check setup. Set a value to None to sample that component from
# each episode seed. These intentionally mirror the editable globals in train.py.
SOURCE_POSITION = (0, 0)
TARGET_POSITIONS = [
    (14, 54), (1, 29), (33, 17), (34, 35),
    (63, 37), (37, 5), (49, 58),
]
TARGET_TYPES = [1, 2, 2, 1, 2, 3, 3]
AGENT_CAPABILITIES = [{0}, {1}, {2}, {3}]


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
        agent_capabilities=AGENT_CAPABILITIES)


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


def _gpu_episode(model, config, env, truth, agents, device, terrain=None):
    from learning.gpu.observation import TensorObservationBuilder
    from learning.gpu.rollout import collect_tensor_episodes
    from learning.gpu.state import TensorEpisodeState
    from learning.gpu.world import TensorWorld

    world = TensorWorld.from_networkx(
        env, config.candidates, device=device, terrain=terrain)
    source, capabilities, target_types = _encode_episode(
        world, truth, agents, config.model.num_target_types, device)
    state = TensorEpisodeState.create(
        world, [source], capabilities, target_types)
    rollout = collect_tensor_episodes(
        model, state,
        TensorObservationBuilder(world, config.model.num_target_types),
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty,
        training=False)
    record = {
        "return": float(rollout.returns[0]),
        "makespan": float(rollout.makespans[0]),
        "completed": bool(rollout.completed[0]),
        "deaths": int(rollout.deaths[0]),
        "remaining_targets": int(rollout.remaining_targets[0]),
        "stalled": bool(rollout.stalled[0]),
        "all_agents_dead": bool(rollout.all_agents_dead[0]),
    }
    return record, world.terrain


def _cpu_episode(model, config, env, truth, agents, device):
    from learning.policy_adapter import LearnedPolicyAdapter
    from learning.rollout import collect_episode

    adapter = LearnedPolicyAdapter(
        model, config.model.num_target_types, training=False,
        candidate_config=config.candidates, device=device)
    rollout = collect_episode(
        env, truth, agents, adapter,
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty)
    result = rollout.result
    return {
        "return": float(rollout.episode_return),
        "makespan": float(result["makespan"]),
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
        "mean_deaths": _mean(records, "deaths"),
        "death_std": _std(records, "deaths"),
        "mean_remaining_targets": _mean(records, "remaining_targets"),
        "stalled_rate": _mean(records, "stalled"),
        "all_agents_dead_rate": _mean(records, "all_agents_dead"),
    }


def evaluate(checkpoint, config_path=None, episodes=1, seed=None,
             device=None):
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

    model = CentralizedPolicy(config.model).to(resolved_device)
    state_dict = torch.load(weights_path, map_location=resolved_device,
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    records = []
    terrain = None
    with torch.no_grad():
        for episode in range(episodes):
            env, truth, agents = _episode_factory(seed + episode, config)
            if resolved_device == "cuda":
                record, terrain = _gpu_episode(
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
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    records, summary, config, weights = evaluate(
        args.checkpoint, args.config, args.episodes, args.seed, args.device)
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
