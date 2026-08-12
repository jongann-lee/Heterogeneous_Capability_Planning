"""Training utilities and CLI for fixed-instance experiments."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import yaml

from learning.configuration import (
    DEFAULT_CONFIG_PATH,
    CandidateConfig,
    LearningConfig,
    ModelConfig,
    ReinforceConfig,
    load_config,
)
from learning.model import CentralizedPolicy
from learning.instances import make_wv_dem_instance
from learning.policy_adapter import LearnedPolicyAdapter
from learning.reinforce import (EMABaseline, batched_optimization_step,
                                optimization_step)
from learning.rollout import collect_episode


# Fixed sanity-check setup. Set any value to None (or comment out its line) to
# sample that component independently for every episode.
SOURCE_POSITION = (0, 0)
TARGET_POSITIONS = [(14,54), (1,29), (33,17), (34,35), (63,37), (37,5), (49,58)]
TARGET_TYPES = [1, 2, 2, 1, 2, 3, 3]
AGENT_CAPABILITIES = [{0}, {1}, {2}, {3}]


def _instance_config():
    """Return serializable values for the active fixed/random setup."""
    capabilities = globals().get("AGENT_CAPABILITIES")
    return {
        "source_position": globals().get("SOURCE_POSITION"),
        "target_positions": globals().get("TARGET_POSITIONS"),
        "target_types": globals().get("TARGET_TYPES"),
        "agent_capabilities": (None if capabilities is None else
                               [sorted(values) for values in capabilities]),
    }


def train(instance_factory, num_target_types, episodes=100,
          model_config: ModelConfig | None = None,
          candidate_config: CandidateConfig | None = None,
          reinforce_config: ReinforceConfig | None = None, device="cpu",
          checkpoint=None, run_config: LearningConfig | None = None):
    """Train on fresh instances returned as ``(env, truth, agents)``."""
    defaults = load_config()
    model_config = model_config or replace(
        defaults.model, num_target_types=num_target_types)
    candidate_config = candidate_config or defaults.candidates
    reinforce_config = reinforce_config or defaults.reinforce
    if model_config.num_target_types != num_target_types:
        raise ValueError("model_config.num_target_types must match the instance")
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_directory = None
    if checkpoint:
        run_directory = Path(checkpoint) / timestamp
        run_directory.mkdir(parents=True, exist_ok=False)
    if run_config is None:
        training_config = replace(
            defaults.training, episodes=episodes, checkpoint=str(checkpoint))
        run_config = LearningConfig(
            model_config, candidate_config, reinforce_config, training_config)

    wandb_run = None
    if run_config.training.wandb:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "training.wandb is enabled, but wandb is not installed") from error
        logged_config = asdict(run_config)
        logged_config["instance"] = _instance_config()
        wandb_run = wandb.init(
            project="heterogeneous-capability-planning",
            name=timestamp,
            config=logged_config,
            dir=str(run_directory) if run_directory is not None else None,
        )

    model = CentralizedPolicy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    history = []
    try:
        for episode in range(episodes):
            env, truth, agents = instance_factory(episode)
            adapter = LearnedPolicyAdapter(
                model, num_target_types, training=True,
                candidate_config=candidate_config, device=device)
            rollout = collect_episode(
                env, truth, agents, adapter,
                reinforce_config.death_penalty,
                reinforce_config.incomplete_penalty)
            loss, _grad_norm = optimization_step(
                optimizer, rollout, baseline,
                reinforce_config.entropy_coefficient,
                reinforce_config.gradient_clip_norm)
            metrics = {
                "episode": episode,
                "return": rollout.episode_return,
                "loss": loss,
                "makespan": rollout.result["makespan"],
                "completed": rollout.result["completed"],
            }
            history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=episode)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if run_directory is not None:
        torch.save(model.state_dict(), run_directory / "trained_weights.pt")
        saved_config = asdict(run_config)
        saved_config["instance"] = _instance_config()
        with (run_directory / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, sort_keys=False)
    train.last_run_directory = run_directory
    return model, history


def train_gpu(env, truth, agents, episodes, batch_size, model_config,
              candidate_config, reinforce_config, run_config, device,
              checkpoint=None):
    """Train with parallel CUDA tensor episodes on one immutable WV world."""
    from learning.gpu.observation import TensorObservationBuilder
    from learning.gpu.rollout import collect_tensor_episodes
    from learning.gpu.state import TensorEpisodeState
    from learning.gpu.world import TensorWorld

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_directory = Path(checkpoint) / timestamp if checkpoint else None
    if run_directory is not None:
        run_directory.mkdir(parents=True, exist_ok=False)
    wandb_run = None
    if run_config.training.wandb:
        import wandb
        logged_config = asdict(run_config)
        logged_config["instance"] = _instance_config()
        logged_config["backend"] = "cuda_tensor"
        wandb_run = wandb.init(
            project="heterogeneous-capability-planning", name=timestamp,
            config=logged_config,
            dir=str(run_directory) if run_directory is not None else None)

    world = TensorWorld.from_networkx(env, candidate_config, device=device)
    builder = TensorObservationBuilder(world, model_config.num_target_types)
    model = CentralizedPolicy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    source = world.node_index[agents[0].position]
    caps = torch.zeros((len(agents), model_config.num_target_types + 1),
                       dtype=torch.bool, device=device)
    for agent_index, agent in enumerate(agents):
        for capability in agent.capabilities:
            caps[agent_index, capability] = True
    types = torch.tensor(
        [truth.nodes[target]["rps_type"] for target in world.targets],
        dtype=torch.long, device=device)
    history = []
    completed_episodes = 0
    try:
        while completed_episodes < episodes:
            current_batch = min(batch_size, episodes - completed_episodes)
            state = TensorEpisodeState.create(
                world, torch.full((current_batch,), source, device=device),
                caps[None].expand(current_batch, -1, -1).clone(),
                types[None].expand(current_batch, -1).clone())
            rollout = collect_tensor_episodes(
                model, state, builder, reinforce_config.death_penalty,
                reinforce_config.incomplete_penalty, training=True)
            loss, _gradient_norm = batched_optimization_step(
                optimizer, rollout, baseline,
                reinforce_config.entropy_coefficient,
                reinforce_config.gradient_clip_norm)
            for item in range(current_batch):
                episode = completed_episodes + item
                metrics = {
                    "episode": episode,
                    "return": float(rollout.returns[item]),
                    "loss": loss,
                    "makespan": float(rollout.makespans[item]),
                    "completed": bool(rollout.completed[item]),
                }
                history.append(metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=episode)
            completed_episodes += current_batch
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if run_directory is not None:
        torch.save(model.state_dict(), run_directory / "trained_weights.pt")
        saved_config = asdict(run_config)
        saved_config["instance"] = _instance_config()
        saved_config["backend"] = "cuda_tensor"
        with (run_directory / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, sort_keys=False)
    return model, history, world, run_directory


def main():
    parser = argparse.ArgumentParser(
        description="Train the centralized policy on the 64x64 WV DEM")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-target-types", type=int)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--checkpoint", "--checkpoint-dir", dest="checkpoint",
        help="directory under which a timestamped run folder is created")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config.training
    episodes = args.episodes if args.episodes is not None else training.episodes
    batch_size = (args.batch_size if args.batch_size is not None
                  else training.batch_size)
    num_target_types = (args.num_target_types if args.num_target_types is not None
                        else config.model.num_target_types)
    num_agents = args.num_agents if args.num_agents is not None else training.num_agents
    seed = args.seed if args.seed is not None else training.seed
    requested_device = args.device or training.device
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if requested_device == "auto" else requested_device
    checkpoint = args.checkpoint or training.checkpoint
    model_config = replace(config.model, num_target_types=num_target_types)
    resolved_training = replace(
        training, episodes=episodes, batch_size=batch_size,
        num_agents=num_agents, seed=seed,
        device=requested_device, checkpoint=checkpoint)
    resolved_config = LearningConfig(
        model_config, config.candidates, config.reinforce, resolved_training)
    torch.manual_seed(seed)

    def factory(episode):
        return make_wv_dem_instance(
            seed + episode, num_target_types, num_agents,
            source_position=globals().get("SOURCE_POSITION"),
            target_positions=globals().get("TARGET_POSITIONS"),
            target_types=globals().get("TARGET_TYPES"),
            agent_capabilities=globals().get("AGENT_CAPABILITIES"))

    if device == "cuda":
        env, truth, agents = factory(0)
        _model, history, _world, run_directory = train_gpu(
            env, truth, agents, episodes, batch_size, model_config,
            config.candidates, config.reinforce, resolved_config, device,
            checkpoint=checkpoint)
        train.last_run_directory = run_directory
    else:
        _model, history = train(
            factory, num_target_types, episodes,
            model_config=model_config, candidate_config=config.candidates,
            reinforce_config=config.reinforce, device=device,
            checkpoint=checkpoint, run_config=resolved_config)
    last = history[-1] if history else {}
    print(f"run_directory={train.last_run_directory} episodes={episodes} "
          f"last_return={last.get('return')} completed={last.get('completed')}")


if __name__ == "__main__":
    main()
