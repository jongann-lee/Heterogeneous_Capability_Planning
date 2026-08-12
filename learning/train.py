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
from learning.reinforce import EMABaseline, optimization_step
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


def main():
    parser = argparse.ArgumentParser(
        description="Train the centralized policy on the 64x64 WV DEM")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int)
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
        training, episodes=episodes, num_agents=num_agents, seed=seed,
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
