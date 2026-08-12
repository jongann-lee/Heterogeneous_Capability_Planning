"""Training utilities and CLI for fixed-instance experiments."""

import argparse
from dataclasses import asdict, replace

import torch

from learning.configuration import (
    DEFAULT_CONFIG_PATH,
    CandidateConfig,
    ModelConfig,
    ReinforceConfig,
    load_config,
)
from learning.model import CentralizedPolicy
from learning.instances import make_fixed_grid
from learning.policy_adapter import LearnedPolicyAdapter
from learning.reinforce import EMABaseline, optimization_step
from learning.rollout import collect_episode


def train(instance_factory, num_target_types, episodes=100,
          model_config: ModelConfig | None = None,
          candidate_config: CandidateConfig | None = None,
          reinforce_config: ReinforceConfig | None = None, device="cpu",
          checkpoint=None):
    """Train on fresh instances returned as ``(env, truth, agents)``."""
    defaults = load_config()
    model_config = model_config or replace(
        defaults.model, num_target_types=num_target_types)
    candidate_config = candidate_config or defaults.candidates
    reinforce_config = reinforce_config or defaults.reinforce
    if model_config.num_target_types != num_target_types:
        raise ValueError("model_config.num_target_types must match the instance")
    model = CentralizedPolicy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    history = []
    for episode in range(episodes):
        env, truth, agents = instance_factory(episode)
        adapter = LearnedPolicyAdapter(
            model, num_target_types, training=True,
            candidate_config=candidate_config, device=device)
        rollout = collect_episode(
            env, truth, agents, adapter,
            reinforce_config.death_penalty,
            reinforce_config.incomplete_penalty)
        loss, grad_norm = optimization_step(
            optimizer, rollout, baseline,
            reinforce_config.entropy_coefficient,
            reinforce_config.gradient_clip_norm)
        history.append({"episode": episode, "return": rollout.episode_return,
                        "loss": loss, "gradient_norm": grad_norm,
                        "completed": rollout.result["completed"]})
    if checkpoint:
        torch.save({"model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "candidate_config": asdict(candidate_config)}, checkpoint)
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train the centralized policy on a fixed synthetic map")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-target-types", type=int)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config.training
    episodes = args.episodes if args.episodes is not None else training.episodes
    num_target_types = (args.num_target_types if args.num_target_types is not None
                        else config.model.num_target_types)
    num_agents = args.num_agents if args.num_agents is not None else training.num_agents
    grid_size = args.grid_size if args.grid_size is not None else training.grid_size
    seed = args.seed if args.seed is not None else training.seed
    requested_device = args.device or training.device
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if requested_device == "auto" else requested_device
    checkpoint = args.checkpoint or training.checkpoint
    model_config = replace(config.model, num_target_types=num_target_types)
    torch.manual_seed(seed)

    def factory(episode):
        return make_fixed_grid(
            seed + episode, grid_size, num_target_types, num_agents)

    _model, history = train(
        factory, num_target_types, episodes,
        model_config=model_config, candidate_config=config.candidates,
        reinforce_config=config.reinforce, device=device,
        checkpoint=checkpoint)
    last = history[-1] if history else {}
    print(f"checkpoint={checkpoint} episodes={episodes} "
          f"last_return={last.get('return')} completed={last.get('completed')}")


if __name__ == "__main__":
    main()
