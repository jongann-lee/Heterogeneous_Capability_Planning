"""Training utilities and CLI for fixed-instance experiments."""

import argparse

import torch

from learning.config import ModelConfig, ReinforceConfig
from learning.model import CentralizedPolicy
from learning.instances import make_fixed_grid
from learning.policy_adapter import LearnedPolicyAdapter
from learning.reinforce import EMABaseline, optimization_step
from learning.rollout import collect_episode


def train(instance_factory, num_target_types, episodes=100,
          model_config=None, reinforce_config=None, device="cpu",
          checkpoint=None):
    """Train on fresh instances returned as ``(env, truth, agents)``."""
    model_config = model_config or ModelConfig(num_target_types)
    reinforce_config = reinforce_config or ReinforceConfig()
    model = CentralizedPolicy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    history = []
    for episode in range(episodes):
        env, truth, agents = instance_factory(episode)
        adapter = LearnedPolicyAdapter(
            model, num_target_types, training=True, device=device)
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
                    "model_config": vars(model_config)}, checkpoint)
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train the centralized policy on a fixed synthetic map")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-target-types", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--checkpoint", default="learning_policy.pt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    def factory(episode):
        return make_fixed_grid(
            args.seed + episode, args.grid_size, args.num_target_types,
            args.num_agents)

    _model, history = train(
        factory, args.num_target_types, args.episodes,
        device=args.device, checkpoint=args.checkpoint)
    last = history[-1] if history else {}
    print(f"checkpoint={args.checkpoint} episodes={args.episodes} "
          f"last_return={last.get('return')} completed={last.get('completed')}")


if __name__ == "__main__":
    main()
