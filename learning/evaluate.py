"""Greedy checkpoint evaluation helpers and fixed-map CLI."""

import argparse

import torch

from learning.configuration import CandidateConfig, ModelConfig, load_config
from learning.model import CentralizedPolicy
from learning.instances import make_fixed_grid
from learning.policy_adapter import LearnedPolicyAdapter
from simulation.engine import run_simulation


def load_policy(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    if "candidate_config" in payload:
        candidate_payload = dict(payload["candidate_config"])
        candidate_payload.setdefault("include_continue", True)
        candidate_config = CandidateConfig(**candidate_payload)
    else:
        candidate_config = load_config().candidates
    model = CentralizedPolicy(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, LearnedPolicyAdapter(
        model, config.num_target_types, training=False,
        candidate_config=candidate_config, device=device)


def evaluate_instance(checkpoint, env_map, ground_truth, agents,
                      device="cpu", **simulation_kwargs):
    _model, policy = load_policy(checkpoint, device)
    return run_simulation(env_map, ground_truth, agents, policy=policy,
                          **simulation_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Greedy checkpoint evaluation")
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ntypes = int(payload["model_config"]["num_target_types"])
    env, truth, agents = make_fixed_grid(
        args.seed, args.grid_size, ntypes, args.num_agents)
    result = evaluate_instance(
        args.checkpoint, env, truth, agents, device=args.device)
    print({key: result[key] for key in (
        "completed", "makespan", "num_deaths", "remaining_targets")})


if __name__ == "__main__":
    main()
