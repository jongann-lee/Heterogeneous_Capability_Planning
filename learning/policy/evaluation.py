"""Greedy checkpoint evaluation helpers and fixed-map CLI."""

import argparse
from pathlib import Path

import torch
import yaml

from learning.policy.configuration import CandidateConfig, ModelConfig, load_config
from learning.policy.model import build_policy
from learning.gpu_sim.instances import make_wv_dem_instance
from learning.policy.adapter import LearnedPolicyAdapter
from simulation.engine import run_simulation


def load_policy(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    weights_path = (checkpoint / "trained_weights.pt"
                    if checkpoint.is_dir() else checkpoint)
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if weights_path.name == "trained_weights.pt":
        saved_config = load_config(weights_path.with_name("config.yaml"))
        config = saved_config.model
        candidate_config = saved_config.candidates
        state_dict = payload
    else:
        # Backward compatibility with the original bundled checkpoint format.
        config = ModelConfig(**payload["model_config"])
        state_dict = payload["model"]
        candidate_payload = dict(payload.get("candidate_config", {}))
        candidate_payload.pop("include_continue", None)
        candidate_config = (CandidateConfig(**candidate_payload)
                            if candidate_payload else load_config().candidates)
    model = build_policy(config).to(device)
    model.load_state_dict(state_dict)
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
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    model, _policy = load_policy(args.checkpoint, "cpu")
    ntypes = model.config.num_target_types
    checkpoint = Path(args.checkpoint)
    instance = {}
    if checkpoint.is_dir():
        with (checkpoint / "config.yaml").open("r", encoding="utf-8") as stream:
            instance = (yaml.safe_load(stream) or {}).get("instance", {})
    env, truth, agents = make_wv_dem_instance(
        args.seed, ntypes, args.num_agents,
        source_position=instance.get("source_position"),
        target_positions=instance.get("target_positions"),
        target_types=instance.get("target_types"),
        agent_capabilities=instance.get("agent_capabilities"))
    result = evaluate_instance(
        args.checkpoint, env, truth, agents, device=args.device)
    print({key: result[key] for key in (
        "completed", "makespan", "num_deaths", "remaining_targets")})


if __name__ == "__main__":
    main()
