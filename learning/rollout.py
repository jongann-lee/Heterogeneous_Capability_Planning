"""Complete episodic rollout collection."""

from dataclasses import dataclass

import torch

from simulation.engine import run_simulation


@dataclass
class Rollout:
    result: dict
    episode_return: float
    log_probability: torch.Tensor
    entropy: torch.Tensor


def calculate_episode_return(result, death_penalty=100.0,
                             incomplete_penalty=1000.0):
    return (-float(result["makespan"])
            - death_penalty * int(result["num_deaths"])
            - incomplete_penalty * len(result["remaining_targets"]))


def collect_episode(env_map, ground_truth, agents, adapter,
                    death_penalty=100.0, incomplete_penalty=1000.0,
                    max_events=100000, verbose=False):
    """Run one episode and retain the policy's differentiable trace."""
    adapter.reset_trace()
    result = run_simulation(
        env_map, ground_truth, agents, policy=adapter,
        death_penalty=death_penalty, max_events=max_events, verbose=verbose)
    parameter = next(adapter.model.parameters())
    zero = parameter.sum() * 0.0
    logp = torch.stack(adapter.decision_log_probs).sum() \
        if adapter.decision_log_probs else zero
    entropy = torch.stack(adapter.decision_entropies).sum() \
        if adapter.decision_entropies else zero
    episode_return = calculate_episode_return(
        result, death_penalty, incomplete_penalty)
    return Rollout(result, episode_return, logp, entropy)
