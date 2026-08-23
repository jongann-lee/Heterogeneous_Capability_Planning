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
                             incomplete_penalty=1000.0,
                             oracle_makespan=None):
    makespan = float(result["makespan"])
    deaths = int(result["num_deaths"])
    remaining = len(result["remaining_targets"])
    if oracle_makespan is None:
        return (-makespan - death_penalty * deaths
                - incomplete_penalty * remaining)
    oracle = float(oracle_makespan)
    if oracle <= 0:
        raise ValueError("oracle_makespan must be positive")
    normalized_regret = makespan / oracle - 1.0
    return (-normalized_regret - death_penalty * deaths
            - incomplete_penalty * remaining)


def collect_episode(env_map, ground_truth, agents, adapter,
                    death_penalty=100.0, incomplete_penalty=1000.0,
                    max_events=100000, verbose=False,
                    render_dir=None, render_dt=1.0,
                    oracle_makespan=None):
    """Run one episode and retain the policy's differentiable trace."""
    adapter.reset_trace()
    result = run_simulation(
        env_map, ground_truth, agents, policy=adapter,
        death_penalty=death_penalty, max_events=max_events, verbose=verbose,
        render_dir=render_dir, render_dt=render_dt)
    parameter = next(adapter.model.parameters())
    zero = parameter.sum() * 0.0
    logp = torch.stack(adapter.decision_log_probs).sum() \
        if adapter.decision_log_probs else zero
    entropy = torch.stack(adapter.decision_entropies).sum() \
        if adapter.decision_entropies else zero
    episode_return = calculate_episode_return(
        result, death_penalty, incomplete_penalty, oracle_makespan)
    result["oracle_makespan"] = oracle_makespan
    result["normalized_regret"] = (
        None if oracle_makespan is None else
        float(result["makespan"]) / float(oracle_makespan) - 1.0)
    return Rollout(result, episode_return, logp, entropy)
