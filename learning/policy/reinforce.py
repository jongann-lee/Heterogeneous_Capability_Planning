"""Dependency-light episodic REINFORCE optimization."""

import torch


class EMABaseline:
    def __init__(self, decay=0.95):
        self.decay = float(decay)
        self.value = None

    def update(self, episode_return):
        old = float(episode_return) if self.value is None else self.value
        self.value = self.decay * old + (1.0 - self.decay) * float(episode_return)
        return old


def reinforce_loss(rollout, baseline, entropy_coefficient=0.01):
    advantage = rollout.episode_return - float(baseline)
    return -(advantage * rollout.log_probability) \
        - entropy_coefficient * rollout.entropy


def optimization_step(optimizer, rollout, baseline: EMABaseline,
                      entropy_coefficient=0.01, gradient_clip_norm=1.0,
                      critic_coefficient=0.5):
    optimizer.zero_grad(set_to_none=True)
    if rollout.state_values.numel():
        targets = torch.full_like(
            rollout.state_values, float(rollout.episode_return))
        advantages = targets - rollout.state_values.detach()
        actor_loss = -(advantages * rollout.decision_log_probabilities).mean()
        entropy = rollout.decision_entropies.mean()
        critic_loss = torch.nn.functional.mse_loss(
            rollout.state_values, targets)
        loss = (actor_loss - entropy_coefficient * entropy
                + critic_coefficient * critic_loss)
        baseline_value = None
    else:
        baseline_value = baseline.update(rollout.episode_return)
        loss = reinforce_loss(rollout, baseline_value, entropy_coefficient)
        actor_loss = loss + entropy_coefficient * rollout.entropy
        critic_loss = loss.new_zeros(())
        entropy = rollout.entropy
    loss.backward()
    parameters = [p for group in optimizer.param_groups
                  for p in group["params"]]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, gradient_clip_norm)
    optimizer.step()
    optimization_step.last_metrics = {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "entropy": float(entropy.detach()),
        "baseline": baseline_value,
    }
    return float(loss.detach()), float(gradient_norm)


def batched_optimization_step(optimizer, rollout, baseline: EMABaseline,
                              entropy_coefficient=0.01,
                              gradient_clip_norm=1.0):
    """REINFORCE update for a vector of parallel episode rollouts."""
    optimizer.zero_grad(set_to_none=True)
    mean_return = float(rollout.returns.mean().detach())
    baseline_value = baseline.update(mean_return)
    advantages = rollout.returns.detach() - baseline_value
    loss = (-(advantages * rollout.log_probabilities)
            - entropy_coefficient * rollout.entropies).mean()
    loss.backward()
    parameters = [p for group in optimizer.param_groups
                  for p in group["params"]]
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters,
                                                   gradient_clip_norm)
    optimizer.step()
    return float(loss.detach()), float(gradient_norm)
