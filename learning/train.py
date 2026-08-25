"""Training utilities and CLI for WV-terrain experiments."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import yaml

from learning.policy.configuration import (
    DEFAULT_CONFIG_PATH,
    CandidateConfig,
    LearningConfig,
    ModelConfig,
    ReinforceConfig,
    load_config,
)
from learning.policy.model import build_policy
from learning.gpu_sim.instances import make_wv_dem_instance
from learning.policy.oracle import parallel_tsp
from learning.policy.adapter import LearnedPolicyAdapter
from learning.policy.reinforce import (EMABaseline, batched_optimization_step,
                                       optimization_step)
from learning.gpu_sim.rollout_cpu import collect_episode


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
            model_config, candidate_config, reinforce_config, training_config,
            defaults.instances)

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

    model = build_policy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    history = []
    try:
        for episode in range(episodes):
            env, truth, agents = instance_factory(episode)
            oracle_makespan = parallel_tsp(truth, agents)
            adapter = LearnedPolicyAdapter(
                model, num_target_types, training=True,
                candidate_config=candidate_config, device=device)
            rollout = collect_episode(
                env, truth, agents, adapter,
                reinforce_config.death_penalty,
                reinforce_config.incomplete_penalty,
                oracle_makespan=oracle_makespan)
            loss, grad_norm = optimization_step(
                optimizer, rollout, baseline,
                reinforce_config.entropy_coefficient,
                reinforce_config.gradient_clip_norm,
                reinforce_config.critic_coefficient)
            loss_metrics = optimization_step.last_metrics
            metrics = {
                "episode": episode,
                "return": rollout.episode_return,
                "loss": loss,
                "makespan": rollout.result["makespan"],
                "oracle_makespan": oracle_makespan,
                "normalized_regret": rollout.result["normalized_regret"],
                "completed": rollout.result["completed"],
                "actor_loss": loss_metrics["actor_loss"],
                "critic_loss": loss_metrics["critic_loss"],
                "entropy": loss_metrics["entropy"],
                "gradient_norm": grad_norm,
                "decisions": int(rollout.decision_log_probabilities.numel()),
                "state_value": (float(rollout.state_values.mean().detach())
                                if rollout.state_values.numel() else 0.0),
                "target_count": (len(rollout.result["eliminated_targets"])
                                 + len(rollout.result["remaining_targets"])),
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


def train_gpu(env, truth, agents, episodes, simulation_batch_size,
              reinforce_batch_size, model_config,
              candidate_config, reinforce_config, run_config, device,
              checkpoint=None, instance_factory=None):
    """Train with parallel CUDA tensor episodes on one immutable WV world."""
    from learning.gpu_sim.observation_gpu import TensorObservationBuilder
    from learning.gpu_sim.rollout_gpu import (collect_tensor_episodes,
                                              replay_tensor_gradients)
    from learning.gpu_sim.state import TensorEpisodeState
    from learning.gpu_sim.world import TensorWorld

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
    terrain = world.terrain
    builder = TensorObservationBuilder(
        world, model_config.num_target_types,
        task_graph=model_config.architecture == "task_graph")
    model = build_policy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=reinforce_config.learning_rate)
    baseline = EMABaseline(reinforce_config.baseline_decay)
    def encode_episode(episode_world, episode_truth, episode_agents):
        episode_source = episode_world.node_index[episode_agents[0].position]
        episode_caps = torch.zeros(
            (len(episode_agents), model_config.num_target_types + 1),
            dtype=torch.bool, device=device)
        for agent_index, agent in enumerate(episode_agents):
            for capability in agent.capabilities:
                episode_caps[agent_index, capability] = True
        episode_types = torch.tensor(
            [episode_truth.nodes[target]["rps_type"]
             for target in episode_world.targets],
            dtype=torch.long, device=device)
        return episode_source, episode_caps, episode_types

    source, caps, types = encode_episode(world, truth, agents)
    oracle_makespan = parallel_tsp(truth, agents)
    random_overlay = instance_factory is not None and any(
        globals().get(name) is None for name in (
            "SOURCE_POSITION", "TARGET_POSITIONS", "TARGET_TYPES",
            "AGENT_CAPABILITIES"))
    history = []
    completed_episodes = 0
    try:
        while completed_episodes < episodes:
            update_size = min(reinforce_batch_size,
                              episodes - completed_episodes)
            optimizer.zero_grad(set_to_none=True)
            # Keep the baseline fixed across all microbatches in this update.
            # Zero is the conventional initial baseline and avoids retaining a
            # first pass of trajectory graphs just to initialize the EMA.
            baseline_value = 0.0 if baseline.value is None else baseline.value
            update_return_sum = 0.0
            update_records = []
            update_rollouts = []
            accumulated = 0
            while accumulated < update_size:
                current_batch = min(simulation_batch_size,
                                    update_size - accumulated)
                if random_overlay:
                    episode_number = completed_episodes + accumulated
                    batch_env, batch_truth, batch_agents = instance_factory(
                        episode_number)
                    world = TensorWorld.from_networkx(
                        batch_env, candidate_config, device=device,
                        terrain=terrain)
                    builder = TensorObservationBuilder(
                        world, model_config.num_target_types,
                        task_graph=model_config.architecture == "task_graph")
                    source, caps, types = encode_episode(
                        world, batch_truth, batch_agents)
                    oracle_makespan = parallel_tsp(
                        batch_truth, batch_agents)
                state = TensorEpisodeState.create(
                    world, torch.full((current_batch,), source, device=device),
                    caps[None].expand(current_batch, -1, -1).clone(),
                    types[None].expand(current_batch, -1).clone())
                rollout = collect_tensor_episodes(
                    model, state, builder, reinforce_config.death_penalty,
                    reinforce_config.incomplete_penalty, training=True,
                    oracle_makespans=oracle_makespan)
                update_return_sum += float(rollout.returns.detach().sum())
                update_rollouts.append(rollout)
                accumulated += current_batch
                del state
                # Return PyTorch's freed activation blocks to CUDA so RAPIDS
                # can service subsequent cuGraph allocations.
                torch.cuda.empty_cache()

            batch_returns = torch.cat(
                [rollout.returns.detach() for rollout in update_rollouts])
            raw_advantages = (batch_returns if model.has_critic
                              else batch_returns - baseline_value)
            advantage_mean = raw_advantages.mean()
            advantage_std = raw_advantages.std(unbiased=False)
            episode_signals = (
                batch_returns if model.has_critic else
                (raw_advantages - advantage_mean)
                / advantage_std.clamp_min(1.0e-8))

            advantage_offset = 0
            for rollout in update_rollouts:
                current_batch = rollout.returns.numel()
                rollout_signals = episode_signals[
                    advantage_offset:advantage_offset + current_batch]
                (detached_losses, decision_counts, critic_losses,
                 entropies, state_values) = replay_tensor_gradients(
                    model, rollout, rollout_signals,
                    reinforce_config.entropy_coefficient, update_size, device,
                    reinforce_config.critic_coefficient)
                for item in range(current_batch):
                    update_records.append({
                        "return": float(rollout.returns[item]),
                        "loss": float(detached_losses[item]),
                        "makespan": float(rollout.makespans[item]),
                        "oracle_makespan": float(
                            rollout.oracle_makespans[item]),
                        "normalized_regret": float(
                            rollout.normalized_regrets[item]),
                        "completed": bool(rollout.completed[item]),
                        "deaths": int(rollout.deaths[item]),
                        "remaining_targets": int(
                            rollout.remaining_targets[item]),
                        "stalled": bool(rollout.stalled[item]),
                        "all_agents_dead": bool(
                            rollout.all_agents_dead[item]),
                        "decisions": int(decision_counts[item]),
                        "critic_loss": float(critic_losses[item]),
                        "entropy": float(entropies[item]),
                        "state_value": float(state_values[item]),
                        "target_count": int(rollout.target_counts[item]),
                    })
                advantage_offset += current_batch
                del rollout, detached_losses, decision_counts, critic_losses
                del entropies, state_values
                torch.cuda.empty_cache()
            del update_rollouts, batch_returns, raw_advantages
            del episode_signals

            parameters = [parameter for group in optimizer.param_groups
                          for parameter in group["params"]]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, reinforce_config.gradient_clip_norm)
            optimizer.step()
            if not model.has_critic:
                baseline.update(update_return_sum / update_size)
            for offset, metrics in enumerate(update_records):
                episode = completed_episodes + offset
                metrics["episode"] = episode
                history.append(metrics)
            if wandb_run is not None:
                update_index = completed_episodes // reinforce_batch_size
                completed_regrets = [
                    record["normalized_regret"] for record in update_records
                    if record["completed"]]
                return_tensor = torch.tensor(
                    [record["return"] for record in update_records])
                value_tensor = torch.tensor(
                    [record["state_value"] for record in update_records])
                return_variance = return_tensor.var(unbiased=False)
                explained_variance = (
                    1.0 - (return_tensor - value_tensor).var(unbiased=False)
                    / return_variance
                    if model.has_critic and return_variance > 1e-12 else
                    torch.tensor(0.0))
                logged_metrics = {
                    "update": update_index,
                    "episodes_seen": completed_episodes + update_size,
                    "mean_return": sum(record["return"]
                                       for record in update_records) / update_size,
                    "mean_loss": sum(record["loss"]
                                     for record in update_records) / update_size,
                    "mean_critic_loss": sum(record["critic_loss"]
                                             for record in update_records) / update_size,
                    "mean_policy_entropy": sum(record["entropy"]
                                                for record in update_records) / update_size,
                    "gradient_norm": float(gradient_norm),
                    "mean_decision_count": sum(record["decisions"]
                                                for record in update_records) / update_size,
                    "critic_explained_variance": float(explained_variance),
                    "completed_only_normalized_regret": (
                        sum(completed_regrets) / len(completed_regrets)
                        if completed_regrets else float("nan")),
                    "advantage_mean": float(advantage_mean),
                    "advantage_std": float(advantage_std),
                    "mean_makespan": sum(record["makespan"]
                                         for record in update_records) / update_size,
                    "mean_oracle_makespan": sum(
                        record["oracle_makespan"]
                        for record in update_records) / update_size,
                    "mean_normalized_regret": sum(
                        record["normalized_regret"]
                        for record in update_records) / update_size,
                    "completion_rate": sum(record["completed"]
                                           for record in update_records) / update_size,
                    "mean_deaths": sum(record["deaths"]
                                       for record in update_records) / update_size,
                    "mean_remaining_targets": sum(
                        record["remaining_targets"]
                        for record in update_records) / update_size,
                    "stalled_rate": sum(record["stalled"]
                                        for record in update_records) / update_size,
                    "all_agents_dead_rate": sum(
                        record["all_agents_dead"]
                        for record in update_records) / update_size,
                }
                target_counts = sorted({
                    record["target_count"] for record in update_records})
                for target_count in target_counts:
                    group = [record for record in update_records
                             if record["target_count"] == target_count]
                    logged_metrics[
                        f"completion_rate/targets_{target_count}"] = sum(
                            record["completed"] for record in group) / len(group)
                wandb_run.log(logged_metrics, step=update_index)
            completed_episodes += update_size
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
        description="Train the configured learned policy on the 64x64 WV DEM")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--simulation-batch-size", type=int)
    parser.add_argument("--reinforce-batch-size", type=int)
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
    simulation_batch_size = (
        args.simulation_batch_size
        if args.simulation_batch_size is not None
        else training.simulation_batch_size)
    reinforce_batch_size = (
        args.reinforce_batch_size
        if args.reinforce_batch_size is not None
        else training.reinforce_batch_size)
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
        training, episodes=episodes,
        simulation_batch_size=simulation_batch_size,
        reinforce_batch_size=reinforce_batch_size,
        num_agents=num_agents, seed=seed,
        device=requested_device, checkpoint=checkpoint)
    resolved_config = LearningConfig(
        model_config, config.candidates, config.reinforce, resolved_training,
        config.instances)
    torch.manual_seed(seed)

    def factory(episode):
        return make_wv_dem_instance(
            seed + episode, num_target_types, num_agents,
            source_position=globals().get("SOURCE_POSITION"),
            target_positions=globals().get("TARGET_POSITIONS"),
            target_types=globals().get("TARGET_TYPES"),
            agent_capabilities=globals().get("AGENT_CAPABILITIES"),
            min_targets=config.instances.min_targets,
            max_targets=config.instances.max_targets)

    if device == "cuda":
        env, truth, agents = factory(0)
        _model, history, _world, run_directory = train_gpu(
            env, truth, agents, episodes, simulation_batch_size,
            reinforce_batch_size, model_config,
            config.candidates, config.reinforce, resolved_config, device,
            checkpoint=checkpoint, instance_factory=factory)
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
