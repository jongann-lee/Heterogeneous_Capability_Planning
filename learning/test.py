"""Deterministic checkpoint evaluation on a selected prepared terrain."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from Real_Life_Maps.prepared_map import DEFAULT_PREPARED_MAP_PATH
from learning.evaluation_suite import (
    load_evaluation_suite,
    select_evaluation_cases,
    validate_evaluation_suite_locations,
)
from learning.policy.configuration import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    RAW_DISTANCE_FEATURE_SCHEMA_VERSION,
    feature_schema_metadata,
    load_config,
)
from learning.gpu_sim.instances import (
    inspect_prepared_map,
    make_prepared_map_instance,
)
from learning.policy.model import build_policy
from learning.policy.oracle import full_information_makespan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "outputs" / "my_policy_simulation"
DEFAULT_RENDER_DIR = DEFAULT_RENDER_ROOT / "frames"
DEFAULT_OUTPUT_MP4 = DEFAULT_RENDER_ROOT / "render_result.mp4"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def _resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        weights = next((candidate for candidate in (
            path / "trained_weights.pt",
            path / "best_weights.pt",
            path / "latest_weights.pt",
        ) if candidate.is_file()), path / "trained_weights.pt")
        config = path / "config.yaml"
    else:
        weights = path
        config = path.with_name("config.yaml")
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights}")
    return weights, config if config.is_file() else None


def _evaluation_output_path(output=None, now=None):
    """Resolve an explicit output or create a timestamped default JSON path."""
    if output is not None:
        return Path(output).expanduser().resolve()
    timestamp = (now or datetime.now().astimezone()).strftime(
        "%Y-%m-%d_%H-%M-%S_%f")
    return DEFAULT_EVALUATION_DIR / f"{timestamp}.json"


def _case_factory(case, map_path=DEFAULT_PREPARED_MAP_PATH):
    metadata = case.instance_metadata()
    return make_prepared_map_instance(
        seed=0,
        num_target_types=case.num_target_types,
        num_agents=case.agent.agent_count,
        source_position=metadata["source_position"],
        target_positions=metadata["target_positions"],
        target_types=metadata["target_types"],
        agent_capabilities=metadata["agent_capabilities"],
        min_targets=case.target.target_count,
        max_targets=case.target.target_count,
        map_path=map_path)


def _checkpoint_prepared_map(config_path):
    """Read map identity recorded beside checkpoint weights, if available."""
    if config_path is None:
        return None
    with Path(config_path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint configuration must be a YAML mapping")
    prepared_map = payload.get("prepared_map")
    if prepared_map is None:
        return None
    if not isinstance(prepared_map, dict):
        raise ValueError("checkpoint prepared_map identity must be a mapping")
    sha256 = prepared_map.get("sha256")
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(character not in "0123456789abcdefABCDEF"
                   for character in sha256)):
        raise ValueError(
            "checkpoint prepared_map.sha256 must be a 64-character hex digest")
    return dict(prepared_map)


def _select_prepared_map_path(cli_map_path, explicit_config,
                              checkpoint_config):
    """Apply CLI, explicit-config, checkpoint-config, then default precedence."""
    if cli_map_path is not None:
        return cli_map_path
    if (explicit_config is not None
            and explicit_config.instances.map_path is not None):
        return explicit_config.instances.map_path
    if (checkpoint_config is not None
            and checkpoint_config.instances.map_path is not None):
        return checkpoint_config.instances.map_path
    return DEFAULT_PREPARED_MAP_PATH


def _encode_episode(world, truth, agents, num_target_types, device):
    source = world.node_index[agents[0].position]
    capabilities = torch.zeros(
        (1, len(agents), num_target_types + 1),
        dtype=torch.bool, device=device)
    for agent_index, agent in enumerate(agents):
        for capability in agent.capabilities:
            capabilities[0, agent_index, capability] = True
    target_types = torch.tensor([[
        truth.nodes[target]["rps_type"] for target in world.targets
    ]], dtype=torch.long, device=device)
    return source, capabilities, target_types


def _capture_state(trace):
    """Return a callback that stores only one CPU snapshot per event time."""
    def capture(state):
        names = ("positions", "alive", "target_live", "target_known", "moving",
                 "transit_from", "transit_to", "arrival_time", "clock",
                 "route_next")
        snapshot = {name: getattr(state, name)[0].detach().cpu().clone()
                    for name in names}
        if trace and float(trace[-1]["clock"]) == float(snapshot["clock"]):
            trace[-1] = snapshot
        else:
            trace.append(snapshot)
    return capture


def _gpu_episode(model, config, env, truth, agents, device, terrain=None,
                 trace=None):
    from learning.gpu_sim.observation_gpu import TensorObservationBuilder
    from learning.gpu_sim.rollout_gpu import collect_tensor_episodes
    from learning.gpu_sim.state import TensorEpisodeState
    from learning.gpu_sim.world import TensorWorld

    world = TensorWorld.from_networkx(
        env, config.candidates, device=device, terrain=terrain)
    source, capabilities, target_types = _encode_episode(
        world, truth, agents, config.model.num_target_types, device)
    state = TensorEpisodeState.create(
        world, [source], capabilities, target_types)
    oracle_makespan = full_information_makespan(truth, agents)
    rollout = collect_tensor_episodes(
        model, state,
        TensorObservationBuilder(
            world, config.model.num_target_types,
            task_graph=config.model.architecture == "task_graph",
            edge_normalization=config.model.edge_normalization,
            edge_normalization_epsilon=(
                config.model.edge_normalization_epsilon)),
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty,
        training=False,
        state_callback=None if trace is None else _capture_state(trace),
        oracle_makespans=oracle_makespan)
    record = _tensor_rollout_record(rollout, 0)
    return record, world.terrain, world


def _tensor_rollout_record(rollout, index):
    """Detach one scenario result from a tensor rollout batch."""
    return {
        "return": float(rollout.returns[index]),
        "makespan": float(rollout.makespans[index]),
        "oracle_makespan": float(rollout.oracle_makespans[index]),
        "fi_opt_makespan": float(rollout.oracle_makespans[index]),
        "normalized_regret": float(rollout.normalized_regrets[index]),
        "completed": bool(rollout.completed[index]),
        "deaths": int(rollout.deaths[index]),
        "remaining_targets": int(rollout.remaining_targets[index]),
        "stalled": bool(rollout.stalled[index]),
        "all_agents_dead": bool(rollout.all_agents_dead[index]),
    }


def _cuda_batch_plan(cases):
    """Group cases that can share one TensorWorld and tensor-state shape."""
    by_target = {}
    for case_index, case in enumerate(cases):
        by_target.setdefault(case.target.id, []).append((case_index, case))
    plan = []
    for target_id, target_cases in by_target.items():
        by_agent_count = {}
        for indexed_case in target_cases:
            count = indexed_case[1].agent.agent_count
            by_agent_count.setdefault(count, []).append(indexed_case)
        plan.append((target_id, tuple(by_agent_count.values())))
    return tuple(plan)


def _agents_for_case(case):
    from simulation.agent import Agent

    return [
        Agent(case.source_position, capabilities=capabilities)
        for capabilities in case.agent.capabilities
    ]


def _gpu_suite_records(model, config, cases, device, progress=None,
                       map_path=None):
    """Evaluate compatible fixed-suite cases in CUDA tensor batches."""
    from learning.gpu_sim.observation_gpu import TensorObservationBuilder
    from learning.gpu_sim.rollout_gpu import collect_tensor_episodes
    from learning.gpu_sim.state import TensorEpisodeState
    from learning.gpu_sim.world import TensorWorld

    records = [None] * len(cases)
    terrain = None
    batch_id = 0
    for _target_id, agent_batches in _cuda_batch_plan(cases):
        first_case = agent_batches[0][0][1]
        env, truth, _agents = (
            _case_factory(first_case) if map_path is None
            else _case_factory(first_case, map_path=map_path))
        world = TensorWorld.from_networkx(
            env, config.candidates, device=device, terrain=terrain)
        terrain = world.terrain
        builder = TensorObservationBuilder(
            world, config.model.num_target_types,
            task_graph=config.model.architecture == "task_graph",
            edge_normalization=config.model.edge_normalization,
            edge_normalization_epsilon=(
                config.model.edge_normalization_epsilon))
        source = world.node_index[first_case.source_position]
        target_types = torch.tensor([
            truth.nodes[target]["rps_type"] for target in world.targets
        ], dtype=torch.long, device=device)

        for indexed_cases in agent_batches:
            batch_size = len(indexed_cases)
            agent_count = indexed_cases[0][1].agent.agent_count
            capabilities = torch.zeros(
                (batch_size, agent_count,
                 config.model.num_target_types + 1),
                dtype=torch.bool, device=device)
            case_agents = []
            for row, (_case_index, case) in enumerate(indexed_cases):
                agents = _agents_for_case(case)
                case_agents.append(agents)
                for agent_index, agent in enumerate(agents):
                    for capability in agent.capabilities:
                        capabilities[row, agent_index, capability] = True
            oracle_makespans = torch.tensor([
                full_information_makespan(truth, agents) for agents in case_agents
            ], dtype=torch.float32, device=device)
            state = TensorEpisodeState.create(
                world,
                torch.full((batch_size,), source, device=device),
                capabilities,
                target_types[None].expand(batch_size, -1).clone())
            rollout = collect_tensor_episodes(
                model, state, builder,
                config.reinforce.death_penalty,
                config.reinforce.incomplete_penalty,
                training=False,
                oracle_makespans=oracle_makespans)
            for row, (case_index, _case) in enumerate(indexed_cases):
                record = _tensor_rollout_record(rollout, row)
                record.update({
                    "simulation_backend": "cuda_tensor",
                    "evaluation_batch_id": batch_id,
                    "evaluation_batch_size": batch_size,
                })
                records[case_index] = record
            if progress is not None:
                progress.update(batch_size)
            batch_id += 1
            del state, rollout
            torch.cuda.empty_cache()
    return records


def _route_from_snapshot(snapshot, agent_index, node, node_count):
    route = [node]
    successors = snapshot["route_next"][agent_index]
    seen = {node}
    while len(route) <= node_count:
        next_node = int(successors[node])
        if next_node < 0 or next_node in seen:
            break
        route.append(next_node)
        seen.add(next_node)
        node = next_node
    return route


def _render_gpu_trace(trace, env, truth, source_agents, world, frames_dir,
                      render_dt):
    """Render a device rollout after it completes; no CPU policy replay."""
    from simulation.agent import Agent
    from simulation.domain import UNKNOWN_TYPE
    from simulation.rendering import render_frame

    if not trace:
        return 0
    display_agents = [Agent(source_agents[0].position,
                            capabilities=agent.capabilities)
                      for agent in source_agents]
    trajectories = [[int(trace[0]["positions"][i])]
                    for i in range(len(display_agents))]
    snapshot_trajectories = []
    for snapshot in trace:
        for i in range(len(display_agents)):
            node = int(snapshot["positions"][i])
            if trajectories[i][-1] != node:
                trajectories[i].append(node)
        snapshot_trajectories.append([path.copy() for path in trajectories])

    final_time = float(trace[-1]["clock"])
    frame_times = []
    time_value = 0.0
    while time_value < final_time:
        frame_times.append(time_value)
        time_value += render_dt
    frame_times.append(final_time)
    snapshot_index = 0
    targets = list(world.targets)
    for frame_index, tau in enumerate(frame_times):
        while (snapshot_index + 1 < len(trace)
               and float(trace[snapshot_index + 1]["clock"]) <= tau):
            snapshot_index += 1
        snapshot = trace[snapshot_index]
        visible_env = env.copy()
        for target_index, target in enumerate(targets):
            live = bool(snapshot["target_live"][target_index])
            known = bool(snapshot["target_known"][target_index])
            visible_env.nodes[target]["type"] = (
                "target_unreached" if live else "target_reached")
            visible_env.nodes[target]["rps_type"] = (
                int(truth.nodes[target]["rps_type"]) if known else UNKNOWN_TYPE)
        xys = []
        positions = []
        for i, agent in enumerate(display_agents):
            node_index = int(snapshot["positions"][i])
            positions.append(node_index)
            agent.position = world.nodes[node_index]
            agent.alive = bool(snapshot["alive"][i])
            agent.trajectory = [world.nodes[node]
                                for node in snapshot_trajectories[snapshot_index][i]]
            route_start = node_index
            if bool(snapshot["moving"][i]):
                start = int(snapshot["transit_from"][i])
                end = int(snapshot["transit_to"][i])
                arrival = float(snapshot["arrival_time"][i])
                clock = float(snapshot["clock"])
                fraction = 0.0 if arrival <= clock else max(
                    0.0, min(1.0, (tau - clock) / (arrival - clock)))
                p0 = truth.nodes[world.nodes[start]]["pos"]
                p1 = truth.nodes[world.nodes[end]]["pos"]
                xys.append((p0[0] + fraction * (p1[0] - p0[0]),
                            p0[1] + fraction * (p1[1] - p0[1])))
                route_start = end
                prefix = [start, end]
            else:
                xys.append(truth.nodes[world.nodes[node_index]]["pos"])
                prefix = [node_index]
            route = _route_from_snapshot(
                snapshot, i, route_start, len(world.nodes))
            indices = prefix + route[1:] if prefix[-1] == route[0] else prefix
            agent.planned_path = [world.nodes[node] for node in indices]
        render_frame(
            visible_env, truth, display_agents, frame_index,
            str(Path(frames_dir) / f"frame_{frame_index:04d}.png"),
            title=f"t = {tau:.1f}", agent_xy=xys)
    return len(frame_times)


def _cpu_episode(model, config, env, truth, agents, device,
                 render_dir=None, render_dt=1.0):
    from learning.policy.adapter import LearnedPolicyAdapter
    from learning.gpu_sim.rollout_cpu import collect_episode

    adapter = LearnedPolicyAdapter(
        model, config.model.num_target_types, training=False,
        candidate_config=config.candidates, device=device)
    oracle_makespan = full_information_makespan(truth, agents)
    rollout = collect_episode(
        env, truth, agents, adapter,
        config.reinforce.death_penalty,
        config.reinforce.incomplete_penalty,
        render_dir=render_dir, render_dt=render_dt,
        oracle_makespan=oracle_makespan)
    result = rollout.result
    return {
        "return": float(rollout.episode_return),
        "makespan": float(result["makespan"]),
        "oracle_makespan": float(oracle_makespan),
        "fi_opt_makespan": float(oracle_makespan),
        "normalized_regret": float(result["normalized_regret"]),
        "completed": bool(result["completed"]),
        "deaths": int(result["num_deaths"]),
        "remaining_targets": len(result["remaining_targets"]),
        "stalled": bool(not result["completed"] and result["survivors"]),
        "all_agents_dead": bool(not result["survivors"]),
    }


class _PreserveFullInformationRoutes:
    """Simulator policy adapter for routes committed before time zero."""

    def __call__(self, _env_map, _agents, **_kwargs):
        # Agent.move consumes the installed path one edge at a time. Replanning
        # callbacks must not replace an oracle route with a new shortest path.
        return None


def _classical_episode(policy_name, config, env, truth, agents,
                       render_dir=None, render_dt=1.0):
    from planning.full_information import solve_full_information
    from planning.policies.scout_then_execute import ScoutThenExecutePolicy
    from simulation.engine import run_simulation

    oracle_makespan = full_information_makespan(truth, agents)
    diagnostics = {}
    if policy_name == "fi-opt":
        plan = solve_full_information(truth, agents)
        if not plan.feasible:
            raise ValueError(plan.diagnostic.get(
                "message", "FI-OPT evaluation instance is infeasible"))
        for agent, path in zip(agents, plan.paths):
            agent.planned_path = list(path)
        policy = _PreserveFullInformationRoutes()
        diagnostics = {
            "phase": "full_information",
            "predicted_makespan": plan.makespan,
            "assignments": [list(values) for values in plan.assignments],
            "target_orders": [list(values) for values in plan.target_orders],
            "paths": [list(values) for values in plan.paths],
            "per_agent_finish_times": list(plan.finish_times),
            "solver": dict(plan.diagnostic),
        }
    elif policy_name == "scout-then-execute":
        policy = ScoutThenExecutePolicy()
    else:
        raise ValueError(f"unknown classical policy: {policy_name}")

    result = run_simulation(
        env, truth, agents, policy=policy,
        death_penalty=config.reinforce.death_penalty,
        render_dir=render_dir, render_dt=render_dt,
    )
    if policy_name == "scout-then-execute":
        diagnostics = dict(policy.diagnostics)
    normalized_regret = (
        result["makespan"] / oracle_makespan - 1.0
        if oracle_makespan > 0.0 else 0.0
    )
    episode_return = (
        -normalized_regret
        - config.reinforce.death_penalty * result["num_deaths"]
        - config.reinforce.incomplete_penalty * len(result["remaining_targets"])
    )
    predicted = diagnostics.get("predicted_makespan")
    if predicted is not None and math.isfinite(float(predicted)):
        diagnostics["executed_makespan"] = float(result["makespan"])
        diagnostics["prediction_error"] = float(result["makespan"] - predicted)
    return {
        "return": float(episode_return),
        "makespan": float(result["makespan"]),
        "oracle_makespan": float(oracle_makespan),
        "fi_opt_makespan": float(oracle_makespan),
        "normalized_regret": float(normalized_regret),
        "completed": bool(result["completed"]),
        "deaths": int(result["num_deaths"]),
        "remaining_targets": len(result["remaining_targets"]),
        "stalled": bool(not result["completed"] and result["survivors"]),
        "all_agents_dead": bool(not result["survivors"]),
        "policy_diagnostics": diagnostics,
    }


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


def _std(records, key):
    mean = _mean(records, key)
    return (sum((float(record[key]) - mean) ** 2 for record in records)
            / len(records)) ** 0.5


def _attach_case_metadata(record, suite, case):
    record.setdefault("fi_opt_makespan", record["oracle_makespan"])
    record.update({
        "suite_id": suite.suite_id,
        "scenario_id": case.scenario_id,
        "agent_configuration_id": case.agent.id,
        "target_configuration_id": case.target.id,
        "agent_count": case.agent.agent_count,
        "target_count": case.target.target_count,
        "agent_profile": case.agent.profile,
        "target_profile": case.target.profile,
        "instance": case.instance_metadata(),
    })
    return record


def summarize(records):
    """Return aggregate deterministic-policy statistics."""
    batch_ids = {
        record.get("evaluation_batch_id", index)
        for index, record in enumerate(records)
    }
    return {
        "scenario_count": len(records),
        # Retained for readers of evaluation JSON written before fixed suites.
        "episodes": len(records),
        "rollout_batch_count": len(batch_ids),
        "max_rollout_batch_size": max(
            record.get("evaluation_batch_size", 1) for record in records),
        "completion_rate": _mean(records, "completed"),
        "mean_return": _mean(records, "return"),
        "return_std": _std(records, "return"),
        "mean_makespan": _mean(records, "makespan"),
        "makespan_std": _std(records, "makespan"),
        "mean_oracle_makespan": _mean(records, "oracle_makespan"),
        "mean_fi_opt_makespan": _mean(records, "fi_opt_makespan"),
        "mean_normalized_regret": _mean(records, "normalized_regret"),
        "normalized_regret_std": _std(records, "normalized_regret"),
        "mean_deaths": _mean(records, "deaths"),
        "death_std": _std(records, "deaths"),
        "mean_remaining_targets": _mean(records, "remaining_targets"),
        "stalled_rate": _mean(records, "stalled"),
        "all_agents_dead_rate": _mean(records, "all_agents_dead"),
    }


def _episode_output_path(path, episode, episodes):
    """Give each rendered repetition its own video output."""

    path = Path(path)
    if episodes == 1:
        return path
    return path.with_name(f"{path.stem}_episode_{episode:04d}{path.suffix}")


def evaluate(checkpoint=None, config_path=None, suite="development", seed=None,
             device=None, agent_config=None, target_config=None,
             agent_count=None, target_count=None, limit=None,
             render=False, render_dir=None, output_mp4=None, mp4_fps=4,
             render_dt=1.0, policy="learned", episodes=1, progress=False,
             map_path=None, allow_map_mismatch=False,
             allow_feature_schema_mismatch=False):
    if policy not in {"learned", "fi-opt", "scout-then-execute"}:
        raise ValueError(f"unknown evaluation policy: {policy}")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    suite_definition, suite_path = load_evaluation_suite(suite)
    selected_cases = select_evaluation_cases(
        suite_definition,
        agent_config=agent_config,
        target_config=target_config,
        agent_count=agent_count,
        target_count=target_count,
        limit=limit)
    evaluation_cases = tuple(
        case for _episode_index in range(episodes) for case in selected_cases
    )
    if render and len(selected_cases) != 1:
        raise ValueError(
            "--render requires exactly one selected evaluation scenario")

    if policy == "learned" and checkpoint is None:
        raise ValueError("the learned policy requires a checkpoint")
    weights_path = None
    checkpoint_config = None
    if checkpoint is not None:
        weights_path, checkpoint_config = _resolve_checkpoint(checkpoint)
    explicit_config = load_config(config_path) if config_path is not None else None
    saved_config = (
        load_config(checkpoint_config) if checkpoint_config is not None else None)
    config = explicit_config or saved_config or load_config(DEFAULT_CONFIG_PATH)
    selected_feature_schema = feature_schema_metadata(config.model)
    checkpoint_feature_schema = (
        feature_schema_metadata(saved_config.model)
        if saved_config is not None else {
            "version": RAW_DISTANCE_FEATURE_SCHEMA_VERSION,
            "edge_normalization": "none",
            "edge_normalization_epsilon": None,
        })
    feature_schema_issue = bool(
        policy == "learned" and (
            checkpoint_feature_schema["version"]
            != CURRENT_FEATURE_SCHEMA_VERSION
            or checkpoint_feature_schema != selected_feature_schema))
    if feature_schema_issue and not allow_feature_schema_mismatch:
        raise ValueError(
            "checkpoint feature schema is incompatible with the selected "
            "neural input schema: checkpoint "
            f"{checkpoint_feature_schema!r}, selected "
            f"{selected_feature_schema!r}. Retrain the policy or pass "
            "--allow-feature-schema-mismatch for an explicit compatibility "
            "override.")
    feature_schema = dict(selected_feature_schema)
    feature_schema.update({
        "checkpoint": checkpoint_feature_schema if policy == "learned" else None,
        "allow_mismatch": bool(allow_feature_schema_mismatch),
        "compatibility_override_used": bool(
            feature_schema_issue and allow_feature_schema_mismatch),
    })
    configured_map_path = _select_prepared_map_path(
        map_path, explicit_config, saved_config)
    (resolved_map_path, terrain_template, _map_metadata,
     prepared_map) = inspect_prepared_map(configured_map_path)
    validate_evaluation_suite_locations(suite_definition, terrain_template)
    checkpoint_map = (
        _checkpoint_prepared_map(checkpoint_config)
        if policy == "learned" else None)
    checkpoint_sha256 = (
        None if checkpoint_map is None else checkpoint_map["sha256"].lower())
    hash_matches = (
        None if checkpoint_sha256 is None
        else prepared_map["sha256"].lower() == checkpoint_sha256)
    mismatch_override_used = bool(
        policy == "learned" and hash_matches is False and allow_map_mismatch)
    if (policy == "learned" and hash_matches is False
            and not allow_map_mismatch):
        raise ValueError(
            "selected prepared map does not match the checkpoint: "
            f"checkpoint SHA-256 {checkpoint_sha256}, selected SHA-256 "
            f"{prepared_map['sha256']}. Pass --allow-map-mismatch to "
            "evaluate deliberately on different terrain.")
    prepared_map.update({
        "checkpoint_sha256": checkpoint_sha256,
        "hash_matches_checkpoint": hash_matches,
        "allow_map_mismatch": bool(allow_map_mismatch),
        "mismatch_override_used": mismatch_override_used,
    })
    config = replace(
        config,
        instances=replace(config.instances, map_path=str(resolved_map_path)))
    if (policy == "learned"
            and config.model.num_target_types != suite_definition.num_target_types):
        raise ValueError(
            "checkpoint model num_target_types does not match evaluation suite")
    if seed is None:
        seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = (device or config.training.device).lower()
    if policy == "learned":
        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if requested_device == "auto" else requested_device
    else:
        resolved_device = "cpu"
    if (policy == "learned" and resolved_device == "cuda"
            and not torch.cuda.is_available()):
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    model = None
    if policy == "learned":
        model = build_policy(config.model).to(resolved_device)
        state_dict = torch.load(weights_path, map_location=resolved_device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

    records = []
    terrain = None
    if render:
        from simulation.rendering import clear_frame_dir
        render_dir = Path(
            render_dir or DEFAULT_RENDER_DIR).resolve()
        output_mp4 = Path(
            output_mp4 or DEFAULT_OUTPUT_MP4).resolve()
    progress_bar = tqdm(
        total=len(evaluation_cases), desc="Evaluating", unit="scenario",
        disable=not progress, file=sys.stderr)
    with torch.no_grad(), progress_bar:
        if policy == "learned" and resolved_device == "cuda" and not render:
            records = _gpu_suite_records(
                model, config, evaluation_cases, resolved_device,
                progress=progress_bar, map_path=resolved_map_path)
            records = [
                _attach_case_metadata(record, suite_definition, case)
                for record, case in zip(records, evaluation_cases)
            ]
        for case_index, case in enumerate(
                () if records else evaluation_cases):
            episode_index = case_index // len(selected_cases)
            episode_seed = seed + episode_index
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env, truth, agents = _case_factory(
                case, map_path=resolved_map_path)
            if render:
                from simulation.rendering import make_mp4_from_frames
                episode_frames = (
                    render_dir if episodes == 1
                    else render_dir / f"episode_{episode_index:04d}"
                )
                episode_video = _episode_output_path(
                    output_mp4, episode_index, episodes
                )
                clear_frame_dir(episode_frames)
                if policy == "learned" and resolved_device == "cuda":
                    trace = []
                    record, terrain, world = _gpu_episode(
                        model, config, env, truth, agents, resolved_device,
                        terrain=terrain, trace=trace)
                    _render_gpu_trace(
                        trace, env, truth, agents, world, episode_frames,
                        render_dt)
                    record["simulation_backend"] = "cuda_tensor"
                elif policy == "learned":
                    record = _cpu_episode(
                        model, config, env, truth, agents, resolved_device,
                        render_dir=episode_frames, render_dt=render_dt)
                    record["simulation_backend"] = "cpu_render"
                else:
                    record = _classical_episode(
                        policy, config, env, truth, agents,
                        render_dir=episode_frames, render_dt=render_dt)
                    record["simulation_backend"] = "cpu_classical_render"
                try:
                    make_mp4_from_frames(
                        episode_frames, episode_video, fps=mp4_fps)
                    record["video"] = str(episode_video)
                except (FileNotFoundError, subprocess.CalledProcessError) as error:
                    print(f"warning: MP4 creation failed: {error}", file=sys.stderr)
                record["frames"] = str(episode_frames)
            elif policy == "learned" and resolved_device == "cuda":
                record, terrain, _world = _gpu_episode(
                    model, config, env, truth, agents, resolved_device,
                    terrain=terrain)
                record["simulation_backend"] = "cuda_tensor"
            elif policy == "learned":
                record = _cpu_episode(
                    model, config, env, truth, agents, resolved_device)
                record["simulation_backend"] = "cpu_simulator"
            else:
                record = _classical_episode(
                    policy, config, env, truth, agents)
                record["simulation_backend"] = "cpu_classical"
            record.update({
                "evaluation_batch_id": case_index,
                "evaluation_batch_size": 1,
                "episode_index": episode_index,
                "seed": episode_seed,
                "policy": policy,
            })
            records.append(_attach_case_metadata(
                record, suite_definition, case))
            progress_bar.update(1)
    if records and policy == "learned" and resolved_device == "cuda" and not render:
        for case_index, record in enumerate(records):
            episode_index = case_index // len(selected_cases)
            record.update({
                "episode_index": episode_index,
                "seed": seed + episode_index,
                "policy": policy,
            })
    evaluate.last_prepared_map = prepared_map
    evaluate.last_feature_schema = feature_schema
    return (records, summarize(records), config, weights_path,
            suite_definition, suite_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate learned or classical policies on a prepared map and "
            "fixed suite"))
    parser.add_argument(
        "checkpoint", nargs="?",
        help="timestamped checkpoint directory or weights file (learned policy only)")
    parser.add_argument(
        "--policy", default="learned",
        choices=["learned", "fi-opt", "scout-then-execute"],
        help="policy to evaluate; learned is the default")
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="repeat every selected fixed-suite case N times")
    parser.add_argument("--config", help="config override; defaults to checkpoint config")
    parser.add_argument(
        "--map-path",
        help="prepared-map artifact; overrides config and checkpoint map paths")
    parser.add_argument(
        "--allow-map-mismatch", action="store_true",
        help="allow learned evaluation when selected and checkpoint map hashes differ")
    parser.add_argument(
        "--allow-feature-schema-mismatch", action="store_true",
        help=("allow an older or otherwise incompatible checkpoint feature "
              "schema; retraining is recommended"))
    parser.add_argument(
        "--suite", default="development",
        help="development, test, or a factorized suite JSON path")
    parser.add_argument(
        "--agent-config", action="append",
        help="select an agent configuration ID; may be repeated")
    parser.add_argument(
        "--target-config", action="append",
        help="select a target configuration ID; may be repeated")
    parser.add_argument("--agent-count", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument(
        "--limit", type=int,
        help="evaluate only the first N cases after applying other filters")
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable the evaluation progress bar")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"],
        help="learned-policy device; classical benchmarks currently execute on CPU")
    parser.add_argument(
        "--cuda", action="store_true",
        help="shortcut for --device cuda (learned only; classical policies remain CPU)")
    parser.add_argument(
        "--output",
        help=("JSON result path; defaults to a timestamped file under "
              "outputs/evaluation"))
    parser.add_argument("--render", action="store_true",
                        help="write one PNG per turn and compile an MP4")
    parser.add_argument("--render-dir",
                        default=str(DEFAULT_RENDER_DIR))
    parser.add_argument("--output-mp4",
                        default=str(DEFAULT_OUTPUT_MP4))
    parser.add_argument("--mp4-fps", type=int, default=4)
    parser.add_argument("--render-dt", type=float, default=1.0,
                        help="sim-time between rendered frames (continuous time)")
    args = parser.parse_args()
    if args.mp4_fps < 1:
        parser.error("--mp4-fps must be positive")
    if args.render_dt <= 0:
        parser.error("--render-dt must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.agent_count is not None and args.agent_count < 1:
        parser.error("--agent-count must be positive")
    if args.target_count is not None and args.target_count < 1:
        parser.error("--target-count must be positive")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.cuda and args.device is not None:
        parser.error("use either --cuda or --device, not both")
    selected_device = "cuda" if args.cuda else args.device

    try:
        (records, summary, config, weights,
         suite_definition, suite_path) = evaluate(
            args.checkpoint, config_path=args.config, suite=args.suite,
            seed=args.seed, device=selected_device,
            agent_config=args.agent_config,
            target_config=args.target_config,
            agent_count=args.agent_count,
            target_count=args.target_count,
            limit=args.limit,
            render=args.render, render_dir=args.render_dir,
            output_mp4=args.output_mp4, mp4_fps=args.mp4_fps,
            render_dt=args.render_dt, policy=args.policy,
            episodes=args.episodes, progress=not args.no_progress,
            map_path=args.map_path,
            allow_map_mismatch=args.allow_map_mismatch,
            allow_feature_schema_mismatch=(
                args.allow_feature_schema_mismatch))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    prepared_map = evaluate.last_prepared_map
    feature_schema = evaluate.last_feature_schema
    output = _evaluation_output_path(args.output)
    payload = {
        "schema_version": 1,
        "checkpoint": str(weights) if weights is not None else None,
        "policy": args.policy,
        "deterministic": True,
        "suite_id": suite_definition.suite_id,
        "suite_path": str(suite_path),
        "terrain_id": suite_definition.terrain_id,
        "output_path": str(output),
        "prepared_map": prepared_map,
        "allow_map_mismatch": args.allow_map_mismatch,
        "feature_schema": feature_schema,
        "allow_unknown_target_actions": (
            config.candidates.allow_unknown_target_actions),
        "scenario_ids": [record["scenario_id"] for record in records],
        "selection": {
            "agent_config": args.agent_config,
            "target_config": args.target_config,
            "agent_count": args.agent_count,
            "target_count": args.target_count,
            "limit": args.limit,
            "episodes": args.episodes,
            "seed": args.seed if args.seed is not None else config.training.seed,
        },
        "summary": summary,
        "scenarios": records,
    }
    rendered = json.dumps(payload, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"saved evaluation: {output}")


if __name__ == "__main__":
    main()
