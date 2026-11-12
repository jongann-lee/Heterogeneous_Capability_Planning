"""Render an HGT decision beside four tables of top action candidates.

    uv run python figure_scripts/05_render_decision.py

Defaults: 1103/best_weights.pt, WV, agents_04_b, targets_08_a.
Run the real deterministic CUDA evaluator, then replay to an actual replan with
both known and unknown live tasks. Prefer four revealed tasks, then the time
nearest 25% of the completed run. --time instead selects the nearest replan to
an explicit simulation time. Task IDs T1..T8 follow suite file order.

The map occupies the left half; a 2x2 table layout occupies the right half.
Each table shows one agent's top four physical actions, logits, full-action
softmax probabilities, and joint-assignment choices. Borders match agent colors.
PDF, editable 600 px-wide SVG, and the printed analysis as TXT are written
beside one another under figure_scripts/outputs, replacing them on reruns.
The simulator and policy use their existing evaluation machinery unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from textwrap import fill
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.export_utils import save_pdf_and_svg
from learning.evaluation_suite import (
    load_evaluation_suite, select_evaluation_cases,
    validate_evaluation_suite_locations,
)
from learning.gpu_sim.instances import inspect_prepared_map
from learning.gpu_sim.observation_gpu import TensorObservationBuilder
from learning.gpu_sim.rollout_gpu import collect_tensor_episodes
from learning.gpu_sim.state import TensorEpisodeState
from learning.gpu_sim.world import TensorWorld
from learning.modules.decoder import AssignmentDecoder
from learning.policy.adapter import LearnedPolicyAdapter
from learning.policy.configuration import CURRENT_FEATURE_SCHEMA_VERSION, load_config
from learning.policy.model import build_policy
from learning.test import (
    _case_factory, _checkpoint_prepared_map, _encode_episode,
    _resolve_checkpoint, _route_from_snapshot,
)
from simulation.domain import UNKNOWN_TYPE
from simulation.rendering import DEAD_AGENT_COLOR, _agent_color, draw_frame


DEFAULT_CHECKPOINT = ROOT / "learning/checkpoints/2026-11-03_11-34-21_798508"
DEFAULT_MAP = ROOT / "Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz"
DEFAULT_SUITE = ROOT / "learning/evaluation_suites/wv_factorial_test_v1.json"
TOP_K = 4
FIGSIZE = (13.6, 6.8)


@dataclass(frozen=True)
class Decision:
    index: int
    clock: float
    revealed: int
    known_live: int
    unknown_live: int
    completed: int
    alive: int


@dataclass
class Snapshot:
    decision: Decision
    graph: object
    agents: list
    transit: list
    observation: object
    logits: torch.Tensor
    decoded: object
    candidate_destinations: list | None = None


class SnapshotCaptured(Exception):
    """Stop replay immediately after scoring the selected real decision."""


class DecisionBuilder(TensorObservationBuilder):
    """Observe the standard CUDA evaluator without changing scores or routes."""

    def __init__(self, *args, capture_index=None, graph=None, agents=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeline = []
        self.capture_index = capture_index
        self.snapshot = None
        self.graph = graph
        self.agents = agents
        self.trajectories = [[agent.position] for agent in agents]
        self.captured = {}

    def track_positions(self, state):
        for index, position in enumerate(state.positions[0].tolist()):
            node = self.world.nodes[position]
            if self.trajectories[index][-1] != node:
                self.trajectories[index].append(node)

    def build(self, state, planning_episode_mask=None):
        known = state.target_known[0].cpu()
        live = state.target_live[0].cpu()
        decision = Decision(
            len(self.timeline), float(state.clock[0]), int(known.sum()),
            int((known & live).sum()), int((~known & live).sum()),
            int((~live).sum()), int(state.alive[0].sum()),
        )
        self.timeline.append(decision)
        result = super().build(state, planning_episode_mask)
        if decision.index == self.capture_index:
            self.captured["candidate_entries"] = result[-1][0].cpu()
        return result

    def record_decode(self, model, original_decode, observation, *args, **kwargs):
        if self.timeline[-1].index != self.capture_index:
            return original_decode(observation, *args, **kwargs)
        original_score = model.actor_critic

        def capture_scores(obs):
            logits, values = original_score(obs)
            self.captured["logits"] = logits.detach().cpu().clone()
            return logits, values

        with patch.object(model, "actor_critic", capture_scores):
            decoded = original_decode(observation, *args, **kwargs)
        self.captured["observation"] = observation.to("cpu")
        self.captured["decoded"] = decoded
        return decoded

    def next_hops(self, state, *args):
        result = super().next_hops(state, *args)
        if self.timeline[-1].index != self.capture_index:
            return result
        self.track_positions(state)
        world = self.world
        graph = self.graph.copy()
        known = state.target_known[0].tolist()
        live = state.target_live[0].tolist()
        for index, node in enumerate(world.targets):
            graph.nodes[node]["type"] = "target_unreached" if live[index] else "target_reached"
            graph.nodes[node]["rps_type"] = int(state.target_types[0, index]) \
                if known[index] else UNKNOWN_TYPE
        agents = deepcopy(self.agents)
        transit = []
        routes = {"route_next": result[2][0].cpu()}
        destinations = []
        for index, agent in enumerate(agents):
            agent.position = world.nodes[int(state.positions[0, index])]
            agent.alive = bool(state.alive[0, index])
            agent.trajectory = list(self.trajectories[index])
            source = agent.position
            travel = None
            if bool(state.moving[0, index]):
                start = world.nodes[int(state.transit_from[0, index])]
                source = world.nodes[int(state.transit_to[0, index])]
                arrival = float(state.arrival_time[0, index])
                travel = (start, source, arrival - graph.edges[start, source]["distance"], arrival)
            transit.append(travel)
            route = _route_from_snapshot(routes, index, world.node_index[source], len(world.nodes))
            agent.planned_path = [world.nodes[node] for node in route]
            destinations.append([world.nodes[int(node)] for node in
                                 self.captured["candidate_entries"][index]])
        obs = self.captured["observation"]
        obs.candidates = [deepcopy(obs.candidates[0])]
        # GPU slots are stable across revelations. Describe their CURRENT
        # revealed/staged target sets, rather than their initial signatures.
        for index, candidate in enumerate(obs.candidates[0]):
            candidate.observed_targets = {node for target, node in enumerate(world.targets)
                                         if bool(obs.action_target_relations[0, index, target, 1])}
            candidate.staging_targets = {node for target, node in enumerate(world.targets)
                                        if bool(obs.action_target_relations[0, index, target, 2])}
        self.snapshot = Snapshot(self.timeline[-1], graph, agents, transit, obs,
                                 self.captured["logits"], self.captured["decoded"], destinations)
        raise SnapshotCaptured


def select_decision(timeline, makespan, progress=0.25, revealed=4, time=None):
    if time is not None:
        return min(timeline, key=lambda item: (abs(item.clock - time), item.index))
    mixed = [item for item in timeline if item.known_live and item.unknown_live
             and item.alive == timeline[0].alive and item.clock > 0]
    if not mixed:
        raise ValueError("run has no decision with both known and unknown live tasks")
    return min(mixed, key=lambda item: (
        abs(item.revealed - revealed), abs(item.clock / makespan - progress), item.index))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_episode(model, config, case, map_path, seed, progress, revealed, time):
    env, truth, agents = _case_factory(case, map_path=map_path)
    world = TensorWorld.from_networkx(env, config.candidates, device="cuda")
    source, capabilities, target_types = _encode_episode(
        world, truth, agents, case.num_target_types, "cuda")

    def new_episode(capture_index=None):
        seed_everything(seed)
        state = TensorEpisodeState.create(world, [source], capabilities, target_types)
        builder = DecisionBuilder(
            world, case.num_target_types, capture_index=capture_index, graph=env, agents=agents,
            use_agent_alive=config.model.use_agent_alive,
            use_task_completed=config.model.use_task_completed,
            edge_normalization=config.model.edge_normalization,
            edge_normalization_epsilon=config.model.edge_normalization_epsilon,
        )
        return state, builder

    state, builder = new_episode()
    print("Running deterministic episode to locate a decision...", file=sys.stderr, flush=True)
    result = collect_tensor_episodes(model, state, builder, training=False,
                                     state_callback=builder.track_positions)
    makespan = float(result.makespans[0])
    if not bool(result.completed[0]) or makespan <= 0:
        raise ValueError("episode did not complete; cannot select a fraction of its makespan")
    if time is not None and time > makespan:
        raise ValueError(f"--time exceeds episode makespan {makespan:.3f}")
    decision = select_decision(builder.timeline, makespan, progress, revealed, time)
    print(f"Replaying to decision {decision.index} at t={decision.clock:.3f} "
          f"({decision.clock / makespan:.1%}); "
          f"{decision.revealed} revealed, {decision.unknown_live} unknown...",
          file=sys.stderr, flush=True)
    state, builder = new_episode(decision.index)
    original_decode = model.decode

    def capture_decode(observation, *args, **kwargs):
        return builder.record_decode(model, original_decode, observation, *args, **kwargs)

    try:
        with patch.object(model, "decode", capture_decode):
            collect_tensor_episodes(model, state, builder, training=False,
                                     state_callback=builder.track_positions)
    except SnapshotCaptured:
        pass
    if builder.snapshot is None or builder.snapshot.decision != decision:
        raise RuntimeError("replay did not reproduce the selected decision")
    return builder.snapshot, makespan


def candidate_label(candidate, target_ids, graph):
    def task_names(nodes):
        return ",".join(sorted((target_ids[node] for node in nodes),
                               key=lambda value: int(value[1:])))

    if candidate.is_wait:
        return "Wait"
    if candidate.is_target:
        known = graph.nodes[candidate.node].get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE
        return ("Execute " if known else "Visit ") + target_ids[candidate.node]
    if candidate.is_observation:
        return "Observe " + task_names(candidate.observed_targets)
    if candidate.is_staging:
        return "Stage " + task_names(candidate.staging_targets)
    return f"Move to {candidate.node}"


def rank_actions(snapshot, target_ids, top_k=TOP_K):
    """Rank the same physical actions used by the assignment decoder."""
    obs = snapshot.observation
    groups = obs.candidate_physical_group
    group_logits = AssignmentDecoder.grouped_logits(
        snapshot.logits, obs.feasible_action_mask, groups,
        obs.physical_group_mask,
    )[0].cpu()
    group_logits[:, obs.physical_group_capacity[0].cpu() <= 0] = -torch.inf
    selected = {agent: int(groups[0, action])
                for agent, action in snapshot.decoded.assignments[0]}
    candidates = obs.candidates[0]
    live = {node for node, data in snapshot.graph.nodes(data=True)
            if data.get("type") == "target_unreached"}
    results = []
    for agent_index, agent in enumerate(snapshot.agents):
        scores = group_logits[agent_index]
        finite = torch.isfinite(scores)
        valid_groups = finite.nonzero().flatten().tolist()
        probabilities = torch.softmax(scores, dim=0) if valid_groups else torch.zeros_like(scores)
        ordered = sorted(valid_groups, key=lambda group: (-float(scores[group]), group))

        def describe(group):
            indices = [index for index in range(len(candidates))
                       if int(groups[0, index]) == group
                       and bool(obs.feasible_action_mask[0, agent_index, index])]
            labels = list(dict.fromkeys(candidate_label(candidates[index], target_ids,
                                                        snapshot.graph) for index in indices))
            representative = int(obs.physical_group_representative[0, group])
            candidate = candidates[representative]
            travel = snapshot.transit[agent_index]
            source = travel[1] if travel is not None else agent.position
            if candidate.is_wait:
                destination = source
            elif candidate.is_target:
                destination = candidate.node
            elif snapshot.candidate_destinations is not None:
                destination = snapshot.candidate_destinations[agent_index][representative]
            else:
                destination = LearnedPolicyAdapter._route(snapshot.graph, source, candidate, live)[-1]
            return {
                "group": group, "action": " / ".join(labels), "destination": destination,
                "logit": float(scores[group]), "probability": float(probabilities[group]),
                "selected": selected.get(agent_index) == group,
            }

        top = [describe(group) for group in ordered[:top_k]]
        chosen = selected.get(agent_index)
        chosen_row = None if chosen is None else next(
            (row for row in top if row["group"] == chosen), None)
        if chosen is not None and chosen_row is None:
            chosen_row = describe(chosen)
        results.append((top, chosen_row))
    return results


def format_analysis(snapshot, case, checkpoint, makespan, rankings):
    decision = snapshot.decision
    lines = [
        "CAAT Graph + HGT decision: West Virginia (WV)", f"Checkpoint: {checkpoint}",
        f"Scenario: {case.scenario_id}",
        "Backend: deterministic CUDA evaluator; logits captured before assignment.",
        f"Decision {decision.index}: t={decision.clock:.3f}; "
        f"{decision.clock / makespan:.1%} of episode makespan {makespan:.3f}",
        f"Tasks: {decision.revealed}/{case.target.target_count} revealed, "
        f"{decision.known_live} known live, {decision.unknown_live} unknown, "
        f"{decision.completed} completed.",
        "Task IDs follow suite file order; '?' never exposes a hidden type.",
        "", "Task | Node (row, col) | Known type | State",
    ]
    for index, target in enumerate(case.target.targets, 1):
        data = snapshot.graph.nodes[target.position]
        known_type = data.get("rps_type", UNKNOWN_TYPE)
        label = "?" if known_type == UNKNOWN_TYPE else str(known_type)
        state = "completed" if data["type"] == "target_reached" else "live"
        lines.append(f"T{index} | {target.position} | {label} | {state}")
    lines.extend([
        "", "Rankings use feasible physical actions BEFORE joint capacity conflicts.",
        "Logit = logsumexp of feasible semantic logits sharing a physical action.",
        "P(agent) = softmax across that agent's feasible physical actions, not a joint probability.",
        "The constrained joint decoder's selected action is marked '*'.",
        "Solid map lines show travel history; dotted lines show selected future routes.",
    ])
    for agent_index, (agent, (top, chosen)) in enumerate(zip(snapshot.agents, rankings)):
        capabilities = ",".join(map(str, sorted(agent.capabilities)))
        travel = snapshot.transit[agent_index]
        location = f"at {agent.position}"
        if travel is not None:
            location = (f"moving {travel[0]} -> {travel[1]}, "
                        f"arrival in {travel[3] - decision.clock:.3f}")
        lines.extend(["", f"A{agent_index}: {{{capabilities}}}; {location}",
                      "Rank | Action | Destination (row, col) | Logit | P(agent) | Selected"])
        for rank, row in enumerate(top, 1):
            lines.append(f"{rank} | {row['action']} | {row['destination']} | "
                         f"{row['logit']:.6f} | {row['probability']:.2%} | "
                         + ("*" if row["selected"] else ""))
        if not top:
            lines.append("No feasible actions.")
        elif len(top) < TOP_K:
            lines.append(f"Only {len(top)} feasible physical actions.")
        if chosen is not None:
            lines.append(f"Joint choice: {chosen['action']} -> {chosen['destination']} "
                         f"(logit {chosen['logit']:.6f}, P(agent) {chosen['probability']:.2%})")
    return "\n".join(lines)


def agent_positions(snapshot):
    positions = []
    for agent, travel in zip(snapshot.agents, snapshot.transit):
        if travel is None:
            positions.append(snapshot.graph.nodes[agent.position]["pos"])
            continue
        start, end, departure, arrival = travel
        fraction = 0 if arrival <= departure else np.clip(
            (snapshot.decision.clock - departure) / (arrival - departure), 0, 1)
        a, b = (snapshot.graph.nodes[node]["pos"] for node in (start, end))
        positions.append(tuple(x + fraction * (y - x) for x, y in zip(a, b)))
    return positions


def draw_action_table(ax, agent_index, agent, ranking):
    """Draw a table from the exact physical-action scores used by the decoder."""
    top, _ = ranking
    color = _agent_color(agent_index)
    ax.set_axis_off()
    border = FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes, facecolor="white", edgecolor=color,
        linewidth=2.2, clip_on=False, zorder=0,
    )
    border.set_gid(f"agent-{agent_index}-table-border")
    ax.add_patch(border)
    ax.add_patch(Rectangle((0, 0.83), 1, 0.17, transform=ax.transAxes,
                           facecolor=to_rgba(color, 0.08), edgecolor="none", zorder=1))
    marker, = ax.plot(0.055, 0.915, marker="o", linestyle="none", markersize=9,
                      markerfacecolor=color, markeredgecolor="black", markeredgewidth=1,
                      transform=ax.transAxes, zorder=3)
    marker.set_gid(f"agent-{agent_index}-table-marker")
    capabilities = ",".join(map(str, sorted(agent.capabilities)))
    ax.text(0.105, 0.915, f"A{agent_index}  ·  {{{capabilities}}}",
            transform=ax.transAxes, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=color, zorder=3)
    rows = []
    for rank, row in enumerate(top, 1):
        action = fill(f"{rank}. {row['action']}", width=20, break_long_words=False,
                      break_on_hyphens=False)
        rows.append([f"{action}\n{row['destination']}", f"{row['logit']:.3f}",
                     f"{row['probability']:.2%}", "Yes" if row["selected"] else "No"])
    if not rows:
        ax.text(0.5, 0.45, "No feasible actions", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        return
    table = ax.table(
        cellText=rows, colLabels=["Candidate", "Logit", "Softmax", "Chosen"],
        colWidths=[0.40, 0.18, 0.23, 0.19], cellLoc="center",
        bbox=(0.035, 0.10, 0.93, 0.71),
    )
    table.set_gid(f"agent-{agent_index}-action-table")
    table.set_zorder(2)
    table.auto_set_font_size(False)
    max_lines = max(row[0].count("\n") + 1 for row in rows)
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    font_size = min(9.5, (height_pt * 0.71 / (len(rows) + 1) - 2) / (1.15 * max_lines))
    table.set_fontsize(font_size)
    for (row_index, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DAE0")
        cell.set_linewidth(0.55)
        cell.PAD = 0.06
        if row_index == 0:
            cell.set_facecolor("#F2F4F6")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(min(font_size, 9.0))
        else:
            selected = top[row_index - 1]["selected"]
            cell.set_facecolor(to_rgba(color, 0.11) if selected else "white")
            if selected:
                cell.get_text().set_fontweight("bold")
            if column == 0:
                cell.get_text().set_ha("left")
            if column == 3:
                cell.get_text().set_color(color if selected else "#66717C")


def render_snapshot(snapshot, case, makespan, output, description, rankings):
    if len(snapshot.agents) != 4:
        raise ValueError("the 2x2 decision-table layout requires four agents")
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    try:
        ax = fig.add_axes((0.035, 0.16, 0.445, 0.74))
        image = draw_frame(
            ax, snapshot.graph, snapshot.graph, snapshot.agents,
            agent_xy=agent_positions(snapshot), scale=0.9, labels_inside=True,
        )
        target_ids = {target.position: f"T{index}" for index, target in enumerate(case.target.targets, 1)}
        for label in ax.texts:
            text = label.get_text()
            label.set_fontsize(10)
            if text.startswith("A"):
                label.set_text(text.split(":", 1)[0])
            elif text.startswith("T:"):
                label.set_text(f"{target_ids[label.xy]}: {text.split(':', 1)[1].strip()}")
        for artist in ax.patches:
            artist.set_rasterized(True)
        handles = [
            Line2D([], [], marker="X", linestyle="none", markersize=10,
                   color="red", markeredgecolor="black", label="Live tasks"),
            Line2D([], [], marker="X", linestyle="none", markersize=10,
                   color=DEAD_AGENT_COLOR, markeredgecolor="black", label="Completed"),
            Line2D([], [], marker="*", linestyle="none", markersize=12,
                   color="gold", markeredgecolor="black", label="Source"),
            Line2D([], [], color="#404040", linewidth=1.6, label="Roads"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.255, 0.12),
                   ncols=4, fontsize=11, frameon=False, handlelength=1, handletextpad=0.5,
                   borderaxespad=0, columnspacing=1.1)
        fig.text(0.255, 0.055, "Task label = ID: type;  ? = unknown", fontsize=11,
                 ha="center", va="center")
        colorbar = fig.colorbar(image, cax=fig.add_axes((0.025, 0.23, 0.010, 0.58)))
        colorbar.set_label("Height", fontsize=11)
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.set_ticks([image.norm.vmin, image.norm.vmax], labels=["Low", "High"])
        colorbar.ax.tick_params(labelsize=10, length=0)
        fig.text(0.255, 0.985, "West Virginia (WV): CAAT Graph + HGT", ha="center",
                 va="top", fontsize=16)
        decision = snapshot.decision
        fig.text(0.255, 0.945,
                 f"t = {decision.clock:.1f} ({decision.clock / makespan:.0%} of run)  ·  "
                 f"{decision.revealed}/{case.target.target_count} revealed",
                 ha="center", va="top", fontsize=11)
        fig.text(0.75, 0.985, f"Top {TOP_K} action candidates per agent", ha="center",
                 va="top", fontsize=16)
        for index, (agent, ranking) in enumerate(zip(snapshot.agents, rankings)):
            column, row = index % 2, index // 2
            table_ax = fig.add_axes((0.515 + column * 0.24, 0.54 - row * 0.42, 0.225, 0.36))
            draw_action_table(table_ax, index, agent, ranking)
        output.parent.mkdir(parents=True, exist_ok=True)
        return save_pdf_and_svg(fig, output, metadata={
            "Title": "WV CAAT Graph + HGT decision", "Subject": description,
            "Creator": "figure_scripts/05_render_decision.py",
        })
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--map-path", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--agent-config", default="agents_04_b")
    parser.add_argument("--target-config", default="targets_08_a")
    parser.add_argument("--progress", type=float, default=0.25)
    parser.add_argument("--revealed", type=int, default=4)
    parser.add_argument("--time", type=float, help="nearest actual decision to this simulation time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/05_WV_decision.pdf")
    args = parser.parse_args(argv)
    if not math.isfinite(args.progress) or not 0 < args.progress < 1:
        parser.error("--progress must be between zero and one")
    if args.revealed < 1:
        parser.error("--revealed must be positive")
    if args.time is not None and (not math.isfinite(args.time) or args.time < 0):
        parser.error("--time must be finite and nonnegative")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension (SVG and TXT are also written)")
    try:
        weights, config_path = _resolve_checkpoint(args.checkpoint)
        if config_path is None:
            raise ValueError("checkpoint must have its saved config.yaml")
        config = load_config(config_path)
        if config.model.architecture != "task_graph":
            raise ValueError("this figure requires an HGT task_graph checkpoint")
        if config.model.feature_schema_version != CURRENT_FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema does not match the current observation builder")
        map_path, terrain, _, identity = inspect_prepared_map(args.map_path)
        checkpoint_map = _checkpoint_prepared_map(config_path)
        if checkpoint_map is None or checkpoint_map["sha256"] != identity["sha256"]:
            raise ValueError("prepared map does not match the checkpoint's recorded terrain")
        suite, _ = load_evaluation_suite(args.suite)
        validate_evaluation_suite_locations(suite, terrain)
        case, = select_evaluation_cases(suite, agent_config=[args.agent_config],
                                       target_config=[args.target_config])
        if case.agent.agent_count != 4:
            raise ValueError("the 2x2 decision-table layout requires a four-agent configuration")
        if config.model.num_target_types != case.num_target_types:
            raise ValueError("checkpoint target-type count does not match the suite")
        if args.revealed > case.target.target_count:
            raise ValueError("--revealed exceeds the scenario's task count")
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise ValueError("this figure uses the CUDA evaluation/routing backend; CUDA is unavailable")
        model = build_policy(config.model).cuda()
        model.load_state_dict(torch.load(weights, map_location="cuda", weights_only=True))
        model.eval()
        with torch.no_grad():
            snapshot, makespan = capture_episode(
                model, config, case, map_path, args.seed, args.progress, args.revealed, args.time)
            target_ids = {target.position: f"T{index}"
                          for index, target in enumerate(case.target.targets, 1)}
            rankings = rank_actions(snapshot, target_ids)
        analysis = format_analysis(snapshot, case, weights, makespan, rankings)
        output = args.output.expanduser().resolve()
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                             "savefig.bbox": None}):
            svg = render_snapshot(snapshot, case, makespan, output, analysis, rankings)
        report = output.with_suffix(".txt")
        report.write_text(analysis + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{output}\n{svg}\n{report}\n\n{analysis}")


if __name__ == "__main__":
    main()
