"""Planner-visible tensor observations and padding utilities."""

from dataclasses import dataclass, fields
from typing import Any

import networkx as nx
import torch

from learning.candidates import Candidate, generate_candidates
from simulation.domain import UNKNOWN_TYPE


@dataclass
class PlannerObservation:
    agent_features: torch.Tensor
    target_features: torch.Tensor
    action_features: torch.Tensor
    agent_mask: torch.Tensor
    target_mask: torch.Tensor
    action_mask: torch.Tensor
    agent_target_relations: torch.Tensor
    agent_action_relations: torch.Tensor
    action_target_relations: torch.Tensor
    feasible_action_mask: torch.Tensor
    candidates: list[list[Candidate]] | None = None
    agents: list[list[Any]] | None = None
    targets: list[list[Any]] | None = None

    def to(self, device):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if torch.is_tensor(value) else value
        return PlannerObservation(**values)


def feature_dimensions(num_target_types: int) -> tuple[int, int, int, int, int, int]:
    """Return ``(Fa, Ft, Fc, Fat, Fac, Fct)``."""
    return (9 + num_target_types, 8 + num_target_types, 12, 6, 7, 7)


def _positions(graph):
    raw = {n: graph.nodes[n].get("pos", (0.0, 0.0)) for n in graph.nodes}
    xs = [float(p[0]) for p in raw.values()] or [0.0]
    ys = [float(p[1]) for p in raw.values()] or [0.0]
    x0, y0 = min(xs), min(ys)
    scale = max(max(xs) - x0, max(ys) - y0, 1.0)
    return {n: ((float(p[0]) - x0) / scale,
                (float(p[1]) - y0) / scale) for n, p in raw.items()}


def _distance_scale(graph):
    return max(sum(float(d.get("distance", 1.0))
                   for _, _, d in graph.edges(data=True)), 1.0)


def _safe_path(graph, source, goal, live_targets, allow_goal=True):
    if source == goal:
        return [source]
    avoid = set(live_targets) - ({goal} if allow_goal else set()) - {source}
    view = graph.copy() if avoid else graph
    if avoid:
        view.remove_nodes_from(avoid)
    try:
        return nx.shortest_path(view, source, goal, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _path_distance(graph, path):
    if not path:
        return float("inf")
    return sum(float(graph.edges[u, v].get("distance", 1.0))
               for u, v in zip(path, path[1:]))


def build_observation(graph: nx.Graph, agents, num_target_types: int,
                      candidates: list[Candidate] | None = None,
                      transit: list | None = None, clock: float = 0.0,
                      committed_targets: dict[Any, Any] | None = None
                      ) -> PlannerObservation:
    """Build one observation using only ``graph`` (the planner's view)."""
    if candidates is None:
        candidates = generate_candidates(graph)
    agents = list(agents)
    transit = list(transit) if transit is not None else [None] * len(agents)
    if len(transit) != len(agents):
        raise ValueError("transit must align with agents")
    committed_targets = committed_targets or {}
    targets = [n for n, d in graph.nodes(data=True)
               if d.get("type") in ("target_unreached", "target_reached")]
    targets.sort(key=repr)
    pos = _positions(graph)
    distance_scale = _distance_scale(graph)
    live = {n for n in targets if graph.nodes[n].get("type") == "target_unreached"}
    fa, ft, fc, fat, fac, fct = feature_dimensions(num_target_types)

    agent_x = torch.zeros((len(agents), fa), dtype=torch.float32)
    target_x = torch.zeros((len(targets), ft), dtype=torch.float32)
    action_x = torch.zeros((len(candidates), fc), dtype=torch.float32)
    at_rel = torch.zeros((len(agents), len(targets), fat), dtype=torch.float32)
    ac_rel = torch.zeros((len(agents), len(candidates), fac), dtype=torch.float32)
    ct_rel = torch.zeros((len(candidates), len(targets), fct), dtype=torch.float32)
    feasible = torch.zeros((len(agents), len(candidates)), dtype=torch.bool)

    for i, (agent, travel) in enumerate(zip(agents, transit)):
        x, y = pos.get(agent.position, (0.0, 0.0))
        destination = travel[1] if travel is not None else (
            agent.planned_path[-1] if agent.planned_path else agent.position)
        dx, dy = pos.get(destination, (0.0, 0.0))
        remaining = max(0.0, float(travel[3]) - clock) if travel else 0.0
        base = [x, y, float(agent.alive), float(agent.scout_capable),
                float(travel is None), float(travel is not None), dx, dy,
                remaining / distance_scale]
        caps = [float(k in agent.capabilities)
                for k in range(1, num_target_types + 1)]
        agent_x[i] = torch.tensor(base + caps)

    living_capable = {
        kind: sum(a.alive and a.can_service(kind) for a in agents)
        for kind in range(1, num_target_types + 1)
    }
    for j, target in enumerate(targets):
        data = graph.nodes[target]
        known_type = int(data.get("rps_type", UNKNOWN_TYPE))
        known = 1 <= known_type <= num_target_types
        one_hot = [float(known and known_type == k)
                   for k in range(1, num_target_types + 1)]
        x, y = pos[target]
        target_x[j] = torch.tensor([
            x, y, float(target in live), float(target not in live),
            float(known), float(not known),
            float(target in committed_targets),
            (living_capable.get(known_type, 0) / max(len(agents), 1))
            if known else 0.0,
            *one_hot,
        ])

    for c, candidate in enumerate(candidates):
        x, y = pos.get(candidate.node, (0.0, 0.0))
        heights = [float(d.get("height", 0.0)) for _, d in graph.nodes(data=True)]
        hscale = max((abs(h) for h in heights), default=1.0) or 1.0
        height = (float(graph.nodes[candidate.node].get("height", 0.0)) / hscale
                  if candidate.node in graph else 0.0)
        action_x[c] = torch.tensor([
            x, y, float(candidate.is_target), float(candidate.is_observation),
            float(candidate.is_staging), float(candidate.is_wait),
            float(candidate.is_continue), len(candidate.associated_targets) /
            max(len(targets), 1), len(candidate.observed_targets) /
            max(len(targets), 1), height,
            float(candidate.capacity is None),
            0.0 if candidate.capacity is None else float(candidate.capacity),
        ])

    for i, (agent, travel) in enumerate(zip(agents, transit)):
        for j, target in enumerate(targets):
            path = _safe_path(graph, agent.position, target, live)
            dist = _path_distance(graph, path)
            kind = int(graph.nodes[target].get("rps_type", UNKNOWN_TYPE))
            known = kind != UNKNOWN_TYPE
            at_rel[i, j] = torch.tensor([
                0.0 if dist == float("inf") else dist / distance_scale,
                0.0 if dist == float("inf") else dist / distance_scale,
                float(known), float(not known),
                float(known and agent.can_service(kind)),
                float(committed_targets.get(target) is agent),
            ])
        for c, candidate in enumerate(candidates):
            if candidate.is_wait:
                reachable, dist, crosses = True, 0.0, False
            elif candidate.is_continue:
                reachable, dist, crosses = travel is not None, 0.0, False
            else:
                path = _safe_path(graph, agent.position, candidate.node, live,
                                  allow_goal=candidate.is_target)
                reachable = path is not None
                dist = _path_distance(graph, path)
                crosses = bool(path and (set(path[1:-1]) & live))
            category_ok = not (candidate.is_observation and
                               not candidate.is_target and
                               not candidate.is_staging and
                               not agent.scout_capable)
            compatible = True
            if candidate.is_target and candidate.node in graph:
                kind = int(graph.nodes[candidate.node].get("rps_type", UNKNOWN_TYPE))
                compatible = kind == UNKNOWN_TYPE or agent.can_service(kind)
            valid = bool(agent.alive and reachable and category_ok and compatible)
            if travel is not None:
                valid = candidate.is_continue
            elif candidate.is_continue:
                valid = False
            feasible[i, c] = valid
            ac_rel[i, c] = torch.tensor([
                0.0 if dist == float("inf") else dist / distance_scale,
                0.0 if dist == float("inf") else dist / distance_scale,
                float(reachable), float(crosses),
                float(candidate.is_continue and travel is not None),
                float(category_ok), float(compatible),
            ])

    for c, candidate in enumerate(candidates):
        for j, target in enumerate(targets):
            if candidate.node is None:
                dist = float("inf")
            else:
                try:
                    dist = nx.shortest_path_length(
                        graph, candidate.node, target, weight="distance")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dist = float("inf")
            known = graph.nodes[target].get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE
            ct_rel[c, j] = torch.tensor([
                float(candidate.node == target),
                float(target in candidate.observed_targets),
                float(target in candidate.staging_targets),
                0.0 if dist == float("inf") else dist / distance_scale,
                float(target in live), float(known), float(not known),
            ])

    return PlannerObservation(
        agent_x.unsqueeze(0), target_x.unsqueeze(0), action_x.unsqueeze(0),
        torch.ones((1, len(agents)), dtype=torch.bool),
        torch.ones((1, len(targets)), dtype=torch.bool),
        torch.ones((1, len(candidates)), dtype=torch.bool),
        at_rel.unsqueeze(0), ac_rel.unsqueeze(0), ct_rel.unsqueeze(0),
        feasible.unsqueeze(0), [candidates], [agents], [targets],
    )


def batch_observations(items: list[PlannerObservation]) -> PlannerObservation:
    """Pad single-example observations into a batch."""
    if not items:
        raise ValueError("cannot batch an empty observation list")
    maxima = [max(x.agent_features.shape[1] for x in items),
              max(x.target_features.shape[1] for x in items),
              max(x.action_features.shape[1] for x in items)]

    def pad(tensor, shape, value=0):
        out = tensor.new_full(shape, value)
        slices = (slice(0, 1),) + tuple(slice(0, n) for n in tensor.shape[1:])
        out[slices] = tensor
        return out

    a, t, c = maxima
    kwargs = {}
    specs = {
        "agent_features": (1, a, items[0].agent_features.shape[-1]),
        "target_features": (1, t, items[0].target_features.shape[-1]),
        "action_features": (1, c, items[0].action_features.shape[-1]),
        "agent_mask": (1, a), "target_mask": (1, t), "action_mask": (1, c),
        "agent_target_relations": (1, a, t, items[0].agent_target_relations.shape[-1]),
        "agent_action_relations": (1, a, c, items[0].agent_action_relations.shape[-1]),
        "action_target_relations": (1, c, t, items[0].action_target_relations.shape[-1]),
        "feasible_action_mask": (1, a, c),
    }
    for name, shape in specs.items():
        kwargs[name] = torch.cat([pad(getattr(item, name), shape) for item in items])
    kwargs.update(candidates=sum((x.candidates or [] for x in items), []),
                  agents=sum((x.agents or [] for x in items), []),
                  targets=sum((x.targets or [] for x in items), []))
    return PlannerObservation(**kwargs)
