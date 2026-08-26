"""CPU/NetworkX planner observations and padding utilities."""

from dataclasses import dataclass, fields
from typing import Any

import networkx as nx
import torch

from learning.policy.candidates import Candidate, generate_candidates
from learning.policy.configuration import CandidateConfig, load_config
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
    action_capacities: torch.Tensor | None = None
    task_agent_features: torch.Tensor | None = None
    task_target_features: torch.Tensor | None = None
    task_action_features: torch.Tensor | None = None
    agent_target_distances: torch.Tensor | None = None
    agent_action_distances: torch.Tensor | None = None
    action_target_distances: torch.Tensor | None = None
    agent_target_distance_mask: torch.Tensor | None = None
    agent_action_distance_mask: torch.Tensor | None = None
    action_target_distance_mask: torch.Tensor | None = None
    serves_mask: torch.Tensor | None = None
    reveals_mask: torch.Tensor | None = None
    stages_for_mask: torch.Tensor | None = None

    def to(self, device):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if torch.is_tensor(value) else value
        return PlannerObservation(**values)


def feature_dimensions(num_target_types: int) -> tuple[int, int, int, int, int, int]:
    """Return ``(Fa, Ft, Fc, Fat, Fac, Fct)``."""
    return (9 + num_target_types, 8 + num_target_types, 11, 6, 6, 7)


def attach_task_graph_fields(observation: PlannerObservation,
                             agent_target_reachable: torch.Tensor,
                             action_target_reachable: torch.Tensor,
                             agent_target_distances: torch.Tensor | None = None,
                             agent_action_distances: torch.Tensor | None = None,
                             action_target_distances: torch.Tensor | None = None,
                             agent_remaining_times: torch.Tensor | None = None,
                             ) -> PlannerObservation:
    """Attach the geometry-free heterogeneous task-graph view in-place.

    The legacy observation remains intact for the Transformer control policy.
    Task-graph callers provide raw traversal times for both the agent node
    feature and distance relations. The fallbacks preserve compatibility for
    callers that only construct the legacy normalized view. Explicit masks
    distinguish an unreachable pair from a genuine zero-distance pair.
    """
    agent = observation.agent_features
    target = observation.target_features
    action = observation.action_features
    num_target_types = agent.shape[-1] - 9

    remaining_feature = (
        agent[..., 8:9] if agent_remaining_times is None else
        agent_remaining_times[..., None])
    # alive, raw remaining transit time, scout capability, then every positive
    # service capability.
    observation.task_agent_features = torch.cat(
        (agent[..., 2:3], remaining_feature, agent[..., 3:4], agent[..., 9:]),
        dim=-1)

    known = target[..., 4:5]
    unknown = target[..., 5:6]
    prior = unknown.expand(*unknown.shape[:-1], num_target_types) \
        / float(num_target_types)
    belief = target[..., 8:] * known + prior
    observation.task_target_features = torch.cat(
        (target[..., 3:4], belief), dim=-1)
    observation.task_action_features = action[..., 2:6]

    legacy_remaining = agent[..., 8:9].unsqueeze(2)
    observation.agent_target_distances = (
        observation.agent_target_relations[..., 0:1] + legacy_remaining
        if agent_target_distances is None else agent_target_distances)
    observation.agent_action_distances = (
        observation.agent_action_relations[..., 0:1] + legacy_remaining
        if agent_action_distances is None else agent_action_distances)
    observation.action_target_distances = (
        observation.action_target_relations[..., 3:4]
        if action_target_distances is None else action_target_distances)

    at_nodes = observation.agent_mask[:, :, None] & observation.target_mask[:, None]
    aa_nodes = observation.agent_mask[:, :, None] & observation.action_mask[:, None]
    ct_nodes = observation.action_mask[:, :, None] & observation.target_mask[:, None]
    observation.agent_target_distance_mask = (
        agent_target_reachable.bool() & at_nodes)
    observation.agent_action_distance_mask = (
        observation.agent_action_relations[..., 2].bool() & aa_nodes)
    observation.action_target_distance_mask = (
        action_target_reachable.bool() & ct_nodes)
    observation.serves_mask = (
        observation.action_target_relations[..., 0].bool() & ct_nodes)
    observation.reveals_mask = (
        observation.action_target_relations[..., 1].bool() & ct_nodes)
    observation.stages_for_mask = (
        observation.action_target_relations[..., 2].bool() & ct_nodes)
    return observation


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


def candidate_path(graph, source, candidate, live_targets):
    """Return the scored route to a physical action or nearest region entry."""
    goals = (candidate.region_nodes if candidate.is_observation and
             candidate.region_nodes else {candidate.node})
    best_path, best_key = None, None
    for goal in goals:
        path = _safe_path(graph, source, goal, live_targets,
                          allow_goal=candidate.is_target)
        if path is None:
            continue
        key = (_path_distance(graph, path), repr(goal))
        if best_key is None or key < best_key:
            best_path, best_key = path, key
    return best_path


def build_observation(graph: nx.Graph, agents, num_target_types: int,
                      candidates: list[Candidate] | None = None,
                      transit: list | None = None, clock: float = 0.0,
                      committed_targets: dict[Any, Any] | None = None,
                      candidate_config: CandidateConfig | None = None,
                      replan_transit: bool = False,
                      ) -> PlannerObservation:
    """Build one observation using only ``graph`` (the planner's view)."""
    if candidates is None:
        candidates = generate_candidates(
            graph, candidate_config or load_config().candidates)
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
    at_reachable = torch.zeros((len(agents), len(targets)), dtype=torch.bool)
    ct_reachable = torch.zeros((len(candidates), len(targets)), dtype=torch.bool)
    task_at_distance = torch.zeros(
        (len(agents), len(targets), 1), dtype=torch.float32)
    task_ac_distance = torch.zeros(
        (len(agents), len(candidates), 1), dtype=torch.float32)
    task_ct_distance = torch.zeros(
        (len(candidates), len(targets), 1), dtype=torch.float32)
    raw_remaining = torch.zeros(len(agents), dtype=torch.float32)

    for i, (agent, travel) in enumerate(zip(agents, transit)):
        x, y = pos.get(agent.position, (0.0, 0.0))
        destination = travel[1] if travel is not None else (
            agent.planned_path[-1] if agent.planned_path else agent.position)
        dx, dy = pos.get(destination, (0.0, 0.0))
        remaining = max(0.0, float(travel[3]) - clock) if travel else 0.0
        raw_remaining[i] = remaining
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
        region = candidate.region_nodes or ({candidate.node}
                                             if candidate.node is not None else set())
        if region:
            x = sum(pos[node][0] for node in region) / len(region)
            y = sum(pos[node][1] for node in region) / len(region)
        else:
            x, y = 0.0, 0.0
        heights = [float(d.get("height", 0.0)) for _, d in graph.nodes(data=True)]
        hscale = max((abs(h) for h in heights), default=1.0) or 1.0
        height = (sum(float(graph.nodes[node].get("height", 0.0))
                      for node in region) / len(region) / hscale
                  if region else 0.0)
        action_x[c] = torch.tensor([
            x, y, float(candidate.is_target), float(candidate.is_observation),
            float(candidate.is_staging), float(candidate.is_wait),
            len(candidate.associated_targets) / max(len(targets), 1),
            len(candidate.observed_targets) /
            max(len(targets), 1), height,
            float(candidate.capacity is None),
            0.0 if candidate.capacity is None else float(candidate.capacity),
        ])

    for i, (agent, travel) in enumerate(zip(agents, transit)):
        planning_position = (travel[1]
                             if travel is not None and replan_transit
                             else agent.position)
        for j, target in enumerate(targets):
            path = _safe_path(graph, planning_position, target, live)
            dist = _path_distance(graph, path)
            at_reachable[i, j] = path is not None
            if path is not None:
                task_at_distance[i, j, 0] = raw_remaining[i] + dist
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
            else:
                path = candidate_path(
                    graph, planning_position, candidate, live)
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
            if travel is not None and not replan_transit:
                valid = False
            feasible[i, c] = valid
            if reachable:
                task_ac_distance[i, c, 0] = raw_remaining[i] + dist
            ac_rel[i, c] = torch.tensor([
                0.0 if dist == float("inf") else dist / distance_scale,
                0.0 if dist == float("inf") else dist / distance_scale,
                float(reachable), float(crosses),
                float(category_ok), float(compatible),
            ])

    for c, candidate in enumerate(candidates):
        for j, target in enumerate(targets):
            region = candidate.region_nodes or ({candidate.node}
                                                 if candidate.node is not None else set())
            if not region:
                dist = task_dist = float("inf")
            else:
                distances, task_distances = [], []
                for node in region:
                    try:
                        distances.append(nx.shortest_path_length(
                            graph, node, target, weight="distance"))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                    path = _safe_path(graph, node, target, live)
                    if path is not None:
                        task_distances.append(_path_distance(graph, path))
                dist = min(distances, default=float("inf"))
                task_dist = min(task_distances, default=float("inf"))
            known = graph.nodes[target].get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE
            ct_rel[c, j] = torch.tensor([
                float(candidate.node == target),
                float(target in candidate.observed_targets),
                float(target in candidate.staging_targets),
                0.0 if dist == float("inf") else dist / distance_scale,
                float(target in live), float(known), float(not known),
            ])
            ct_reachable[c, j] = bool(
                region and task_dist != float("inf"))
            if task_dist != float("inf"):
                task_ct_distance[c, j, 0] = task_dist

    observation = PlannerObservation(
        agent_x.unsqueeze(0), target_x.unsqueeze(0), action_x.unsqueeze(0),
        torch.ones((1, len(agents)), dtype=torch.bool),
        torch.ones((1, len(targets)), dtype=torch.bool),
        torch.ones((1, len(candidates)), dtype=torch.bool),
        at_rel.unsqueeze(0), ac_rel.unsqueeze(0), ct_rel.unsqueeze(0),
        feasible.unsqueeze(0), [candidates], [agents], [targets],
    )
    return attach_task_graph_fields(
        observation, at_reachable.unsqueeze(0), ct_reachable.unsqueeze(0),
        agent_target_distances=task_at_distance.unsqueeze(0),
        agent_action_distances=task_ac_distance.unsqueeze(0),
        action_target_distances=task_ct_distance.unsqueeze(0),
        agent_remaining_times=raw_remaining.unsqueeze(0))


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
        "task_agent_features": (1, a, items[0].task_agent_features.shape[-1]),
        "task_target_features": (1, t, items[0].task_target_features.shape[-1]),
        "task_action_features": (1, c, items[0].task_action_features.shape[-1]),
        "agent_target_distances": (1, a, t, 1),
        "agent_action_distances": (1, a, c, 1),
        "action_target_distances": (1, c, t, 1),
        "agent_target_distance_mask": (1, a, t),
        "agent_action_distance_mask": (1, a, c),
        "action_target_distance_mask": (1, c, t),
        "serves_mask": (1, c, t),
        "reveals_mask": (1, c, t),
        "stages_for_mask": (1, c, t),
    }
    for name, shape in specs.items():
        kwargs[name] = torch.cat([pad(getattr(item, name), shape) for item in items])
    kwargs.update(candidates=sum((x.candidates or [] for x in items), []),
                  agents=sum((x.agents or [] for x in items), []),
                  targets=sum((x.targets or [] for x in items), []),
                  action_capacities=None)
    return PlannerObservation(**kwargs)
