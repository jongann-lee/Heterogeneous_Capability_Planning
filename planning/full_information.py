"""Exact full-information heterogeneous makespan planning.

Routes are executable in the simulator. Search states are physical nodes plus
the exact mask of targets already serviced, so every supported target crossed
by a route is recorded, already serviced targets may be revisited, and an
unsupported target can never be used as transit. No route relies on another
agent clearing a target first.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
import math
from numbers import Integral
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx


Node = Hashable
_EPS = 1.0e-12


class FullInformationInfeasibleError(ValueError):
    """Raised by the scalar interface when no executable plan exists."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.diagnostic = dict(diagnostic)
        super().__init__(self.diagnostic.get("message", "full-information problem is infeasible"))


@dataclass(frozen=True)
class FullInformationPlan:
    """An exact plan and the evidence needed to execute or inspect it."""

    feasible: bool
    makespan: float
    assignments: tuple[tuple[Node, ...], ...]
    target_orders: tuple[tuple[Node, ...], ...]
    paths: tuple[tuple[Node, ...], ...]
    finish_times: tuple[float, ...]
    diagnostic: Mapping[str, Any]
    exact: bool = True


# Descriptive compatibility name used by the public-interface documentation.
OraclePlan = FullInformationPlan


@dataclass
class _AgentOptions:
    costs: list[float]
    endpoints: list[tuple[Node, int] | None]
    parents: dict[tuple[Node, int], tuple[Node, int] | None] | None


def _stable_nodes(nodes: Sequence[Node]) -> list[Node]:
    return sorted(nodes, key=lambda node: (type(node).__name__, repr(node)))


def _live_targets(graph: nx.Graph) -> list[Node]:
    return _stable_nodes([
        node for node, data in graph.nodes(data=True)
        if data.get("type") == "target_unreached"
    ])


def _target_type(graph: nx.Graph, node: Node) -> int:
    value = graph.nodes[node].get("rps_type")
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(
            f"full-information target {node!r} needs a known positive integer "
            f"rps_type, got {value!r}"
        )
    return int(value)


def _can_service(agent: Any, target_type: int) -> bool:
    can_service = getattr(agent, "can_service", None)
    if callable(can_service):
        return bool(can_service(target_type))
    return target_type in getattr(agent, "capabilities", ())


def _edge_distance(graph: nx.Graph, u: Node, v: Node) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        return math.inf
    if graph.is_multigraph():
        return min(
            (float(attrs.get("distance", 1.0)) for attrs in data.values()),
            default=math.inf,
        )
    return float(data.get("distance", 1.0))


def _validate_edge_weights(graph: nx.Graph) -> None:
    for u, v, data in graph.edges(data=True):
        value = float(data.get("distance", 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"edge {(u, v)!r} has invalid distance {value!r}")


def _agent_options(
    graph: nx.Graph,
    start: Node,
    allowed_mask: int,
    targets: Sequence[Node],
    target_index: Mapping[Node, int],
    *,
    reconstruct: bool,
) -> _AgentOptions:
    """Shortest physical route for every exact serviced-target mask."""

    full_size = 1 << len(targets)
    costs = [math.inf] * full_size
    endpoints: list[tuple[Node, int] | None] = [None] * full_size
    parents: dict[tuple[Node, int], tuple[Node, int] | None] | None = (
        {} if reconstruct else None
    )
    start_mask = 0
    if start in target_index:
        start_mask = 1 << target_index[start]
    start_state = (start, start_mask)
    if allowed_mask == 0:
        costs[0] = 0.0
        endpoints[0] = start_state
        if parents is not None:
            parents[start_state] = None
        return _AgentOptions(costs=costs, endpoints=endpoints, parents=parents)
    best = {start_state: 0.0}
    if parents is not None:
        parents[start_state] = None
    sequence = count()
    heap = [(0.0, next(sequence), start, start_mask)]

    while heap:
        distance, _tie, node, mask = heapq.heappop(heap)
        state = (node, mask)
        if distance > best.get(state, math.inf) + _EPS:
            continue
        if distance < costs[mask] - _EPS:
            costs[mask] = distance
            endpoints[mask] = state

        neighbors = graph.successors(node) if graph.is_directed() else graph.neighbors(node)
        for neighbor in neighbors:
            next_mask = mask
            if neighbor in target_index:
                bit = 1 << target_index[neighbor]
                if not (allowed_mask & bit):
                    continue
                next_mask |= bit
            candidate = distance + _edge_distance(graph, node, neighbor)
            next_state = (neighbor, next_mask)
            if candidate < best.get(next_state, math.inf) - _EPS:
                best[next_state] = candidate
                if parents is not None:
                    parents[next_state] = state
                heapq.heappush(
                    heap, (candidate, next(sequence), neighbor, next_mask)
                )

    return _AgentOptions(costs=costs, endpoints=endpoints, parents=parents)


def _reconstruct_route(
    mask: int,
    options: _AgentOptions,
    targets: Sequence[Node],
) -> tuple[tuple[Node, ...], tuple[Node, ...]]:
    if options.parents is None:
        raise RuntimeError("route reconstruction was not requested")
    state = options.endpoints[mask]
    if state is None:
        raise RuntimeError("missing endpoint for a finite assignment")
    chain: list[tuple[Node, int]] = []
    while state is not None:
        chain.append(state)
        state = options.parents[state]
    chain.reverse()

    order: list[Node] = []
    seen = 0
    for _node, current_mask in chain:
        newly_serviced = current_mask & ~seen
        while newly_serviced:
            bit = newly_serviced & -newly_serviced
            order.append(targets[bit.bit_length() - 1])
            newly_serviced ^= bit
        seen = current_mask
    return tuple(node for node, _mask in chain), tuple(order)


def _infeasible_plan(
    agent_count: int,
    starts: Sequence[Node],
    releases: Sequence[float],
    diagnostic: Mapping[str, Any],
) -> FullInformationPlan:
    return FullInformationPlan(
        feasible=False,
        makespan=math.inf,
        assignments=tuple(() for _ in range(agent_count)),
        target_orders=tuple(() for _ in range(agent_count)),
        paths=tuple((start,) for start in starts),
        finish_times=tuple(releases),
        diagnostic=dict(diagnostic),
    )


def _solve(
    graph: nx.Graph,
    agents: Sequence[Any],
    *,
    start_nodes: Sequence[Node] | None,
    release_times: Sequence[float] | None,
    reconstruct: bool,
) -> FullInformationPlan:
    agents = tuple(agents)
    starts = tuple(start_nodes) if start_nodes is not None else tuple(
        agent.position for agent in agents
    )
    releases = tuple(float(value) for value in (
        release_times if release_times is not None else (0.0,) * len(agents)
    ))
    if len(starts) != len(agents):
        raise ValueError("start_nodes must have one entry per agent")
    if len(releases) != len(agents):
        raise ValueError("release_times must have one entry per agent")
    if any(not math.isfinite(value) or value < 0.0 for value in releases):
        raise ValueError("release_times must be finite and nonnegative")
    missing_starts = [start for start in starts if start not in graph]
    if missing_starts:
        raise ValueError(f"start nodes are absent from the graph: {missing_starts!r}")

    _validate_edge_weights(graph)
    targets = _live_targets(graph)
    if not targets:
        return FullInformationPlan(
            feasible=True,
            makespan=0.0,
            assignments=tuple(() for _ in agents),
            target_orders=tuple(() for _ in agents),
            paths=tuple((start,) for start in starts),
            finish_times=releases,
            diagnostic={"message": "no live targets", "target_count": 0},
        )
    if not agents:
        return _infeasible_plan(
            0, starts, releases,
            {"message": "live targets exist but no agents were supplied", "targets": targets},
        )

    target_types = [_target_type(graph, target) for target in targets]
    target_index = {target: index for index, target in enumerate(targets)}
    required_masks = [0] * len(agents)
    for agent_index, start in enumerate(starts):
        if start in target_index:
            start_target_index = target_index[start]
            target_type = target_types[start_target_index]
            if not _can_service(agents[agent_index], target_type):
                return _infeasible_plan(
                    len(agents), starts, releases,
                    {
                        "message": "an agent starts on an unsupported live target",
                        "agent_index": agent_index,
                        "target": start,
                        "target_type": target_type,
                    },
                )
            required_masks[agent_index] = 1 << start_target_index

    allowed_masks: list[int] = []
    for agent in agents:
        mask = 0
        for index, target_type in enumerate(target_types):
            if _can_service(agent, target_type):
                mask |= 1 << index
        allowed_masks.append(mask)
    union_mask = 0
    for mask in allowed_masks:
        union_mask |= mask
    uncovered = [
        target for index, target in enumerate(targets)
        if not (union_mask & (1 << index))
    ]
    if uncovered:
        return _infeasible_plan(
            len(agents), starts, releases,
            {
                "message": "some target types have no compatible agent",
                "targets": uncovered,
                "target_types": [target_types[target_index[target]] for target in uncovered],
            },
        )

    # Agents with identical starts and service masks share the same physical
    # product-state search. Release offsets affect only the outer min-max DP.
    option_cache: dict[tuple[Node, int], _AgentOptions] = {}
    options = []
    for index in range(len(agents)):
        key = (starts[index], allowed_masks[index])
        if key not in option_cache:
            option_cache[key] = _agent_options(
                graph, starts[index], allowed_masks[index], targets,
                target_index, reconstruct=reconstruct,
            )
        options.append(option_cache[key])

    full_mask = (1 << len(targets)) - 1
    dp: dict[int, float] = {0: 0.0}
    backtrack: list[dict[int, tuple[int, int]]] = []
    for agent_index, agent_options in enumerate(options):
        next_dp: dict[int, float] = {}
        next_parent: dict[int, tuple[int, int]] = {}
        for assigned, makespan in dp.items():
            available = full_mask & ~assigned & allowed_masks[agent_index]
            subset = available
            while True:
                if required_masks[agent_index] & ~subset:
                    if subset == 0:
                        break
                    subset = (subset - 1) & available
                    continue
                travel = agent_options.costs[subset]
                finish = (
                    0.0 if subset == 0
                    else releases[agent_index] + travel
                    if math.isfinite(travel)
                    else math.inf
                )
                if math.isfinite(finish):
                    combined = assigned | subset
                    candidate = max(makespan, finish)
                    if candidate < next_dp.get(combined, math.inf) - _EPS:
                        next_dp[combined] = candidate
                        next_parent[combined] = (assigned, subset)
                if subset == 0:
                    break
                subset = (subset - 1) & available
        dp = next_dp
        backtrack.append(next_parent)

    if full_mask not in dp:
        return _infeasible_plan(
            len(agents), starts, releases,
            {
                "message": "no executable assignment can reach every target",
                "targets": targets,
                "scope": "routes may not cross live targets outside their assignment",
            },
        )

    assigned_masks = [0] * len(agents)
    mask = full_mask
    for agent_index in range(len(agents) - 1, -1, -1):
        previous_mask, subset = backtrack[agent_index][mask]
        assigned_masks[agent_index] = subset
        mask = previous_mask

    assignments: list[tuple[Node, ...]] = []
    target_orders: list[tuple[Node, ...]] = []
    paths: list[tuple[Node, ...]] = []
    finish_times: list[float] = []
    for agent_index, subset in enumerate(assigned_masks):
        if reconstruct:
            route, order = _reconstruct_route(
                subset, options[agent_index], targets
            )
        else:
            route = (starts[agent_index],)
            order = tuple(
                targets[index] for index in range(len(targets))
                if subset & (1 << index)
            )
        assignments.append(tuple(
            targets[index] for index in range(len(targets)) if subset & (1 << index)
        ))
        target_orders.append(order)
        paths.append(route)
        travel = options[agent_index].costs[subset]
        finish_times.append(releases[agent_index] + (travel if subset else 0.0))

    return FullInformationPlan(
        feasible=True,
        makespan=dp[full_mask],
        assignments=tuple(assignments),
        target_orders=tuple(target_orders),
        paths=tuple(paths),
        finish_times=tuple(finish_times),
        diagnostic={
            "message": "optimal executable full-information plan",
            "target_count": len(targets),
            "scope": "no route crosses a live target outside its assignment",
        },
    )


def solve_full_information(
    graph: nx.Graph,
    agents: Sequence[Any],
    *,
    start_nodes: Sequence[Node] | None = None,
    release_times: Sequence[float] | None = None,
) -> FullInformationPlan:
    """Return an exact heterogeneous min--max plan with physical routes."""

    return _solve(
        graph, agents, start_nodes=start_nodes, release_times=release_times,
        reconstruct=True,
    )


def full_information_makespan(
    graph: nx.Graph,
    agents: Sequence[Any],
    *,
    start_nodes: Sequence[Node] | None = None,
    release_times: Sequence[float] | None = None,
) -> float:
    """Return the exact makespan without retaining route reconstruction data."""

    plan = _solve(
        graph, agents, start_nodes=start_nodes, release_times=release_times,
        reconstruct=False,
    )
    if not plan.feasible:
        raise FullInformationInfeasibleError(plan.diagnostic)
    return plan.makespan


__all__ = [
    "FullInformationInfeasibleError",
    "FullInformationPlan",
    "OraclePlan",
    "full_information_makespan",
    "solve_full_information",
]
