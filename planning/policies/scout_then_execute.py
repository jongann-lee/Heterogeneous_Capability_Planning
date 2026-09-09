"""Strict cooperative Scout-Then-Execute benchmark policy.

Every service-only agent waits while all scout-capable agents jointly minimize
the time of the last target-type reveal.  Once no live target remains unknown,
the policy commits to the exact executable full-information plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Hashable, Mapping, Sequence

from planning.full_information import FullInformationPlan, solve_full_information
from planning.policies.scout_wrp import (
    DEFAULT_EXACT_TARGET_CAP,
    DEFAULT_SUBOPT_WEIGHT,
    _compile,
    solve_cover_walk,
)
from simulation.domain import UNKNOWN_TYPE


Node = Hashable
_INF = math.inf
_EPS = 1.0e-12


@dataclass(frozen=True)
class ScoutCoveragePlan:
    feasible: bool
    exact: bool
    makespan: float
    responsibilities: tuple[tuple[Node, ...], ...]
    paths: tuple[tuple[Node, ...], ...]
    reveal_schedules: tuple[Mapping[Node, float], ...]
    unscoutable: tuple[Node, ...]
    diagnostic: Mapping[str, Any]


@dataclass
class _CoverageOptions:
    costs: list[float]
    endpoints: list[tuple[int, int] | None]
    parents: dict[tuple[int, int], tuple[int, int] | None]
    nodes: list[Node]
    adjacency: list[list[tuple[int, float]]]
    cell_masks: list[int]
    touched: int


def _unknown_live_targets(env_map) -> list[Node]:
    return sorted(
        (
            node for node, data in env_map.nodes(data=True)
            if data.get("type") == "target_unreached"
            and data.get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE
        ),
        key=lambda node: (type(node).__name__, repr(node)),
    )


def _all_live_targets(env_map) -> set[Node]:
    return {
        node for node, data in env_map.nodes(data=True)
        if data.get("type") == "target_unreached"
    }


def _coverage_options(env_map, start, targets, live_set) -> _CoverageOptions:
    nodes, index, adjacency, _reverse, cell_masks = _compile(
        env_map, live_set, targets
    )
    size = 1 << len(targets)
    costs = [_INF] * size
    endpoints: list[tuple[int, int] | None] = [None] * size
    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    if start not in index:
        return _CoverageOptions(
            costs, endpoints, parents, nodes, adjacency, cell_masks, 0
        )

    start_index = index[start]
    start_mask = cell_masks[start_index]
    start_state = (start_mask, start_index)
    best = {start_state: 0.0}
    parents[start_state] = None
    heap = [(0.0, start_mask, start_index)]
    unsettled = size

    while heap and unsettled:
        distance, mask, node_index = heapq.heappop(heap)
        state = (mask, node_index)
        if distance > best.get(state, _INF) + _EPS:
            continue

        # The first state popped whose mask contains a requested subset is an
        # optimal route for that subset. It may reveal additional targets for
        # free; those reveals remain valid cooperative progress.
        subset = mask
        while True:
            if not math.isfinite(costs[subset]):
                costs[subset] = distance
                endpoints[subset] = state
                unsettled -= 1
            if subset == 0:
                break
            subset = (subset - 1) & mask

        for neighbor, edge_cost in adjacency[node_index]:
            next_mask = mask | cell_masks[neighbor]
            candidate = distance + float(edge_cost)
            next_state = (next_mask, neighbor)
            if candidate < best.get(next_state, _INF) - _EPS:
                best[next_state] = candidate
                parents[next_state] = state
                heapq.heappush(heap, (candidate, next_mask, neighbor))

    return _CoverageOptions(
        costs, endpoints, parents, nodes, adjacency, cell_masks, len(best)
    )


def _reconstruct_coverage(
    option: _CoverageOptions,
    responsibility_mask: int,
    targets: Sequence[Node],
) -> tuple[tuple[Node, ...], Mapping[Node, float]]:
    state = option.endpoints[responsibility_mask]
    if state is None:
        return (), {}
    chain: list[tuple[int, int]] = []
    while state is not None:
        chain.append(state)
        state = option.parents[state]
    chain.reverse()
    path = tuple(option.nodes[node_index] for _mask, node_index in chain)

    schedule: dict[Node, float] = {}
    seen = 0
    clock = 0.0
    for chain_index, (mask, node_index) in enumerate(chain):
        if chain_index:
            previous_index = chain[chain_index - 1][1]
            clock += next(
                float(cost) for neighbor, cost in option.adjacency[previous_index]
                if neighbor == node_index
            )
        newly_seen = mask & ~seen
        while newly_seen:
            bit = newly_seen & -newly_seen
            schedule[targets[bit.bit_length() - 1]] = clock
            newly_seen ^= bit
        seen = mask
    return path, schedule


def _infeasible_coverage(scout_count, starts, targets, message):
    return ScoutCoveragePlan(
        feasible=False,
        exact=True,
        makespan=_INF,
        responsibilities=tuple(() for _ in range(scout_count)),
        paths=tuple((start,) for start in starts),
        reveal_schedules=tuple({} for _ in range(scout_count)),
        unscoutable=tuple(targets),
        diagnostic={"message": message, "unscoutable": tuple(targets)},
    )


def _solve_exact_coverage(
    env_map, scouts, targets, starts, releases
) -> ScoutCoveragePlan:
    if not targets:
        return ScoutCoveragePlan(
            True, True, 0.0,
            tuple(() for _ in scouts), tuple((start,) for start in starts),
            tuple({} for _ in scouts), (),
            {"message": "no unknown live targets", "states_touched": 0},
        )
    if not scouts:
        return _infeasible_coverage(
            0, starts, targets, "unknown targets exist but no scout is alive"
        )

    live_set = _all_live_targets(env_map)
    options = [
        _coverage_options(env_map, start, targets, live_set) for start in starts
    ]
    unscoutable = [
        target for index, target in enumerate(targets)
        if all(not math.isfinite(option.costs[1 << index]) for option in options)
    ]
    if unscoutable:
        return _infeasible_coverage(
            len(scouts), starts, unscoutable,
            "some unknown targets have no reachable visibility witness",
        )

    full_mask = (1 << len(targets)) - 1
    dp: dict[int, float] = {0: 0.0}
    parents: list[dict[int, tuple[int, int]]] = []
    for scout_index, option in enumerate(options):
        next_dp: dict[int, float] = {}
        next_parent: dict[int, tuple[int, int]] = {}
        for assigned, makespan in dp.items():
            available = full_mask & ~assigned
            subset = available
            while True:
                travel = option.costs[subset]
                cost = (
                    0.0 if subset == 0
                    else releases[scout_index] + travel
                    if math.isfinite(travel)
                    else _INF
                )
                if math.isfinite(cost):
                    combined = assigned | subset
                    candidate = max(makespan, cost)
                    if candidate < next_dp.get(combined, _INF) - _EPS:
                        next_dp[combined] = candidate
                        next_parent[combined] = (assigned, subset)
                if subset == 0:
                    break
                subset = (subset - 1) & available
        dp = next_dp
        parents.append(next_parent)

    if full_mask not in dp:
        return _infeasible_coverage(
            len(scouts), starts, targets,
            "no cooperative visibility-cover assignment is executable",
        )

    masks = [0] * len(scouts)
    current = full_mask
    for scout_index in range(len(scouts) - 1, -1, -1):
        previous, subset = parents[scout_index][current]
        masks[scout_index] = subset
        current = previous

    responsibilities = []
    paths = []
    schedules = []
    for scout_index, (mask, option) in enumerate(zip(masks, options)):
        responsibilities.append(tuple(
            target for index, target in enumerate(targets) if mask & (1 << index)
        ))
        path, schedule = _reconstruct_coverage(option, mask, targets)
        paths.append(path)
        schedules.append({
            target: releases[scout_index] + reveal_time
            for target, reveal_time in schedule.items()
        })
    return ScoutCoveragePlan(
        feasible=True,
        exact=True,
        makespan=dp[full_mask],
        responsibilities=tuple(responsibilities),
        paths=tuple(paths),
        reveal_schedules=tuple(schedules),
        unscoutable=(),
        diagnostic={
            "message": "optimal cooperative visibility cover",
            "states_touched": sum(option.touched for option in options),
        },
    )


def _solve_weighted_fallback(
    env_map, scouts, targets, starts, releases, weight
) -> ScoutCoveragePlan:
    """Bounded-size fallback above the configured exact target cap."""

    live_set = _all_live_targets(env_map)
    assigned: list[list[Node]] = [[] for _ in scouts]
    unscoutable = []
    for target in targets:
        choices = []
        for index, start in enumerate(starts):
            path, schedule, missing, _stats = solve_cover_walk(
                env_map, start, [target], live_set, weight=1.0
            )
            if path is not None and target not in missing:
                choices.append((
                    releases[index] + schedule.get(target, 0.0), index
                ))
        if not choices:
            unscoutable.append(target)
        else:
            _cost, index = min(choices)
            assigned[index].append(target)
    if unscoutable:
        plan = _infeasible_coverage(
            len(scouts), starts, unscoutable,
            "some unknown targets have no reachable visibility witness",
        )
        return ScoutCoveragePlan(
            plan.feasible, False, plan.makespan, plan.responsibilities,
            plan.paths, plan.reveal_schedules, plan.unscoutable, plan.diagnostic,
        )

    paths = []
    schedules = []
    makespan = 0.0
    for scout_index, (start, responsibility) in enumerate(zip(starts, assigned)):
        if not responsibility:
            paths.append((start,))
            schedules.append({})
            continue
        path, schedule, missing, _stats = solve_cover_walk(
            env_map, start, responsibility, live_set, weight=weight
        )
        if path is None or missing:
            return ScoutCoveragePlan(
                False, False, _INF, tuple(tuple(values) for values in assigned),
                tuple((value,) for value in starts), tuple({} for _ in scouts),
                tuple(missing), {"message": "weighted coverage fallback failed"},
            )
        paths.append(tuple(path))
        adjusted_schedule = {
            target: releases[scout_index] + reveal_time
            for target, reveal_time in schedule.items()
        }
        schedules.append(adjusted_schedule)
        makespan = max(
            makespan,
            max(adjusted_schedule.values(), default=releases[scout_index]),
        )
    return ScoutCoveragePlan(
        True, False, makespan, tuple(tuple(values) for values in assigned),
        tuple(paths), tuple(schedules), (),
        {"message": f"weighted-A* cooperative fallback (weight={weight:g})"},
    )


def solve_cooperative_scouting(
    env_map,
    scouts: Sequence[Any],
    *,
    unknown_targets: Sequence[Node] | None = None,
    start_nodes: Sequence[Node] | None = None,
    release_times: Sequence[float] | None = None,
    exact_target_cap: int = DEFAULT_EXACT_TARGET_CAP,
    suboptimal_weight: float = DEFAULT_SUBOPT_WEIGHT,
) -> ScoutCoveragePlan:
    """Minimize the final reveal time over all supplied scout agents."""

    scouts = tuple(scouts)
    if any(not getattr(scout, "scout_capable", False) for scout in scouts):
        raise ValueError("solve_cooperative_scouting accepts scout-capable agents only")
    targets = tuple(
        _unknown_live_targets(env_map) if unknown_targets is None
        else sorted(set(unknown_targets), key=lambda node: (type(node).__name__, repr(node)))
    )
    starts = tuple(start_nodes) if start_nodes is not None else tuple(
        scout.position for scout in scouts
    )
    releases = tuple(float(value) for value in (
        release_times if release_times is not None else (0.0,) * len(scouts)
    ))
    if len(starts) != len(scouts):
        raise ValueError("start_nodes must have one entry per scout")
    if len(releases) != len(scouts):
        raise ValueError("release_times must have one entry per scout")
    if any(not math.isfinite(value) or value < 0.0 for value in releases):
        raise ValueError("release_times must be finite and nonnegative")
    if len(targets) <= exact_target_cap:
        return _solve_exact_coverage(
            env_map, scouts, targets, starts, releases
        )
    return _solve_weighted_fallback(
        env_map, scouts, targets, starts, releases, suboptimal_weight
    )


class ScoutThenExecutePolicy:
    """Stateful strict two-phase benchmark compatible with ``run_simulation``."""

    replan_in_transit = True

    def __init__(
        self,
        *,
        exact_target_cap: int = DEFAULT_EXACT_TARGET_CAP,
        suboptimal_weight: float = DEFAULT_SUBOPT_WEIGHT,
    ):
        self.exact_target_cap = int(exact_target_cap)
        self.suboptimal_weight = float(suboptimal_weight)
        self.phase = "scouting"
        self.diagnostics: dict[str, Any] = {
            "phase": self.phase,
            "infeasible": False,
        }
        self._roster: list[Any] = []
        self._transit: list[Any] = []
        self._clock = 0.0
        self._roster_ids: tuple[int, ...] | None = None
        self._scout_plan: ScoutCoveragePlan | None = None
        self._scout_plan_key = None
        self._fi_plan: FullInformationPlan | None = None

    def set_runtime_state(self, agents, transit, clock):
        roster_ids = tuple(id(agent) for agent in agents)
        if self._roster_ids is not None and roster_ids != self._roster_ids:
            self.__init__(
                exact_target_cap=self.exact_target_cap,
                suboptimal_weight=self.suboptimal_weight,
            )
        self._roster_ids = roster_ids
        self._roster = list(agents)
        self._transit = list(transit)
        self._clock = float(clock)

    def _wait_everyone(self):
        for index, agent in enumerate(self._roster):
            if not agent.alive or self._transit[index] is not None:
                continue
            agent.planned_path = [agent.position]

    def _mark_infeasible(self, message, details=None):
        self._wait_everyone()
        self.diagnostics.update({
            "phase": self.phase,
            "infeasible": True,
            "reason": message,
        })
        if details is not None:
            self.diagnostics["details"] = details

    def _begin_scouting(self, env_map):
        roster_indices = [
            index for index, agent in enumerate(self._roster)
            if agent.alive and agent.scout_capable
        ]
        scouts = [self._roster[index] for index in roster_indices]
        starts = []
        releases = []
        for index, scout in zip(roster_indices, scouts):
            transit = self._transit[index]
            if transit is None:
                starts.append(scout.position)
                releases.append(0.0)
            else:
                _source, destination, _depart, arrival = transit
                starts.append(destination)
                releases.append(max(0.0, float(arrival) - self._clock))
        self._wait_everyone()
        self._scout_plan = solve_cooperative_scouting(
            env_map,
            scouts,
            start_nodes=starts,
            release_times=releases,
            exact_target_cap=self.exact_target_cap,
            suboptimal_weight=self.suboptimal_weight,
        )
        if not self._scout_plan.feasible:
            self._mark_infeasible(
                self._scout_plan.diagnostic.get("message", "scouting is infeasible"),
                self._scout_plan.diagnostic,
            )
            return
        for scout, path, schedule in zip(
            scouts, self._scout_plan.paths, self._scout_plan.reveal_schedules
        ):
            scout.planned_path = list(path)
            scout.reveal_schedule = dict(schedule)
        self.diagnostics.update({
            "exact": self._scout_plan.exact,
            "scouting_exact": self._scout_plan.exact,
            "scouting_makespan_estimate": self._scout_plan.makespan,
            "scout_responsibilities": [
                list(values) for values in self._scout_plan.responsibilities
            ],
            "scout_routes": [list(path) for path in self._scout_plan.paths],
            "scout_reveal_schedules": [
                [
                    {"target": target, "time": float(reveal_time)}
                    for target, reveal_time in schedule.items()
                ]
                for schedule in self._scout_plan.reveal_schedules
            ],
            "scouting": dict(self._scout_plan.diagnostic),
        })
        self.diagnostics.setdefault(
            "initial_scouting_makespan_estimate", self._scout_plan.makespan
        )

    def _begin_execution(self, env_map):
        self.phase = "execution"
        scouting_routes = []
        for index, agent in enumerate(self._roster):
            if not agent.scout_capable:
                continue
            route = list(agent.trajectory)
            transit = self._transit[index]
            if transit is not None:
                destination = transit[1]
                if not route or route[-1] != destination:
                    route.append(destination)
            scouting_routes.append(route)
        living = [agent for agent in self._roster if agent.alive]
        roster_index = {id(agent): index for index, agent in enumerate(self._roster)}
        starts = []
        releases = []
        for agent in living:
            index = roster_index[id(agent)]
            transit = self._transit[index]
            if transit is None:
                starts.append(agent.position)
                releases.append(0.0)
            else:
                _u, destination, _depart, arrival = transit
                starts.append(destination)
                releases.append(max(0.0, float(arrival) - self._clock))

        self._fi_plan = solve_full_information(
            env_map, living, start_nodes=starts, release_times=releases
        )
        self.diagnostics.update({
            "phase": self.phase,
            "scouting_completion_time": self._clock,
            "scout_executed_routes": scouting_routes,
            "execution_makespan_estimate": self._fi_plan.makespan,
            "execution_exact": self._fi_plan.exact,
            "predicted_makespan": self._clock + self._fi_plan.makespan,
            "full_information": dict(self._fi_plan.diagnostic),
            "fi_assignments": [list(values) for values in self._fi_plan.assignments],
            "fi_target_orders": [list(values) for values in self._fi_plan.target_orders],
            "fi_paths": [list(path) for path in self._fi_plan.paths],
        })
        if not self._fi_plan.feasible:
            self._mark_infeasible(
                self._fi_plan.diagnostic.get("message", "execution is infeasible"),
                self._fi_plan.diagnostic,
            )
            return
        for agent, path in zip(living, self._fi_plan.paths):
            agent.planned_path = list(path)

    def __call__(self, env_map, agents, **_kwargs):
        del agents  # The runtime hook supplies the complete living roster.
        if self.diagnostics.get("infeasible"):
            return
        if self.phase == "scouting":
            unknown = _unknown_live_targets(env_map)
            if not unknown:
                self._begin_execution(env_map)
            else:
                future_states = []
                for index, agent in enumerate(self._roster):
                    if not agent.alive or not agent.scout_capable:
                        continue
                    transit = self._transit[index]
                    if transit is None:
                        future_states.append((agent.position, 0.0))
                    else:
                        _u, destination, _depart, arrival = transit
                        future_states.append((
                            destination,
                            max(0.0, float(arrival) - self._clock),
                        ))
                key = (
                    tuple(unknown), env_map.number_of_edges(),
                    tuple(future_states),
                )
                if key != self._scout_plan_key:
                    self._scout_plan_key = key
                    self._begin_scouting(env_map)


def make_policy(**kwargs) -> ScoutThenExecutePolicy:
    return ScoutThenExecutePolicy(**kwargs)


__all__ = [
    "ScoutCoveragePlan",
    "ScoutThenExecutePolicy",
    "make_policy",
    "solve_cooperative_scouting",
]
