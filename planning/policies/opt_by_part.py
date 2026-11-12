"""Concurrent optimal-by-part scouting and known-target execution.

At each simulator replan, pure scouts cover unknown targets and service-only
agents execute known targets. Known-task counts by type minus service-only
capability counts determine the gap that hybrids fill. Remaining hybrids scout.
The routing solvers are shared with Scout-Then-Execute; the combined mission is
not jointly optimal.
"""

from __future__ import annotations

from collections import Counter
import random
from typing import Any

import networkx as nx

from planning.full_information import solve_full_information
from planning.policies.scout_then_execute import solve_cooperative_scouting
from planning.policies.scout_wrp import (
    DEFAULT_EXACT_TARGET_CAP,
    DEFAULT_SUBOPT_WEIGHT,
)
from simulation.domain import UNKNOWN_TYPE


def _select_hybrid_executors(env_map, roster, known):
    """Greedily fill per-type demand, counting each agent capability once.

    One service-only agent with capabilities {1, 2} subtracts one unit of each
    type, as does a selected hybrid with those positive capabilities. This is a
    role-selection heuristic; FI-OPT still lets an agent service multiple tasks.
    Prefer the hybrid covering the most remaining deficit units, breaking ties
    uniformly at random using Python's seeded RNG. Never recruit a hybrid whose
    marginal contribution is zero.
    """
    demand = Counter(env_map.nodes[target]["rps_type"] for target in known)
    service_supply = Counter(
        capability
        for agent in roster if agent.alive and not agent.scout_capable
        for capability in agent.capabilities if capability > 0
    )
    gap = demand - service_supply
    remaining = gap.copy()
    candidates = [
        index for index, agent in enumerate(roster)
        if agent.alive and agent.scout_capable and agent.can_engage
    ]
    selected = set()
    while candidates:
        scores = {
            index: sum(remaining[capability] > 0
                       for capability in roster[index].capabilities
                       if capability > 0)
            for index in candidates
        }
        best_score = max(scores.values())
        if best_score == 0:
            break
        tied = [index for index in candidates if scores[index] == best_score]
        best = random.choice(tied) if len(tied) > 1 else tied[0]
        contribution = Counter({
            capability: 1 for capability in roster[best].capabilities
            if capability > 0 and remaining[capability] > 0
        })
        selected.add(best)
        remaining -= contribution
        candidates.remove(best)
    return selected, demand, service_supply, gap, remaining


class OptByPartPolicy:
    """Recompute both parts on every callback, preserving committed edges.

    Only planner-visible state is used. Unknown and deferred known targets are
    excluded from the execution graph, so they cannot become transit shortcuts.
    Known targets with no compatible executor are deferred until roles change.
    A failed solve leaves that group waiting; the other group can still act and
    later reveals, service, or blockage observations trigger another solve.
    """

    replan_in_transit = True

    def __init__(
        self,
        *,
        exact_target_cap: int = DEFAULT_EXACT_TARGET_CAP,
        suboptimal_weight: float = DEFAULT_SUBOPT_WEIGHT,
    ):
        self.exact_target_cap = int(exact_target_cap)
        self.suboptimal_weight = float(suboptimal_weight)
        self.diagnostics: dict[str, Any] = {}
        self._roster: list[Any] = []
        self._transit: list[Any] = []
        self._clock = 0.0
        self._scout_plan = None
        self._fi_plan = None

    def set_runtime_state(self, agents, transit, clock):
        self._roster = list(agents)
        self._transit = list(transit)
        self._clock = float(clock)

    def __call__(self, env_map, agents, **_kwargs):
        # The engine hook supplies every agent, including those in transit.
        # Also support ordinary direct calls when no runtime hook was used.
        roster = self._roster or list(agents)
        transit = self._transit if self._roster else [None] * len(roster)
        live = {
            node for node, data in env_map.nodes(data=True)
            if data.get("type") == "target_unreached"
        }
        unknown = {
            node for node in live
            if env_map.nodes[node].get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE
        }
        known = live - unknown
        hybrid_executors, demand, service_supply, gap, remaining_gap = (
            _select_hybrid_executors(env_map, roster, known)
        )

        starts = []
        releases = []
        scouts = []
        executors = []
        committed_scouts = []
        roles = ["wait"] * len(roster)
        for index, agent in enumerate(roster):
            moving = transit[index]
            start = agent.position if moving is None else moving[1]
            starts.append(start)
            releases.append(
                0.0 if moving is None
                else max(0.0, float(moving[3]) - self._clock)
            )
            if not agent.alive:
                continue
            # Replace only the route after the committed arrival, never the edge.
            agent.planned_path = [start]
            agent.reveal_schedule = {}
            if agent.scout_capable and index not in hybrid_executors:
                if unknown:
                    roles[index] = "scout"
                    if start in live:
                        # A former executor may already be entering a known
                        # target. Its encounter must finish before it can scout.
                        committed_scouts.append(index)
                    else:
                        scouts.append(index)
            elif agent.can_engage and known:
                roles[index] = "execute"
                executors.append(index)

        self._scout_plan = None
        if unknown:
            self._scout_plan = solve_cooperative_scouting(
                env_map,
                [roster[index] for index in scouts],
                start_nodes=[starts[index] for index in scouts],
                release_times=[releases[index] for index in scouts],
                exact_target_cap=self.exact_target_cap,
                suboptimal_weight=self.suboptimal_weight,
            )
            if self._scout_plan.feasible:
                for index, path, schedule in zip(
                    scouts, self._scout_plan.paths,
                    self._scout_plan.reveal_schedules,
                ):
                    roster[index].planned_path = list(path)
                    roster[index].reveal_schedule = dict(schedule)

        executable = {
            target for target in known
            if any(roster[index].can_service(env_map.nodes[target]["rps_type"])
                   for index in executors)
        }
        deferred = known - executable
        blocked = unknown | deferred
        execution_graph = nx.subgraph_view(
            env_map, filter_node=lambda node: node not in blocked
        )
        self._fi_plan = solve_full_information(
            execution_graph,
            [roster[index] for index in executors],
            start_nodes=[starts[index] for index in executors],
            release_times=[releases[index] for index in executors],
        )
        if self._fi_plan.feasible:
            for index, path in zip(executors, self._fi_plan.paths):
                roster[index].planned_path = list(path)

        scout_plan = self._scout_plan
        stable = lambda values: sorted(values, key=lambda n: (type(n).__name__, repr(n)))
        self.diagnostics = {
            "phase": "concurrent" if unknown else "execution",
            "unknown_target_count": len(unknown),
            "known_target_count": len(known),
            "hybrid_assignment_rule": "capability_gap",
            "hybrid_executor_indices": sorted(hybrid_executors),
            "known_capability_demand": dict(sorted(demand.items())),
            "service_only_capability_supply": dict(sorted(service_supply.items())),
            "initial_capability_gap": dict(sorted(gap.items())),
            "remaining_capability_gap": dict(sorted(remaining_gap.items())),
            "agent_roles": roles,
            "scout_agent_indices": scouts,
            "executor_agent_indices": executors,
            "scouts_finishing_committed_encounters": committed_scouts,
            "deferred_known_targets": stable(deferred),
            "scouting_feasible": scout_plan.feasible if scout_plan else True,
            "scouting_exact": scout_plan.exact if scout_plan else True,
            "scouting_makespan_estimate": scout_plan.makespan if scout_plan else 0.0,
            "scout_responsibilities": (
                [list(values) for values in scout_plan.responsibilities]
                if scout_plan else []
            ),
            "scout_routes": [list(path) for path in scout_plan.paths] if scout_plan else [],
            "scouting": dict(scout_plan.diagnostic) if scout_plan else {},
            "execution_feasible": self._fi_plan.feasible,
            "execution_exact": self._fi_plan.exact,
            "execution_makespan_estimate": self._fi_plan.makespan,
            "fi_assignments": [list(values) for values in self._fi_plan.assignments],
            "fi_paths": [list(path) for path in self._fi_plan.paths],
            "full_information": dict(self._fi_plan.diagnostic),
        }


def make_policy(**kwargs) -> OptByPartPolicy:
    return OptByPartPolicy(**kwargs)


__all__ = ["OptByPartPolicy", "make_policy"]
