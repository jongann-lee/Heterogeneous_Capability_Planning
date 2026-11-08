"""Evaluation-only task event logs and cumulative progress curves."""

from itertools import groupby
import math

from simulation.domain import AGENT_DIES, AGENT_WINS, UNKNOWN_TYPE


class TaskProgressRecorder:
    """Record task facts after transitions, without exposing truth to a policy."""

    def __init__(self, targets, truth):
        self.targets = tuple(targets)
        self._indices = {node: index for index, node in enumerate(self.targets)}
        self._tasks = [
            {
                "target_id": index,
                "node": node,
                "target_type": int(truth.nodes[node]["rps_type"]),
                "discovered_at": None,
                "discovery_source": None,
                "completed_at": None,
            }
            for index, node in enumerate(self.targets)
        ]
        self.events = []
        self._previous_alive = None

    def _event(self, time, event, node, **details):
        time = float(time)
        if not math.isfinite(time) or time < 0:
            raise ValueError("task event time must be finite and non-negative")
        if self.events and time < self.events[-1]["time"]:
            raise ValueError("task events must be chronological")
        task = self._tasks[self._indices[node]]
        self.events.append({
            "time": time,
            "event": event,
            "target_id": task["target_id"],
            "node": node,
            "target_type": task["target_type"],
            **details,
        })

    def discover(self, time, node, source):
        task = self._tasks[self._indices[node]]
        if task["discovered_at"] is not None:
            return
        self._event(time, "task_discovered", node, source=source)
        task["discovered_at"] = float(time)
        task["discovery_source"] = source

    def complete(self, time, node):
        task = self._tasks[self._indices[node]]
        if task["completed_at"] is not None:
            return
        # Contact reveals a previously unknown type even without scouting.
        self.discover(time, node, "contact")
        self._event(time, "task_completed", node)
        task["completed_at"] = float(time)

    def initialize(self, env, *, full_information=False):
        for node in self.targets:
            if full_information:
                self.discover(0.0, node, "full_information")
            elif env.nodes[node].get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE:
                self.discover(0.0, node, "initial_knowledge")
            if env.nodes[node].get("type") == "target_reached":
                self.complete(0.0, node)

    def consume_cpu_events(self, events):
        for event in events:
            node = event["node"]
            if node not in self._indices:
                continue
            kind = event["event"]
            if kind == "reveal":
                self.discover(event["time"], node, "scout")
            elif kind in {AGENT_WINS, AGENT_DIES}:
                self.discover(event["time"], node, "contact")
                if kind == AGENT_WINS:
                    self.complete(event["time"], node)

    def capture_tensor_state(self, state):
        """Keep task transitions only; do not retain terrain or route snapshots."""
        known = state.target_known[0].detach().cpu().tolist()
        live = state.target_live[0].detach().cpu().tolist()
        alive = state.alive[0].detach().cpu()
        # A scout senses before contact can kill it in the same transition.
        observers = alive if self._previous_alive is None else (
            alive | self._previous_alive)
        scouts = state.capabilities[0, :, 0].detach().cpu() & observers
        visible = state.world.visible_targets[
            state.positions[0]].detach().cpu()
        scout_visible = (visible & scouts[:, None]).any(dim=0).tolist()
        time = float(state.clock[0])
        for index, node in enumerate(state.world.targets):
            if known[index]:
                self.discover(
                    time, node, "scout" if scout_visible[index] else "contact")
            if not live[index]:
                self.complete(time, node)
        self._previous_alive = alive.clone()

    def finish(self, makespan):
        makespan = float(makespan)
        if not math.isfinite(makespan) or makespan < 0:
            raise ValueError("task progress makespan must be finite and non-negative")
        if self.events and self.events[-1]["time"] > makespan:
            raise ValueError("task events cannot occur after makespan")
        discovered = scouted = completed = 0
        total = len(self.targets)

        def point(time):
            return {
                "time": time,
                "discovered_count": discovered,
                "scouted_count": scouted,
                "completed_count": completed,
                "unknown_count": total - discovered,
                "remaining_count": total - completed,
            }

        timeline = [point(0.0)]
        for time, events in groupby(self.events, key=lambda event: event["time"]):
            for event in events:
                if event["event"] == "task_discovered":
                    discovered += 1
                    scouted += int(event["source"] == "scout")
                else:
                    completed += 1
            if timeline[-1]["time"] == time:
                timeline[-1] = point(time)
            else:
                timeline.append(point(time))
        if timeline[-1]["time"] != makespan:
            timeline.append(point(makespan))
        return {
            "schema_version": 1,
            "total_tasks": total,
            "time_units": "prepared-map traversal time",
            "timeline_convention": "counts after all task events at each timestamp",
            "discovered_count_definition": "types known through scouting, contact, or initial information",
            "scouted_count_definition": "tasks whose type was first discovered by scout sensing",
            "completed_count_definition": "successfully serviced tasks",
            "targets": self._tasks,
            "events": self.events,
            "timeline": timeline,
        }
