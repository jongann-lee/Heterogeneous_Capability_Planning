"""Mutable execution state for a simulation agent."""

from simulation.domain import ROCK, SCOUT, validate_capabilities


class Agent:
    """An agent's mutable state during one simulation episode.

    Capability ``0`` grants scouting. Positive capabilities identify target
    types the agent can service. ``agent_type`` remains as a deprecated
    singleton-capability shorthand for older experiment code.
    """

    def __init__(self, position, capabilities=None, agent_type: int | None = None,
                 cost_multiplier: float = 1.0, movement_modifier: int = 1):
        self.position = position
        if capabilities is not None and agent_type is not None:
            raise ValueError("pass capabilities or agent_type, not both")
        if capabilities is None:
            shorthand = ROCK if agent_type is None else int(agent_type)
            capabilities = {shorthand}
        self.capabilities = validate_capabilities(capabilities)

        # Temporary compatibility field for legacy policies and renderers.
        positive = sorted(value for value in self.capabilities if value > 0)
        self.agent_type = (
            int(agent_type) if agent_type is not None
            else (positive[0]
                  if len(positive) == 1 and SCOUT not in self.capabilities
                  else SCOUT if self.capabilities == {SCOUT} else None)
        )
        self.alive = True
        self.cost_multiplier = float(cost_multiplier)
        self.movement_modifier = int(movement_modifier)
        self.total_traversal_cost = 0.0
        self.trajectory = [position]
        self.planned_path: list = []

    @property
    def can_engage(self) -> bool:
        return any(value > 0 for value in self.capabilities)

    @property
    def scout_capable(self) -> bool:
        return SCOUT in self.capabilities

    def can_service(self, target_type: int) -> bool:
        return target_type > 0 and target_type in self.capabilities

    def move(self, from_node, to_node, cost: float):
        if not self.alive:
            return
        if from_node != self.position:
            raise ValueError(
                f"Agent is at {self.position} but move() was called with "
                f"from_node={from_node}"
            )
        self.position = to_node
        self.total_traversal_cost += cost * self.cost_multiplier
        self.trajectory.append(to_node)
        if self.planned_path and self.planned_path[0] == from_node:
            self.planned_path = self.planned_path[1:]
