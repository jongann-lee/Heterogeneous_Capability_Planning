"""Capability-based target interaction primitives.

The operative model is general:

* target types are positive integers ``1..n``;
* an agent owns a set of integer capabilities;
* capability ``0`` means the agent can scout;
* a positive capability ``k`` means the agent can service target type ``k``;
* contacting a supported target succeeds, while contacting an unsupported
  target kills the agent and leaves the target active.

The graph attribute ``rps_type``, old symbolic constants, and :func:`beats`
remain temporarily so older experiment artifacts can still load. The simulator
does not use the cyclic RPS rule.
"""

import random
from collections.abc import Iterable


SCOUT = 0
UNKNOWN_TYPE = -1

# Legacy symbolic names retained for old notebooks and policies.
ROCK = 1
SCISSOR = 2
PAPER = 3
COMBAT_TYPES = (ROCK, SCISSOR, PAPER)
TARGET_TYPES = (ROCK, SCISSOR, PAPER)
AGENT_TYPES = (SCOUT, ROCK, SCISSOR, PAPER)
TYPE_NAMES = {
    SCOUT: "scout",
    ROCK: "type-1",
    SCISSOR: "type-2",
    PAPER: "type-3",
    UNKNOWN_TYPE: "unknown",
}


AGENT_WINS = "agent_wins"
AGENT_DIES = "agent_dies"

# Kept only so legacy policy modules import cleanly.  Generalized encounters
# never produce a draw.
DRAW = "draw"


def target_type_name(target_type: int) -> str:
    """Return a display label for a dynamic target type."""
    if target_type == UNKNOWN_TYPE:
        return "unknown"
    return f"type-{target_type}"


def capability_label(capabilities: Iterable[int]) -> str:
    """Compact, deterministic display label for a capability set."""
    values = sorted(set(capabilities))
    return "{" + ",".join("scout" if value == SCOUT else str(value)
                          for value in values) + "}"


def validate_capabilities(capabilities: Iterable[int],
                          num_target_types: int | None = None) -> frozenset[int]:
    """Validate and normalize an agent capability set."""
    values = frozenset(int(value) for value in capabilities)
    if any(value < 0 for value in values):
        raise ValueError("agent capabilities must be non-negative integers")
    if num_target_types is not None:
        if num_target_types < 1:
            raise ValueError("num_target_types must be at least 1")
        invalid = sorted(value for value in values if value > num_target_types)
        if invalid:
            raise ValueError(
                f"capabilities {invalid} exceed target type range 1..{num_target_types}"
            )
    return values


def can_service(capabilities: Iterable[int], target_type: int) -> bool:
    """Whether ``capabilities`` contains the positive ``target_type``."""
    return target_type > 0 and target_type in capabilities


def resolve_encounter(capabilities, target_type: int) -> str:
    """Resolve contact using binary capability matching.

    ``capabilities`` is normally an iterable.  A single integer is accepted as
    a temporary compatibility shorthand and interpreted as a singleton set.
    """
    if isinstance(capabilities, int):
        capabilities = {capabilities}
    return AGENT_WINS if can_service(capabilities, target_type) else AGENT_DIES


def assign_target_types(targets, num_target_types: int = 3, rng=None) -> dict:
    """Assign reproducible random target types from ``1..num_target_types``."""
    if num_target_types < 1:
        raise ValueError("num_target_types must be at least 1")
    chooser = rng or random
    return {target: chooser.randint(1, num_target_types) for target in targets}


def assign_agent_capabilities(
        num_agents: int,
        num_target_types: int,
        capability_probability: float = 0.5,
        scout_probability: float = 0.25,
        ensure_target_coverage: bool = True,
        ensure_scout: bool = True,
        rng=None) -> list[frozenset[int]]:
    """Generate a reproducible random capability subset for every agent.

    Positive target capabilities and scouting are sampled independently.
    With the default safeguards, every target type is supported by at least
    one agent and at least one agent can scout.
    """
    if num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    if num_target_types < 1:
        raise ValueError("num_target_types must be at least 1")
    for name, value in (
        ("capability_probability", capability_probability),
        ("scout_probability", scout_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")

    generator = rng or random
    capabilities = []
    for _ in range(num_agents):
        values = {
            target_type
            for target_type in range(1, num_target_types + 1)
            if generator.random() < capability_probability
        }
        if generator.random() < scout_probability:
            values.add(SCOUT)
        capabilities.append(values)

    if ensure_target_coverage:
        for target_type in range(1, num_target_types + 1):
            if not any(target_type in values for values in capabilities):
                capabilities[generator.randrange(num_agents)].add(target_type)

    if ensure_scout and not any(SCOUT in values for values in capabilities):
        capabilities[generator.randrange(num_agents)].add(SCOUT)

    return [frozenset(values) for values in capabilities]


def init_target_types(env_map, ground_truth, target_types) -> None:
    """Store truth on the ground-truth graph and unknowns on planner targets."""
    for target, true_type in target_types.items():
        if true_type < 1:
            raise ValueError("target types must be positive integers")
        if ground_truth.has_node(target):
            ground_truth.nodes[target]["rps_type"] = true_type
        if env_map.has_node(target):
            env_map.nodes[target]["rps_type"] = UNKNOWN_TYPE


def beats(a: int, b: int) -> bool:
    """Legacy RPS helper retained temporarily for old policy imports."""
    return b == (a % 3) + 1
