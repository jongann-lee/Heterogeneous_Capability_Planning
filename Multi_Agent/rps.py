"""
Rock-Scissor-Paper model primitives for the multi-agent simulation.

This is the *mechanics* layer for the `rock_scissor_paper` research direction.
It is deliberately policy-free: it defines the type ids, the cyclic-dominance
combat rule, and small helpers for stamping target types onto the planner /
ground-truth graphs. The assignment policy lives elsewhere (see
rps_simulation.naive_type_aware_replan for a placeholder).

Type ids
--------
Agents and targets carry an integer type:

    SCOUT   = 0   (agents only) -- long-range visibility, CANNOT engage targets
    ROCK    = 1
    SCISSOR = 2
    PAPER   = 3

Targets are always one of {ROCK, SCISSOR, PAPER}. The planner may not know a
target's type yet: UNKNOWN_TYPE (-1) is the "not yet revealed" sentinel that
sits on env_map targets until an agent observes them (mirrors the
UNKNOWN_LOCK sentinel from the fixed_key branch).

Combat (cyclic dominance, exactly like the game)
------------------------------------------------
ROCK(1) beats SCISSOR(2) beats PAPER(3) beats ROCK(1) -- each id beats the
*next*, with 3 wrapping back to 1:

    a beats b   iff   b == (a % 3) + 1

When a combat agent steps onto a *live* target:

    same type           -> DRAW       (target and agent both unchanged)
    agent beats target   -> AGENT_WINS (target eliminated, agent survives)
    target beats agent   -> AGENT_DIES (agent removed, target unchanged)

A SCOUT that steps onto a live target always dies -- it cannot fight back.
"""

import random


# --- Type ids -------------------------------------------------------------
SCOUT = 0
ROCK = 1
SCISSOR = 2
PAPER = 3

#: Sentinel for a target whose type the planner has not yet revealed.
UNKNOWN_TYPE = -1

#: Agent types that can engage targets.
COMBAT_TYPES = (ROCK, SCISSOR, PAPER)
#: Types a target may take.
TARGET_TYPES = (ROCK, SCISSOR, PAPER)
#: All valid agent types.
AGENT_TYPES = (SCOUT, ROCK, SCISSOR, PAPER)

TYPE_NAMES = {
    SCOUT: "scout",
    ROCK: "rock",
    SCISSOR: "scissor",
    PAPER: "paper",
    UNKNOWN_TYPE: "unknown",
}


# --- Combat outcomes ------------------------------------------------------
DRAW = "draw"
AGENT_WINS = "agent_wins"
AGENT_DIES = "agent_dies"


def beats(a: int, b: int) -> bool:
    """True iff RPS type ``a`` beats type ``b``.

    Both must be in {ROCK, SCISSOR, PAPER}. Encodes the cycle
    ROCK->SCISSOR->PAPER->ROCK as ``b == (a % 3) + 1``.
    """
    return b == (a % 3) + 1


def resolve_encounter(agent_type: int, target_type: int) -> str:
    """Outcome when ``agent_type`` steps onto a live target of ``target_type``.

    Returns one of :data:`DRAW`, :data:`AGENT_WINS`, :data:`AGENT_DIES`.
    A scout never wins or draws -- it always dies on a live target.
    """
    if agent_type == SCOUT:
        return AGENT_DIES
    if agent_type == target_type:
        return DRAW
    if beats(agent_type, target_type):
        return AGENT_WINS
    return AGENT_DIES


def assign_target_types(targets, types=TARGET_TYPES, rng=None) -> dict:
    """Assign each target a (random) RPS type.

    Args:
        targets: iterable of target node ids.
        types: pool of types to draw from (default rock/scissor/paper).
        rng: optional ``random.Random`` instance; defaults to the module-level
            ``random`` so a single ``random.seed(...)`` in ``main()`` makes the
            assignment reproducible (see CLAUDE.md seeding convention).

    Returns:
        ``{target_node: rps_type}``.
    """
    choose = (rng or random).choice
    pool = list(types)
    return {t: choose(pool) for t in targets}


def init_target_types(env_map, ground_truth, target_types) -> None:
    """Stamp the RPS types onto the two graphs (in place).

    Ground truth gets the *real* type on every target; the planner's env_map
    gets :data:`UNKNOWN_TYPE` so types must be discovered by observation.

    Args:
        env_map: planner's view (gets ``rps_type = UNKNOWN_TYPE`` on targets).
        ground_truth: true graph (gets the real ``rps_type`` on targets).
        target_types: ``{target_node: rps_type}`` from :func:`assign_target_types`.
    """
    for t, true_type in target_types.items():
        if ground_truth.has_node(t):
            ground_truth.nodes[t]["rps_type"] = true_type
        if env_map.has_node(t):
            env_map.nodes[t]["rps_type"] = UNKNOWN_TYPE
