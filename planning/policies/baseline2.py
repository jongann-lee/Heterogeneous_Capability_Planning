"""Public entry point for baseline 2.

Baseline 2 is implemented in :mod:`planning.policies.scout_wrp`: the scout-capable
agent with the fewest positive service capabilities performs the WRP covering
walk first, then joins the capability-aware attacker layer.
"""

from planning.policies.scout_wrp import (
    DEFAULT_EXACT_TARGET_CAP,
    DEFAULT_HEDGE_TAU,
    DEFAULT_SUBOPT_WEIGHT,
    make_policy,
    solve_cover_walk,
    solve_visit_walk,
)


replan = make_policy()

__all__ = [
    "DEFAULT_EXACT_TARGET_CAP",
    "DEFAULT_HEDGE_TAU",
    "DEFAULT_SUBOPT_WEIGHT",
    "make_policy",
    "replan",
    "solve_cover_walk",
    "solve_visit_walk",
]
