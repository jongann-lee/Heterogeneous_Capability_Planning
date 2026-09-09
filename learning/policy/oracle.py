"""Compatibility imports for the executable full-information oracle."""

from planning.full_information import full_information_makespan


# Existing training/evaluation imports keep working while now using FI-OPT.
parallel_tsp = full_information_makespan

__all__ = ["full_information_makespan", "parallel_tsp"]
