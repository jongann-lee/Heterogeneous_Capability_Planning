# Learning package

This package implements the centralized learned planner described in
`IMPLEMENTATION_PLAN.md`. It uses deterministic candidate generation, padded
planner-visible observations, agent/target/action attention, an explicit
agent-action pointer head, and constrained autoregressive joint assignment.

The simulator adapter observes the full living team but only changes routes for
agents currently at nodes. Agents traversing an edge are restricted to the
`CONTINUE_CURRENT_ROUTE` action. No ground-truth graph is accepted by the
observation builder or policy.

Experiment settings live in `learning/config.yaml`; `learning/configuration.py`
only loads and validates that file. The attention stack uses PyTorch's
`TransformerDecoderLayer` for self-attention, cross-attention, residuals,
normalization, and feed-forward processing.

Quick smoke training and greedy evaluation on the included fixed grid:

```bash
uv run python -m learning.train --episodes 100 --checkpoint learning_policy.pt
uv run python -m learning.train --config learning/config.yaml
uv run python -m learning.evaluate learning_policy.pt
```

For research runs, import `learning.train.train` and supply an
`instance_factory(episode)` returning fresh `(env_map, ground_truth, agents)`
objects. The code uses complete episodic REINFORCE with an EMA baseline and has
no Gym or RL-framework dependency.

Fast checks:

```bash
uv run python -m tests.test_learning
uv run python -m tests.test_simulation
```
