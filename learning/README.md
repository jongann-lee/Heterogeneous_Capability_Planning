# Learning package

This package implements the centralized learned planner described in
`IMPLEMENTATION_PLAN.md`. It uses deterministic candidate generation, padded
planner-visible observations, agent/target/action attention, an explicit
agent-action pointer head, and constrained autoregressive joint assignment.

The simulator adapter observes and jointly replans the full living team. An
agent traversing an edge must finish that edge, but its replacement route is
chosen immediately from the committed arrival node and begins on arrival. No
ground-truth graph is accepted by the observation builder or policy.

Experiment settings live in `learning/config.yaml`; `learning/configuration.py`
only loads and validates that file. The attention stack uses PyTorch's
`TransformerDecoderLayer` for self-attention, cross-attention, residuals,
normalization, and feed-forward processing.

Training returns are normalized against `learning.orcale.parallel_tsp`, a
full-information min-max open-TSP oracle over the terrain's shortest-path
metric closure. The logged `normalized_regret` is
`makespan / oracle_makespan - 1`; zero matches the oracle. Death and incomplete
penalties are dimensionless and applied directly after makespan normalization,
so an incomplete episode cannot exploit the oracle credit by stopping early.

Training and greedy evaluation use the clockwise-rotated 64x64 WV DEM:

```bash
uv run python -m learning.train --episodes 100
uv run python -m learning.train --config learning/config.yaml
uv run python -m learning.test learning/checkpoints/<run-timestamp> --device cuda
uv run python -m learning.test learning/checkpoints/<run-timestamp> --device cuda --render
```

The optional test renderer records the deterministic CUDA tensor rollout and
draws its event trace afterward; it does not recompute the policy or routes on
the CPU. Matching the non-learning benchmark defaults exactly, it writes
interpolated frames to `outputs/my_policy_simulation/frames/`, creates
`outputs/my_policy_simulation/render_result.mp4`, samples every 1.0 simulation
time unit, and encodes at 4 FPS. Use
`--render-dt` to control the simulation-time spacing between frames and
`--mp4-fps` to control video playback speed. PNG drawing and video encoding
remain CPU-side, but simulation and policy inference stay on CUDA.

Each training run creates a timestamped directory beneath
`learning/checkpoints/` containing separate `trained_weights.pt` and
`config.yaml` files. Pass `--checkpoint-dir` to use a different parent
directory. With `training.wandb: true`, the same resolved configuration and
one aggregate record per REINFORCE optimizer update are logged to the
`heterogeneous-capability-planning` Weights & Biases project. Metrics include
mean return, mean policy loss, mean makespan, and completion rate across the
update's episode batch. Failure diagnostics include mean deaths, mean remaining
targets, stalled rate, and all-agents-dead rate.

For research runs, import `learning.train.train` and supply an
`instance_factory(episode)` returning fresh `(env_map, ground_truth, agents)`
objects. The code uses complete episodic REINFORCE with an EMA baseline and has
no Gym or RL-framework dependency.

Fast checks:

```bash
uv run python -m tests.test_learning
uv run python -m tests.test_simulation
```
