# Learning package

This package provides two centralized learned planners over the same
deterministic candidates, planner-visible routes, and constrained
autoregressive joint assignment:

- `task_graph` (the default in `learning/config.yaml`) is a typed,
  edge-conditioned heterogeneous GNN over agent, target, and action nodes.
  Its graph critic is implemented but currently disabled in the default
  configuration, so training uses the scalar EMA REINFORCE baseline.
- `transformer` is the original set-attention control architecture. Its
  preserved configuration is `learning/config_transformer.yaml`; it retains
  the scalar EMA REINFORCE baseline for checkpoint compatibility.

Set `model.architecture` in a configuration file to select the implementation.
The task graph is not the terrain graph: routing, visibility, candidate
generation, and partial-observability updates remain outside the network.

The simulator adapter observes and jointly replans the full living team. An
agent traversing an edge must finish that edge, but its replacement route is
chosen immediately from the committed arrival node and begins on arrival. No
ground-truth graph is accepted by the observation builder or policy.

The CUDA task-graph router deduplicates independent exact route queries on the
GPU, constructs disjoint vertex-offset copies of the terrain graph with each
query's own blocked-node mask, and resolves the route bank with one cuGraph
SSSP operation. These blocked graph copies and their results are temporary.
Only the original terrain graph and bounded SSSP rows produced by the older
single-source helper may persist.

`learning/policy/configuration.py` loads and validates experiment settings. The
task-graph policy consumes only capabilities, raw remaining transit time,
completion/type beliefs, action categories, raw safe-route distances, and
typed action-target semantic relations. It receives no absolute coordinates,
heights, or ground truth.

Training returns are normalized against `learning.policy.oracle.parallel_tsp`, a
full-information min-max open-TSP oracle over the terrain's shortest-path
metric closure. The logged `normalized_regret` is
`makespan / oracle_makespan - 1`; zero matches the oracle. Death and incomplete
penalties are dimensionless and applied directly after makespan normalization,
so an incomplete episode cannot exploit the oracle credit by stopping early.

Training and greedy evaluation use the clockwise-rotated 64x64 WV DEM:

```bash
uv run python -m learning.train --episodes 100
uv run python -m learning.train --config learning/config.yaml
uv run python -m learning.train --config learning/config_transformer.yaml
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
`learning/checkpoints/`. It writes `config.yaml` immediately, refreshes
`latest_weights.pt` after every optimizer update, and updates
`best_weights.pt` whenever mean episodic return improves. The associated
progress is recorded in `checkpoint_state.yaml`; `trained_weights.pt` remains
the final snapshot for compatibility. Pass `--checkpoint-dir` to use a
different parent directory. Evaluation accepts any of these weight files
directly; a run directory prefers the final snapshot when it exists and falls
back to the rolling snapshots during an active run. With
`training.wandb: true`, the same resolved
configuration and one aggregate record per optimizer update are logged to the
`heterogeneous-capability-planning` Weights & Biases project. Metrics include
mean return, mean policy loss, mean makespan, and completion rate across the
update's episode batch. Failure diagnostics include mean deaths, mean remaining
targets, stalled rate, and all-agents-dead rate.

For research runs, import `learning.train.train` and supply an
`instance_factory(episode)` returning fresh `(env_map, ground_truth, agents)`
objects. The Transformer uses complete episodic REINFORCE with an EMA baseline;
the default task graph now uses the same baseline approach. Set
`model.use_critic: true` to restore its shared state-conditioned critic.
Neither path depends on Gym or another RL framework.

Fast checks:

```bash
uv run python -m tests.test_learning
uv run python -m tests.test_simulation
```

## Synchronizing with DeltaAI

The DeltaAI repository is expected at:

```text
/projects/bhmi/jongann2/Heterogeneous_Capability_Planning/
```

### Send desktop code to DeltaAI

Run this from the desktop repository root. It updates source code and cluster
scripts without copying generated data or overwriting DeltaAI's environment,
checkpoints, logs, W&B data, or persistent preprocessing cache. The command
intentionally omits `--delete`, so files that exist only on DeltaAI are
preserved.

```bash
rsync -avP --itemize-changes \
  --exclude='/.git/' \
  --exclude='/.venv/' \
  --exclude='/.uv-cache/' \
  --exclude='/.cluster-cache/' \
  --exclude='/learning/checkpoints/' \
  --exclude='/outputs/' \
  --exclude='/logs/' \
  --exclude='/wandb/' \
  --exclude='/cache/' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='.DS_Store' \
  ./ \
  Delta_AI:/projects/bhmi/jongann2/Heterogeneous_Capability_Planning/
```

If `pyproject.toml` or `uv.lock` changed, rebuild/synchronize the DeltaAI
environment from a GPU allocation with `cluster/delta/HCP_build_env.sh`. Do not
run the CUDA environment verification or training on a login node.

### Pull one checkpoint run back to the desktop

First identify the desired timestamped run on DeltaAI:

```bash
ssh Delta_AI \
  'ls -dt /projects/bhmi/jongann2/Heterogeneous_Capability_Planning/learning/checkpoints/20* | head -1'
```

Then, from the desktop repository root, set `RUN` to that directory's basename
and pull only its checkpoint results. The W&B directory is excluded because it
may be large and may still be receiving writes during an active run.

```bash
RUN=2026-08-26_17-02-11_300431
mkdir -p "learning/checkpoints/${RUN}"

rsync -avP \
  --exclude='wandb/' \
  --exclude='*.tmp' \
  Delta_AI:/projects/bhmi/jongann2/Heterogeneous_Capability_Planning/learning/checkpoints/"${RUN}"/ \
  learning/checkpoints/"${RUN}"/
```

Rolling `latest_weights.pt` and `best_weights.pt` files are replaced atomically,
so this result command can be rerun safely while training continues. The final
`trained_weights.pt` appears only after training completes.
