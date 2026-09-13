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

Staging actions retain semantic intent through encoding. Each unknown target
keeps its configured single-target staging actions, while the closest `n`
finite pairs among `n` unknown live targets receive one safe directed minimax
staging action each. Single staging uses feature value `0.5`; pair staging uses
`1.0`. Multiple semantic actions may therefore point to the same terrain node.
The decoder combines their logits with `logsumexp`, selects one physical
destination, and enforces capacity once for that destination. Its policy log
probability and entropy are calculated over physical destinations rather than
semantic aliases. Target-directed safe distance maps, target-pair separation,
and pair minimax locations are computed once per sampled target layout; replans
only filter that scenario cache using the current live/known target beliefs.

The simulator adapter observes and jointly replans the full living team. An
agent traversing an edge must finish that edge, but its replacement route is
chosen immediately from the committed arrival node and begins on arrival. No
ground-truth graph is accepted by the observation builder or policy.

The CUDA task-graph router deduplicates independent exact route queries on the
GPU, constructs disjoint vertex-offset copies of the terrain graph with each
query's own blocked-node mask, and resolves the route bank with one cuGraph
SSSP operation. These blocked graph copies and their results are temporary.
Only the original terrain graph and bounded SSSP rows produced by the older
single-source helper may persist. This dynamic route bank is separate from the
scenario-static staging-geometry cache described above.

`learning/policy/configuration.py` loads and validates experiment settings. The
task-graph policy consumes only capabilities, normalized remaining transit
time, completion/type beliefs, action categories, normalized safe-route
distances, and typed action-target semantic relations. For every decision,
`model.edge_normalization: per_decision_zscore` computes one population mean
and standard deviation over all finite, reachable agent-target, agent-action,
and action-target edges in that episode. CUDA batches compute those statistics
independently per episode. Wait, unreachable, and padded relations are excluded.
The normalization changes neural features only; terrain edges, routing,
makespan, FI-OPT, and reports continue to use real seconds.

`candidates.allow_unknown_target_actions` controls risky direct contact with
unrevealed targets. The default `false` requires scouting first. When enabled,
only agents with a positive service capability may contact an unknown target;
pure scouts are never eligible. Known targets always require the matching
service capability. Both this toggle and the model feature-schema metadata are
saved in checkpoints, W&B configuration, and evaluation JSON. Schema-1 raw
distance checkpoints require retraining or the explicit
`--allow-feature-schema-mismatch` evaluation override. No observation path
receives absolute coordinates, heights, or ground truth.

Training returns are normalized against
`planning.full_information.full_information_makespan`, the exact FI-OPT
heterogeneous min-max open-route result. Unlike the retired optimistic metric
closure, FI-OPT returns executable target-aware routes: crossing a supported
live target records it in that agent's assignment, while targets outside the
assignment cannot be transit nodes. The logged `normalized_regret` is
`makespan / oracle_makespan - 1`; zero matches the oracle. Death and incomplete
penalties are dimensionless and applied directly after makespan normalization,
so an incomplete episode cannot exploit the oracle credit by stopping early.

Training and evaluation load a validated offline-built prepared map. The
clockwise-rotated 64x64 WV artifact remains the backward-compatible default,
but `instances.map_path` or `--map-path` may select another artifact size:

```bash
uv run python -m learning.train --episodes 100
uv run python -m learning.train --config learning/config.yaml
uv run python -m learning.train --config learning/config_transformer.yaml
uv run python -m learning.train \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz
uv run python -m learning.test learning/checkpoints/<run-timestamp> --device cuda
uv run python -m learning.test learning/checkpoints/<run-timestamp> \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --device cuda
uv run python -m learning.test --policy fi-opt --suite test
uv run python -m learning.test --policy scout-then-execute --suite test
uv run python -m learning.test learning/checkpoints/<run-timestamp> \
  --suite test --output outputs/evaluation/<checkpoint-name>.json
uv run python -m learning.analyze outputs/evaluation/<checkpoint-name>.json
uv run python -m learning.test learning/checkpoints/<run-timestamp> \
  --suite test --agent-config agents_04_b \
  --target-config targets_08_c --render
```

Evaluation always uses a validated fixed suite. `learned` is the default policy
and requires the positional checkpoint; `fi-opt` and `scout-then-execute` do
not load model weights and currently execute in the CPU simulator. FI-OPT knows
all target types at time zero. Scout-Then-Execute jointly routes every living
scout-capable agent, forces all service-only agents to wait for the last reveal,
and then invokes the same FI-OPT solver. `--episodes N` repeats each selected
case with consecutive recorded seeds. The default
`development` alias resolves to the original single RPS scenario in
`learning/evaluation_suites/wv_rps_fixed_v1.json`. The `test` alias resolves to
`wv_factorial_test_v1.json`, whose 12 agent configurations and 15 target
configurations form 180 deterministic scenarios. For each target count, the
suite contains a dispersed balanced layout, one dispersed type-heavy layout
(alternating type 1 and type 2), and a balanced multi-cluster layout. The
multi-cluster case uses two separated clusters for 5-6 targets and three for
7-9 targets, with only 2-3 targets in each cluster. CUDA
evaluation batches the three compatible agent profiles sharing a target layout
and agent count, requiring 60 tensor rollouts and 15 world constructions for
the complete suite. `--agent-config`,
`--target-config`, `--agent-count`, `--target-count`, and `--limit` filter that
Cartesian product. An explicit suite JSON path can replace either alias.
Rendering is accepted only when the final filtered selection contains exactly
one scenario. Every result records both its resolved suite ID and scenario ID.
Suite coordinates are checked against actual nodes in the selected graph, not
a fixed coordinate bound. `terrain_id` remains descriptive suite metadata.
Every evaluation writes its full JSON result to
`outputs/evaluation/<current-date-and-time>.json` by default. Pass
`--output <path>` to choose the filename or destination explicitly. The CLI
prints only the saved path instead of echoing the complete JSON payload.

`learning.analyze` keeps the full evaluation JSON as the raw reproducible
artifact and prints a compact report derived from it. The report contains
makespan mean/population-standard-deviation and completion matrices indexed by
agent and target count. Its overall section reports failure, scenarios with at
least one death, deaths per deployed agent, mean deaths, stalled and all-dead
rates, remaining targets, and raw/completed-only normalized regret. Pass
`--output <path>` to additionally save the compact aggregate as JSON; it will
not overwrite an existing file.

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

The saved run configuration also records the resolved artifact path, prepared
map schema, serialized-content SHA-256, dimensions, node/edge counts, and build
metadata. Evaluation map precedence is explicit `--map-path`, an explicitly
selected config, the checkpoint config, then the default WV artifact. This
allows an rsynced map to be relocated with `--map-path`: matching content is
accepted by hash. A different hash fails learned evaluation by default;
`--allow-map-mismatch` is the deliberate override and is recorded in result
JSON. Classical policies have no checkpoint-map hash requirement.

For research runs, import `learning.train.train` and supply an
`instance_factory(episode)` returning fresh `(env_map, ground_truth, agents)`
objects. The Transformer uses complete episodic REINFORCE with an EMA baseline;
the default task graph now uses the same baseline approach. Set
`model.use_critic: true` to restore its shared state-conditioned critic.
Neither path depends on Gym or another RL framework.

Although this change preserves model tensor dimensions, checkpoints trained
before semantic staging and physical-location decoding should be treated as a
different policy version and retrained for performance comparisons.

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
RUN=2026-09-11_14-58-56_040053
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
