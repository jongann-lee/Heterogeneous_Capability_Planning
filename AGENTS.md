# AGENTS.md

Repository guide for coding agents. This describes the code currently on
`main`; verify the branch and implementation after a branch switch.

## Project

This repository studies centralized planning for heterogeneous agents on an
uncertain, weighted terrain graph. Target locations are known but their
positive integer types are initially hidden. Agent capabilities are integers:

- `0` permits scouting (edge/blockage sensing and target-type revelation);
- positive `k` permits servicing target type `k`;
- contacting an unsupported live target kills the agent and leaves the target
  active.

The simulator objective is mission makespan plus a configurable death penalty.
The learning objective normalizes makespan by the executable FI-OPT
heterogeneous min-max makespan and adds dimensionless death and incompletion
penalties. Logs and result JSON retain `oracle_makespan` as a compatibility
name and also expose the same value as `fi_opt_makespan`.

The package is `heterogeneous-capability-planning`, requires Python 3.12 or
3.13, and has a working root `main.py` for the real-map benchmark.

## Source map

- `simulation/`
  - `agent.py`: mutable per-episode `Agent` state. Prefer `capabilities`;
    singleton `agent_type` exists only for legacy compatibility.
  - `domain.py`: capability validation/generation, target-type initialization,
    and encounters. `rps_type`, `ROCK`, and `beats` are compatibility remnants;
    the active model is direct capability matching, not cyclic RPS.
  - `engine.py`: map-independent continuous-time discrete-event simulator and
    safe placeholder policy.
  - `real_map_benchmark.py`: loads a prepared real-terrain map, applies episode
    overlays, selects a baseline, runs it, and optionally writes JSON, CSV,
    PNG, and MP4 output.
  - `rendering.py`: visualization and ffmpeg integration, kept outside the core
    engine import path.
- `planning/`
  - `full_information.py`: exact heterogeneous min-max open-route solver. Its
    per-agent Dijkstra state is `(physical node, serviced-target mask)`, so it
    exposes consistent scalar and reconstructed assignment/order/path results.
    `solve_full_information` returns a diagnostic plan (including infeasible
    plans); `full_information_makespan` is the scalar training interface and
    raises `FullInformationInfeasibleError` when no plan exists.
  - `policies/baseline1.py`: independent distance routing. Service agents
    prefer supported, unknown, then unsupported targets; pure scouts move to
    the tallest safe node.
  - `policies/baseline2.py` / `scout_wrp.py`: assigns the least
    service-capable scout a Watchman Route Problem covering walk while the
    other agents use the PTSP-style claim/hedge/probe attacker layer. The
    baseline-1 attacker layer remains an explicit ablation option. Exact A* is
    used through 12 scoutable unknown targets, with weighted A* above that
    threshold.
  - `policies/scout_then_execute.py`: strict cooperative two-phase benchmark.
    All scouts minimize the final reveal time while service-only agents wait;
    the execution phase then commits to FI-OPT, including transit release
    offsets at the phase boundary. Cooperative coverage is exact through 12
    unknown targets; larger cases use greedy scout assignment followed by
    weighted-A* covering walks and report `exact=False`.
  - `finite_horizon.py`: older reward-driven Hungarian and sequential-greedy
    comparison planners; these are not benchmark defaults.
  - `legacy/`: retained comparison code, not an active entry point.
- `learning/`
  - `config.yaml`: task-graph defaults with the optional critic currently
    disabled. The original Transformer defaults are preserved in
    `config_transformer.yaml`.
    `policy/configuration.py` loads and validates both; CLI flags override
    common training fields.
  - `modules/` and `policy/model.py`: selectable Transformer or typed
    heterogeneous graph actor, optional graph critic, and constrained joint
    decoder. Semantic action logits are marginalized by physical destination
    before assignment and capacity enforcement.
  - `policy/candidates.py`: deterministic target, observation, single-target
    staging, closest-pair minimax staging, and wait candidates. Semantic and
    physical candidate identities are intentionally distinct.
  - `gpu_sim/observation_cpu.py`: canonical planner-visible feature builder and
    CPU batching despite the historical module path.
  - `policy/adapter.py` and `gpu_sim/rollout_cpu.py`: simulator-backed policy
    and CPU rollout/training path.
  - `gpu_sim/world.py`, `state.py`, `observation_gpu.py`, `rollout_gpu.py`, and
    `cugraph_router.py`: batched CUDA simulation and cuGraph routing path.
    Independent exact route queries are deduplicated on-device and combined
    as disjoint graph copies for one GPU SSSP operation.
  - `policy/oracle.py`: compatibility import for the FI-OPT normalization
    scalar in `planning/full_information.py`.
  - `train.py`: CPU/CUDA REINFORCE training. Timestamped checkpoints contain
    the resolved config, rolling latest/best-return weights, progress metadata,
    and final `trained_weights.pt`.
  - `evaluation_suite.py` and `evaluation_suites/`: validated factorized fixed
    evaluation definitions. `development` is the original one-case RPS setup;
    `test` is the 12-by-15, 180-case WV factorial suite.
  - `test.py`: deterministic suite-based learned, FI-OPT, and strict
    Scout-Then-Execute evaluation with optional single-scenario rendering.
    Only the learned mode requires a checkpoint. `policy/evaluation.py` is a
    smaller legacy interface.
  - `analyze.py`: compact post-processing for full evaluation JSON, including
    agent-count/target-count matrices and overall failure/death diagnostics.
- `Graph_Generation/`: visibility, blockage, target-graph, and stochastic
  diverse-path helpers used by older planners.
- `Real_Life_Maps/`: bundled `WV_DEM.tif` and `WV_roads.pkl`, the explicit
  offline `build_map.py` compiler, prepared-map serialization, GRASS viewshed
  worker, and retained older benchmark scripts. The compiler derives directed
  Tobler travel times on a dense physical raster, computes node-first GRASS
  viewsheds, induces edge visibility, and writes the target-independent map
  consumed by simulation and learning.
- `Single_Agent/`: original reward-driven implementation and TSP solver,
  retained as dependencies/comparisons.
- `tests/`: executable synthetic regressions for simulation and learning.
- `outputs/` and `learning/checkpoints/`: ignored generated artifacts. Never
  delete or overwrite them without checking with the user.

`NEW_REPOSITORY_HANDOFF.md`, `learning/IMPLEMENTATION_PLAN.md`, and
`learning/IMPLEMENTATION_REPORT.md` are design/history documents. They are
useful context but are not authoritative when they disagree with code or tests.
The root notebook and older `Real_Life_Maps/` scripts are also exploratory or
legacy.

## Non-negotiable simulation contracts

### Partial observability uses two graphs

`env_map` is the planner's optimistic, partially observed graph;
`ground_truth` contains true target types and traversable edges. Types start as
`UNKNOWN_TYPE` (`-1`) in `env_map`. Only sensing and contact may copy facts from
`ground_truth` into `env_map`. Never pass ground truth to a learned or ordinary
partially observed policy, candidate generator, or observation builder.
FI-OPT is the deliberate exception: its normalization scalar and `fi-opt`
benchmark receive the truth graph by definition. Strict Scout-Then-Execute may
not use truth during scouting; its execution solver receives `env_map` only
after every remaining target type has been revealed.

Active graph conventions:

- node `type` includes `source`, `intermediate`, `target_unreached`, and
  `target_reached`;
- target types use the legacy `rps_type` attribute;
- nodes expose `visible_edges`; a scout sees their true endpoints plus itself;
- directed edges carry `distance` and commonly `observed_edge`; grid instances
  usually contain both orientations, and uphill/downhill costs may differ.

### Time, policies, and routes

`run_simulation` is event-driven. Edge `distance` is traversal time. Agents
observe and interact on node arrival, and an agent already traversing an edge
must reach its committed next node. Normal policies receive living agents at
nodes. A learned-style policy may set `replan_in_transit = True` and implement
`set_runtime_state(...)`; a moving agent's replacement route must begin at its
committed arrival node.

A policy has the shape `policy(env_map, agents, reward_ratio=...,
obs_discount_factor=..., sample_recursion=..., sample_num_obstacle=...,
sample_obstacle_hop=..., verbose=...)` and mutates each supplied agent's
`planned_path`. Paths include their start node. Preserve the route selected or
scored by a planner: do not replace it later with a fresh `nx.shortest_path`.
This matters especially for diverse-path routines in `finite_horizon.py`.

Use `simulation.domain` helpers rather than reimplementing encounters.
Capability sets may be empty, pure-scout, pure-service, or hybrid. When adding
randomness, seed Python `random`, NumPy, and PyTorch as applicable, and prefer a
passed RNG where the API supports one.

## Learning invariants

- Observations contain only belief-state information; the builder intentionally
  accepts no truth graph.
- Agent, target, and candidate counts vary. Masks/padding must remain correct
  and permutation equivariant.
- Constrained decoding enforces feasibility and candidate capacity.
- Staging aliases at one node remain distinct graph actions but share one
  physical decoder group. Group logits use masked `logsumexp`; log probability,
  entropy, and finite capacity are defined over physical groups, not aliases.
- Single-target staging has arity feature `0.5`; pair staging has `1.0`. Among
  `n` unknown live targets, only the closest `n` finite pairs are active, using
  conservative directed separation and Candidate-to-Target minimax locations.
- Incomplete episodes need an explicit penalty so stopping early cannot beat
  completion.
- Keep CPU and tensor/CUDA transition semantics aligned. The CUDA path batches
  world state, routing, rollout, and gradient replay, not just inference.
- CUDA task-graph routing batches independent exact queries as disjoint,
  vertex-offset terrain copies connected to one super-source. Blocked copies
  and their SSSP results are temporary and must not persist after the route
  bank is consumed. The older single-source helper may persistently cache only
  the original terrain graph and bounded SSSP rows computed on it.
- Candidate staging geometry is scenario-static: compute target-directed safe
  distance maps, pair separation, and pair minimax locations once per sampled
  target layout, then only filter cached definitions as target beliefs change.
- Real terrain is an explicit offline-built artifact. The backward-compatible
  default is `Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz`; simulation,
  training, and evaluation do not resample the DEM or calculate visibility.
  Re-run `python -m Real_Life_Maps.build_map` deliberately to replace or create
  a map. Node visibility is primary, and visible edges are induced only when
  both endpoints are visible.
- `learning.train` and `learning.test` must support selecting a prepared map
  through both configuration and `--map-path`, with the CLI taking precedence.
  Do not hard-code a 64x64 terrain size, WV-only identifier, or WV-only factory
  in those paths. Derive valid nodes and dimensions from the validated artifact,
  and validate suite source/target positions by graph membership.
- Prepared maps may differ in size but must share one validated schema. Nodes
  expose `pos`, `height`, `elevation_m`, `type`, `visible_nodes`, and
  `visible_edges`; directed edges expose `distance`, `is_road`,
  `observed_edge`, and `num_used`. Preserve additional diagnostic attributes.
  A checkpoint records the selected map identity, content hash, and relevant
  metadata. Evaluation must detect a checkpoint/map mismatch and require an
  explicit override rather than silently using different terrain.
- `simulation_batch_size` controls simultaneous tensor episodes;
  `reinforce_batch_size` controls optimizer accumulation. Legacy configs with
  `batch_size` map it to both fields.
- Learned evaluation accepts a checkpoint run directory or weights path and
  normally uses the configuration saved beside the weights. FI-OPT and
  Scout-Then-Execute need no checkpoint and always use the CPU event-driven
  simulator; `--device` and `--cuda` apply only to the learned policy.
  A relocated prepared map is accepted when its content hash matches the
  checkpoint; `--allow-map-mismatch` is required and recorded for deliberate
  learned-policy evaluation on different map content.
  Evaluation JSON is written by default to a timestamped file under
  `outputs/evaluation`; `--output` selects an explicit path.
  Evaluation defaults to the `development` suite; rendering is valid only
  after filters select one case. Non-rendering learned CUDA evaluation batches
  cases only when they share a target world and agent count, while reusing each
  target world across those batches.
- FI-OPT routes are target-aware and independently executable: a supported
  target encountered on a route belongs to that agent, and a target outside
  its assignment is never used as a transit node. It deliberately does not
  model one agent clearing an unsupported target for another agent to cross
  later. Do not replace this with an ordinary all-node shortest-path metric
  closure or describe it as unrestricted joint temporal optimality.
- FI-OPT is exponential in the number of live targets: the per-agent physical
  product search scales with target masks and the outer assignment DP
  enumerates compatible subsets. The built-in suites cap target counts at 9.
  Preserve the scalar no-reconstruction path used once per training instance.
- A targetless FI-OPT instance returns zero. Reward code permits a zero
  denominator only when both the episode and oracle are targetless; other zero
  oracle values are errors. `parallel_tsp` remains only as a compatibility
  alias for `full_information_makespan`.
- Strict Scout-Then-Execute uses only planner-visible state during scouting.
  Every scout-capable agent may participate, service-only agents wait, and the
  final reveal triggers one FI-OPT plan over remaining targets. Moving agents
  use committed destinations as starts and remaining edge times as releases.
  Targets without a reachable safe visibility witness make scouting
  explicitly infeasible; the policy never substitutes sacrificial contact.
- Evaluation suite JSON is a validated Cartesian product of agent and target
  configurations. Every agent configuration must contain a scout and cover
  all positive target types. Filters preserve file order, and `--episodes N`
  repeats each selected fixed case rather than sampling new layouts.

## Environment and commands

Use `uv`; system Python is not expected to have the dependencies. The lockfile
includes PyTorch 2.12 and CUDA 13 cuGraph. CUDA training needs a compatible
NVIDIA/RAPIDS environment, while the CPU simulator and synthetic checks do not
need to run the CUDA backend.

```bash
uv sync

# Explicit offline map compilation (requires GRASS GIS)
uv run python -m Real_Life_Maps.build_map

# Fast regression suites
uv run python -m tests.test_simulation
uv run python -m tests.test_learning
uv run python -m tests.test_real_map_builder

# Real-map benchmark (equivalent root entry: uv run python main.py)
uv run python -m simulation.real_map_benchmark --help
uv run python -m simulation.real_map_benchmark --policy baseline1 --seed 0
uv run python -m simulation.real_map_benchmark --policy baseline2 --render

# Learned policy
uv run python -m learning.train --config learning/config.yaml --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --episodes 100 --device cpu
uv run python -m learning.test learning/checkpoints/<run-directory> --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --device cuda
uv run python -m learning.test learning/checkpoints/<run-directory> --suite test --device cuda
uv run python -m learning.test learning/checkpoints/<run-directory> --suite development --device cuda --render

# Classical benchmark policies (CPU; no checkpoint)
uv run python -m learning.test --policy fi-opt --suite test --output outputs/evaluation/fi-opt.json
uv run python -m learning.test --policy scout-then-execute --suite test --output outputs/evaluation/scout-then-execute.json

# Fixed-suite analysis and a filtered classical render
uv run python -m learning.analyze outputs/evaluation/<result>.json
uv run python -m learning.test --policy scout-then-execute --suite test --agent-config agents_04_b --target-config targets_08_c --render
```

The default learning config currently uses 3 target types, 5-9 targets, and
3-6 randomly generated agents whose capabilities collectively cover every
target type and include at least one scout. It uses CUDA and Weights & Biases
logging. Pair staging and wait actions are enabled. `--num-agents` fixes the
count for a run. For local smoke runs, override `--episodes` and `--device`;
disable `training.wandb` in a temporary config when external logging is not
intended. MP4 creation requires `ffmpeg`.

## Working conventions

- Use module entry points (`python -m ...`) so imports resolve consistently.
- Run both test modules after shared simulation/domain changes; run at least
  `tests.test_learning` after model, decoder, observation, config, routing,
  return, oracle, benchmark-policy, evaluation-suite, or checkpoint changes.
- Keep FI-OPT scalar and reconstructed results semantically identical. Validate
  reconstructed routes through `run_simulation`; do not post-process an
  optimal assignment with unrelated shortest paths.
- Preserve the strict phase boundary in Scout-Then-Execute: service-only agents
  wait for the final reveal, then one FI-OPT plan is installed. Keep
  `replan_in_transit` and `set_runtime_state` handling aligned with the engine.
- Preserve outputs/checkpoints and prepared maps; do not overwrite a prepared
  map without an explicit map-builder invocation, and do not commit caches,
  frames, videos, or weights.
- Check signatures before reviving old benchmarks/notebooks; retained scripts
  may predate the generalized capability API.
- Keep rendering optional and outside the core simulation import path.
- Test directed/asymmetric costs and unreachable routes when graph semantics
  change; do not assume an undirected connected grid.
