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
The learning objective normalizes makespan by a full-information parallel
open-TSP oracle and adds dimensionless death and incompletion penalties.

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
  - `real_map_benchmark.py`: builds the WV DEM instance, selects a baseline,
    runs it, and optionally writes JSON, CSV, PNG, and MP4 output.
  - `rendering.py`: visualization and ffmpeg integration, kept outside the core
    engine import path.
- `planning/`
  - `policies/baseline1.py`: independent distance routing. Service agents
    prefer supported, unknown, then unsupported targets; pure scouts move to
    the tallest safe node.
  - `policies/baseline2.py` / `scout_wrp.py`: assigns the least
    service-capable scout a Watchman Route Problem covering walk, then uses the
    baseline-1 attacker layer. Exact A* is used through 12 scoutable unknown
    targets, with weighted A* above that threshold.
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
    decoder.
  - `policy/candidates.py`: deterministic target, staging, and wait candidates.
  - `gpu_sim/observation_cpu.py`: canonical planner-visible feature builder and
    CPU batching despite the historical module path.
  - `policy/adapter.py` and `gpu_sim/rollout_cpu.py`: simulator-backed policy
    and CPU rollout/training path.
  - `gpu_sim/world.py`, `state.py`, `observation_gpu.py`, `rollout_gpu.py`, and
    `cugraph_router.py`: batched CUDA simulation and cuGraph routing path.
  - `policy/oracle.py`: full-information min-max open-TSP normalization oracle.
  - `train.py`: CPU/CUDA REINFORCE training. Timestamped checkpoints contain
    `trained_weights.pt` and the resolved `config.yaml`.
  - `test.py`: deterministic checkpoint evaluation and optional post-rollout
    rendering. `policy/evaluation.py` is a smaller legacy interface.
- `Graph_Generation/`: visibility, blockage, target-graph, and stochastic
  diverse-path helpers used by older planners.
- `Real_Life_Maps/`: bundled `WV_DEM.tif`, `WV_roads.pkl`, terrain builder, and
  retained older benchmark scripts.
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
`ground_truth` into `env_map`. Never pass ground truth to a policy, candidate
generator, or observation builder.

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
- Incomplete episodes need an explicit penalty so stopping early cannot beat
  completion.
- Keep CPU and tensor/CUDA transition semantics aligned. The CUDA path batches
  world state, routing, rollout, and gradient replay, not just inference.
- `simulation_batch_size` controls simultaneous tensor episodes;
  `reinforce_batch_size` controls optimizer accumulation. Legacy configs with
  `batch_size` map it to both fields.
- Evaluation accepts a checkpoint run directory or weights path and normally
  uses the configuration saved beside the weights.

## Environment and commands

Use `uv`; system Python is not expected to have the dependencies. The lockfile
includes PyTorch 2.12 and CUDA 13 cuGraph. CUDA training needs a compatible
NVIDIA/RAPIDS environment, while the CPU simulator and synthetic checks do not
need to run the CUDA backend.

```bash
uv sync

# Fast regression suites
uv run python -m tests.test_simulation
uv run python -m tests.test_learning

# Real-map benchmark (equivalent root entry: uv run python main.py)
uv run python -m simulation.real_map_benchmark --help
uv run python -m simulation.real_map_benchmark --policy baseline1 --seed 0
uv run python -m simulation.real_map_benchmark --policy baseline2 --render

# Learned policy
uv run python -m learning.train --config learning/config.yaml --episodes 100 --device cpu
uv run python -m learning.test learning/checkpoints/<run-directory> --device cuda
uv run python -m learning.test learning/checkpoints/<run-directory> --device cuda --render
```

The default learning config currently uses 3 target types, 5-9 targets, 4
agents, CUDA, and Weights & Biases logging. For local smoke runs, override
`--episodes` and `--device`; disable `training.wandb` in a temporary config when
external logging is not intended. MP4 creation requires `ffmpeg`.

## Working conventions

- Use module entry points (`python -m ...`) so imports resolve consistently.
- Run both test modules after shared simulation/domain changes; run at least
  `tests.test_learning` after model, decoder, observation, config, routing,
  return, or checkpoint changes.
- Preserve outputs/checkpoints; do not commit caches, frames, videos, or
  weights.
- Check signatures before reviving old benchmarks/notebooks; retained scripts
  may predate the generalized capability API.
- Keep rendering optional and outside the core simulation import path.
- Test directed/asymmetric costs and unreachable routes when graph semantics
  change; do not assume an undirected connected grid.
