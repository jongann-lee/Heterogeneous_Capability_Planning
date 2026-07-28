# AGENTS.md

Project guide for Codex working in this repo. Created from an earlier
session's accumulated context. **This repo has several diverging git branches
that each explore a different extension — verify the actual code on the current
branch before trusting any branch-specific claim below.**

## What this is

A multi-agent extension of the paper *"Navigating Uncertain Environments with
Heterogeneous Visibility"* (arXiv 2603.03495). The PDF is at the repo root
(`Navigating_Uncertain_Environments_with_Heterogeneous_Visibility(anonymous).pdf`).

The single-agent algorithm from the paper is already implemented; the active
work is lifting it to multiple cooperating agents and exploring variants
(lock-and-key constraints, moving targets, etc.). The `pyproject.toml` name
`uncertain-edge-tsp` is legacy single-agent naming.

**Core problem.** Agents traverse a weighted graph and must collectively visit
every target (TSP-like), but the planner has imperfect knowledge of the
environment and learns by *observing* as agents move. Path reward trades off
visibility/observation gain against traversal distance.

## Branches (each is a different research direction — they diverge!)

- **`main`** — base multi-agent version. Single `reward_ratio`, `Agent` has a
  `cost_multiplier`, no locks/keys, no policy dispatch.
- **`fixed_key`** — lock-and-key model with *fixed* agent↔key pairs. Targets
  carry a `lock`; agents carry `possessed_keys`; a target unlocks only with the
  matching key. Reward split into `edge_reward_ratio` + `target_reward_ratio`;
  `cost_multiplier == 1 + len(possessed_keys)`; partial-observability locks via
  sentinel `UNKNOWN_LOCK = -1`; `--policy` dispatch with `baseline1_shortest_path`.
- **`moving_targets`** — explores targets that move (branched from the 0527
  base). Uses `reward_ratio` / `cost_multiplier` like `main`; has leftover
  `baseline1_all_key.py` / `baseline2_greedy_single_key.py` files but no policy
  dispatch.

Because abstractions are renamed across branches (`reward_ratio` vs
`edge_reward_ratio`+`target_reward_ratio`; `cost_multiplier` vs
`possessed_keys`; locks present only on `fixed_key`), **always grep the current
branch rather than assuming.**

## Repo layout

- `Single_Agent/` — baseline. `repeated_topk.py` is the main algorithm class
  `RepeatedTopK` (builds a target graph, reweights edges by path reward, solves
  a TSP via `lin_kernighan_tsp.py`). `reward_functions.py` defines the
  per-node visibility reward. `calculate_path_reward` (in `repeated_topk.py`)
  is the shared reward function used by both single- and multi-agent code.
- `Multi_Agent/` — the multi-agent extension.
  - `finite_horizon_MA.py` — `Agent` class + assignment policies:
    `finite_horizon_assignment` (Hungarian) and `sequential_greedy_assignment`
    (submodular-aware greedy; the one actually used).
  - `multi_agent_simulation.py` — turn-based simulation driver + the real-DEM
    benchmark. CLI entry point (`main()` with argparse; see `--help`).
  - `simulation_utils.py` — PNG/MP4 rendering of runs.
  - `TSP_solver.py` — standalone nearest-neighbor / brute-force Hamiltonian helpers.
- `Graph_Generation/` — environment construction. `target_graph.py` has
  `stochastic_accumulated_blockage_path` (the diverse-path sampler used
  everywhere) and `create_fully_connected_target_graph`. Also `visibility.py`,
  `height_graph_generation.py`, `edge_block_generation.py`.
- `Real_Life_Maps/` — real DEM (`WV_DEM.tif`) + OSM roads (`WV_roads.pkl`)
  benchmark. `real_map_generation.py` builds `RealTerrainGrid`.
- `Automatic_Generated_Maps/` — synthetic-map benchmark suite.
- `main.py` is a stub, not a real entry point. Root `*.ipynb` are exploratory.

## Core abstractions (durable across branches)

- **Two-graph partial observability.** The simulation holds two graphs:
  `env_graph`/`env_map` = the planner's (optimistic) view, and
  `blocked_env_graph` = ground truth (with obstacles applied). Agents reveal
  truth only by observing from their position; the planner replans when reality
  contradicts its plan. Never let the planner read ground truth directly.
- **Node visibility attributes.** Nodes carry `visible_edges` (and on some
  branches `visible_nodes`) listing what's observable from that node. Edges
  carry `distance`, `observed_edge`, `num_used`.
- **Reward.** Roughly `reward_ratio * (visibility of newly-observed edges) −
  distance`, optionally discounted per step. On `fixed_key` this is split into
  an edge-visibility term and a target-observation term.
- **Assignment.** `sequential_greedy_assignment` picks one (agent, target) pair
  at a time, commits the best, and replays its path on the shared state so its
  observations propagate to the remaining candidates (respects submodularity).

## IMPORTANT gotcha: follow the *scored* path, not a fresh shortest path

When routing an agent to its assigned target, send it down the
**reward-maximizing path that the assignment scored**, NOT a recomputed
`nx.shortest_path`. The scorer evaluates diverse candidate paths and the best
one is generally NOT the shortest; recomputing a shortest path here silently
discards the planning. The assignment functions return `(target, path)` tuples
for this reason, and `_replan` / `replan` must consume `path` directly. (This
was a real bug that was fixed on `main` and `fixed_key`; `moving_targets`
already has the fix.) If you touch the replan/assignment code, preserve this.

## Running it

Dependencies (scipy, networkx, rasterio, osmnx, matplotlib) live in the project
venv — **system `python` lacks them.** Use `uv run python ...` (pyproject is
present on `main`/`fixed_key`); the `.venv` is untracked so it persists across
branch switches, so `.venv/bin/python` also works.

Multi-agent simulation on the real DEM map:
```
uv run python Multi_Agent/multi_agent_simulation.py --render        # writes PNGs + MP4
uv run python Multi_Agent/multi_agent_simulation.py --debug         # per-replan debug frames
uv run python Multi_Agent/multi_agent_simulation.py --help          # all flags (vary by branch)
```
Outputs land in `Multi_Agent/my_policy_simulation/`. Algorithm hyperparameters
are set inside `main()` (not all are CLI flags). MP4 export needs `ffmpeg`.

## Conventions / things that bite

- **Seeds:** set `random.seed(...)` and `np.random.seed(...)` once in `main()`;
  the diverse-path sampler and any lock/target randomization derive from there.
- **Verify before deleting/overwriting** the generated frames/CSVs in
  `my_policy_simulation/` — they're run artifacts.
- Some older benchmark scripts (`benchmark.py`, `Real_Life_Maps/*benchmark*.py`,
  `Automatic_Generated_Maps/run_benchmark.py`) call `RepeatedTopK` with an
  out-of-date signature and may be stale on some branches — check before relying
  on them.
