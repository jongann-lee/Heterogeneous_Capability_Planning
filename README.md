# Coordination of Heterogeneous Teams with Uncertain Task Requirements

Code accompanying the paper of the same name. This `submission1` branch
contains the **CAAT Graph + HGT** planner, policy pretraining on WV, and fixed
suite evaluations on WV, WA, and KTH.

The problem is to coordinate heterogeneous agents to complete tasks at known
locations whose capability requirements are initially unknown. Agents can
possess multiple capabilities: capability `0` permits **scouting**, while each
positive capability permits **executing tasks** with the matching requirement.
A **hybrid agent** can both scout and execute tasks. Scouts reveal task
requirements according to a location-dependent **visibility mapping**.

Our hierarchical planner constructs the **Capability–Agent–Action–Task (CAAT)
graph**, processes it using a **heterogeneous graph transformer (HGT)**, and
uses an **MLP scorer** and **sequential decoder** to assign actions to agents.
A shortest-path planner then finds a path to each action's location. The four
action types are **execute**, **observe**, **stage**, and **wait**, allowing the
policy to jointly coordinate task discovery, execution, and staging.

## Setup

Use Python 3.12 or 3.13 and `uv`:

```bash
uv sync
```

CUDA runs require a compatible NVIDIA/CUDA 13 environment. The lockfile pins
PyTorch 2.12 and Linux-only cuGraph. CPU training and evaluation are also
supported; macOS skips cuGraph. MP4 rendering additionally requires `ffmpeg`.

## Prepared maps and evaluation scenarios

Training and evaluation use bundled maps with precomputed traversal costs and
visibility mappings. Outdoor map rebuilding and its raw DEM/road inputs are
excluded from this checkout.

| Map | Prepared map under `Real_Life_Maps/` | Evaluation suite under `learning/evaluation_suites/` | Scenarios |
| --- | --- | --- | --- |
| WV | `WV_tobler_viewshed_64.pkl.gz` | `wv_factorial_test_v1.json` | 180 |
| WA | `WA_tobler_viewshed_64.pkl.gz` | `wa_factorial_test_v1.json` | 180 |
| KTH | `KTH_Indoor_map.pkl.gz` | `KTH_indoor_big.json` | 9 |

WV is the training map. WA is an unseen outdoor map with the same traversal
and visibility models and the same ranges of agent, task, and capability
counts. KTH is an unseen indoor map with different traversal and visibility
models and more agents, tasks, and capabilities than seen during training.
WV and WA use three non-scout capabilities; KTH uses eight agents, fourteen
tasks, and five non-scout capabilities. The indoor source rasters and helper
code remain for floor-plan rendering.

## Configuration

`learning/config.yaml` supplies the defaults for a new training run. Its five
sections control:

| Section | Settings |
| --- | --- |
| `model` | CAAT/HGT dimensions, position and status features, travel-time normalization, scorer, and optional critic. |
| `candidates` | Single-task and two-task staging, wait actions, action capacity, and whether agents may attempt tasks with unknown requirements. |
| `instances` | Prepared training map and the ranges of randomly sampled agent and task counts. |
| `reinforce` | Learning rate, entropy regularization, return baseline, death and incompletion penalties, and gradient clipping. |
| `training` | Episode count, batch sizes, random seed, device, checkpoint directory, and Weights & Biases logging. |

The current defaults train for 12,800 episodes on WV with 3–6 agents, 5–9 tasks,
and three non-scout capabilities. For CUDA training, `simulation_batch_size: 8`
controls the number of simultaneous episodes; `reinforce_batch_size: 32`
controls the number of episodes accumulated per optimizer update. The agent-count range in
`instances` takes precedence over `training.num_agents`; `--num-agents` fixes
the count explicitly.

Pass `--config path/to/config.yaml` to use another configuration. Supported
command-line options, such as `--map-path`, `--episodes`, and `--device`,
override their configuration values. Each training run saves its resolved
`config.yaml` beside the weights. Learned evaluation uses that saved file by
default, so editing `learning/config.yaml` does not change an existing
checkpoint's evaluation settings. Evaluation suites supply the fixed agent
capabilities, task locations, and task requirements.

## Policy pretraining on WV

```bash
uv run python -m learning.train --config learning/config.yaml \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --device cuda
```

The policy is pretrained using **REINFORCE** on simulated rollouts. Each run
creates a new timestamped directory under `learning/checkpoints/` containing
its resolved configuration and best, latest, and final weights. The default
config enables Weights & Biases; set `training.wandb: false` in a separate
config or set `WANDB_MODE=disabled` for local runs. For a short CPU run, add
`--episodes 2 --device cpu --checkpoint-dir /tmp/hcp-training`.

The paper's default policy, run **1103**, uses
`learning/checkpoints/2026-11-03_11-34-21_798508/best_weights.pt` and the
`config.yaml` beside it. This configuration includes position features,
single-task and two-task staging, z-score normalization of travel-time edges,
and the MLP scorer.

`learning/config.yaml` matches run 1103's effective settings, including its
alive/completed status inputs, learning rate of `3e-4`, entropy coefficient of
`0.01`, and death penalty of `1.0`. The WV map path is relative for portability.
The training command above therefore uses the paper's default configuration.
Checkpoints are Git-ignored and must be supplied separately for evaluation
without retraining.

Alternatively, select run 1103's saved config directly:

```bash
uv run python -m learning.train \
  --config learning/checkpoints/2026-11-03_11-34-21_798508/config.yaml \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --device cuda
```

## Evaluating CAAT Graph + HGT

Set `HCP_WEIGHTS` to the desired weights file. Evaluation loads its saved
configuration by default; an explicit weights file selects the exact
checkpoint.

```bash
HCP_WEIGHTS=learning/checkpoints/2026-11-03_11-34-21_798508/best_weights.pt

uv run python -m learning.test "$HCP_WEIGHTS" --device cuda \
  --suite learning/evaluation_suites/wv_factorial_test_v1.json \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz \
  --output results/wv-learned.json

uv run python -m learning.test "$HCP_WEIGHTS" --device cuda \
  --suite learning/evaluation_suites/wa_factorial_test_v1.json \
  --map-path Real_Life_Maps/WA_tobler_viewshed_64.pkl.gz \
  --allow-map-mismatch --output results/wa-learned.json

uv run python -m learning.test "$HCP_WEIGHTS" --device cuda \
  --suite learning/evaluation_suites/KTH_indoor_big.json \
  --map-path Real_Life_Maps/KTH_Indoor_map.pkl.gz \
  --allow-map-mismatch --output results/kth-learned.json
```

All policy weights are frozen during evaluation, and actions are selected
deterministically. Evaluating the WV-pretrained policy on WA and KTH therefore
tests **zero-shot transfer**, without further training or fine-tuning.

Use `--device cpu` for the CPU simulator. Evaluation checks scenario locations
against the selected map. `--allow-map-mismatch` explicitly permits evaluating
WV-pretrained weights on a different map. Results record the selected map's
hash and any mismatch override.

## Baselines and result analysis

The paper compares CAAT Graph + HGT with three baselines:

- **Oracle** (`oracle`): uses ground-truth task requirements to minimize
  makespan over independently executable task assignments and routes.
- **Scout Then Engage** (`scout-then-engage`): first uses the Watchman Route
  Problem solver to discover all task requirements, then applies the
  Oracle solver to execute the remaining tasks.
- **Engage While Scout** (`engage-while-scout`): runs the Watchman Route Problem
  solver for a scouting group and the Oracle solver for an execution group
  simultaneously, using a heuristic to assign hybrid agents between the groups. The two
  solvers optimize their respective subproblems independently.

The baselines run on CPU and require no checkpoint.

### WV

```bash
uv run python -m learning.test --policy oracle \
  --suite learning/evaluation_suites/wv_factorial_test_v1.json \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz \
  --output results/wv-oracle.json

uv run python -m learning.test --policy scout-then-engage \
  --suite learning/evaluation_suites/wv_factorial_test_v1.json \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz \
  --output results/wv-scout-then-engage.json

uv run python -m learning.test --policy engage-while-scout \
  --suite learning/evaluation_suites/wv_factorial_test_v1.json \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz \
  --output results/wv-engage-while-scout.json
```

### WA

```bash
uv run python -m learning.test --policy oracle \
  --suite learning/evaluation_suites/wa_factorial_test_v1.json \
  --map-path Real_Life_Maps/WA_tobler_viewshed_64.pkl.gz \
  --output results/wa-oracle.json

uv run python -m learning.test --policy scout-then-engage \
  --suite learning/evaluation_suites/wa_factorial_test_v1.json \
  --map-path Real_Life_Maps/WA_tobler_viewshed_64.pkl.gz \
  --output results/wa-scout-then-engage.json

uv run python -m learning.test --policy engage-while-scout \
  --suite learning/evaluation_suites/wa_factorial_test_v1.json \
  --map-path Real_Life_Maps/WA_tobler_viewshed_64.pkl.gz \
  --output results/wa-engage-while-scout.json
```

### KTH

```bash
uv run python -m learning.test --policy oracle \
  --suite learning/evaluation_suites/KTH_indoor_big.json \
  --map-path Real_Life_Maps/KTH_Indoor_map.pkl.gz \
  --output results/kth-oracle.json

uv run python -m learning.test --policy scout-then-engage \
  --suite learning/evaluation_suites/KTH_indoor_big.json \
  --map-path Real_Life_Maps/KTH_Indoor_map.pkl.gz \
  --output results/kth-scout-then-engage.json

uv run python -m learning.test --policy engage-while-scout \
  --suite learning/evaluation_suites/KTH_indoor_big.json \
  --map-path Real_Life_Maps/KTH_Indoor_map.pkl.gz \
  --output results/kth-engage-while-scout.json
```

### Result analysis

Analyze a learned or baseline result with:

```bash
uv run python -m learning.analyze results/wv-learned.json
```

The Oracle does not model one agent completing a task so that another can
later pass through its location. Its search is exponential in task count, so
fourteen-task KTH baseline evaluations can take substantial CPU time. The
Watchman Route Problem solver is exact through twelve unknown tasks and uses
an approximation above that threshold.

The paper reports **regret** as the ratio of a policy's makespan to the
Oracle's makespan, `M / M*`, with the Oracle at `1`. The result field
`normalized_regret` stores `M / M* - 1`, with the Oracle at `0`.

Use `--agent-config`, `--target-config`, or `--limit` to select scenarios.
Rendering requires exactly one selected scenario.
`learning.test_single_detailed` records task requirement discovery and task
completion times using the same evaluator. Result paths must be new;
evaluation refuses to overwrite an existing file.

## Repository layout

- `learning/`: CAAT graph construction, HGT, scorer and sequential decoder,
  CPU/CUDA rollouts, REINFORCE pretraining, fixed-suite evaluation, and analysis.
- `planning/`: the Oracle and the solvers used by the two scouting/execution
  baselines.
- `simulation/`: agents, task interactions, event simulation, visibility, and
  rendering.
- `Real_Life_Maps/`: prepared maps, loading, and indoor rendering support.

Use module entry points (`python -m ...`) so imports resolve consistently.
