# Heterogeneous Capability Planning

Centralized planning for heterogeneous agents on uncertain weighted terrain.
Targets have hidden positive integer types; capability `0` grants scouting and
each positive agent capability identifies a target type it can service.

This checkout contains WV training and fixed-suite evaluation on WV, WA, and
KTH. It includes the learned task-graph policy and the `oracle`,
`scout-then-engage`, and `engage-while-scout` comparison policies.

## Setup

Use Python 3.12 or 3.13 and `uv`:

```bash
uv sync
```

CUDA runs require a compatible NVIDIA/CUDA 13 environment. The lockfile pins
PyTorch 2.12 and Linux-only cuGraph. CPU training and evaluation are also
supported; macOS skips cuGraph. MP4 rendering additionally requires `ffmpeg`.

## Prepared maps and evaluation suites

Training and evaluation use bundled, precomputed terrain and visibility.
Outdoor map rebuilding and its raw DEM/road inputs are excluded from this
checkout.

| Terrain | Map under `Real_Life_Maps/` | Suite under `learning/evaluation_suites/` | Cases |
| --- | --- | --- | --- |
| WV | `WV_tobler_viewshed_64.pkl.gz` | `wv_factorial_test_v1.json` | 180 |
| WA | `WA_tobler_viewshed_64.pkl.gz` | `wa_factorial_test_v1.json` | 180 |
| KTH | `KTH_Indoor_map.pkl.gz` | `KTH_indoor_big.json` | 9 |

The indoor source rasters and helper code remain for floor-plan rendering and
regression tests. The KTH suite uses eight agents, fourteen targets, and five
target types. WV and WA use three target types.

## WV training

```bash
uv run python -m learning.train --config learning/config.yaml \
  --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz --device cuda
```

Each run creates a new timestamped directory under `learning/checkpoints/`
with its resolved configuration and best, latest, and final weights. The
default config enables Weights & Biases; set `training.wandb: false` in a
separate config or set `WANDB_MODE=disabled` for local runs. For a short CPU
run, add `--episodes 2 --device cpu --checkpoint-dir /tmp/hcp-training`.

The November 3 results use
`learning/checkpoints/2026-11-03_11-34-21_798508/best_weights.pt` and the
`config.yaml` beside it. The current `learning/config.yaml` has different
experiment settings. To retrain the November 3 recipe, select that saved config
with `--config` and explicitly pass the WV map path. Checkpoints are Git-ignored
and must be supplied separately for evaluation without retraining.

## Learned evaluation

Set `HCP_WEIGHTS` to the desired weights file. Evaluation loads its saved
configuration by default; an explicit file selects the exact snapshot.

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

Use `--device cpu` for the CPU simulator. Evaluation is deterministic and
checks suite locations against the selected graph. `--allow-map-mismatch`
explicitly permits evaluating WV-trained weights on different terrain.
Results record the selected map's hash and any mismatch override.

## Baselines and analysis

The comparison policies run on CPU and require no checkpoint. For each row in
the map/suite table, pass the same map and suite to each policy:

```bash
for HCP_POLICY in oracle scout-then-engage engage-while-scout; do
  uv run python -m learning.test --policy "$HCP_POLICY" \
    --suite learning/evaluation_suites/wv_factorial_test_v1.json \
    --map-path Real_Life_Maps/WV_tobler_viewshed_64.pkl.gz \
    --output "results/wv-${HCP_POLICY}.json"
done

uv run python -m learning.analyze results/wv-learned.json
```

The `oracle` computes executable target-aware min-max routes; it does not model one
agent clearing a target so another can transit through it later. Its search is
exponential in target count, so fourteen-target KTH baseline cases can take
substantial CPU time.

Use `--agent-config`, `--target-config`, or `--limit` to select cases. Rendering
requires exactly one selected case. `learning.test_single_detailed` records
task discovery and completion times using the same evaluator. Result paths
must be new; evaluation refuses to overwrite an existing file.

## Repository layout

- `learning/`: model, candidates, constrained decoder, CPU/CUDA rollouts,
  training, fixed-suite evaluation, and analysis.
- `planning/`: the oracle and the shared comparison-policy solvers.
- `simulation/`: agents, domain rules, event engine, visibility, and rendering.
- `Real_Life_Maps/`: prepared maps, loader, and indoor rendering support.

Use module entry points (`python -m ...`) so imports resolve consistently.
See `learning/README.md` for model and evaluation details.
