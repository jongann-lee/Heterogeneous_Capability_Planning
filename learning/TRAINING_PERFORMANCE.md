# CUDA training diagnostics

The CUDA backend runs simulation state, route queries, policy inference, and
gradient replay on the GPU. It still uses Python for the event loop,
NetworkX for scenario preparation and the exact FI-OPT normalization, and CPU
memory for replay observations. GPU utilization alone does not identify which
of those stages dominates.

## Findings and changes

- The GPU leak was reproduced on the host RTX 5090 (32 GB), using the
  repository's existing `.venv`. Sandboxed processes cannot see this GPU;
  host GPU access is required for these checks.
- The installed cuGraph 26.06 Python `sssp` wrapper checks source membership
  using `Graph.has_node`, which extracts a native vertex array without freeing
  its owner. The lower-level Python membership check also omits its result
  free. A 32-query bank on the 4,096-node map leaked approximately 0.5 MiB per
  call. Ten repeated calls accumulated 5 MiB; garbage collection and
  `torch.cuda.empty_cache()` did not release it. In a 32-episode training run,
  live RMM allocations increased from 1.8 to 651.7 MiB.
- `native_sssp.py` now calls the same libcugraph SSSP algorithm through its
  public C API and explicitly frees every graph, result, and array view.
  Source membership is checked against the explicit contiguous vertex range.
  No installed packages are patched. Repeating the isolated workload now
  leaves both RMM live bytes and allocation counts unchanged.
- Temporary cuGraph routing graphs previously included every distinct query
  at once. Their size depends on the target layout and diverging episode
  states, so later batches can have larger allocation peaks. Routing now
  limits each traversal to 1,000,000 copied edges (including source links) and
  262,144 copied terrain vertices, with one terrain copy as the minimum. Larger
  banks use multiple exact traversals. Final distance/predecessor banks still
  scale with the total query count; this is a scratch-space bound, not a total
  VRAM limit. Temporary graph construction buffers are released before dense
  result conversion. No blocked graph or result cache was added.
- Compact candidate-to-target relation minima are cached by the planner-visible
  live-target mask. Agent motion and type revelations do not change them.
  This per-world cache is capped at 512 masks and 8 MiB; it contains no graph,
  predecessor, path, or full SSSP row. Larger layouts bypass the cache.
- Gradient replay combines episode-decision records across traces, with a
  configurable `training.replay_batch_size` (default 64; CLI
  `--replay-batch-size`). This preserves per-episode decision averaging and
  optimizer accumulation. Lower it to reduce activation memory. It is separate
  from simultaneous simulation episodes and the optimizer's episode batch.
- Training gives RAPIDS a reusable scratch pool when its allocator is the
  default direct CUDA allocator. It starts at 64 MiB and is capped at the
  smaller of 1 GiB and one eighth of initially free VRAM. Existing custom
  resources are preserved. An isolated routing benchmark improved from
  approximately 54 to 37 ms per call with pooling.
- The assignment decoder previously looped over episodes and read CUDA
  scalars into Python for every assignment. It now batches episodes, aggregates
  semantic aliases once per decision, and transfers legacy Python assignments
  in one operation. Tensor rollouts use device action indices directly.
- PyTorch's cache was flushed after every simulation batch and replay batch.
  It is now released once per optimizer update before the rollout phase to
  make unused replay activation memory available to RAPIDS. Caches remain
  reusable within each phase.
- Metric export uses one bulk tensor transfer per rollout, and consumed CPU
  traces are released after replay. Rollout collection explicitly disables
  autograd throughout.

The leak is in native RAPIDS allocations, so PyTorch's live-memory counters
alone do not reveal it. The original profile also identified routing as the
largest cost: 111 seconds of a 186-second, 32-episode diagnostic run.

## Measuring a run

Add `--profile` to the existing training command. For a short diagnostic run
with checkpoints in a new timestamped directory under `/tmp`:

```bash
uv run python -m learning.train --config learning/config.yaml --device cuda \
  --episodes 64 --profile --checkpoint /tmp/hcp-training-profile
```

This retains the config's W&B setting. Use a separate config with
`training.wandb: false` if external logging is not wanted.

Each completed optimizer update prints a `training_profile={...}` JSON record.
The same metrics go to W&B when enabled. A PyTorch OOM or Python allocation
failure also prints an allocator snapshot before re-raising the exception.

- `time/instance_generation_seconds`, `world_preparation_seconds`, and
  `fi_opt_seconds` measure preparation between simulation batches. Initial
  terrain/model construction and the initial FI-OPT solve precede these
  per-update measurements.
- `time/rollout_seconds` includes observation/routing, policy decoding,
  trace copying, route reconstruction, and simulation. The separately logged
  rollout components overlap this total; do not add them to it.
- `time/gradient_replay_seconds`, `optimizer_seconds`, `checkpoint_seconds`,
  and `metrics_to_cpu_seconds` identify the remaining training costs.
- `memory/{start,after_rollout,end}/torch_allocated_mib` measures live PyTorch
  allocations; `torch_reserved_mib` includes its reusable cache.
- `torch_peak_allocated_mib` is the peak since the start of the update.
  `rmm_allocated_mib` measures tracked live RAPIDS allocations;
  `rmm_peak_since_start_mib` is cumulative since profiling was enabled.
- `rmm_pool_reserved_mib` reports reusable RMM pool storage separately from
  live allocations. A stable nonzero pool reservation is expected.
- `device_used_mib` and `device_free_mib` measure the entire device, including
  other processes and allocations outside those tracked allocators.

Compare end-of-update live allocation counts after warm-up across many updates
with comparable scenario sizes. A rising peak alone is not evidence of a leak;
growing live allocations on a repeated fixed workload warrant retention
analysis. Profiling synchronizes phase boundaries and adds overhead, so disable
it when measuring normal end-to-end throughput.

PyTorch documents the distinction between live allocations and reserved memory
in its [CUDA memory management guide](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management).
RAPIDS allocation counters use its existing resource through the
[RMM statistics API](https://docs.rapids.ai/api/rmm/stable/python/statistics/).

## Local validation

The same 32-episode GPU diagnostic workload (seed 0, bundled WV terrain,
simulation batch 8, optimizer batch 32) gave these measurements:

| Measurement | Before GPU fixes | After GPU fixes |
| --- | ---: | ---: |
| Total wall time | 186.0 s | 77.8 s |
| Observation and routing | 111.1 s | 32.2 s |
| Gradient replay | 31.8 s | 3.5 s |
| Live RMM memory at the end | 651.7 MiB | 0.95 MiB |
| Peak live PyTorch memory | 232.6 MiB | 993.7 MiB |

That is a 2.39x end-to-end speedup for this workload. Both measurements include
Python profiling and synchronized CUDA phase timing; both already include the
batched assignment decoder. They compare the subsequent GPU routing, allocator,
and replay fixes, not an untouched checkout. The same final episode return was
recorded before the first optimizer update. Larger replay batches deliberately
use more activation memory to reduce launch and backward overhead; cached
PyTorch reservation can therefore be higher even though the leak is fixed.

A further 128-episode run completed four optimizer updates in 240.4 seconds
with CUDA phase timing enabled and Python profiling disabled. Every update
ended with exactly 0.9541 MiB of live RMM memory and a 64 MiB RMM pool. End-of-
update live PyTorch memory ranged from 168.7 to 174.0 MiB as layouts changed;
per-update activation peaks ranged from 832 to 994 MiB. This validates repeated
updates, rather than claiming a full multi-thousand-episode training run.
W&B was disabled and all diagnostic logs and checkpoints were stored under
`/tmp/hcp-gpu-diagnosis.ZqV2fe`, outside repository outputs and checkpoints.

A separate real-terrain stress check repeated 2,000 routing banks with 1, 8,
16, and 32 queries, varying blocked nodes. After warm-up, live RMM memory stayed
at exactly 387,072 bytes in three allocations, and live PyTorch memory stayed
at exactly 235,520 bytes throughout. The check took 65.9 seconds and asserted
exact equality every 250 banks, without garbage collection or cache flushing.

A CPU microbenchmark with eight episodes, six agents, and 80 actions measured
decoder time at approximately 7.4 ms before and 0.82 ms after, and decoder replay
at 6.4 ms before and 0.73 ms after (single CPU thread, 50 iterations).
The CPU profiler counted 392 scalar reads before and none after. Assignments,
joint log probabilities, entropies, and gradients matched the previous decoder.
These numbers do not measure total training speed or CUDA performance.

On the bundled 4,096-node WV terrain with seed 0 and four agents, standalone
CPU FI-OPT solves took about 0.25 s for five targets and 1.15 s for seven and
nine targets. Each sampled simulation batch runs this solve before its rollout.
These examples establish CPU work between batches; they do not determine its
fraction of wall time on the training GPU.

Tests cover mixed active/padded episodes, alias capacities, stochastic replay
gradients, partial optimizer batches, and route chunk limits. The CUDA routing
test compares chunked and unchunked results with asymmetric and unreachable
routes when a CUDA device is available. A separate CUDA regression compares
live PyTorch and RMM allocations after repeated routing workloads and warm-up.

The complete `uv run --no-sync python -m tests.test_learning` run on the RTX
5090 passed 79 of 80 checks, including CUDA parity, independent-blocker routing,
exact native allocation ownership, and relation-cache invalidation. Replay
losses and gradients match across batches that split and combine traces on
both CPU and CUDA, with the critic enabled and disabled. The sole failure is
the pre-existing mismatch between the default config's enabled unknown-target
actions and `test_yaml_configuration_loads_and_validates`, which expects them
disabled; the same assertion fails in the unmodified HEAD test.
