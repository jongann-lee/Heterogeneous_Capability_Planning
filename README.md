# Heterogeneous Capability Planning

Centralized planning for heterogeneous agents on uncertain weighted terrain.
Targets have hidden positive integer types; capability `0` grants scouting and
each positive agent capability identifies a target type it can service.

## Repository layout

- `simulation/`: domain model, agent state, event engine, rendering, and the
  real-map benchmark entry point.
- `planning/`: hand-written policies and retained comparison planners.
- `learning/`: centralized PyTorch attention policy, YAML configuration,
  candidate generation, constrained decoding, and REINFORCE training.
- `tests/`: small synthetic regression suite.
- `Graph_Generation/`, `Real_Life_Maps/`, `Single_Agent/`: retained map and
  algorithm dependencies from the original research repository.
- `outputs/`: ignored simulation artifacts.

## Run

```bash
uv sync
uv run python -m tests.test_simulation
uv run python -m simulation.real_map_benchmark --help
uv run python -m simulation.real_map_benchmark --policy baseline2 --seed 0 \
  --agent-capabilities '0;1;2;3' --verbose --render
```

Use module entry points (`python -m ...`) so package imports resolve
consistently. MP4 generation additionally requires the `ffmpeg` executable.
