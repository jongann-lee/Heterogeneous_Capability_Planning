"""Evaluate one fixed case and export exact task discovery/completion times.

Uses the same CLI and CPU/CUDA policy execution as ``learning.test``. Defaults
to the balanced-hybrid team and balanced target layout on KTH; no render is needed.
"""

from pathlib import Path

from learning.evaluation_suite import resolve_suite_path
from learning.test import evaluate as _evaluate, main as _main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "learning/evaluation_suites/KTH_indoor_big.json"
DEFAULT_MAP_PATH = PROJECT_ROOT / "Real_Life_Maps/KTH_Indoor_map.pkl.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation/KTH-detailed"
DEFAULT_AGENT_CONFIG = "agents_08_b"
DEFAULT_TARGET_CONFIG = "targets_14_a"


def evaluate(*args, **kwargs):
    """Run exactly one episode of one scenario with task progress recording."""
    if len(args) < 3:
        kwargs.setdefault("suite", DEFAULT_SUITE)
    suite = args[2] if len(args) >= 3 else kwargs["suite"]
    if resolve_suite_path(suite) == DEFAULT_SUITE:
        if len(args) < 6:
            kwargs.setdefault("agent_config", DEFAULT_AGENT_CONFIG)
        if len(args) < 7:
            kwargs.setdefault("target_config", DEFAULT_TARGET_CONFIG)
    if len(args) < 19:
        kwargs.setdefault("map_path", DEFAULT_MAP_PATH)
    kwargs["detailed"] = True
    return _evaluate(*args, **kwargs)


def main(argv=None):
    _main(argv, detailed=True, default_suite=str(DEFAULT_SUITE),
          default_map_path=str(DEFAULT_MAP_PATH),
          default_output_dir=DEFAULT_OUTPUT_DIR,
          default_agent_config=DEFAULT_AGENT_CONFIG,
          default_target_config=DEFAULT_TARGET_CONFIG)


if __name__ == "__main__":
    main()
