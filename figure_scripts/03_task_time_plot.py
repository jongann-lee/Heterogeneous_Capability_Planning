"""Compare task discovery and completion progress for three detailed KTH runs.

    uv run python figure_scripts/03_task_time_plot.py

Reads scout-then-engage.json, engage-while-scout.json, and 1103-default.json
from outputs/evaluation/KTH-detailed, in that panel order, and writes
figure_scripts/outputs/03_KTH_task_time.pdf and .svg, overwriting on reruns.
--results-dir, --engage-while-scout, and --output select other paths.
The SVG is 600 px wide, preserves proportions, and retains editable text and
vector curves. All three panels have equal heights and matching axis layouts.

Blue shows cumulative tasks discovered. Red shows total tasks minus cumulative
completions (tasks remaining), so it decreases as tasks are completed. Curves
use the saved post-event timeline, including simultaneous events, and stop at
each run's makespan. All panels share identical time and integer count axes.
Dashed vertical lines mark finish times; the shared time axis includes padding
beyond the longest run.
Time is the prepared map's traversal time, not wall-clock time or seconds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.export_utils import save_pdf_and_svg
from figure_scripts.policy_labels import OPT_BY_PART_LABEL


DISCOVERED_COLOR = "#7297B5"
REMAINING_COLOR = "#B97676"
POLICY_NAMES = {"scout-then-execute": "Scout Then Engage",
                "opt-by-part": OPT_BY_PART_LABEL, "learned": "CAAT Graph + HGT"}
RUN_FILES = ("scout-then-engage.json", "engage-while-scout.json", "1103-default.json")
DEFAULT_ENGAGE_WHILE_SCOUT = ROOT / "outputs/evaluation/KTH-detailed/engage-while-scout.json"


@dataclass(frozen=True)
class TaskRun:
    path: Path
    policy: str
    map_hash: str
    instance: dict
    time_units: str
    total: int
    makespan: float
    times: tuple[float, ...]
    discovered: tuple[int, ...]
    remaining: tuple[int, ...]


def event_time(value, context):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{context}: expected finite, nonnegative time")
    return float(value)


def load_run(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("scenarios", [])
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one scenario")
    record = records[0]
    progress = record.get("task_progress", {})
    total = progress.get("total_tasks")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError(f"{path}: missing or invalid total_tasks")
    targets = progress.get("targets", [])
    timeline = progress.get("timeline", [])
    if len(targets) != total or not timeline:
        raise ValueError(f"{path}: missing target event times or timeline")
    makespan = event_time(record.get("makespan"), str(path))
    detections, completions = [], []
    for target in targets:
        discovered = target.get("discovered_at")
        completed = target.get("completed_at")
        discovered = None if discovered is None else event_time(discovered, str(path))
        completed = None if completed is None else event_time(completed, str(path))
        if (discovered is not None and discovered > makespan
                or completed is not None and completed > makespan):
            raise ValueError(f"{path}: task event occurs after makespan")
        if completed is not None and (discovered is None or completed < discovered):
            raise ValueError(f"{path}: task completes before discovery")
        detections.append(discovered)
        completions.append(completed)
    times, discovered_counts, remaining_counts = [], [], []
    for point in timeline:
        time = event_time(point.get("time"), str(path))
        if time > makespan or times and time <= times[-1]:
            raise ValueError(f"{path}: timeline must be chronological within the episode")
        discovered = sum(value is not None and value <= time for value in detections)
        completed = sum(value is not None and value <= time for value in completions)
        if (point.get("discovered_count") != discovered
                or point.get("completed_count") != completed
                or point.get("remaining_count") != total - completed):
            raise ValueError(f"{path}: timeline counts disagree with per-target times")
        times.append(time)
        discovered_counts.append(discovered)
        remaining_counts.append(total - completed)
    if times[0] != 0 or times[-1] != makespan:
        raise ValueError(f"{path}: timeline must span zero through makespan")
    # Every recorded transition must be represented, including a batch of
    # discoveries or completions at the same timestamp.
    if any(value is not None and value not in times for value in detections + completions):
        raise ValueError(f"{path}: timeline omits a task event")
    policy = payload.get("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"{path}: missing policy name")
    return TaskRun(
        path, policy, payload.get("prepared_map", {}).get("sha256"),
        record.get("instance"), progress.get("time_units"), total, makespan,
        tuple(times), tuple(discovered_counts), tuple(remaining_counts),
    )


def validate_runs(runs):
    if len(runs) != len(RUN_FILES):
        raise ValueError("expected exactly three detailed runs")
    first = runs[0]
    for run, expected_policy in zip(runs, ("scout-then-execute", "opt-by-part", "learned")):
        if run.policy != expected_policy:
            raise ValueError(f"{run.path}: expected policy {expected_policy}")
        if not first.map_hash or first.map_hash != run.map_hash:
            raise ValueError("detailed runs must use the same prepared map")
        if not first.instance or first.instance != run.instance:
            raise ValueError("detailed runs must use the same agent and task setup")
        if first.total != run.total or not first.time_units or first.time_units != run.time_units:
            raise ValueError("detailed runs must have the same task count and time units")


def run_title(run):
    return POLICY_NAMES.get(run.policy, run.policy)


def render_plot(runs, output):
    panel_count = len(runs)
    figure_height = 2.8 * panel_count + 0.4
    fig, axes = plt.subplots(
        panel_count, 1, figsize=(6.4, figure_height), sharex=True, sharey=True,
        gridspec_kw={"height_ratios": (1,) * panel_count},
    )
    try:
        # Fixed physical header/footer margins preserve each panel's size as
        # another comparison is added, including in the 600 px-wide SVG.
        fig.subplots_adjust(left=0.12, right=0.95, bottom=0.66 / figure_height,
                            top=1 - 0.72 / figure_height, hspace=0.50)
        longest = max(run.makespan for run in runs)
        horizon = math.ceil((longest + max(100, 0.05 * longest)) / 100) * 100
        for ax, run in zip(axes, runs):
            ax.step(run.times, run.discovered, where="post", color=DISCOVERED_COLOR,
                    linewidth=2, label="Tasks discovered", zorder=3)
            ax.step(run.times, run.remaining, where="post", color=REMAINING_COLOR,
                    linewidth=2, label="Tasks remaining", zorder=3)
            finish = ax.axvline(run.makespan, color="#666666", linestyle="--",
                                linewidth=1.1, zorder=2)
            finish.set_gid(f"finish-{run.policy}")
            ax.set_title(run_title(run), loc="left", fontsize=11)
            ax.set_xlim(0, horizon)
            ax.set_ylim(-0.3, run.total + 0.3)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.set_ylabel("Number of tasks", fontsize=10)
            ax.set_xlabel("Time", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(color="#DDDDDD", linewidth=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.99),
                   ncols=2, frameon=False, fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        return save_pdf_and_svg(
            fig, output, metadata={
                "Title": "KTH task discovery and completion over time",
                "Subject": ("Blue: cumulative discoveries. Red: total tasks minus completions. "
                            "Dashed gray: finish time. Time: " + runs[0].time_units + ". Inputs: "
                            + "; ".join(str(run.path) for run in runs)),
                "Creator": "figure_scripts/03_task_time_plot.py",
            },
        )
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "outputs" / "evaluation" / "KTH-detailed")
    parser.add_argument("--engage-while-scout", type=Path, default=DEFAULT_ENGAGE_WHILE_SCOUT,
                        help="detailed opt-by-part JSON for the middle panel")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "03_KTH_task_time.pdf",
                        help="PDF path; a 600 px-wide SVG is also written alongside it")
    args = parser.parse_args(argv)
    paths = [args.results_dir.expanduser().resolve() / name for name in RUN_FILES]
    paths[1] = args.engage_while_scout.expanduser().resolve()
    runs = [load_run(path) for path in paths]
    validate_runs(runs)
    output = args.output.expanduser().resolve()
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "savefig.bbox": None}):
        svg_output = render_plot(runs, output)
    print(output)
    print(svg_output)
    for run in runs:
        print(f"  {run_title(run)}: {run.total} tasks, "
              f"makespan {run.makespan:g}; {run.path.name}")


if __name__ == "__main__":
    main()
