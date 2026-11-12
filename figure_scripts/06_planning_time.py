"""Compare saved replan times for 1103 and both classical baselines.

    uv run python figure_scripts/06_planning_time.py

WV scenarios are grouped by the numeric product agent_count * target_count:
3x8 and 4x6 both contribute to workload 24. Include only KTH's largest
product, currently 8x14=112, in the same policy series. Circles, squares, and triangles
distinguish the policies. Use a logarithmic seconds axis for both maps.

Each point is the equally weighted mean of saved per-scenario replan means;
error bars show their population standard deviation, matching learning.analyze.
Scenarios with more replans do not receive more weight. Read result JSONs only;
do not rerun evaluations. Export PDF, editable 600 px-wide SVG, and a CSV with
the plotted values to figure_scripts/outputs, replacing them on reruns.
--wv-scout-then-engage selects a newer timed WV Scout Then Engage result;
the legacy result has no timings, so only its KTH point is plotted until then.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.export_utils import save_pdf_and_svg
from figure_scripts.policy_labels import OPT_BY_PART_LABEL
from learning.analyze import analyze_evaluation


POLICIES = ("learned", "opt-by-part", "scout-then-execute")
LABELS = {"learned": "CAAT Graph + HGT", "opt-by-part": OPT_BY_PART_LABEL,
          "scout-then-execute": "Scout Then Engage"}
COLORS = {"learned": "#0072B2", "opt-by-part": "#7A9E7E", "scout-then-execute": "#B97676"}
MARKERS = {"learned": "o", "opt-by-part": "s", "scout-then-execute": "^"}
LINESTYLES = {policy: "-" for policy in POLICIES}
RESULT_FILES = {
    "WV": ("WV/WV-1103-position-best.json", "WV/WV-opt-by-part.json",
           "WV/2026-09-09-scout-then-execute.json"),
    "KTH": ("KTH-Factorial-v2/1103-position-pair-staging-best.json",
            "KTH-Factorial-v2/opt-by-part.json", "KTH-Factorial-v2/scout-then-execute.json"),
}


@dataclass(frozen=True)
class TimingPoint:
    map_name: str
    policy: str
    workload: int
    count: int
    mean: float
    std: float
    pairs: tuple[tuple[int, int], ...]


def validate_timings(payload):
    analyze_evaluation(payload)
    for record in payload["scenarios"]:
        identity = record["scenario_id"]
        calls = record.get("planning_call_count")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
            raise ValueError(f"{identity}: missing or invalid planning_call_count")
        for field in ("mean_planning_time_seconds", "total_planning_time_seconds"):
            value = record.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{identity}: missing or invalid {field}")
        if not math.isclose(record["mean_planning_time_seconds"],
                            record["total_planning_time_seconds"] / calls,
                            rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"{identity}: inconsistent planning timing fields")


def validate_comparison(first, second):
    if first["suite_id"] != second["suite_id"]:
        raise ValueError("policy results use different evaluation suites")
    map_hash = first.get("prepared_map", {}).get("sha256")
    if not map_hash or map_hash != second.get("prepared_map", {}).get("sha256"):
        raise ValueError("policy results use different or unidentified prepared maps")
    indexed = [{(row["scenario_id"], row.get("episode_index", 0)): row
                for row in payload["scenarios"]} for payload in (first, second)]
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("policy results have different scenario coverage")
    for key, row in indexed[0].items():
        reference = indexed[1][key]
        if not isinstance(row.get("instance"), dict) or not row["instance"] \
                or row["instance"] != reference.get("instance"):
            raise ValueError(f"{key}: policy results have different or unidentified instances")
        for field in ("agent_count", "target_count", "seed"):
            if row.get(field) != reference.get(field):
                raise ValueError(f"{key}: policy results have different {field}")


def aggregate_timings(payload, map_name):
    records = payload["scenarios"]
    if map_name == "KTH":
        largest = max(row["agent_count"] * row["target_count"] for row in records)
        records = [row for row in records
                   if row["agent_count"] * row["target_count"] == largest]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["agent_count"] * record["target_count"]].append(record)
    points = []
    for workload, rows in sorted(grouped.items()):
        means = [row["mean_planning_time_seconds"] for row in rows]
        pairs = tuple(sorted({(row["agent_count"], row["target_count"]) for row in rows}))
        points.append(TimingPoint(map_name, payload["policy"], workload, len(rows),
                                  statistics.fmean(means), statistics.pstdev(means), pairs))
    return points


def load_points(results_dir, wv_scout_result=None):
    points, sources = [], []
    for map_name, files in RESULT_FILES.items():
        payloads = []
        for name, policy in zip(files, POLICIES):
            path = wv_scout_result if (wv_scout_result is not None and map_name == "WV"
                                       and policy == "scout-then-execute") else results_dir / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            analyze_evaluation(payload, source=path)
            if payload.get("policy") != policy:
                raise ValueError(f"{path}: expected policy {policy!r}")
            untimed_legacy = (map_name == "WV" and policy == "scout-then-execute"
                             and wv_scout_result is None
                             and all(row.get("mean_planning_time_seconds") is None
                                     for row in payload["scenarios"]))
            if untimed_legacy:
                print("WV Scout Then Engage has no saved replan timings; only its KTH point "
                      "is plotted. Use --wv-scout-then-engage with a timed WV result.",
                      file=sys.stderr)
            else:
                validate_timings(payload)
                points.extend(aggregate_timings(payload, map_name))
            payloads.append(payload)
            sources.append(path)
        for payload in payloads[1:]:
            validate_comparison(payloads[0], payload)
    return points, sources


def render_plot(points, sources, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor="white")
    try:
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.89)
        for policy in POLICIES:
            series = sorted((point for point in points if point.policy == policy),
                            key=lambda point: point.workload)
            ax.errorbar([point.workload for point in series], [point.mean for point in series],
                        yerr=[point.std for point in series], color=COLORS[policy],
                        marker=MARKERS[policy], linestyle=LINESTYLES[policy],
                        markersize=5, linewidth=1.4, elinewidth=0.8,
                        capsize=2, label=LABELS[policy], zorder=3)
            point = series[-1]
            ax.annotate(f"{point.mean:.2f} s", (point.workload, point.mean),
                        xytext=(-11, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=11, color=COLORS[policy])
        ax.set_yscale("log")
        # A symmetric SD can cross zero; show only its positive portion on the
        # logarithmic axis, while preserving the full SD in the CSV/metadata.
        lower = min(max(point.mean - point.std, point.mean * 0.1) for point in points)
        upper = max(point.mean + point.std for point in points)
        ax.set_ylim(lower / 1.6, upper * 2.5)
        workloads = [point.workload for point in points]
        ax.set_xlim(min(workloads) - 4, max(workloads) + 6)
        ax.set_xticks([15, 24, 36, 54, 80, 112])
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", color="#DADADA", linewidth=0.7)
        ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of agents × number of tasks", fontsize=12)
        ax.set_ylabel("Average replan time (s)", fontsize=12)
        ax.set_title("Average replanning time", fontsize=15, pad=12)
        ax.legend(loc="upper left", frameon=False, fontsize=12)
        description = (
            "Wall-clock seconds per replan; equal-weight average of saved per-scenario means; "
            "error bars: population SD across scenario means. WV grouped by numeric "
            "agent_count * target_count, including all pairs with the same product. "
            "KTH: largest product only, included in the same policy series as WV. "
            "Circles: CAAT Graph + HGT; squares: "
            f"{OPT_BY_PART_LABEL}; triangles: Scout Then Engage. "
            "All lines solid; logarithmic y axis. "
            "Inputs: " + "; ".join(map(str, sources))
        )
        if not any(point.map_name == "WV" and point.policy == "scout-then-execute" for point in points):
            description += ". WV Scout Then Engage omitted because its result has no timing fields."
        output.parent.mkdir(parents=True, exist_ok=True)
        return save_pdf_and_svg(fig, output, metadata={
            "Title": "Average replanning time", "Subject": description,
            "Creator": "figure_scripts/06_planning_time.py",
        })
    finally:
        plt.close(fig)


def write_csv(points, output):
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["map", "policy", "agents_times_tasks", "scenario_count",
                         "mean_replan_seconds", "population_std_seconds", "agent_task_pairs"])
        for point in points:
            writer.writerow([point.map_name, LABELS[point.policy], point.workload, point.count,
                             point.mean, point.std,
                             "; ".join(f"{a}x{t}" for a, t in point.pairs)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--wv-scout-then-engage", type=Path,
                        help="timed WV Scout Then Engage evaluation JSON")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/06_planning_time.pdf")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension (SVG and CSV are also written)")
    try:
        wv_scout_result = None if args.wv_scout_then_engage is None \
            else args.wv_scout_then_engage.expanduser().resolve()
        points, sources = load_points(args.results_dir.expanduser().resolve(), wv_scout_result)
        output = args.output.expanduser().resolve()
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                             "savefig.bbox": None}):
            svg = render_plot(points, sources, output)
        data_path = output.with_suffix(".csv")
        write_csv(points, data_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{output}\n{svg}\n{data_path}")
    for point in points:
        print(f"  {point.map_name} {point.workload:3d} | {LABELS[point.policy]}: "
              f"{point.mean:.6f} ± {point.std:.6f} s (n={point.count})")


if __name__ == "__main__":
    main()
