"""Plot one three-policy regret comparison PDF and SVG for each map.

Run from any working directory::

    uv run python figure_scripts/02_main_regrets.py

Outputs: figure_scripts/outputs/02_{WA,WV,KTH}_regret.pdf and .svg. SVGs are
600 px wide with preserved proportions, editable text, and vector plot elements.
Reruns overwrite both formats. --results-dir and --output-dir override the
default directories.
The learned-policy default is 1103; KTH uses the factorial v2 results.

Regret here is makespan / FI-OPT, i.e. the stored normalized regret plus one.
Each bar averages the per-scenario ratios, with population standard deviation
error bars (not standard error or a confidence interval). Every saved scenario
has equal weight. The oracle bar is exactly 1.0 with zero error. The plots use
the full factorial results, rather than the example layouts in 01_render_maps.

Map content and scenario definitions are checked before plotting. Any legacy
WA start difference is reported in the console and PDF metadata. The figures
contain no bottom captions.
Input evaluation JSONs are read only; no simulation or evaluation is run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.analyze import analyze_evaluation
from figure_scripts.export_utils import save_pdf_and_svg


FIGSIZE = (5.6, 4.0)
POLICIES = ("fi-opt", "scout-then-execute", "learned")
LABELS = ("Oracle", "Scout-Then-Engage", "CAAT Graph + HGT")
COLORS = ("#A7ABB0", "#7A9E7E", "#0072B2")
# File order is bar order: oracle, Scout-Then-Engage, and 1103.
RESULT_FILES = {
    "WA": (
        "WA/WA-Oracle.json",
        "WA/WA-scout-then-execute.json",
        "WA/WA-1103-position-best.json",
    ),
    "WV": (
        "2026-09-09-FI-OPT.json",
        "2026-09-09-scout-then-execute.json",
        "WV/WV-1103-position-best.json",
    ),
    "KTH": (
        "KTH-Factorial-v2/fi-opt.json",
        "KTH-Factorial-v2/scout-then-execute.json",
        "KTH-Factorial-v2/1103-position-pair-staging-best.json",
    ),
}
TITLES = {"WA": "Washington (WA)", "WV": "West Virginia (WV)", "KTH": "KTH Indoor"}


@dataclass(frozen=True)
class RegretStats:
    count: int
    mean: float
    std: float


def load_result(path, expected_policy):
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Reuse the existing evaluator analysis's record and outcome validation.
    analyze_evaluation(payload, source=path)
    if payload.get("policy") != expected_policy:
        raise ValueError(f"{path}: expected policy {expected_policy!r}")
    records = payload["scenarios"]
    ratios = []
    for record in records:
        identity = record["scenario_id"]
        denominator = record.get("fi_opt_makespan", record.get("oracle_makespan"))
        if (isinstance(denominator, bool)
                or not isinstance(denominator, (int, float))
                or not math.isfinite(denominator) or denominator <= 0):
            raise ValueError(f"{path}: {identity} has invalid FI-OPT makespan")
        if not record["completed"] or record["remaining_targets"]:
            raise ValueError(
                f"{path}: {identity} is incomplete; a makespan-only ratio "
                "would misrepresent this outcome")
        ratio = record["makespan"] / denominator
        if not math.isclose(ratio - 1, record["normalized_regret"],
                            rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{path}: {identity} has inconsistent regret fields")
        ratios.append(ratio)
    if expected_policy == "fi-opt":
        if any(not math.isclose(value, 1, rel_tol=1e-6) for value in ratios):
            raise ValueError(f"{path}: oracle results do not match FI-OPT")
        stats = RegretStats(len(ratios), 1.0, 0.0)
    else:
        stats = RegretStats(len(ratios), statistics.fmean(ratios), statistics.pstdev(ratios))
    return payload, stats


def scenario_key(record):
    return record["scenario_id"], record.get("episode_index", 0)


def validate_comparison(map_name, payloads):
    """Match actual instances, allowing only the disclosed WA start difference."""
    hashes = [payload.get("prepared_map", {}).get("sha256") for payload in payloads]
    if not all(hashes) or len(set(hashes)) != 1:
        raise ValueError(f"{map_name}: result files use different or unidentified maps")
    indexed = [{scenario_key(record): record for record in payload["scenarios"]}
               for payload in payloads]
    if any(indexed[0].keys() != records.keys() for records in indexed[1:]):
        raise ValueError(f"{map_name}: result files have different scenario coverage")
    different_starts = False
    for key, reference in indexed[0].items():
        if not isinstance(reference.get("instance"), dict):
            raise ValueError(f"{map_name}: {key} has no instance metadata")
        for records in indexed[1:]:
            record = records[key]
            instance = record.get("instance", {})
            if instance.keys() != reference["instance"].keys():
                raise ValueError(f"{map_name}: {key} has different instance fields")
            for field, value in reference["instance"].items():
                if instance[field] != value:
                    if map_name == "WA" and field == "source_position":
                        different_starts = True
                    else:
                        raise ValueError(f"{map_name}: {key} has different {field}")
            if record.get("seed") != reference.get("seed"):
                raise ValueError(f"{map_name}: {key} has different episode seeds")
            if instance["source_position"] == reference["instance"]["source_position"]:
                # CPU and CUDA saves can round the same FI-OPT scalar differently.
                if not math.isclose(record["oracle_makespan"], reference["oracle_makespan"],
                                    rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"{map_name}: {key} has different FI-OPT makespans")
    if not different_starts:
        return ""
    starts = []
    for label, payload in zip(LABELS, payloads):
        positions = sorted({tuple(record["instance"]["source_position"])
                            for record in payload["scenarios"]})
        starts.append(f"{label}: " + ", ".join(map(str, positions)))
    return "Different starts — " + "; ".join(starts)


def render_chart(map_name, stats, paths, start_note, destination, y_max):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    try:
        ax = fig.add_axes((0.15, 0.17, 0.82, 0.73))
        bars = ax.bar(
            range(3), [item.mean for item in stats],
            yerr=[item.std for item in stats], width=0.62, color=COLORS,
            edgecolor="black", linewidth=0.6, capsize=4,
            error_kw={"elinewidth": 1.1, "capthick": 1.1}, zorder=3,
        )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=0.9, zorder=2)
        ax.set_ylim(0, y_max)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("Regret (makespan / oracle)", fontsize=10)
        ax.set_xticks(range(3), [label.replace("-Engage", "-\nEngage")
                                for label in LABELS], fontsize=10)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#DADADA", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, item in zip(bars, stats):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    item.mean + item.std + 0.035 * y_max,
                    f"{item.mean:.2f}", ha="center", va="bottom", fontsize=10)
        fig.text(0.56, 0.96, TITLES[map_name], ha="center", va="top", fontsize=12)
        subject = (
            "Metric: per-scenario makespan / FI-OPT; bars: mean; errors: population SD. "
            + "Inputs: " + "; ".join(str(path) for path in paths)
            + (". " + start_note if start_note else "")
        )
        save_pdf_and_svg(
            fig, destination,
            metadata={"Title": f"{TITLES[map_name]} regret",
                      "Subject": subject,
                      "Creator": "figure_scripts/02_main_regrets.py"},
        )
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    comparisons = []
    # Validate all inputs before replacing any figures.
    for map_name, filenames in RESULT_FILES.items():
        paths = [results_dir / filename for filename in filenames]
        results = [load_result(path, policy) for path, policy in zip(paths, POLICIES)]
        payloads, stats = zip(*results)
        note = validate_comparison(map_name, payloads)
        comparisons.append((map_name, stats, paths, note))
    y_max = math.ceil(max(item.mean + item.std
                          for _, stats, _, _ in comparisons for item in stats) * 1.15 * 2) / 2
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "savefig.bbox": None}):
        for map_name, stats, paths, note in comparisons:
            destination = output_dir / f"02_{map_name}_regret.pdf"
            render_chart(map_name, stats, paths, note, destination, y_max)
            print(destination)
            print(destination.with_suffix(".svg"))
            for label, item in zip(LABELS, stats):
                print(f"  {label}: {item.mean:.6f} ± {item.std:.6f} (n={item.count})")
            if note:
                print(f"  {note}")


if __name__ == "__main__":
    main()
