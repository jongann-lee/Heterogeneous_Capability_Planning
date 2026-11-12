"""Plot regret minus one for every scenario in one wide appendix figure.

    uv run python figure_scripts/08_each_scenario_plot.py

WV and WA contain 180 dots per policy; KTH contains nine per policy. Use
the same 1103 and baseline results/colors as figure 02. Oracle is a gray
zero reference line. Each map has its own panel, with identical y limits.

Regret minus one = makespan / FI-OPT makespan - 1. Keep suite file order
and match scenarios across policies. Small fixed horizontal offsets separate
the three methods at each scenario index. No scenarios are averaged away.

Export a wide PDF, editable 600 px-wide SVG, and a CSV containing every point
under figure_scripts/outputs, replacing them on reruns. Input results are
read only. --results-dir and --output override the default paths.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.export_utils import save_pdf_and_svg


regret = import_module("figure_scripts.02_main_regrets")
MAPS = ("WV", "WA", "KTH")
POLICIES = ("scout-then-execute", "opt-by-part", "learned")
LABELS = dict(zip(regret.POLICIES, regret.LABELS))
COLORS = dict(zip(regret.POLICIES, regret.COLORS))
MARKERS = {"scout-then-execute": "^", "opt-by-part": "s", "learned": "o"}
OFFSETS = {"scout-then-execute": -0.18, "opt-by-part": 0, "learned": 0.18}
FIGSIZE = (12, 4.2)


@dataclass(frozen=True)
class MapScenarios:
    name: str
    keys: tuple[tuple[str, int], ...]
    values: dict[str, tuple[float, ...]]


def load_scenarios(results_dir):
    maps, sources = [], []
    for name in MAPS:
        paths = [results_dir / filename for filename in regret.RESULT_FILES[name]]
        payloads = [regret.load_result(path, policy)[0]
                    for path, policy in zip(paths, regret.POLICIES)]
        note = regret.validate_comparison(name, payloads)
        if note:
            raise ValueError(f"{name}: individual scenario plots require identical starts; {note}")
        indexed = {
            policy: {regret.scenario_key(row): row for row in payload["scenarios"]}
            for policy, payload in zip(regret.POLICIES, payloads)
        }
        keys = tuple(indexed["fi-opt"])
        values = {}
        for policy in POLICIES:
            values[policy] = tuple(
                indexed[policy][key]["makespan"]
                / indexed[policy][key].get("fi_opt_makespan", indexed[policy][key]["oracle_makespan"])
                - 1 for key in keys)
        maps.append(MapScenarios(name, keys, values))
        sources.extend(paths)
    return maps, sources


def render_plot(maps, sources, output):
    fig, axes = plt.subplots(
        1, len(maps), figsize=FIGSIZE, sharey=True,
        gridspec_kw={"width_ratios": (3.4, 3.4, 1.4)},
    )
    try:
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.16, top=0.82, wspace=0.12)
        all_values = [value for result in maps for values in result.values.values() for value in values]
        low, high = min(0, min(all_values)), max(0, max(all_values))
        padding = max((high - low) * 0.06, 0.05)
        for ax, result in zip(axes, maps):
            count = len(result.keys)
            ax.axhline(0, color=COLORS["fi-opt"], linestyle="--", linewidth=1.1, zorder=2)
            for policy in POLICIES:
                points = ax.scatter(
                    [index + 1 + OFFSETS[policy] for index in range(count)],
                    result.values[policy], s=16, marker=MARKERS[policy],
                    color=COLORS[policy], alpha=0.85, edgecolors="none", zorder=3,
                )
                points.set_gid(f"{result.name}-{policy}-points")
            ax.set_title(f"{result.name} ({count} scenarios)", fontsize=14, pad=10)
            ax.set_xlim(0.3, count + 0.7)
            ax.set_ylim(low - padding, high + padding)
            ax.set_xticks([1, 30, 60, 90, 120, 150, 180] if count == 180
                          else list(range(1, count + 1, 2)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_xlabel("Scenario index", fontsize=13)
            ax.tick_params(labelsize=12)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Regret − 1", fontsize=15)
        handles = [Line2D([], [], color=COLORS["fi-opt"], linestyle="--",
                          linewidth=1.1, label="Oracle (0)")]
        handles.extend(Line2D([], [], marker=MARKERS[policy], linestyle="none",
                              color=COLORS[policy], markersize=6, label=LABELS[policy])
                       for policy in POLICIES)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncols=4, frameon=False, fontsize=13, handlelength=1.3,
                   handletextpad=0.5, columnspacing=1.4)
        output.parent.mkdir(parents=True, exist_ok=True)
        description = (
            "Every dot is one policy's result for one scenario: makespan / FI-OPT makespan - 1. "
            "No averaging. WV/WA: 180 points per policy; KTH: nine per policy. "
            "Scenario indices preserve reference suite file order and match across policies. "
            "Horizontal offsets: Scout Then Engage -0.18; Engage While Scout 0; CAAT +0.18. "
            "Oracle: gray zero line. All panels share identical y limits. Inputs: "
            + "; ".join(map(str, sources))
        )
        return save_pdf_and_svg(fig, output, metadata={
            "Title": "Per-scenario regret minus one", "Subject": description,
            "Creator": "figure_scripts/08_each_scenario_plot.py",
        })
    finally:
        plt.close(fig)


def write_csv(maps, output):
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["map", "scenario_index", "scenario_id", "episode_index",
                         "policy", "regret_minus_one"])
        for result in maps:
            for index, (scenario_id, episode_index) in enumerate(result.keys):
                for policy in POLICIES:
                    writer.writerow([result.name, index + 1, scenario_id, episode_index,
                                     LABELS[policy], result.values[policy][index]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/08_each_scenario_plot.pdf")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension (SVG and CSV are also written)")
    try:
        maps, sources = load_scenarios(args.results_dir.expanduser().resolve())
        output = args.output.expanduser().resolve()
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                             "savefig.bbox": None}):
            svg = render_plot(maps, sources, output)
        csv_path = output.with_suffix(".csv")
        write_csv(maps, csv_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{output}\n{svg}\n{csv_path}")
    for result in maps:
        print(f"  {result.name}: {len(result.keys)} points per policy; "
              "Oracle shown as the zero reference")


if __name__ == "__main__":
    main()
