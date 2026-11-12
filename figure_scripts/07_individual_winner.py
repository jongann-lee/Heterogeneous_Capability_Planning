"""Show per-scenario policy wins as one square-grid figure per map.

    uv run python figure_scripts/07_individual_winner.py

Use the same 1103, Scout Then Engage, and Engage While Scout results and
colors as figure 02. WV/WA each use 5 rows and 36 columns; KTH uses a 1x9
grid of larger squares. Fill from the top left, down each column, grouping wins in
that policy order. Grid positions represent win counts, not suite ordering.

Lowest makespan wins. Ties within 1e-6 relative/absolute tolerance are counted
separately and colored gray if present. Match map, scenario, instance, and
seed metadata through the existing figure-02 validation before comparing.

Export 07_{WV,WA,KTH}_individual_winner.pdf, editable 600 px-wide SVGs, and
per-map CSVs under figure_scripts/outputs, replacing them on reruns. Match
figure 02's page width and horizontal plot bounds for stacking below its
charts. Show win counts without map headings or a color legend.
--output-dir selects another destination. Inputs are read only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.export_utils import save_pdf_and_svg

# Reuse the checked source selection and comparison contracts of figure 02.
regret = import_module("figure_scripts.02_main_regrets")
POLICIES = ("scout-then-execute", "opt-by-part", "learned")
LABELS = dict(zip(regret.POLICIES, regret.LABELS))
COLORS = dict(zip(regret.POLICIES, regret.COLORS))
LABELS["tie"] = "Tie"
COLORS["tie"] = "#B8BCC1"
MAPS = (("WV", 5, 36), ("WA", 5, 36), ("KTH", 1, 9))
GRID_LEFT = 0.15
GRID_WIDTH = 0.82
GRID_ROWS = 5
GRID_COLUMNS = 36
FIGSIZE = (regret.FIGSIZE[0],
           regret.FIGSIZE[0] * GRID_WIDTH * GRID_ROWS / GRID_COLUMNS + 0.53)


@dataclass(frozen=True)
class ScenarioWinner:
    scenario_id: str
    episode_index: int
    makespans: tuple[float, ...]
    winner: str


@dataclass(frozen=True)
class MapWinners:
    name: str
    rows: int
    columns: int
    scenarios: tuple[ScenarioWinner, ...]

    @property
    def counts(self):
        return Counter(scenario.winner for scenario in self.scenarios)


def winning_policy(makespans):
    best = min(makespans)
    winners = [policy for policy, value in zip(POLICIES, makespans)
               if math.isclose(value, best, rel_tol=1e-6, abs_tol=1e-6)]
    return winners[0] if len(winners) == 1 else "tie"


def load_winners(results_dir):
    maps, sources = [], {}
    for name, rows, columns in MAPS:
        paths = [results_dir / filename for filename in regret.RESULT_FILES[name]]
        payloads = [regret.load_result(path, policy)[0]
                    for path, policy in zip(paths, regret.POLICIES)]
        note = regret.validate_comparison(name, payloads)
        if note:
            raise ValueError(f"{name}: individual winners require identical starts; {note}")
        indexed = {
            policy: {regret.scenario_key(record): record for record in payload["scenarios"]}
            for policy, payload in zip(regret.POLICIES, payloads)
        }
        scenarios = []
        for key in indexed[POLICIES[0]]:
            makespans = tuple(float(indexed[policy][key]["makespan"]) for policy in POLICIES)
            scenarios.append(ScenarioWinner(key[0], key[1], makespans, winning_policy(makespans)))
        if len(scenarios) != rows * columns:
            raise ValueError(f"{name}: {rows}x{columns} grid needs {rows * columns} "
                             f"scenarios, found {len(scenarios)}")
        order = {policy: index for index, policy in enumerate((*POLICIES, "tie"))}
        scenarios.sort(key=lambda item: (order[item.winner], item.scenario_id, item.episode_index))
        maps.append(MapWinners(name, rows, columns, tuple(scenarios)))
        sources[name] = paths
    return maps, sources


def draw_grid(ax, result):
    # Identical physical grid bounds, with KTH cells four times larger.
    width, height = GRID_COLUMNS, GRID_ROWS
    size = min(width / result.columns, height / result.rows)
    x_offset = (width - result.columns * size) / 2
    y_offset = (height - result.rows * size) / 2
    gap = size * 0.08
    for index, scenario in enumerate(result.scenarios):
        column, row = divmod(index, result.rows)
        square = Rectangle(
            (x_offset + column * size + gap / 2, y_offset + row * size + gap / 2),
            size - gap, size - gap, facecolor=COLORS[scenario.winner], linewidth=0,
        )
        square.set_gid(f"{result.name}-scenario-{index + 1}-{scenario.winner}")
        ax.add_patch(square)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def render_map(result, sources, output):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    try:
        categories = [*POLICIES]
        if result.counts["tie"]:
            categories.append("tie")
        grid_height = FIGSIZE[0] * GRID_WIDTH * GRID_ROWS / GRID_COLUMNS
        ax = fig.add_axes((GRID_LEFT, 0.45 / FIGSIZE[1],
                           GRID_WIDTH, grid_height / FIGSIZE[1]))
        draw_grid(ax, result)
        for offset, policy in enumerate(categories):
            count = result.counts[policy]
            label = f"{count} win" + ("" if count == 1 else "s") \
                if policy != "tie" else f"{count} tied"
            x = GRID_LEFT + GRID_WIDTH * (offset + 0.5) / len(categories)
            fig.text(x, 0.19 / FIGSIZE[1], label, ha="center", va="center", fontsize=14,
                     fontweight="bold", color=COLORS[policy])
        description = (
            "One square per evaluated scenario. Minimum makespan among Scout Then Engage, "
            f"{LABELS['opt-by-part']}, and CAAT Graph + HGT (1103) determines the winner. "
            "Fill downward within columns, then move right, grouped by policy, "
            "not original scenario order. "
            f"{result.name}: {result.rows} rows x {result.columns} columns. "
            "Ties: gray, if present. "
            "Inputs: " + "; ".join(map(str, sources))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        return save_pdf_and_svg(fig, output, metadata={
            "Title": f"{result.name} individual scenario winners", "Subject": description,
            "Creator": "figure_scripts/07_individual_winner.py",
        })
    finally:
        plt.close(fig)


def write_csv(maps, output):
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["map", "square", "row", "column", "scenario_id", "episode_index",
                         "winner", *(f"{policy}_makespan" for policy in POLICIES)])
        for result in maps:
            for index, scenario in enumerate(result.scenarios):
                column, row = divmod(index, result.rows)
                writer.writerow([result.name, index + 1, row + 1, column + 1,
                                 scenario.scenario_id, scenario.episode_index,
                                 LABELS[scenario.winner], *scenario.makespans])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args(argv)
    try:
        maps, sources = load_winners(args.results_dir.expanduser().resolve())
        output_dir = args.output_dir.expanduser().resolve()
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                             "savefig.bbox": None}):
            for result in maps:
                output = output_dir / f"07_{result.name}_individual_winner.pdf"
                svg = render_map(result, sources[result.name], output)
                csv_path = output.with_suffix(".csv")
                write_csv((result,), csv_path)
                print(f"{output}\n{svg}\n{csv_path}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for result in maps:
        print(f"  {result.name}: " + "; ".join(
            f"{LABELS[policy]} {result.counts[policy]}" for policy in (*POLICIES, "tie")))


if __name__ == "__main__":
    main()
