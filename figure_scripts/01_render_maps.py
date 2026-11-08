"""Render WV, WA, and KTH test layouts as landscape PDFs and 600 px-wide SVGs.

Run from the repository root::

    uv run python figure_scripts/01_render_maps.py

The fewest-agent configuration is selected from each factorial test suite,
with ties resolved in file order. Outdoor maps use targets_06_a; KTH uses its
fewest-task configuration, also breaking ties in file order. All maps share
the evaluation renderer's agent colours and target-type labels. Capabilities
appear in the legends; a pie marker with equal agent-coloured slices and a
compact agent-ID range mark the shared start.
Outdoor maps also show roads and normalized terrain height; KTH shows
its fine occupancy floor plan. Map aspect ratios
are preserved within identical 5.6-by-4.8-inch pages; PDF text and markers are
vector graphics. Each PDF also gets an SVG with editable text and vectors,
600 px wide with the original aspect ratio. Reruns overwrite both formats.
Use --output-dir to choose a different destination.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Wedge
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
# Support the requested direct script invocation from any working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.evaluation_suite import (
    load_evaluation_suite,
    validate_evaluation_suite_locations,
)
from figure_scripts.export_utils import save_pdf_and_svg
from Real_Life_Maps.indoor_map_generation import load_walkable
from Real_Life_Maps.prepared_map import load_prepared_map
from simulation.agent import Agent
from simulation.rendering import draw_frame, _agent_color, _legend_handles


FIGSIZE = (5.6, 4.8)
# Match the outdoor axes' physical width and height to avoid equal-aspect
# centering adding whitespace on either side of the square terrain.
MAP_RECT = (0.025, 0.035, 0.775, 0.905)
INDOOR_MAP_RECT = (0.025, 0.105, 0.95, 0.83)
MAPS = (
    ("wv", "West Virginia (WV)", "WV_tobler_viewshed_64.pkl.gz",
     "wv_factorial_test_v1.json"),
    ("wa", "Washington (WA)", "WA_tobler_viewshed_64.pkl.gz",
     "wa_factorial_test_v1.json"),
    ("kth", "KTH indoor", "KTH_Indoor_map.pkl.gz", "KTH_indoor_big.json"),
)


def draw_scene(ax, graph, suite, agents, targets, *, indoor=False):
    """Reuse the test-loop actors with the selected static layout revealed."""
    # This is a labelled scenario illustration, rather than a policy
    # observation: show the configured types instead of UNKNOWN_TYPE.
    scene = graph.copy()
    scene.nodes[suite.source_position]["type"] = "source"
    for target in targets.targets:
        scene.nodes[target.position].update(
            type="target_unreached", rps_type=target.target_type)
    team = [Agent(suite.source_position, capabilities=capabilities)
            for capabilities in agents.capabilities]
    image = draw_frame(
        ax, scene, scene, team, scale=0.8, labels_inside=True,
        show_source=False, draw_background=not indoor)
    # Replace the overlapping per-agent circles with a single physical-size
    # pie so every co-located agent is visible, independent of map coordinates.
    for collection in ax.collections:
        if collection.get_zorder() == 11:
            collection.set_visible(False)
    draw_team_marker(ax, scene.nodes[suite.source_position]["pos"], len(team))
    # All agents share one physical start. Keep a single ID range there and
    # put individual capability sets in the legend instead of stacking boxes.
    labels = [text for text in ax.texts if text.get_text().startswith("A")]
    for label in labels:
        label.set_visible(False)
    if labels:
        label = labels[0]
        label.set_text(f"A0–A{len(team) - 1}" if len(team) > 1 else "A0")
        label.get_bbox_patch().set_edgecolor("#333333")
        label.set_visible(True)
    return image


def draw_team_marker(ax, position, count):
    """Draw an equal-slice team pie at the shared physical position."""
    diameter = 14
    radius = diameter / 2
    drawing = DrawingArea(diameter, diameter, clip=False)
    angle = 360 / count
    for index in range(count):
        drawing.add_artist(Wedge(
            (radius, radius), radius,
            90 - (index + 1) * angle, 90 - index * angle,
            facecolor=_agent_color(index), edgecolor="white", linewidth=0.4,
        ))
    drawing.add_artist(Circle(
        (radius, radius), radius, fill=False, edgecolor="black", linewidth=1,
    ))
    ax.add_artist(AnnotationBbox(
        drawing, position, xycoords="data", frameon=False, pad=0,
        box_alignment=(0.5, 0.5), annotation_clip=False, clip_on=False,
        zorder=11,
    ))


def draw_outdoor(fig, ax, graph, suite, agents, targets):
    image = draw_scene(ax, graph, suite, agents, targets)
    # Rasterize only the terrain to avoid seams between adjacent PDF cells.
    for patch in ax.patches:
        patch.set_rasterized(True)
    colorbar = fig.colorbar(image, cax=fig.add_axes((0.835, 0.13, 0.023, 0.57)))
    colorbar.set_label("Height", labelpad=5)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=9, width=0.6)
    draw_legend(fig, agents)


def draw_indoor(ax, metadata):
    """Use the fine occupancy background and its original image orientation."""
    walkable, _ = load_walkable(ROOT / "Real_Life_Maps" / "50052749")
    if tuple(walkable.shape) != tuple(metadata["fine_shape"]):
        raise ValueError("KTH occupancy dimensions do not match the prepared map")
    height, width = walkable.shape
    ax.imshow(
        walkable, cmap=mcolors.ListedColormap(("#30363B", "#F5F5F2")),
        vmin=0, vmax=1, origin="upper", interpolation="nearest",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
    )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)


def draw_layout(fig, ax, graph, suite, agents, targets):
    draw_scene(ax, graph, suite, agents, targets, indoor=True)
    draw_legend(fig, agents, indoor=True)


def draw_legend(fig, agents, *, indoor=False):
    handles = [Line2D(
        [], [], linestyle="none", marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor="black", label="Agents:",
    )]
    for index, capabilities in enumerate(agents.capabilities):
        values = ",".join(map(str, sorted(capabilities)))
        handles.append(Line2D(
            [], [], linestyle="none", marker="o", markersize=7,
            markerfacecolor=_agent_color(index), markeredgecolor="black",
            markeredgewidth=0.7, label=f"A{index}: {{{values}}}",
        ))
    target_handle = _legend_handles()[1]
    target_handle.set_label("Tasks")
    handles.append(target_handle)
    legend = fig.legend(
        handles=handles, loc="upper center" if indoor else "upper left",
        bbox_to_anchor=(0.5, 0.085) if indoor else (0.805, 0.91),
        ncols=(len(handles) + 1) // 2 if indoor else 1, fontsize=8,
        frameon=False, borderaxespad=0, handletextpad=0.5, labelspacing=0.5,
        handlelength=1, columnspacing=1.2,
    )
    legend.get_texts()[0].set_fontweight("bold")


def render_map(name, title, map_name, suite_name, destination):
    graph, metadata = load_prepared_map(ROOT / "Real_Life_Maps" / map_name)
    suite, suite_path = load_evaluation_suite(
        ROOT / "learning" / "evaluation_suites" / suite_name)
    validate_evaluation_suite_locations(suite, graph)
    agents = min(suite.agent_configurations, key=lambda item: item.agent_count)
    targets = min(suite.target_configurations, key=lambda item: item.target_count)
    if name in ("wv", "wa"):
        targets = next(item for item in suite.target_configurations
                       if item.id == "targets_06_a")
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    try:
        ax = fig.add_axes(INDOOR_MAP_RECT if name == "kth" else MAP_RECT)
        if name == "kth":
            draw_indoor(ax, metadata)
            draw_layout(fig, ax, graph, suite, agents, targets)
        else:
            draw_outdoor(fig, ax, graph, suite, agents, targets)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        fig.text(0.5 if name == "kth" else 0.4125, 0.99, title,
                 ha="center", va="top", fontsize=12)
        description = (
            f"{suite_path.name}: {agents.id} ({agents.agent_count} agents), "
            f"{targets.id} ({targets.target_count} tasks)"
        )
        svg_path = save_pdf_and_svg(
            fig, destination,
            metadata={"Title": title, "Subject": description,
                      "Creator": "figure_scripts/01_render_maps.py"},
        )
        print(f"{destination}\n{svg_path}\n  {description}")
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs",
        help="PDF/SVG destination (default: figure_scripts/outputs; overwrites existing figures)",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    destinations = [output_dir / f"map_{name}.pdf" for name, *_ in MAPS]
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.bbox": None}):
        for specification, destination in zip(MAPS, destinations):
            render_map(*specification, destination)


if __name__ == "__main__":
    main()
