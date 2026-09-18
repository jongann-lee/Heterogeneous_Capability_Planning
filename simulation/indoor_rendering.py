"""Render indoor episodes over the fine occupancy map.

The prepared indoor graph stores a fine-pixel route for each directed edge.
Those routes are used for trails, plans, and positions between event times so
that a shortcut never appears to pass through a wall.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from simulation.domain import UNKNOWN_TYPE
from simulation.rendering import (
    DEFAULT_AGENT_COLORS,
    DEAD_AGENT_COLOR,
    _stacked_agent_label_offsets,
)


def _edge_points(graph, first, second):
    data = graph.get_edge_data(first, second)
    if data is not None and "fine_path" in data:
        return [(float(col), float(row)) for row, col in data["fine_path"]]
    return [graph.nodes[node]["pos"] for node in (first, second)]


def interpolate_edge(graph, first, second, fraction):
    """Return a position along an edge's stored fine-pixel route."""
    points = _edge_points(graph, first, second)
    if len(points) < 2:
        return tuple(points[0])
    progress = min(1.0, max(0.0, float(fraction))) * (len(points) - 1)
    index = min(int(progress), len(points) - 2)
    portion = progress - index
    x0, y0 = points[index]
    x1, y1 = points[index + 1]
    return (x0 + portion * (x1 - x0), y0 + portion * (y1 - y0))


def _route_points(graph, nodes):
    if not nodes:
        return []
    points = [graph.nodes[nodes[0]]["pos"]]
    for first, second in zip(nodes, nodes[1:]):
        if first == second:
            continue
        segment = _edge_points(graph, first, second)
        points.extend(segment[1:])
    return points


def _load_background(graph, map_path, metadata, raster_path):
    """Use the local fine raster when present; otherwise draw stored routes."""
    fine_shape = metadata.get("fine_shape")
    if fine_shape is None or len(fine_shape) != 2:
        raise ValueError("indoor rendering requires prepared-map fine_shape")
    shape = tuple(map(int, fine_shape))
    if raster_path is not None:
        raster = np.load(raster_path, allow_pickle=False)
        if raster.shape != shape or raster.ndim != 2:
            raise ValueError("indoor raster shape does not match prepared map")
        return np.asarray(raster != 0, dtype=bool)

    candidates = []
    if map_path is not None:
        parent = Path(map_path).resolve().parent
        candidates.extend((parent / "50052749", parent))
    source = metadata.get("source")
    if source:
        candidates.append(Path(source))
    for directory in candidates:
        if ((directory / "occ_map.npy").is_file()
                and (directory / "valid_space.npy").is_file()):
            from Real_Life_Maps.indoor_map_generation import load_walkable
            walkable, _ = load_walkable(directory)
            if walkable.shape == shape:
                return walkable.astype(bool)

    # The prepared graph is portable even without its source rasters. Its
    # exact edge routes still provide a useful, though thinner, floor plan.
    background = np.zeros(shape, dtype=bool)
    for _, _, data in graph.edges(data=True):
        path = data.get("fine_path")
        if path:
            rows, cols = zip(*path)
            background[rows, cols] = True
    for data in graph.nodes.values():
        row, col = data["dense_anchor"]
        background[row, col] = True
    return background


class IndoorRenderer:
    """Reusable frame renderer for one prepared indoor terrain."""

    def __init__(self, graph, *, map_path=None, metadata=None, raster_path=None):
        self.graph = graph
        self.metadata = dict(metadata or {})
        self.walkable = _load_background(
            graph, map_path, self.metadata, raster_path)
        self.height, self.width = self.walkable.shape
        self.background = np.where(self.walkable, 0.96, 0.16)

    def interpolate_edge(self, first, second, fraction):
        return interpolate_edge(self.graph, first, second, fraction)

    def __call__(self, env_map, ground_truth, agents, frame_index, output_path,
                 *, title=None, agent_xy=None):
        """Write one frame using planner-visible target state."""
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"render frame already exists: {output_path}")
        fig, ax = plt.subplots(figsize=(15, 10))
        try:
            ax.set_position([0.025, 0.06, 0.75, 0.86])
            ax.imshow(self.background, cmap="gray", vmin=0, vmax=1,
                      origin="upper", interpolation="nearest",
                      extent=(-0.5, self.width - 0.5,
                              self.height - 0.5, -0.5))

            observed = [self.graph.nodes[node]["pos"]
                        for node, data in env_map.nodes(data=True)
                        if data.get("observed_node", False)]
            if observed:
                ax.scatter(*zip(*observed), s=8, c="#111111",
                           linewidths=0, alpha=0.95, zorder=2,
                           rasterized=True)

            sources = [node for node, data in env_map.nodes(data=True)
                       if data.get("type") == "source"]
            if sources:
                points = [self.graph.nodes[node]["pos"] for node in sources]
                ax.scatter(*zip(*points), marker="*", s=210, c="#e8b400",
                           edgecolors="black", linewidths=0.8, zorder=6)

            agent_points = [
                agent_xy[index] if agent_xy is not None
                else self.graph.nodes[agent.position]["pos"]
                for index, agent in enumerate(agents)
            ]
            label_offsets = _stacked_agent_label_offsets(
                agent_points, base_offset=8, spacing=18)
            for index, agent in enumerate(agents):
                color = (DEFAULT_AGENT_COLORS[index % len(DEFAULT_AGENT_COLORS)]
                         if agent.alive else DEAD_AGENT_COLOR)
                trail = _route_points(self.graph, agent.trajectory)
                position = agent_points[index]
                if trail and trail[-1] != position:
                    trail.append(position)
                if len(trail) > 1:
                    ax.plot(*zip(*trail), color=color, linewidth=2.2,
                            alpha=0.9, zorder=3)
                if agent.alive and len(agent.planned_path) > 1:
                    planned = _route_points(self.graph, agent.planned_path)
                    ax.plot(*zip(*planned), color=color, linewidth=1.6,
                            linestyle=(0, (3, 2)), alpha=0.8, zorder=4)
                ax.scatter(*position, s=95, c=color, edgecolors="black",
                           linewidths=0.8, zorder=8)
                ax.annotate(f"A{index}", xy=position, xytext=label_offsets[index],
                            textcoords="offset points", ha="center",
                            fontsize=9, fontweight="bold", zorder=10,
                            bbox={"boxstyle": "round,pad=0.12",
                                  "facecolor": "white", "edgecolor": color,
                                  "alpha": 0.9})

            for node, data in env_map.nodes(data=True):
                state = data.get("type")
                if state not in ("target_unreached", "target_reached"):
                    continue
                position = self.graph.nodes[node]["pos"]
                color = "#d62728" if state == "target_unreached" else "#777777"
                ax.scatter(*position, marker="X", s=115, c=color,
                           edgecolors="black", linewidths=0.6, zorder=9)
                value = data.get("rps_type", UNKNOWN_TYPE)
                label = "?" if value == UNKNOWN_TYPE else str(value)
                ax.annotate(f"T:{label}", xy=position, xytext=(0, -17),
                            textcoords="offset points", ha="center",
                            fontsize=8, fontweight="bold", zorder=10,
                            bbox={"boxstyle": "round,pad=0.12",
                                  "facecolor": "white", "alpha": 0.85})

            handles = [
                Line2D([], [], marker="o", color="none", markerfacecolor="#111111",
                       markersize=5, label="Observed node"),
                Line2D([], [], marker="*", color="none", markerfacecolor="#e8b400",
                       markeredgecolor="black", markersize=12, label="Source"),
                Line2D([], [], marker="X", color="none", markerfacecolor="#d62728",
                       markeredgecolor="black", markersize=10, label="Live target"),
                Line2D([], [], marker="X", color="none", markerfacecolor="#777777",
                       markeredgecolor="black", markersize=10, label="Serviced target"),
            ]
            for index, agent in enumerate(agents):
                color = (DEFAULT_AGENT_COLORS[index % len(DEFAULT_AGENT_COLORS)]
                         if agent.alive else DEAD_AGENT_COLOR)
                capabilities = ",".join(map(str, sorted(agent.capabilities)))
                status = " (dead)" if not agent.alive else ""
                handles.append(Line2D([], [], color=color, linewidth=3,
                                      label=f"A{index} {{{capabilities}}}{status}"))
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(1.01, 1), fontsize=9, title="Legend")
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(self.height - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.suptitle(title or f"Frame {frame_index}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=100)
        finally:
            plt.close(fig)
