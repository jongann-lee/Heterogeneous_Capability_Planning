"""
Visualization helpers for the multi-agent simulation.

Mirrors the rendering rules used in Real_Life_Maps/visualization.ipynb:
- terrain by height, obstacles in black, road edges in dark grey
- source = green circle, target_unreached = red X, target_reached = grey X
- each agent gets a single color; solid line = trajectory so far,
  dotted line = planned path
"""
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.collections as mc
from matplotlib.lines import Line2D
import networkx as nx

from simulation.domain import UNKNOWN_TYPE


DEFAULT_AGENT_COLORS = ["blue", "red", "green"]

DEAD_AGENT_COLOR = "grey"


def _agent_color(idx, agent_colors=None):
    colors = agent_colors if agent_colors is not None else DEFAULT_AGENT_COLORS
    if idx < len(colors):
        return colors[idx]
    return plt.cm.tab10(idx % 10)


def render_simulation_frame(env_map, blocked_env_graph, agents, turn_idx, output_path,
                            agent_colors=None, title=None, extra_paths=None):
    """Render one frame of the multi-agent simulation to PNG.

    Args:
        env_map: planner's view of the graph (used for target_reached state).
        blocked_env_graph: ground-truth graph (used for terrain, obstacles, roads).
        agents: list of Agent instances (each with .position, .trajectory, .planned_path).
        turn_idx: integer turn number — used in title and filename ordering.
        output_path: absolute path to write the PNG.
        agent_colors: optional list of matplotlib colors per agent.
        title: optional title override (default "Turn N").
        extra_paths: optional list of (path, color, linestyle, linewidth) tuples to
            overlay before agent markers. Used by the replan-debug renderer to
            show every candidate (agent, target) shortest path.
    """
    pos = nx.get_node_attributes(blocked_env_graph, 'pos')
    all_heights = [d.get('height', 0) for _, d in blocked_env_graph.nodes(data=True)]
    max_height = max(all_heights) if all_heights else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_height + 1)
    cmap_terrain = plt.cm.terrain

    xs = sorted(set(p[0] for p in pos.values()))
    ys = sorted(set(p[1] for p in pos.values()))
    cell_w = xs[1] - xs[0]
    cell_h = ys[1] - ys[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    for node, data in blocked_env_graph.nodes(data=True):
        x, y = pos[node]
        if data.get("type") == "obstacle":
            color = "black"
        else:
            color = cmap_terrain(norm(data.get('height', 0)))
        ax.add_patch(patches.Rectangle(
            (x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
            linewidth=0, facecolor=color
        ))

    road_segments = [[pos[u], pos[v]] for u, v, d in blocked_env_graph.edges(data=True) if d.get('is_road')]
    if road_segments:
        ax.add_collection(mc.LineCollection(road_segments, colors='#404040', linewidths=2.0, zorder=4))

    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        if len(agent.trajectory) >= 2:
            traj_segments = [[pos[agent.trajectory[k]], pos[agent.trajectory[k + 1]]]
                             for k in range(len(agent.trajectory) - 1)]
            ax.add_collection(mc.LineCollection(traj_segments, colors=color, linewidths=4.0, zorder=5))
        if len(agent.planned_path) >= 2:
            plan_segments = [[pos[agent.planned_path[k]], pos[agent.planned_path[k + 1]]]
                             for k in range(len(agent.planned_path) - 1)]
            ax.add_collection(mc.LineCollection(plan_segments, colors=color, linewidths=3.2,
                                                zorder=6, linestyles='dotted'))

    if extra_paths:
        for path, color, linestyle, linewidth in extra_paths:
            if path is None or len(path) < 2:
                continue
            segs = [[pos[path[k]], pos[path[k + 1]]] for k in range(len(path) - 1)]
            ax.add_collection(mc.LineCollection(
                segs, colors=color, linewidths=linewidth,
                zorder=4, linestyles=linestyle, alpha=0.7,
            ))

    src_pts, unreached_pts, reached_pts = [], [], []
    for node, data in env_map.nodes(data=True):
        t = data.get("type")
        if t == "source":
            src_pts.append(pos[node])
        elif t == "target_unreached":
            unreached_pts.append(pos[node])
        elif t == "target_reached":
            reached_pts.append(pos[node])

    if src_pts:
        ax.scatter([p[0] for p in src_pts], [p[1] for p in src_pts],
                   marker='o', s=220, facecolor='limegreen', edgecolor='darkgreen',
                   linewidths=1.5, zorder=10)
    if unreached_pts:
        ax.scatter([p[0] for p in unreached_pts], [p[1] for p in unreached_pts],
                   marker='x', s=200, color='red', linewidths=3, zorder=10)
    if reached_pts:
        ax.scatter([p[0] for p in reached_pts], [p[1] for p in reached_pts],
                   marker='x', s=200, color='grey', linewidths=3, zorder=10)

    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        x, y = pos[agent.position]
        ax.scatter([x], [y], marker='o', s=130, facecolor=color, edgecolor='black',
                   linewidths=1.2, zorder=11)

    ax.set_xlim(xs[0] - cell_w / 2, xs[-1] + cell_w / 2)
    ax.set_ylim(ys[0] - cell_h / 2, ys[-1] + cell_h / 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title if title is not None else f"Turn {turn_idx}")
    # Keep every frame on the same fixed-size canvas. Using bbox_inches="tight"
    # here makes the PNG dimensions change as floating labels move, which
    # prevents ffmpeg from encoding the sequence as one video stream.
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _agent_capability_text(agent_index, agent):
    values = ",".join(str(value) for value in sorted(agent.capabilities))
    return f"A{agent_index}: {{{values}}}"


def _target_type_text(target_type):
    return "T: ?" if target_type == UNKNOWN_TYPE else f"T: {target_type}"


def _legend_handles():
    """Proxy artists for marker/state meaning; types are shown as text."""
    return [
        Line2D([], [], marker="o", linestyle="None", markersize=10,
               markerfacecolor="white", markeredgecolor="black",
               label="agent (label = capabilities)"),
        Line2D([], [], marker="X", linestyle="None", markersize=11,
               markerfacecolor="red", markeredgecolor="black",
               label="live target (label = type)"),
        Line2D([], [], marker="X", linestyle="None", markersize=11,
               markerfacecolor=DEAD_AGENT_COLOR, markeredgecolor="black",
               label="serviced target"),
        Line2D([], [], marker="*", linestyle="None", markersize=14,
               markerfacecolor="gold", markeredgecolor="black", label="source"),
    ]


def render_frame(env_map, ground_truth, agents, turn_idx, output_path,
                 title=None, agent_xy=None):
    """Render one frame of the capability-based simulation to PNG.

    Agent colors identify trajectories only; they do not encode capabilities.
    Floating labels show ``A<index>: {capabilities}`` above each agent and
    ``T: <type>`` above each target (``T: ?`` until revealed).

    Args:
        env_map: planner's view (target types/reached state, source).
        ground_truth: reality (terrain, obstacles, roads, positions).
        agents: list of Agent (with .capabilities, .alive, .trajectory,
            .planned_path, .position).
        agent_xy: optional list of (x, y) coordinates, one per agent, giving
            each agent's *interpolated* position (used by the continuous-time
            driver to draw agents mid-edge). When None, agents are drawn at
            their current node.
    """
    pos = nx.get_node_attributes(ground_truth, 'pos')
    all_heights = [d.get('height', 0) for _, d in ground_truth.nodes(data=True)]
    max_height = max(all_heights) if all_heights else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_height + 1)
    cmap_terrain = plt.cm.terrain

    xs = sorted(set(p[0] for p in pos.values()))
    ys = sorted(set(p[1] for p in pos.values()))
    cell_w = xs[1] - xs[0] if len(xs) > 1 else 1
    cell_h = ys[1] - ys[0] if len(ys) > 1 else 1

    # A 9x9-inch map panel plus a compact legend column. Manual positioning
    # avoids the large vertical whitespace produced by tight_layout + equal
    # aspect while retaining identical dimensions across frames.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_position([0.025, 0.04, 0.75, 0.90])

    for node, data in ground_truth.nodes(data=True):
        x, y = pos[node]
        color = "black" if data.get("type") == "obstacle" else cmap_terrain(norm(data.get('height', 0)))
        ax.add_patch(patches.Rectangle(
            (x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
            linewidth=0, facecolor=color))

    road_segments = [[pos[u], pos[v]] for u, v, d in ground_truth.edges(data=True) if d.get('is_road')]
    if road_segments:
        ax.add_collection(mc.LineCollection(road_segments, colors='#404040', linewidths=2.0, zorder=4))

    def _apos(i, agent):
        return agent_xy[i] if agent_xy is not None else pos[agent.position]

    # Trajectory (solid) + planned path (dotted), colored per agent identity.
    for i, agent in enumerate(agents):
        color = DEAD_AGENT_COLOR if not agent.alive else _agent_color(i)
        traj_pts = [pos[nd] for nd in agent.trajectory]
        if agent_xy is not None:
            traj_pts.append(agent_xy[i])   # extend the trail to the interpolated point
        if len(traj_pts) >= 2:
            segs = [[traj_pts[k], traj_pts[k + 1]] for k in range(len(traj_pts) - 1)]
            ax.add_collection(mc.LineCollection(segs, colors=color, linewidths=4.0, zorder=5))
        if agent.alive and len(agent.planned_path) >= 2:
            segs = [[pos[agent.planned_path[k]], pos[agent.planned_path[k + 1]]]
                    for k in range(len(agent.planned_path) - 1)]
            ax.add_collection(mc.LineCollection(segs, colors=color, linewidths=3.2,
                                                zorder=6, linestyles='dotted'))

    # Source (star), targets (X by known/eliminated color), agents (circle).
    src_pts = [pos[n] for n, d in env_map.nodes(data=True) if d.get("type") == "source"]
    if src_pts:
        ax.scatter([p[0] for p in src_pts], [p[1] for p in src_pts], marker='*',
                   s=420, facecolor='gold', edgecolor='black', linewidths=1.2, zorder=10)

    # Agents first, then targets ON TOP: when an agent occupies a target node
    # (e.g. it just engaged, or died there) the target's revealed type stays
    # visible -- the agent's presence is still shown by its trajectory line.
    for i, agent in enumerate(agents):
        point = _apos(i, agent)
        color = DEAD_AGENT_COLOR if not agent.alive else _agent_color(i)
        ax.scatter([point[0]], [point[1]], marker='o', s=140,
                   facecolor=color, edgecolor='black', linewidths=1.2, zorder=11)
        ax.annotate(
            _agent_capability_text(i, agent),
            xy=point,
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.85),
            zorder=15,
        )

    for n, d in env_map.nodes(data=True):
        state = d.get("type")
        if state not in ("target_unreached", "target_reached"):
            continue
        point = pos[n]
        color = "red" if state == "target_unreached" else DEAD_AGENT_COLOR
        target_type = d.get("rps_type", UNKNOWN_TYPE)
        ax.scatter([point[0]], [point[1]], marker='X', s=200,
                   facecolor=color, edgecolor='black', linewidths=1.0, zorder=12)
        ax.annotate(
            _target_type_text(target_type),
            xy=point,
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor=color, alpha=0.85),
            zorder=14,
        )

    ax.legend(handles=_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=10,
              frameon=True, title="legend")

    ax.set_xlim(xs[0] - cell_w / 2, xs[-1] + cell_w / 2)
    ax.set_ylim(ys[0] - cell_h / 2, ys[-1] + cell_h / 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title if title is not None else f"Turn {turn_idx}")
    # Do not call tight_layout: it recenters the equal-aspect map inside the
    # legend-reserved rectangle and recreates the excessive whitespace.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# Compatibility for notebooks/scripts written before the package cleanup.
RPS_DEAD_COLOR = DEAD_AGENT_COLOR
render_rps_frame = render_frame


def clear_frame_dir(frames_dir):
    """Delete existing frame_*.png from frames_dir (creates the dir if absent)."""
    p = Path(frames_dir)
    p.mkdir(parents=True, exist_ok=True)
    for f in p.glob("frame_*.png"):
        f.unlink()


def clear_debug_dir(debug_dir):
    """Delete existing replan_*.png from debug_dir (creates the dir if absent)."""
    p = Path(debug_dir)
    p.mkdir(parents=True, exist_ok=True)
    for f in p.glob("replan_*.png"):
        f.unlink()


def render_replan_debug_frame(env_map, blocked_env_graph, agents, replan_idx,
                              turn_idx, output_path, agent_colors=None):
    """Render a debug frame showing every (agent, remaining target) shortest path
    as a dashed line in the agent's color, on top of the regular simulation view.

    Saved separately from the MP4 frames.
    """
    import networkx as nx  # local to keep module-level lazy

    targets = [
        n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
    ]
    extras = []
    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        for t in targets:
            try:
                path = nx.shortest_path(env_map, agent.position, t, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            extras.append((path, color, "dashed", 2.0))

    title = f"Replan #{replan_idx} @ turn {turn_idx} ({len(targets)} targets remaining)"
    render_simulation_frame(
        env_map, blocked_env_graph, agents, turn_idx, output_path,
        agent_colors=agent_colors, title=title, extra_paths=extras,
    )


def make_mp4_from_frames(frames_dir, output_mp4, fps=4):
    """Combine frame_*.png in frames_dir into an MP4 at the given fps via ffmpeg."""
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return None
    # Normalize every frame to the first frame's aspect ratio and an even
    # width/height. libx264 with yuv420p rejects odd dimensions, and this also
    # tolerates older frame sets whose tight bounding boxes varied slightly.
    first_image = plt.imread(frames[0])
    height, width = first_image.shape[:2]
    width += width % 2
    height += height % 2
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    pattern = str(Path(frames_dir) / "frame_%04d.png")
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-vf", video_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_mp4
