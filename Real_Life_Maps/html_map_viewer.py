"""Create a standalone, zoomable node-coordinate viewer for prepared maps.

Run from the repository root, for example::

    uv run python -m Real_Life_Maps.html_map_viewer \
        Real_Life_Maps/KTH_Indoor_map.pkl.gz --open

The input is a trusted prepared ``.pkl.gz`` map. The viewer embeds its terrain
image and node coordinates, so the resulting HTML needs no Python or map file.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import webbrowser

import numpy as np
from PIL import Image

from Real_Life_Maps.prepared_map import load_prepared_map


TEMPLATE = Path(__file__).with_name("html_map_viewer_template.html")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "cache" / "html_map_viewer"


def _walkable_background(map_path: Path, metadata: dict,
                         raster_path: Path | None) -> np.ndarray | None:
    """Find the indoor occupancy raster when it is available beside the map."""
    if raster_path is not None:
        raster = np.load(raster_path, allow_pickle=False)
        if raster.ndim != 2:
            raise ValueError("--raster must be a two-dimensional .npy array")
        return np.asarray(raster != 0, dtype=bool)

    if "fine_shape" not in metadata:
        return None
    source = metadata.get("source")
    candidates = []
    if source:
        candidates.extend((
            Path(source),
            map_path.parent / Path(source).name,
            Path(__file__).resolve().parent / Path(source).name,
        ))
    candidates.append(map_path.parent)
    for directory in candidates:
        if (directory / "occ_map.npy").is_file() and \
                (directory / "valid_space.npy").is_file():
            from Real_Life_Maps.indoor_map_generation import load_walkable
            walkable, _ = load_walkable(directory)
            return walkable.astype(bool)
        if (directory / "walkable.npy").is_file():
            raster = np.load(directory / "walkable.npy", allow_pickle=False)
            if raster.ndim == 2:
                return raster.astype(bool)
    return None


def _png_data_uri(image: Image.Image) -> str:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def _node_label(node) -> str:
    if isinstance(node, tuple):
        return "(" + ", ".join(map(str, node)) + ")"
    return str(node)


def _scene(map_path: Path, graph, metadata: dict,
           raster_path: Path | None) -> dict:
    positions = {node: tuple(map(float, data["pos"]))
                 for node, data in graph.nodes(data=True)}
    walkable = _walkable_background(map_path, metadata, raster_path)
    fine_shape = metadata.get("fine_shape")
    if walkable is not None:
        height, width = walkable.shape
        if fine_shape is not None and list(walkable.shape) != list(fine_shape):
            raise ValueError("occupancy raster shape differs from map metadata")
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[walkable] = (248, 248, 244)
        pixels[~walkable] = (37, 43, 49)
        background = Image.fromarray(pixels, mode="RGB")
        transform = lambda xy: xy
        background_note = "Fine occupancy"
    elif fine_shape is not None:
        height, width = map(int, fine_shape)
        background = Image.new("RGB", (width, height), (247, 247, 242))
        transform = lambda xy: xy
        background_note = "Occupancy raster unavailable; showing graph nodes"
    else:
        xs = [xy[0] for xy in positions.values()]
        ys = [xy[1] for xy in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        span = max(x_max - x_min, y_max - y_min, 1.0)
        factor = min(1.0, 1400.0 / span)
        width = max(2, int(round((x_max - x_min) * factor)) + 1)
        height = max(2, int(round((y_max - y_min) * factor)) + 1)
        background = Image.new("RGB", (width, height), (239, 239, 232))
        transform = lambda xy: ((xy[0] - x_min) * factor,
                                (y_max - xy[1]) * factor)
        heights = [float(data.get("height", 0.0))
                   for _, data in graph.nodes(data=True)]
        low, high = min(heights), max(heights)
        points = background.load()
        for node, data in graph.nodes(data=True):
            value = float(data.get("height", 0.0))
            fraction = (value - low) / (high - low) if high > low else 0.5
            color = (int(226 - 64 * fraction),
                     int(231 - 47 * fraction),
                     int(207 - 104 * fraction))
            x, y = transform(positions[node])
            points[min(width - 1, max(0, round(x))),
                   min(height - 1, max(0, round(y)))] = color
        background_note = "Node elevation"

    nodes = []
    for node, position in positions.items():
        x, y = transform(position)
        nodes.append([_node_label(node), round(x, 3), round(y, 3),
                      round(position[0], 3), round(position[1], 3)])
    return {
        "name": map_path.name,
        "width": width,
        "height": height,
        "background": _png_data_uri(background),
        "background_note": background_note,
        "nodes": nodes,
    }


def make_viewer(map_path: Path | str, output_path: Path | str | None = None,
                raster_path: Path | str | None = None) -> Path:
    """Write one self-contained HTML viewer and return its path."""
    map_path = Path(map_path).expanduser().resolve()
    if not map_path.name.endswith(".pkl.gz"):
        raise ValueError("map input must be a prepared .pkl.gz file")
    graph, metadata = load_prepared_map(map_path)
    raster = None if raster_path is None else Path(raster_path).expanduser().resolve()
    scene = _scene(map_path, graph, metadata, raster)
    data = json.dumps(scene, separators=(",", ":"), ensure_ascii=False)
    data = data.replace("</", "<\\/")
    output = (DEFAULT_OUTPUT_DIR / (map_path.name[:-7] + "_viewer.html")
              if output_path is None else Path(output_path).expanduser())
    output = output.resolve()
    html = TEMPLATE.read_text(encoding="utf-8").replace("__MAP_DATA__", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="validated prepared .pkl.gz map")
    parser.add_argument("--output", type=Path,
                        help="HTML path; defaults under cache/html_map_viewer")
    parser.add_argument("--raster", type=Path,
                        help="optional 2D .npy walkability raster for background")
    parser.add_argument("--open", action="store_true",
                        help="open the generated HTML in your browser")
    args = parser.parse_args(argv)
    try:
        output = make_viewer(args.map, args.output, args.raster)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(output)
    if args.open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
