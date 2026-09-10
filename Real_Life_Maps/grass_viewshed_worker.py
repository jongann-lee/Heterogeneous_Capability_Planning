"""Internal worker executed inside a temporary GRASS GIS session."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, required=True)
    parser.add_argument("--observer-height", type=float, required=True)
    parser.add_argument("--target-height", type=float, required=True)
    parser.add_argument("--memory", type=int, required=True)
    return parser


def main(argv=None):
    try:
        import grass.script as gs
        import grass.script.array as garray
    except ImportError as error:
        raise RuntimeError(
            "the viewshed worker must be launched by the GRASS executable"
        ) from error

    args = _parser().parse_args(argv)
    anchors = np.load(args.anchors, allow_pickle=False)
    rows = np.asarray(anchors["rows"], dtype=np.int64)
    cols = np.asarray(anchors["cols"], dtype=np.int64)
    coordinates = np.asarray(anchors["coordinates"], dtype=np.float64)
    if coordinates.shape != (len(rows), 2) or len(cols) != len(rows):
        raise ValueError("invalid viewshed anchor file")

    elevation_name = "dense_elevation"
    viewshed_name = "current_viewshed"
    gs.run_command(
        "r.in.gdal", input=str(args.dem), output=elevation_name,
        flags="o", overwrite=True, quiet=True)
    gs.run_command("g.region", raster=elevation_name, quiet=True)

    visible = np.zeros((len(rows), len(rows)), dtype=bool)
    for index, (east, north) in enumerate(coordinates):
        gs.run_command(
            "r.viewshed",
            input=elevation_name,
            output=viewshed_name,
            coordinates=f"{east:.9f},{north:.9f}",
            observer_elevation=float(args.observer_height),
            target_elevation=float(args.target_height),
            max_distance=float(args.max_distance),
            memory=int(args.memory),
            flags="b",
            overwrite=True,
            quiet=True,
        )
        raster = np.asarray(
            garray.array(mapname=viewshed_name, null=0), dtype=np.float64)
        visible[index] = raster[rows, cols] > 0.0
        visible[index, index] = True
        if (index + 1) % 64 == 0 or index + 1 == len(rows):
            print(
                f"  viewsheds: {index + 1}/{len(rows)}",
                flush=True,
            )
    np.save(args.output, visible, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
