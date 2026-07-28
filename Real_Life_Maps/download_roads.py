"""
Download road data from OpenStreetMap for the WV DEM region
and rasterize it onto an N x N grid matching the DEM resampling.

Outputs:
  - road_nodes: set of (row, col) grid cells that a road passes through
  - road_edges: set of ((r1,c1),(r2,c2)) grid edges that lie along roads

Saved as a pickle file for use by RealTerrainGrid.
"""

import numpy as np
import pickle
import rasterio
from pyproj import Transformer
import osmnx as ox


def get_dem_metadata(dem_path):
    """Read bounds and CRS from the DEM file."""
    with rasterio.open(dem_path) as ds:
        bounds = ds.bounds  # left, bottom, right, top
        crs = ds.crs
    return bounds, crs


def fetch_roads(bounds, crs):
    """
    Fetch road geometries from OSM within the DEM bounds.
    Converts DEM CRS bounds to WGS84 (lat/lon) for the OSM query.
    Returns a GeoDataFrame of road edges.
    """
    # Transform bounds from DEM CRS to WGS84
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = transformer.transform(bounds.right, bounds.top)

    print(f"DEM bounds (native CRS): {bounds}")
    print(f"DEM bounds (WGS84): N={lat_max:.6f}, S={lat_min:.6f}, E={lon_max:.6f}, W={lon_min:.6f}")

    # Download road network from OSM
    # osmnx 2.x bbox order: (west, south, east, north)
    G_roads = ox.graph_from_bbox(
        bbox=(lon_min, lat_min, lon_max, lat_max),
        network_type="drive",
        retain_all=True,
    )
    edges_gdf = ox.graph_to_gdfs(G_roads, nodes=False, edges=True)
    print(f"Fetched {len(edges_gdf)} road segments from OSM")
    return edges_gdf, (lat_min, lat_max, lon_min, lon_max)


def rasterize_roads(edges_gdf, bounds, crs, n_size, wgs84_bounds):
    """
    Map OSM road geometries onto the N x N grid.

    The DEM is rotated 90 degrees clockwise (np.rot90(k=-1)) before being
    used in RealTerrainGrid. We apply the same transform here so road
    coordinates align with the graph's node indices.

    rot90(k=-1) on an NxN grid: (r, c) -> (c, N-1-r)

    Returns road_nodes (set of grid cells) and road_edges (set of grid edge tuples).
    """
    lat_min, lat_max, lon_min, lon_max = wgs84_bounds

    # Transformer: WGS84 -> DEM CRS (so we can work in the DEM's coordinate space)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # DEM extent in native CRS
    x_min, y_min = bounds.left, bounds.bottom
    x_max, y_max = bounds.right, bounds.top
    x_range = x_max - x_min
    y_range = y_max - y_min

    road_nodes = set()
    road_edges = set()

    for _, row in edges_gdf.iterrows():
        geom = row["geometry"]
        coords = list(geom.coords)  # list of (lon, lat) points

        prev_cell = None
        for lon, lat in coords:
            # Convert to DEM CRS
            x, y = transformer.transform(lon, lat)

            # Map to raw grid indices (row 0 = top of DEM = y_max)
            raw_col = int((x - x_min) / x_range * n_size)
            raw_r = int((y_max - y) / y_range * n_size)

            # Apply the same rot90(k=-1) transform used on the height grid
            r = raw_col
            col = n_size - 1 - raw_r

            # Clamp to valid range
            col = max(0, min(n_size - 1, col))
            r = max(0, min(n_size - 1, r))

            cell = (r, col)
            road_nodes.add(cell)

            # Connect consecutive road vertices that land on adjacent grid cells
            if prev_cell is not None and prev_cell != cell:
                # Bresenham-style: walk intermediate cells so we don't skip any
                for intermediate in _walk_cells(prev_cell, cell):
                    road_nodes.add(intermediate)
                # Add edges between consecutive cells along the road
                walked = _walk_cells(prev_cell, cell)
                for i in range(len(walked) - 1):
                    road_edges.add((walked[i], walked[i + 1]))
                    road_edges.add((walked[i + 1], walked[i]))  # both directions

            prev_cell = cell

    print(f"Road nodes: {len(road_nodes)}, Road edges: {len(road_edges)}")
    return road_nodes, road_edges


def _walk_cells(start, end):
    """4-connected line between two grid cells.

    Standard Bresenham emits 8-connected lines; when it steps diagonally, the
    two cells only touch at a corner and there is no edge between them in the
    4-connected grid graph. We split every diagonal step into two orthogonal
    steps so consecutive cells in the returned list always differ by exactly
    one row OR one column.
    """
    r0, c0 = start
    r1, c1 = end
    cells = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        step_r = e2 > -dc
        step_c = e2 < dr
        if step_r and step_c:
            err -= dc
            r += sr
            cells.append((r, c))
            err += dr
            c += sc
        elif step_r:
            err -= dc
            r += sr
        else:
            err += dr
            c += sc

    return cells


def download_and_save(dem_path="WV_DEM.tif", n_size=64, output_path="WV_roads.pkl"):
    """Main entry point: fetch roads, rasterize, save."""
    bounds, crs = get_dem_metadata(dem_path)
    edges_gdf, wgs84_bounds = fetch_roads(bounds, crs)
    road_nodes, road_edges = rasterize_roads(edges_gdf, bounds, crs, n_size, wgs84_bounds)

    data = {
        "road_nodes": road_nodes,
        "road_edges": road_edges,
        "n_size": n_size,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved road data to {output_path}")

    return road_nodes, road_edges


if __name__ == "__main__":
    download_and_save()
