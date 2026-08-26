import hashlib
import math
from pathlib import Path as FilePath
import pickle
import tempfile

import networkx as nx
import numpy as np
from matplotlib.path import Path


VISIBILITY_CACHE_SCHEMA = 1
DEFAULT_VISIBILITY_CACHE_DIR = (
    FilePath(__file__).resolve().parents[1] / "cache" / "visibility"
)

class RealTerrainGrid:
    def __init__(self, elevation_array, source, targets, k_up=1.0, k_down=2.0, road_nodes=None, road_edges=None):
        self.m, self.n = elevation_array.shape
        self.source = source
        self.targets = list(targets)

        self.k_up = k_up
        self.k_down = k_down
        self.h_0 = 0.5

        self.road_nodes = road_nodes or set()
        self.road_edges = road_edges or set()

        # 1. Normalize Heights (0.0 to 10.0)
        min_h = np.min(elevation_array)
        max_h = np.max(elevation_array)
        if max_h == min_h:
            self.heights = np.zeros_like(elevation_array)
        else:
            self.heights = (elevation_array - min_h) / (max_h - min_h) * 10.0

        # 2. Create Directed Graph
        self.G = nx.grid_2d_graph(self.m, self.n, create_using=nx.DiGraph)

        targets_set = set(self.targets)

        # 3. Store Attributes
        for r in range(self.m):
            for c in range(self.n):
                node = (r, c)
                self.G.nodes[node]['height'] = float(self.heights[r, c])
                self.G.nodes[node]['pos'] = (r, c)
                self.G.nodes[node]['is_road'] = node in self.road_nodes

                if node == self.source:
                    self.G.nodes[node]['type'] = 'source'
                elif node in targets_set:
                    self.G.nodes[node]['type'] = 'target_unreached'
                else:
                    self.G.nodes[node]['type'] = 'intermediate'

        # Initial distance calculation
        self.calculate_distances()

    def add_obstacle(self, center, rx, ry):
        """
        Labels nodes within an ellipse as 'obstacle'.
        Equation: (x - cx)^2 / rx^2 + (y - cy)^2 / ry^2 <= 1
        """
        cx, cy = center
        for node in self.G.nodes():
            r, c = node
            # Check if point (r, c) is inside the ellipse
            # Note: r is typically y-axis and c is x-axis in grid indexing
            if ((c - cx)**2 / rx**2) + ((r - cy)**2 / ry**2) <= 1:
                # Keep source and target safe from being obstacles
                if node not in [self.source, self.targets]:
                    self.G.nodes[node]['type'] = 'obstacle'
        
        # We must recalculate distances to account for the new obstacles
        # You'll need to pass your coefficients again or store them as self
        self.calculate_distances()

    def calculate_distances(self):
        """
        Calculates edge weights. If either node is an obstacle, weight = 10,000.
        """
        for u, v in self.G.edges():
            # Check for obstacles
            if self.G.nodes[u]['type'] == 'obstacle' or self.G.nodes[v]['type'] == 'obstacle':
                self.G.edges[u, v]['distance'] = 10000.0
                continue

            h_u = self.G.nodes[u]['height']
            h_v = self.G.nodes[v]['height']
            diff = h_v - h_u
            base_dist = 1.0 
            
            if diff > 0: # Uphill
                # weight = base_dist + self.k_up * (diff**2)
                weight = base_dist + self.k_down * (diff + self.h_0) * diff
            elif diff < 0: # Downhill
                weight = base_dist + self.k_down * (diff + self.h_0) * diff
            else: # Flat
                weight = base_dist
                
            # Halve cost if both endpoints are road nodes
            if self.G.nodes[u]['is_road'] and self.G.nodes[v]['is_road']:
                weight *= 0.5

            self.G.edges[u, v]['distance'] = float(weight)
            self.G.edges[u, v]['observed_edge'] = False
            self.G.edges[u, v]['is_road'] = (u, v) in self.road_edges or (v, u) in self.road_edges


    def _visibility_cache_path(self, max_radius, angular_res, cache_dir):
        """Return a content-addressed cache path for this loaded terrain."""
        heights = np.ascontiguousarray(self.heights)
        digest = hashlib.sha256()
        digest.update(f"visibility-v{VISIBILITY_CACHE_SCHEMA}\0".encode())
        digest.update(str(heights.shape).encode())
        digest.update(heights.dtype.str.encode())
        digest.update(heights.tobytes())
        digest.update(f"\0{int(max_radius)}\0{int(angular_res)}".encode())
        return FilePath(cache_dir) / f"{digest.hexdigest()}.pkl"

    def _load_visibility_cache(self, path):
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            nodes = tuple(self.G.nodes())
            if (payload.get("schema") != VISIBILITY_CACHE_SCHEMA or
                    tuple(payload.get("nodes", ())) != nodes):
                return False
            visible_edges = payload.get("visible_edges")
            if (not isinstance(visible_edges, list) or
                    len(visible_edges) != len(nodes)):
                return False
            nx.set_node_attributes(
                self.G, dict(zip(nodes, visible_edges)), name="visible_edges")
            return True
        except (OSError, EOFError, pickle.PickleError, AttributeError,
                TypeError, ValueError):
            return False

    def _write_visibility_cache(self, path):
        nodes = tuple(self.G.nodes())
        payload = {
            "schema": VISIBILITY_CACHE_SCHEMA,
            "nodes": nodes,
            "visible_edges": [
                self.G.nodes[node]["visible_edges"] for node in nodes
            ],
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent,
                    prefix=f".{path.name}.", delete=False) as stream:
                temporary = FilePath(stream.name)
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _calculate_all_visibilities(self, max_radius, angular_res):
        all_vis = {
            node: self._get_polytope_visibility(
                node, max_radius, angular_res)
            for node in self.G.nodes()
        }
        nx.set_node_attributes(self.G, all_vis, name="visible_edges")

    def compute_all_visibilities(
            self, max_radius=30, angular_res=180,
            cache_dir=DEFAULT_VISIBILITY_CACHE_DIR):
        """Load DEM visibility from disk or calculate and persist it once.

        Visibility depends only on the normalized terrain heights and sweep
        parameters, not on episode sources, targets, or agent state. The
        content-addressed cache therefore remains valid for every scenario on
        the same loaded DEM. An advisory lock prevents concurrent training
        jobs from calculating the same missing entry independently.

        Returns ``True`` on a cache hit and ``False`` after calculation.
        """
        cache_path = self._visibility_cache_path(
            max_radius, angular_res, cache_dir)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._calculate_all_visibilities(max_radius, angular_res)
            return False

        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        calculated = False
        try:
            import fcntl

            with lock_path.open("a+b") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                if self._load_visibility_cache(cache_path):
                    return True
                self._calculate_all_visibilities(max_radius, angular_res)
                calculated = True
                try:
                    self._write_visibility_cache(cache_path)
                except OSError:
                    pass
                return False
        except (ImportError, OSError):
            # Cache failures must not prevent simulation. Atomic replacement
            # still protects readers on platforms without advisory locking.
            if self._load_visibility_cache(cache_path):
                return True
            if not calculated:
                self._calculate_all_visibilities(max_radius, angular_res)
            try:
                self._write_visibility_cache(cache_path)
            except OSError:
                pass
            return False

    def _get_polytope_visibility(self, obs_node, max_radius, angular_res):
            r0, c0 = obs_node
            h0 = self.heights[r0, c0] + 0.01
            
            # 1. Determine the "Horizon" points for the polytope
            # We sweep 360 degrees and find the first blockage at each angle
            angles = np.linspace(0, 2 * np.pi, angular_res)
            poly_vertices = [(c0, r0)] # Start with observer as center

            for angle in angles:
                found_blockage = False
                # Step outward along the ray
                for d in range(1, max_radius + 1):
                    curr_r = int(round(r0 + d * np.sin(angle)))
                    curr_c = int(round(c0 + d * np.cos(angle)))
                    
                    # Boundary check
                    if not (0 <= curr_r < self.m and 0 <= curr_c < self.n):
                        poly_vertices.append((curr_c, curr_r))
                        found_blockage = True
                        break
                    
                    # Height-Gate Check
                    if self.heights[curr_r, curr_c] > h0:
                        poly_vertices.append((curr_c, curr_r))
                        found_blockage = True
                        break
                
                # If no blockage was found within max_radius, the horizon is the max_radius point
                if not found_blockage:
                    poly_vertices.append((c0 + max_radius * np.cos(angle), 
                                        r0 + max_radius * np.sin(angle)))

            # 2. Create the Mask using the Polytope vertices
            # We use Matplotlib's Path to find all nodes inside this polygon
            path = Path(poly_vertices)
            
            # Generate a bounding box of nodes to check (for efficiency)
            r_min, r_max = max(0, r0 - max_radius), min(self.m, r0 + max_radius + 1)
            c_min, c_max = max(0, c0 - max_radius), min(self.n, c0 + max_radius + 1)
            
            visible_nodes = []
            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    if path.contains_point((c, r)):
                        visible_nodes.append((r, c))
            
            visible_nodes_set = set(visible_nodes)

            # Always add immediate neighbors to visible nodes
            # This prevents discretization artifacts from missing adjacent grid cells
            for v in self.G.successors(obs_node):
                visible_nodes_set.add(v)
            for v in self.G.predecessors(obs_node):
                visible_nodes_set.add(v)


            # 3. Include ALL edges that lie within this polytope
            vis_edges = []
            for u in visible_nodes_set:
                for v in self.G.neighbors(u):
                    if v in visible_nodes_set:
                        vis_edges.append((u, v))
                for v in self.G.predecessors(u):
                    if v in visible_nodes_set:
                        vis_edges.append((v, u))
            
            return list(set(vis_edges))
        

    def get_graph(self):
        return self.G
