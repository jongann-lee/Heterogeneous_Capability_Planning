"""cuGraph-backed routing for the fixed WV graph."""

from dataclasses import dataclass, field
from collections import OrderedDict

import torch


@dataclass(frozen=True)
class SSSPResult:
    distances: torch.Tensor       # [sources, nodes]
    predecessors: torch.Tensor    # [sources, nodes]


@dataclass
class RoutingBatchCache:
    """Bounded blocked-graph and exact reroute cache for one rollout batch."""

    max_cached_graphs: int = 64
    max_cached_routes: int = 512
    graphs: OrderedDict = field(default_factory=OrderedDict, init=False)
    routes: OrderedDict = field(default_factory=OrderedDict, init=False)

    @staticmethod
    def _get(items, key):
        value = items.get(key)
        if value is not None:
            items.move_to_end(key)
        return value

    @staticmethod
    def _put(items, key, value, limit):
        items[key] = value
        items.move_to_end(key)
        while len(items) > limit:
            items.popitem(last=False)

    def get_graph(self, blocked):
        return self._get(self.graphs, tuple(blocked))

    def put_graph(self, blocked, graph):
        self._put(
            self.graphs, tuple(blocked), graph, self.max_cached_graphs)

    def get_route(self, blocked, source):
        return self._get(self.routes, (tuple(blocked), int(source)))

    def put_route(self, blocked, source, result):
        self._put(
            self.routes, (tuple(blocked), int(source)), result,
            self.max_cached_routes)

    def clear(self):
        """Release every batch-owned RAPIDS graph and rerouted tensor row."""
        self.routes.clear()
        self.graphs.clear()


class CuGraphRouter:
    """Run weighted SSSP with persistent terrain and caller-owned caches.

    The original graph and a bounded set of its SSSP rows persist. Graphs with
    active target nodes removed, and routes calculated on those graphs, are
    retained only when a caller supplies a bounded rollout-batch cache.
    """

    def __init__(self, num_nodes, edge_sources, edge_destinations, edge_costs,
                 target_nodes=(), max_cached_routes=512):
        import cudf

        self.num_nodes = int(num_nodes)
        self.target_nodes = tuple(int(x) for x in target_nodes)
        self._edges = cudf.DataFrame({
            "src": edge_sources,
            "dst": edge_destinations,
            "weight": edge_costs,
        })
        self._base_graph = None
        self.max_cached_routes = int(max_cached_routes)
        self._base_sssp_cache = OrderedDict()

    def _blocked_nodes(self, blocked):
        """Normalize either node IDs or the legacy fixed-target bit mask."""
        if isinstance(blocked, int):
            return tuple(node for index, node in enumerate(self.target_nodes)
                         if blocked & (1 << index))
        return tuple(sorted({int(node) for node in blocked}))

    def _build_graph(self, blocked):
        import cugraph

        keep = None
        for node in blocked:
            allowed = ((self._edges.src != node) &
                       (self._edges.dst != node))
            keep = allowed if keep is None else keep & allowed
        edges = self._edges if keep is None else self._edges[keep]
        graph = cugraph.Graph(directed=True)
        graph.from_cudf_edgelist(
            edges, source="src", destination="dst", edge_attr="weight",
            renumber=False, vertices=list(range(self.num_nodes)))
        return graph

    def graph(self, blocked_nodes=()):
        """Return the cached base graph or a temporary blocked graph."""
        blocked = self._blocked_nodes(blocked_nodes)
        if not blocked:
            if self._base_graph is None:
                self._base_graph = self._build_graph(())
            return self._base_graph
        return self._build_graph(blocked)

    @staticmethod
    def _series_to_torch(series, dtype=None):
        # cuDF -> CuPy -> torch uses CUDA Array Interface/DLPack and therefore
        # does not stage graph results through host memory.
        value = torch.utils.dlpack.from_dlpack(series.to_cupy())
        return value.to(dtype=dtype) if dtype is not None else value

    def _run_sssp(self, source, blocked, graph=None):
        import cugraph

        graph = self.graph(blocked) if graph is None else graph
        frame = cugraph.sssp(graph, source=int(source)).sort_values("vertex")
        vertices = self._series_to_torch(frame.vertex, torch.long)
        if vertices.numel() != self.num_nodes or not torch.equal(
                vertices, torch.arange(self.num_nodes,
                                      device=vertices.device)):
            raise RuntimeError("cuGraph SSSP did not return every world node")
        return (self._series_to_torch(frame.distance, torch.float32),
                self._series_to_torch(frame.predecessor, torch.long))

    def _base_sssp(self, source):
        """Return a cached SSSP row from the unmodified terrain graph."""
        source = int(source)
        cached = self._base_sssp_cache.get(source)
        if cached is None:
            cached = self._run_sssp(source, ())
            self._base_sssp_cache[source] = cached
            while len(self._base_sssp_cache) > self.max_cached_routes:
                self._base_sssp_cache.popitem(last=False)
        else:
            self._base_sssp_cache.move_to_end(source)
        return cached

    def _base_routes_are_safe(self, predecessor, source, blocked,
                              required_nodes):
        """Whether base-tree paths to every required node avoid blockers."""
        if not blocked:
            return True
        device = predecessor.device
        required = torch.as_tensor(required_nodes, dtype=torch.long,
                                   device=device).flatten()
        blocked_tensor = torch.as_tensor(blocked, dtype=torch.long,
                                         device=device)
        required = required[~torch.isin(required, blocked_tensor)]
        cursor = required.clone()
        active = (cursor >= 0) & (cursor != int(source))
        for _ in range(self.num_nodes):
            if torch.isin(cursor[active], blocked_tensor).any():
                return False
            if not active.any():
                return True
            previous = predecessor[cursor.clamp_min(0)]
            active &= (previous >= 0) & (previous != int(source))
            cursor = torch.where(active, previous, cursor)
        return False

    def sssp(self, sources, blocked_nodes=(), required_nodes=None,
             batch_cache: RoutingBatchCache | None = None) -> SSSPResult:
        """Return exact SSSP rows, reusing safe target-independent rows.

        When every required destination's base-tree path avoids the sparse
        blocked-node set, the unblocked row is exact and no blocked cuGraph
        variant is constructed. Otherwise this falls back to exact SSSP.
        """
        blocked = self._blocked_nodes(blocked_nodes)
        distance_rows, predecessor_rows = [], []
        temporary_graphs = {}
        for source in torch.as_tensor(sources).flatten().tolist():
            effective_blocked = tuple(node for node in blocked
                                      if node != int(source))
            cached = (None if batch_cache is None else
                      batch_cache.get_route(effective_blocked, source))
            if cached is None:
                base = self._base_sssp(source)
                result = base
                if effective_blocked and (
                        required_nodes is None or not self._base_routes_are_safe(
                            base[1], source, effective_blocked, required_nodes)):
                    graph = (temporary_graphs.get(effective_blocked)
                             if batch_cache is None else
                             batch_cache.get_graph(effective_blocked))
                    if graph is None:
                        graph = self.graph(effective_blocked)
                        if batch_cache is None:
                            temporary_graphs[effective_blocked] = graph
                        else:
                            batch_cache.put_graph(effective_blocked, graph)
                    result = self._run_sssp(
                        source, effective_blocked, graph=graph)
                    if batch_cache is not None:
                        batch_cache.put_route(
                            effective_blocked, source, result)
            else:
                result = cached
            distance_rows.append(result[0])
            predecessor_rows.append(result[1])
        return SSSPResult(torch.stack(distance_rows),
                          torch.stack(predecessor_rows))

    def reconstruct_path(self, result, source_row, source, goal):
        """Reconstruct one route for diagnostics/CPU equivalence tests."""
        predecessor = result.predecessors[source_row]
        cursor = int(goal)
        reverse = [cursor]
        while cursor != int(source):
            cursor = int(predecessor[cursor].item())
            if cursor < 0 or len(reverse) > self.num_nodes:
                return None
            reverse.append(cursor)
        return list(reversed(reverse))
