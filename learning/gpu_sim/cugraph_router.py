"""cuGraph-backed routing for the fixed WV graph."""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SSSPResult:
    distances: torch.Tensor       # [sources, nodes]
    predecessors: torch.Tensor    # [sources, nodes]


class CuGraphRouter:
    """Run weighted SSSP with a persistent immutable terrain graph.

    The original graph and a bounded set of its SSSP rows may persist. Graphs
    with active targets removed and results calculated on them are temporary.
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

    def sssp(self, sources, blocked_nodes=()) -> SSSPResult:
        """Return exact rows, caching only rows on the unmodified graph."""
        blocked = self._blocked_nodes(blocked_nodes)
        distance_rows, predecessor_rows = [], []
        temporary_graphs = {}
        for source in torch.as_tensor(sources).flatten().tolist():
            effective_blocked = tuple(node for node in blocked
                                      if node != int(source))
            if effective_blocked:
                graph = temporary_graphs.get(effective_blocked)
                if graph is None:
                    graph = self.graph(effective_blocked)
                    temporary_graphs[effective_blocked] = graph
                result = self._run_sssp(
                    source, effective_blocked, graph=graph)
            else:
                result = self._base_sssp(source)
            distance_rows.append(result[0])
            predecessor_rows.append(result[1])
        return SSSPResult(torch.stack(distance_rows),
                          torch.stack(predecessor_rows))

    def sssp_batch(self, sources, blocked_node_mask) -> SSSPResult:
        """Run independent exact SSSP queries in one GPU graph traversal.

        cuGraph exposes weighted SSSP for one source. To keep a whole rollout
        routing bank in one GPU operation, this method constructs disjoint,
        vertex-offset copies of the sparse terrain graph and connects a single
        super-source to each query source. Each copy receives its own blocked
        node mask. The copies cannot reach one another, so the one SSSP result
        reshapes exactly into independent distance and predecessor rows.

        This path intentionally performs no route or graph-variant caching.
        """
        import cudf
        import cugraph
        import cupy as cp

        source_tensor = torch.as_tensor(
            sources, dtype=torch.long, device="cuda").flatten()
        query_count = int(source_tensor.numel())
        if query_count == 0:
            empty_distances = torch.empty(
                (0, self.num_nodes), dtype=torch.float32,
                device=source_tensor.device)
            empty_predecessors = torch.empty(
                (0, self.num_nodes), dtype=torch.long,
                device=source_tensor.device)
            return SSSPResult(empty_distances, empty_predecessors)
        if query_count * self.num_nodes >= torch.iinfo(torch.int32).max:
            raise ValueError("batched SSSP vertex IDs exceed int32 capacity")

        blocked_tensor = torch.as_tensor(
            blocked_node_mask, dtype=torch.bool,
            device=source_tensor.device)
        if blocked_tensor.shape != (query_count, self.num_nodes):
            raise ValueError(
                "blocked_node_mask must have shape [queries, nodes]")

        source_array = cp.from_dlpack(source_tensor).astype(
            cp.int32, copy=False)
        blocked = cp.from_dlpack(blocked_tensor).copy()
        query_rows = cp.arange(query_count, dtype=cp.int32)
        blocked[query_rows, source_array] = False
        offsets = query_rows * self.num_nodes

        edge_sources = self._series_to_cupy(self._edges.src, cp.int32)
        edge_destinations = self._series_to_cupy(self._edges.dst, cp.int32)
        edge_costs = self._series_to_cupy(self._edges.weight, cp.float32)
        allowed = (~blocked[:, edge_sources] &
                   ~blocked[:, edge_destinations])
        copied_sources = (
            edge_sources[None] + offsets[:, None]).ravel()[allowed.ravel()]
        copied_destinations = (
            edge_destinations[None] + offsets[:, None]).ravel()[allowed.ravel()]
        copied_costs = cp.broadcast_to(
            edge_costs, allowed.shape).ravel()[allowed.ravel()]

        super_source = query_count * self.num_nodes
        copied_sources = cp.concatenate((
            copied_sources,
            cp.full(query_count, super_source, dtype=cp.int32),
        ))
        copied_destinations = cp.concatenate((
            copied_destinations, source_array + offsets,
        ))
        copied_costs = cp.concatenate((
            copied_costs,
            cp.zeros(query_count, dtype=cp.float32),
        ))
        edges = cudf.DataFrame({
            "src": copied_sources,
            "dst": copied_destinations,
            "weight": copied_costs,
        })
        graph = cugraph.Graph(directed=True)
        graph.from_cudf_edgelist(
            edges, source="src", destination="dst", edge_attr="weight",
            renumber=False)
        frame = cugraph.sssp(graph, source=int(super_source))

        vertices = self._series_to_cupy(frame.vertex, cp.int32)
        frame_distances = self._series_to_cupy(
            frame.distance, cp.float32)
        frame_predecessors = self._series_to_cupy(
            frame.predecessor, cp.int32)
        dense_size = query_count * self.num_nodes
        distances = cp.full(dense_size, cp.inf, dtype=cp.float32)
        predecessors = cp.full(dense_size, -1, dtype=cp.int32)
        ordinary = vertices < dense_size
        ordinary_vertices = vertices[ordinary]
        ordinary_distances = frame_distances[ordinary]
        distances[ordinary_vertices] = cp.where(
            ordinary_distances < 1.0e30, ordinary_distances, cp.inf)
        predecessor_values = frame_predecessors[ordinary]
        valid_predecessor = ((predecessor_values >= 0) &
                             (predecessor_values != super_source))
        predecessors[ordinary_vertices] = cp.where(
            valid_predecessor,
            predecessor_values % self.num_nodes,
            -1)
        distances = distances.reshape(query_count, self.num_nodes)
        predecessors = predecessors.reshape(query_count, self.num_nodes)
        return SSSPResult(
            torch.utils.dlpack.from_dlpack(distances),
            torch.utils.dlpack.from_dlpack(predecessors).to(torch.long))

    @staticmethod
    def _series_to_cupy(series, dtype):
        """Return a zero-copy cuDF view, casting only when necessary."""
        value = series.to_cupy()
        return value.astype(dtype, copy=False)

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
