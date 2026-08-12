"""cuGraph-backed routing for the fixed WV graph."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SSSPResult:
    distances: torch.Tensor       # [sources, nodes]
    predecessors: torch.Tensor    # [sources, nodes]


class CuGraphRouter:
    """Cache GPU graph variants and run weighted SSSP from many sources.

    Graph variants are keyed by a bit mask of blocked target nodes. The source
    and optional goal are restored per query by selecting a compatible cached
    variant; callers should omit a goal target from ``blocked_target_mask``.
    """

    def __init__(self, num_nodes, edge_sources, edge_destinations, edge_costs,
                 target_nodes):
        import cudf

        self.num_nodes = int(num_nodes)
        self.target_nodes = tuple(int(x) for x in target_nodes)
        self._edges = cudf.DataFrame({
            "src": edge_sources,
            "dst": edge_destinations,
            "weight": edge_costs,
        })
        self._graphs = {}
        self._sssp_cache = {}

    def graph(self, blocked_target_mask=0):
        """Return a lazily-built directed graph for one target blockage mask."""
        import cugraph

        mask = int(blocked_target_mask)
        graph = self._graphs.get(mask)
        if graph is not None:
            return graph
        keep = None
        for index, node in enumerate(self.target_nodes):
            if mask & (1 << index):
                allowed = ((self._edges.src != node) &
                           (self._edges.dst != node))
                keep = allowed if keep is None else keep & allowed
        edges = self._edges if keep is None else self._edges[keep]
        graph = cugraph.Graph(directed=True)
        graph.from_cudf_edgelist(
            edges, source="src", destination="dst", edge_attr="weight",
            renumber=False, vertices=list(range(self.num_nodes)))
        self._graphs[mask] = graph
        return graph

    @staticmethod
    def _series_to_torch(series, dtype=None):
        # cuDF -> CuPy -> torch uses CUDA Array Interface/DLPack and therefore
        # does not stage graph results through host memory.
        value = torch.utils.dlpack.from_dlpack(series.to_cupy())
        return value.to(dtype=dtype) if dtype is not None else value

    def sssp(self, sources, blocked_target_mask=0) -> SSSPResult:
        """Return dense CUDA distance/predecessor rows for each source."""
        import cugraph

        graph = self.graph(blocked_target_mask)
        distance_rows, predecessor_rows = [], []
        for source in torch.as_tensor(sources).flatten().tolist():
            key = (int(blocked_target_mask), int(source))
            cached = self._sssp_cache.get(key)
            if cached is None:
                frame = cugraph.sssp(graph, source=int(source)).sort_values("vertex")
                vertices = self._series_to_torch(frame.vertex, torch.long)
                if vertices.numel() != self.num_nodes or not torch.equal(
                        vertices, torch.arange(self.num_nodes,
                                              device=vertices.device)):
                    raise RuntimeError("cuGraph SSSP did not return every world node")
                cached = (
                    self._series_to_torch(frame.distance, torch.float32),
                    self._series_to_torch(frame.predecessor, torch.long))
                self._sssp_cache[key] = cached
            distance_rows.append(cached[0])
            predecessor_rows.append(cached[1])
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
