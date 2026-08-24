"""Batched shortest paths for the fixed sparse world graph using PyTorch."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouteResult:
    distances: torch.Tensor
    predecessors: torch.Tensor
    goals_reached: torch.Tensor


class GridRouter:
    """Device-resident adjacency and batched masked Dijkstra routing.

    Every row is an independent routing query. ``blocked_nodes`` may therefore
    differ across episodes. Sources and goals are always made traversable; the
    caller is responsible for blocking other live targets.
    """

    def __init__(self, neighbors: torch.Tensor, costs: torch.Tensor):
        if neighbors.ndim != 2 or costs.shape != neighbors.shape:
            raise ValueError("neighbors and costs must have shape [nodes, degree]")
        if neighbors.dtype != torch.long:
            raise ValueError("neighbors must use torch.long indices")
        self.neighbors = neighbors
        self.costs = costs
        self.num_nodes, self.max_degree = neighbors.shape

    @classmethod
    def from_edges(cls, num_nodes, edge_sources, edge_destinations, edge_costs,
                   device=None):
        """Create padded outgoing adjacency tensors from edge arrays."""
        source = torch.as_tensor(edge_sources, dtype=torch.long)
        destination = torch.as_tensor(edge_destinations, dtype=torch.long)
        weight = torch.as_tensor(edge_costs, dtype=torch.float32)
        if not (source.shape == destination.shape == weight.shape):
            raise ValueError("edge arrays must have matching shapes")
        degree = torch.bincount(source, minlength=num_nodes)
        max_degree = int(degree.max().item()) if degree.numel() else 0
        neighbors = torch.full((num_nodes, max_degree), -1, dtype=torch.long)
        costs = torch.full((num_nodes, max_degree), torch.inf)
        offsets = torch.zeros(num_nodes, dtype=torch.long)
        for src, dst, cost in zip(source.tolist(), destination.tolist(), weight.tolist()):
            slot = int(offsets[src])
            neighbors[src, slot] = dst
            costs[src, slot] = cost
            offsets[src] += 1
        return cls(neighbors.to(device), costs.to(device))

    def shortest_paths(self, sources, goals, blocked_nodes=None) -> RouteResult:
        """Run exact Dijkstra queries entirely on the adjacency tensor's device."""
        device = self.neighbors.device
        sources = torch.as_tensor(sources, dtype=torch.long, device=device).flatten()
        goals = torch.as_tensor(goals, dtype=torch.long, device=device).flatten()
        if sources.shape != goals.shape:
            raise ValueError("sources and goals must have the same shape")
        batch = sources.numel()
        if blocked_nodes is None:
            blocked = torch.zeros((batch, self.num_nodes), dtype=torch.bool,
                                  device=device)
        else:
            blocked = torch.as_tensor(blocked_nodes, dtype=torch.bool,
                                      device=device).clone()
            if blocked.shape != (batch, self.num_nodes):
                raise ValueError("blocked_nodes must have shape [batch, nodes]")
        rows = torch.arange(batch, device=device)
        blocked[rows, sources] = False
        blocked[rows, goals] = False

        distances = torch.full((batch, self.num_nodes), torch.inf, device=device)
        predecessors = torch.full((batch, self.num_nodes), -1, dtype=torch.long,
                                  device=device)
        visited = blocked.clone()
        distances[rows, sources] = 0.0

        # Each iteration settles one node per query. Queries whose goal has
        # already settled become inactive while the remaining rows continue.
        active = torch.ones(batch, dtype=torch.bool, device=device)
        for _ in range(self.num_nodes):
            unsettled = distances.masked_fill(visited, torch.inf)
            current_distance, current = unsettled.min(dim=1)
            active = active & torch.isfinite(current_distance)
            current = torch.where(active, current, goals)
            visited[rows, current] |= active

            reached_now = active & (current == goals)
            active = active & ~reached_now
            adjacent = self.neighbors[current]
            edge_cost = self.costs[current]
            valid_edge = active[:, None] & (adjacent >= 0)
            safe_adjacent = adjacent.clamp_min(0)
            valid_edge &= ~visited.gather(1, safe_adjacent)
            proposed = current_distance[:, None] + edge_cost
            old = distances.gather(1, safe_adjacent)
            improve = valid_edge & (proposed < old)
            if improve.any():
                # Grid adjacency contains at most one edge from the current
                # node to a destination, so scatter assignment is unambiguous.
                update_rows, slots = improve.nonzero(as_tuple=True)
                update_nodes = safe_adjacent[update_rows, slots]
                distances[update_rows, update_nodes] = proposed[update_rows, slots]
                predecessors[update_rows, update_nodes] = current[update_rows]
            if not active.any():
                break

        goal_distances = distances[rows, goals]
        return RouteResult(goal_distances, predecessors,
                           torch.isfinite(goal_distances))

    def reconstruct_paths(self, result: RouteResult, sources, goals):
        """Return padded node paths on-device and their lengths."""
        device = self.neighbors.device
        sources = torch.as_tensor(sources, dtype=torch.long, device=device).flatten()
        goals = torch.as_tensor(goals, dtype=torch.long, device=device).flatten()
        batch = sources.numel()
        rows = torch.arange(batch, device=device)
        reverse = torch.full((batch, self.num_nodes), -1, dtype=torch.long,
                             device=device)
        cursor = goals.clone()
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        active = result.goals_reached.clone()
        for step in range(self.num_nodes):
            reverse[rows[active], step] = cursor[active]
            lengths += active.long()
            done = cursor == sources
            active &= ~done
            if not active.any():
                break
            cursor = torch.where(active,
                                 result.predecessors[rows, cursor], cursor)
            active &= cursor >= 0
        forward = torch.full_like(reverse, -1)
        for row in range(batch):
            length = lengths[row]
            if length > 0:
                forward[row, :length] = reverse[row, :length].flip(0)
        return forward, lengths
