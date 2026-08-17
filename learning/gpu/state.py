"""Batched CUDA episode state and exact event transitions."""

from dataclasses import dataclass

import torch


@dataclass
class TensorEpisodeState:
    world: object
    positions: torch.Tensor       # [B, A] node ids
    capabilities: torch.Tensor    # [B, A, K+1]
    target_types: torch.Tensor    # [B, T], values 1..K
    alive: torch.Tensor           # [B, A]
    target_live: torch.Tensor     # [B, T]
    target_known: torch.Tensor    # [B, T]
    moving: torch.Tensor          # [B, A]
    transit_from: torch.Tensor    # [B, A]
    transit_to: torch.Tensor      # [B, A]
    arrival_time: torch.Tensor    # [B, A]
    clock: torch.Tensor           # [B]
    traversal_cost: torch.Tensor  # [B, A]
    deaths: torch.Tensor          # [B]
    goal_nodes: torch.Tensor      # [B, A]
    route_next: torch.Tensor      # [B, A, N], committed successors
    route_active: torch.Tensor    # [B, A]
    needs_replan: torch.Tensor    # [B, A]
    stalled: torch.Tensor         # [B]

    @classmethod
    def create(cls, world, source_nodes, capabilities, target_types):
        device = world.positions.device
        capabilities = torch.as_tensor(capabilities, dtype=torch.bool,
                                       device=device)
        target_types = torch.as_tensor(target_types, dtype=torch.long,
                                       device=device)
        batch, agents, _ = capabilities.shape
        sources = torch.as_tensor(source_nodes, dtype=torch.long,
                                  device=device).flatten()
        if sources.numel() == 1:
            sources = sources.expand(batch)
        positions = sources[:, None].expand(batch, agents).clone()
        shape = (batch, agents)
        state = cls(
            world, positions, capabilities, target_types,
            torch.ones(shape, dtype=torch.bool, device=device),
            torch.ones((batch, len(world.targets)), dtype=torch.bool,
                       device=device),
            torch.zeros((batch, len(world.targets)), dtype=torch.bool,
                        device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
            positions.clone(), positions.clone(),
            torch.full(shape, torch.inf, device=device),
            torch.zeros(batch, device=device),
            torch.zeros(shape, device=device),
            torch.zeros(batch, dtype=torch.long, device=device),
            positions.clone(),
            torch.full((batch, agents, len(world.positions)), -1,
                       dtype=torch.long, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
            torch.ones(shape, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device))
        state.observe(torch.ones_like(state.alive))
        return state

    @property
    def batch_size(self):
        return self.positions.shape[0]

    def observe(self, observer_mask):
        """Reveal targets and return only previously unknown revelations."""
        scouts = self.capabilities[:, :, 0] & self.alive & observer_mask
        visible = self.world.visible_targets[self.positions]
        revealed = (visible & scouts[:, :, None]).any(dim=1)
        newly_revealed = revealed & self.target_live & ~self.target_known
        self.target_known |= revealed & self.target_live
        return newly_revealed

    def dispatch_next_hops(self, next_nodes, agent_mask):
        """Start one edge traversal for selected at-node agents."""
        next_nodes = torch.as_tensor(next_nodes, dtype=torch.long,
                                     device=self.positions.device)
        selected = agent_mask & self.alive & ~self.moving
        current = self.positions
        adjacent = self.world.neighbors[current]
        match = adjacent == next_nodes[:, :, None]
        valid = match.any(dim=2) & selected & (next_nodes != current)
        slots = match.long().argmax(dim=2)
        costs = self.world.edge_cost[current, slots]
        self.transit_from = torch.where(valid, current, self.transit_from)
        self.transit_to = torch.where(valid, next_nodes, self.transit_to)
        self.arrival_time = torch.where(
            valid, self.clock[:, None] + costs, self.arrival_time)
        self.moving |= valid
        return valid

    def advance(self):
        """Process one heap-equivalent arrival per active episode.

        Equal-time arrivals use the lowest agent index, matching the CPU
        ``(arrival_time, agent_index)`` heap ordering.
        """
        pending = self.arrival_time.masked_fill(~self.moving, torch.inf)
        next_time, arriving_agent = pending.min(dim=1)
        active = torch.isfinite(next_time)
        rows = torch.arange(self.batch_size, device=self.positions.device)
        chosen = torch.zeros_like(self.moving)
        chosen[rows[active], arriving_agent[active]] = True
        old = self.positions.clone()
        self.clock = torch.where(active, next_time, self.clock)
        self.positions = torch.where(chosen, self.transit_to, self.positions)
        self.traversal_cost += torch.where(
            chosen, self.arrival_time - self.clock[:, None] + 0.0,
            torch.zeros_like(self.traversal_cost))
        # Use immutable edge tensors for exact traversal cost; the expression
        # above is zero after clock update and is replaced for chosen entries.
        adjacent = self.world.neighbors[old]
        slots = (adjacent == self.positions[:, :, None]).long().argmax(dim=2)
        edge_cost = self.world.edge_cost[old, slots]
        self.traversal_cost += torch.where(
            chosen, edge_cost, torch.zeros_like(edge_cost))
        self.moving &= ~chosen
        self.arrival_time = torch.where(
            chosen, torch.full_like(self.arrival_time, torch.inf),
            self.arrival_time)

        newly_revealed = self.observe(chosen)
        at_target = self.positions[:, :, None] == self.world.target_nodes
        encounter = at_target & self.target_live[:, None, :] & chosen[:, :, None]
        target_hit = encounter.any(dim=1)
        capability = self.capabilities.gather(
            2, self.target_types[:, None, :].expand(
                -1, self.capabilities.shape[1], -1))
        wins = encounter & capability
        loses = encounter & ~capability
        self.target_known |= target_hit
        self.target_live &= ~wins.any(dim=1)
        died = loses.any(dim=2)
        self.alive &= ~died
        self.deaths += died.sum(dim=1)
        information_changed = newly_revealed.any(dim=1) | target_hit.any(dim=1)
        return active, arriving_agent, chosen, information_changed

    def completed(self):
        return ~self.target_live.any(dim=1)
