"""Initial vanilla Transformer policy developed around August 11 (0811)."""

import math

import torch
from torch import nn

from learning.modules import (AssignmentDecoder, CrossTransformerBlock,
                              EntityEncoder, HeterogeneousGraphBlock,
                              WorldBlock)
from learning.gpu_sim.observation_cpu import feature_dimensions
from learning.policy.configuration import ModelConfig


class VanillaTransformerPolicy(nn.Module):
    """The initial 0811 centralized Transformer policy implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        fa, ft, fc, _fat, fac, fct = feature_dimensions(config.num_target_types)
        d = config.model_dim
        self.agent_encoder = EntityEncoder(fa, d)
        self.target_encoder = EntityEncoder(ft, d)
        self.action_encoder = EntityEncoder(fc, d)
        self.world_blocks = nn.ModuleList([
            WorldBlock(d, config.num_heads, config.dropout)
            for _ in range(config.num_world_blocks)])
        self.action_reads_target = CrossTransformerBlock(
            d, config.num_heads, config.dropout)
        self.action_target_relation = nn.Sequential(
            nn.Linear(fct, config.relation_hidden_dim), nn.GELU(),
            nn.Linear(config.relation_hidden_dim, d))
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.pair_relation = nn.Sequential(
            nn.Linear(fac, config.relation_hidden_dim), nn.GELU(),
            nn.Linear(config.relation_hidden_dim, 1))
        self.decoder = AssignmentDecoder()
        self.has_critic = False

    def forward(self, observation):
        a = self.agent_encoder(observation.agent_features)
        t = self.target_encoder(observation.target_features)
        c = self.action_encoder(observation.action_features)
        for block in self.world_blocks:
            a, t = block(a, t, observation.agent_mask, observation.target_mask)
        relation = self.action_target_relation(
            observation.action_target_relations)
        mask = observation.target_mask[:, None, :, None]
        relation = (relation * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        c = c + relation
        c = self.action_reads_target(
            c, t, observation.action_mask, observation.target_mask)
        logits = torch.einsum("bad,bcd->bac", self.query(a), self.key(c))
        logits = logits / math.sqrt(self.config.model_dim)
        logits = logits + self.pair_relation(
            observation.agent_action_relations).squeeze(-1)
        return logits.masked_fill(~observation.feasible_action_mask, -torch.inf)

    def decode(self, observation, candidates=None, training=None):
        logits = self(observation)
        candidate_batches = candidates or observation.candidates
        if candidate_batches is None:
            raise ValueError("candidates are required for decoding capacities")
        if observation.action_capacities is not None:
            capacities = observation.action_capacities
        else:
            unlimited = torch.iinfo(torch.long).max
            capacities = torch.zeros(logits.shape[0], logits.shape[2],
                                     dtype=torch.long, device=logits.device)
            for b, items in enumerate(candidate_batches):
                for c, item in enumerate(items):
                    capacities[b, c] = (unlimited if item.capacity is None
                                        else item.capacity)
        return self.decoder(logits, observation.feasible_action_mask,
                            capacities, self.training if training is None else training)

    def actor_critic(self, observation):
        return self(observation), None

    def decode_with_value(self, observation, candidates=None, training=None):
        return self.decode(observation, candidates, training), None


class HeterogeneousGraphPolicy(nn.Module):
    """Typed edge-conditioned task-graph actor with a shared graph critic."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.has_critic = True
        d = config.model_dim
        k = config.num_target_types
        edge_dim = config.distance_embedding_dim
        self.agent_encoder = EntityEncoder(k + 3, d)
        self.target_encoder = EntityEncoder(k + 1, d)
        self.action_encoder = EntityEncoder(4, d)
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, edge_dim), nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.graph_blocks = nn.ModuleList([
            HeterogeneousGraphBlock(
                d, config.num_heads, edge_dim, config.dropout)
            for _ in range(config.message_passing_blocks)
        ])
        self.actor = nn.Sequential(
            nn.Linear(2 * d + edge_dim, config.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(config.relation_hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(3 * d, config.critic_hidden_dim),
            nn.GELU(),
            nn.Linear(config.critic_hidden_dim, 1),
        )
        self.decoder = AssignmentDecoder()

    @staticmethod
    def _require_task_graph(observation):
        required = (
            "task_agent_features", "task_target_features",
            "task_action_features", "agent_target_distances",
            "agent_action_distances", "action_target_distances",
            "agent_target_distance_mask", "agent_action_distance_mask",
            "action_target_distance_mask", "serves_mask", "reveals_mask",
            "stages_for_mask",
        )
        missing = [name for name in required
                   if getattr(observation, name, None) is None]
        if missing:
            raise ValueError(
                "task-graph observation fields are missing: "
                + ", ".join(missing))

    def encode(self, observation):
        self._require_task_graph(observation)
        agents = self.agent_encoder(observation.task_agent_features)
        targets = self.target_encoder(observation.task_target_features)
        actions = self.action_encoder(observation.task_action_features)
        at_distance = self.distance_encoder(observation.agent_target_distances)
        aa_distance = self.distance_encoder(observation.agent_action_distances)
        ct_distance = self.distance_encoder(observation.action_target_distances)
        for block in self.graph_blocks:
            agents, targets, actions = block(
                agents, targets, actions, observation,
                at_distance, aa_distance, ct_distance)
        return agents, targets, actions, aa_distance

    def actor_critic(self, observation):
        agents, targets, actions, aa_distance = self.encode(observation)
        batch, num_agents, _ = agents.shape
        num_actions = actions.shape[1]
        agent_pairs = agents[:, :, None].expand(
            batch, num_agents, num_actions, -1)
        action_pairs = actions[:, None].expand(
            batch, num_agents, num_actions, -1)
        logits = self.actor(torch.cat(
            (agent_pairs, action_pairs, aa_distance), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(
            ~observation.feasible_action_mask, -torch.inf)

        def masked_sum(values, mask):
            return (values * mask[..., None]).sum(dim=1)

        graph_state = torch.cat((
            masked_sum(agents, observation.agent_mask),
            masked_sum(targets, observation.target_mask),
            masked_sum(actions, observation.action_mask),
        ), dim=-1)
        values = self.critic(graph_state).squeeze(-1)
        return logits, values

    def forward(self, observation):
        return self.actor_critic(observation)[0]

    def _capacities(self, observation, candidates, logits):
        if observation.action_capacities is not None:
            return observation.action_capacities
        candidate_batches = candidates or observation.candidates
        if candidate_batches is None:
            raise ValueError("candidates are required for decoding capacities")
        unlimited = torch.iinfo(torch.long).max
        capacities = torch.zeros(logits.shape[0], logits.shape[2],
                                 dtype=torch.long, device=logits.device)
        for batch, items in enumerate(candidate_batches):
            for action, item in enumerate(items):
                capacities[batch, action] = (
                    unlimited if item.capacity is None else item.capacity)
        return capacities

    def decode_with_value(self, observation, candidates=None, training=None):
        logits, values = self.actor_critic(observation)
        capacities = self._capacities(observation, candidates, logits)
        decoded = self.decoder(
            logits, observation.feasible_action_mask, capacities,
            self.training if training is None else training)
        return decoded, values

    def decode(self, observation, candidates=None, training=None):
        return self.decode_with_value(
            observation, candidates, training)[0]


def build_policy(config: ModelConfig) -> nn.Module:
    """Construct the policy architecture selected by a saved configuration."""
    if config.architecture == "task_graph":
        return HeterogeneousGraphPolicy(config)
    return VanillaTransformerPolicy(config)
