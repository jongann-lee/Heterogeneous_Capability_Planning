"""Initial vanilla Transformer policy developed around August 11 (0811)."""

import math

import torch
from torch import nn

from learning.modules import (AssignmentDecoder, CrossTransformerBlock,
                              EntityEncoder, WorldBlock)
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
