"""Masked residual attention blocks used by the learned policy."""

import torch
from torch import nn


class ResidualAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * model_dim, model_dim))
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask, query_mask):
        update, _ = self.attention(
            query, context, context,
            key_padding_mask=~context_mask, need_weights=False)
        value = self.norm1(query + self.dropout(update))
        value = self.norm2(value + self.dropout(self.ff(value)))
        return value.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class WorldBlock(nn.Module):
    """Agent/target self-attention followed by bidirectional cross-attention."""

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super().__init__()
        self.agent_self = ResidualAttention(model_dim, num_heads, dropout)
        self.target_self = ResidualAttention(model_dim, num_heads, dropout)
        self.agent_reads_target = ResidualAttention(model_dim, num_heads, dropout)
        self.target_reads_agent = ResidualAttention(model_dim, num_heads, dropout)

    def forward(self, agents, targets, agent_mask, target_mask):
        a = self.agent_self(agents, agents, agent_mask, agent_mask)
        t = self.target_self(targets, targets, target_mask, target_mask)
        return (self.agent_reads_target(a, t, target_mask, agent_mask),
                self.target_reads_agent(t, a, agent_mask, target_mask))
