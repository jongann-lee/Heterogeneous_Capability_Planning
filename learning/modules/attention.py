"""Transformer and typed task-graph attention blocks."""

import math

import torch
from torch import nn


class CrossTransformerBlock(nn.Module):
    """Self-attend query tokens, then cross-attend to context tokens.

    ``TransformerDecoderLayer`` already provides the residual connections,
    normalization, self-attention, cross-attention, feed-forward network, and
    dropout needed by this operation. No positional encoding is added because
    agents, targets, and actions are unordered sets; spatial information lives
    in their features and relation tensors.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

    def forward(self, query, context, query_mask, context_mask):
        value = self.layer(
            tgt=query,
            memory=context,
            tgt_key_padding_mask=~query_mask,
            memory_key_padding_mask=~context_mask,
        )
        return value.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class WorldBlock(nn.Module):
    """Bidirectional agent-target Transformer decoding block."""

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super().__init__()
        self.agent_reads_target = CrossTransformerBlock(
            model_dim, num_heads, dropout)
        self.target_reads_agent = CrossTransformerBlock(
            model_dim, num_heads, dropout)

    def forward(self, agents, targets, agent_mask, target_mask):
        # Both directions read the same incoming state so one direction does
        # not receive an arbitrary within-layer update advantage.
        return (
            self.agent_reads_target(agents, targets, agent_mask, target_mask),
            self.target_reads_agent(targets, agents, target_mask, agent_mask),
        )


class TypedRelationAttention(nn.Module):
    """One directed relation with its own multi-head message parameters."""

    def __init__(self, model_dim, num_heads, distance_dim=None, dropout=0.0):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.distance_bias = (
            nn.Linear(distance_dim, num_heads, bias=False)
            if distance_dim is not None else None)
        self.distance_value = (
            nn.Linear(distance_dim, model_dim, bias=False)
            if distance_dim is not None else None)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, destination, source, edge_mask, distance=None):
        """Aggregate source messages into destinations.

        ``edge_mask`` has shape ``[batch, destination, source]``. Distance
        embeddings, when present, have one additional trailing feature axis.
        An all-masked neighborhood produces an exact zero rather than NaNs.
        """
        batch, destinations, model_dim = destination.shape
        sources = source.shape[1]
        q = self.query(destination).view(
            batch, destinations, self.num_heads, self.head_dim)
        k = self.key(source).view(
            batch, sources, self.num_heads, self.head_dim)
        v = self.value(source).view(
            batch, sources, self.num_heads, self.head_dim)
        scores = torch.einsum("bdhe,bshe->bdhs", q, k) \
            / math.sqrt(self.head_dim)
        if self.distance_bias is not None:
            if distance is None:
                raise ValueError("distance embedding required for this relation")
            scores = scores + self.distance_bias(distance).permute(0, 1, 3, 2)

        mask = edge_mask[:, :, None, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = self.dropout(weights)

        aggregate = torch.einsum("bdhs,bshe->bdhe", weights, v)
        if self.distance_value is not None:
            distance_value = self.distance_value(distance).view(
                batch, destinations, sources,
                self.num_heads, self.head_dim)
            aggregate = aggregate + torch.einsum(
                "bdhs,bdshe->bdhe", weights, distance_value)
        return self.output(aggregate.reshape(batch, destinations, model_dim))


class _TypedNodeUpdate(nn.Module):
    def __init__(self, model_dim, relation_count, dropout=0.0):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear((relation_count + 1) * model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, state, messages, node_mask):
        update = self.feed_forward(torch.cat((state, *messages), dim=-1))
        value = self.norm(state + update)
        return value.masked_fill(~node_mask[..., None], 0.0)


class HeterogeneousGraphBlock(nn.Module):
    """Synchronous typed message passing over agent/target/action nodes."""

    def __init__(self, model_dim, num_heads, distance_dim, dropout=0.0):
        super().__init__()

        def distance_relation():
            return TypedRelationAttention(
                model_dim, num_heads, distance_dim, dropout)

        def semantic_relation():
            return TypedRelationAttention(model_dim, num_heads, None, dropout)

        # Every entry is directional; reverse directions never share weights.
        self.distance_relations = nn.ModuleDict({
            "target_to_agent": distance_relation(),
            "agent_to_target": distance_relation(),
            "action_to_agent": distance_relation(),
            "agent_to_action": distance_relation(),
            "target_to_action": distance_relation(),
            "action_to_target": distance_relation(),
        })
        self.semantic_relations = nn.ModuleDict({
            f"{kind}_{direction}": semantic_relation()
            for kind in ("serves", "reveals", "stages_for")
            for direction in ("target_to_action", "action_to_target")
        })
        self.agent_update = _TypedNodeUpdate(model_dim, 2, dropout)
        self.target_update = _TypedNodeUpdate(model_dim, 5, dropout)
        self.action_update = _TypedNodeUpdate(model_dim, 5, dropout)

    def forward(self, agents, targets, actions, observation,
                agent_target_distance, agent_action_distance,
                action_target_distance):
        at_mask = observation.agent_target_distance_mask
        aa_mask = observation.agent_action_distance_mask
        ct_mask = observation.action_target_distance_mask
        semantic = (
            ("serves", observation.serves_mask),
            ("reveals", observation.reveals_mask),
            ("stages_for", observation.stages_for_mask),
        )

        agent_messages = [
            self.distance_relations["target_to_agent"](
                agents, targets, at_mask, agent_target_distance),
            self.distance_relations["action_to_agent"](
                agents, actions, aa_mask, agent_action_distance),
        ]
        target_messages = [
            self.distance_relations["agent_to_target"](
                targets, agents, at_mask.transpose(1, 2),
                agent_target_distance.transpose(1, 2)),
            self.distance_relations["action_to_target"](
                targets, actions, ct_mask.transpose(1, 2),
                action_target_distance.transpose(1, 2)),
        ]
        action_messages = [
            self.distance_relations["agent_to_action"](
                actions, agents, aa_mask.transpose(1, 2),
                agent_action_distance.transpose(1, 2)),
            self.distance_relations["target_to_action"](
                actions, targets, ct_mask, action_target_distance),
        ]
        for name, relation_mask in semantic:
            action_messages.append(
                self.semantic_relations[f"{name}_target_to_action"](
                    actions, targets, relation_mask))
            target_messages.append(
                self.semantic_relations[f"{name}_action_to_target"](
                    targets, actions, relation_mask.transpose(1, 2)))

        # All messages above read the same incoming node states.
        return (
            self.agent_update(
                agents, agent_messages, observation.agent_mask),
            self.target_update(
                targets, target_messages, observation.target_mask),
            self.action_update(
                actions, action_messages, observation.action_mask),
        )
