"""PyTorch Transformer blocks used by learned policies."""

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
