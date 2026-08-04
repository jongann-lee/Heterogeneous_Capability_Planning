"""Entity feature encoders."""

from torch import nn


class EntityEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim))

    def forward(self, features):
        return self.network(features)
