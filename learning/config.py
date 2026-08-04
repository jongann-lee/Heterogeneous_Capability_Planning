"""Configuration objects for the centralized learned planner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    num_target_types: int
    model_dim: int = 128
    num_heads: int = 4
    num_world_blocks: int = 2
    dropout: float = 0.0
    relation_hidden_dim: int = 64

    def __post_init__(self):
        if self.num_target_types < 1:
            raise ValueError("num_target_types must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")


@dataclass(frozen=True)
class CandidateConfig:
    staging_per_target: int = 2
    staging_capacity: int = 1
    include_wait: bool = True
    include_continue: bool = True


@dataclass(frozen=True)
class ReinforceConfig:
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.01
    baseline_decay: float = 0.95
    death_penalty: float = 100.0
    incomplete_penalty: float = 1000.0
    gradient_clip_norm: float = 1.0
