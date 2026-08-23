"""Typed loading and validation for ``learning/config.yaml``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class ModelConfig:
    num_target_types: int
    model_dim: int
    num_heads: int
    num_world_blocks: int
    dropout: float
    relation_hidden_dim: int

    def __post_init__(self):
        if self.num_target_types < 1:
            raise ValueError("num_target_types must be positive")
        if self.model_dim <= 0 or self.num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_world_blocks < 1:
            raise ValueError("num_world_blocks must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.relation_hidden_dim <= 0:
            raise ValueError("relation_hidden_dim must be positive")


@dataclass(frozen=True)
class CandidateConfig:
    staging_per_target: int
    staging_capacity: int
    include_wait: bool

    def __post_init__(self):
        if self.staging_per_target < 0:
            raise ValueError("staging_per_target must be non-negative")
        if self.staging_capacity < 1:
            raise ValueError("staging_capacity must be positive")


@dataclass(frozen=True)
class ReinforceConfig:
    learning_rate: float
    entropy_coefficient: float
    baseline_decay: float
    death_penalty: float
    incomplete_penalty: float
    gradient_clip_norm: float

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if not 0.0 <= self.baseline_decay < 1.0:
            raise ValueError("baseline_decay must lie in [0, 1)")
        if self.death_penalty < 0 or self.incomplete_penalty < 0:
            raise ValueError("penalties must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class InstanceConfig:
    min_targets: int = 7
    max_targets: int = 7

    def __post_init__(self):
        if self.min_targets < 1:
            raise ValueError("instances.min_targets must be positive")
        if self.max_targets < self.min_targets:
            raise ValueError(
                "instances.max_targets must be at least instances.min_targets")


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int
    simulation_batch_size: int
    reinforce_batch_size: int
    num_agents: int
    seed: int
    device: str
    checkpoint: str
    wandb: bool

    def __post_init__(self):
        if (self.episodes < 1 or self.simulation_batch_size < 1 or
                self.reinforce_batch_size < 1 or self.num_agents < 1):
            raise ValueError(
                "episodes, simulation_batch_size, reinforce_batch_size, and "
                "num_agents must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if not isinstance(self.wandb, bool):
            raise ValueError("wandb must be a boolean")


@dataclass(frozen=True)
class LearningConfig:
    model: ModelConfig
    candidates: CandidateConfig
    reinforce: ReinforceConfig
    training: TrainingConfig
    instances: InstanceConfig = InstanceConfig()


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return value


def load_config(path: str | Path | None = None) -> LearningConfig:
    """Load the learning configuration from YAML and validate every section."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("learning configuration must be a YAML mapping")
    training = dict(_section(payload, "training"))
    # Read checkpoints/configs made before rollout and optimizer batching were
    # separated. The old value controlled both behaviors.
    legacy_batch_size = training.pop("batch_size", None)
    if legacy_batch_size is not None:
        training.setdefault("simulation_batch_size", legacy_batch_size)
        training.setdefault("reinforce_batch_size", legacy_batch_size)
    if isinstance(training.get("device"), str):
        training["device"] = training["device"].lower()
    instance_payload = payload.get("instances", {})
    if not isinstance(instance_payload, dict):
        raise ValueError("configuration section 'instances' must be a mapping")
    candidate_payload = dict(_section(payload, "candidates"))
    # Checkpoints created before joint in-transit replanning had a CONTINUE
    # candidate. It no longer exists, but its saved config remains loadable.
    candidate_payload.pop("include_continue", None)
    return LearningConfig(
        model=ModelConfig(**_section(payload, "model")),
        candidates=CandidateConfig(**candidate_payload),
        reinforce=ReinforceConfig(**_section(payload, "reinforce")),
        training=TrainingConfig(**training),
        instances=InstanceConfig(**instance_payload),
    )
