"""Validated, factorized fixed-suite definitions for learned-policy evaluation."""

from dataclasses import dataclass
import json
from pathlib import Path

from simulation.domain import validate_capabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = Path(__file__).resolve().parent / "evaluation_suites"
SUITE_ALIASES = {
    "development": SUITE_ROOT / "wv_rps_fixed_v1.json",
    "test": SUITE_ROOT / "wv_factorial_test_v1.json",
}


@dataclass(frozen=True)
class AgentConfiguration:
    id: str
    agent_count: int
    profile: str
    capabilities: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class TargetDefinition:
    position: tuple[int, int]
    target_type: int


@dataclass(frozen=True)
class TargetConfiguration:
    id: str
    target_count: int
    profile: str
    targets: tuple[TargetDefinition, ...]


@dataclass(frozen=True)
class EvaluationSuite:
    schema_version: int
    suite_id: str
    terrain_id: str
    num_target_types: int
    source_position: tuple[int, int]
    agent_configurations: tuple[AgentConfiguration, ...]
    target_configurations: tuple[TargetConfiguration, ...]


@dataclass(frozen=True)
class EvaluationCase:
    suite_id: str
    scenario_id: str
    source_position: tuple[int, int]
    num_target_types: int
    agent: AgentConfiguration
    target: TargetConfiguration

    def instance_metadata(self):
        return {
            "source_position": list(self.source_position),
            "agent_capabilities": [
                sorted(capabilities)
                for capabilities in self.agent.capabilities
            ],
            "target_positions": [
                list(target.position) for target in self.target.targets
            ],
            "target_types": [
                target.target_type for target in self.target.targets
            ],
        }


def resolve_suite_path(suite="development"):
    """Resolve a built-in alias or an explicit suite JSON path."""
    suite_value = "development" if suite is None else str(suite)
    if suite_value in SUITE_ALIASES:
        return SUITE_ALIASES[suite_value].resolve()
    candidate = Path(suite_value).expanduser()
    if not candidate.is_absolute():
        local = candidate.resolve()
        candidate = local if local.is_file() else PROJECT_ROOT / candidate
    return candidate.resolve()


def _identifier(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _count(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _position(value, context):
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in value)):
        raise ValueError(f"{context} must be a two-integer list")
    position = tuple(value)
    if any(coordinate < 0 or coordinate >= 64 for coordinate in position):
        raise ValueError(f"{context} must lie inside the 64x64 WV terrain")
    return position


def _parse_agent_configuration(payload, num_target_types):
    context = "agent configuration"
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    identifier = _identifier(payload.get("id"), f"{context} id")
    count = _count(payload.get("agent_count"), f"{identifier}.agent_count")
    profile = _identifier(payload.get("profile"), f"{identifier}.profile")
    raw_capabilities = payload.get("capabilities")
    if not isinstance(raw_capabilities, list) or len(raw_capabilities) != count:
        raise ValueError(
            f"{identifier}.capabilities must contain {count} entries")
    capabilities = tuple(
        validate_capabilities(values, num_target_types)
        for values in raw_capabilities
    )
    if not any(0 in values for values in capabilities):
        raise ValueError(f"{identifier} must contain at least one scout")
    supported = set().union(*capabilities)
    required = set(range(1, num_target_types + 1))
    if not required <= supported:
        missing = sorted(required - supported)
        raise ValueError(
            f"{identifier} does not cover target capabilities {missing}")
    return AgentConfiguration(identifier, count, profile, capabilities)


def _parse_target_configuration(payload, num_target_types, source_position):
    context = "target configuration"
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    identifier = _identifier(payload.get("id"), f"{context} id")
    count = _count(payload.get("target_count"), f"{identifier}.target_count")
    profile = _identifier(payload.get("profile"), f"{identifier}.profile")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != count:
        raise ValueError(f"{identifier}.targets must contain {count} entries")
    targets = []
    positions = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"{identifier}.targets[{index}] must be an object")
        position = _position(
            raw_target.get("position"),
            f"{identifier}.targets[{index}].position")
        target_type = raw_target.get("type")
        if (isinstance(target_type, bool) or not isinstance(target_type, int)
                or not 1 <= target_type <= num_target_types):
            raise ValueError(
                f"{identifier}.targets[{index}].type must lie in "
                f"1..{num_target_types}")
        if position == source_position:
            raise ValueError(f"{identifier} places a target at the source")
        if position in positions:
            raise ValueError(f"{identifier} contains duplicate target positions")
        positions.add(position)
        targets.append(TargetDefinition(position, target_type))
    return TargetConfiguration(identifier, count, profile, tuple(targets))


def load_evaluation_suite(suite="development"):
    """Load and validate one fixed evaluation suite."""
    path = resolve_suite_path(suite)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation suite not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation suite JSON in {path}: {error}") \
            from error
    if not isinstance(payload, dict):
        raise ValueError("evaluation suite root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("evaluation suite schema_version must be 1")
    suite_id = _identifier(payload.get("suite_id"), "suite_id")
    terrain_id = _identifier(payload.get("terrain_id"), "terrain_id")
    if terrain_id != "wv_dem_64_v1":
        raise ValueError(
            f"unsupported evaluation terrain_id {terrain_id!r}")
    num_target_types = _count(
        payload.get("num_target_types"), "num_target_types")
    source_position = _position(payload.get("source_position"),
                                "source_position")
    raw_agents = payload.get("agent_configurations")
    raw_targets = payload.get("target_configurations")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("agent_configurations must be a non-empty list")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("target_configurations must be a non-empty list")
    agents = tuple(
        _parse_agent_configuration(item, num_target_types)
        for item in raw_agents
    )
    targets = tuple(
        _parse_target_configuration(item, num_target_types, source_position)
        for item in raw_targets
    )
    for context, identifiers in (
        ("agent configuration", [item.id for item in agents]),
        ("target configuration", [item.id for item in targets]),
    ):
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate {context} id")
    return EvaluationSuite(
        schema_version=1, suite_id=suite_id, terrain_id=terrain_id,
        num_target_types=num_target_types, source_position=source_position,
        agent_configurations=agents, target_configurations=targets,
    ), path


def _selected_ids(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def select_evaluation_cases(suite, agent_config=None, target_config=None,
                            agent_count=None, target_count=None, limit=None):
    """Filter a suite's deterministic Cartesian product in file order."""
    agent_ids = _selected_ids(agent_config)
    target_ids = _selected_ids(target_config)
    known_agent_ids = {item.id for item in suite.agent_configurations}
    known_target_ids = {item.id for item in suite.target_configurations}
    if agent_ids is not None and not agent_ids <= known_agent_ids:
        unknown = sorted(agent_ids - known_agent_ids)
        raise ValueError(f"unknown agent configuration(s): {', '.join(unknown)}")
    if target_ids is not None and not target_ids <= known_target_ids:
        unknown = sorted(target_ids - known_target_ids)
        raise ValueError(f"unknown target configuration(s): {', '.join(unknown)}")
    if agent_count is not None:
        _count(agent_count, "agent_count")
    if target_count is not None:
        _count(target_count, "target_count")
    if limit is not None:
        _count(limit, "limit")

    agents = [
        item for item in suite.agent_configurations
        if (agent_ids is None or item.id in agent_ids)
        and (agent_count is None or item.agent_count == agent_count)
    ]
    targets = [
        item for item in suite.target_configurations
        if (target_ids is None or item.id in target_ids)
        and (target_count is None or item.target_count == target_count)
    ]
    cases = [
        EvaluationCase(
            suite_id=suite.suite_id,
            scenario_id=f"{agent.id}__{target.id}",
            source_position=suite.source_position,
            num_target_types=suite.num_target_types,
            agent=agent, target=target,
        )
        for agent in agents for target in targets
    ]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("suite filters selected no evaluation scenarios")
    return cases
