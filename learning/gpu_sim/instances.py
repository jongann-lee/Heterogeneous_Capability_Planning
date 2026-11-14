"""Prepared-map simulation instance factories for training and evaluation."""

import random
from functools import lru_cache


from Real_Life_Maps.prepared_map import (
    DEFAULT_PREPARED_MAP_PATH,
    load_prepared_map,
    prepared_map_identity,
    resolve_prepared_map_path,
)
from simulation.agent import Agent
from simulation.domain import (assign_agent_capabilities, assign_target_types,
                               init_target_types, validate_capabilities)


@lru_cache(maxsize=8)
def _prepared_terrain_template_cached(resolved_path):
    """Cache target-independent terrain by canonical artifact path."""
    graph, metadata = load_prepared_map(resolved_path)
    graph = graph.copy()
    for node in graph:
        graph.nodes[node]["type"] = "intermediate"
    return graph, metadata


def _prepared_terrain_template(map_path=DEFAULT_PREPARED_MAP_PATH):
    resolved = resolve_prepared_map_path(map_path)
    return _prepared_terrain_template_cached(str(resolved))[0]


def inspect_prepared_map(map_path=DEFAULT_PREPARED_MAP_PATH):
    """Resolve and validate an artifact before any episode is constructed."""
    resolved = resolve_prepared_map_path(map_path)
    graph, metadata = _prepared_terrain_template_cached(str(resolved))
    identity = prepared_map_identity(
        resolved, graph, metadata, configured_path=map_path)
    return resolved, graph, dict(metadata), identity


def make_prepared_map_instance(seed=0, num_target_types=3, num_agents=4,
                               source_position=None, target_positions=None,
                               target_types=None, agent_capabilities=None,
                               min_targets=7, max_targets=7,
                               map_path=DEFAULT_PREPARED_MAP_PATH):
    """Create one episode overlay on a validated prepared terrain map."""
    rng = random.Random(seed)
    terrain = _prepared_terrain_template(map_path)
    nodes = list(terrain.nodes)
    source = rng.choice(nodes) if source_position is None else tuple(source_position)
    if source not in terrain:
        raise ValueError(
            f"source position {source!r} is not present in the prepared map")
    if target_positions is None:
        if min_targets < 1 or max_targets < min_targets:
            raise ValueError("target count range must satisfy 1 <= min <= max")
        available = [node for node in nodes if node != source]
        if max_targets > len(available):
            raise ValueError("max_targets exceeds the available prepared-map nodes")
        targets = rng.sample(available, rng.randint(min_targets, max_targets))
    else:
        targets = [tuple(position) for position in target_positions]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target positions must be non-empty and unique")
    if source in targets:
        raise ValueError("target positions must differ from the source")
    missing_targets = [target for target in targets if target not in terrain]
    if missing_targets:
        raise ValueError(
            "target positions are not present in the prepared map: "
            f"{missing_targets!r}")

    env, truth = terrain.copy(), terrain.copy()

    env.nodes[source]["type"] = "source"
    truth.nodes[source]["type"] = "source"
    for target in targets:
        env.nodes[target]["type"] = "target_unreached"
        truth.nodes[target]["type"] = "target_unreached"
    if target_types is None:
        types = assign_target_types(targets, num_target_types, rng)
    else:
        values = list(target_types)
        if len(values) != len(targets):
            raise ValueError("target_types must align with target_positions")
        if any(not 1 <= int(value) <= num_target_types for value in values):
            raise ValueError("target types must lie in 1..num_target_types")
        types = dict(zip(targets, map(int, values)))
    init_target_types(env, truth, types)

    if agent_capabilities is None:
        capabilities = assign_agent_capabilities(
            num_agents, num_target_types, ensure_target_coverage=True,
            ensure_scout=True, rng=rng)
    else:
        capabilities = [validate_capabilities(values, num_target_types)
                        for values in agent_capabilities]
        if len(capabilities) != num_agents:
            raise ValueError("agent_capabilities must contain num_agents entries")
    return env, truth, [Agent(source, capabilities=values)
                        for values in capabilities]
