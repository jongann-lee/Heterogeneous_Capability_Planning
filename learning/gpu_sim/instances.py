"""Prepared-map simulation instance factories for training and evaluation."""

import random
from functools import lru_cache
from pathlib import Path

import networkx as nx

from Real_Life_Maps.prepared_map import (
    DEFAULT_PREPARED_MAP_PATH,
    load_prepared_map,
    prepared_map_identity,
    resolve_prepared_map_path,
)
from simulation.agent import Agent
from simulation.domain import (assign_agent_capabilities, assign_target_types,
                               init_target_types, validate_capabilities)


WV_GRID_SIZE = 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WV_PREPARED_MAP_PATH = (
    PROJECT_ROOT / "Real_Life_Maps" / "WV_tobler_viewshed_64.pkl.gz"
)


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


def _wv_terrain_template():
    """Compatibility accessor for the default WV terrain template."""
    graph, metadata = _prepared_terrain_template_cached(
        str(resolve_prepared_map_path(WV_PREPARED_MAP_PATH)))
    if metadata["coarse_size"] != WV_GRID_SIZE:
        raise ValueError(
            f"prepared WV map is {metadata['coarse_size']}x"
            f"{metadata['coarse_size']}, expected {WV_GRID_SIZE}x{WV_GRID_SIZE}"
        )
    return graph


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


def make_wv_dem_instance(seed=0, num_target_types=3, num_agents=4,
                         source_position=None, target_positions=None,
                         target_types=None, agent_capabilities=None,
                         min_targets=7, max_targets=7):
    """Compatibility wrapper for the default 64x64 WV prepared map."""
    return make_prepared_map_instance(
        seed=seed, num_target_types=num_target_types, num_agents=num_agents,
        source_position=source_position, target_positions=target_positions,
        target_types=target_types, agent_capabilities=agent_capabilities,
        min_targets=min_targets, max_targets=max_targets,
        map_path=WV_PREPARED_MAP_PATH)


def make_fixed_grid(seed=0, size=5, num_target_types=3, num_agents=4,
                    source_position=None, target_positions=None,
                    target_types=None, agent_capabilities=None):
    """Create a blockage-free partially observable training instance.

    Positions, target types, and agent capabilities are sampled from ``seed``
    when their corresponding override is ``None``.
    """
    if size < 3:
        raise ValueError("size must be at least 3")
    rng = random.Random(seed)
    env = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
    for node in env.nodes:
        env.nodes[node].update(pos=node, height=0.0, type="intermediate")
    for u, v in env.edges:
        env.edges[u, v].update(distance=1.0, observed_edge=False, num_used=1.0)
    nodes = list(env.nodes)
    if source_position is None:
        source = rng.choice(nodes)
    else:
        source = tuple(source_position)
        if source not in env:
            raise ValueError(f"source position {source!r} is outside the grid")
    env.nodes[source]["type"] = "source"
    if target_positions is None:
        available = [node for node in nodes if node != source]
        if len(available) < 4:
            raise ValueError("grid must have four non-source target positions")
        targets = rng.sample(available, 4)
    else:
        targets = [tuple(position) for position in target_positions]
        if not targets:
            raise ValueError("target_positions must not be empty")
        if len(set(targets)) != len(targets):
            raise ValueError("target positions must be unique")
        if source in targets:
            raise ValueError("source and target positions must be different")
        if any(target not in env for target in targets):
            raise ValueError("a target position is outside the grid")
    for target in targets:
        env.nodes[target]["type"] = "target_unreached"
    # Local footprints prevent all target types being revealed at time zero.
    for node in env.nodes:
        env.nodes[node]["visible_edges"] = (
            [(node, neighbor) for neighbor in env.successors(node)]
            + [(neighbor, node) for neighbor in env.predecessors(node)])
    truth = env.copy()
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
        capabilities = list(agent_capabilities)
        if len(capabilities) != num_agents:
            raise ValueError("agent_capabilities must contain num_agents entries")
    agents = [Agent(source, capabilities=values) for values in capabilities]
    return env, truth, agents
