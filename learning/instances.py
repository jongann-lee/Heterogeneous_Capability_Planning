"""Small fixed-map instance factory for smoke tests and CLI experiments."""

import random
from functools import lru_cache
from pathlib import Path

import networkx as nx

from simulation.agent import Agent
from simulation.domain import (assign_agent_capabilities, assign_target_types,
                               init_target_types, validate_capabilities)


WV_GRID_SIZE = 64
WV_DEM_PATH = Path(__file__).resolve().parents[1] / "Real_Life_Maps" / "WV_DEM.tif"
WV_ROADS_PATH = Path(__file__).resolve().parents[1] / "Real_Life_Maps" / "WV_roads.pkl"


@lru_cache(maxsize=8)
def _wv_graph_template(source, targets, num_target_types):
    """Build expensive visibility data once for each spatial configuration."""
    from simulation.real_map_benchmark import build_graphs

    built = build_graphs(
        str(WV_DEM_PATH), str(WV_ROADS_PATH), WV_GRID_SIZE, source, targets,
        num_target_types=num_target_types, rng=random.Random(0))
    if built is None:
        raise ValueError("a target is unreachable on the WV DEM")
    return built[0], built[1]


def make_wv_dem_instance(seed=0, num_target_types=3, num_agents=4,
                         source_position=None, target_positions=None,
                         target_types=None, agent_capabilities=None):
    """Create one 64x64 episode on the clockwise-rotated WV DEM."""
    rng = random.Random(seed)
    nodes = [(row, col) for row in range(WV_GRID_SIZE)
             for col in range(WV_GRID_SIZE)]
    source = rng.choice(nodes) if source_position is None else tuple(source_position)
    if source not in nodes:
        raise ValueError(f"source position {source!r} is outside the WV grid")
    if target_positions is None:
        targets = rng.sample([node for node in nodes if node != source], 7)
    else:
        targets = [tuple(position) for position in target_positions]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target positions must be non-empty and unique")
    if source in targets or any(target not in nodes for target in targets):
        raise ValueError("target positions must be on-grid and differ from source")

    template_env, template_truth = _wv_graph_template(
        source, tuple(targets), num_target_types)
    env, truth = template_env.copy(), template_truth.copy()
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
