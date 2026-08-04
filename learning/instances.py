"""Small fixed-map instance factory for smoke tests and CLI experiments."""

import random

import networkx as nx

from simulation.agent import Agent
from simulation.domain import (assign_agent_capabilities, assign_target_types,
                               init_target_types)


def make_fixed_grid(seed=0, size=5, num_target_types=3, num_agents=4):
    """Create a blockage-free partially observable training instance."""
    if size < 3:
        raise ValueError("size must be at least 3")
    rng = random.Random(seed)
    env = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
    for node in env.nodes:
        env.nodes[node].update(pos=node, height=0.0, type="intermediate")
    for u, v in env.edges:
        env.edges[u, v].update(distance=1.0, observed_edge=False, num_used=1.0)
    source = (size // 2, size // 2)
    env.nodes[source]["type"] = "source"
    targets = [(0, 0), (0, size - 1), (size - 1, 0),
               (size - 1, size - 1)]
    for target in targets:
        env.nodes[target]["type"] = "target_unreached"
    # Local footprints prevent all target types being revealed at time zero.
    for node in env.nodes:
        env.nodes[node]["visible_edges"] = (
            [(node, neighbor) for neighbor in env.successors(node)]
            + [(neighbor, node) for neighbor in env.predecessors(node)])
    truth = env.copy()
    types = assign_target_types(targets, num_target_types, rng)
    init_target_types(env, truth, types)
    capabilities = assign_agent_capabilities(
        num_agents, num_target_types, ensure_target_coverage=True,
        ensure_scout=True, rng=rng)
    agents = [Agent(source, capabilities=values) for values in capabilities]
    return env, truth, agents
