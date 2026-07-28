import numpy as np
import networkx as nx

def visibility_reward(env_graph: nx.Graph, input_node) -> float:
    """
    Defines a reward function on each node of the graph based on
    the combined value of the visible, unexplored edges from that node.
    """

    visible_edges = env_graph.nodes[input_node]["visible_edges"]
    visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
    edge_value = np.array([env_graph.edges[edge]["num_used"] for edge in visible_unexplored_edges])
    visibility_reward = np.sum(edge_value)

    return visibility_reward


def target_and_visibility_reward(env_graph: nx.Graph, input_node, unreached_targets) -> float:
    """
    Defines a reward function on each node of the graph based on
    1. The number of remaining targets in the environment.
    2. The combined value of the visible, unexplored edges from that node
    """

    reward_ratio = 0.0 # Weighting factor between target count and visibility

    if input_node in unreached_targets:
        target_reward = -1 * len(unreached_targets) + 1
    else:
        target_reward = -1 * len(unreached_targets) 

    visible_edges = env_graph.nodes[input_node]["visible_edges"]
    visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
    edge_value = np.array([env_graph.edges[edge]["num_used"] for edge in visible_unexplored_edges])
    visibility_reward = np.sum(edge_value)

    total_reward = target_reward + reward_ratio * visibility_reward

    return total_reward

    
