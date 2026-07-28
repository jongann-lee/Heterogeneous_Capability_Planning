import numpy as np
import networkx as nx

# an implementation of the KSPD algorithm.

# First we define the similarity measures

def canonical_edge(u, v):
    return (u, v) if u <= v else (v, u)

def sim1(input_graph, path1, path2):
    edge_path1 = [canonical_edge(u, v) for u, v in nx.utils.pairwise(path1)]
    edge_path2 = [canonical_edge(u, v) for u, v in nx.utils.pairwise(path2)]
    
    set_edge_path1 = set(edge_path1)
    set_edge_path2 = set(edge_path2)

    intersection = set_edge_path1.intersection(set_edge_path2)
    union = set_edge_path1.union(set_edge_path2)

    length_intersection = sum(input_graph[u][v]["distance"] for u, v in intersection)
    length_union = sum(input_graph[u][v]["distance"] for u, v in union)

    if length_union == 0:
        return 0  # avoid division by zero
    
    return length_intersection / length_union


# The naive implementation (algorithm 1 of the paper)

def approximate_KSPD(input_graph: nx.graph, source, target, k: int, tau: float):

    """
    This is an implementation of the appriximate KSPD from the paper
    "Finding Top-k Shortest Paths with Diversity"

    Input:
        input_graph (nx.graph): The input graph.
        source: The source node.
        target: The target node.
        k (int): The number of paths to find.
        tau (int): The diversity threshold.
    
    Output:
        A list of k paths from source to target.
    """
    top_k_diverse_paths = []

    # Generate a sorted list of the all the simple paths from source to target
    paths_generator = nx.shortest_simple_paths(input_graph, source, target, weight="distance")

    for path in paths_generator:
        # If we have found k paths we are done
        if len(top_k_diverse_paths) >= k:
            break

        # Check if the path is diverse enough
        is_diverse = True
        for existing_path_data in top_k_diverse_paths:
            if sim1(input_graph, path, existing_path_data['path']) > tau:
                is_diverse = False
                break
        
        if is_diverse:
            path_length = nx.path_weight(input_graph, path, weight="distance")
            top_k_diverse_paths.append({'path': path, 'length': path_length})
        
    return top_k_diverse_paths



