import networkx as nx
import numpy as np
from typing import List

def nearest_neighbor(input_graph: nx.Graph) -> List:
    """
    A simple way to generate the Hamiltonian path
    using the Nearest Neighbor heuristic
    
    Expects:
    - Edge attribute 'distance' for weights
    - Node attribute 'type' with value 'source' to identify the starting node
    - Node attribute 'type' with value 'target' for destination nodes
    """
    
    if len(input_graph) == 0:
        return []
    
    # Find the source node (where node attribute 'type' == 'source')
    source_node = None
    for node, data in input_graph.nodes(data=True):
        if data.get('type') == 'source':  # Changed this line
            source_node = node
            break
    
    # Fallback to first node if no source found
    if source_node is None:
        source_node = list(input_graph.nodes())[0]
    
    # Initialize path and unvisited set
    path = [source_node]
    unvisited = set(input_graph.nodes()) - {source_node}
    current = source_node
    
    # Nearest neighbor loop
    while unvisited:
        # Get all neighbors that are unvisited
        neighbors = set(input_graph.neighbors(current)) & unvisited
        
        if not neighbors:
            # No more reachable unvisited nodes
            break
        
        # Find nearest neighbor using min with key
        nearest = min(
            neighbors,
            key=lambda node: input_graph[current][node].get('distance', float('inf'))
        )
        
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return path

import itertools

def brute_force_hamiltonian_path(input_graph: nx.Graph) -> List:
    """
    Find the optimal Hamiltonian path using brute force.
    Only practical for small graphs (<12 nodes).
    """
    if len(input_graph) == 0:
        return []
    
    # Find the source node
    source_node = None
    for node, data in input_graph.nodes(data=True):
        if data.get('type') == 'source':
            source_node = node
            break
    
    if source_node is None:
        source_node = list(input_graph.nodes())[0]
    
    # Get all other nodes
    other_nodes = [n for n in input_graph.nodes() if n != source_node]
    
    best_path = None
    best_distance = float('inf')
    
    # Try all permutations starting from source
    for perm in itertools.permutations(other_nodes):
        path = [source_node] + list(perm)
        
        # Check if this path is valid (all edges exist) and calculate distance
        total_distance = 0
        valid = True
        
        for i in range(len(path) - 1):
            if input_graph.has_edge(path[i], path[i+1]):
                total_distance += input_graph[path[i]][path[i+1]].get('distance', float('inf'))
            else:
                valid = False
                break
        
        if valid and total_distance < best_distance:
            best_distance = total_distance
            best_path = path
    
    return best_path if best_path else [source_node]