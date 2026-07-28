import numpy as np
import networkx as nx

from typing import List

def block_edge_on_path(env_graph: nx.Graph, path: List[int], block_per_section: int) -> nx.Graph:
    """
    Blocks edges on the given path in the environment graph.

    Parameters:
    - env_graph: The environment graph (networkx Graph).
    - path: A list of node indices representing the path.
    - block_per_section: Number of edges to block in each section of the path.
    
    Returns:
    - A new environment graph with specified edges blocked. 
        This is done with a boolean indicator. This edge may be removed later.
    """
    # Create a copy of the graph to avoid modifying the original
    blocked_graph = env_graph.copy()

    # Define a blocked attribute
    nx.set_edge_attributes(blocked_graph, False, 'blocked')
    
    # Identify target nodes (source and target_unreached) in the path
    target_indices = []
    for i, node in enumerate(path):
        node_type = env_graph.nodes[node].get("type", None)
        if node_type in ["source", "target_unreached"]:
            target_indices.append(i)
    
    # If there are no sections (less than 2 targets), return the graph as-is
    if len(target_indices) < 2:
        raise ValueError("Path must contain at least two target nodes to form sections.")
    
    # Process each section between consecutive target nodes
    for i in range(len(target_indices) - 1):
        start_idx = target_indices[i]
        end_idx = target_indices[i + 1]
        
        # Extract edges in this section
        section_edges = []
        for j in range(start_idx, end_idx):
            edge = (int(path[j]), int(path[j + 1]))
            section_edges.append(edge)
        
        # Randomly select edges to block
        if len(section_edges) > 0:
            num_to_block = min(block_per_section, len(section_edges))
            indices_to_block = np.random.choice(len(section_edges), num_to_block, replace=False)
            edges_to_block = [section_edges[int(idx)] for idx in indices_to_block]
            
            # Remove the selected edges from the graph
            for edge in edges_to_block:
                if blocked_graph.has_edge(edge[0], edge[1]):
                    blocked_graph.edges[edge[0], edge[1]]['blocked'] = True
    
    return blocked_graph    


def block_edges_maintain_connectivity(env_graph: nx.Graph, block_ratio: float) -> nx.Graph:
    """
    Blocks a specified number of edges randomly while ensuring the graph remains connected.
    
    Parameters:
    - env_graph: The environment graph (networkx Graph).
    - block_ratio: ratio of edges to block (set between 0 and 1)
    
    Returns:
    - A new environment graph with specified edges blocked (marked with 'blocked'=True).
      The graph connectivity is guaranteed to be maintained.
    """

    if block_ratio < 0 or block_ratio > 1:
        raise ValueError("block_ratio has to be between 0 and 1")

    # Create a copy of the graph to avoid modifying the original
    blocked_graph = env_graph.copy()

    num_edges_to_block = int(block_ratio * env_graph.number_of_edges())
    
    # Initialize blocked attribute if not present
    if not all('blocked' in blocked_graph.edges[e] for e in blocked_graph.edges()):
        nx.set_edge_attributes(blocked_graph, False, 'blocked')
    
    # Get all edges that are not already blocked
    available_edges = [e for e in blocked_graph.edges() if not blocked_graph.edges[e].get('blocked', False)]
    
    if len(available_edges) == 0:
        return blocked_graph
    
    # Shuffle edges for random selection
    np.random.shuffle(available_edges)
    
    blocked_count = 0
    attempted_edges = 0
    
    for edge in available_edges:
        if blocked_count >= num_edges_to_block:
            break
            
        attempted_edges += 1
        
        # Try blocking this edge
        blocked_graph.edges[edge[0], edge[1]]['blocked'] = True
        
        # Create a temporary graph with blocked edges removed to check connectivity
        temp_graph = blocked_graph.copy()
        edges_to_remove = [(u, v) for u, v, data in temp_graph.edges(data=True) if data.get('blocked', False)]
        temp_graph.remove_edges_from(edges_to_remove)
        
        # Check if graph is still connected
        if nx.is_connected(temp_graph):
            # Keep this edge blocked
            blocked_count += 1
        else:
            # Unblock this edge - it's critical for connectivity
            blocked_graph.edges[edge[0], edge[1]]['blocked'] = False
    
    return blocked_graph

def block_corridor_to_target(env_graph: nx.Graph, num_remove: int, 
                             sorted_pick: bool = False, seed: int = None) -> nx.Graph:
    """
    Blocks specific edges connecting corridor ends to the target node.
    
    Instead of removing the edges from the graph object, this sets a 'blocked'=True 
    attribute, consistent with the style of block_edges_maintain_connectivity.

    Args:
        env_graph (nx.Graph): The multiple corridor graph.
        num_remove (int): The number of corridor-to-target connections to block.
        sorted_pick (bool, optional): If True, blocks edges starting from the "top" 
                                      corridor (shortest path if sorted). 
                                      If False, blocks edges randomly. 
                                      Defaults to False.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        nx.Graph: The graph with specific edges marked as blocked.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create a copy to avoid modifying the original
    blocked_graph = env_graph.copy()

    # 1. Identify Target Node
    # In the create_multiple_corridor_graph function, target is always fixed at index 1
    target_node = 1
    
    if target_node not in blocked_graph:
        raise ValueError("Target node (1) not found in graph.")

    # 2. Identify Candidate Edges
    # Find all edges connected to the target. 
    # In this specific topology, only corridor ends connect to the target.
    # We store them as tuples (corridor_end_node, target_node)
    candidate_edges = []
    for neighbor in blocked_graph.neighbors(target_node):
        candidate_edges.append((neighbor, target_node))

    total_corridors = len(candidate_edges)

    if num_remove > total_corridors:
        raise ValueError(f"num_remove ({num_remove}) cannot exceed total corridors ({total_corridors})")

    # 3. Sort Candidates
    # In the creation function, corridors are created top-to-bottom.
    # Node indices are assigned sequentially. Therefore, lower neighbor indices
    # correspond to the 'top' corridors.
    candidate_edges.sort(key=lambda x: x[0])

    # 4. Select Edges to Block
    edges_to_block = []
    
    if sorted_pick:
        # Pick the top 'num_remove' edges
        edges_to_block = candidate_edges[:num_remove]
    else:
        # Random sample
        indices = np.random.choice(len(candidate_edges), size=num_remove, replace=False)
        edges_to_block = [candidate_edges[i] for i in indices]

    # 5. Apply Blocking
    # Initialize 'blocked' attribute to False for all edges if not present
    if not all('blocked' in blocked_graph.edges[e] for e in blocked_graph.edges()):
        nx.set_edge_attributes(blocked_graph, False, 'blocked')

    for u, v in edges_to_block:
        # Set the attribute
        blocked_graph[u][v]['blocked'] = True

    return blocked_graph

def block_specific_edges(env_graph: nx.Graph, edges_to_block: list) -> nx.Graph:
    """
    Blocks a specific list of edges in the environment graph.
    """
    
    # 1. Create a copy
    blocked_graph = env_graph.copy()
    
    # 2. Ensure 'blocked' attribute exists
    if not all('blocked' in blocked_graph.edges[e] for e in blocked_graph.edges()):
        nx.set_edge_attributes(blocked_graph, False, 'blocked')

    # 3. Apply the blocks
    for u, v in edges_to_block:
        if blocked_graph.has_edge(u, v):
            blocked_graph.edges[u, v]['blocked'] = True
        else:
            # Check for reversed edge if not found directly (just in case, though nx handles this)
            if blocked_graph.has_edge(v, u):
                blocked_graph.edges[v, u]['blocked'] = True
            else:
                pass # Edge doesn't exist (maybe wall removed it)

    # 4. Safety Check: Verify Connectivity

    # CORRECTED: Only accept (u, v) for standard Graphs
    def is_traversable(u, v):
        return not blocked_graph.edges[u, v].get('blocked', False)

    traversable_view = nx.subgraph_view(blocked_graph, filter_edge=is_traversable)

    # Global Connectivity — use is_connected for undirected, has_path for directed
    is_directed = isinstance(blocked_graph, nx.DiGraph)
    if not is_directed:
        if not nx.is_connected(traversable_view):
            num_components = nx.number_connected_components(traversable_view)
            print(f"WARNING: The blocked edges disconnected the graph into {num_components} components.")

    # Source-Target Connectivity (assuming standard keys)
    # If your source/target are stored in node attributes, you can look them up dynamically:
    source = next((n for n, d in blocked_graph.nodes(data=True) if d.get('type') == 'source'), (0,0))
    target = next((n for n, d in blocked_graph.nodes(data=True) if d.get('type') == 'target_unreached'), None)

    if source in traversable_view and target is not None and target in traversable_view:
        if not nx.has_path(traversable_view, source, target):
            print(f"CRITICAL WARNING: No path exists between {source} and {target} after blocking edges!")

    return blocked_graph