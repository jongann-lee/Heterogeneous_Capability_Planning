import numpy as np
import random
import networkx as nx
from typing import List


# This module defines the visibility mapping

def line_of_sight_visibility(input_graph: nx.Graph, visibility_range: int = None) -> nx.Graph:
    """
    Calculates the line-of-sight visibility graph from an input grid graph.

    An edge is considered visible to if they it is in the same row or column
    as the node and there is an unbroken path of  between them in the input graph.
    The visibility can be limited by a maximum range.

    Args:
        input_graph (nx.Graph): The input graph, where nodes are assumed to be 
                                (row, col) or (x, y) tuples.
        visibility_range (int, optional): The maximum number of grid cells a node 
                                          can see. If None, visibility is unlimited 
                                          across the grid's extent. Defaults to None.

    Returns:
        nx.Graph: A new graph containing edges between all mutually visible nodes.
    """
    # Use a set for efficient O(1) average time complexity for node lookups
    nodes_set = set(input_graph.nodes())
    if not nodes_set:
        # Set an empty attribute for an empty graph and return
        nx.set_node_attributes(input_graph, {}, name='visibility')
        return input_graph

    # Find chains of edges where any node inside that chain can see all of the edges in it
    horizontal_chains = []
    vertical_chains = []

    # Determine the grid's bounding box to define scan area
    min_x = min(n[0] for n in nodes_set)
    max_x = max(n[0] for n in nodes_set)
    min_y = min(n[1] for n in nodes_set)
    max_y = max(n[1] for n in nodes_set)

    """
    Finds all continuous horizontal and vertical chains by scanning the grid
    and checking for both node and edge existence at each step.
    """
    horizontal_chains = []
    vertical_chains = []
    
    # A set of nodes for fast lookups
    nodes_set = set(input_graph.nodes())
    # A canonical set of edges for fast lookups
    edge_set = {tuple(sorted(edge)) for edge in input_graph.edges()}

    # Get the grid's bounding box to define the scan area
    if not nodes_set:
        return [], []
    min_x = min(n[0] for n in nodes_set)
    max_x = max(n[0] for n in nodes_set)
    min_y = min(n[1] for n in nodes_set)
    max_y = max(n[1] for n in nodes_set)

    # --- Horizontal Scan ---
    for r in range(min_x, max_x + 1):
        current_chain = []
        for c in range(min_y, max_y + 1):
            node = (r, c)
            
            # If the node doesn't exist, any current chain is broken.
            if node not in nodes_set:
                if len(current_chain) > 1:
                    horizontal_chains.append(current_chain)
                current_chain = []
                continue

            # The node exists.
            if not current_chain:
                # If there's no active chain, start a new one with this node.
                current_chain.append(node)
            else:
                # If a chain is active, check for an edge from the previous node.
                previous_node = current_chain[-1]
                if tuple(sorted((previous_node, node))) in edge_set:
                    # If connected, add it to the chain.
                    current_chain.append(node)
                else:
                    # If not connected, the old chain ends...
                    if len(current_chain) > 1:
                        horizontal_chains.append(current_chain)
                    # ...and a new one begins with the current node.
                    current_chain = [node]
        
        # After the row scan is complete, save any remaining valid chain.
        if len(current_chain) > 1:
            horizontal_chains.append(current_chain)
            
    # --- Vertical Scan (Same logic, just swapping axes) ---
    for c in range(min_y, max_y + 1):
        current_chain = []
        for r in range(min_x, max_x + 1):
            node = (r, c)
            if node not in nodes_set:
                if len(current_chain) > 1:
                    vertical_chains.append(current_chain)
                current_chain = []
                continue
            
            if not current_chain:
                current_chain.append(node)
            else:
                previous_node = current_chain[-1]
                if tuple(sorted((previous_node, node))) in edge_set:
                    current_chain.append(node)
                else:
                    if len(current_chain) > 1:
                        vertical_chains.append(current_chain)
                    current_chain = [node]

        if len(current_chain) > 1:
            vertical_chains.append(current_chain)


    # Create a lookup map from each node to the chain(s) it belongs to
    node_to_chain = {node: {} for node in nodes_set}
    for chain in horizontal_chains:
        chain_tuple = tuple(chain) # Use tuple for immutability
        for node in chain:
            node_to_chain[node]['horizontal'] = chain_tuple
    for chain in vertical_chains:
        chain_tuple = tuple(chain)
        for node in chain:
            node_to_chain[node]['vertical'] = chain_tuple

    # For each node, define its visibility as the union of its chains
    visibility_mapping = {}
    for node in input_graph.nodes():
        visible_edges = set()
        
        # Helper to convert a list of nodes into a set of canonical edge tuples
        def nodes_to_edges(node_list: list) -> set:
            edges = set()
            for i in range(len(node_list) - 1):
                # Sort nodes within the tuple to create a canonical edge representation
                # e.g., ((1,0), (0,0)) becomes ((0,0), (1,0))
                edge = tuple(sorted((node_list[i], node_list[i+1])))
                edges.add(edge)
            return edges
        
        # --- Horizontal Visibility ---
        if 'horizontal' in node_to_chain.get(node, {}):
            h_chain = node_to_chain[node]['horizontal']
            if visibility_range is None:
                visible_edges.update(nodes_to_edges(h_chain))
            else:
                node_index = h_chain.index(node)
                start = max(0, node_index - visibility_range)
                end = min(len(h_chain), node_index + visibility_range + 1)
                # Convert the slice of visible nodes into edges
                visible_edges.update(nodes_to_edges(h_chain[start:end]))

        # --- Vertical Visibility ---
        if 'vertical' in node_to_chain.get(node, {}):
            v_chain = node_to_chain[node]['vertical']
            if visibility_range is None:
                visible_edges.update(nodes_to_edges(v_chain))
            else:
                node_index = v_chain.index(node)
                start = max(0, node_index - visibility_range)
                end = min(len(v_chain), node_index + visibility_range + 1)
                # Convert the slice of visible nodes into edges
                visible_edges.update(nodes_to_edges(v_chain[start:end]))
            
        visibility_mapping[node] = list(visible_edges)
    
    nx.set_node_attributes(input_graph, visibility_mapping, name="visible_edges")
    return input_graph


def two_hop_visibility(input_graph: nx.Graph, num_extended_nodes: int = None, 
                       extended_fraction: float = 0.1, seed: int = None, node_list: list = None) -> nx.Graph:
    """
    Creates a visibility mapping where most nodes see only their connected edges,
    but a select number of random nodes can see edges up to 2 hops away.

    Args:
        input_graph (nx.Graph): The input graph to create visibility mapping for.
        num_extended_nodes (int, optional): The exact number of nodes to give extended 
                                           visibility. If None, uses extended_fraction.
        extended_fraction (float, optional): The fraction of nodes (0.0 to 1.0) that 
                                            should have extended 2-hop visibility. 
                                            Defaults to 0.1 (10%).
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        node_list (list, optional): Explicit list of nodes to give extended visibility.
                                   If provided, overrides num_extended_nodes, 
                                   extended_fraction, and seed parameters.

    Returns:
        nx.Graph: The input graph with a 'visible_edges' node attribute containing
                 the list of visible edges for each node.
    """
    if seed is not None:
        random.seed(seed)
    
    intermediate_nodes_list = [node for node in input_graph.nodes() 
                  if input_graph.nodes[node].get("type", "intermediate")]
    
    # Handle empty graph
    if not intermediate_nodes_list:
        nx.set_node_attributes(input_graph, {}, name='visible_edges')
        return input_graph
    
    # Determine which nodes should have extended visibility
    if node_list is not None:
        # Use provided node list
        # Validate that all nodes in node_list exist in the graph
        invalid_nodes = [n for n in node_list if n not in input_graph.nodes()]
        if invalid_nodes:
            raise ValueError(f"The following nodes in node_list are not in the graph: {invalid_nodes}")
        extended_nodes = set(node_list)
    else:
        # Randomly select nodes for extended visibility
        if num_extended_nodes is None:
            num_extended_nodes = max(1, int(len(intermediate_nodes_list) * extended_fraction))
        else:
            num_extended_nodes = min(num_extended_nodes, len(intermediate_nodes_list))
        
        extended_nodes = set(random.sample(intermediate_nodes_list, num_extended_nodes))
    
    # Randomly select nodes for extended visibility if list not given
    if node_list == None:
        extended_nodes = set(random.sample(intermediate_nodes_list, num_extended_nodes))
    else:
        extended_nodes = set(node_list)
    
    visibility_mapping = {}
    
    for node in list(input_graph.nodes()):
        visible_edges = set()
        
        if node in extended_nodes:
            # Extended visibility: see edges up to 2 hops away
            # Get all neighbors (1-hop)
            neighbors_1hop = set(input_graph.neighbors(node))
            
            # Add all edges connected to this node (1-hop edges)
            for neighbor in neighbors_1hop:
                edge = tuple(sorted((int(node), int(neighbor))))
                visible_edges.add(edge)
            
            # Add all edges connected to neighbors (2-hop edges)
            for neighbor in neighbors_1hop:
                neighbors_2hop = set(input_graph.neighbors(neighbor))
                for neighbor_2hop in neighbors_2hop:
                    edge = tuple(sorted((int(neighbor), int(neighbor_2hop))))
                    visible_edges.add(edge)
        else:
            # Standard visibility: see only directly connected edges
            neighbors = input_graph.neighbors(node)
            for neighbor in neighbors:
                edge = tuple(sorted((int(node), int(neighbor))))
                visible_edges.add(edge)
        
        visibility_mapping[node] = list(visible_edges)
    
    # Set the visibility mapping as a node attribute
    nx.set_node_attributes(input_graph, visibility_mapping, name='visible_edges')
    
    # Optionally mark which nodes have extended visibility
    extended_visibility_flag = {node: (node in extended_nodes) for node in list(input_graph.nodes())}
    nx.set_node_attributes(input_graph, extended_visibility_flag, name='has_extended_visibility')
    
    return input_graph

def n_corridor_visibility(input_graph: nx.Graph, see_all_node_idx: int = 2) -> nx.Graph:
    """
    Creates a specific visibility mapping intended ONLY for the multiple_corridor_graph.
    
    Most nodes have standard 1-hop visibility (they see only the edges they are 
    directly connected to). The designated "see all" node (default index 2) 
    has global visibility and can observe every edge in the graph.

    Args:
        input_graph (nx.Graph): The n-corridor graph to create visibility mapping for.
        see_all_node_idx (int, optional): The index of the node designated to see 
                                          everything. Defaults to 2, which matches 
                                          the creation logic of create_multiple_corridor_graph.

    Returns:
        nx.Graph: The input graph with a 'visible_edges' node attribute containing
                  the list of visible edges for each node.
    """
    
    # 1. Pre-calculate the set of ALL edges for the "see all" node
    all_edges_set = set()
    for u, v in input_graph.edges():
        # Sort node indices for consistency in edge representation
        edge = tuple(sorted((int(u), int(v))))
        all_edges_set.add(edge)

    visibility_mapping = {}

    for node in list(input_graph.nodes()):
        visible_edges = set()

        # 2. Check if this is the special "see all" node
        if node == see_all_node_idx:
            # This node sees the entire graph structure
            visible_edges = all_edges_set.copy()
        
        # 3. Standard visibility for all other nodes
        else:
            neighbors = input_graph.neighbors(node)
            for neighbor in neighbors:
                # Add only the edge connecting the node to its neighbor
                edge = tuple(sorted((int(node), int(neighbor))))
                visible_edges.add(edge)
        
        visibility_mapping[node] = list(visible_edges)

    # 4. Set the visibility mapping as a node attribute
    nx.set_node_attributes(input_graph, visibility_mapping, name="visible_edges")

    return input_graph

def hill_visibility(input_graph: nx.Graph, hill_nodes: list) -> nx.Graph:
    """
    Applies "Hill Node" properties to the graph:
    1. Hill Nodes have global visibility (can see all edges).
    2. Edges connecting a Hill Node to a Non-Hill Node have 3x distance (uphill/downhill penalty).
    
    Args:
        input_graph (nx.Graph): The occupancy grid graph.
        hill_nodes (list): A list of node identifiers (e.g., (r, c) tuples) 
                           designated as high ground.

    Returns:
        nx.Graph: The modified graph with updated edge weights and 
                  a 'visible_edges' node attribute.
    """
    
    # Convert list to set for O(1) lookups
    hill_set = set(hill_nodes)

    # --- 1. Modify Edge Weights ---
    # We iterate over edges to find connections between Hill and Non-Hill
    for u, v, data in input_graph.edges(data=True):
        u_is_hill = u in hill_set
        v_is_hill = v in hill_set

        # XOR check: True if exactly one of them is a hill node
        if u_is_hill ^ v_is_hill:
            if 'distance' in data:
                data['distance'] *= 3.0

    # --- 2. Assign Visibility Mapping ---
    
    # Pre-calculate the list of ALL edges in a canonical sorted format
    # Sorting ensures (u, v) is treated identical to (v, u)
    all_edges_list = []
    for u, v in input_graph.edges():
        edge_rep = tuple(sorted((u, v)))
        all_edges_list.append(edge_rep)

    visibility_mapping = {}

    for node in input_graph.nodes():
        if node in hill_set:
            # Hill nodes see everything
            visibility_mapping[node] = all_edges_list[:] # Copy the full list
        else:
            # Standard nodes only see their immediate connections
            local_edges = []
            for neighbor in input_graph.neighbors(node):
                edge_rep = tuple(sorted((node, neighbor)))
                local_edges.append(edge_rep)
            visibility_mapping[node] = local_edges

    # Set the attribute
    nx.set_node_attributes(input_graph, visibility_mapping, name="visible_edges")

    return input_graph