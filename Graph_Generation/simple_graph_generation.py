import random
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay, Voronoi

def generate_simple_graph(n_nodes, n_edges):
    """
    Generates a random, connected, undirected graph using NetworkX.

    Nodes are assigned a 2D position, and edge weights ('distance')
    represent the Euclidean distance between them.

    Args:
        n_nodes (int): The number of nodes in the graph.
        n_edges (int): The total number of edges in the graph.

    Returns:
        nx.Graph: The generated NetworkX graph.
        Returns None if parameters are invalid.
    """
    # --- Parameter Validation ---
    if n_nodes <= 0:
        print("Error: Number of nodes must be positive.")
        return None
    
    min_edges = n_nodes - 1
    if n_edges < min_edges:
        print(f"Error: A connected graph with {n_nodes} nodes needs at least {min_edges} edges.")
        return None

    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        print(f"Error: Cannot have more than {max_edges} edges in a simple graph with {n_nodes} nodes.")
        return None

    # 1. Create a graph and assign 2D positions to nodes
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    nx.set_node_attributes(G, pos, 'pos')

    # Helper function to calculate distance
    def get_distance(n1, n2):
        p1 = pos[n1]
        p2 = pos[n2]
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # 2. Ensure connectivity by creating a random path
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    for i in range(n_nodes - 1):
        u, v = nodes[i], nodes[i+1]
        dist = get_distance(u, v)
        G.add_edge(u, v, distance=dist)

    # 3. Add remaining edges randomly
    while G.number_of_edges() < n_edges:
        u, v = random.sample(range(n_nodes), 2)
        # Ensure it's not a self-loop or an existing edge
        if u != v and not G.has_edge(u, v):
            dist = get_distance(u, v)
            G.add_edge(u, v, distance=dist)
    
    return G

def create_sparse_connected_grid(m, n, node_removal_fraction=0.0, edge_removal_fraction=0.0, target_ratio=0.1, source_node=None, seed=None):
    """
    Generates a 2D grid graph, removes a portion of nodes and edges, and ensures
    the graph remains connected.

    The process involves shuffling a list of edges and attempting to remove
    a specified fraction of them. An edge is only removed if it is not a
    "bridge" (i.e., its removal does not disconnect the graph).

    Args:
        m (int): The number of rows in the grid.
        n (int): The number of columns in the grid.
        node_removal_fraction (float): The fraction of total nodes to attempt to remove.
                                  Must be between 0.0 and 1.0. A higher value
                                  will result in a sparser graph.
        edge_removal_fraction (float): The fraction of total edges to attempt to remove.
                                  Must be between 0.0 and 1.0. A higher value
                                  will result in a sparser graph.
        seed (int, optional): A seed for the random number generator to ensure
                              reproducibility.

    Returns:
        networkx.Graph: The sparse, yet fully connected, graph.
    """
    if not 0.0 <= node_removal_fraction <= 1.0:
        raise ValueError("node_removal_fraction must be between 0 and 1.")

    if not 0.0 <= edge_removal_fraction <= 1.0:
        raise ValueError("edge_removal_fraction must be between 0 and 1.")

    if seed is not None:
        random.seed(seed)

    # 1. Create the complete 2D grid graph
    G = nx.grid_2d_graph(m, n)
    

    # 1.1 Remove a fraction of the nodes randomly
    nodes_to_consider = list(G.nodes())
    random.shuffle(nodes_to_consider)
    nodes_to_remove_count = int(len(nodes_to_consider) * node_removal_fraction)
    nodes_removed_count = 0
    for node in nodes_to_consider:

        if nodes_removed_count >= nodes_to_remove_count:
            break

        # Temporarily remove the edge
        neighbors = list(G.neighbors(node))
        G.remove_node(node)

        # If the graph has become disconnected, add the node back
        if not nx.is_connected(G):
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        else:    
            nodes_removed_count += 1

    # 2. Assign node attributes
    # 2.1. Set a default type for all nodes
    node_attributes = {node: {"type": "intermediate"} for node in G.nodes()}

    # 2.2. Select and set the source node
    if source_node is None:
        # If no source is specified, pick one randomly
        source_node = random.choice(list(G.nodes()))
    node_attributes[source_node]["type"] = "source"

    # 2.3. Select and set the target nodes
    # Create a pool of potential targets (all nodes except the source)
    potential_targets = [n for n in G.nodes() if n != source_node]
    num_targets = int(len(potential_targets) * target_ratio)

    # Randomly sample from the pool
    target_nodes = random.sample(potential_targets, num_targets)
    for node in target_nodes:
        node_attributes[node]["type"] = "target_unreached"

    # 2.4. Apply the attributes to the graph
    nx.set_node_attributes(G, node_attributes)

    # 3. Determine the target number of edges to remove
    initial_edge_count = G.number_of_edges()
    edges_to_consider = list(G.edges())
    random.shuffle(edges_to_consider)
    edges_to_remove_count = int(initial_edge_count * edge_removal_fraction)
    edges_removed_count = 0

    # 4. Iterate through edges and remove them if they don't disconnect the graph
    for u, v in edges_to_consider:
        if edges_removed_count >= edges_to_remove_count:
            break

        # Temporarily remove the edge
        G.remove_edge(u, v)

        # If the graph has become disconnected, add the edge back
        if not nx.is_connected(G):
            G.add_edge(u, v)
        else:
            edges_removed_count += 1

    # 5. Add edge weights based on Euclidian distance
    distances = {}
    for u, v in G.edges():
        # The nodes are the coordinates (row, col)
        dist = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
        distances[(u, v)] = dist
    
    nx.set_edge_attributes(G, distances, name="distance")
    nx.set_edge_attributes(G, False, name="observed_edge")

    return G

def create_delaunay_graph(n_points=30, target_ratio=0.1, source_node=None, seed=None):
    """
    Generates a Delaunay triangulation graph by randomly placing points in a 
    [0, 1] x [0, 1] square, computing their Voronoi cells, moving points to 
    cell centers, and connecting points whose Voronoi cells are adjacent.

    Args:
        n_points (int): The number of points to generate in the square.
        target_ratio (float): The fraction of nodes to designate as targets.
                              Must be between 0.0 and 1.0.
        source_node (int, optional): The index of the source node. If None,
                                     a random node will be selected.
        seed (int, optional): A seed for the random number generator to ensure
                              reproducibility.

    Returns:
        networkx.Graph: The Delaunay triangulation graph with node attributes
                        (type, pos) and edge attributes (distance, observed_edge).
    """
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError("target_ratio must be between 0 and 1.")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1. Generate random points in [0, 1] x [0, 1]
    points = np.random.rand(n_points, 2)

    # 2. Create Voronoi diagram
    vor = Voronoi(points)

    # 3. Move points to centers of their Voronoi cells
    def get_voronoi_cell_center(vor, point_idx):
        """Calculate the centroid of a Voronoi cell, handling infinite regions"""
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        # Filter out -1 (infinite vertex) and empty regions
        if -1 in region or len(region) == 0:
            # For infinite regions, just use the original point
            return vor.points[point_idx]
        
        # Get vertices of the cell
        vertices = vor.vertices[region]
        
        # Calculate centroid
        centroid = vertices.mean(axis=0)
        
        # Clip to [0, 1] bounds to keep within our square
        centroid = np.clip(centroid, 0, 1)
        
        return centroid

    centered_points = np.array([get_voronoi_cell_center(vor, i) for i in range(len(points))])

    # 4. Create Delaunay triangulation with centered points
    tri = Delaunay(centered_points)

    # 5. Create NetworkX graph from Delaunay triangulation
    G = nx.Graph()

    # 5.1. Add nodes with their positions
    for i, point in enumerate(centered_points):
        G.add_node(i, pos=tuple(point))

    # 5.2. Add edges from Delaunay triangulation
    for simplex in tri.simplices:
        # Each simplex is a triangle, so we connect all three vertices
        G.add_edge(simplex[0], simplex[1])
        G.add_edge(simplex[1], simplex[2])
        G.add_edge(simplex[2], simplex[0])

    # 6. Assign node attributes
    # 6.1. Set a default type for all nodes
    node_attributes = {node: {"type": "intermediate"} for node in G.nodes()}

    # 6.2. Select and set the source node
    if source_node is None:
        # If no source is specified, pick one randomly
        source_node = random.choice(list(G.nodes()))
    node_attributes[source_node]["type"] = "source"

    # 6.3. Select and set the target nodes
    # Create a pool of potential targets (all nodes except the source)
    potential_targets = [n for n in G.nodes() if n != source_node]
    num_targets = int(len(potential_targets) * target_ratio)

    # Randomly sample from the pool
    target_nodes = random.sample(potential_targets, num_targets)
    for node in target_nodes:
        node_attributes[node]["type"] = "target_unreached"

    # 6.4. Apply the attributes to the graph
    nx.set_node_attributes(G, node_attributes)

    # 7. Add edge weights based on Euclidean distance
    distances = {}
    for u, v in G.edges():
        # Get the positions of the nodes
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        # Calculate Euclidean distance
        dist = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
        distances[(u, v)] = dist
    
    nx.set_edge_attributes(G, distances, name="distance")
    nx.set_edge_attributes(G, False, name="observed_edge")

    return G


def create_multiple_corridor_graph(n_corridors, sort_corridors=False, seed=None):
    """
    Generates a graph consisting of multiple parallel corridors connecting a 
    source node to a single target node. Includes a specific side-node attached
    to the source.

    Args:
        n_corridors (int): The number of parallel corridors to generate.
        sort_corridors (bool): If True, corridors are sorted by length (shortest 
                               to longest). Defaults to False.
        seed (int, optional): A seed for the random number generator to ensure
                              reproducibility.

    Returns:
        networkx.Graph: The corridor graph with node attributes (type, pos) 
                        and edge attributes (distance, observed_edge).
    """
    if n_corridors < 1:
        raise ValueError("n_corridors must be at least 1.")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1. Generate corridor lengths
    # Sample from uniform distribution [0.5, 1.5]
    corridor_lengths = np.random.uniform(0.5, 1.5, n_corridors)
    
    if sort_corridors:
        corridor_lengths.sort()

    # 2. Initialize Graph
    G = nx.Graph()

    # 3. Create Special Nodes
    # We assign fixed indices for clarity: 0 for Source, 1 for Target, 2 for "See All"
    source_node = 0
    target_node = 1
    see_all_node = 2

    # 4. Position Logic (for visualization)
    # Source is at (0,0). "See All" is to the left. Target is to the far right.
    # Corridors are stacked vertically.
    G.add_node(source_node, pos=(0, 0))
    G.add_node(see_all_node, pos=(-1, 0))
    
    # We place the target at x=3
    G.add_node(target_node, pos=(3, 0))

    # 5. Build Corridors
    # Calculate vertical spacing
    y_positions = np.linspace(np.sqrt(n_corridors // 2), -np.sqrt(n_corridors // 2), n_corridors)

    # Keep track of next available node index
    current_node_idx = 3

    # Dictionary to store logical edge weights to override Euclidean distance later
    logical_weights = {}

    # Connection: Source -> See All (Arbitrary weight, set to 1.0 for consistency)
    G.add_edge(source_node, see_all_node)
    logical_weights[(source_node, see_all_node)] = 1.0

    for i in range(n_corridors):
        length = corridor_lengths[i]
        y = y_positions[i]
        
        # Create Start and End nodes for this corridor
        c_start = current_node_idx
        c_end = current_node_idx + 1
        current_node_idx += 2

        # 5.1 Add Nodes with Positions
        # Start node is visually at x=1 (distance 1 from source)
        G.add_node(c_start, pos=(1, y))
        # End node is visually at x=2
        G.add_node(c_end, pos=(2, y))

        # 5.2 Add Edges and record logical weights
        # Source -> Corridor Start (Weight = 1.0)
        G.add_edge(source_node, c_start)
        logical_weights[(source_node, c_start)] = 1.0

        # Corridor Start -> Corridor End (Weight = sampled length)
        G.add_edge(c_start, c_end)
        logical_weights[(c_start, c_end)] = length

        # Corridor End -> Target (Weight = 1.0)
        G.add_edge(c_end, target_node)
        logical_weights[(c_end, target_node)] = 1.0

    # 6. Assign node attributes
    # 6.1. Set a default type for all nodes
    node_attributes = {node: {"type": "intermediate"} for node in G.nodes()}

    # 6.2. Set specific types
    node_attributes[source_node]["type"] = "source"
    node_attributes[target_node]["type"] = "target_unreached"
    
    # Note: The "see_all_node" remains "intermediate" as per instructions
    # ("Later, I will designate this..."), but it is topologically distinct.

    # 6.3. Apply the attributes to the graph
    nx.set_node_attributes(G, node_attributes)

    # 7. Add edge weights
    # Unlike the Delaunay graph which uses pure Euclidean distance, 
    # this environment requires specific logical weights defined in the prompt.
    
    # Ensure undirected consistency in the weights dictionary
    final_weights = {}
    for (u, v), w in logical_weights.items():
        final_weights[(u, v)] = w
        final_weights[(v, u)] = w

    nx.set_edge_attributes(G, final_weights, name="distance")
    nx.set_edge_attributes(G, False, name="observed_edge")

    return G

def create_occupancy_grid(m, n, nodes_to_remove):
    """
    Generates a 2D grid graph with specific nodes removed.
    Source is always (0,0) and Target is always (m-1, n-1).

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        nodes_to_remove (list of tuples): List of (r, c) coordinates to remove.

    Returns:
        networkx.Graph: The grid with nodes removed and attributes set.
    """
    # 1. Create the complete 2D grid graph
    G = nx.grid_2d_graph(m, n)
    
    # 2. Remove the specific list of nodes
    # We use set intersection to avoid errors if you list a node that doesn't exist
    valid_nodes_to_remove = [node for node in nodes_to_remove if node in G]
    G.remove_nodes_from(valid_nodes_to_remove)

    # 3. Define Source and Target
    source_node = (0, 0)
    target_node = (m - 1, n - 1)

    # --- Safety Checks ---
    if source_node not in G:
        raise ValueError(f"Source node {source_node} was removed from the graph!")
    if target_node not in G:
        raise ValueError(f"Target node {target_node} was removed from the graph!")
    
    # Optional: Ensure the graph is actually solvable (connected start to end)
    if not nx.has_path(G, source_node, target_node):
        print("WARNING: The obstacles provided block all paths from Source to Target.")

    # 4. Assign node attributes
    node_attributes = {node: {"type": "intermediate"} for node in G.nodes()}
    node_attributes[source_node]["type"] = "source"
    node_attributes[target_node]["type"] = "target_unreached" # Changed from "target_unreached" for simplicity
    nx.set_node_attributes(G, node_attributes)

    # 5. Add edge weights (Euclidean distance)
    distances = {}
    for u, v in G.edges():
        dist = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
        distances[(u, v)] = dist
    
    nx.set_edge_attributes(G, distances, name="distance")

    # 6. Assign 'pos' attribute (The 2D indices)
    # The nodes in grid_2d_graph are already (row, col) tuples.
    # We map node -> node so G.nodes[(r,c)]['pos'] = (r,c)
    pos_attributes = {node: node for node in G.nodes()}
    nx.set_node_attributes(G, pos_attributes, name="pos")
    nx.set_edge_attributes(G, False, name="observed_edge")
    
    return G