import networkx as nx
import numpy as np
from collections import deque

class HeightMapGrid:
    def __init__(self, m, n):
        """
        Initializes a grid graph where every node has height 0.
        Source is (0,0), Target is (m-1, n-1).
        Distance weights are NOT calculated yet.
        """
        self.m = m
        self.n = n
        self.source = (0, 0)
        self.target = (m - 1, n - 1)
        
        # 1. Create the complete 2D grid
        self.G = nx.grid_2d_graph(m, n)
        
        # 2. Initialize "height" to 0.0
        nx.set_node_attributes(self.G, 0.0, name="height")

        # 3. Assign Types (source, target, intermediate)
        node_types = {node: "intermediate" for node in self.G.nodes()}
        node_types[self.source] = "source"
        node_types[self.target] = "target_unreached"
        nx.set_node_attributes(self.G, node_types, name="type")

        # 4. Assign 'pos' attribute
        pos_attributes = {node: node for node in self.G.nodes()}
        nx.set_node_attributes(self.G, pos_attributes, name="pos")

    def set_heights(self, nodes, height_value):
        """
        Assigns a specific height value to a list of nodes.
        
        Args:
            nodes (list of tuples): List of (r, c) coordinates.
            height_value (float): The height to assign to these nodes.
        """
        # Filter out invalid nodes just in case
        valid_nodes = [node for node in nodes if node in self.G]
        
        # Create a dict for the update
        height_update = {node: float(height_value) for node in valid_nodes}
        nx.set_node_attributes(self.G, height_update, name="height")

    def add_mountain(self, blob_nodes):
            """
            Takes a list of nodes (a 'blob') and assigns heights.
            Nodes in the blob directly connected to non-blob nodes get height 1.
            Nodes connected to those get height 2, and so on (BFS layering).
            """
            # Convert to set for O(1) lookup
            blob_set = set(node for node in blob_nodes if node in self.G)
            
            if not blob_set:
                return

            # 1. Find the "Boundary" (Height 1)
            # These are nodes in the blob that have at least one neighbor NOT in the blob
            queue = deque()
            visited_in_blob = set()
            new_heights = {}

            for node in blob_set:
                is_boundary = False
                for neighbor in self.G.neighbors(node):
                    if neighbor not in blob_set:
                        is_boundary = True
                        break
                
                if is_boundary:
                    # This node touches the "outside", so it starts at height 1
                    queue.append((node, 1))
                    visited_in_blob.add(node)
                    new_heights[node] = 1.0

            # 2. BFS to fill the center of the blob (Height 2, 3, ...)
            while queue:
                current_node, current_height = queue.popleft()
                
                for neighbor in self.G.neighbors(current_node):
                    # We only care about neighbors that are part of the mountain (blob)
                    # and haven't been assigned a height yet.
                    if neighbor in blob_set and neighbor not in visited_in_blob:
                        next_height = current_height + 1
                        visited_in_blob.add(neighbor)
                        new_heights[neighbor] = float(next_height)
                        queue.append((neighbor, next_height))
            
            # 3. Update the graph attributes
            # Nodes inside the blob that were "unreachable" (e.g. if the blob covers the WHOLE grid)
            # will simply retain their default height (0) or you can handle that edge case here.
            nx.set_node_attributes(self.G, new_heights, name="height")

    def add_plataeu(self, blob_nodes):
        """
        Instead of adding a mountain where the inner nodes get higher and higher
        in this all of the blob nodes share the same height: 1
        
        """
        if not blob_nodes:
            return
        
        new_heights = {}
        
        for node in blob_nodes:
            new_heights[node] = 1.0

        
        nx.set_node_attributes(self.G, new_heights, name="height")


    def calculate_distances(self):
        """
        Calculates edge weights based on:
        Base Distance (1.0) + 2 * |height_diff|
        """
        distances = {}
        
        for u, v in self.G.edges():
            # Get heights
            h_u = self.G.nodes[u]['height']
            h_v = self.G.nodes[v]['height']
            
            # Base grid distance is always 1.0 in a 4-connected grid
            # If you later allow diagonals, you might need euclidean distance here
            base_dist = 1.0 
            
            # Calculate cost
            weight = base_dist + 2.0 * abs(h_u - h_v)
            distances[(u, v)] = weight
            
        # Update the graph edges
        nx.set_edge_attributes(self.G, distances, name="distance")

        # Initialize the 'observed_edge' attirubtion to False
        nx.set_edge_attributes(self.G, False, name="observed_edge")


    def calculate_visibility(self):
            """
            Calculates recursive visibility with cascading flat extensions.
            
            Logic:
            1. Default: You see 1 hop (immediate edges).
            2. Drop: If you look down a drop, you gain a 'Flat Extension' (Radius 2) on that new level.
            3. Recursion: If during that extension you drop again, the extension refreshes (Reset to 2).
            """
            # Pre-compute heights for O(1) lookups
            heights = {n: self.G.nodes[n]['height'] for n in self.G.nodes()}
            
            visibility = {}
            all_nodes = list(self.G.nodes())

            for source in all_nodes:
                visible_edges = set()
                
                # State: {node: max_budget_remaining}
                # We track the best budget we've arrived at a node with.
                # If we arrive again with less fuel, we ignore it.
                # Initialize with -1 so even 0 is "better" (though we only push if >0)
                max_budget_seen = {n: -1 for n in all_nodes}
                
                # Queue: [(node, current_budget)]
                # Start with budget 0: We see immediate edges, but flat neighbors get budget -1 (stop)
                queue = [(source, 0)] 
                max_budget_seen[source] = 0

                idx = 0
                while idx < len(queue):
                    curr, budget = queue[idx]
                    idx += 1
                    
                    curr_h = heights[curr]

                    # 1. VISIBILITY: If we process 'curr', we see all its attached edges
                    for neighbor in self.G.neighbors(curr):
                        # Add edge to set (sorted tuple for uniqueness)
                        if curr < neighbor:
                            edge = (curr, neighbor)
                        else:
                            edge = (neighbor, curr)
                        visible_edges.add(edge)

                        # 2. TRANSITION LOGIC
                        neigh_h = heights[neighbor]
                        
                        if neigh_h < curr_h:
                            # DROP: Recursion Trigger. Reset budget to 2 (The "Flat Extension")
                            new_budget = 2
                        elif neigh_h == curr_h:
                            # FLAT: Consume budget
                            new_budget = budget - 1
                        else:
                            # RISE: Vision blocked
                            new_budget = -1

                        # 3. PROPAGATION CHECK
                        # We only continue traversing if we have fuel (>0)
                        # AND if this is the best fuel state we've seen for this neighbor
                        if new_budget > 0:
                            if new_budget > max_budget_seen[neighbor]:
                                max_budget_seen[neighbor] = new_budget
                                queue.append((neighbor, new_budget))

                visibility[source] = list(visible_edges)

            nx.set_node_attributes(self.G, visibility, name="visible_edges")
    
    def calculate_simple_visibility(self, blobs):
            """
            Simplified visibility for testing.
            
            Logic:
            1. Nodes NOT in a blob see only their immediate edges (1-hop).
            2. Nodes IN a blob see the 'Induced Subgraph' of the 'Covering Blob'.
            - The 'Covering Blob' (Halo) is the set of all nodes in the blob 
                PLUS all their 8-connected neighbors (including diagonals).
            - Visibility = All graph edges where BOTH endpoints are in this Halo.
            
            Args:
                blobs (list of list of tuples): A list of blobs, where each blob 
                                                is a list of (r, c) coordinates.
            """
            # 1. Base Case: Initialize basic 1-hop visibility for everyone
            visibility = {}
            for node in self.G.nodes():
                visible_edges = set()
                for neighbor in self.G.neighbors(node):
                    edge = tuple(sorted((node, neighbor)))
                    visible_edges.add(edge)
                visibility[node] = visible_edges

            # 2. Process each blob
            for blob_nodes in blobs:
                # A. Identify the 'Halo' (The Covering Set of Nodes)
                # The Halo includes the blob nodes + all 8-connected neighbors
                halo_nodes = set()
                
                for r, c in blob_nodes:
                    # Check all 8 neighbors + self (dr, dc in -1, 0, 1)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            
                            # We only add it if it actually exists in the grid
                            if (nr, nc) in self.G:
                                halo_nodes.add((nr, nc))

                # B. Collect all edges strictly contained within the Halo
                # (The Induced Subgraph of the Halo)
                halo_edges = set()
                
                for u in halo_nodes:
                    # Check actual graph neighbors (4-connected)
                    for v in self.G.neighbors(u):
                        # Only add the edge if the neighbor is ALSO in the halo
                        if v in halo_nodes:
                            edge = tuple(sorted((u, v)))
                            halo_edges.add(edge)

                # C. Broadcast this visibility to the nodes inside the original blob
                for node in blob_nodes:
                    if node in self.G:
                        # Update (union) with the new halo edges
                        visibility[node].update(halo_edges)

            # 3. Save attributes to the graph
            final_mapping = {k: list(v) for k, v in visibility.items()}
            nx.set_node_attributes(self.G, final_mapping, name="visible_edges")
            
    def remove_edges(self, edges_to_remove):
            """
            Manually removes a specific list of edges from the graph AND
            updates all visibility lists to reflect these removals.
            """
            # 1. Standardize edges to remove (sort them for consistent matching)
            # We use a set for fast O(1) lookups during the scrubbing phase
            edges_to_remove_set = set()
            for u, v in edges_to_remove:
                # Store the sorted version so it matches how we store visible_edges
                edges_to_remove_set.add(tuple(sorted((u, v))))

            # 2. Remove from the actual Graph structure
            for u, v in edges_to_remove:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)

            # 3. Scrub these edges from every node's 'visible_edges' list
            # This ensures no node claims to 'see' an edge that no longer exists
            for node in self.G.nodes():
                if "visible_edges" in self.G.nodes[node]:
                    current_visible = self.G.nodes[node]["visible_edges"]
                    
                    # Filter out any edge that is in our removal set
                    # We check sorted tuples because (u,v) == (v,u) for undirected graphs
                    updated_visible = [
                        edge for edge in current_visible 
                        if tuple(sorted(edge)) not in edges_to_remove_set
                    ]
                    
                    self.G.nodes[node]["visible_edges"] = updated_visible

    def get_graph(self):
        """Returns the internal NetworkX graph object."""
        return self.G