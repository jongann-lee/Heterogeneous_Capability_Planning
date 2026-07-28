import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add it to sys.path if it isn't there already
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import numpy as np
import networkx as nx

from Single_Agent.reward_functions import visibility_reward
from Single_Agent.lin_kernighan_tsp import solve_tsp_lin_kernighan
from Graph_Generation.target_graph import stochastic_accumulated_blockage_path, create_fully_connected_target_graph

def calculate_path_reward(path, env_graph: nx.Graph, reward_ratio: float, discount_factor: float = 1.0) -> float:
    """
    Calculates the total reward for a given path in the enviroment graph.

    The environment graph contains nodes with a 'visible_edges' attribute,
    the edges have a 'distance', 'observed_edge', 'num_used' attribute. 

    They are used to calculate reward function, which is defined in reward_functions.py

    Note that we do not calculate the visibility reward at the final node in the path for consistency.

    Args:
        path (list): A list of nodes representing the path taken by the agent.
        env_graph (nx.Graph): The environment graph containing nodes and edges with attributes.
        reward_ratio (float): A weighting factor to balance visibility reward and distance penalty.
        discount_factor (float): A factor to discount future observaton rewards.
    Returns: 
        float: The total calculated reward for the given path.
    """

    total_visibility_reward = 0.0
    total_distance = 0.0

    for i in range(len(path) - 1):

        current_node = path[i]
        next_node = path[i+1]
        
        # Calculate visibility reward at the current node
        node_visibility_reward = visibility_reward(env_graph, current_node)
        total_visibility_reward += node_visibility_reward * (discount_factor ** i)

        # Calculate the distance to the next node
        edge_distance = env_graph.edges[current_node, next_node]["distance"]
        total_distance += edge_distance

        # Update the visibility of edges after observing from the current node
        if "visible_edges" in env_graph.nodes[current_node]:
            visible_edges = env_graph.nodes[current_node]["visible_edges"]
            visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
            
            if len(visible_unexplored_edges) > 0:
                for edge in visible_unexplored_edges:
                    env_graph.edges[edge]["observed_edge"] = True

        # Move to the next node in the path
        current_node = next_node
    
    # Calculate the final reward

    total_reward = reward_ratio * total_visibility_reward - total_distance

    return total_reward
    
        
class RepeatedTopK:
    def __init__(self, reward_ratio: float, env_graph: nx.Graph, 
                    sample_recursion: int, sample_num_obstacle: int, sample_obstacle_hop: int,
                    target_graph_num_neighbors: int = 4, target_graph_recursions: int = 2,
                    target_graph_num_obstacles: int = 2, target_graph_obstacle_hop: int = 2, obs_discount_factor: float = 1.0):
            """
            Initializes the RepeatedTopK method with the specified parameters.

            Args:
                reward_ratio (float): A weighting factor to balance visibility reward and distance penalty.
                env_graph (nx.Graph): The environment graph.
                sample_recursion (int): Recursion depth for stochastic path sampling.
                sample_num_obstacle (int): Number of obstacles per recursion level for sampling.
                sample_obstacle_hop (int): Hop radius for obstacle removal during sampling.
                target_graph_num_neighbors (int): Number of nearest neighbors for target graph construction.
                target_graph_recursions (int): Recursion depth for target graph diverse paths.
                target_graph_num_obstacles (int): Number of obstacles for target graph diverse paths.
                target_graph_obstacle_hop (int): Hop radius for target graph diverse paths.
                obs_discount_factor (float): Discount factor for visibility reward in path reward calculation.
            """
            self.reward_ratio = reward_ratio
            self.env_graph = env_graph
            self.target_graph = None  # Built dynamically in find_best_path
            self.obs_discount_factor = obs_discount_factor

            # sampling parameters
            self.recursion = sample_recursion
            self.num_obstacle = sample_num_obstacle
            self.obstacle_hop = sample_obstacle_hop

            # target graph generation parameters
            self.target_graph_num_neighbors = target_graph_num_neighbors
            self.target_graph_recursions = target_graph_recursions
            self.target_graph_num_obstacles = target_graph_num_obstacles
            self.target_graph_obstacle_hop = target_graph_obstacle_hop

    def build_target_graph(self, source):
        """
        Builds the target graph dynamically from the env_graph, then reweights
        each edge using process_section to account for reward-aware path costs.

        Args:
            source: The source node.

        Returns:
            The reweighted target graph (also stored as self.target_graph).
        """
        targets = [
            node for node, data in self.env_graph.nodes(data=True)
            if data.get('type') == 'target_unreached'
        ]
        if not targets:
            raise ValueError("No target nodes found in env_graph (type='target_unreached')")

        self.target_graph = create_fully_connected_target_graph(
            self.env_graph,
            source=source,
            targets=targets,
            num_neighbors=self.target_graph_num_neighbors,
            recursions=self.target_graph_recursions,
            num_obstacles=self.target_graph_num_obstacles,
            obstacle_hop=self.target_graph_obstacle_hop,
        )

        # Reweight each edge: run process_section to find the best path,
        # then use the negative reward as the initial weight (since TSP minimizes).
        for u, v in self.target_graph.edges():
            best_path = self.process_section(u, v)
            best_reward = calculate_path_reward(best_path, self.env_graph.copy(), self.reward_ratio, self.obs_discount_factor)
            self.target_graph.edges[u, v]['distance'] = -best_reward
            self.target_graph.edges[u, v]['best_path'] = best_path

        # Shift all weights so the minimum becomes 1:
        # new_weight = (-reward) + max_reward + 1
        max_reward = max(-self.target_graph.edges[u, v]['distance'] for u, v in self.target_graph.edges())
        for u, v in self.target_graph.edges():
            self.target_graph.edges[u, v]['distance'] += max_reward + 1

        return self.target_graph

    def generate_Hamiltonian_path(self, source):
        """
        Builds the target graph with reweighted edges, then generates the
        shortest Hamiltonian path using Lin-Kernighan TSP.

        Args:
            source: The source node.

        Returns:
            The Hamiltonian path (ordered list of target nodes starting from source).
        """
        self.build_target_graph(source)

        # Get TSP cycle
        node_list = list(self.target_graph.nodes())
        distance_matrix = nx.to_numpy_array(self.target_graph, nodelist=node_list, weight='distance', nonedge=np.inf)
        tsp_cycle, _ = solve_tsp_lin_kernighan(distance_matrix)
        tsp_cycle = [node_list[i] for i in tsp_cycle]

        if len(tsp_cycle) > 1 and tsp_cycle[0] == tsp_cycle[-1]:
            tsp_path_nodes = tsp_cycle[:-1]
        else:
            tsp_path_nodes = tsp_cycle

        # Rotate the cycle to start from source
        try:
            source_idx = tsp_path_nodes.index(source)
        except ValueError:
            raise ValueError(f"Source node {source} not found in TSP path {tsp_path_nodes}")
        hamiltonian_path = tsp_path_nodes[source_idx:] + tsp_path_nodes[:source_idx]

        return hamiltonian_path
    
    def deviate_path_at_node(self, base_path, node_in_path):
            """
            Creates a list of alternate paths by deviating at a certain node_in_path.
            
            Returns:
                list: A list of augmented paths (lists of nodes). Returns empty list if no deviations found.
            """
            candidate_paths = []

            # Find the index of the deviation node in the base path
            if node_in_path not in base_path:
                return []
            
            deviation_idx = base_path.index(node_in_path)
            
            # Can't deviate at the end node
            if deviation_idx == len(base_path) - 1:
                return []
            
            destination = base_path[-1]

            # Get all neighbors of the deviation node
            neighbors = list(self.env_graph.neighbors(node_in_path))
            
            # Filter out neighbors that are already in the base path
            valid_neighbors = [n for n in neighbors if n not in base_path]
            
            if not valid_neighbors:
                return []
            
            # Select neighbors to sample (up to num_neighbor_samples)
            sample_size = min(len(valid_neighbors), self.num_neighbor_samples)
            chosen_indices = np.random.choice(len(valid_neighbors), size=sample_size, replace=False)
            chosen_neighbors = [valid_neighbors[i] for i in chosen_indices]

            for neighbor in chosen_neighbors:
                # --- Strategy 1: Direct Deviation ---
                # Path: [... -> deviation_node -> neighbor -> ... -> dest]
                try:
                    # 1. Keep path up to deviation point
                    path_1 = base_path[:deviation_idx + 1]
                    # 2. Add the deviation node
                    path_1.append(neighbor)
                    # 3. Find shortest path back
                    return_path = nx.shortest_path(
                        self.env_graph, source=neighbor, target=destination, weight="distance"
                    )
                    path_1.extend(return_path[1:])
                    candidate_paths.append(path_1)
                except nx.NetworkXNoPath:
                    pass # Skip if no path back

                # --- Strategy 2: Neighbor of Neighbor Deviation ---
                if self.sample_neighbor_of_neighbor:
                    # Get neighbors of the current neighbor
                    nn_candidates = list(self.env_graph.neighbors(neighbor))
                    # Filter: shouldn't be in base_path and shouldn't be the deviation_node itself
                    valid_nn = [nn for nn in nn_candidates if nn not in base_path and nn != node_in_path]

                    if valid_nn:
                        # Pick ONE random neighbor of the neighbor
                        nn_target_idx = np.random.choice(len(valid_nn), replace=False)
                        nn_target = valid_nn[nn_target_idx]
                        
                        try:
                            path_2 = base_path[:deviation_idx + 1]
                            path_2.append(neighbor)
                            path_2.append(nn_target)
                            
                            return_path_nn = nx.shortest_path(
                                self.env_graph, source=nn_target, target=destination, weight="distance"
                            )
                            path_2.extend(return_path_nn[1:])
                            candidate_paths.append(path_2)
                        except nx.NetworkXNoPath:
                            pass

            return candidate_paths
    
    def deviate_path_stochastic_block(self, base_path):
        """
        Use the stochastic blocking strategy to generate candidate paths between two targets

        Returns: 
            list of potential paths between two targets
        """
        
        candidate_paths = []

        # Get paths with depth information
        candidate_paths_with_depth = stochastic_accumulated_blockage_path(
            self.env_graph,
            source=base_path[0],
            target=base_path[-1],
            recursions=self.recursion,
            num_obstacles_per_path=self.num_obstacle,
            obstacle_hop=self.obstacle_hop
        )
        
        # Extract just the paths (discard depth info for this use case)
        candidate_paths = [path for path, depth in candidate_paths_with_depth]
        
        return candidate_paths



    def process_section(self, begin_node, end_node):
            """
            Processes a single edge from the target graph and finds the best path
            """
            if begin_node not in self.target_graph.nodes() or end_node not in self.target_graph.nodes():
                raise ValueError("Begin or end node not in target graph.")
            
            if not self.target_graph.has_edge(begin_node, end_node):
                raise ValueError("The specified edge does not exist in the target graph.")

            diverse_paths_data = self.target_graph.edges[begin_node, end_node]['diverse_paths']
            shortest_path = diverse_paths_data[0].copy()

            best_reward = -np.inf
            best_path = None
            
            # Check if the path is in the REVERSE order
            if shortest_path[0] == end_node and shortest_path[-1] == begin_node:
                shortest_path = shortest_path[::-1]
            
            # Calculate the reward for the base path
            # Note: Assuming calculate_path_reward is available in scope or imported
            reward = calculate_path_reward(shortest_path, self.env_graph.copy(), self.reward_ratio, self.obs_discount_factor)
            if reward > best_reward:
                best_reward = reward
                best_path = shortest_path

            # Now get the list of alternate paths
                
            t_sample = time.perf_counter()
            deviated_paths = self.deviate_path_stochastic_block(shortest_path)
            t_sample = time.perf_counter() - t_sample
            num_samples = len(deviated_paths)
            
            t_reward = time.perf_counter()
            for d_path in deviated_paths:
                reward = calculate_path_reward(d_path, self.env_graph.copy(), self.reward_ratio, self.obs_discount_factor)
                if reward > best_reward:
                    best_reward = reward
                    best_path = d_path
            t_reward = time.perf_counter() - t_reward

            return best_path
    

    
    def alternate_path_online(self, begin_node, end_node):
            """
            Given a remaining path to a target, finds an alternate path.
            """
            if end_node not in self.target_graph.nodes():
                raise ValueError("End node should be a target node")

            best_reward = -np.inf
            best_path = None

            # First calculate the reward of the shortest path
            try:
                base_path = nx.shortest_path(self.env_graph, begin_node, end_node, "distance")
                reward = calculate_path_reward(base_path, self.env_graph.copy(), self.reward_ratio, self.obs_discount_factor)
                if reward > best_reward:
                    best_reward = reward
                    best_path = base_path
                
                deviated_paths = self.deviate_path_stochastic_block(base_path)

                    

                for d_path in deviated_paths:
                    reward = calculate_path_reward(d_path, self.env_graph.copy(), self.reward_ratio, self.obs_discount_factor)
                    if reward > best_reward:
                        best_reward = reward
                        best_path = d_path


            except nx.NetworkXNoPath:
                # Handle case where destination is unreachable
                return None

            return best_path

    def find_best_path(self, source):
        """
        Finds the best Hamiltonian path by:
        1. Building the target graph and reweighting edges via process_section
        2. Solving TSP on the reweighted target graph
        3. Stitching together the best section paths

        Args:
            source: The source node.

        Returns:
            The full path (list of nodes) visiting all targets.
        """
        base_Hamiltonian_path = self.generate_Hamiltonian_path(source)

        best_path = []

        # Stitch together the best paths for each section
        # (already computed and stored during build_target_graph)
        for i in range(len(base_Hamiltonian_path) - 1):
            begin_node = base_Hamiltonian_path[i]
            end_node = base_Hamiltonian_path[i + 1]

            section_best_path = self.target_graph.edges[begin_node, end_node]['best_path']

            # Ensure correct direction
            if section_best_path[0] == end_node and section_best_path[-1] == begin_node:
                section_best_path = section_best_path[::-1]

            if len(best_path) > 0 and best_path[-1] == section_best_path[0]:
                best_path.extend(section_best_path[1:])
            else:
                best_path.extend(section_best_path)

        return best_path
            




