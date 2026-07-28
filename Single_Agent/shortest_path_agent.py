import networkx as nx
import numpy as np
from Single_Agent.lin_kernighan_tsp import solve_tsp_lin_kernighan
from Graph_Generation.target_graph import create_fully_connected_target_graph


class ShortestPathAgent:
    def __init__(self, env_graph, target_graph_num_neighbors, target_graph_recursions, target_graph_obstacle_hop, target_graph_num_obstacles):
        self.env_graph = env_graph

        self.target_graph_num_neighbors = target_graph_num_neighbors
        self.target_graph_recursions = target_graph_recursions
        self.target_graph_obstacle_hop = target_graph_obstacle_hop
        self.target_graph_num_obstacles = target_graph_num_obstacles


    def single_target_path(self, source, target):
        """
        Compute the shortest path between source and target on self.env_graph.
        """
        return nx.shortest_path(self.env_graph, source=source, target=target, weight="distance")

    def target_order(self, source, targets):
        """
        Determine the order in which to visit targets.

        Builds a fully connected target graph over the source + targets,
        then solves a Hamiltonian path starting from the source.

        Returns:
            Ordered list of nodes to visit (starting with source).
        """
        target_graph = create_fully_connected_target_graph(
            self.env_graph,
            source=source,
            targets=targets,
            num_neighbors=self.target_graph_num_neighbors,
            recursions=self.target_graph_recursions,
            obstacle_hop=self.target_graph_obstacle_hop,
            num_obstacles=self.target_graph_num_obstacles
        )

        # --- Solve TSP for Hamiltonian path ---
        node_list = list(target_graph.nodes())
        distance_matrix = nx.to_numpy_array(target_graph, nodelist=node_list, weight="distance", nonedge=np.inf)
        tsp_cycle, _ = solve_tsp_lin_kernighan(distance_matrix)
        tsp_cycle = [node_list[i] for i in tsp_cycle]

        # Convert cycle to path
        if len(tsp_cycle) > 1 and tsp_cycle[0] == tsp_cycle[-1]:
            tsp_path_nodes = tsp_cycle[:-1]
        else:
            tsp_path_nodes = tsp_cycle

        # Rotate so the source is first
        try:
            source_idx = tsp_path_nodes.index(source)
        except ValueError:
            raise ValueError(f"Source node {source} not found in TSP path {tsp_path_nodes}")
        hamiltonian_path = tsp_path_nodes[source_idx:] + tsp_path_nodes[:source_idx]

        return hamiltonian_path

    def generate_path(self, source):
        """
        Generate a full path visiting all targets in an efficient order.

        Discovers targets from self.env_graph (nodes with type 'target_unreached'),
        determines visitation order via Hamiltonian path, then connects consecutive
        waypoints with shortest paths.

        Returns:
            List of nodes representing the full path from source through all targets.
        """
        targets = [
            node for node, data in self.env_graph.nodes(data=True)
            if data.get('type') == 'target_unreached'
        ]
        if not targets:
            raise ValueError("No target nodes found in env_graph (type='target_unreached')")

        ordered_waypoints = self.target_order(source, targets)

        full_path = []
        for i in range(len(ordered_waypoints) - 1):
            segment = self.single_target_path(ordered_waypoints[i], ordered_waypoints[i + 1])
            if full_path:
                # Avoid duplicating the connecting node
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)

        return full_path
