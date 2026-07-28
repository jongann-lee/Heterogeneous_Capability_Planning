import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import random
import pickle
import rasterio

from real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Single_Agent.repeated_topk import RepeatedTopK

# ==============================================================================
# CONFIGURATION TOGGLES
# ==============================================================================

# --- Agent Selection ---
USE_OUR_AGENT = True          # If True, uses RepeatedTopK agent. If False, uses shortest path agent.

# --- Run Settings ---
NUM_RUNS = 32

# --- Obstacle Toggles (which mountain obstacles to include in the map) ---
USE_OBSTACLE_1 = False   # (top-right)
USE_OBSTACLE_2 = False   # (center)
USE_OBSTACLE_3 = False   # (bottom-left)

# --- Our Agent Parameters ---
REWARD_RATIO = 1.0
TARGET_RECURSION = 4
TARGET_NUM_OBSTACLES = 8
TARGET_OBSTACLE_HOP = 4
SAMPLE_RECURSION = 5
SAMPLE_NUM_OBSTACLES = 4
SAMPLE_OBSTACLE_HOP = 4

# --- Map Save/Load ---
MAP_SAVE_PATH = "saved_WV_NP_map.pkl"

# ==============================================================================
# MAP DEFINITION
# ==============================================================================


def get_grid_from_local_dem(file_path, n_size):
    """
    Automatically detects bounds from a local 1m DEM and 
    resamples it to an N x N grid.
    """
    with rasterio.open(file_path) as dataset:
        # 1. Automatically get metadata/bounds
        bounds = dataset.bounds
        crs = dataset.crs
        print(f"File Bounds: {bounds}")
        print(f"Coordinate System: {crs}")

        # 2. Resample during read (Memory efficient)
        # We specify the output shape as (1, n_size, n_size) for (band, height, width)
        data = dataset.read(
            1,
            out_shape=(n_size, n_size),
            resampling=rasterio.enums.Resampling.bilinear # Or Resampling.max to preserve peaks
        )
        
        # 3. Handle NoData values (common in 1m DEMs)
        if dataset.nodata is not None:
            data = np.where(data == dataset.nodata, np.nan, data)

        return data, bounds

# --- Execution ---
# Replace with your West Virginia file path
dem_path = 'WV_DEM.tif' 
N = 64 

height_grid, dem_bounds = get_grid_from_local_dem(dem_path, N)



# --- Map Save/Load ---

def build_map_key():
    """Build a string key representing the current map configuration."""
    return (
        f"obstacles_{int(USE_OBSTACLE_1)}_{int(USE_OBSTACLE_2)}_{int(USE_OBSTACLE_3)}"
        f"_rec{TARGET_RECURSION}_nobs{TARGET_NUM_OBSTACLES}_hop{TARGET_OBSTACLE_HOP}"
    )

def generate_map():
    """Generate the environment graph and related structures."""
    corrected_height_grid = np.rot90(height_grid, k=-1)

    graph_generator = RealTerrainGrid(corrected_height_grid, k_up=1.0, k_down=2.0)
    graph_generator.compute_all_visibilities()
    env_graph = graph_generator.get_graph().copy()

    if USE_OBSTACLE_1:
        graph_generator.add_obstacle(center=(43,46), rx=5, ry=3)
    if USE_OBSTACLE_2:
        graph_generator.add_obstacle(center=(30,36), rx=3, ry=5)
    if USE_OBSTACLE_3:
        graph_generator.add_obstacle(center=(12,19), rx=4, ry=4)

    blocked_env_graph = graph_generator.get_graph().copy()

    target_graph = create_fully_connected_target_graph(env_graph, recursions=TARGET_RECURSION, num_obstacles=TARGET_NUM_OBSTACLES, obstacle_hop=TARGET_OBSTACLE_HOP)

    return env_graph, blocked_env_graph, target_graph


def load_or_generate_map():
    """Load map from disk if it matches current config, otherwise generate and save."""
    map_key = build_map_key()

    if os.path.exists(MAP_SAVE_PATH):
        print(f"Found saved map at '{MAP_SAVE_PATH}'. Checking configuration...")
        with open(MAP_SAVE_PATH, "rb") as f:
            saved_data = pickle.load(f)

        if saved_data.get("map_key") == map_key:
            print("Configuration matches. Loading saved map.")
            return saved_data["env_graph"], saved_data["blocked_env_graph"], saved_data["target_graph"]
        else:
            print(f"Configuration mismatch (saved: {saved_data.get('map_key')}, current: {map_key}). Regenerating map...")
    else:
        print("No saved map found. Generating map...")

    env_graph, blocked_env_graph, target_graph = generate_map()

    print(f"Saving map to '{MAP_SAVE_PATH}'...")
    with open(MAP_SAVE_PATH, "wb") as f:
        pickle.dump({"map_key": map_key, "env_graph": env_graph, "blocked_env_graph": blocked_env_graph, "target_graph": target_graph}, f)
    print("Map saved.")

    return env_graph, blocked_env_graph, target_graph


env_graph, blocked_env_graph, target_graph = load_or_generate_map()


def run_benchmark():
    # --- Load or generate map ---
    env_graph, blocked_env_graph, target_graph = load_or_generate_map()

    # Remove all edges connected to obstacle nodes from the blocked_env_graph
    edges_to_remove = [(u, v) for u, v in blocked_env_graph.edges() 
                    if blocked_env_graph.nodes[u].get('type') == 'obstacle' 
                    or blocked_env_graph.nodes[v].get('type') == 'obstacle']

    blocked_env_graph.remove_edges_from(edges_to_remove)

    # Update the visible_edges for all nodes to reflect the removed edges
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            current_visible = blocked_env_graph.nodes[node]["visible_edges"]
            # Remove any blocked edges from the visible_edges set
            updated_visible = set(current_visible) - set(edges_to_remove)
            blocked_env_graph.nodes[node]["visible_edges"] = list(updated_visible)

    shortest_path = nx.shortest_path(env_graph, source=(N-1, N-1), target=(0, 0), weight="distance")

    path_length_list = []
    runtimes = []
    num_samples_list = []
    t_sample_list = []
    t_reward_list = []

    agent_label = "Our Agent (RepeatedTopK)" if USE_OUR_AGENT else "Shortest Path Agent"
    print(f"\nRunning benchmark with: {agent_label}")

    for run_idx in tqdm(range(NUM_RUNS)):
        np.random.seed(42 + run_idx)
        random.seed(42 + run_idx)

        start_time = time.time()

        if not nx.has_path(blocked_env_graph, (0, 0), (N-1, N-1)):
            continue

        # --- Shortest Path Agent ---
        if not USE_OUR_AGENT:
            path_1 = shortest_path.copy()
            target_nodes = [(N-1, N-1), (0,0)]
            env_graph1 = env_graph.copy()

            current_node = path_1[0]
            next_target_index = 1
            index = 0
            total_travel_distance = 0.0

            while index < len(path_1) - 1:
                next_node = path_1[index + 1]

                # Update the next_target_index 
                if current_node == target_nodes[next_target_index]:
                    next_target_index += 1
                
                # Check out all the observable edges
                observable_edges = set(blocked_env_graph.nodes[current_node]["visible_edges"]) # From the actual blocked env graph
                assumed_observable_edges = set(env_graph1.nodes[current_node]["visible_edges"]) # From the agent's world model 
                blocked_edges = assumed_observable_edges - observable_edges
                
                if len(blocked_edges) > 0:
                    # Remove blocked edges from agent's world model
                    for edge in blocked_edges:
                        u, v = edge
                        if env_graph1.has_edge(u, v):
                            env_graph1.remove_edge(u, v)
                    # Update the visibility mapping as well
                    for node in env_graph1.nodes():
                        if "visible_edges" in env_graph1.nodes[node]:
                            current_visible = env_graph1.nodes[node]["visible_edges"]
                            # Remove any blocked edges from the visible_edges set
                            updated_visible = set(current_visible) - blocked_edges
                            env_graph1.nodes[node]["visible_edges"] = list(updated_visible)
                    
                    # Check if any blocked edge is in our current path
                    path_edges = [(path_1[i], path_1[i+1]) for i in range(index, len(path_1) - 1)]

                    # Create a set that includes both directions of blocked edges
                    blocked_edges_both_directions = set()
                    for u, v in blocked_edges:
                        blocked_edges_both_directions.add((u, v))
                        blocked_edges_both_directions.add((v, u))
                    
                    if any(edge in blocked_edges_both_directions for edge in path_edges):
                        
                        # Replan through ALL remaining targets
                        remaining_targets = target_nodes[next_target_index:]
                        new_path = [current_node]
                        
                        for target in remaining_targets:
                            segment = nx.shortest_path(env_graph1, source=new_path[-1], target=target, weight="distance")
                            new_path.extend(segment[1:])  # Append segment excluding the first node (already in new_path)
                        
                        # Replace the rest of path_1 with the new path
                        path_1 = path_1[:index + 1] + new_path[1:]

                # Get the next node from the (possibly updated) path
                next_node = path_1[index + 1]
                total_travel_distance += blocked_env_graph.edges[current_node, next_node]["distance"]
                
                # Now that we have a traversable path, just go to the next node in path
                current_node = next_node
                index += 1

            path_length_list.append(total_travel_distance)

        # --- Our Agent ---
        else:
            env_graph2 = env_graph.copy()
            path2_generator = RepeatedTopK(
                reward_ratio=REWARD_RATIO,
                env_graph=env_graph2,
                target_graph=target_graph,
                sample_recursion=SAMPLE_RECURSION,
                sample_num_obstacle=SAMPLE_NUM_OBSTACLES,
                sample_obstacle_hop=SAMPLE_OBSTACLE_HOP
            )

            path_2, num_samples, t_sample, t_reward = path2_generator.find_best_path()
            target_nodes = [(N-1, N-1), (0,0)]

            current_node = path_2[0]
            next_target_index = 1
            index = 0
            total_travel_distance = 0.0

            while index < len(path_2) - 1:
                next_node = path_2[index + 1]

                # Update the next_target_index 
                if current_node == target_nodes[next_target_index]:
                    next_target_index += 1
                
                # Check out all the observable edges
                observable_edges = set(blocked_env_graph.nodes[current_node]["visible_edges"]) # From the actual blocked env graph
                assumed_observable_edges = set(env_graph2.nodes[current_node]["visible_edges"]) # From the agent's world model 
                blocked_edges = assumed_observable_edges - observable_edges

                # MARK OBSERVED EDGES AS SEEN
                for edge in assumed_observable_edges:
                    if env_graph2.has_edge(*edge):
                        env_graph2.edges[edge]["observed_edge"] = True

                if len(blocked_edges) > 0:
                    # Remove blocked edges from agent's world model
                    for edge in blocked_edges:
                        u, v = edge
                        if env_graph2.has_edge(u, v):
                            env_graph2.remove_edge(u, v)
                    
                    # Update the visibility mapping as well
                    for node in env_graph2.nodes():
                        if "visible_edges" in env_graph2.nodes[node]:
                            current_visible = env_graph2.nodes[node]["visible_edges"]
                            # Remove any blocked edges from the visible_edges set
                            updated_visible = set(current_visible) - blocked_edges
                            env_graph2.nodes[node]["visible_edges"] = list(updated_visible)
                    
                    # Check if any blocked edge is in our current path
                    path_edges = [(path_2[i], path_2[i+1]) for i in range(index, len(path_2) - 1)]

                    # Create a set that includes both directions of blocked edges
                    blocked_edges_both_directions = set()
                    for u, v in blocked_edges:
                        blocked_edges_both_directions.add((u, v))
                        blocked_edges_both_directions.add((v, u))

                    if any(edge in blocked_edges_both_directions for edge in path_edges):
                        
                        # Replan through ALL remaining targets
                        remaining_targets = target_nodes[next_target_index:]
                        new_path = [current_node]
                        
                        for target in remaining_targets:
                            segment = path2_generator.alternate_path_online(new_path[-1], target)
                            new_path.extend(segment[1:])  # Append segment excluding the first node (already in new_path)
                        
                        # Replace the rest of path_1 with the new path
                        path_2 = path_2[:index + 1] + new_path[1:]

                # Get the next node from the (possibly updated) path
                next_node = path_2[index + 1]
                total_travel_distance += blocked_env_graph.edges[current_node, next_node]["distance"]
                current_node = next_node
                index += 1

            path_length_list.append(total_travel_distance)
            num_samples_list.append(num_samples)
            t_sample_list.append(t_sample)
            t_reward_list.append(t_reward)

        end_time = time.time()
        runtimes.append(end_time - start_time)

    # --- Print Results ---
    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)

    print(f"\nAGENT: {'Our Agent (RepeatedTopK)' if USE_OUR_AGENT else 'Shortest Path Agent'}")

    print(f"\nOBSTACLES USED:")
    print(f"  Obstacle 1 (top-right)  : {'ON' if USE_OBSTACLE_1 else 'OFF'}")
    print(f"  Obstacle 2 (center)     : {'ON' if USE_OBSTACLE_2 else 'OFF'}")
    print(f"  Obstacle 3 (bottom-left): {'ON' if USE_OBSTACLE_3 else 'OFF'}")

    print(f"\nTARGET GRAPH PARAMETERS:")
    print(f"  Recursion:       {TARGET_RECURSION}")
    print(f"  Num obstacles:   {TARGET_NUM_OBSTACLES}")
    print(f"  Obstacle hop:    {TARGET_OBSTACLE_HOP}")

    if USE_OUR_AGENT:
        print(f"\nOUR AGENT PARAMETERS:")
        print(f"  Reward ratio:    {REWARD_RATIO:.2f}")
        print(f"  Sample recursion:     {SAMPLE_RECURSION}")
        print(f"  Sample num obstacles: {SAMPLE_NUM_OBSTACLES}")
        print(f"  Sample obstacle hop:  {SAMPLE_OBSTACLE_HOP}")

    if path_length_list:
        mean_cost = np.mean(path_length_list)
        variance_cost = np.var(path_length_list)
        std_dev = np.std(path_length_list)
        avg_runtime = np.mean(runtimes)

        print(f"\nRUN SETTINGS:")
        print(f"  Num runs:           {NUM_RUNS}")
        print(f"  Successful runs:    {len(path_length_list)}")

        print(f"\nPERFORMANCE:")
        print(f"  Mean path cost:     {mean_cost:.2f}")
        print(f"  Variance:           {variance_cost:.2f}")
        print(f"  Std deviation:      {std_dev:.2f}")
        print(f"  Avg runtime/run:    {avg_runtime:.4f}s")
    else:
        print("\nNo successful runs to report.")
    
    print(f"\nADDITIONAL METRICS FOR OUR AGENT:")
    if num_samples_list:
        mean_samples = np.mean(num_samples_list)
        mean_t_sample = np.mean(t_sample_list)
        mean_t_reward = np.mean(t_reward_list)
        print(f"  Mean samples:       {mean_samples:.2f}")
        print(f"  Mean sample time:   {mean_t_sample:.4f}s")
        print(f"  Mean reward time:   {mean_t_reward:.4f}s")

    print("=" * 40)


if __name__ == "__main__":
    run_benchmark()