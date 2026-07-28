import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import random
import itertools
import pickle

from Graph_Generation.height_graph_generation import HeightMapGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Graph_Generation.edge_block_generation import block_specific_edges
from Single_Agent.repeated_topk import RepeatedTopK
from Single_Agent.RPP import ZeroCostRPP

# Settings
edge_block_prob = float(0.5)
num_runs = 1000

use_shortest_path_agent = False
use_our_agent = True
use_RPP_agent = False

# Generate the graph (same as in single_agent_height_grid.ipynb)

# These are the "mountains"
mountain1 = [(0,11), (1,11), (2,11), (3,11), 
             (0,10), (1,10), (2,10), (3,10),
             (0,9),  (1,9),  (2,9),
             (0,8), (1,8)]
mountain2 = [(3,7), (4,7), (5,7), 
             (1,6), (2,6), (3,6), (4,6), (5,6), 
             (1,5), (2,5), (3,5), (4,5),
             (1,4), (2,4), (3,4), (4,4),
             (3,3), (4,3)]
mountain3 = [(7,5),
         (6,4), (7,4), (8,4), (9,4), (10,4), 
         (6,3), (7,3), (8,3), (9,3), (10,3),
         (6,2), (7,2), (8,2), (9,2),
         (6,1), (7,1)]
mountain4 = [(6,10), (7,10), (8,10), (9,10), (10,10),
             (6,9), (7,9), (8,9), (9,9), (10,9),
             (7,8), (8,8), (9,8), (10,8),
             (9,7), (10,7),
             (9,6), (10,6)]
set_of_blobs = [mountain1, mountain2, mountain3, mountain4]

map_generator = HeightMapGrid(m=12, n=12)
map_generator.add_plataeu(mountain1)
map_generator.add_plataeu(mountain2)
map_generator.add_plataeu(mountain3)
map_generator.add_plataeu(mountain4)
map_generator.calculate_distances()

# map_generator.calculate_visibility()
map_generator.calculate_simple_visibility(set_of_blobs)

# Remove the edges that connect the mountain to the valley
edge_list_1 = [((0,7), (0,8)), ((1,8), (2,8)), ((2,8), (2,9)), ((2,9), (3,9)),
               ((3,9), (3,10)), ((3,11), (4,11))]
edge_list_2 = [((3,7), (3,8)), ((4,7), (4,8)), ((5,7), (5,8)),
               ((5,7), (6,7)), ((5,6), (6,6)), ((5,5), (5,6)),
                ((4,5), (5,5)), ((4,4), (5,4)), ((4,3), (5,3)),
                ((4,2), (4,3)), ((3,2), (3,3)), ((2,3), (2,4)),
                ((1,3), (1,4)), ((0,4), (1,4)), ((0,5), (1,5)),
                ((0,6), (1,6)), ((1,6), (1,7)), ((2,6), (2,7))]
edge_list_3 = [((7,5), (8,5)), ((8,4), (8,5)), ((9,4), (9,5)),
               ((10,4), (10,5)), ((10,4), (11,4)), ((10,3), (11,3)),
               ((10,2), (10,3)), ((8,1), (8,2)),
                ((7,0), (7,1)), ((9,1), (9,2)),
               ((5,1), (6,1)), ((5,2), (6,2)), ((5,3), (6,3)),
               ((5,4), (6,4)), ((6,4), (6,5)), ((6,5), (7,5))]
edge_list_4 = [((7,10), (7,11)), ((8,10), (8,11)), ((9,10), (9,11)),
               ((10,10), (10,11)), ((10,10), (11,10)), ((10,9), (11,9)),
               ((10,8), (11,8)), ((10,7), (11,7)), ((10,5), (10,6)),
               ((9,5), (9,6)), ((8,6), (9,6)), ((8,7), (8,8)),
               ((7,7), (7,8)), ((6,8), (7,8)), ((6,8), (6,9)),
               ((5,9), (6,9)), ((5,10), (6,10))]

edge_list = edge_list_1 + edge_list_2 + edge_list_3 + edge_list_4
map_generator.remove_edges(edge_list)

# env_graph = map_generator.get_graph()

#IF YOU ARE USING THE SAVED MAP
with open('Automatic_Generated_Maps/top_bottom_maps.pkl', 'rb') as f:
    data = pickle.load(f)

top3 = data['top3']      # list of (map_data, improvement_pct)
bottom3 = data['bottom3']  # list of (map_data, improvement_pct)
md_top, imp_top = top3[0]
md_bottom, imp_bottom = bottom3[0]


env_graph = md_top['env_graph']

target_recursion = 4
target_num_obstacles = 4
target_obstacle_hop = 1

target_graph = create_fully_connected_target_graph(env_graph, recursions=target_recursion,
                                                    num_obstacles=target_num_obstacles, 
                                                    obstacle_hop=target_obstacle_hop)


# Define the chokepoints (the edges that can be blocked)

chokepoints_list = [((7,11), (8,11)), ((8,11), (9,11)), ((9,11), (10,11)),
                    ((11,7), (11,8)), ((11,8), (11,9)), ((11,9), (11,10)),
                    ((8,5), (8,6)), ((8,5), (9,5)), ((9,5), (10,5)), 
                    ((0,4), (0,5)), ((0,5), (0,6)), ((0,6), (0,7)),
                    ((5,3), (5,4)), ((5,4), (5,5)), ((5,5), (6,5)),
                    ((11,2), (11,3)), ((11,3), (11,4)), ((11,4), (11,5))]

# chokepoints_list = md_top['chokepoints']

chokepoints_list=[((9,0), (10,0)),
                  ((8,5), (9,5)),
                  ((9,9), (10,9)),
                  ((0,12), (0,13)),
                  ((6,13), (6,14)),
                  ((4,15), (5,15)),
                  ((8,15), (9,15)),
                  ((13,15), (14,15))

]

# chokepoints_list = [((9,11), (10,11)),
#                     ((11,9), (11,10)),
#                     ((8,5), (9,5)), 
#                     ((0,6), (0,7)),
#                     ((5,5), (6,5)),
#                     ((11,4), (11,5))]

# Pre calculate shortest path and the Hamiltonian target path(trivial for now)
path_generator = RepeatedTopK(reward_ratio = 0.0, env_graph=env_graph, target_graph=target_graph,
                              sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)

shortest_path = []
hamiltonian_target_path = path_generator.generate_Hamiltonian_path()
for i in range(len(hamiltonian_target_path) - 1):
    begin_node = hamiltonian_target_path[i]
    end_node = hamiltonian_target_path[i + 1]

    section_path = nx.shortest_path(env_graph, source=begin_node, target=end_node, weight="distance")

    # Append the section best path, avoiding duplication of nodes at the end
    if len(shortest_path) > 0 and shortest_path[-1] == section_path[0]:
        shortest_path.extend(section_path[1:])
    else:
        shortest_path.extend(section_path)

path_length_list = []
runtimes = []


# Main repeating run loop
for run_idx in tqdm(range(num_runs)):

    # Set the seed here so that each test gets the same samples of the edge block distribution
    np.random.seed(42+run_idx)
    random.seed(42+run_idx)
    
    start_time = time.time()

    # Block edges
    edges_to_remove = []
    RNG = np.random.rand(len(chokepoints_list)) # if below 0.75 then edge is blocked
    for i, (u, v) in enumerate(chokepoints_list):
        if RNG[i] < edge_block_prob:
            edges_to_remove.append((u, v))
            # print(f" Blocking edge{i}")


    # Create the blocked env graph (real env) and remove the blocked edges
    blocked_env_graph = block_specific_edges(env_graph, edges_to_remove)
    blocked_env_graph.remove_edges_from(edges_to_remove)

    # Skip run if the goal is unreachable
    if not nx.has_path(blocked_env_graph, (0,0), (11,11)):
        continue

    # Update the visible_edges for all nodes to reflect the removed edges
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            current_visible = blocked_env_graph.nodes[node]["visible_edges"]
            # Remove any blocked edges from the visible_edges set
            updated_visible = set(current_visible) - set(edges_to_remove)
            blocked_env_graph.nodes[node]["visible_edges"] = list(updated_visible)

    # Get the path taken by the shortest path agent
    if use_shortest_path_agent:
        path_1 = shortest_path.copy() # Start with the shortest path
        target_nodes = hamiltonian_target_path.copy()  # All target nodes
        env_graph1 = env_graph.copy() # Agent's world model (doesn't know any edges are blocked)

        current_node = path_1[0]
        next_target_index = 1 # since 0 is the source
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
                        # print(f"From {current_node}, observed edge ({u}, {v}) is blocked. Removing from graph.")
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
                    # print(f"Blocked edge detected in planned path. Recalculating entire remaining path...")
                    
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
            
            # Now that we have a traversable path, just go to the next node in path
            # print(f"Moving from {current_node} to {next_node}")
            total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
            current_node = next_node
            index += 1

        path_length_list.append(total_travel_distance)

    elif use_our_agent:
        env_graph2 = env_graph.copy() # Agent's world model (doesn't know any edges are blocked)
        path2_generator = RepeatedTopK(reward_ratio = 10.0, env_graph=env_graph2, target_graph=target_graph,
                                       sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)

        path_2 = path2_generator.find_best_path() # Start with the best path
        target_nodes = hamiltonian_target_path.copy()  # All target nodes
        

        current_node = path_2[0]
        next_target_index = 1 # since 0 is the source
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
                        # print(f"From {current_node}, observed edge ({u}, {v}) is blocked. Removing from graph.")
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
                    # print(f"Blocked edge detected in planned path. Recalculating entire remaining path...")
                    
                    # Replan through ALL remaining targets
                    remaining_targets = target_nodes[next_target_index:]
                    new_path = [current_node]
                    
                    for target in remaining_targets:
                        segment = path2_generator.alternate_path_online(new_path[-1], target)
                        new_path.extend(segment[1:])  # Append segment excluding the first node (already in new_path)
                    
                    # Replace the rest of path_2 with the new path
                    path_2 = path_2[:index + 1] + new_path[1:]

            # Get the next node from the (possibly updated) path
            next_node = path_2[index + 1]
            
            # Now that we have a traversable path, just go to the next node in path
            # print(f"Moving from {current_node} to {next_node}")
            total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
            current_node = next_node
            index += 1

        path_length_list.append(total_travel_distance)

    elif use_RPP_agent:
        env_graph4 = env_graph.copy() # Agent's world model (doesn't know any edges are blocked)

        # Generate all 2^n possible realizations for the n chokepoints
        realizations = []
        chokepoints = chokepoints_list
                
        source_node = (0,0)
        target_node = (15,15)
        edge_block_prob = edge_block_prob

        for bits in itertools.product([0, 1], repeat=len(chokepoints)):
            # 0 = open, 1 = blocked
            blocked_in_this_world = [chokepoints[i] for i, bit in enumerate(bits) if bit == 1]
            
            # Calculate probability of this specific world occurring
            prob = 1.0
            for bit in bits:
                prob *= edge_block_prob if bit == 1 else (1 - edge_block_prob)
            
            # The set of free edges is all edges in env_graph MINUS the blocked ones
            all_edges = set(env_graph.edges())
            free_edges = all_edges - set(blocked_in_this_world)
            
            realizations.append({
                'prob': prob,
                'edges': free_edges,
                'blocked': set(blocked_in_this_world)
            })
        

        path4_generator = ZeroCostRPP(env_graph=env_graph4,
                                    realizations=realizations,
                                    start_node=source_node,
                                    goal_node=target_node)

        policy_tree = path4_generator.plan()

        # Y is the set of indices of all realizations [cite: 119, 159]
        current_belief_indices = tuple(range(len(realizations)))
        current_tree_node = (current_belief_indices, source_node)
        current_node = source_node

        # Edges open in ALL realizations are known safe from the start
        confirmed_safe_edges = set(
            e for e in env_graph4.edges()
            if all(e in r['edges'] for r in realizations)  # open in ALL realizations = never uncertain
        )


        agent4_travel_distance = 0.0

        while True:
            node_data = policy_tree.nodes.get(current_tree_node, {})
            
            # 1. Check for Terminal States [cite: 173, 408]
            if node_data.get('type') == 'terminal_no_goal':
                # print(f"RPP Determination: Goal is unreachable in this environment.")
                break
            
            if current_node == target_node:
                # print(f"Goal Reached! Total distance: {agent4_travel_distance}")
                break

            # 2. Get the next tree action [cite: 170, 250]
            successors = list(policy_tree.successors(current_tree_node))
            if not successors:
                break
            
            next_tree_state = successors[0]
            target_env_vertex = next_tree_state[1] 

            # *** CHANGE: Plan leg only on confirmed safe edges ***
            safe_subgraph = env_graph4.edge_subgraph(confirmed_safe_edges)
            try:
                leg_path = nx.shortest_path(safe_subgraph, current_node, target_env_vertex, weight="distance")
            except nx.NetworkXNoPath:
                print(f"No confirmed-safe path from {current_node} to {target_env_vertex}. Agent is stuck.")
                break
            
            # *** CHANGE: Check each step for blocked edges before committing ***
            for next_step in leg_path[1:]:
                if not blocked_env_graph.has_edge(current_node, next_step):
                    # Discovered a blocked edge mid-leg — stop here
                    print(f"Blocked edge ({current_node}, {next_step}) discovered mid-leg.")
                    confirmed_safe_edges.discard((current_node, next_step))
                    confirmed_safe_edges.discard((next_step, current_node))
                    break
                distance = blocked_env_graph.edges[current_node, next_step]["distance"]
                agent4_travel_distance += distance
                current_node = next_step

            # 4. Perform Observation and Transition Belief [cite: 152, 252]
            # If we aren't at the goal yet, we must update our belief based on sensors [cite: 153, 231]
            if current_node != target_node:
                visible_here = env_graph4.nodes[current_node].get("visible_edges", [])
                actual_outcome = tuple(sorted(
                    [e for e in visible_here if blocked_env_graph.has_edge(*e)],
                    key=lambda x: (x[0], x[1])
                ))
                
                # Look for the specific branch that matches what we just saw 
                found_next_belief = False
                for potential_next_belief in policy_tree.successors(next_tree_state):
                    edge_data = policy_tree.get_edge_data(next_tree_state, potential_next_belief)
                    if edge_data.get('observation_result') == actual_outcome:
                        current_tree_node = potential_next_belief # Jump to the new state 
                        found_next_belief = True
                        break
                
                if not found_next_belief:
                    # print(f"Warning: Outcome {actual_outcome} at {current_node} is inconsistent with policy.")
                    break
            else:
                # Final safety check: if we are physically at target_node, we are done
                # print(f"Goal Reached! Total distance: {agent4_travel_distance}")
                break

        path_length_list.append(agent4_travel_distance)

        
    end_time = time.time()
    runtimes.append(end_time - start_time)
        
if path_length_list:
    mean_cost = np.mean(path_length_list)
    variance_cost = np.var(path_length_list)
    std_dev = np.std(path_length_list)
    avg_runtime = np.mean(runtimes)

    if use_our_agent:
        print("\n" + "="*30)
        print(f"MODEL PARAMETERS")
        print(f"Edge Recursion:     {target_recursion:d}")
        print(f"Edge num obstacles: {target_num_obstacles:d}")
        print(f"Edge obstacle hop:  {target_obstacle_hop:d}")
        print(f"Plan recursion      {path2_generator.recursion:d}")
        print(f"Plan num obstacles: {path2_generator.num_obstacle:d}")
        print(f"Plan obstacle hop:  {path2_generator.obstacle_hop:d}")
        print(f"Reward ratio:       {path2_generator.reward_ratio:.2f}")
     
    print("-" * 30)
    print(f"RESULTS FOR {len(path_length_list)} SUCCESSFUL RUNS")
    print("-" * 30)
    print(f"Mean Path Cost:     {mean_cost:.2f}")
    print(f"Variance:           {variance_cost:.2f}")
    print(f"Std Deviation:      {std_dev:.2f}")
    print(f"Avg Runtime/Run:    {avg_runtime:.4f}s")
    print("="*30)