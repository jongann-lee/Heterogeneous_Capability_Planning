"""
Simple information-value pruning for benchmark maps.

For the shortest path from source to target:
  1. Reroute cost:  for each chokepoint group on the path, remove it and
                    measure how much longer the alternative is.
  2. Observation cost:  for each blob, find the cheapest detour through any
                        of its plateau nodes (source → blob_node → target
                        minus the direct shortest path).

Metric = mean_reroute_cost − mean_observation_cost.
Positive → the reroute penalty exceeds the observation detour → keep the map.
"""

import numpy as np
import networkx as nx


def group_chokepoints(chokepoints):
    """Group chokepoint edges that share a node (union-find)."""
    if not chokepoints:
        return []

    node_to_edges = {}
    for idx, (u, v) in enumerate(chokepoints):
        node_to_edges.setdefault(u, []).append(idx)
        node_to_edges.setdefault(v, []).append(idx)

    parent = list(range(len(chokepoints)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for node, edge_ids in node_to_edges.items():
        for i in range(1, len(edge_ids)):
            union(edge_ids[0], edge_ids[i])

    groups = {}
    for idx in range(len(chokepoints)):
        groups.setdefault(find(idx), []).append(chokepoints[idx])
    return list(groups.values())


def compute_pruning_metric(map_data):
    """
    Returns:
        (metric_value, details_dict) or (None, error_string)
    """
    env_graph = map_data['env_graph']
    chokepoints = map_data['chokepoints']
    blobs = map_data.get('blobs', [])
    source = map_data['source']
    target = map_data['target']

    if not chokepoints:
        return None, "no chokepoints"
    if not blobs:
        return None, "no plateaus"

    # --- Baseline shortest path ---
    try:
        baseline_path = nx.shortest_path(env_graph, source, target, weight="distance")
        baseline_cost = nx.shortest_path_length(env_graph, source, target, weight="distance")
    except nx.NetworkXNoPath:
        return None, "no path source to target"

    # --- Reroute costs ---
    path_edges_norm = set()
    for i in range(len(baseline_path) - 1):
        path_edges_norm.add(tuple(sorted((baseline_path[i], baseline_path[i + 1]))))

    groups = group_chokepoints(list(chokepoints))
    reroute_costs = []

    for group in groups:
        group_norm = {tuple(sorted(e)) for e in group}
        if not group_norm & path_edges_norm:
            continue

        temp = env_graph.copy()
        for u, v in group:
            if temp.has_edge(u, v):
                temp.remove_edge(u, v)
            if temp.has_edge(v, u):
                temp.remove_edge(v, u)

        try:
            alt_cost = nx.shortest_path_length(temp, source, target, weight="distance")
            reroute_costs.append(alt_cost - baseline_cost)
        except nx.NetworkXNoPath:
            continue

    if not reroute_costs:
        return None, "no chokepoints on shortest path"

    # --- Observation detour costs (one per blob) ---
    blob_detours = []

    for blob_nodes in blobs:
        best_detour = float('inf')
        for p in blob_nodes:
            if p not in env_graph:
                continue
            try:
                cost_via = (
                    nx.shortest_path_length(env_graph, source, p, weight="distance")
                    + nx.shortest_path_length(env_graph, p, target, weight="distance")
                )
                best_detour = min(best_detour, cost_via - baseline_cost)
            except nx.NetworkXNoPath:
                continue

        if best_detour < float('inf'):
            blob_detours.append(best_detour)

    if not blob_detours:
        return None, "no reachable plateaus"

    mean_reroute = np.mean(reroute_costs)
    mean_observation = np.mean(blob_detours)
    metric = mean_reroute - mean_observation

    details = {
        'baseline_cost': float(baseline_cost),
        'n_groups': len(groups),
        'n_groups_on_path': len(reroute_costs),
        'reroute_costs': [float(x) for x in reroute_costs],
        'mean_reroute_cost': float(mean_reroute),
        'blob_detours': [float(x) for x in blob_detours],
        'mean_observation_cost': float(mean_observation),
        'metric': float(metric),
    }
    return metric, details


def prune_maps(candidates, threshold=0.0):
    """
    Prune maps based on: mean_reroute_cost − mean_observation_cost.

    Args:
        candidates: list of map_data dicts from generate_map_suite
        threshold:  minimum metric value to keep a map

    Returns:
        list of map_data dicts that passed pruning
    """
    kept = []
    for i, map_data in enumerate(candidates):
        label = map_data.get('label', '?')
        seed = map_data.get('seed', '?')
        print(f"  Pruning {i+1}/{len(candidates)} ({label}, seed={seed})...",
              end="", flush=True)

        metric, details = compute_pruning_metric(map_data)

        if metric is None:
            print(f" SKIP ({details})")
            continue

        if metric < threshold:
            # print(f" PRUNE (metric={metric:.2f}, "
            #       f"reroute={details['mean_reroute_cost']:.2f}, "
            #       f"obs={details['mean_observation_cost']:.2f})")
            continue

        kept.append(map_data)
        # print(f" KEEP (metric={metric:.2f}, "
        #       f"reroute={details['mean_reroute_cost']:.2f}, "
        #       f"obs={details['mean_observation_cost']:.2f})")

    print(f"\n  Pruning complete: {len(kept)}/{len(candidates)} maps kept")
    return kept