"""
Analytic pruning score for generated maps.

Scores maps based on structural properties that predict whether the
visibility-aware RepeatedTopK agent will outperform the shortest-path agent.
No simulation is required -- only graph analysis.

Key insight: Our agent wins when:
1. Chokepoint blockages force detours
2. Those detours pass near blobs (plateaus)
3. Being on/near a blob lets you SEE chokepoint edges before reaching them
4. So the agent can preemptively reroute, avoiding wasted travel

The score captures these structural properties analytically.
"""

import numpy as np
import networkx as nx


def score_map(map_data, verbose=False):
    """
    Computes an analytic quality score for a generated map.
    Higher score = more likely that RepeatedTopK outperforms SP.

    Returns:
        float: composite score (higher is better)
        dict: component scores for diagnostics
    """
    g = map_data['env_graph']
    source = map_data['source']
    target = map_data['target']
    chokepoints = map_data['chokepoints']
    blobs = map_data['blobs']
    block_prob = map_data.get('block_prob', 0.5)

    all_blob_nodes = set()
    for blob in blobs:
        all_blob_nodes.update(blob)

    components = {}

    # ---- Score 1: Chokepoint visibility from blobs ----
    # How many chokepoints can be seen from blob nodes?
    # This is the core value proposition: the agent enters a blob to see
    # whether chokepoints ahead are blocked.
    cp_visibility_score = _score_chokepoint_visibility(g, chokepoints, blobs)
    components['cp_visibility'] = cp_visibility_score

    # ---- Score 2: Detour-near-blob score ----
    # When chokepoints block, does the detour route pass near blobs?
    # If yes, the agent could have scouted from the blob first.
    detour_blob_score = _score_detour_near_blob(
        g, source, target, chokepoints, all_blob_nodes, block_prob
    )
    components['detour_near_blob'] = detour_blob_score

    # ---- Score 3: Blob entry efficiency ----
    # How much extra cost to enter & exit a blob vs staying in corridor?
    # Lower detour cost = more likely agent uses blob scouting.
    entry_efficiency = _score_blob_entry_efficiency(g, blobs, all_blob_nodes)
    components['entry_efficiency'] = entry_efficiency

    # ---- Score 4: Chokepoint impact ----
    # How much does blocking chokepoints increase path cost?
    # Higher impact = more value from preemptive scouting.
    cp_impact = _score_chokepoint_impact(g, source, target, chokepoints, block_prob)
    components['cp_impact'] = cp_impact

    # ---- Score 5: Path diversity through blobs ----
    # Are there meaningfully different routes that go through blobs vs around?
    path_diversity = _score_path_diversity(g, source, target, all_blob_nodes)
    components['path_diversity'] = path_diversity

    # Composite score: weights calibrated from empirical correlation analysis
    # detour_near_blob (r=0.545) is the strongest predictor
    # entry_efficiency (r=0.216) and cp_visibility (r=0.076) are secondary
    # cp_impact (r=0.043) provides minimal signal
    # path_diversity (r=-0.191) is slightly negative -- penalize slightly
    #
    # Hard filter: detour_near_blob < 0.5 means the map is almost certainly bad
    if detour_blob_score < 0.5:
        composite = -10.0  # Reject immediately
    else:
        composite = (
            1.0 * cp_visibility_score +
            5.0 * detour_blob_score +
            2.0 * entry_efficiency +
            0.5 * cp_impact +
            -0.5 * path_diversity
        )
    components['composite'] = composite

    if verbose:
        print(f"  cp_visibility:  {cp_visibility_score:.3f}")
        print(f"  detour_near_blob: {detour_blob_score:.3f}")
        print(f"  entry_efficiency: {entry_efficiency:.3f}")
        print(f"  cp_impact:      {cp_impact:.3f}")
        print(f"  path_diversity: {path_diversity:.3f}")
        print(f"  COMPOSITE:      {composite:.3f}")

    return composite, components


def _score_chokepoint_visibility(g, chokepoints, blobs):
    """
    What fraction of chokepoints are visible from at least one blob node?

    For each chokepoint edge (u,v), check if any blob node can see it
    (i.e., the chokepoint edge appears in that blob node's visible_edges).
    """
    if not chokepoints:
        return 0.0

    # Build set of chokepoint edges (sorted tuples)
    cp_set = set()
    for u, v in chokepoints:
        cp_set.add(tuple(sorted((u, v))))

    # For each blob node, check what chokepoints it can see
    visible_cps = set()
    for blob in blobs:
        for node in blob:
            if node not in g.nodes():
                continue
            vis = g.nodes[node].get('visible_edges', [])
            for edge in vis:
                edge_key = tuple(sorted(edge))
                if edge_key in cp_set:
                    visible_cps.add(edge_key)

    return len(visible_cps) / len(cp_set)


def _score_detour_near_blob(g, source, target, chokepoints, all_blob_nodes,
                             block_prob, num_trials=30):
    """
    Simulates chokepoint blockages analytically and checks whether
    the resulting detour path passes near blobs.

    'Near blob' = the detour path has a node adjacent to a blob node.
    """
    if not chokepoints:
        return 0.0

    rng = np.random.RandomState(12345)

    # Get baseline shortest path
    try:
        baseline_sp = nx.shortest_path(g, source, target, weight='distance')
        baseline_cost = sum(g.edges[baseline_sp[i], baseline_sp[i+1]]['distance']
                           for i in range(len(baseline_sp) - 1))
    except nx.NetworkXNoPath:
        return 0.0

    baseline_set = set(baseline_sp)

    # Build a set of nodes adjacent to any blob
    blob_adjacent = set()
    for node in all_blob_nodes:
        if node in g.nodes():
            for neighbor in g.neighbors(node):
                if neighbor not in all_blob_nodes:
                    blob_adjacent.add(neighbor)
    blob_adjacent |= all_blob_nodes  # Include blob nodes themselves

    detours_near_blob = 0
    total_detours = 0

    for _ in range(num_trials):
        edges_to_remove = []
        rolls = rng.rand(len(chokepoints))
        for i, (u, v) in enumerate(chokepoints):
            if rolls[i] < block_prob:
                if g.has_edge(u, v):
                    edges_to_remove.append((u, v))
                elif g.has_edge(v, u):
                    edges_to_remove.append((v, u))

        if not edges_to_remove:
            continue

        blocked_g = g.copy()
        blocked_g.remove_edges_from(edges_to_remove)

        if not nx.has_path(blocked_g, source, target):
            continue

        try:
            detour_sp = nx.shortest_path(blocked_g, source, target, weight='distance')
            detour_cost = sum(blocked_g.edges[detour_sp[i], detour_sp[i+1]]['distance']
                             for i in range(len(detour_sp) - 1))
        except (nx.NetworkXNoPath, KeyError):
            continue

        # Only count meaningful detours (cost increase > 0.5)
        if detour_cost <= baseline_cost + 0.5:
            continue

        total_detours += 1

        # Check if detour nodes pass near blob
        detour_new_nodes = set(detour_sp) - baseline_set
        if detour_new_nodes & blob_adjacent:
            detours_near_blob += 1

    if total_detours == 0:
        return 0.0

    return detours_near_blob / total_detours


def _score_blob_entry_efficiency(g, blobs, all_blob_nodes):
    """
    How cheap is it to enter and exit a blob from the corridor?

    For each blob, find the minimum 'enter + traverse + exit' cost through
    the blob compared to going around. Lower overhead = higher score.

    Returns a score in [0, 1] where 1 = very efficient blob entry.
    """
    if not blobs:
        return 0.0

    efficiencies = []
    for blob in blobs:
        blob_set = set(blob)
        # Find entry points (edges connecting blob to non-blob)
        entry_points = []
        for node in blob:
            if node not in g.nodes():
                continue
            for neighbor in g.neighbors(node):
                if neighbor not in all_blob_nodes:
                    entry_points.append((neighbor, node))  # (corridor_side, blob_side)

        if len(entry_points) < 2:
            efficiencies.append(0.0)
            continue

        # Find the shortest path through the blob between different entry points
        # Compare to the shortest path around the blob
        best_efficiency = 0.0
        for i in range(len(entry_points)):
            for j in range(i + 1, min(len(entry_points), i + 4)):
                c1, b1 = entry_points[i]  # corridor1, blob1
                c2, b2 = entry_points[j]  # corridor2, blob2

                # Cost through blob: c1->b1->...->b2->c2
                try:
                    through_blob = nx.shortest_path_length(g, c1, c2, weight='distance')
                except nx.NetworkXNoPath:
                    continue

                # Cost around blob: c1->...->c2 avoiding blob
                try:
                    # Create graph without blob interior
                    no_blob_g = g.copy()
                    for node in blob_set:
                        if node in no_blob_g and node != c1 and node != c2:
                            no_blob_g.remove_node(node)
                    around_blob = nx.shortest_path_length(no_blob_g, c1, c2, weight='distance')
                except nx.NetworkXNoPath:
                    # Can't go around -> blob entry is essential
                    best_efficiency = max(best_efficiency, 1.0)
                    continue

                if around_blob > 0:
                    # Ratio of through-blob vs around-blob cost
                    # If through_blob < around_blob, efficiency > 1 (great!)
                    # If through_blob > around_blob, efficiency < 1
                    ratio = around_blob / (through_blob + 0.01)
                    eff = min(ratio, 1.5) / 1.5  # Normalize to [0, 1]
                    best_efficiency = max(best_efficiency, eff)

        efficiencies.append(best_efficiency)

    return np.mean(efficiencies) if efficiencies else 0.0


def _score_chokepoint_impact(g, source, target, chokepoints, block_prob,
                              num_trials=30):
    """
    How much do chokepoint blockages increase path cost?
    Higher coefficient of variation = more value from scouting.
    """
    if not chokepoints:
        return 0.0

    rng = np.random.RandomState(54321)

    try:
        baseline_sp = nx.shortest_path(g, source, target, weight='distance')
        baseline_cost = sum(g.edges[baseline_sp[i], baseline_sp[i+1]]['distance']
                           for i in range(len(baseline_sp) - 1))
    except nx.NetworkXNoPath:
        return 0.0

    costs = []
    for _ in range(num_trials):
        edges_to_remove = []
        rolls = rng.rand(len(chokepoints))
        for i, (u, v) in enumerate(chokepoints):
            if rolls[i] < block_prob:
                if g.has_edge(u, v):
                    edges_to_remove.append((u, v))
                elif g.has_edge(v, u):
                    edges_to_remove.append((v, u))

        if not edges_to_remove:
            costs.append(baseline_cost)
            continue

        blocked_g = g.copy()
        blocked_g.remove_edges_from(edges_to_remove)

        if not nx.has_path(blocked_g, source, target):
            continue

        try:
            sp = nx.shortest_path(blocked_g, source, target, weight='distance')
            cost = sum(blocked_g.edges[sp[i], sp[i+1]]['distance']
                      for i in range(len(sp) - 1))
            costs.append(cost)
        except (nx.NetworkXNoPath, KeyError):
            continue

    if len(costs) < 5:
        return 0.0

    costs_arr = np.array(costs)
    mean_cost = np.mean(costs_arr)
    std_cost = np.std(costs_arr)

    if mean_cost <= 0:
        return 0.0

    # CV (coefficient of variation) normalized to a 0-1 score
    cv = std_cost / mean_cost
    # CV > 0.15 is quite impactful; cap at 0.4
    return min(cv / 0.4, 1.0)


def _score_path_diversity(g, source, target, all_blob_nodes):
    """
    Are there meaningfully different routes, some going through blobs
    and some going around?

    Finds multiple diverse shortest paths and checks if some use blob nodes
    while others don't.
    """
    # Find diverse paths by iteratively penalizing used edges
    paths = []
    temp_g = g.copy()
    for _ in range(8):
        try:
            sp = nx.shortest_path(temp_g, source, target, weight='distance')
            paths.append(sp)
            for i in range(len(sp) - 1):
                u, v = sp[i], sp[i + 1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except nx.NetworkXNoPath:
            break

    if len(paths) < 2:
        return 0.0

    # Check which paths use blob nodes
    blob_paths = 0
    non_blob_paths = 0
    for path in paths:
        blob_nodes_in_path = sum(1 for n in path if n in all_blob_nodes)
        if blob_nodes_in_path > 0:
            blob_paths += 1
        else:
            non_blob_paths += 1

    # Best case: roughly equal split between blob and non-blob paths
    total = blob_paths + non_blob_paths
    if total == 0:
        return 0.0

    # Score based on having BOTH types of paths available
    minority = min(blob_paths, non_blob_paths)
    return min(minority / (total / 2), 1.0)


def prune_maps(maps, target_count=None, min_score=None, verbose=False):
    """
    Prune a list of maps using analytic scoring.

    Args:
        maps: list of map_data dicts
        target_count: keep top N maps (if specified)
        min_score: keep only maps above this score (if specified)
        verbose: print scores

    Returns:
        list of (map_data, score, components) tuples, sorted by score descending
    """
    scored = []
    for i, map_data in enumerate(maps):
        score, components = score_map(map_data, verbose=verbose)
        scored.append((map_data, score, components))
        if verbose:
            label = map_data.get('label', '?')
            seed = map_data.get('seed', '?')
            print(f"  Map {i} ({label}, seed={seed}): score={score:.3f}")

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])

    if min_score is not None:
        scored = [(m, s, c) for m, s, c in scored if s >= min_score]

    if target_count is not None:
        scored = scored[:target_count]

    return scored
