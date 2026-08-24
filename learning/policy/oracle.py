"""Full-information makespan oracle for small parallel routing instances."""

import math

import networkx as nx


def _open_tsp_costs(distances, source, targets, allowed_mask):
    """Return the minimum source-starting, no-return tour cost per subset."""
    count = len(targets)
    costs = [math.inf] * (1 << count)
    costs[0] = 0.0
    states = {}
    for target_index, target in enumerate(targets):
        bit = 1 << target_index
        if not allowed_mask & bit:
            continue
        distance = distances[source].get(target, math.inf)
        if math.isfinite(distance):
            states[(bit, target_index)] = distance
            costs[bit] = distance
    for mask in range(1, 1 << count):
        if mask & ~allowed_mask:
            continue
        for last in range(count):
            current = states.get((mask, last))
            if current is None:
                continue
            costs[mask] = min(costs[mask], current)
            for next_index, next_target in enumerate(targets):
                bit = 1 << next_index
                if mask & bit or not allowed_mask & bit:
                    continue
                step = distances[targets[last]].get(next_target, math.inf)
                if not math.isfinite(step):
                    continue
                key = (mask | bit, next_index)
                states[key] = min(states.get(key, math.inf), current + step)
    return costs


def parallel_tsp(ground_truth, agents):
    """Return the exact metric-closure min-max open-tour makespan.

    Target types are read from ``ground_truth`` and are therefore treated as
    known from time zero. Targets may be partitioned only among compatible
    agents. Each agent starts at its current position, visits its assigned
    targets in the best order, and need not return to the source. The result is
    the minimum possible maximum route length across agents.

    Shortest-path distances form a metric closure. This deliberately ignores
    partial observability and incidental encounters while travelling through a
    target node, making the result an optimistic oracle/lower bound for the
    simulator. The subset dynamic programs are exact for that abstraction and
    are practical for the project's small target sets.
    """
    targets = sorted(
        (node for node, data in ground_truth.nodes(data=True)
         if data.get("type") == "target_unreached"), key=repr)
    if not targets:
        return 0.0
    if not agents:
        raise ValueError("parallel_tsp requires at least one agent")

    relevant = set(targets) | {agent.position for agent in agents}
    distances = {
        node: nx.single_source_dijkstra_path_length(
            ground_truth, node, weight="distance")
        for node in relevant
    }
    target_types = [int(ground_truth.nodes[target]["rps_type"])
                    for target in targets]
    compatible_masks = []
    route_costs = []
    for agent in agents:
        mask = sum(1 << index for index, target_type in enumerate(target_types)
                   if agent.can_service(target_type))
        compatible_masks.append(mask)
        route_costs.append(_open_tsp_costs(
            distances, agent.position, targets, mask))

    full_mask = (1 << len(targets)) - 1
    covered = 0
    for mask in compatible_masks:
        covered |= mask
    if covered != full_mask:
        missing = [targets[index] for index in range(len(targets))
                   if not covered & (1 << index)]
        raise ValueError(f"no compatible agent for targets: {missing}")

    # Partition the targets among agents while minimizing the slowest tour.
    assignments = {0: 0.0}
    for agent_index, compatible in enumerate(compatible_masks):
        updated = {}
        for assigned, current_makespan in assignments.items():
            available = compatible & ~assigned
            subset = available
            while True:
                route_cost = route_costs[agent_index][subset]
                if math.isfinite(route_cost):
                    combined = assigned | subset
                    makespan = max(current_makespan, route_cost)
                    updated[combined] = min(
                        updated.get(combined, math.inf), makespan)
                if subset == 0:
                    break
                subset = (subset - 1) & available
        assignments = updated
    result = assignments.get(full_mask, math.inf)
    if not math.isfinite(result):
        raise ValueError("compatible targets are unreachable in the oracle graph")
    return float(result)
