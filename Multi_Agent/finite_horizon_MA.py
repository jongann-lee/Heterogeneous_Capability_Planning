import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from Single_Agent.repeated_topk import calculate_path_reward
from Graph_Generation.target_graph import stochastic_accumulated_blockage_path
from Multi_Agent.rps import ROCK, COMBAT_TYPES


UNREACHABLE_REWARD = -1e9


class Agent:
    def __init__(self, position, agent_type: int = ROCK,
                 cost_multiplier: float = 1.0, movement_modifier: int = 1):
        """
        Args:
            position: starting node.
            agent_type: one of the rps ids (SCOUT=0, ROCK=1, SCISSOR=2,
                PAPER=3). Scouts have long-range visibility but cannot engage
                targets; rock/scissor/paper can engage but are blind (sense
                nothing), learning a target's type only by stepping onto it.
                See Multi_Agent/rps.py. Defaults to ROCK so legacy callers that
                ignore types still get a valid combat agent.
            cost_multiplier: per-agent multiplier on edge cost. The graph's
                edge 'distance' gets multiplied by this when accumulated into
                total_traversal_cost (and when scoring path rewards).
            movement_modifier: integer >= 1. Number of edges this agent can
                traverse per simulation turn.
        """
        self.position = position
        self.agent_type = int(agent_type)
        self.alive = True
        self.cost_multiplier = float(cost_multiplier)
        self.movement_modifier = int(movement_modifier)
        self.total_traversal_cost = 0.0
        self.trajectory = [position]
        self.planned_path: list = []

    @property
    def can_engage(self) -> bool:
        """True for rock/scissor/paper agents; False for scouts."""
        return self.agent_type in COMBAT_TYPES

    def move(self, from_node, to_node, cost: float):
        if not self.alive:
            return
        if from_node != self.position:
            raise ValueError(
                f"Agent is at {self.position} but move() was called with from_node={from_node}"
            )
        self.position = to_node
        self.total_traversal_cost += cost * self.cost_multiplier
        self.trajectory.append(to_node)
        if self.planned_path and self.planned_path[0] == from_node:
            self.planned_path = self.planned_path[1:]


def finite_horizon_assignment(env_map: nx.Graph, agents, reward_ratio: float,
                              discount_factor: float = 1.0,
                              sample_recursion: int = 0,
                              sample_num_obstacle: int = 0,
                              sample_obstacle_hop: int = 0) -> dict:
    """
    One-step bipartite assignment of agents to targets via the Hungarian algorithm.

    For each (agent, target) pair, scores via _score_pair (which generates a
    candidate set of paths through stochastic_accumulated_blockage_path and
    returns the best). Returns the assignment that maximizes total reward.

    NOTE: Hungarian assumes per-pair rewards are independent. Our visibility
    reward is not — see sequential_greedy_assignment for a submodular-aware
    alternative. This routine is kept for comparison and for callers that
    don't care about observation overlap.

    Targets are nodes in env_map with attribute type == 'target_unreached'.
    Pairs with no path get a large negative reward so the solver still runs.

    Args:
        env_map: planner's view of the environment.
        agents: list of Agent objects.
        reward_ratio: lambda weighting visibility reward against distance cost.
        discount_factor: per-step geometric discount on the visibility reward.
        sample_recursion, sample_num_obstacle, sample_obstacle_hop: parameters
            controlling how many diverse candidate paths to generate per pair
            via stochastic_accumulated_blockage_path. Set sample_recursion=0
            to fall back to pure shortest-path scoring.

    Returns:
        dict mapping agent index (in the input list) to (target_node, path),
        where `path` is the reward-maximizing path that produced the pair's
        score. The caller should send the agent down this path rather than
        recomputing a shortest path. Empty if there are no targets.
    """
    targets = [
        n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
    ]
    if not targets:
        return {}

    n_agents = len(agents)
    n_targets = len(targets)
    reward = np.full((n_agents, n_targets), UNREACHABLE_REWARD, dtype=float)
    paths: dict = {}

    for i, agent in enumerate(agents):
        for j, target in enumerate(targets):
            r, path = _score_pair(
                env_map, agent, target, reward_ratio, discount_factor,
                sample_recursion, sample_num_obstacle, sample_obstacle_hop,
            )
            if r > -np.inf:
                reward[i, j] = r
                paths[(i, j)] = path

    row_ind, col_ind = linear_sum_assignment(reward, maximize=True)
    return {
        int(r): (targets[c], paths.get((int(r), int(c))))
        for r, c in zip(row_ind, col_ind)
    }


def _score_pair(state, agent, target, reward_ratio, discount_factor,
                sample_recursion: int = 0, sample_num_obstacle: int = 0,
                sample_obstacle_hop: int = 0):
    """Score one (agent, target) pair on the current state. Returns (reward, path).

    Mirrors RepeatedTopK.process_section: generates a candidate set of paths
    via stochastic_accumulated_blockage_path (shortest path + diverse
    alternatives produced by random obstacle insertion) and returns the
    (reward, path) of the best one. Set sample_recursion=0 to fall back to
    pure shortest-path scoring.

    Returns (-inf, None) if the source is disconnected from the target.
    Does not mutate `state` (each scoring uses state.copy()).
    """
    candidates_with_depth = stochastic_accumulated_blockage_path(
        state, agent.position, target,
        recursions=sample_recursion,
        num_obstacles_per_path=sample_num_obstacle,
        obstacle_hop=sample_obstacle_hop,
    )
    if not candidates_with_depth:
        return -np.inf, None

    best_reward = -np.inf
    best_path = None
    cm = agent.cost_multiplier
    for path, _depth in candidates_with_depth:
        reward = calculate_path_reward(path, state.copy(), reward_ratio, discount_factor)
        if cm != 1.0:
            path_distance = sum(
                state.edges[path[k], path[k + 1]]["distance"]
                for k in range(len(path) - 1)
            )
            reward -= (cm - 1.0) * path_distance
        if reward > best_reward:
            best_reward = reward
            best_path = path
    return best_reward, best_path


def sequential_greedy_assignment(env_map: nx.Graph, agents, reward_ratio: float,
                                 discount_factor: float = 1.0,
                                 sample_recursion: int = 0,
                                 sample_num_obstacle: int = 0,
                                 sample_obstacle_hop: int = 0,
                                 verbose: bool = False) -> dict:
    """
    Greedy bipartite assignment that respects observation overlap between paths.

    The Hungarian algorithm assumes per-pair rewards are independent, but the
    visibility component of our path reward is NOT — once one agent's path
    "claims" an edge by observing it, any other path that would also observe
    that edge gains nothing from it. This routine respects that submodularity:

      1. Score every remaining (agent, target) pair against the current state.
      2. Commit the single best pair.
      3. Replay that path on the shared state so the edges it observes are
         marked observed for everyone else.
      4. Repeat until no agents or no targets remain.

    Standard greedy on a matroid intersection gives a 1/2-approximation to the
    monotone submodular maximum.

    Args:
        env_map: planner's view of the environment.
        agents: list of Agent objects.
        reward_ratio: lambda weighting visibility reward against distance cost.
        discount_factor: per-step geometric discount on the visibility reward.
        verbose: if True, print per-round reward matrix and the selected pair.

    Returns {agent_idx: (target_node, path)}, where `path` is the
    reward-maximizing path that produced the agent's score. The caller should
    send the agent down this path rather than recomputing a shortest path,
    since for a reward-driven policy the best path is generally not the
    shortest one. Same tuple format as finite_horizon_assignment.
    """
    targets = [
        n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
    ]
    if not targets:
        return {}

    state = env_map.copy()
    remaining_agents = list(range(len(agents)))
    remaining_targets = list(targets)
    assignment: dict = {}

    # Lazy import so non-verbose users don't drag in matplotlib.
    agent_colors = None
    target_id = None
    if verbose:
        from Multi_Agent.simulation_utils import DEFAULT_AGENT_COLORS
        agent_colors = DEFAULT_AGENT_COLORS
        target_id = {t: j for j, t in enumerate(targets)}

        def color_of(ai):
            return agent_colors[ai] if ai < len(agent_colors) else f"agent{ai}"

        print("=" * 60)
        print("Sequential greedy reassignment")
        print("Agents:")
        for ai in range(len(agents)):
            print(f"  Agent {ai} ({color_of(ai)}) @ {agents[ai].position}")
        print("Targets:")
        for j, t in enumerate(targets):
            print(f"  Target {j} @ {t}")

    round_idx = 0
    while remaining_agents and remaining_targets:
        round_idx += 1
        best_reward = -np.inf
        best_pair = None
        best_path = None

        if verbose:
            print(f"\nRound {round_idx}:")

        for ai in remaining_agents:
            for target in remaining_targets:
                reward, path = _score_pair(
                    state, agents[ai], target, reward_ratio, discount_factor,
                    sample_recursion, sample_num_obstacle, sample_obstacle_hop,
                )
                if verbose:
                    label = f"  Agent {ai} ({color_of(ai)}) -> Target {target_id[target]} @ {target}: "
                    if path is None:
                        print(label + "no path")
                    else:
                        print(label + f"reward = {reward:.4f}")
                if reward > best_reward:
                    best_reward = reward
                    best_pair = (ai, target)
                    best_path = path

        if best_pair is None or best_reward <= -np.inf:
            if verbose:
                print("  (no feasible pair this round; stopping)")
            break

        ai, target = best_pair
        assignment[ai] = (target, best_path)
        remaining_agents.remove(ai)
        remaining_targets.remove(target)

        if verbose:
            print(f"  >> Selected: Agent {ai} ({color_of(ai)}) -> "
                  f"Target {target_id[target]} @ {target} (reward = {best_reward:.4f})")

        # Commit: replay the chosen path on the shared state so the edges it
        # observes propagate to remaining candidates.
        calculate_path_reward(best_path, state, reward_ratio, discount_factor)

    if verbose:
        print("=" * 60)
    return assignment
