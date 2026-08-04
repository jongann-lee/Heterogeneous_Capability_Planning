"""Adapter from the neural joint policy to simulator ``planned_path`` values."""

import networkx as nx
import torch

from learning.candidates import generate_candidates
from learning.observation import build_observation


class LearnedPolicyAdapter:
    """Callable simulator policy retaining differentiable decision traces."""

    def __init__(self, model, num_target_types: int, training: bool = False,
                 staging_per_target: int = 2, device=None):
        self.model = model
        self.num_target_types = num_target_types
        self.training = training
        self.staging_per_target = staging_per_target
        self.device = device or next(model.parameters()).device
        self.decision_log_probs = []
        self.decision_entropies = []
        self._all_agents = None
        self._transit = None
        self._clock = 0.0

    def set_runtime_state(self, agents, transit, clock):
        """Receive the full team state from the simulation engine."""
        self._all_agents = list(agents)
        self._transit = list(transit)
        self._clock = float(clock)

    def reset_trace(self):
        self.decision_log_probs.clear()
        self.decision_entropies.clear()

    @staticmethod
    def _route(graph, source, candidate, live):
        if candidate.node is None:
            return [source]
        avoid = set(live)
        if candidate.is_target:
            avoid.discard(candidate.node)
        avoid.discard(source)
        view = graph.copy() if avoid else graph
        if avoid:
            view.remove_nodes_from(avoid)
        try:
            return nx.shortest_path(view, source, candidate.node,
                                    weight="distance")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [source]

    def __call__(self, env_map, at_node_agents, **_kwargs):
        all_agents = self._all_agents or list(at_node_agents)
        transit = self._transit or [None] * len(all_agents)
        candidates = generate_candidates(
            env_map, staging_per_target=self.staging_per_target)
        observation = build_observation(
            env_map, all_agents, self.num_target_types,
            candidates=candidates, transit=transit, clock=self._clock).to(self.device)
        self.model.train(self.training)
        with torch.set_grad_enabled(self.training):
            decoded = self.model.decode(
                observation, candidates=[candidates], training=self.training)
        self.decision_log_probs.append(decoded.log_probabilities[0])
        self.decision_entropies.append(decoded.entropies[0])

        at_node_ids = {id(agent) for agent in at_node_agents}
        live = {n for n, d in env_map.nodes(data=True)
                if d.get("type") == "target_unreached"}
        for agent in at_node_agents:
            agent.planned_path = [agent.position]
        for agent_index, action_index in decoded.assignments[0]:
            agent = all_agents[agent_index]
            candidate = candidates[action_index]
            if id(agent) not in at_node_ids or candidate.is_continue:
                continue
            agent.planned_path = self._route(
                env_map, agent.position, candidate, live)

