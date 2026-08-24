"""Adapter from the neural joint policy to simulator ``planned_path`` values."""

import networkx as nx
import torch

from learning.policy.candidates import generate_candidates
from learning.policy.configuration import CandidateConfig, load_config
from learning.gpu_sim.observation_cpu import build_observation, candidate_path


class LearnedPolicyAdapter:
    """Callable simulator policy retaining differentiable decision traces."""

    replan_in_transit = True

    def __init__(self, model, num_target_types: int, training: bool = False,
                 candidate_config: CandidateConfig | None = None, device=None):
        self.model = model
        self.num_target_types = num_target_types
        self.training = training
        self.candidate_config = candidate_config or load_config().candidates
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
        if candidate.node is None and not candidate.region_nodes:
            return [source]
        return candidate_path(graph, source, candidate, live) or [source]

    def __call__(self, env_map, at_node_agents, **_kwargs):
        all_agents = self._all_agents or list(at_node_agents)
        transit = self._transit or [None] * len(all_agents)
        candidates = generate_candidates(env_map, self.candidate_config)
        observation = build_observation(
            env_map, all_agents, self.num_target_types,
            candidates=candidates, transit=transit, clock=self._clock,
            replan_transit=True).to(self.device)
        self.model.train(self.training)
        with torch.set_grad_enabled(self.training):
            decoded = self.model.decode(
                observation, candidates=[candidates], training=self.training)
        self.decision_log_probs.append(decoded.log_probabilities[0])
        self.decision_entropies.append(decoded.entropies[0])

        active_ids = {id(agent) for agent in at_node_agents}
        live = {n for n, d in env_map.nodes(data=True)
                if d.get("type") == "target_unreached"}
        for agent_index, agent in enumerate(all_agents):
            if id(agent) not in active_ids:
                continue
            travel = transit[agent_index]
            source = travel[1] if travel is not None else agent.position
            agent.planned_path = [source]
        for agent_index, action_index in decoded.assignments[0]:
            agent = all_agents[agent_index]
            candidate = candidates[action_index]
            if id(agent) not in active_ids:
                continue
            travel = transit[agent_index]
            source = travel[1] if travel is not None else agent.position
            agent.planned_path = self._route(
                env_map, source, candidate, live)
