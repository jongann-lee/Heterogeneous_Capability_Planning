# Learning package

This package is reserved for the learned centralized planner. It should depend
on the public simulation interface, not on rendering or real-map loading.

The intended components are:

- candidate generation and agent-candidate masks;
- shared agent and candidate encoders;
- episodic rollout collection;
- REINFORCE training and rollout baselines;
- checkpoint evaluation against policies in `planning/policies/`.

No RL framework is required or installed yet.
