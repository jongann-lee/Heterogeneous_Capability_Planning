# Centralized Learning Planner — Implementation Report

## Summary

The first implementation of the centralized neural planner described in
`learning/IMPLEMENTATION_PLAN.md` is now present in the repository. The new
learning stack covers deterministic action-candidate generation,
planner-visible tensor observations, an attention-based neural policy, an
autoregressive constrained assignment decoder, simulation integration,
episodic REINFORCE training, checkpoint evaluation, and regression tests.

The implementation deliberately follows the plan's initial scope. It uses the
existing fixed graph representation and continuous-time simulator, assumes no
new blockage-specific learning machinery, uses a fixed number of target types
per model, supports variable entity counts through padding and masks, and does
not introduce Gym, MCTS, a map GNN, recurrence, actor-critic training, or
collision avoidance.

No substantive model training or real-map benchmark was run as part of this
implementation. Only short debugging, regression, overfit, and command-line
smoke checks were performed. Full experiment execution remains with the user.

## Files Added

### Core configuration

- `learning/config.yaml`
  - Stores model, candidate-generation, REINFORCE, and smoke-training settings.
- `learning/configuration.py`
  - Loads YAML into typed immutable objects and validates architectural and
    numerical constraints.

### Candidate generation

- `learning/candidates.py`
  - Defines the `Candidate` representation.
  - Generates live-target, observation, and staging candidates
    deterministically from the planner's graph.
  - Adds `WAIT` and `CONTINUE_CURRENT_ROUTE` special candidates.
  - Deduplicates physical locations and represents overlapping roles with
    multi-role flags and associated-target sets.
  - Attaches capacities to candidates. Target and observation actions are
    exclusive by default, staging capacity is configurable, and `WAIT` and
    `CONTINUE` use unlimited global capacity while remaining constrained by
    per-agent feasibility.

### Planner observations

- `learning/observation.py`
  - Defines `PlannerObservation` with the tensor layout requested in the plan:
    agent, target, and action features; entity masks; three relation tensors;
    and the agent-action feasibility mask.
  - Builds observations exclusively from the planner-visible graph. The
    observation function does not accept a ground-truth graph, preventing
    accidental target-type leakage through its API.
  - Encodes known target types with one-hot vectors and represents all unknown
    target types with the same unknown encoding.
  - Encodes positive agent capabilities with a fixed `n`-length multi-hot
    vector and represents scouting separately.
  - Includes normalized graph positions, distances, ETAs, route reachability,
    live/known state, capability compatibility, candidate roles, visibility
    relations, staging relations, and commitment-related fields.
  - Generates feasibility rules for dead agents, moving agents, scouts,
    incompatible known targets, reachability, `WAIT`, and `CONTINUE`.
  - Provides ordinary zero-padding and boolean masks for batching variable
    numbers of agents, targets, and actions.

### Neural architecture

- `learning/encoders.py`
  - Provides small MLP entity encoders for agent, target, and action feature
    vectors.

- `learning/attention.py`
  - Uses PyTorch `TransformerDecoderLayer` rather than custom residual
    attention machinery.
  - Each block performs built-in self-attention, cross-attention, residual,
    normalization, feed-forward, and dropout processing with padding masks.
  - Implements bidirectional agent-target blocks and action-target
    contextualization without positional encoding, preserving set symmetry.

- `learning/model.py`
  - Implements the complete `CentralizedPolicy` network.
  - Contextualizes agents and targets through configurable world blocks.
  - Contextualizes actions through action self-attention, aggregated
    action-target relation embeddings, and action-to-target cross-attention.
  - Retains explicit agent-action query/key scores instead of using only an
    aggregated cross-attention output.
  - Adds an MLP score computed from agent-action relational features.
  - Produces `[B, A, C]` pointer logits and sets infeasible pairs to negative
    infinity.

### Coordinated decoder

- `learning/decoder.py`
  - Implements sequential centralized assignment over flattened
    `(agent, action)` pairs.
  - Samples from a masked categorical distribution in training mode and uses
    greedy selection in evaluation mode.
  - Removes an agent after its first assignment.
  - Decrements action capacity and removes exhausted actions.
  - Allows multiple agents to select unlimited-capacity actions such as
    `WAIT`.
  - Returns assignments, flattened pair indices, joint log probabilities, and
    accumulated entropy.
  - The joint log probability is the sum of the sequential decision log
    probabilities, as required for episodic REINFORCE.

### Simulator integration

- `learning/policy_adapter.py`
  - Adapts neural assignments to the existing simulator policy interface.
  - Builds candidates and planner observations at every replanning event.
  - Converts physical-node actions into graph routes and writes them directly
    to `Agent.planned_path`.
  - Routes around other live targets so a selected route does not create an
    unintended encounter.
  - Preserves agents currently traversing an edge and treats them as committed
    to `CONTINUE_CURRENT_ROUTE`.
  - Retains differentiable log-probability and entropy traces during training.

- `simulation/engine.py`
  - Adds a small optional runtime hook for centralized policies.
  - Before calling a policy, the engine can now provide the full agent list,
    transit records, and current continuous time through
    `set_runtime_state(...)`.
  - Existing baseline call signatures and behavior remain unchanged.
  - The policy still receives the at-node agents as its mutable routing set, so
    an in-transit agent cannot change its committed edge.

### Rollouts and REINFORCE

- `learning/rollout.py`
  - Runs complete episodes through the existing simulation engine.
  - Computes the configurable initial return:

    ```text
    -makespan
    - death_penalty * num_deaths
    - incomplete_penalty * num_remaining_targets
    ```

  - Aggregates all replanning-event log probabilities and entropies into an
    episodic trace.

- `learning/reinforce.py`
  - Implements an exponential-moving-average return baseline.
  - Implements the episodic REINFORCE loss with an entropy bonus.
  - Includes optimizer stepping and gradient clipping.

- `learning/train.py`
  - Provides a reusable `train(instance_factory, ...)` Python API.
  - Creates fresh instances per episode, collects complete rollouts, updates
    the policy, records basic history, and saves checkpoints.
  - Provides a command-line smoke-training entry point using the included
    fixed-grid instance generator.

- `learning/evaluate.py`
  - Loads model configuration and weights from a checkpoint.
  - Evaluates with greedy autoregressive decoding.
  - Provides both a reusable Python function and a fixed-grid command-line
    entry point.

- `learning/instances.py`
  - Provides a small blockage-free fixed-grid instance generator for smoke
    training and evaluation.
  - Creates planner and ground-truth graph copies, locally visible terrain,
    hidden target types, reproducible capabilities, and target-type coverage.

## Files Updated

- `learning/__init__.py`
  - Exports the primary configuration, observation, and policy interfaces.

- `learning/README.md`
  - Replaces the placeholder text with an overview of the implemented stack,
    commands for smoke training and evaluation, and the reusable training API.

- `pyproject.toml`
  - Adds an explicit `torch==2.12.0` dependency matching the prepared project
    environment.

- `uv.lock`
  - Synchronizes the lockfile with PyTorch 2.12.0 and its CUDA 13 dependency
    set.

## Planner-Visibility and Safety Rules

The implementation preserves the repository's two-graph partial-observability
contract:

1. Neural observations are built only from `env_map`, the planner's current
   view.
2. `build_observation` has no `ground_truth` parameter.
3. Unknown targets receive an unknown flag and an all-zero type one-hot vector.
4. The true target type is only available after the simulator has written a
   revealed type into the planner graph through sensing or contact.
5. The learned policy never reads the simulator's ground-truth graph when
   creating candidates, features, feasibility masks, scores, or routes.

The implementation also enforces these assignment constraints:

- dead agents receive no feasible action;
- an in-transit agent can only continue its current route;
- an at-node agent cannot select `CONTINUE`;
- a non-scout cannot select a pure observation action;
- known incompatible targets cannot be selected for service;
- unreachable physical candidates are masked;
- every agent is assigned at most once per joint decision;
- exclusive actions cannot be claimed twice; and
- `WAIT` can be assigned to multiple agents.

## Neural Data Flow

At a replanning event, the main forward path is:

```text
planner-visible env_map + full team execution state
    -> deterministic candidate generation
    -> PlannerObservation tensors and masks
    -> agent / target / action MLP encoders
    -> agent-target bidirectional attention blocks
    -> action relation aggregation and action-target attention
    -> explicit agent-action pointer logits
    -> feasibility masking
    -> sequential constrained joint decoding
    -> candidate-to-route conversion
    -> Agent.planned_path
```

During training, every sequential selection contributes a categorical log
probability. Those values are summed across assignments and replanning events
to obtain the log probability of the complete episode. The episodic return is
then compared with the EMA baseline to produce the REINFORCE advantage.

## Tests Added

`tests/test_learning.py` contains fast checks for the new stack:

1. Hidden ground-truth target types do not enter observations.
2. Variable agent counts batch and pad correctly.
3. Padding does not change valid logits.
4. Reordering agents reorders pointer-score rows.
5. Reordering actions reorders pointer-score columns.
6. Reordering targets leaves the represented physical policy state unchanged.
7. Dead agents are not assignable.
8. Moving agents can only continue.
9. Non-scout agents cannot select pure observation actions.
10. Known incompatible targets are masked.
11. Sequential decoding assigns each agent at most once.
12. Exclusive actions are not assigned twice.
13. Unlimited `WAIT` capacity supports multiple assignments.
14. Masked actions are never sampled.
15. Forward and backward passes produce finite gradients.
16. The model can overfit a tiny fixed supervised decision.
17. A complete policy-adapter episode runs through the existing simulator.

The tests use a standalone runner consistent with the existing repository test
style, so they do not require `pytest`.

## Debugging and Verification Performed

The following checks were run for the initial implementation with the
repository's `.venv` Python:

```bash
.venv/bin/python -m tests.test_learning
.venv/bin/python -m tests.test_simulation
.venv/bin/python -m compileall -q learning simulation tests
git diff --check
```

Results:

- learning checks: **7/7 passed**;
- existing simulator regression checks: **16/16 passed**;
- all new and modified Python files compiled successfully; and
- the Git whitespace/error check passed.

A one-episode training CLI smoke run was also performed:

```bash
.venv/bin/python -m learning.train \
  --episodes 1 \
  --grid-size 3 \
  --num-agents 3 \
  --num-target-types 2 \
  --checkpoint /tmp/hcp_learning_smoke.pt
```

That run completed an episode, performed a backward/optimizer step, and wrote a
checkpoint. The checkpoint loader and greedy evaluation CLI were then invoked
successfully. A model trained for only one sampled episode is not expected to
perform reliably in greedy evaluation, so that smoke evaluation was treated as
an interface/debugging check rather than a performance result.

The subsequent YAML and standard-Transformer refactor was made in
an environment where Python execution was unavailable. Run the commands above
on the GPU desktop before treating those historical results as verification of
the refactored version.

The installed PyTorch build previously reported `2.12.0+cu130`. CUDA was not available to
the isolated debugging process, so the checks ran on CPU. This verifies the CPU
execution path but does not constitute a CUDA performance or correctness run.

## How to Run

### Fast regression checks

```bash
uv run python -m tests.test_learning
uv run python -m tests.test_simulation
```

### Fixed-grid smoke training

```bash
uv run python -m learning.train \
  --config learning/config.yaml \
  --episodes 100 \
  --num-target-types 3 \
  --num-agents 4 \
  --grid-size 5 \
  --checkpoint learning_policy.pt
```

The training CLI selects CUDA automatically when PyTorch reports that it is
available. A device can be selected explicitly with `--device cpu` or
`--device cuda`.

### Greedy checkpoint evaluation

```bash
uv run python -m learning.evaluate learning_policy.pt \
  --seed 10000 \
  --grid-size 5 \
  --num-agents 4
```

### Research-instance training

For real experiments, define an instance factory that creates fresh mutable
objects for every episode:

```python
from learning.train import train


def instance_factory(episode):
    env_map = ...
    ground_truth = ...
    agents = ...
    return env_map, ground_truth, agents


model, history = train(
    instance_factory,
    num_target_types=3,
    episodes=1000,
    device="cuda",
    checkpoint="learning_policy.pt",
)
```

The factory must return fresh agents and graphs because the simulator mutates
agent state and internally evolves the planner's map during each episode.

## Current Limitations and Follow-Up Work

This implementation is an initial research foundation, not a tuned learned
planner. Important current limitations are:

- The included CLI instance is a small synthetic fixed grid. The real DEM
  benchmark has not yet been wired into a dedicated learning experiment
  configuration.
- Candidate staging uses deterministic nearest-node selection. It has not yet
  been tuned against the map geometry or compared with more sophisticated
  staging heuristics.
- Sequential decoding updates feasibility and capacity after each selection,
  but it does not rerun the full transformer after every selected pair. The
  coordination mechanism therefore conditions later selections through the
  shrinking feasible pair set rather than an additional learned claim-state
  embedding.
- `WAIT` is available and unlimited, but the existing event simulator has no
  independent future wake-up event for a team that chooses to wait while no
  agent is moving. Such a decision can correctly terminate as a stalled
  episode. A future experiment may add an explicit timed-wait event if waiting
  without concurrent transit needs to advance simulation time.
- Candidate routes currently use safe shortest paths. The learned network
  selects a destination/action candidate but does not select among the diverse
  reward-scored paths used by some hand-written planners.
- The initial return is undiscounted and uses only makespan, deaths, and
  remaining targets. Reward scale and penalty values still require
  experimental calibration.
- The trainer uses an EMA baseline rather than a learned critic.
- There is no recurrent policy memory; all state must be represented in the
  current planner-visible graph and agent execution state.
- No substantive convergence, generalization, GPU throughput, or baseline
  comparison results have been collected yet.
- The tiny overfit test verifies optimization on a fixed supervised pair
  choice. It is not evidence that episodic REINFORCE will learn the complete
  task without additional reward and curriculum tuning.

## Recommended Next Steps

1. Run the fast tests in the user's normal `uv` environment and confirm CUDA
   visibility separately.
2. Train on the fixed-grid CLI long enough to inspect completion rate, return,
   entropy, and the frequency of stalled `WAIT` assignments.
3. Add structured metric logging and periodic checkpoint evaluation rather
   than relying only on the returned Python history list.
4. Establish a tiny reproducible training curriculum before moving to the real
   DEM map.
5. Add a learned decoder-state or claim embedding if capacity masking alone is
   insufficient for coordinated specialization.
6. Add experiment-specific wrappers that compare the greedy learned policy
   with `baseline1` and `baseline2` on identical seeds and capability sets.
7. Only after the fixed-map behavior is stable, expand toward blockages, map
   encoders, learned baselines, or other extensions excluded from the initial
   scope.
