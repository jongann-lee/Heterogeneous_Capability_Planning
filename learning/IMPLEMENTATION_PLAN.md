# Centralized Learning Planner — Implementation Plan

## Objective

Implement a centralized neural policy for the heterogeneous-capability
simulation using agent, target, and candidate-action tokens with
self-attention, cross-attention, and an autoregressive assignment decoder.

The policy must operate entirely on planner-visible information. Hidden target
types and any other ground-truth state must never enter the model observation.

## Initial scope

Keep the first version deliberately constrained:

- no edge blockages;
- the current fixed map;
- known target locations and hidden target types;
- a fixed number of target types `n` per trained model;
- variable agent, target, and candidate-action counts;
- a centralized planner with instantaneous information sharing;
- the existing continuous-time simulation mechanics;
- episodic REINFORCE training; and
- pure PyTorch—the project does not require Gym.

Do not initially add MCTS, a map GNN, actor-critic training, recurrent memory,
probabilistic target priors, or collision avoidance.

## Actions

The policy's atomic decision is an `(agent, candidate)` pair.

Candidate locations come from three sources:

1. live target locations;
2. safe locations that can observe one or more unknown targets; and
3. staging locations well positioned to approach unknown targets.

Also include `WAIT` and, where appropriate, `CONTINUE_CURRENT_ROUTE`.

Candidate generation must remain deterministic and separate from the neural
network. Deduplicate candidates at the same physical node and encode their
roles using multi-hot flags instead of creating multiple equivalent actions for
one node.

## Planner-visible state

Create a `PlannerObservation` containing padded tensors and masks:

```text
agent_features:           [B, Amax, Fa]
target_features:          [B, Tmax, Ft]
action_features:          [B, Cmax, Fc]

agent_mask:               [B, Amax]
target_mask:              [B, Tmax]
action_mask:              [B, Cmax]

agent_target_relations:   [B, Amax, Tmax, Fat]
agent_action_relations:   [B, Amax, Cmax, Fac]
action_target_relations:  [B, Cmax, Tmax, Fct]

feasible_action_mask:     [B, Amax, Cmax]
```

Use ordinary padding and masks initially instead of nested tensors.

### Agent features

Include:

- current position features or a node embedding;
- alive status;
- scout capability;
- positive capabilities as an `n`-length multi-hot vector;
- idle/in-transit status;
- current destination;
- remaining traversal time; and
- accumulated cost if it affects the objective or desired workload balance.

Do not use an arbitrary agent-ID embedding. Agents with identical observable
state should be interchangeable.

### Target features

Include:

- position features;
- live/serviced status;
- known/unknown status;
- revealed type as a one-hot vector, or an all-zero vector plus the unknown
  flag;
- whether another agent is committed to it; and
- the number of living agents capable of servicing it when its type is known.

Unknown targets must all use the same belief encoding under the current
independent uniform-type assumption. Do not expose their true types.

### Action features

Include:

- position features;
- target-location flag;
- observation-location flag;
- staging-location flag;
- `WAIT` and `CONTINUE` flags;
- number of associated targets;
- number of unknown targets observable from the location; and
- static terrain or node features.

### Relational features

Agent-target relations should include:

- shortest-path distance and ETA;
- target known/unknown;
- whether the agent can service the target, if known; and
- whether the agent is already committed to the target.

Agent-action relations should include:

- shortest-path distance and ETA;
- reachability;
- whether the route crosses a live target;
- whether the agent is already committed to the action; and
- whether the action category is feasible for this agent.

Action-target relations should include:

- whether the action is located at the target;
- whether the action can observe the target;
- whether the action is a staging candidate for the target;
- candidate-to-target shortest-path distance; and
- target live/known status.

Normalize distances and times consistently.

## Neural architecture

Use three initial encoders:

```python
A = AgentEncoder(agent_features)       # [B, A, d]
T = TargetEncoder(target_features)     # [B, T, d]
C = ActionEncoder(action_features)     # [B, C, d]
```

Small MLPs with categorical embeddings are sufficient for the initial
encoders. Transformer layers provide contextual representations afterward.

### Agent-target world blocks

Apply several bidirectional blocks:

```python
A = AgentSelfAttention(A)
T = TargetSelfAttention(T)

A = AgentReadsTargets(query=A, key=T, value=T)
T = TargetsReadAgents(query=T, key=A, value=A)
```

Each operation should use multi-head attention, residual connections, layer
normalization, feed-forward layers, and padding masks.

The directions have separate meanings:

- agent self-attention represents team coordination;
- target self-attention represents spatial and capability competition among
  remaining tasks;
- agents reading targets identifies which tasks matter to each agent; and
- targets reading agents represents how difficult each task is for the current
  team, including capability rarity.

Conceptually, one block is:

\[
\widetilde A = \operatorname{SA}_A(A), \qquad
\widetilde T = \operatorname{SA}_T(T),
\]

\[
A' = \widetilde A +
\operatorname{CA}_{A\leftarrow T}(\widetilde A,\widetilde T),
\]

\[
T' = \widetilde T +
\operatorname{CA}_{T\leftarrow A}(\widetilde T,\widetilde A).
\]

### Action contextualization

Actions must understand what they accomplish before agents score them:

```python
C = ActionSelfAttention(C)
C = ActionsReadTargets(query=C, key=T, value=T)
```

Incorporate `action_target_relations` using either an additive attention bias
or an aggregated relation embedding. Relation aggregation is a reasonable
first implementation:

```python
relation_context = masked_aggregate(
    relation_mlp(action_target_relations),
    target_mask,
)
C = C + relation_context
```

## Agent-action pointer head

Produce one score for every feasible agent-action pair:

\[
z_{ic} =
\frac{(W_q A_i)^\top(W_k C_c)}{\sqrt d}
+ \operatorname{MLP}(r_{ic}).
\]

The output shape is:

```text
pair_logits: [B, A, C]
```

Apply `feasible_action_mask` by setting invalid logits to negative infinity.

Do not use only the aggregated output of ordinary cross-attention as the
policy output. Retain explicit query-key pair scores, as in a pointer-network
decoder.

## Coordinated assignment decoder

Do not independently take the maximum action in every agent row. That can send
similar agents to the same candidate because no decision is conditioned on the
other selected actions.

Implement an autoregressive centralized decoder:

1. Compute logits over every currently feasible `(agent, action)` pair.
2. Flatten the valid pair matrix.
3. During training, sample one pair from the masked softmax.
4. During evaluation, select the maximum-scoring pair.
5. Mark that agent as assigned.
6. Update the selected action's claim and remaining capacity.
7. Recompute or update logits for the remaining assignments.
8. Continue until all assignable agents have actions or `STOP` is selected.

The decoder should return something equivalent to:

```python
DecoderOutput(
    assignments=...,
    selected_pair_indices=...,
    log_probabilities=...,
    entropies=...,
)
```

The log-probability of the joint assignment is the sum of the sequential
decision log-probabilities.

Support configurable action capacities:

- target action: initially capacity 1;
- observation location: initially capacity 1;
- staging location: configurable;
- `WAIT`: unlimited; and
- `CONTINUE_CURRENT_ROUTE`: agent-specific.

These constraints can be relaxed later if intentional redundant assignments
become useful.

## Simulation integration

Add a learned-policy adapter under `learning/` that conforms to the simulation
planner interface.

At every replanning event:

1. construct the planner-visible observation;
2. generate the current candidates;
3. run the neural policy;
4. convert selected candidates into graph routes; and
5. assign `planned_path` without exposing ground truth.

The centralized observation should include all living agents, including agents
currently traversing an edge. In-transit agents must be masked to
`CONTINUE_CURRENT_ROUTE` until they reach their committed destination.

If the current simulation policy interface only receives agents that are at a
node, generalize the interface so the learned policy can observe the entire
team while preserving the rule that an in-transit agent cannot change its
current edge.

## REINFORCE trainer

Use complete episodic rollouts. For one episode:

\[
\log P(\tau) = \sum_k \log \pi_\theta(a_k\mid S_k).
\]

Use the loss:

\[
\mathcal L =
-(R-b)\log P(\tau) - \beta H,
\]

where `R` is the episodic return, `b` is initially an exponential-moving-average
baseline, `H` is an entropy bonus, and `beta` is configurable.

Do not use greedy selection during training. Sample from the masked softmax and
use greedy decoding only during evaluation.

Keep reward calculation configurable. One initial preset is:

```python
episode_return = (
    -makespan
    - death_penalty * num_deaths
    - incomplete_penalty * num_remaining_targets
)
```

The incompletion penalty must be large enough that an early failed termination
does not appear better than completing the mission. Use undiscounted episodic
return initially because decisions occur at irregular continuous-time
intervals.

## Suggested files

```text
learning/
    observation.py        # PlannerObservation and tensor batching
    candidates.py         # deterministic candidate generation
    encoders.py           # initial entity encoders
    attention.py          # self/cross-attention blocks
    model.py              # complete policy network
    decoder.py            # autoregressive pair assignment
    policy_adapter.py     # simulation integration
    rollout.py            # episode collection
    reinforce.py          # loss and optimizer logic
    train.py              # training CLI
    evaluate.py           # greedy benchmark evaluation
    config.py             # configuration dataclasses

tests/
    test_learning_observation.py
    test_learning_model.py
    test_learning_decoder.py
    test_learning_policy.py
```

## Required tests

At minimum, verify that:

- hidden ground-truth target types never enter observations;
- variable numbers of agents, targets, and candidates batch correctly;
- padding does not change valid logits;
- reordering agents reorders score rows;
- reordering actions reorders score columns;
- reordering targets does not change the represented physical state;
- invalid actions receive zero probability;
- dead agents cannot be assigned;
- in-transit agents can only continue;
- non-scout agents cannot select pure observation actions;
- known incompatible targets are masked;
- sequential decoding assigns every agent at most once;
- exclusive actions are not assigned twice;
- `WAIT` can be assigned to multiple agents;
- forward and backward passes produce finite values;
- the model can overfit a tiny fixed instance; and
- a complete learned-policy episode runs through the existing simulator.

## Implementation order

1. Define `PlannerObservation` and deterministic candidate generation.
2. Add observation-leakage, padding, and permutation tests.
3. Implement the initial encoders and attention blocks.
4. Implement the pair-scoring pointer head.
5. Implement autoregressive joint assignment.
6. Add the simulation policy adapter.
7. Add episodic rollout collection.
8. Add REINFORCE training and evaluation.
9. Verify a tiny-instance overfit before beginning large GPU experiments.

## Central design requirement

This must remain a centralized joint policy. Agent-agent attention alone does
not provide coordination if every agent subsequently performs an independent
argmax. The autoregressive pair decoder is what turns the shared representation
into a coordinated assignment.
