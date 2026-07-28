# Generalized Terrain-Aware Multi-Agent Planning — Repository Handoff

This document is the starting context for a new repository derived from
`Multi_Agent_Lock_and_Key`. It captures the research direction, mechanics worth
preserving, proposed generalized formulation, unresolved modeling decisions,
and an implementation roadmap. A new coding assistant should read this file
before modifying the copied code.

## 1. Research direction

The project studies heterogeneous agents operating on uncertain, weighted
terrain. A centralized planner must use scouting-capable agents to acquire
information and task-capable agents to service hidden target types while
balancing mission completion, elapsed time, agent losses, and travel cost.

The previous repository implemented one special case using rock–scissors–paper:

- target types were rock, scissors, and paper;
- there was one pure scout and three pure attackers;
- every attacker had exactly one winning, one drawing, and one losing matchup.

The new repository should generalize that model. Rock–scissors–paper should be
retained only as a regression scenario demonstrating that the generalized model
strictly contains the old one.

The intended contribution is not necessarily a new optimization algorithm. It
may instead be a general problem formulation, a geospatial instance-generation
pipeline, a reproducible benchmark, and an empirical study of information,
capability overlap, specialization, risk, and terrain.

## 2. General problem statement

Let the environment be a directed weighted graph

\[
G=(V,E,w),
\]

where each directed edge \(e=(u,v)\) has traversal time \(w_e>0\). The graph is
typically generated from a digital elevation model (DEM), roads, and terrain
data.

There are \(m\) agents and \(q\) targets. Target \(j\) is located at a known
node \(x_j\in V\), but has a hidden type

\[
\tau_j \in \mathcal T=\{1,\ldots,n\}.
\]

The planner maintains a belief or partial-information state rather than reading
ground truth directly.

### 2.1 Agent capabilities

Each agent \(i\) has two logically separate capability dimensions:

1. **Scouting capability**

   \[
   s_i\in\{0,1\}.
   \]

   If \(s_i=1\), the agent can reveal target types and hidden terrain state
   according to its visibility model. An agent may be both scout-capable and
   task-capable.

2. **Target-interaction capability**

   The most general representation is an outcome function

   \[
   O_i:\mathcal T\rightarrow
   \{\text{success},\text{stalemate},\text{agent-loss}\}.
   \]

   Equivalently, every agent has three sets

   \[
   W_i=\{k:O_i(k)=\text{success}\},
   \]

   \[
   D_i=\{k:O_i(k)=\text{stalemate}\},
   \]

   \[
   L_i=\{k:O_i(k)=\text{agent-loss}\},
   \]

   which partition \(\mathcal T\).

For a simpler user-facing encoding, an agent may expose a capability set

\[
C_i\subseteq\{0,1,\ldots,n\},
\]

where `0 in C_i` means scout-capable and each positive \(k\in C_i\) means the
agent can successfully service target type \(k\). Internally, scouting should
still remain a separate boolean because target type 0 does not actually exist.

The simple binary-capability model is a special case:

\[
W_i=C_i\setminus\{0\},\qquad D_i=\varnothing,\qquad
L_i=\mathcal T\setminus W_i.
\]

Whether this binary model should be the default remains an explicit design
decision; see Section 8.

### 2.2 Interaction

When agent \(i\) reaches a live target \(j\), the target type is revealed and
the configured outcome \(O_i(\tau_j)\) is applied:

- `success`: target \(j\) is eliminated/serviced and the agent survives;
- `stalemate`: target and agent remain active;
- `agent-loss`: the target remains active and the agent is removed.

The old RPS model is recovered by choosing three target types and assigning each
attacker one success, one stalemate, and one loss outcome. A pure scout has
\(s_i=1\), no successful interactions, and loses on contact with every target.

### 2.3 Partial observability

Keep the two-graph contract from the previous repository:

- **Planner graph:** optimistic/partially observed state. Target positions are
  known, target types initially are unknown, and hidden blockages have not yet
  been incorporated.
- **Ground-truth graph:** true target types and true traversable edge set.

Only observations and target interactions may transfer information from ground
truth into planner state.

A scout-capable agent at node \(v\) observes a footprint

\[
\mathcal V_i(v)\subseteq V\cup E.
\]

This can reveal:

- types of live targets in the footprint;
- traversability or blockage state of visible edges.

The initial implementation may use deterministic visibility, but the API should
not prevent later range limits, heterogeneous sensors, noisy observations, or
Bayesian beliefs.

## 3. Continuous-time mechanics to preserve

The previous simulation was converted from fixed turns to a discrete-event
simulation. Preserve this design.

- An edge with cost \(w_{uv}\) takes \(w_{uv}\) units of simulated time to
  traverse, or \(w_{uv}/v_i\) if per-agent speed \(v_i\) is introduced.
- A min-heap stores future arrival events.
- The global clock jumps directly to the next event; do not discretize
  continuous time into small fixed steps.
- Observation and interaction occur at node arrivals.
- Waiting has no movement event, but time still advances while other agents
  move. Therefore delaying an agent is reflected in mission makespan.
- An agent already traversing an edge is committed to reaching that edge's
  destination. New information may alter its onward plan, but it should not
  teleport or reverse without an explicit turnaround model.
- Simultaneous events must be processed deterministically. Ideally, process all
  arrivals sharing the same timestamp as one event batch before invoking the
  planner, so agent-index ordering does not change outcomes.

Rendering should remain decoupled from simulation. Record state-changing events
and transit intervals, then interpolate positions at a fixed rendering interval.
Frame timestamps are rendering sample times, not necessarily event times.

## 4. Objective and reported metrics

Mission completion should be treated as the primary requirement. At minimum,
report:

\[
\text{completion indicator},\qquad
T_{\mathrm{complete}},\qquad
N_{\mathrm{loss}},\qquad
\sum_i C_i,\qquad
\max_i C_i,
\]

where \(T_{\mathrm{complete}}\) is makespan and \(C_i\) is agent \(i\)'s total
traversal cost.

A scalar objective may be exposed:

\[
J=T_{\mathrm{complete}}+\lambda_{\mathrm{loss}}N_{\mathrm{loss}},
\]

but raw metrics and Pareto curves should also be retained because conclusions
can depend strongly on \(\lambda_{\mathrm{loss}}\).

Useful additional metrics include:

- target reveal time;
- target service time;
- number of unsupported engagements;
- idle time per agent;
- information acquired over time;
- regret relative to an omniscient planner;
- success after removal of one or more critical agents.

Incomplete runs must not accidentally appear better because they terminate
early. Either compare completion first lexicographically or assign an explicit,
documented incompletion penalty.

## 5. Why the generalized model is interesting

The generalized model exposes research dimensions hidden by RPS:

- **Capability overlap:** multiple agents can service the same target type.
- **Rare capabilities:** only one agent may be able to service a critical type.
- **Hybrid agents:** an agent may scout and service targets, creating a direct
  information-versus-action trade-off.
- **Generalists and specialists:** capability breadth can be traded against
  speed, sensing, or traversal cost.
- **Redundancy and resilience:** capability loss can make remaining targets
  impossible to service.
- **Joint assignment:** revealed targets require capability-constrained
  allocation and sequencing.
- **Risk under hidden types:** sending an agent to an unknown target can consume
  a capability needed later.
- **Terrain dependence:** travel time and visibility determine whether waiting
  for information is worth the delay.

## 6. Instance-generation requirements

An instance generator should explicitly control:

- number of target types \(n\);
- number of agents and targets;
- prior distribution over target types;
- number of successful agents per target type;
- average capability-set size;
- overlap between agent capabilities;
- fraction of scout-capable agents;
- fraction of hybrid scout/task agents;
- number of critical types with only one successful agent;
- frequency of stalemate and loss outcomes;
- terrain roughness, road density, obstacle uncertainty, and visibility
  fragmentation.

Unless intentionally generating impossible cases, enforce:

\[
\forall k\in\mathcal T,\quad \exists i\text{ such that }O_i(k)=\text{success}.
\]

Also verify instance-level feasibility after considering graph connectivity and
agent starting positions.

Every random instance must be reproducible from a recorded seed and serialized
configuration.

## 7. Recommended software architecture

Do not carry the old RPS names into core abstractions. Suggested modules:

```text
src/
  domain.py              # Agent, Target, capabilities, outcomes
  belief.py              # planner-visible target/edge state
  simulator.py           # discrete-event mechanics
  policy.py              # policy protocol/interface
  terrain/
    dem.py                # geospatial loading and graph construction
    traversal.py          # travel-time model
    visibility.py         # viewshed/observation model
  policies/
    nearest_capable.py
    information_greedy.py
    omniscient.py
  rendering.py
  instances.py
tests/
```

Core domain logic should not import rasterio, matplotlib, or real-map code.
Terrain loading, rendering, and policy implementations should be adapters around
the simulation kernel.

### 7.1 Suggested data model

Conceptually:

```python
class InteractionOutcome(Enum):
    SUCCESS = 1
    STALEMATE = 0
    AGENT_LOSS = -1


@dataclass
class AgentSpec:
    scout_capable: bool
    outcomes: dict[int, InteractionOutcome]
    speed: float = 1.0


@dataclass
class TargetState:
    node: Hashable
    true_type: int
    active: bool = True
```

Keep immutable agent specifications separate from mutable execution state
(position, alive, transit interval, trajectory, and assigned task).

### 7.2 Policy interface

A policy should receive only planner-visible information, never ground truth.
It should see all agents, including those in transit, so it can coordinate joint
assignments. In-transit agents remain committed to their current edge, but the
policy may update their route after the committed destination.

The policy result should preferably be explicit assignments/routes rather than
mutating agent objects in place. This makes policies easier to test and prevents
stale assignments from being hidden in mutable state.

## 8. Modeling decisions that still require owner approval

Do not silently choose these:

1. **Unsupported interaction:** Does an unsupported target always kill the
   agent, always cause stalemate, or use a configurable outcome matrix?
   Recommendation: configurable matrix, with “unsupported means loss” as a
   simple preset.
2. **Scouting encoding:** The external shorthand may use capability `0`, but
   should the public API expose `scout_capable: bool` instead?
   Recommendation: boolean in the formal model and code; support `0` only in
   import/config helpers.
3. **Observation:** Are non-scout-capable agents completely blind, or do they
   retain local navigation sensing while being unable to classify targets?
4. **Target priors:** Uniform independent types, known nonuniform priors, or
   correlated assignments?
5. **Collisions:** Can multiple agents occupy a node or traverse the same edge
   simultaneously? The old simulator did not model inter-agent collisions.
6. **Communication:** Is all observed information shared instantaneously with a
   centralized planner?
7. **Replanning:** Should all same-time arrivals be batched before one joint
   replan? Recommendation: yes.
8. **Objective:** Lexicographic completion/deaths/makespan, weighted scalar, or
   multi-objective reporting?
9. **Agent heterogeneity beyond capabilities:** speed, traversal cost,
   visibility footprint, failure probability, or energy budget?
10. **Target behavior:** static targets only, or future support for deadlines,
    service durations, and moving targets?

## 9. Minimum baseline suite

Implement at least:

1. **Random feasible assignment:** sanity floor.
2. **Nearest target:** ignores hidden type and capability risk.
3. **Nearest revealed compatible target:** waits or gambles when no compatible
   target is known.
4. **Static-vantage scout:** counterpart to the old tallest-point baseline.
5. **Information-greedy scout:** maximize expected target-type revelations per
   unit travel time.
6. **Joint capability-aware assignment:** solve a known-target assignment using
   shortest-path travel times.
7. **Omniscient oracle:** knows all target types and hidden blockages; provides a
   lower bound/reference, not a deployable policy.
8. **Old RPS baseline:** regression and continuity with the previous project.

## 10. Minimum test suite

Tests should establish:

- valid capability and outcome matrices;
- RPS is reproduced exactly as a special case;
- scout-only, attacker-only, and hybrid agents behave correctly;
- hidden target types cannot leak into policies;
- observations update planner state only when permitted;
- success, stalemate, and agent-loss interactions;
- simultaneous event batching;
- traversal time equals edge cost divided by speed;
- waiting increases makespan when it lies on the critical path;
- in-transit agents finish their committed edge after replanning;
- completion and infeasibility detection;
- deterministic behavior under a fixed seed;
- rendering interpolation does not affect simulation state.

Use `uv` for environment and command execution. Establish a real
`pyproject.toml` and lockfile in the new repository rather than depending only
on a copied `.venv`.

## 11. Terrain realism: future work, not yet solved

The copied terrain code is useful scaffolding but should not yet be described as
physically realistic. Known limitations include:

- elevation normalized separately on each map rather than preserved in meters;
- heuristic, uncalibrated slope-to-time conversion;
- a fixed road cost multiplier;
- custom height-gated visibility rather than a validated viewshed model;
- manually specified elliptical obstacles;
- synthetic target positions/types;
- limited geographic diversity.

Future realism work should preserve geospatial units and coordinate reference
systems, calibrate directed travel times, use validated viewsheds, derive
uncertainty from real terrain/land-cover data, generate many geographic
instances, and perform external validation.

## 12. Starting a new coding session

To begin work in the new repository:

1. Copy this document into the new repository.
2. Add a short `AGENTS.md` telling Codex to read this document and verify the
   current branch/files before acting.
3. Add or update `CLAUDE.md` similarly if Claude will also work in the repo.
4. Tell the assistant which decisions in Section 8 have been resolved.
5. Ask it first to inventory the copied code and propose what to retain,
   generalize, or delete. Do not immediately rename everything without a tested
   migration plan.
6. Build the generalized domain model and event simulator before adapting the
   real-map pipeline or rendering.

The minimum information needed from the owner is:

- the chosen unsupported-interaction semantics;
- whether non-scout agents are completely blind;
- the initial objective/metric ordering;
- whether collisions and communication constraints are in scope;
- whether the first milestone should use synthetic graphs or immediately adapt
  the DEM pipeline.

