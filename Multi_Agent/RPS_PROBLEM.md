# Rock–Scissor–Paper Multi-Agent Problem — Briefing

A self-contained statement of the problem, meant to be handed to a fresh
assistant so it can help design **policies** (agent-to-target assignment +
routing). You do **not** need the codebase to reason about solutions here; the
"Interface" and "How to run" sections at the end are only relevant if you will
write or execute code.

This is a research extension of the paper *"Navigating Uncertain Environments
with Heterogeneous Visibility"* — a single planner controls several agents that
must visit/clear targets on a weighted graph while learning the environment by
observation. The rock-scissor-paper combat model below is the new part.

---

## 1. Environment

- A **weighted graph** — in practice a real terrain (DEM) grid, ~64×64,
  4-connected. Each edge has a **traversal cost** (`distance`) derived from
  terrain (up/down-hill penalty, roads cheaper). Agents move along edges one
  step per turn and accumulate cost.
- A fixed set of **target nodes** (their *positions* are known to everyone) and
  one **source** node where every agent starts.

## 2. Partial observability — two graphs

The planner never sees ground truth directly. There are two graphs:

- **Planner view** (`env_map`): optimistic. It assumes no blockages, and every
  target's *type* is **UNKNOWN**.
- **Ground truth**: the real edge set (some edges blocked/removed) and the
  **true type** of every target.

The planner only closes the gap by **observing**, and it must replan when
reality contradicts its plan.

## 3. Agents (two roles)

| Role | Visibility | Can engage? | On a live target |
|------|-----------|-------------|------------------|
| **Scout** (type id 0) | **Long-range**, terrain/height-limited (sees far; a high vantage sees over low terrain) | **No** | **Dies** (can't fight) |
| **Attacker** = rock(1) / scissor(2) / paper(3) | **Blind** — senses *nothing* | **Yes** | Combat (see below) |

Consequences:

- **Only scouts observe.** They reveal (a) the *types* of targets within sight
  and (b) terrain blockages. Attackers contribute *no* information by looking.
- An **attacker learns a target's type only by stepping onto it** (combat
  always reveals the type — but stepping on may kill the attacker).
- Default roster: **one of each** — scout, rock, scissor, paper.

## 4. Targets & combat (the RPS rule)

Every target has a hidden type ∈ {rock, scissor, paper}. When an **attacker**
steps onto a **live** target:

- **same type** → **draw**: nothing changes (target and attacker both stay).
- **attacker beats target** → **win**: the target is **eliminated**, attacker
  survives.
- **target beats attacker** → **loss**: the **attacker dies**, target remains.

Cycle (each id beats the next, wrapping): **rock ▸ scissor ▸ paper ▸ rock**,
i.e. `a beats b  ⇔  b == (a % 3) + 1`.

A **scout** stepping onto any live target simply **dies**.

Type ids used throughout: `scout=0, rock=1, scissor=2, paper=3`, and
`UNKNOWN=-1` for a target the planner hasn't revealed yet.

## 5. Objective

- **Primary:** clear (eliminate) every target.
- **Secondary:** minimize attacker/scout **deaths**, total (and max per-agent)
  **traversal cost**, and **turns**.
- A run can end **incomplete**: e.g. the only attacker that beats a surviving
  target is dead or can't reach it, or a target stays UNKNOWN and no correct-
  type attacker ever engages it.

**The central tension.** Attackers are blind gamblers. Committing an attacker
to an unrevealed target is a bet: it might win, draw, or *die*. The scout is
the only safe source of information, but it can't be everywhere, and some
targets may be occluded from every vantage. So a good policy is fundamentally
about **managing information vs. risk**: how much scouting to buy before
committing attackers, and which gambles are worth taking.

## 6. Current baseline (this is what a better policy should beat)

**"baseline-1"** — simple, uncoordinated, deliberately weak:

- **Attacker:** rank live targets by the outcome category **win > draw >
  unknown > lose**; pick the closest target in the best non-empty category;
  then **engage it literally** — walk all the way on. So a blind attacker that
  picks an *unknown* target which turns out to beat it **dies** (intended
  weakness). Routes detour around *other* live targets so it doesn't die on a
  target it didn't choose.
- **Scout:** go to the single **tallest node** on the map (best static vantage,
  since visibility is height-gated) and observe from there.
- No coordination between attackers (each chooses independently); no use of the
  base problem's visibility/observation reward.

### Where it's weak (good discussion seeds)

- The scout optimizes **vantage height**, not **information** — it doesn't tour
  to maximize the number/priority of targets revealed, and it ignores which
  targets are still unknown.
- Attackers **gamble greedily** on the nearest unknown with no notion of "is
  this bet worth it?" or "should I wait for the scout?".
- **No joint assignment**: which attacker type should take which target, and in
  what order, to avoid deadlock and wasted deaths, is left to chance.
- **Occluded targets** (never seen by the scout) are handled only by attackers
  blundering into them.
- The base problem also rewards **observing high-traffic edges** (reducing
  blockage uncertainty); baseline-1 ignores this entirely.

## 7. Open questions to discuss

- **Scouting policy:** route the scout to maximize *type reveals* over time
  (an information-greedy / vantage-tour objective) rather than one tallest spot.
- **Commit vs. wait:** when is gambling a blind attacker on an unknown better
  than paying for more scouting first? Can you quantify the expected cost of a
  gamble (⅓ win / ⅓ draw / ⅓ die, absent priors) vs. the delay of waiting?
- **Joint assignment & sequencing:** match attacker types to (revealed) targets
  and order engagements to guarantee completion and minimize deaths.
- **Risk-aware routing:** if a target must be gambled on, which attacker should
  take the risk (e.g. the one whose loss is least costly to the remaining
  plan)?
- **Metrics:** define the scoring you actually care about (completion rate,
  deaths, total/makespan cost, turns) and how to evaluate across map seeds.

---

## Interface & how to run (only if the other assistant will write/run code)

- A **policy** is a callable invoked on every replan; it sets each *living*
  agent's `planned_path` on the planner graph. Replans are triggered by: a
  newly **revealed** target type, a discovered **blockage** on some agent's
  path, or any **combat** outcome.
- State it can read: each agent's `agent_type`, `alive`, `position`; each target
  node's `type` (`target_unreached` / `target_reached`) and `rps_type`
  (`-1`/UNKNOWN until revealed).
- Reference implementation lives in `Multi_Agent/`:
  `rps.py` (rules), `rps_simulation.py` (the loop + a placeholder policy),
  `baseline1_all_key.py` (baseline-1), `rps_real_map.py` (real-DEM runner),
  `test_rps.py` (fast synthetic tests). Run tests with
  `uv run python Multi_Agent/test_rps.py`; run on the map with
  `uv run python Multi_Agent/rps_real_map.py --policy baseline1 --render`.
