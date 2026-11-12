"""Print two regret tables for one policy's full evaluation JSON.

Run from the repository root, for example::

    uv run python figure_scripts/04_regret_table.py \
        outputs/evaluation/WV/WV-1103-position-best.json

Add --baseline PATH to divide by another policy's per-scenario makespan.
PATH may be a result JSON, or a directory containing exactly one matching
Scout Then Engage evaluation (searched recursively).

The first table groups by agent and task counts; the second groups by agent
and task profiles. Regret matches figure 02: makespan / FI-OPT makespan, with
Oracle at 1.0. Entries show the mean, population standard deviation, and
number of evaluated scenarios. Every scenario receives equal weight.
Only stdout is written; evaluation files are read without modification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.analyze import analyze_evaluation
from figure_scripts.policy_labels import OPT_BY_PART_LABEL


AGENT_TYPES = ("specialist", "balanced hybrid", "scout redundant")
TASK_TYPES = ("balanced", "one type heavy", "clustered")
AGENT_PROFILES = {
    "specialist_minimal": "specialist",
    "specialists": "specialist",
    "specialists_with_generalist": "specialist",
    "specialists_with_redundancy": "specialist",
    "balanced_hybrid": "balanced hybrid",
    "scout_redundant": "scout redundant",
}
POLICY_NAMES = {
    "fi-opt": "Oracle",
    "scout-then-execute": "Scout Then Engage",
    "learned": "CAAT Graph + HGT",
    "opt-by-part": OPT_BY_PART_LABEL,
}


def task_type(profile):
    if profile == "balanced":
        return "balanced"
    if profile == "multi_cluster_balanced":
        return "clustered"
    if isinstance(profile, str) and profile.startswith(("type_", "types_")) \
            and profile.endswith("_heavy"):
        return "one type heavy"
    raise ValueError(f"unsupported task profile {profile!r}")


def matched_baseline_records(payload, baseline):
    """Match each evaluated scenario to the same instance in the baseline."""
    analyze_evaluation(baseline)
    if payload["suite_id"] != baseline["suite_id"]:
        raise ValueError("baseline uses a different evaluation suite")
    map_hash = payload.get("prepared_map", {}).get("sha256")
    baseline_hash = baseline.get("prepared_map", {}).get("sha256")
    if not map_hash or map_hash != baseline_hash:
        raise ValueError("baseline uses a different or unidentified prepared map")
    indexed = {(row["scenario_id"], row.get("episode_index", 0)): row
               for row in baseline["scenarios"]}
    matched = {}
    for row in payload["scenarios"]:
        key = row["scenario_id"], row.get("episode_index", 0)
        reference = indexed.get(key)
        if reference is None:
            raise ValueError(f"{key}: missing from baseline")
        if not isinstance(row.get("instance"), dict) or not row["instance"] \
                or row["instance"] != reference.get("instance"):
            raise ValueError(f"{key}: baseline uses a different or unidentified instance")
        for field in ("seed", "agent_count", "target_count"):
            if row.get(field) != reference.get(field):
                raise ValueError(f"{key}: baseline has different {field}")
        if not reference["completed"] or reference["remaining_targets"]:
            raise ValueError(f"{key}: baseline run is incomplete")
        if reference["makespan"] <= 0:
            raise ValueError(f"{key}: baseline makespan must be positive")
        matched[key] = reference
    return matched


def load_baseline(path, payload):
    """Resolve an explicit baseline or the unique matching scout result."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        baseline = json.loads(path.read_text(encoding="utf-8"))
        matched_baseline_records(payload, baseline)
        return baseline, path
    matches = []
    rejected = []
    for candidate in sorted(path.rglob("*.json")):
        try:
            baseline = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(baseline, dict) \
                or baseline.get("policy") != "scout-then-execute" \
                or baseline.get("suite_id") != payload["suite_id"]:
            continue
        try:
            matched_baseline_records(payload, baseline)
        except ValueError as error:
            rejected.append(f"{candidate}: {error}")
        else:
            matches.append((baseline, candidate))
    if not matches:
        details = "\n" + "\n".join(rejected) if rejected else ""
        raise ValueError(f"no matching Scout Then Engage evaluation in {path}{details}")
    if len(matches) > 1:
        choices = "\n".join(str(candidate) for _, candidate in matches)
        raise ValueError(
            "multiple matching Scout Then Engage evaluations; "
            f"pass a specific JSON to --baseline:\n{choices}")
    return matches[0]


def group_regrets(payload, baseline=None):
    """Validate the evaluation and group per-scenario ratios both ways."""
    analyze_evaluation(payload)
    matched = None if baseline is None else matched_baseline_records(payload, baseline)
    by_count = defaultdict(list)
    by_type = defaultdict(list)
    for record in payload["scenarios"]:
        identity = record["scenario_id"]
        oracle = record.get("fi_opt_makespan", record.get("oracle_makespan"))
        if (isinstance(oracle, bool) or not isinstance(oracle, (int, float))
                or not math.isfinite(oracle) or oracle <= 0):
            raise ValueError(f"{identity}: invalid FI-OPT makespan")
        if not record["completed"] or record["remaining_targets"]:
            raise ValueError(
                f"{identity}: incomplete run; makespan-only regret would "
                "misrepresent this outcome")
        ratio = record["makespan"] / oracle
        if not math.isfinite(ratio) or not math.isclose(
                ratio - 1, record["normalized_regret"],
                rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{identity}: inconsistent regret fields")
        if matched is not None:
            reference = matched[identity, record.get("episode_index", 0)]
            ratio = record["makespan"] / reference["makespan"]
            if not math.isfinite(ratio):
                raise ValueError(f"{identity}: non-finite baseline ratio")
        agent_profile = record.get("agent_profile")
        if agent_profile not in AGENT_PROFILES:
            raise ValueError(f"{identity}: unsupported agent profile {agent_profile!r}")
        agent_type = AGENT_PROFILES[agent_profile]
        target_type = task_type(record.get("target_profile"))
        by_count[record["agent_count"], record["target_count"]].append(ratio)
        by_type[agent_type, target_type].append(ratio)
    return by_count, by_type


def format_stats(values):
    if not values:
        return "—"
    return (f"{statistics.fmean(values):.3f} ± {statistics.pstdev(values):.3f} "
            f"(n={len(values)})")


def format_table(headers, rows):
    widths = [max(len(str(row[index])) for row in [headers, *rows])
              for index in range(len(headers))]

    def line(row):
        return " | ".join(str(value).ljust(width)
                          for value, width in zip(row, widths))

    return "\n".join([
        line(headers), "-+-".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ])


def format_report(payload, source, baseline=None, baseline_source=None):
    by_count, by_type = group_regrets(payload, baseline)
    agent_counts = sorted({key[0] for key in by_count})
    task_counts = sorted({key[1] for key in by_count})
    size_rows = [
        [count, *(format_stats(by_count.get((count, tasks))) for tasks in task_counts)]
        for count in agent_counts
    ]
    type_rows = [
        [agents, tasks, format_stats(by_type.get((agents, tasks)))]
        for agents in AGENT_TYPES for tasks in TASK_TYPES
    ]
    all_ratios = [ratio for ratios in by_count.values() for ratio in ratios]
    policy = payload.get("policy", "unspecified")
    baseline_policy = "fi-opt" if baseline is None else baseline.get("policy", "unspecified")
    baseline_name = POLICY_NAMES.get(baseline_policy, baseline_policy)
    lines = [
        f"Policy: {POLICY_NAMES.get(policy, policy)}",
        f"Suite: {payload['suite_id']}",
        f"Source: {source}",
        f"Baseline: {baseline_name}"
        + (f" ({baseline_source})" if baseline_source is not None else ""),
        f"Regret = makespan / {baseline_name}; mean ± population std (n=scenarios)",
        "Ratios: 1.0 = baseline; below 1.0 = faster; above 1.0 = slower.",
        f"Overall: {format_stats(all_ratios)}",
        "", "Regret by number of agents and tasks",
        format_table(["Agents", *(f"Tasks={count}" for count in task_counts)], size_rows),
        "", "Regret by agent type and task type",
        format_table(["Agent type", "Task type", "Regret"], type_rows),
    ]
    if any(record["target_profile"].startswith("types_")
           for record in payload["scenarios"]):
        lines.extend([
            "", "Note: the 'one type heavy' category includes this suite's "
            "multiple-heavy-type profile (e.g. KTH types_1_2_heavy).",
        ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="full policy evaluation JSON")
    parser.add_argument(
        "--baseline", type=Path,
        help="baseline result JSON, or directory with matching Scout Then Engage results "
             "(default: Oracle makespan stored in result)",
    )
    args = parser.parse_args(argv)
    path = args.result.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        analyze_evaluation(payload, source=path)
        baseline, baseline_source = (None, None) if args.baseline is None \
            else load_baseline(args.baseline, payload)
        report = format_report(payload, path, baseline, baseline_source)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report)


if __name__ == "__main__":
    main()
