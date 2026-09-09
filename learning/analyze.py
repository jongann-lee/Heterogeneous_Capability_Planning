"""Compact analysis for full JSON files produced by :mod:`learning.test`."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def _mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"count": len(values), "mean": mean, "std": variance ** 0.5}


def _require_number(record, name, scenario_id, *, minimum=None):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"scenario {scenario_id!r} has invalid {name}")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"scenario {scenario_id!r} has invalid {name}")
    return value


def _require_integer(record, name, scenario_id, *, minimum=0):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"scenario {scenario_id!r} has invalid {name}")
    return value


def _validate_records(payload):
    if not isinstance(payload, dict):
        raise ValueError("evaluation result root must be an object")
    suite_id = payload.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("evaluation result is missing suite_id")
    records = payload.get("scenarios")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation result must contain non-empty scenarios")
    scenario_ids = set()
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"scenarios[{index}] must be an object")
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"scenarios[{index}] is missing scenario_id")
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id {scenario_id!r}")
        scenario_ids.add(scenario_id)
        if record.get("suite_id") != suite_id:
            raise ValueError(
                f"scenario {scenario_id!r} does not match suite_id {suite_id!r}")
        completed = record.get("completed")
        stalled = record.get("stalled")
        all_agents_dead = record.get("all_agents_dead")
        if not all(isinstance(value, bool)
                   for value in (completed, stalled, all_agents_dead)):
            raise ValueError(
                f"scenario {scenario_id!r} has invalid outcome flags")
        agent_count = _require_integer(
            record, "agent_count", scenario_id, minimum=1)
        target_count = _require_integer(
            record, "target_count", scenario_id, minimum=1)
        deaths = _require_integer(record, "deaths", scenario_id)
        remaining_targets = _require_integer(
            record, "remaining_targets", scenario_id)
        if deaths > agent_count:
            raise ValueError(
                f"scenario {scenario_id!r} has more deaths than agents")
        if remaining_targets > target_count:
            raise ValueError(
                f"scenario {scenario_id!r} has too many remaining targets")
        validated.append({
            "scenario_id": scenario_id,
            "agent_count": agent_count,
            "target_count": target_count,
            "makespan": _require_number(
                record, "makespan", scenario_id, minimum=0.0),
            "normalized_regret": _require_number(
                record, "normalized_regret", scenario_id),
            "completed": completed,
            "deaths": deaths,
            "remaining_targets": remaining_targets,
            "stalled": stalled,
            "all_agents_dead": all_agents_dead,
        })
    return suite_id, validated


def _group_summary(records):
    scenario_count = len(records)
    completed = [record for record in records if record["completed"]]
    deaths = sum(record["deaths"] for record in records)
    agents_deployed = sum(record["agent_count"] for record in records)
    death_scenarios = sum(record["deaths"] > 0 for record in records)
    failures = scenario_count - len(completed)
    return {
        "scenario_count": scenario_count,
        "completion_count": len(completed),
        "failure_count": failures,
        "completion_rate": len(completed) / scenario_count,
        "failure_rate": failures / scenario_count,
        "makespan": _mean_std(record["makespan"] for record in records),
        "completed_makespan": _mean_std(
            record["makespan"] for record in completed),
        "normalized_regret": _mean_std(
            record["normalized_regret"] for record in records),
        "completed_normalized_regret": _mean_std(
            record["normalized_regret"] for record in completed),
        "total_deaths": deaths,
        "mean_deaths_per_scenario": deaths / scenario_count,
        "death_scenario_count": death_scenarios,
        "death_scenario_rate": death_scenarios / scenario_count,
        "agents_deployed": agents_deployed,
        "agent_mortality_rate": deaths / agents_deployed,
        "stalled_count": sum(record["stalled"] for record in records),
        "stalled_rate": sum(record["stalled"] for record in records)
        / scenario_count,
        "all_agents_dead_count": sum(
            record["all_agents_dead"] for record in records),
        "all_agents_dead_rate": sum(
            record["all_agents_dead"] for record in records) / scenario_count,
        "mean_remaining_targets": sum(
            record["remaining_targets"] for record in records)
        / scenario_count,
    }


def analyze_evaluation(payload, source=None):
    """Aggregate one fixed-suite evaluation result by problem size."""
    suite_id, records = _validate_records(payload)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["agent_count"], record["target_count"])].append(record)
    by_size = []
    for (agent_count, target_count), items in sorted(grouped.items()):
        summary = _group_summary(items)
        summary.update({
            "agent_count": agent_count,
            "target_count": target_count,
        })
        by_size.append(summary)
    return {
        "schema_version": 1,
        "analysis_type": "fixed_evaluation_summary",
        "source_result": None if source is None else str(source),
        "suite_id": suite_id,
        "overall": _group_summary(records),
        "by_agent_and_target_count": by_size,
    }


def load_and_analyze(path):
    """Read one evaluation JSON file and return its compact analysis."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluation result not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation JSON in {path}: {error}") \
            from error
    return analyze_evaluation(payload, source=path)


def _matrix(analysis, cell, title, width=20):
    groups = {
        (item["agent_count"], item["target_count"]): item
        for item in analysis["by_agent_and_target_count"]
    }
    agent_counts = sorted({key[0] for key in groups})
    target_counts = sorted({key[1] for key in groups})
    first_width = max(6, len("agents"))
    lines = [title]
    lines.append(
        f"{'agents':>{first_width}} | "
        + " | ".join(f"targets={count:02d}".center(width)
                     for count in target_counts))
    lines.append(
        "-" * first_width + "-+-"
        + "-+-".join("-" * width for _ in target_counts))
    for agent_count in agent_counts:
        values = []
        for target_count in target_counts:
            item = groups.get((agent_count, target_count))
            values.append("—" if item is None else cell(item))
        lines.append(
            f"{agent_count:>{first_width}} | "
            + " | ".join(value.center(width) for value in values))
    return lines


def _format_stats(stats):
    if stats["mean"] is None:
        return "n/a"
    return f"{stats['mean']:.2f} ± {stats['std']:.2f}"


def format_analysis(analysis):
    """Render a compact human-readable analysis report."""
    overall = analysis["overall"]
    lines = [
        "Fixed evaluation analysis",
        f"Suite: {analysis['suite_id']}",
    ]
    if analysis.get("source_result"):
        lines.append(f"Source: {analysis['source_result']}")
    lines.extend((
        "",
        *_matrix(
            analysis,
            lambda item: _format_stats(item["makespan"]),
            "Makespan by problem size (mean ± population std; all scenarios)"),
        "",
        *_matrix(
            analysis,
            lambda item: (
                f"{item['completion_count']}/{item['scenario_count']} "
                f"({item['completion_rate']:.1%})"),
            "Completion by problem size"),
        "",
        "Overall",
        f"  Scenarios: {overall['scenario_count']}",
        f"  Completion: {overall['completion_count']}/"
        f"{overall['scenario_count']} ({overall['completion_rate']:.2%})",
        f"  Failure: {overall['failure_count']}/"
        f"{overall['scenario_count']} ({overall['failure_rate']:.2%})",
        f"  Makespan: {_format_stats(overall['makespan'])}",
        "  Completed-only makespan: "
        f"{_format_stats(overall['completed_makespan'])}",
        f"  Normalized regret: {_format_stats(overall['normalized_regret'])}",
        "  Completed-only normalized regret: "
        f"{_format_stats(overall['completed_normalized_regret'])}",
        f"  Scenarios with a death: {overall['death_scenario_count']}/"
        f"{overall['scenario_count']} ({overall['death_scenario_rate']:.2%})",
        f"  Agent deaths: {overall['total_deaths']}/"
        f"{overall['agents_deployed']} deployed "
        f"({overall['agent_mortality_rate']:.2%})",
        "  Mean deaths per scenario: "
        f"{overall['mean_deaths_per_scenario']:.3f}",
        f"  Stalled: {overall['stalled_count']}/"
        f"{overall['scenario_count']} ({overall['stalled_rate']:.2%})",
        f"  All agents dead: {overall['all_agents_dead_count']}/"
        f"{overall['scenario_count']} ({overall['all_agents_dead_rate']:.2%})",
        "  Mean remaining targets: "
        f"{overall['mean_remaining_targets']:.3f}",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a JSON result produced by learning.test")
    parser.add_argument("result", help="full evaluation result JSON")
    parser.add_argument(
        "--output",
        help="optional path for the compact aggregate JSON (must not exist)")
    args = parser.parse_args()
    try:
        analysis = load_and_analyze(args.result)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    output = None
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            parser.error(f"refusing to overwrite existing output: {output}")
    print(format_analysis(analysis))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(analysis, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
