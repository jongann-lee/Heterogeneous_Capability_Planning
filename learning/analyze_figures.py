"""Plot makespan comparisons from fixed-suite evaluation JSON files."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning.analyze import analyze_evaluation


POLICY_COLORS = ("#377eb8", "#e41a1c", "#4daf4a", "#984ea3")
POLICY_NAMES = {
    "learned": "Learned",
    "fi-opt": "FI-OPT",
    "scout-then-execute": "Scout-Then-Execute",
}
TARGET_PROFILE_NAMES = {
    "balanced": "Balanced",
    "heavy": "Heavy",
    "cluster": "Cluster",
}
TARGET_PROFILE_ORDER = {name: index for index, name in enumerate(
    ("balanced", "heavy", "cluster"))}


def _load_result(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluation result not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation JSON in {path}: {error}") \
            from error
    analysis = analyze_evaluation(payload, source=path)
    policy = payload.get("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"evaluation result is missing policy: {path}")
    records = payload["scenarios"]
    for record in records:
        profile = record.get("target_profile")
        if not isinstance(profile, str) or not profile:
            raise ValueError(
                f"scenario {record['scenario_id']!r} is missing target_profile")
    return {
        "path": path,
        "policy": policy,
        "analysis": analysis,
        "records": records,
    }


def _default_labels(results):
    labels = [POLICY_NAMES.get(item["policy"], item["policy"])
              for item in results]
    duplicate_labels = {
        label for label, count in Counter(labels).items() if count > 1
    }
    labels = [
        f"{label} ({item['path'].stem})"
        if label in duplicate_labels else label
        for label, item in zip(labels, results)
    ]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "result labels are not unique; provide one --label per result")
    return labels


def _grouped_makespans(analysis):
    return {
        (item["agent_count"], item["target_count"]): item["makespan"]
        for item in analysis["by_agent_and_target_count"]
    }


def _target_profile(profile):
    """Normalize the suite's count-specific profile names for plotting."""
    if profile == "balanced":
        return "balanced"
    if profile.endswith("_heavy"):
        return "heavy"
    if profile == "multi_cluster_balanced":
        return "cluster"
    return profile


def _mean_std(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"count": len(values), "mean": mean, "std": math.sqrt(variance)}


def _profile_makespans(records):
    grouped = defaultdict(list)
    for record in records:
        profile = _target_profile(record["target_profile"])
        grouped[profile].append(record["makespan"])
    return {profile: _mean_std(values)
            for profile, values in grouped.items()}


def _profile_sort_key(profile):
    return TARGET_PROFILE_ORDER.get(profile, len(TARGET_PROFILE_ORDER)), profile


def _profile_label(profile):
    return TARGET_PROFILE_NAMES.get(
        profile, profile.replace("_", " ").title())


def create_makespan_figures(result_paths, output_dir="analysis_figures",
                            labels=None, dpi=180, overwrite=False):
    """Create one row of agent-count bar graphs for each target count."""
    if not 1 <= len(result_paths) <= len(POLICY_COLORS):
        raise ValueError("provide between one and four evaluation results")
    if labels is not None and len(labels) != len(result_paths):
        raise ValueError("provide exactly one --label per evaluation result")
    if dpi < 1:
        raise ValueError("dpi must be positive")

    results = [_load_result(path) for path in result_paths]
    suite_ids = {item["analysis"]["suite_id"] for item in results}
    if len(suite_ids) != 1:
        raise ValueError("all evaluation results must use the same suite")
    labels = list(labels) if labels is not None else _default_labels(results)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("policy labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("policy labels must be unique")

    grouped = [_grouped_makespans(item["analysis"]) for item in results]
    profile_grouped = [
        _profile_makespans(item["records"]) for item in results
    ]
    combinations = set(grouped[0])
    for item, values in zip(results[1:], grouped[1:]):
        if set(values) != combinations:
            raise ValueError(
                f"problem-size coverage does not match the first result: "
                f"{item['path']}")
    profiles = set(profile_grouped[0])
    for item, values in zip(results[1:], profile_grouped[1:]):
        if set(values) != profiles:
            raise ValueError(
                f"target-profile coverage does not match the first result: "
                f"{item['path']}")

    agent_counts = sorted({agent for agent, _target in combinations})
    target_counts = sorted({target for _agent, target in combinations})
    output_dir = Path(output_dir).expanduser().resolve()
    output_paths = [
        output_dir / f"makespan_targets_{target_count:02d}.png"
        for target_count in target_counts
    ]
    output_paths.append(output_dir / "makespan_by_target_profile.png")
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing figure(s): {paths}")

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = POLICY_COLORS[:len(results)]
    x_positions = list(range(len(results)))
    created = []
    for target_count in target_counts:
        output_path = (
            output_dir / f"makespan_targets_{target_count:02d}.png")
        figure, axes = plt.subplots(
            1, len(agent_counts),
            figsize=(4.2 * len(agent_counts), 4.8),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes[0]
        for axis, agent_count in zip(axes, agent_counts):
            stats = [values[(agent_count, target_count)] for values in grouped]
            bars = axis.bar(
                x_positions,
                [item["mean"] for item in stats],
                yerr=[item["std"] for item in stats],
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                capsize=4,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )
            axis.set_title(f"Agents = {agent_count}")
            axis.set_xticks(x_positions, labels, rotation=25, ha="right")
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            axis.set_axisbelow(True)
            for bar, item in zip(bars, stats):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + item["std"],
                    f"{item['mean']:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )
            axis.margins(y=0.12)
        axes[0].set_ylabel("Makespan")
        figure.suptitle(
            f"Makespan by policy — {target_count} targets\n"
            "Bars show mean; error bars show population standard deviation",
            fontsize=13,
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        created.append(output_path)

    profiles = sorted(profiles, key=_profile_sort_key)
    profile_output_path = output_dir / "makespan_by_target_profile.png"
    figure, axes = plt.subplots(
        1, len(profiles), figsize=(4.2 * len(profiles), 4.8),
        sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    x_positions = list(range(len(results)))
    for axis, profile in zip(axes, profiles):
        stats = [policy_stats[profile] for policy_stats in profile_grouped]
        bars = axis.bar(
            x_positions,
            [item["mean"] for item in stats],
            yerr=[item["std"] for item in stats],
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        axis.set_title(_profile_label(profile))
        axis.set_xticks(x_positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
        axis.margins(y=0.15)
        for bar, item in zip(bars, stats):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + item["std"],
                f"{item['mean']:.1f}",
                ha="center", va="bottom", fontsize=8,
            )
    axes[0].set_ylabel("Makespan")
    figure.suptitle(
        "Makespan by target profile — all agent and target counts\n"
        "Bars show mean; error bars show population standard deviation",
        fontsize=13)
    figure.savefig(profile_output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    created.append(profile_output_path)
    return created


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one makespan comparison PNG per target count from up to "
            "four learning.test result JSON files"))
    parser.add_argument(
        "results", nargs="+",
        help="evaluation result JSON files, in desired bar/color order")
    parser.add_argument(
        "--label", action="append", dest="labels",
        help="display label for the corresponding result; repeat for each file")
    parser.add_argument(
        "--output-dir", default="analysis_figures",
        help="figure directory (default: ./analysis_figures)")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing figure files")
    args = parser.parse_args()

    try:
        paths = create_makespan_figures(
            args.results,
            output_dir=args.output_dir,
            labels=args.labels,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
