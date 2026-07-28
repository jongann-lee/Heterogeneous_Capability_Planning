"""
Statistical analysis of benchmark results.

Reads the CSV output from run_benchmark.py and produces:
1. Summary statistics tables (printed and saved)
2. Per-category breakdowns
3. Statistical significance tests (paired t-test, Wilcoxon)
4. Formatted LaTeX table output for the paper
"""

import sys
import os
import csv
import json
import argparse

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_results(csv_path):
    """Loads results from the benchmark CSV file."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def categorize_results(results):
    """Groups results by their label prefix (narrow, medium, dense, etc.)."""
    categories = {}
    for row in results:
        label = row['label']
        # Extract category from label (e.g., "narrow_12x12" -> "narrow")
        category = label.split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(row)
    return categories


def compute_category_stats(rows):
    """Computes aggregate statistics for a category of maps."""
    sp_means = [float(r['sp_mean']) for r in rows]
    our_means = [float(r['our_mean']) for r in rows]
    improvements = [float(r['improvement_pct']) for r in rows]
    win_rates = [float(r['our_wins_pct']) for r in rows]
    valid_runs = [int(r['valid_runs']) for r in rows]

    return {
        'count': len(rows),
        'total_runs': sum(valid_runs),
        'sp_mean': np.mean(sp_means),
        'sp_std': np.std(sp_means),
        'our_mean': np.mean(our_means),
        'our_std': np.std(our_means),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'improvement_median': np.median(improvements),
        'win_rate_mean': np.mean(win_rates),
        'win_rate_std': np.std(win_rates),
        'maps_with_improvement': sum(1 for x in improvements if x > 0),
    }


def compute_pooled_agent_stats(rows):
    """
    Computes pooled mean, variance, and std dev for each agent across all runs.

    Uses the law of total variance to reconstruct exact pooled statistics
    from per-map mean, std, and run count:
        pooled_mean = sum(n_i * mean_i) / N
        pooled_var  = [sum(n_i * (std_i^2 + mean_i^2)) / N] - pooled_mean^2
    """
    n_list = np.array([int(r['valid_runs']) for r in rows])
    sp_mean_list = np.array([float(r['sp_mean']) for r in rows])
    sp_std_list = np.array([float(r['sp_std']) for r in rows])
    our_mean_list = np.array([float(r['our_mean']) for r in rows])
    our_std_list = np.array([float(r['our_std']) for r in rows])

    N = np.sum(n_list)
    if N == 0:
        return None

    # SP agent pooled stats
    sp_pooled_mean = np.sum(n_list * sp_mean_list) / N
    sp_pooled_var = np.sum(n_list * (sp_std_list**2 + sp_mean_list**2)) / N - sp_pooled_mean**2
    sp_pooled_std = np.sqrt(sp_pooled_var)

    # Our agent pooled stats
    our_pooled_mean = np.sum(n_list * our_mean_list) / N
    our_pooled_var = np.sum(n_list * (our_std_list**2 + our_mean_list**2)) / N - our_pooled_mean**2
    our_pooled_std = np.sqrt(our_pooled_var)

    return {
        'total_runs': int(N),
        'sp_mean': float(sp_pooled_mean),
        'sp_var': float(sp_pooled_var),
        'sp_std': float(sp_pooled_std),
        'our_mean': float(our_pooled_mean),
        'our_var': float(our_pooled_var),
        'our_std': float(our_pooled_std),
    }


def print_agent_stats_table(results, categories):
    """
    Prints per-agent mean, variance, and std dev for each category and overall.
    These are pooled across all runs (not per-map averages).
    """
    print("\n" + "=" * 90)
    print("AGENT COST STATISTICS (pooled across all runs)")
    print("=" * 90)
    print(f"{'Category':<18} {'Agent':<7} {'Mean':>10} {'Variance':>12} {'Std Dev':>10} {'Runs':>7}")
    print("-" * 90)

    for category, rows in sorted(categories.items()):
        stats = compute_pooled_agent_stats(rows)
        if stats is None:
            continue
        print(f"{category:<18} {'SP':<7} {stats['sp_mean']:>10.2f} "
              f"{stats['sp_var']:>12.2f} {stats['sp_std']:>10.2f} {stats['total_runs']:>7}")
        print(f"{'':<18} {'Ours':<7} {stats['our_mean']:>10.2f} "
              f"{stats['our_var']:>12.2f} {stats['our_std']:>10.2f} {stats['total_runs']:>7}")
        print(f"{'-'*90}")

    # Overall
    overall = compute_pooled_agent_stats(results)
    if overall:
        print(f"{'OVERALL':<18} {'SP':<7} {overall['sp_mean']:>10.2f} "
              f"{overall['sp_var']:>12.2f} {overall['sp_std']:>10.2f} {overall['total_runs']:>7}")
        print(f"{'':<18} {'Ours':<7} {overall['our_mean']:>10.2f} "
              f"{overall['our_var']:>12.2f} {overall['our_std']:>10.2f} {overall['total_runs']:>7}")

    print("=" * 90)

    return overall


def print_summary_table(results):
    """Prints a summary table of all results."""
    print("\n" + "=" * 100)
    print(f"{'Map ID':<8} {'Label':<20} {'Grid':<6} {'BP':<5} "
          f"{'SP Mean':>9} {'Our Mean':>9} {'Impr%':>7} {'Win%':>6} {'Runs':>5}")
    print("-" * 100)

    for row in results:
        print(f"{row['map_id']:<8} {row['label']:<20} "
              f"{row['grid_size']:<6} {row['block_prob']:<5} "
              f"{float(row['sp_mean']):>9.2f} {float(row['our_mean']):>9.2f} "
              f"{float(row['improvement_pct']):>7.2f} "
              f"{float(row['our_wins_pct']):>6.1f} {row['valid_runs']:>5}")

    print("=" * 100)


def print_category_table(categories):
    """Prints a category-level summary table."""
    print("\n" + "=" * 100)
    print("CATEGORY SUMMARY")
    print("=" * 100)
    print(f"{'Category':<15} {'Maps':>5} {'Runs':>7} "
          f"{'SP Mean':>9} {'Our Mean':>9} "
          f"{'Impr% Mean':>11} {'Impr% Med':>10} "
          f"{'Win% Mean':>10} {'Maps+':>6}")
    print("-" * 100)

    for category, rows in sorted(categories.items()):
        stats = compute_category_stats(rows)
        print(f"{category:<15} {stats['count']:>5} {stats['total_runs']:>7} "
              f"{stats['sp_mean']:>9.2f} {stats['our_mean']:>9.2f} "
              f"{stats['improvement_mean']:>11.2f} {stats['improvement_median']:>10.2f} "
              f"{stats['win_rate_mean']:>10.1f} "
              f"{stats['maps_with_improvement']:>3}/{stats['count']}")

    print("=" * 100)


def run_statistical_tests(results):
    """
    Runs paired statistical tests on the per-map mean costs.
    Tests whether RepeatedTopK is statistically significantly better.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\nscipy not available, skipping statistical tests.")
        print("Install with: pip install scipy")
        return None

    sp_means = np.array([float(r['sp_mean']) for r in results])
    our_means = np.array([float(r['our_mean']) for r in results])
    differences = sp_means - our_means  # Positive = our agent is better

    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    # Paired t-test (assumes normally distributed differences)
    t_stat, p_value_t = scipy_stats.ttest_rel(sp_means, our_means)
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value_t:.6f}")
    print(f"  Significant at alpha=0.05: {'Yes' if p_value_t < 0.05 else 'No'}")
    print(f"  Significant at alpha=0.01: {'Yes' if p_value_t < 0.01 else 'No'}")

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, p_value_w = scipy_stats.wilcoxon(differences, alternative='greater')
        print(f"\nWilcoxon signed-rank test (one-sided, H1: our < SP):")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value:     {p_value_w:.6f}")
        print(f"  Significant at alpha=0.05: {'Yes' if p_value_w < 0.05 else 'No'}")
        print(f"  Significant at alpha=0.01: {'Yes' if p_value_w < 0.01 else 'No'}")
    except ValueError as e:
        print(f"\nWilcoxon test not applicable: {e}")
        p_value_w = None
        w_stat = None

    # Effect size (Cohen's d)
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        effect_label = "large"
    elif abs(cohens_d) >= 0.5:
        effect_label = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_label = "small"
    else:
        effect_label = "negligible"
    print(f"  Interpretation: {effect_label}")

    # 95% confidence interval for the mean improvement
    n = len(differences)
    se = diff_std / np.sqrt(n)
    ci_lower = diff_mean - 1.96 * se
    ci_upper = diff_mean + 1.96 * se
    print(f"\n95% CI for mean cost reduction: [{ci_lower:.4f}, {ci_upper:.4f}]")

    print("=" * 60)

    return {
        't_stat': float(t_stat),
        'p_value_ttest': float(p_value_t),
        'w_stat': float(w_stat) if w_stat is not None else None,
        'p_value_wilcoxon': float(p_value_w) if p_value_w is not None else None,
        'cohens_d': float(cohens_d),
        'effect_label': effect_label,
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
    }


def generate_latex_table(results, categories):
    """Generates a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results on automatically generated maps. "
                 r"Improvement shows the percentage reduction in expected travel cost "
                 r"of RepeatedTopK over shortest-path replanning. "
                 r"Mean, Var, and Std are pooled across all runs.}")
    lines.append(r"\label{tab:auto_benchmark}")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Category & Maps & Runs & \multicolumn{3}{c}{SP Agent} & \multicolumn{3}{c}{Our Agent} & Impr. (\%) \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r" & & & Mean & Var & Std & Mean & Var & Std & \\")
    lines.append(r"\midrule")

    for category, rows in sorted(categories.items()):
        cat_stats = compute_category_stats(rows)
        pooled = compute_pooled_agent_stats(rows)
        cat_display = category.replace("_", " ").title()
        if pooled:
            lines.append(
                f"{cat_display} & {cat_stats['count']} & {pooled['total_runs']} & "
                f"{pooled['sp_mean']:.2f} & {pooled['sp_var']:.2f} & {pooled['sp_std']:.2f} & "
                f"{pooled['our_mean']:.2f} & {pooled['our_var']:.2f} & {pooled['our_std']:.2f} & "
                f"{cat_stats['improvement_mean']:.2f} \\\\"
            )

    # Overall
    all_cat_stats = compute_category_stats(results)
    all_pooled = compute_pooled_agent_stats(results)
    lines.append(r"\midrule")
    if all_pooled:
        lines.append(
            f"\\textbf{{Overall}} & {all_cat_stats['count']} & {all_pooled['total_runs']} & "
            f"{all_pooled['sp_mean']:.2f} & {all_pooled['sp_var']:.2f} & {all_pooled['sp_std']:.2f} & "
            f"{all_pooled['our_mean']:.2f} & {all_pooled['our_var']:.2f} & {all_pooled['our_std']:.2f} & "
            f"\\textbf{{{all_cat_stats['improvement_mean']:.2f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", type=str, default="benchmark_results.csv",
                        help="Input CSV file from run_benchmark.py")
    parser.add_argument("--output-latex", type=str, default="results_table.tex",
                        help="Output LaTeX table file")
    parser.add_argument("--output-stats", type=str, default="statistical_tests.json",
                        help="Output file for statistical test results")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(input_dir, args.input)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run run_benchmark.py first.")
        sys.exit(1)

    results = load_results(csv_path)
    if not results:
        print("No results found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(results)} map results from {csv_path}")

    # Per-map table
    print_summary_table(results)

    # Category breakdown
    categories = categorize_results(results)
    print_category_table(categories)

    # Per-agent pooled cost statistics (mean, variance, std dev)
    overall_agent_stats = print_agent_stats_table(results, categories)

    # Statistical tests
    stat_results = run_statistical_tests(results)

    # Save combined stats to JSON
    combined_stats = {}
    if stat_results:
        combined_stats.update(stat_results)
    if overall_agent_stats:
        combined_stats['overall_agent_stats'] = overall_agent_stats
        # Also add per-category agent stats
        category_agent_stats = {}
        for category, rows in sorted(categories.items()):
            cat_pooled = compute_pooled_agent_stats(rows)
            if cat_pooled:
                category_agent_stats[category] = cat_pooled
        combined_stats['category_agent_stats'] = category_agent_stats

    if combined_stats:
        stats_path = os.path.join(input_dir, args.output_stats)
        with open(stats_path, 'w') as f:
            json.dump(combined_stats, f, indent=2)
        print(f"\nStatistical test results saved to: {stats_path}")

    # LaTeX table
    latex_table = generate_latex_table(results, categories)
    latex_path = os.path.join(input_dir, args.output_latex)
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    # Print the LaTeX table
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(latex_table)


if __name__ == "__main__":
    main()
