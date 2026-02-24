"""Generate a LaTeX table from the poisoned apple experiment CSV results."""

import csv
import os

TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")

# Map CSV filenames to environment labels
ENV_CONFIGS = {
    "results_simple_5x5.csv": r"$5 \times 5$",
    "results_simple_6x6.csv": r"$6 \times 6$",
    "results_simple_7x7.csv": r"$7 \times 7$",
    "results_simple_8x8.csv": r"$8 \times 8$",
}

# Readable metric names
METRIC_LABELS = {
    "avg_reward": "Avg Reward",
    "avg_performance_success": "Perf. Success",
    "avg_safety_success": "Safety Success",
    "avg_overall_success": "Overall Success",
}

# Adaptation strategies (column groups)
STRATEGIES = ["NoAdapt", "UnsafeAdapt", "SafeAdapt"]
STRATEGY_LABELS = {
    "NoAdapt": "No Adapt.",
    "UnsafeAdapt": "Unsafe Adapt.",
    "SafeAdapt": "Safe Adapt.",
}
TASKS = ["Task 1", "Task 2"]


def load_csv(filepath: str) -> dict[str, dict[str, str]]:
    """Load CSV into {metric_name: {column_name: value}} dict."""
    data: dict[str, dict[str, str]] = {}
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # first row is header
        col_names = header[1:]  # skip the index column
        for row in reader:
            metric = row[0]
            data[metric] = {col_names[i]: row[i + 1] for i in range(len(col_names))}
    return data


def fmt(val: float) -> str:
    """Format a numeric value for the table."""
    if val == int(val):
        return f"{val:.1f}"
    return f"{val:.2f}"


def build_latex_table() -> str:
    # Collect all data: env_label -> {metric -> {col -> val}}
    all_data: dict[str, dict[str, dict[str, str]]] = {}
    for csv_name, env_label in ENV_CONFIGS.items():
        path = os.path.join(TABLES_DIR, csv_name)
        if os.path.exists(path):
            all_data[env_label] = load_csv(path)

    if not all_data:
        raise FileNotFoundError("No CSV files found in tables directory.")

    metrics = list(METRIC_LABELS.keys())

    # Build column spec: env | metric | (Task1 Task2) x 3 strategies
    col_spec = "ll" + "".join(["cc" for _ in STRATEGIES])
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Poisoned Apple environment results across grid sizes and adaptation strategies.}")
    lines.append(r"\label{tab:poisoned_apple_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: strategy group headers
    header1_parts = [" ", " "]
    for strat in STRATEGIES:
        header1_parts.append(
            r"\multicolumn{2}{c}{" + STRATEGY_LABELS[strat] + "}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    # Cmidrules under each strategy group
    cmidrules = ""
    for i, _ in enumerate(STRATEGIES):
        col_start = 3 + i * 2
        col_end = col_start + 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
    lines.append(cmidrules.strip())

    # Header row 2: Task labels
    header2_parts = ["Grid Size", "Metric"]
    for _ in STRATEGIES:
        header2_parts.extend(["Task 1", "Task 2"])
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows, grouped by environment
    for env_idx, (env_label, data) in enumerate(all_data.items()):
        n_metrics = len(metrics)
        for m_idx, metric in enumerate(metrics):
            row_parts = []
            # Environment label only on first metric row (multirow)
            if m_idx == 0:
                row_parts.append(
                    r"\multirow{" + str(n_metrics) + "}{*}{" + env_label + "}"
                )
            else:
                row_parts.append("")

            row_parts.append(METRIC_LABELS[metric])

            for strat in STRATEGIES:
                for task in TASKS:
                    col_name = f"{strat} / {task}"
                    val = data[metric][col_name]
                    row_parts.append(fmt(float(val)))

            lines.append(" & ".join(row_parts) + r" \\")

        # Separator between environment groups (except last)
        if env_idx < len(all_data) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")  # close resizebox
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    latex = build_latex_table()
    print(latex)
    # Also write to file
    out_path = os.path.join(os.path.dirname(__file__), "tables", "combined_results.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table written to {out_path}")
