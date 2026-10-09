#!/usr/bin/env python3
"""One MASA table of final total reward and safe-trajectory rate.

PSPO is the region-first line-segment variant. Statistics use ten seed-level
observations and two sample standard errors. Bold marks the top three means
per environment and metric, including ties at the third ordered position.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_masa_learning_curve_comparisons as curves

DEFAULT_OUTPUT = curves.FIGURES / "masa_all_methods_table"
GROUPS = ("PSPO (default)", "Baselines", "Projected baselines", "Other PSPO variants")
VARIANT_LABELS = {"default": "PSPO orthotope (region-first)",
                  "verify_first": "PSPO orthotope (verify-first)",
                  "segment_verify_first": "PSPO-LS (verify-first)"}
METRICS = (("reward", "Total reward", 1, 2, False),
           ("safety", r"Safety (\%)", 100, 1, False))


@dataclasses.dataclass(frozen=True)
class Method:
    key: str
    label: str
    group: str
    series: curves.Series
    variant: str | None = None


def methods(environment, rl_sgf_runs) -> list[Method]:
    main = curves.series_for("a_pspo_vs_baselines", environment, rl_sgf_runs)
    projected = curves.series_for("b_pspo_vs_projected_baselines", environment, rl_sgf_runs)
    other = curves.series_for("c_four_pspo_variants", environment, rl_sgf_runs)
    methods = [Method("pspo", "PSPO (PSPO-LS, region-first)", GROUPS[0], main[-1], "segment")]
    methods += [Method(s.spec.key, s.spec.label, GROUPS[1], s) for s in main[:-1]]
    methods += [Method("projected_" + s.spec.key,
                       s.spec.label.replace(" + final projection", " + projection"), GROUPS[2], s)
                for s in projected[:-1]]
    methods += [Method("pspo_" + s.spec.key, VARIANT_LABELS[s.spec.key], GROUPS[3], s, s.spec.key)
                for s in other if s.spec.key != "segment"]
    if len(methods) != 17 or len({m.key for m in methods}) != 17:
        raise ValueError("Expected seventeen distinct methods")
    return methods


class Sources:
    def __init__(self):
        self.files = {}

    def record(self, path):
        path = path.resolve()
        if str(path) not in self.files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.files[str(path)] = {"path": str(path), "sha256": digest.hexdigest(),
                                     "bytes": path.stat().st_size}

    def json(self, path):
        self.record(path)
        return json.loads(path.read_text())


def summary(values):
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected ten finite observations")
    return statistics.fmean(values), 2 * statistics.stdev(values) / math.sqrt(10)


def collect(rl_sgf_runs):
    sources = Sources()
    rows = []
    for environment in curves.canon.ENVIRONMENTS:
        for method in methods(environment, rl_sgf_runs):
            for seed in range(10):
                directory = method.series.root / f"seed{seed}"
                metrics_path = directory / method.series.terminal_path
                node = sources.json(metrics_path)
                if method.series.projected:
                    if node["status"] != "complete" or node["smoke"] or node["episodes"] != 100 or node["seed"] != seed:
                        raise ValueError(f"Incomplete projected evaluation: {metrics_path}")
                    if node["evaluation_policy"] != "nominal_unshielded_deterministic":
                        raise ValueError(f"Unexpected projected evaluation policy: {metrics_path}")
                    reward, safety = node["deployed"]["mean_total_reward"], node["deployed"]["safe_trajectory_rate"]
                else:
                    for section in method.series.terminal_section:
                        node = node[section]
                    if node["eval_episodes"] != 100:
                        raise ValueError(f"Unexpected final evaluation budget: {metrics_path}")
                    reward, safety = node["reward"]["mean_total_reward"], node["safety"]["safety_rate"]
                if method.variant:
                    config = sources.json(directory / "config.json")
                    saved = sources.json(directory / "summary.json")
                    expected_shape = "segment" if method.variant in {"segment", "segment_verify_first"} else "orthotope"
                    expected_verify_first = method.variant in {"verify_first", "segment_verify_first"}
                    if (config["adaptive"]["safe_region_shape"] != expected_shape or
                        config["adaptive"]["verify_first"] != expected_verify_first or
                        config["total_timesteps"] != environment.nominal_budget or
                        config["evaluation_policy"] != "unshielded" or saved["early_stop_triggered"]):
                        raise ValueError(f"Unexpected PSPO variant/configuration: {directory}")
                if not math.isfinite(float(reward)) or not 0 <= float(safety) <= 1:
                    raise ValueError(f"Invalid reward/safety observation: {metrics_path}")
                rows.append({"environment_key": environment.key, "environment": environment.label,
                             "nominal_timesteps": environment.nominal_budget, "group": method.group,
                             "method_key": method.key, "method": method.label, "seed": seed,
                             "reward": float(reward), "safety": float(safety),
                             "metrics_source": str(metrics_path.resolve()), "episodes_per_seed": 100})
    return rows, sources


def aggregate(seed_rows):
    rows = []
    for environment in curves.canon.ENVIRONMENTS:
        keys = list(dict.fromkeys(row["method_key"] for row in seed_rows if row["environment_key"] == environment.key))
        for key in keys:
            selected = [row for row in seed_rows if row["environment_key"] == environment.key and row["method_key"] == key]
            if [row["seed"] for row in selected] != list(range(10)):
                raise ValueError("Expected exactly seeds 0--9")
            row = {field: selected[0][field] for field in ("environment_key", "environment", "nominal_timesteps", "group", "method_key", "method")}
            row["n_seeds"] = 10
            for metric, *_ in METRICS:
                values = [item[metric] for item in selected if item[metric] is not None]
                row[metric + "_n"] = len(values)
                row[metric + "_mean"], row[metric + "_two_se"] = summary(values) if len(values) == 10 else (None, None)
            rows.append(row)
        for metric, _, _, _, minimise in METRICS:
            selected = [row for row in rows if row["environment_key"] == environment.key]
            values = [row[metric + "_mean"] for row in selected if row[metric + "_mean"] is not None]
            for row in selected:
                value = row[metric + "_mean"]
                rank = None if value is None else 1 + sum(
                    (other < value if minimise else other > value) and not math.isclose(other, value, rel_tol=0, abs_tol=1e-12)
                    for other in values)
                row[metric + "_rank"] = rank
                row[metric + "_bold"] = rank is not None and rank <= 3
    return rows


def cell(row, metric_spec, latex=True):
    metric, _, multiplier, decimals, _ = metric_spec
    mean, error = row[metric + "_mean"], row[metric + "_two_se"]
    if mean is None:
        return r"\textemdash" if latex else "—"
    mean = mean * multiplier if round(mean * multiplier, decimals) else 0.0
    text = f"{mean:.{decimals}f} \\pm {error * multiplier:.{decimals}f}"
    if latex:
        if row[metric + "_bold"]:
            text = r"\mathbf{" + text + "}"
        text = "$" + text + "$"
    else:
        text = text.replace(r"\pm", "±")
    return text


def table(rows):
    order = [(row["method_key"], row["method"], row["group"]) for row in rows
             if row["environment_key"] == curves.canon.ENVIRONMENTS[0].key]
    lookup = {(row["environment_key"], row["method_key"]): row for row in rows}
    lines = ["% Generated by scripts/generate_masa_all_methods_table.py.",
             "% Requires booktabs and graphicx. Two panels form one table.",
             r"\begin{table*}[p]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.08}"]
    for panel in range(2):
        environments = curves.canon.ENVIRONMENTS[3 * panel:3 * panel + 3]
        lines += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}l*{3}{cc}@{}}", r"\toprule",
                  "Method & " + " & ".join(r"\multicolumn{2}{c}{" + env.label + "}" for env in environments) + r" \\",
                  r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                  " & " + " & ".join(spec[1] for _ in environments for spec in METRICS) + r" \\"]
        current = None
        for key, label, group in order:
            if current != group:
                lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textbf{" + group + r"}} \\"]
                current = group
            lines.append(label + " & " + " & ".join(cell(lookup[env.key, key], spec)
                         for env in environments for spec in METRICS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}%", "}"]
        if panel == 0:
            lines.append(r"\par\vspace{0.7em}")
    caption = (
        "Test-time total reward and safe-trajectory rate for the six MASA environments. "
        "Values are means $\\pm$ two standard errors over ten seeds; final evaluations use 100 episodes per seed. "
        "PSPO (default) is region-first PSPO-LS. Bold marks the three highest reward means and the three highest safety means "
        "within each environment across all groups, including ties at the third position; ranking uses unrounded means. "
        "For projected baselines, unsafe final actors are projected; actors already safe are retained. "
        "PSPO-LS and projected baselines have different initial actor weights."
    )
    lines += [r"\caption{" + caption + "}", r"\label{tab:masa-all-methods-reward-safety}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preview(output, rows):
    """Readable vector preview; LaTeX remains the publication table source."""
    import matplotlib.pyplot as plt
    order = [row for row in rows if row["environment_key"] == curves.canon.ENVIRONMENTS[0].key]
    lookup = {(row["environment_key"], row["method_key"]): row for row in rows}
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.7, 12.4))
    fig.suptitle("MASA: test-time total reward and safety rate", fontsize=13, y=0.986)
    for panel in range(2):
        environments = curves.canon.ENVIRONMENTS[3 * panel:3 * panel + 3]
        axis = fig.add_axes([0.015, 0.57 - panel * 0.445, 0.97, 0.385])
        axis.axis("off")
        data = [["Method"] + [""] * 6,
                [""] + [spec[1].replace(r"\%", "%") for _ in environments for spec in METRICS]]
        bold = set()
        group_rows = []
        current = None
        for method in order:
            if current != method["group"]:
                group_rows.append(len(data)); data.append([method["group"]] + [""] * 6)
                current = method["group"]
            index = len(data)
            data.append([method["method"]] + [cell(lookup[env.key, method["method_key"]], spec, False)
                        for env in environments for spec in METRICS])
            for column, (env, spec) in enumerate([(env, spec) for env in environments for spec in METRICS], 1):
                if lookup[env.key, method["method_key"]][spec[0] + "_bold"]:
                    bold.add((index, column))
        rendered = axis.table(cellText=data, colWidths=[0.30] + [0.70 / 6] * 6,
                              cellLoc="center", bbox=[0, 0, 1, 1])
        rendered.auto_set_font_size(False); rendered.set_fontsize(7.8)
        # Span the two metric columns with text drawn above the blank header
        # cells; a single-cell label would be covered by its neighbour's face.
        for column, environment in enumerate(environments):
            axis.text(0.30 + (column + 0.5) * 0.70 / 3, 1 - 0.5 / len(data),
                      environment.label, ha="center", va="center", weight="bold",
                      fontsize=7.8, transform=axis.transAxes, zorder=4)
        for (r, c), item in rendered.get_celld().items():
            item.set_linewidth(0.2); item.set_edgecolor("#D5D5D5")
            if c == 0:
                item.set_text_props(ha="left"); item.PAD = 0.015
            if r < 2 or r in group_rows:
                item.set_facecolor("#E9EFF3" if r < 2 else "#F2F2F2")
                item.set_text_props(weight="bold")
            elif (r, c) in bold:
                item.set_text_props(weight="bold")
    notes = (
        "Means ± 2 standard errors across ten seeds; 100 test episodes per seed. PSPO default = region-first PSPO-LS.\n"
        "Bold: top three reward and safety means per environment across all method groups, including ties; unrounded ranking.\n"
        "For projected baselines, unsafe final actors are projected; actors already safe are retained.\n"
        "PSPO-LS and projected baselines have different initial actor weights. This preview is rendered directly from the exported statistics."
    )
    fig.text(0.02, 0.025, notes, va="bottom", fontsize=7.3, linespacing=1.6)
    fig.savefig(output / "masa_all_methods_table_preview.pdf")
    fig.savefig(output / "masa_all_methods_table_preview.png", dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rl-sgf-runs", type=Path, default=curves.RL_SGF_RUNS)
    args = parser.parse_args(argv)
    seed_rows, sources = collect(args.rl_sgf_runs.resolve())
    rows = aggregate(seed_rows)
    if len(seed_rows) != 1020 or len(rows) != 102:
        raise ValueError("Expected six environments, seventeen methods, and ten seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "masa_all_methods_seed_metrics.csv", seed_rows)
    write_csv(args.output_dir / "masa_all_methods_summary.csv", rows)
    (args.output_dir / "masa_all_methods_table.tex").write_text(table(rows))
    preview(args.output_dir, rows)
    metadata = {"generator": str(Path(__file__).resolve()),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed_count": 10, "eval_episodes_per_seed": 100,
                "metrics": [spec[0] for spec in METRICS],
                "error": "two sample standard errors across seed means",
                "primary_pspo": "region-first line segment", "groups": GROUPS,
                "ranking": "Competition ranks over unrounded means across all seventeen rows per environment; rank <= 3 is bold. Absolute tie tolerance 1e-12.",
                "source_files": list(sources.files.values())}
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# All-method MASA reward and safety table\n\n"
        "`masa_all_methods_table.tex` is one full-width table with two stacked three-environment panels. "
        "Groups: PSPO default (region-first PSPO-LS), seven baselines, six projected baselines, three other PSPO variants. "
        "Requires `booktabs` and `graphicx`. The two metrics are test-time total reward and safe-trajectory rate, expressed as a percentage. "
        "The PDF/PNG preview is rendered from the same summary statistics; it is not a compiled LaTeX document.\n\n"
        "All final evaluations use 100 deterministic episodes for each of ten seeds. Errors are twice the sample standard error "
        "of seed-level means. Bold uses unrounded competition ranks <=3 across all methods in each environment, including ties "
        "at the third ordered position. This can highlight more than three methods, particularly safety.\n\n"
        "Projected actors are kept when already safe. PSPO-LS and projected baselines do not have identical initial actor weights. "
        "Aggregate and seed-level CSVs contain reward and safety measurements and their source provenance.\n\n"
        "Regenerate from the repository root:\n\n```bash\n"
        "MPLCONFIGDIR=/tmp/masa-all-methods-mpl .venv/bin/python "
        "projects/safe_policy_optimisation/scripts/generate_masa_all_methods_table.py\n```\n")
    print(f"Saved {len(rows)} reward/safety summaries from {len(seed_rows)} seed observations to {args.output_dir.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
