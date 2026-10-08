"""Analyse the PSPO four-ablation suite with paired seed-level inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from projects.safe_policy_optimisation.scripts.run_pspo_four_ablations import (
    ENVIRONMENTS,
    OUTPUT_ROOT,
    VARIANTS,
    source_dir,
)
from projects.safe_policy_optimisation.utils.pspo_defaults import environment_defaults

COMPARISONS = (
    ("no_entropy", "control", "entropy"),
    ("ce_only", "no_entropy", "margin_conditional_on_no_entropy"),
    ("ce_only", "control", "ce_only_vs_control"),
    ("fixed_lid", "control", "fixed_lid"),
    ("no_gradient", "control", "no_gradient"),
    ("verify_first", "region_first_instrumented", "verify_first"),
)
PRIMARY_ENDPOINTS = (
    "final_unshielded_reward",
    "normalized_reward_auc",
    "exact_shield_alignment",
)
VERIFY_FIRST_ENDPOINTS = (
    *PRIMARY_ENDPOINTS,
    "false_negative_rate",
    "safety_enforcement_s",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def seed_dir(
    variant: str,
    environment: str,
    seed: int,
    *,
    root: Path,
    architecture: str,
) -> Path:
    if variant == "control":
        return source_dir(environment, architecture) / f"seed{seed}"
    return root / variant / architecture / environment / f"seed{seed}"


def read_curve(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "timestep": float(raw["timestep"]),
                        "reward": float(raw["mean_total_reward"]),
                        "alignment": float(raw.get("shield_alignment_rate", "nan")),
                        "wall_time_s": float(raw.get("training_wall_time_s", "nan")),
                        "safety_s": float(raw.get("safety_enforcement_s", "nan")),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda row: row["timestep"])


def normalized_auc(curve: list[dict[str, float]], horizon: int) -> float:
    if not curve or horizon <= 0:
        return float("nan")
    x = np.asarray([row["timestep"] for row in curve], dtype=float)
    y = np.asarray([row["reward"] for row in curve], dtype=float)
    keep = x <= float(horizon)
    x, y = x[keep], y[keep]
    if not x.size:
        return float("nan")
    if x[0] > 0:
        x, y = np.r_[0.0, x], np.r_[y[0], y]
    if x[-1] < horizon:
        x, y = np.r_[x, float(horizon)], np.r_[y, y[-1]]
    return float(np.trapezoid(y, x) / float(horizon))


def first_attainment(
    curve: list[dict[str, float]], threshold: float
) -> tuple[float, float, float]:
    for row in curve:
        if row["reward"] >= threshold:
            return row["timestep"], row["wall_time_s"], row["safety_s"]
    return float("nan"), float("nan"), float("nan")


def load_row(
    variant: str,
    environment: str,
    seed: int,
    *,
    root: Path,
    architecture: str,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    directory = seed_dir(
        variant, environment, seed, root=root, architecture=architecture
    )
    row: dict[str, Any] = {
        "variant": variant,
        "environment": environment,
        "seed": seed,
        "status": "missing",
        "seed_dir": str(directory),
    }
    summary_path, metrics_path = directory / "summary.json", directory / "metrics.json"
    config_path = directory / "config.json"
    curve = read_curve(directory / "learning_curves/evaluation_unshielded_summary.csv")
    if not summary_path.exists() or not metrics_path.exists():
        row["failure_reason"] = "missing summary.json or metrics.json"
        return row, curve
    try:
        summary, metrics = read_json(summary_path), read_json(metrics_path)
        config = read_json(config_path) if config_path.exists() else {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        row["status"] = "failed"
        row["failure_reason"] = str(exc)
        return row, curve
    diagnostics = summary.get("adaptive_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    timing = summary.get("timing")
    timing = timing if isinstance(timing, dict) else {}
    proposed = summary.get("evaluation_proposed_action_safety")
    proposed = proposed if isinstance(proposed, dict) else {}
    checks = int(proposed.get("proposed_action_checks", 0) or 0)
    unsafe_eval = int(proposed.get("unsafe_proposed_action_count", 0) or 0)
    alignment = (
        float(summary["final_exact_all_state_alignment"])
        if "final_exact_all_state_alignment" in summary
        else float(1.0 - unsafe_eval / checks) if checks else float("nan")
    )
    horizon = environment_defaults(environment).total_timesteps
    training = summary.get("training") if isinstance(summary.get("training"), dict) else {}
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    false_negatives = int(diagnostics.get("region_first_false_negatives", 0) or 0)
    audits = int(diagnostics.get("candidate_audits", 0) or 0)
    audited_safe = int(diagnostics.get("audited_exactly_safe", 0) or 0)
    adaptive_config = config.get("adaptive")
    verify_first = bool(
        adaptive_config.get("verify_first", False)
        if isinstance(adaptive_config, dict)
        else False
    )
    exactly_safe_proposals = (
        int(diagnostics.get("candidates_verified_safe", 0) or 0)
        if verify_first
        else audited_safe
    )
    row.update(
        status="complete",
        failure_reason="",
        final_unshielded_reward=float(summary.get("unshielded_eval_mean_total_reward", metrics.get("mean_reward", float("nan")))),
        normalized_reward_auc=normalized_auc(curve, horizon),
        exact_shield_alignment=alignment,
        empirical_training_safety=float(training.get("safety_rate", float("nan"))),
        empirical_evaluation_safety=float(evaluation.get("safety_rate", float("nan"))),
        unsafe_proposed_actions=int(summary.get("unsafe_proposed_actions_during_exploration", 0) or 0),
        unshielded_eval_unsafe_actions=unsafe_eval,
        lid_computations=int(diagnostics.get("rashomon_computations", 0) or 0),
        lid_iterations=int(diagnostics.get("rashomon_iters_spent", 0) or 0),
        projections=int(diagnostics.get("projections_applied", 0) or 0),
        reverts=int(diagnostics.get("fallback_reverts", 0) or 0),
        initialization_failures=int(diagnostics.get("initial_region_failures", 0) or 0),
        false_negatives=false_negatives,
        candidate_audits=audits,
        audited_exactly_safe=audited_safe,
        exactly_safe_proposals=exactly_safe_proposals,
        false_negative_rate=(
            float(false_negatives / exactly_safe_proposals)
            if exactly_safe_proposals
            else float("nan")
        ),
        exact_verification_s=float(timing.get("exact_verification_s", diagnostics.get("exact_verification_wall_time_total_s", float("nan")))),
        lid_complete_s=float(timing.get("lid_complete_s", diagnostics.get("rashomon_wall_time_total_s", float("nan")))),
        projection_s=float(timing.get("projection_s", diagnostics.get("projection_wall_time_total_s", float("nan")))),
        safety_enforcement_s=float(timing.get("safety_enforcement_s", diagnostics.get("safety_enforcement_wall_time_total_s", float("nan")))),
        diagnostic_audit_s=float(timing.get("diagnostic_audit_s", diagnostics.get("diagnostic_audit_wall_time_total_s", float("nan")))),
        decision_path_s=float(timing.get("decision_path_s", float("nan"))),
        training_wall_time_s=float(timing.get("training_wall_time_s", float("nan"))),
        proposal_information_used=bool(diagnostics.get("proposal_information_used", False)),
    )
    base_path = Path(str(config.get("base_policy_path", "")))
    base_summary_path = base_path.parent / "summary.json"
    if base_summary_path.exists():
        base_summary = read_json(base_summary_path)
        base_metrics = base_summary.get("base_policy")
        if isinstance(base_metrics, dict):
            row.update(
                initialization_reached_target=bool(base_metrics.get("reached_target", False)),
                initialization_epochs=int(base_metrics.get("epochs_run", 0) or 0),
                initialization_final_min_all_margin=float(base_metrics.get("final_min_all_margin", float("nan"))),
                initialization_final_safe_entropy=float(base_metrics.get("final_normalized_safe_action_entropy_min", float("nan"))),
                initialization_margin_loss_weight=float(base_metrics.get("margin_loss_weight", 1.0)),
                initialization_entropy_weight=float(base_metrics.get("safe_action_entropy_weight", 0.0)),
            )
    return row, curve


def finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([float(value) for value in values], dtype=float)
    return result[np.isfinite(result)]


def mean_se(values: Iterable[Any]) -> tuple[float, float, int]:
    data = finite(values)
    if not data.size:
        return float("nan"), float("nan"), 0
    se = float(data.std(ddof=1) / math.sqrt(data.size)) if data.size > 1 else 0.0
    return float(data.mean()), se, int(data.size)


def bootstrap_ci(values: Iterable[Any], *, seed: int, reps: int = 10_000) -> tuple[float, float]:
    data = finite(values)
    if not data.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(reps, data.size))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def stable_seed(*parts: object) -> int:
    """Return a process-independent seed for reproducible bootstrap intervals."""

    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def exact_paired_randomization(differences: Iterable[Any]) -> float:
    data = finite(differences)
    data = data[data != 0.0]
    if not data.size:
        return 1.0
    observed = abs(float(data.mean()))
    extreme = 0
    total = 2 ** int(data.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(data.size)):
        statistic = abs(float(np.mean(data * np.asarray(signs))))
        extreme += int(statistic >= observed - 1e-15)
    return float(extreme / total)


def holm_adjust(records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["comparison"], record["endpoint"])].append(record)
    for group in groups.values():
        ordered = sorted(group, key=lambda record: record["p_value"])
        running = 0.0
        count = len(ordered)
        for rank, record in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(record["p_value"]))
            running = max(running, adjusted)
            record["holm_p_value"] = running


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    rows: list[dict[str, Any]],
    curves: dict[tuple[str, str, int], list[dict[str, float]]],
    *,
    output: Path,
) -> None:
    complete = [row for row in rows if row["status"] == "complete"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    reward_axes = list(axes[:2].flat)
    safety_axes = list(axes[2:].flat)
    for reward_axis, safety_axis, environment in zip(
        reward_axes, safety_axes, ENVIRONMENTS
    ):
        horizon = environment_defaults(environment).total_timesteps
        grid = np.linspace(0, horizon, 101)
        for variant in VARIANTS:
            series = []
            for (v, env, _seed), curve in curves.items():
                if v != variant or env != environment or not curve:
                    continue
                x = np.asarray([item["timestep"] for item in curve])
                y = np.asarray([item["reward"] for item in curve])
                series.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            if series:
                reward_axis.plot(grid, np.mean(series, axis=0), label=variant)
            safety_series = []
            for (v, env, _seed), curve in curves.items():
                if v != variant or env != environment or not curve:
                    continue
                x = np.asarray([item["timestep"] for item in curve])
                y = np.asarray([item["alignment"] for item in curve])
                if np.isfinite(y).any():
                    safety_series.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            if safety_series:
                safety_axis.plot(grid, np.nanmean(safety_series, axis=0), label=variant)
        reward_axis.set_title(environment)
        reward_axis.set_xlabel("training steps")
        reward_axis.set_ylabel("unshielded reward")
        safety_axis.set_title(f"{environment} safety")
        safety_axis.set_xlabel("training steps")
        safety_axis.set_ylabel("evaluation action alignment")
        safety_axis.set_ylim(-0.02, 1.02)
    handles, labels = reward_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(output / "reward_safety_learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, environment in zip(axes.flat, ENVIRONMENTS):
        environment_curves = [
            curve
            for (variant, env, _seed), curve in curves.items()
            if env == environment
            and variant in VARIANTS
            and curve
            and any(math.isfinite(item["safety_s"]) for item in curve)
        ]
        maximum_safety_s = max(
            (
                max(
                    item["safety_s"]
                    for item in curve
                    if math.isfinite(item["safety_s"])
                )
                for curve in environment_curves
            ),
            default=0.0,
        )
        grid = np.linspace(0.0, maximum_safety_s, 101)
        for variant in VARIANTS:
            series = []
            for (candidate_variant, env, _seed), curve in curves.items():
                if candidate_variant != variant or env != environment or not curve:
                    continue
                points = [
                    (item["safety_s"], item["reward"])
                    for item in curve
                    if math.isfinite(item["safety_s"])
                    and math.isfinite(item["reward"])
                ]
                if not points:
                    continue
                x = np.maximum.accumulate(np.asarray([point[0] for point in points]))
                y = np.asarray([point[1] for point in points])
                series.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            if series:
                axis.plot(grid, np.mean(series, axis=0), label=variant)
        axis.set_title(environment)
        axis.set_xlabel("cumulative safety-enforcement seconds")
        axis.set_ylabel("unshielded reward")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(output / "reward_vs_safety_enforcement_seconds.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    timing_variants = ("region_first_instrumented", "verify_first")
    for index, environment in enumerate(ENVIRONMENTS):
        offset = index - (len(ENVIRONMENTS) - 1) / 2
        for variant_index, variant in enumerate(timing_variants):
            subset = [row for row in complete if row["environment"] == environment and row["variant"] == variant]
            mean_fn, se_fn, _ = mean_se(row["false_negative_rate"] for row in subset)
            mean_time, se_time, _ = mean_se(row["decision_path_s"] for row in subset)
            x = variant_index + offset * 0.09
            axes[0].errorbar(x, mean_fn, yerr=se_fn, fmt="o", label=environment if variant_index == 0 else None)
            axes[1].errorbar(x, mean_time, yerr=se_time, fmt="o")
    axes[0].set_xticks(range(2), timing_variants, rotation=15)
    axes[0].set_ylabel("false-negative rate (seed mean ± SE)")
    axes[1].set_xticks(range(2), timing_variants, rotation=15)
    axes[1].set_ylabel("decision-path seconds (seed mean ± SE)")
    axes[0].legend(fontsize=7)
    fig.savefig(output / "verify_first_false_negative_compute.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    labels, means, errors = [], [], []
    for environment in ENVIRONMENTS:
        for variant in ("control", "fixed_lid", "no_gradient"):
            subset = [row for row in complete if row["environment"] == environment and row["variant"] == variant]
            mean, se, _ = mean_se(row["lid_iterations"] for row in subset)
            labels.append(f"{environment}\n{variant}")
            means.append(mean)
            errors.append(se)
    axis.bar(range(len(labels)), means, yerr=errors)
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_ylabel("LID iterations (seed mean ± SE)")
    fig.savefig(output / "lid_compute_summaries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    init_variants = ("control", "no_entropy", "ce_only")
    labels = []
    margins = []
    entropies = []
    for environment in ENVIRONMENTS:
        for variant in init_variants:
            subset = [
                row for row in complete
                if row["environment"] == environment and row["variant"] == variant
            ]
            labels.append(f"{environment}\n{variant}")
            margins.append(mean_se(row.get("initialization_final_min_all_margin", float("nan")) for row in subset)[0])
            entropies.append(mean_se(row.get("initialization_final_safe_entropy", float("nan")) for row in subset)[0])
    axes[0].bar(range(len(labels)), margins)
    axes[0].set_ylabel("final minimum all-safe logit margin")
    axes[1].bar(range(len(labels)), entropies)
    axes[1].set_ylabel("final minimum normalized safe-action entropy")
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    fig.savefig(output / "initialization_diagnostics.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--architecture", default="two_hidden")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args(argv)
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, int], list[dict[str, float]]] = {}
    for variant in args.variants:
        for environment in args.envs:
            for seed in args.seeds:
                row, curve = load_row(
                    variant, environment, seed,
                    root=args.root, architecture=args.architecture,
                )
                rows.append(row)
                curves[(variant, environment, seed)] = curve

    thresholds: dict[str, float] = {}
    for environment in args.envs:
        controls = [
            row["final_unshielded_reward"] for row in rows
            if row["variant"] == "control" and row["environment"] == environment
            and row["status"] == "complete"
        ]
        thresholds[environment] = mean_se(controls)[0]
    for row in rows:
        if row["status"] != "complete":
            continue
        curve = curves[(row["variant"], row["environment"], row["seed"])]
        steps, wall_s, safety_s = first_attainment(
            curve, thresholds[row["environment"]]
        )
        row["canonical_reward_threshold"] = thresholds[row["environment"]]
        row["steps_to_canonical_final_reward"] = steps
        row["wall_time_to_canonical_final_reward_s"] = wall_s
        row["safety_time_to_canonical_final_reward_s"] = safety_s

    numeric_endpoints = (
        *PRIMARY_ENDPOINTS,
        "empirical_training_safety", "empirical_evaluation_safety",
        "unsafe_proposed_actions", "lid_computations", "lid_iterations",
        "projections", "reverts", "initialization_failures",
        "false_negative_rate", "decision_path_s", "safety_enforcement_s",
        "training_wall_time_s", "steps_to_canonical_final_reward",
        "wall_time_to_canonical_final_reward_s",
        "safety_time_to_canonical_final_reward_s",
    )
    summaries: list[dict[str, Any]] = []
    for variant in args.variants:
        for environment in args.envs:
            subset = [row for row in rows if row["variant"] == variant and row["environment"] == environment]
            complete = [row for row in subset if row["status"] == "complete"]
            for endpoint in numeric_endpoints:
                mean, se, count = mean_se(row.get(endpoint, float("nan")) for row in complete)
                low, high = bootstrap_ci(
                    (row.get(endpoint, float("nan")) for row in complete),
                    seed=stable_seed(variant, environment, endpoint),
                )
                summaries.append({
                    "variant": variant, "environment": environment,
                    "endpoint": endpoint, "mean": mean, "standard_error": se,
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "complete_seeds": count, "requested_seeds": len(args.seeds),
                    "failed_or_missing_seeds": len(subset) - len(complete),
                })

    paired: list[dict[str, Any]] = []
    by_key = {(row["variant"], row["environment"], row["seed"]): row for row in rows}
    for treatment, reference, comparison in COMPARISONS:
        if treatment not in args.variants or reference not in args.variants:
            continue
        for environment in args.envs:
            endpoints = (
                VERIFY_FIRST_ENDPOINTS
                if comparison == "verify_first"
                else PRIMARY_ENDPOINTS
            )
            for endpoint in endpoints:
                differences, seed_values = [], []
                for seed in args.seeds:
                    left = by_key[(treatment, environment, seed)]
                    right = by_key[(reference, environment, seed)]
                    if left["status"] != "complete" or right["status"] != "complete":
                        continue
                    difference = float(left[endpoint]) - float(right[endpoint])
                    if math.isfinite(difference):
                        differences.append(difference)
                        seed_values.append(f"{seed}:{difference:.9g}")
                mean, se, count = mean_se(differences)
                low, high = bootstrap_ci(
                    differences,
                    seed=stable_seed(comparison, environment, endpoint),
                )
                paired.append({
                    "comparison": comparison, "treatment": treatment,
                    "reference": reference, "environment": environment,
                    "endpoint": endpoint, "paired_seeds": count,
                    "requested_seeds": len(args.seeds), "mean_difference": mean,
                    "standard_error": se, "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "p_value": exact_paired_randomization(differences),
                    "seed_differences": ";".join(seed_values),
                })
    holm_adjust(paired)

    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_seed_tables.csv", paired)
    (output / "results.json").write_text(
        json.dumps(clean_json({
            "per_seed": rows, "summary": summaries, "paired": paired,
            "canonical_reward_thresholds": thresholds,
        }), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_outputs(rows, curves, output=output)

    lines = [
        "# PSPO four-ablation study", "",
        "Seeds are the independent units. Intervals are seed-bootstrap 95% CIs; ",
        "paired p-values are exact sign-randomization tests with Holm correction ",
        "across environments for each comparison and endpoint. Verify-first also ",
        "tests false-negative rate and total safety-enforcement time.", "",
        "## Completion", "",
        "| Variant | Environment | Complete / requested |", "|---|---|---:|",
    ]
    for variant in args.variants:
        for environment in args.envs:
            complete = sum(
                row["status"] == "complete" for row in rows
                if row["variant"] == variant and row["environment"] == environment
            )
            lines.append(f"| {variant} | {environment} | {complete} / {len(args.seeds)} |")
    lines += ["", "## Paired primary endpoints", "", "| Comparison | Environment | Endpoint | Mean difference | 95% CI | Exact p | Holm p | n |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for record in paired:
        lines.append(
            "| {comparison} | {environment} | {endpoint} | {mean_difference:.6g} | "
            "[{bootstrap_ci_low:.6g}, {bootstrap_ci_high:.6g}] | {p_value:.6g} | "
            "{holm_p_value:.6g} | {paired_seeds} |".format(**record)
        )
    lines += [
        "", "## Definitions", "",
        "`normalized_reward_auc` is the trapezoidal integral of mean unshielded ",
        "reward divided by the environment horizon (time-axis normalization). ",
        "Diagnostic exact-audit time is excluded from region-first `decision_path_s`. ",
        "Failed and non-attaining seeds remain in `per_seed.csv` and completion counts; ",
        "non-attainment is represented as missing time/step values, never silently dropped.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
