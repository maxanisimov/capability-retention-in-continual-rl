#!/usr/bin/env python
"""Plot the best completed total reward for each PSPO algorithm variant.

Precomputed results are loaded through the existing architecture-result
catalogue. Adaptive results are aggregated from seed-level ``config.json`` and
``metrics.json`` files. A candidate cohort must contain at least ten completed
seeds; selection is then by maximum mean total reward across architectures and
hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.scripts import plot_architecture_results as precomputed_results  # noqa: E402


DEFAULT_ADAPTIVE_ROOTS = (
    REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs",
    REPO / "outputs/_pspo_hparam",
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/results/pspo/comparisons/best_variants"
)

ENVIRONMENTS = [
    "Bridge Crossing v1",
    "Bridge Crossing v2",
    "Colour Bomb v1",
    "Colour Bomb v2",
    "Media Streaming",
    "MiniPacman",
]

VARIANTS = [
    ("precomputed", "Precomputed", "#16865C"),
    ("adaptive_v1_projection", "Adaptive v1\nprojection", "#2673B8"),
    ("adaptive_v1_monitor", "Adaptive v1\nmonitor", "#7A7A7A"),
    ("adaptive_v2_union", "Adaptive v2\nunion", "#D18B00"),
    ("adaptive_v2_replace", "Adaptive v2\nreplace", "#C54A3D"),
]

ARCHITECTURE_LABELS = {
    "tabular": "tabular",
    "one_hidden": "1 hidden",
    "two_hidden": "2 hidden",
}

ARCHITECTURE_TITLES = {
    "tabular": "Tabular Actor-Critic",
    "one_hidden": "One-Hidden-Layer Actor-Critic",
    "two_hidden": "Two-Hidden-Layer Actor-Critic",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class Result:
    variant: str
    environment: str
    architecture: str
    reward_mean: float
    reward_std: float
    reward_sem: float
    safety_mean: float
    n_seeds: int
    source: str
    configuration: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root containing adaptive seed runs. Repeat to scan multiple roots. "
            "Defaults to the canonical artifact tree and legacy PSPO sweeps."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-seeds", type=int, default=10)
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def normalise_environment(name: str) -> str:
    return "Colour Bomb v1" if name == "Colour Bomb" else name


def classify_adaptive(config: dict[str, Any]) -> str | None:
    algorithm = config.get("algorithm")
    adaptive = config.get("adaptive") or {}
    if algorithm == "adaptive_safe_ppo":
        strategy = adaptive.get("unsafe_update_strategy", "rashomon_project")
        if strategy == "none":
            return "adaptive_v1_monitor"
        if strategy == "rashomon_project":
            return "adaptive_v1_projection"
        return None
    if algorithm == "adaptive_safe_ppo_v2":
        update_mode = adaptive.get("region_update_mode", "union")
        if update_mode in {"union", "replace"}:
            return f"adaptive_v2_{update_mode}"
    if algorithm == "pspo_adaptive":
        if adaptive.get("verify_first", False):
            return "adaptive_v1_projection"
        update_mode = adaptive.get("region_update_mode", "replace")
        if update_mode in {"union", "replace"}:
            return f"adaptive_v2_{update_mode}"
    return None


def architecture_from_config(config: dict[str, Any], source: Path) -> str:
    architecture = config.get("base_policy_architecture") or {}
    return precomputed_results.arch_name(architecture.get("n_hidden"), source)


def environment_from_config(config: dict[str, Any], source: Path) -> str:
    name = precomputed_results.env_name(config.get("env_id"), source)
    return normalise_environment(name)


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def precomputed_candidates(minimum_seeds: int) -> list[Result]:
    candidates: list[Result] = []
    records = precomputed_results.hparam_records() + precomputed_results.aggregate_records()
    for record in records:
        environment = normalise_environment(record.environment)
        if environment not in ENVIRONMENTS or record.reward.n < minimum_seeds:
            continue
        architecture = precomputed_results.arch_name(
            record.hyperparameters.get("n_hidden"), record.source
        )
        if architecture == "unknown":
            architecture = record.architecture
        candidates.append(
            Result(
                variant="precomputed",
                environment=environment,
                architecture=architecture,
                reward_mean=record.reward.mean,
                reward_std=record.reward.std,
                reward_sem=record.reward.sem,
                safety_mean=record.safety.mean,
                n_seeds=record.reward.n,
                source=relative(record.source),
                configuration=record.hyperparameters,
            )
        )
    return candidates


def adaptive_candidates(roots: list[Path], minimum_seeds: int) -> list[Result]:
    cohorts: dict[Path, list[tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    for root in roots:
        if root.name == "_pspo_hparam":
            config_paths = root.glob(
                "*/adaptive/iters_*__margin_*/runs/seed*/config.json"
            )
        else:
            config_paths = root.rglob("config.json")
        for config_path in config_paths:
            config = load_json(config_path)
            if config is None or classify_adaptive(config) is None:
                continue
            match = re.fullmatch(r"seed_?(\d+)", config_path.parent.name)
            if match is None:
                continue
            metrics_path = config_path.with_name("metrics.json")
            metrics = load_json(metrics_path)
            if metrics is None:
                continue
            cohorts.setdefault(config_path.parent.parent, []).append(
                (config, metrics, metrics_path)
            )

    candidates: list[Result] = []
    for cohort, entries in cohorts.items():
        by_seed: dict[int, tuple[dict[str, Any], dict[str, Any], Path]] = {}
        for config, metrics, metrics_path in entries:
            seed = int(config.get("seed", -1))
            if seed >= 0:
                by_seed[seed] = (config, metrics, metrics_path)
        if len(by_seed) < minimum_seeds:
            continue

        ordered = [by_seed[seed] for seed in sorted(by_seed)]
        configs = [entry[0] for entry in ordered]
        variants = {classify_adaptive(config) for config in configs}
        environments = {environment_from_config(config, cohort) for config in configs}
        architectures = {architecture_from_config(config, cohort) for config in configs}
        if len(variants) != 1 or len(environments) != 1 or len(architectures) != 1:
            continue

        rewards: list[float] = []
        safeties: list[float] = []
        for _config, metrics, _metrics_path in ordered:
            try:
                rewards.append(float(metrics["reward"]["mean_total_reward"]))
                safeties.append(float(metrics["safety"]["safety_rate"]))
            except (KeyError, TypeError, ValueError):
                break
        if len(rewards) != len(ordered):
            continue

        reward_mean, reward_std = mean_std(rewards)
        safety_mean, _ = mean_std(safeties)
        config = configs[0]
        adaptive = dict(config.get("adaptive") or {})
        candidates.append(
            Result(
                variant=next(iter(variants)) or "unknown",
                environment=next(iter(environments)),
                architecture=next(iter(architectures)),
                reward_mean=reward_mean,
                reward_std=reward_std,
                reward_sem=reward_std / math.sqrt(len(rewards)),
                safety_mean=safety_mean,
                n_seeds=len(rewards),
                source=relative(cohort),
                configuration={
                    "algorithm": config.get("algorithm"),
                    "adaptive": adaptive,
                    "base_policy_path": config.get("base_policy_path"),
                },
            )
        )
    return candidates


def select_best(
    candidates: list[Result], *, architecture: str | None = None
) -> dict[tuple[str, str], Result]:
    selected: dict[tuple[str, str], Result] = {}
    for candidate in candidates:
        if architecture is not None and candidate.architecture != architecture:
            continue
        key = (candidate.environment, candidate.variant)
        incumbent = selected.get(key)
        rank = (
            candidate.reward_mean,
            candidate.safety_mean,
            candidate.n_seeds,
            candidate.source,
        )
        if incumbent is None:
            selected[key] = candidate
            continue
        incumbent_rank = (
            incumbent.reward_mean,
            incumbent.safety_mean,
            incumbent.n_seeds,
            incumbent.source,
        )
        if rank > incumbent_rank:
            selected[key] = candidate
    return selected


def reward_limits(results: list[Result | None]) -> tuple[float, float]:
    completed = [result for result in results if result is not None]
    if not completed:
        return -1.0, 1.0
    lows = [min(0.0, result.reward_mean - result.reward_sem) for result in completed]
    highs = [max(0.0, result.reward_mean + result.reward_sem) for result in completed]
    low, high = min(lows), max(highs)
    span = high - low
    if span < 1e-9:
        span = max(1.0, abs(high))
    return low - 0.16 * span, high + 0.22 * span


def draw_environment(
    ax: plt.Axes,
    environment: str,
    selected: dict[tuple[str, str], Result],
    *,
    detailed_labels: bool,
) -> None:
    results = [selected.get((environment, key)) for key, _label, _colour in VARIANTS]
    positions = list(range(len(VARIANTS)))
    means = [result.reward_mean if result is not None else 0.0 for result in results]
    errors = [result.reward_sem if result is not None else 0.0 for result in results]
    colours = [colour if result is not None else "white" for result, (_key, _label, colour) in zip(results, VARIANTS)]
    edges = [colour for _key, _label, colour in VARIANTS]

    bars = ax.bar(
        positions,
        means,
        yerr=errors,
        width=0.68,
        color=colours,
        edgecolor=edges,
        linewidth=1.2,
        error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
    )
    low, high = reward_limits(results)
    ax.set_ylim(low, high)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(environment, pad=8)
    ax.set_ylabel("Mean total reward")
    ax.set_xticks(positions, [label for _key, label, _colour in VARIANTS])
    ax.tick_params(axis="x", length=0, pad=6)

    span = high - low
    for bar, result in zip(bars, results):
        x = bar.get_x() + bar.get_width() / 2
        if result is None:
            ax.text(x, 0.0 + 0.025 * span, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")
            bar.set_hatch("//")
            continue
        if result.reward_mean >= 0:
            y = result.reward_mean + result.reward_sem + 0.025 * span
            va = "bottom"
        else:
            y = result.reward_mean - result.reward_sem - 0.025 * span
            va = "top"
        value = f"{result.reward_mean:.3g} +/- {result.reward_sem:.2g}"
        if detailed_labels:
            architecture = ARCHITECTURE_LABELS.get(result.architecture, result.architecture)
            value += f"\n{architecture}"
        ax.text(x, y, value, ha="center", va=va, fontsize=7.5, color="#222222")


def plot_individual_charts(
    selected: dict[tuple[str, str], Result],
    output_dir: Path,
    *,
    architecture: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for environment in ENVIRONMENTS:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        draw_environment(
            ax,
            environment,
            selected,
            detailed_labels=architecture is None,
        )
        if architecture is not None:
            ax.set_title(f"{environment}\n{ARCHITECTURE_TITLES[architecture]}")
        fig.tight_layout()
        stem = environment.lower().replace(" ", "_")
        fig.savefig(output_dir / f"{stem}_total_reward.png", dpi=240, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}_total_reward.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_overview(
    selected: dict[tuple[str, str], Result],
    output_dir: Path,
    *,
    architecture: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for ax, environment in zip(axes.flat, ENVIRONMENTS):
        draw_environment(ax, environment, selected, detailed_labels=False)
    title = "Best completed PSPO variant by environment"
    if architecture is not None:
        title += f": {ARCHITECTURE_TITLES[architecture]}"
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    stem = (
        "all_environments_total_reward"
        if architecture is None
        else f"{architecture}_all_environments_total_reward"
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    selected: dict[tuple[str, str], Result],
    output_dir: Path,
    minimum_seeds: int,
    *,
    architecture: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_scope = (
        "across available architectures and hyperparameters"
        if architecture is None
        else f"within the {architecture} architecture and across available hyperparameters"
    )
    payload: dict[str, Any] = {
        "metric": "reward.mean_total_reward",
        "selection": (
            "Maximum completed cohort mean per environment and semantic PSPO "
            f"variant, {selection_scope}."
        ),
        "minimum_completed_seeds": minimum_seeds,
        "architecture": architecture,
        "variants": [key for key, _label, _colour in VARIANTS],
        "environments": {},
    }
    rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        payload["environments"][environment] = {}
        for variant, label, _colour in VARIANTS:
            result = selected.get((environment, variant))
            payload["environments"][environment][variant] = asdict(result) if result else None
            rows.append(
                {
                    "environment": environment,
                    "variant": variant,
                    "label": label.replace("\n", " "),
                    "reward_mean": result.reward_mean if result else "",
                    "reward_std": result.reward_std if result else "",
                    "reward_sem": result.reward_sem if result else "",
                    "safety_mean": result.safety_mean if result else "",
                    "n_seeds": result.n_seeds if result else "",
                    "architecture": result.architecture if result else "",
                    "source": result.source if result else "",
                }
            )
    (output_dir / "best_pspo_variant_rewards.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "best_pspo_variant_rewards.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_architecture_outputs(
    candidates: list[Result], output_dir: Path, minimum_seeds: int
) -> None:
    architecture_root = output_dir / "by_architecture"
    architecture_root.mkdir(parents=True, exist_ok=True)
    for architecture in ARCHITECTURE_LABELS:
        selected = select_best(candidates, architecture=architecture)
        architecture_dir = architecture_root / architecture
        write_manifest(
            selected,
            architecture_dir,
            minimum_seeds,
            architecture=architecture,
        )
        plot_individual_charts(
            selected,
            architecture_dir,
            architecture=architecture,
        )
        plot_overview(
            selected,
            architecture_root,
            architecture=architecture,
        )
        print(
            f"  {ARCHITECTURE_TITLES[architecture]}: "
            f"{len(selected)} completed environment/variant result(s)"
        )


def main() -> int:
    args = parse_args()
    if args.minimum_seeds <= 0:
        raise SystemExit("--minimum-seeds must be positive")
    adaptive_roots = [
        path.resolve() for path in (args.adaptive_root or DEFAULT_ADAPTIVE_ROOTS)
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = precomputed_candidates(args.minimum_seeds)
    candidates.extend(adaptive_candidates(adaptive_roots, args.minimum_seeds))
    selected = select_best(candidates)
    write_manifest(selected, output_dir, args.minimum_seeds)
    plot_individual_charts(selected, output_dir)
    plot_overview(selected, output_dir)
    write_architecture_outputs(candidates, output_dir, args.minimum_seeds)

    print(f"Selected {len(selected)} completed environment/variant result(s).")
    print(f"Wrote charts and manifests to {relative(output_dir)}")
    for environment in ENVIRONMENTS:
        available = [
            key for key, _label, _colour in VARIANTS if (environment, key) in selected
        ]
        print(f"  {environment}: {', '.join(available) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
