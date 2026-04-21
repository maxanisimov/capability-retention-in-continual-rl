"""Evaluate a trained LunarLander policy for reward/success/failure metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.pipelines.lunarlander.core.env.env_factory import (
    _make_lunarlander_env,
)
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.eval.evaluate_policy import (
    POLICY_TO_SUBDIR,
    _build_actor_from_state_dict,
    _resolve_actor_path,
    _sanitize,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_outputs_root,
    default_task_settings_file,
    resolve_policy_dir as _resolve_policy_dir,
)

LATEX_POLICY_ROW_ORDER = [
    "source",
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
]

POLICY_DISPLAY_NAMES = {
    "source": "Source",
    "downstream_unconstrained": "Unconstrained Adaptation",
    "downstream_ewc": "EWC",
    "downstream_rashomon": "Rashomon",
}

REPORT_METRIC_SPECS = {
    "total_reward": {
        "latex_label": "total reward",
        "mean_key": "total_reward_mean_across_seeds_mean",
        "std_key": "total_reward_mean_across_seeds_std",
    },
    "success_rate": {
        "latex_label": "success rate",
        "mean_key": "success_rate_across_seeds_mean",
        "std_key": "success_rate_across_seeds_std",
    },
    "failure_rate": {
        "latex_label": "failure rate",
        "mean_key": "failure_rate_across_seeds_mean",
        "std_key": "failure_rate_across_seeds_std",
    },
}


def _seed_from_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    suffix = name.removeprefix("seed_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _discover_available_seeds(
    *,
    outputs_root: Path,
    train_task_setting: str,
    policy_subdir: str,
) -> list[int]:
    seeds: set[int] = set()

    task_root = outputs_root / train_task_setting
    if task_root.exists():
        for seed_dir in task_root.glob("seed_*"):
            if not seed_dir.is_dir():
                continue
            seed = _seed_from_name(seed_dir.name)
            if seed is None:
                continue
            if (seed_dir / policy_subdir).exists():
                seeds.add(seed)

    # Legacy fallback layout: outputs_root/seed_<n>/<policy_subdir>
    if outputs_root.exists():
        for seed_dir in outputs_root.glob("seed_*"):
            if not seed_dir.is_dir():
                continue
            seed = _seed_from_name(seed_dir.name)
            if seed is None:
                continue
            if (seed_dir / policy_subdir).exists():
                seeds.add(seed)

    return sorted(seeds)


def _evaluate_success_metrics(
    *,
    env,
    actor: torch.nn.Module,
    episodes: int,
    device: str,
    seed: int = 0,
    deterministic: bool = True,
) -> tuple[float, float, float, float]:
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}.")

    actor.eval()
    total_rewards: list[float] = []
    successes = 0

    for episode_num in range(episodes):
        obs, _ = env.reset(seed=seed * episode_num)
        done = False
        episodic_reward = 0.0
        episode_success = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                if deterministic:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = env.step(action)
            episodic_reward += float(reward)
            episode_success = episode_success or bool(info.get("is_success", False))
            done = bool(terminated or truncated)

        total_rewards.append(float(episodic_reward))
        if episode_success:
            successes += 1

    total_reward_mean = float(np.mean(total_rewards))
    total_reward_std = float(np.std(total_rewards))
    success_rate = float(successes / episodes)
    failure_rate = float(1.0 - success_rate)
    return total_reward_mean, total_reward_std, success_rate, failure_rate


def _format_mean_std(mean: float | None, std: float | None, precision: int) -> str:
    if mean is None or std is None:
        return "--"
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def _policy_display_name(policy_name: str) -> str:
    return POLICY_DISPLAY_NAMES.get(policy_name, policy_name.replace("_", " ").title())


def _metric_mean_std(aggregate: dict[str, Any] | None, mean_key: str, std_key: str) -> tuple[float | None, float | None]:
    if aggregate is None:
        return None, None
    mean_raw = aggregate.get(mean_key)
    std_raw = aggregate.get(std_key)
    if mean_raw is None or std_raw is None:
        return None, None
    return float(mean_raw), float(std_raw)


def _sanitize_latex_label_fragment(s: str) -> str:
    cleaned = []
    for ch in s.lower():
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif ch == "-":
            cleaned.append("_")
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "env"


def _build_latex_table(
    policy_results: list[dict[str, Any]],
    selected_policies: list[str],
    report_metrics: list[str],
    precision: int,
    env_setting: str,
) -> str:
    aggregates_by_policy_role: dict[tuple[str, str], dict[str, Any]] = {}
    for result in policy_results:
        policy_name = str(result.get("policy_name"))
        env_role = str(result.get("env_role"))
        aggregate = result.get("aggregate_across_seeds")
        if isinstance(aggregate, dict):
            aggregates_by_policy_role[(policy_name, env_role)] = aggregate

    ordered_policies: list[str] = []
    for policy_name in LATEX_POLICY_ROW_ORDER:
        if policy_name in selected_policies:
            ordered_policies.append(policy_name)
    for policy_name in selected_policies:
        if policy_name not in ordered_policies:
            ordered_policies.append(policy_name)

    seed_counts = []
    for result in policy_results:
        aggregate = result.get("aggregate_across_seeds")
        if isinstance(aggregate, dict) and aggregate.get("num_seeds") is not None:
            seed_counts.append(int(aggregate["num_seeds"]))
    num_seeds_text = str(seed_counts[0]) if seed_counts else "0"
    if seed_counts and any(count != seed_counts[0] for count in seed_counts):
        num_seeds_text = f"{min(seed_counts)}-{max(seed_counts)}"

    header_cells = ["Policy"]
    for role_name in ("Source", "Downstream"):
        for metric in report_metrics:
            metric_label = str(REPORT_METRIC_SPECS[metric]["latex_label"])
            header_cells.append(f"{role_name} {metric_label}")

    column_spec = "l" + ("c" * (len(header_cells) - 1))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for policy_name in ordered_policies:
        source_agg = aggregates_by_policy_role.get((policy_name, "source"))
        downstream_agg = aggregates_by_policy_role.get((policy_name, "downstream"))

        row_cells = [
            _policy_display_name(policy_name),
        ]
        for aggregate in (source_agg, downstream_agg):
            for metric in report_metrics:
                metric_spec = REPORT_METRIC_SPECS[metric]
                mean_val, std_val = _metric_mean_std(
                    aggregate,
                    str(metric_spec["mean_key"]),
                    str(metric_spec["std_key"]),
                )
                row_cells.append(_format_mean_std(mean_val, std_val, precision))
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{"
                f"Lunar Lander ({env_setting.replace('_', ' ')}): aggregated metrics across {num_seeds_text} seeds."
                "}"
            ),
            f"\\label{{tab:lunarlander_{_sanitize_latex_label_fragment(env_setting)}_aggregated_metrics}}",
            "\\end{table}",
        ],
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained LunarLander policy and report total reward, "
            "success rate (from info['is_success']), and failure rate."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Seed used during policy training (required unless --all-seeds).",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=(
            "Evaluate all available seeds and aggregate metrics. "
            "If no policy selector is provided, evaluates all policies."
        ),
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        choices=sorted(POLICY_TO_SUBDIR.keys()),
        help="Single policy checkpoint group to evaluate.",
    )
    parser.add_argument(
        "--policy-names",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(POLICY_TO_SUBDIR.keys()),
        help="Multiple policy checkpoint groups to evaluate in one run.",
    )
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Evaluate all known policy groups.",
    )
    parser.add_argument(
        "--all-env-roles",
        action="store_true",
        help="Evaluate each selected policy on both source and downstream tasks.",
    )
    parser.add_argument(
        "--env-setting",
        type=str,
        required=True,
        help="Environment configuration key from LunarLander task settings.",
    )
    parser.add_argument(
        "--train-task-setting",
        type=str,
        default=None,
        help="Task setting key used when training the policy. Defaults to --env-setting.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Evaluation seed.",
    )
    parser.add_argument(
        "--env-role",
        type=str,
        choices=["source", "downstream"],
        default=None,
        help=(
            "Task role to evaluate on. Defaults to source for source policy, "
            "downstream otherwise. Mutually exclusive with --all-env-roles."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic argmax actions (default: True).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task settings YAML with source/downstream environment configs.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Root outputs directory containing <task_setting>/seed_<N> policy folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional metrics YAML output path.",
    )
    parser.add_argument(
        "--latex-output-path",
        type=Path,
        default=None,
        help="Optional LaTeX table output path. Defaults to YAML path with .tex suffix.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        help="Decimal precision for LaTeX table values (default: 2).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["total_reward", "success_rate"],
        choices=sorted(REPORT_METRIC_SPECS.keys()),
        help=(
            "Metrics to report in console and LaTeX outputs. "
            "Default: total_reward success_rate."
        ),
    )
    args = parser.parse_args()

    if int(args.eval_episodes) <= 0:
        raise ValueError(f"--eval-episodes must be > 0, got {args.eval_episodes}.")
    if int(args.round) < 0:
        raise ValueError(f"--round must be >= 0, got {args.round}.")
    if len(args.metrics) == 0:
        raise ValueError("--metrics must include at least one metric.")
    if not args.all_seeds and args.train_seed is None:
        raise ValueError("Provide --train-seed, or use --all-seeds.")
    if args.all_seeds and args.train_seed is not None:
        raise ValueError("Use either --train-seed or --all-seeds, not both.")
    if args.all_policies and (args.policy_name is not None or args.policy_names is not None):
        raise ValueError("Use either --all-policies or --policy-name/--policy-names, not both.")
    if args.policy_name is not None and args.policy_names is not None:
        raise ValueError("Use either --policy-name or --policy-names, not both.")
    if args.all_env_roles and args.env_role is not None:
        raise ValueError("Use either --env-role or --all-env-roles, not both.")

    selected_all_policies = bool(
        args.all_policies
        or (args.all_seeds and args.policy_name is None and args.policy_names is None),
    )

    if selected_all_policies:
        selected_policies = sorted(POLICY_TO_SUBDIR.keys())
    elif args.policy_names is not None:
        selected_policies = list(dict.fromkeys(args.policy_names))
    elif args.policy_name is not None:
        selected_policies = [str(args.policy_name)]
    else:
        raise ValueError(
            "Provide --policy-name/--policy-names, or use --all-policies/--all-seeds.",
        )

    train_task_setting = args.train_task_setting or args.env_setting
    report_metrics = list(dict.fromkeys([str(m) for m in args.metrics]))

    policy_results: list[dict[str, Any]] = []
    for policy_name in selected_policies:
        policy_subdir = POLICY_TO_SUBDIR[policy_name]
        if args.all_env_roles:
            env_roles = ["source", "downstream"]
        else:
            env_role = args.env_role
            if env_role is None:
                env_role = "source" if policy_name == "source" else "downstream"
            env_roles = [env_role]

        if args.all_seeds:
            eval_seeds = _discover_available_seeds(
                outputs_root=args.outputs_root,
                train_task_setting=train_task_setting,
                policy_subdir=policy_subdir,
            )
            if not eval_seeds:
                raise FileNotFoundError(
                    "No seeds found for requested policy under outputs root. "
                    f"Policy='{policy_name}', task_setting='{train_task_setting}'.",
                )
        else:
            eval_seeds = [int(args.train_seed)]

        for env_role in env_roles:
            env_cfg = _load_task_settings(args.task_settings_file, args.env_setting, env_role)
            env_id = str(env_cfg.get("env_id") or "LunarLander-v3")
            gravity_raw = env_cfg.get("gravity")
            gravity = None if gravity_raw is None else float(gravity_raw)
            task_id_default = 0.0 if env_role == "source" else 1.0
            task_id = float(env_cfg.get("task_id", task_id_default))
            append_task_id = bool(env_cfg.get("append_task_id", True))
            continuous = bool(env_cfg.get("continuous", False))
            if continuous:
                raise ValueError("Only discrete-action LunarLander is supported in this evaluator.")
            dynamics_cfg = _resolve_lunarlander_dynamics(
                env_cfg,
                cfg_name=f"task_settings[{args.env_setting}:{env_role}]",
            )
            env_kwargs = {
                "gravity": gravity,
                "task_id": task_id,
                "append_task_id": append_task_id,
                **dynamics_cfg,
            }

            per_seed_results: list[dict[str, Any]] = []
            for seed in eval_seeds:
                policy_dir = _resolve_policy_dir(
                    args.outputs_root,
                    train_task_setting,
                    int(seed),
                    policy_subdir,
                )
                if not policy_dir.exists():
                    raise FileNotFoundError(f"Policy directory does not exist: {policy_dir}")

                actor_path = _resolve_actor_path(policy_dir, policy_name)
                actor_state = torch.load(actor_path, map_location="cpu")
                if not isinstance(actor_state, dict):
                    raise ValueError(f"Expected actor checkpoint state_dict dict at {actor_path}.")

                actor = _build_actor_from_state_dict(actor_state).to(args.device)
                actor.eval()

                eval_env = _make_lunarlander_env(env_id, render_mode=None, **env_kwargs)
                try:
                    total_reward_mean, total_reward_std, success_rate, failure_rate = _evaluate_success_metrics(
                        env=eval_env,
                        actor=actor,
                        episodes=int(args.eval_episodes),
                        deterministic=bool(args.deterministic),
                        device=args.device,
                        seed=int(args.eval_seed),
                    )
                finally:
                    eval_env.close()

                per_seed_results.append(
                    {
                        "train_seed": int(seed),
                        "policy_dir": str(policy_dir),
                        "actor_path": str(actor_path),
                        "total_reward_mean": float(total_reward_mean),
                        "total_reward_std": float(total_reward_std),
                        "success_rate": float(success_rate),
                        "failure_rate": float(failure_rate),
                    },
                )

            reward_means = np.asarray([r["total_reward_mean"] for r in per_seed_results], dtype=np.float64)
            success_rates = np.asarray([r["success_rate"] for r in per_seed_results], dtype=np.float64)
            failure_rates = np.asarray([r["failure_rate"] for r in per_seed_results], dtype=np.float64)
            aggregate = {
                "num_seeds": int(len(per_seed_results)),
                "seeds": [int(r["train_seed"]) for r in per_seed_results],
                "total_reward_mean_across_seeds_mean": float(np.mean(reward_means)),
                "total_reward_mean_across_seeds_std": float(np.std(reward_means)),
                "success_rate_across_seeds_mean": float(np.mean(success_rates)),
                "success_rate_across_seeds_std": float(np.std(success_rates)),
                "failure_rate_across_seeds_mean": float(np.mean(failure_rates)),
                "failure_rate_across_seeds_std": float(np.std(failure_rates)),
            }
            policy_results.append(
                {
                    "policy_name": str(policy_name),
                    "env_role": str(env_role),
                    "env_id": env_id,
                    "task_id": float(task_id),
                    "gravity": gravity,
                    "append_task_id": bool(append_task_id),
                    "dynamics": dynamics_cfg,
                    "all_seeds": bool(args.all_seeds),
                    "train_seed": (int(eval_seeds[0]) if not args.all_seeds else None),
                    "per_seed_results": per_seed_results,
                    "aggregate_across_seeds": aggregate,
                },
            )

    if args.all_seeds:
        eval_dir = args.outputs_root / train_task_setting / "evals"
    else:
        first_policy = policy_results[0]["policy_name"]
        first_seed = int(policy_results[0]["per_seed_results"][0]["train_seed"])
        first_policy_dir = _resolve_policy_dir(
            args.outputs_root,
            train_task_setting,
            first_seed,
            POLICY_TO_SUBDIR[first_policy],  # type: ignore[index]
        )
        eval_dir = first_policy_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if selected_all_policies:
        policy_tag = "all-policies"
    elif len(selected_policies) > 1:
        policy_tag = "multi-policies"
    else:
        policy_tag = f"policy-{_sanitize(selected_policies[0])}"

    if args.all_seeds:
        file_stem = (
            f"success_eval_all_seeds_{policy_tag}"
            f"_env-{_sanitize(args.env_setting)}"
            f"_episodes-{int(args.eval_episodes)}"
        )
    else:
        file_stem = (
            f"success_eval_{policy_tag}"
            f"_env-{_sanitize(args.env_setting)}"
            f"_seed-{int(args.train_seed)}"
            f"_episodes-{int(args.eval_episodes)}"
        )
    if args.all_env_roles:
        file_stem += "_roles-both"
    elif args.env_role is not None:
        file_stem += f"_role-{_sanitize(args.env_role)}"
    output_path = args.output_path or (eval_dir / f"{file_stem}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex_output_path = args.latex_output_path or output_path.with_suffix(".tex")
    latex_output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "policy_names": [str(p) for p in selected_policies],
        "metrics": report_metrics,
        "all_policies": bool(selected_all_policies),
        "all_seeds": bool(args.all_seeds),
        "all_env_roles": bool(args.all_env_roles),
        "env_role": (str(args.env_role) if args.env_role is not None else None),
        "train_seed": (int(args.train_seed) if not args.all_seeds else None),
        "task_settings_file": str(args.task_settings_file),
        "train_task_setting": str(train_task_setting),
        "env_setting": str(args.env_setting),
        "eval_seed": int(args.eval_seed),
        "eval_episodes": int(args.eval_episodes),
        "round": int(args.round),
        "deterministic": bool(args.deterministic),
        "policy_results": policy_results,
    }

    # Backward-compatible aliases for single-policy runs.
    if len(policy_results) == 1:
        single = policy_results[0]
        summary["policy_name"] = str(single["policy_name"])
        summary["env_role"] = str(single["env_role"])
        summary["env_id"] = str(single["env_id"])
        summary["task_id"] = float(single["task_id"])
        summary["gravity"] = single["gravity"]
        summary["append_task_id"] = bool(single["append_task_id"])
        summary["dynamics"] = single["dynamics"]
        summary["per_seed_results"] = single["per_seed_results"]
        summary["aggregate_across_seeds"] = single["aggregate_across_seeds"]

    output_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    latex_table = _build_latex_table(
        policy_results,
        [str(p) for p in selected_policies],
        report_metrics,
        int(args.round),
        str(args.env_setting),
    )
    latex_output_path.write_text(latex_table, encoding="utf-8")

    print(f"Evaluation done for {len(policy_results)} policy/policies.")
    for result in policy_results:
        agg = result["aggregate_across_seeds"]
        metrics_text = []
        for metric in report_metrics:
            metric_spec = REPORT_METRIC_SPECS[metric]
            mean_key = str(metric_spec["mean_key"])
            std_key = str(metric_spec["std_key"])
            metrics_text.append(
                f"{metric}_mean={float(agg[mean_key]):.3f}, {metric}_std={float(agg[std_key]):.3f}",
            )
        metrics_joined = ", ".join(metrics_text)
        print(
            f"  policy={result['policy_name']}, role={result['env_role']}: "
            f"num_seeds={agg['num_seeds']}, {metrics_joined}",
        )
    print(f"Saved metrics: {output_path}")
    print(f"Saved LaTeX table: {latex_output_path}")


if __name__ == "__main__":
    main()
