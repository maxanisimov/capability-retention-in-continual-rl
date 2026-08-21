#!/usr/bin/env python
"""PSPO hyperparameter sweep launcher with disjoint CPU-core allocation.

The sweep is intentionally limited to PSPO-specific hyperparameters:

* ``rashomon_n_iters``;
* ``bc_target_margin``.
* ``safe_region_shape`` / ``zonotope_rank`` for selecting the PSPO safe-region
  geometry.

For precomputed PSPO, ``rashomon_n_iters`` is the offline budget used to build
the fixed Rashomon set.  For adaptive PSPO, it is the per-update/on-demand
Rashomon budget passed to ``train_pspo_adaptive.py``; the base policy artifact
is built separately and does not vary with this value.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings
from projects.safe_policy_optimisation.utils.cpu_allocation import apply_cpu_affinity, parse_cpu_ids
from projects.safe_policy_optimisation.utils.parallel import (
    Job,
    available_cores,
    partition_cores,
    run_core_slot_jobs,
)


REPO = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO / "projects" / "safe_policy_optimisation"
SET_STAGE = PROJECT_ROOT / "stages" / "compute_shield_rashomon_set.py"
PRECOMPUTED_STAGE = PROJECT_ROOT / "stages" / "train_pspo_precomputed.py"
ADAPTIVE_STAGE = PROJECT_ROOT / "stages" / "train_pspo_adaptive.py"

ENV_PIPELINE = {
    "media_streaming": "paper_2503_07671_media_streaming",
    "colour_bomb": "paper_2503_07671_colour_bomb",
    "colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
    "bridge_crossing": "paper_2503_07671_bridge_crossing",
    "bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
    "mini_pacman": "paper_2503_07671_minipacman",
}

METHODS = ("precomputed", "adaptive")
INTERNAL_RUN_FLAG = "--_run-setting"


@dataclass(frozen=True)
class Setting:
    method: str
    rashomon_iters: int
    bc_target_margin: float
    bc_margin_mode: str = "any"
    rashomon_multi_label_mode: str = "any"
    rashomon_surrogate: str = "auto"

    @property
    def tag(self) -> str:
        tag = (
            f"{self.method}/"
            f"iters_{self.rashomon_iters}__margin_{format_tag_float(self.bc_target_margin)}"
        )
        if self.bc_margin_mode != "any":
            tag += f"__bc_{self.bc_margin_mode}"
        if self.rashomon_multi_label_mode != "any":
            tag += f"__cert_{self.rashomon_multi_label_mode}"
        if self.rashomon_surrogate != "auto":
            tag += f"__surrogate_{self.rashomon_surrogate}"
        return tag


@dataclass(frozen=True)
class RuntimePlan:
    settings: list[Setting]
    slots: list[list[int]]
    sweep_root: Path
    use_taskset: bool
    safety_demo_size: int
    config: dict[str, Any]
    n_hidden: int
    hidden_dim: int
    state_representation: str
    seeds: list[int]
    device: str
    adaptive_base_set_iters: int
    safe_region_shape: str
    zonotope_rank: int | None


def format_tag_float(value: float) -> str:
    return ("%g" % float(value)).replace("-", "m").replace(".", "p")


def resolve_pipeline(args: argparse.Namespace) -> str:
    if args.pipeline is not None:
        return str(args.pipeline)
    if args.env not in ENV_PIPELINE:
        raise SystemExit(f"Unknown --env {args.env!r}; valid={sorted(ENV_PIPELINE)}")
    return ENV_PIPELINE[str(args.env)]


def resolve_methods(value: str) -> list[str]:
    if value == "both":
        return list(METHODS)
    if value not in METHODS:
        raise ValueError(f"Unknown method {value!r}; valid={(*METHODS, 'both')}")
    return [value]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def default_sweep_root() -> Path:
    return REPO / "outputs" / "_pspo_hparam" / datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PSPO-only hyperparameter sweep over Rashomon iterations and "
            "BC target logit margin with disjoint CPU-core allocation."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--pipeline", help="Pipeline registry key.")
    source.add_argument("--env", help=f"Short paper env name. Valid: {', '.join(sorted(ENV_PIPELINE))}.")
    parser.add_argument("--method", choices=("precomputed", "adaptive", "both"), default="precomputed")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--rashomon-iters", type=int, nargs="+", default=None)
    parser.add_argument("--bc-target-margins", type=float, nargs="+", default=None)
    parser.add_argument(
        "--bc-margin-mode",
        choices=("any", "all"),
        default="any",
        help=(
            "BC safety-margin semantics for the initial safe policy. 'any' is "
            "the historical best-safe-vs-best-unsafe criterion; 'all' requires "
            "every safe action to beat every unsafe action."
        ),
    )
    parser.add_argument(
        "--rashomon-multi-label-mode",
        choices=("any", "all"),
        default="any",
        help=(
            "Rashomon admissible-set certificate/surrogate. 'any' requires at "
            "least one safe action to beat every unsafe action; 'all' requires "
            "every safe action to beat every unsafe action."
        ),
    )
    parser.add_argument(
        "--rashomon-surrogate",
        choices=("auto", "probability", "logsumexp"),
        default="auto",
        help="Rashomon optimization surrogate form.",
    )
    parser.add_argument(
        "--safe-region-shape",
        choices=("orthotope", "zonotope"),
        default="orthotope",
        help="Safe parameter-region geometry used by PSPO.",
    )
    parser.add_argument(
        "--zonotope-rank",
        type=int,
        default=None,
        help="Number of zonotope generator directions when --safe-region-shape=zonotope.",
    )
    parser.add_argument("--sweep-root", type=Path, default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--cores-per-setting", type=int, default=1)
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=None)
    parser.add_argument("--no-taskset", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--state-representation", choices=("one_hot", "features"), default=None)
    parser.add_argument(
        "--adaptive-base-set-iters",
        type=int,
        default=None,
        help=(
            "Offline set-building budget used only to create adaptive base_policy.pt. "
            "Defaults to the selected pipeline's rashomon_n_iters."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional runtime override for smoke tests; not a swept hyperparameter.",
    )
    parser.add_argument(
        INTERNAL_RUN_FLAG,
        dest="_run_setting",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args._run_setting is None and args.pipeline is None and args.env is None:
        parser.error("one of --pipeline or --env is required")
    return args


def unique_ints(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value <= 0:
            raise SystemExit(f"Iteration counts must be positive, got {value}.")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        value = float(value)
        if value <= 0:
            raise SystemExit(f"BC target margins must be positive, got {value}.")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_settings(
    methods: list[str],
    rashomon_iters: list[int],
    margins: list[float],
    *,
    bc_margin_mode: str = "any",
    rashomon_multi_label_mode: str = "any",
    rashomon_surrogate: str = "auto",
) -> list[Setting]:
    return [
        Setting(
            method=method,
            rashomon_iters=int(n_iters),
            bc_target_margin=float(margin),
            bc_margin_mode=bc_margin_mode,
            rashomon_multi_label_mode=rashomon_multi_label_mode,
            rashomon_surrogate=rashomon_surrogate,
        )
        for method in methods
        for n_iters in rashomon_iters
        for margin in margins
    ]


def resolve_slots(args: argparse.Namespace, n_settings: int) -> list[list[int]]:
    cores = list(args.cpu_ids) if args.cpu_ids is not None else available_cores()
    cores_per_setting = int(args.cores_per_setting)
    if cores_per_setting <= 0:
        raise SystemExit("--cores-per-setting must be positive.")
    max_slots_by_cores = len(cores) // cores_per_setting
    if max_slots_by_cores <= 0:
        raise SystemExit(
            f"Need at least {cores_per_setting} core(s) per setting, but only "
            f"{len(cores)} available: {cores}."
        )
    n_slots = min(n_settings, max_slots_by_cores)
    if args.max_parallel is not None:
        if int(args.max_parallel) <= 0:
            raise SystemExit("--max-parallel must be positive when provided.")
        n_slots = min(n_slots, int(args.max_parallel))
    return partition_cores(cores, max(1, n_slots), cores_per_group=cores_per_setting)


def safety_demo_size(cfg: dict[str, Any], *, shield_path: Path, state_representation: str) -> int:
    """Return the full safety-demonstration dataset size for the selected env."""

    from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
        make_safe_behaviour_payload,
    )
    from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs
    from projects.safe_policy_optimisation.utils.shield import load_shield_mask

    state_to_features = None
    if state_representation == "features":
        from projects.safe_policy_optimisation.utils.safe_crl_bridge import make_custom_masa_env

        feature_env = make_custom_masa_env(
            cfg["env_id"],
            max_episode_steps=None,
            env_kwargs=parse_env_kwargs(cfg.get("env_kwargs")),
        ).unwrapped
        state_to_features = feature_env.state_to_features

    mask = load_shield_mask(shield_path)
    _dataset, metadata = make_safe_behaviour_payload(mask, state_to_features)
    return int(metadata["dataset_size"])


def build_runtime_plan(args: argparse.Namespace) -> RuntimePlan:
    pipeline = resolve_pipeline(args)
    cfg, _pipeline_cfg, _task_cfg = compose_pipeline_settings(pipeline)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)
    methods = resolve_methods(args.method)
    if args.rashomon_iters is None:
        raise SystemExit("--rashomon-iters is required.")
    if args.bc_target_margins is None:
        raise SystemExit("--bc-target-margins is required.")
    settings = build_settings(
        methods,
        unique_ints(list(args.rashomon_iters)),
        unique_floats(list(args.bc_target_margins)),
        bc_margin_mode=str(args.bc_margin_mode),
        rashomon_multi_label_mode=str(args.rashomon_multi_label_mode),
        rashomon_surrogate=str(args.rashomon_surrogate),
    )
    if not settings:
        raise SystemExit("No hyperparameter settings to run.")
    n_hidden = int(args.n_hidden if args.n_hidden is not None else cfg.get("n_hidden", 2))
    hidden_dim = int(args.hidden_dim if args.hidden_dim is not None else cfg.get("hidden_dim", 64))
    state_representation = str(args.state_representation or cfg.get("state_representation", "one_hot"))
    sweep_root = args.sweep_root or default_sweep_root()
    slots = resolve_slots(args, len(settings))
    use_taskset = (not args.no_taskset) and shutil.which("taskset") is not None
    if not args.no_taskset and not use_taskset:
        print("Note: taskset not found; relying on per-stage --cpu-ids affinity.", file=sys.stderr)
    shield_path = resolve_path(cfg["shield_path"])
    demo_size = safety_demo_size(cfg, shield_path=shield_path, state_representation=state_representation)
    return RuntimePlan(
        settings=settings,
        slots=slots,
        sweep_root=sweep_root,
        use_taskset=use_taskset,
        safety_demo_size=demo_size,
        config=cfg,
        n_hidden=n_hidden,
        hidden_dim=hidden_dim,
        state_representation=state_representation,
        seeds=list(dict.fromkeys(int(seed) for seed in args.seeds)),
        device=str(args.device or cfg.get("device", "cpu")),
        adaptive_base_set_iters=int(
            args.adaptive_base_set_iters
            if args.adaptive_base_set_iters is not None
            else cfg.get("rashomon_n_iters", 2000)
        ),
        safe_region_shape=str(args.safe_region_shape),
        zonotope_rank=args.zonotope_rank,
    )


def setting_dir(sweep_root: Path, setting: Setting) -> Path:
    return sweep_root / setting.tag


def set_dir_for(sweep_root: Path, setting: Setting) -> Path:
    return setting_dir(sweep_root, setting) / "set"


def run_root_for(sweep_root: Path, setting: Setting) -> Path:
    return setting_dir(sweep_root, setting) / "runs"


def setting_log_dir(sweep_root: Path, setting: Setting) -> Path:
    return setting_dir(sweep_root, setting) / "logs"


def common_training_args(
    cfg: dict[str, Any],
    *,
    seed: int,
    device: str,
    state_representation: str,
    cpu_ids: list[int],
) -> list[str]:
    return [
        "--env-id",
        str(cfg["env_id"]),
        "--env-kwargs",
        json.dumps(cfg.get("env_kwargs") or {}),
        "--state-representation",
        state_representation,
        "--max-episode-steps",
        str(cfg["max_episode_steps"]),
        "--cost-limit",
        str(cfg["cost_limit"]),
        "--total-timesteps",
        str(cfg["total_timesteps"]),
        "--eval-episodes",
        str(cfg["eval_episodes"]),
        "--seed",
        str(seed),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--n-steps",
        str(cfg["n_steps"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--n-epochs",
        str(cfg["n_epochs"]),
        "--gamma",
        str(cfg["gamma"]),
        "--gae-lambda",
        str(cfg.get("gae_lambda", 0.95)),
        "--clip-range",
        str(cfg.get("clip_range", 0.2)),
        "--ent-coef",
        str(cfg.get("ent_coef", 0.0)),
        "--vf-coef",
        str(cfg.get("vf_coef", 0.5)),
        "--max-grad-norm",
        str(cfg.get("max_grad_norm", 0.5)),
        "--device",
        device,
        "--early-stop-eval-policy",
        "unshielded",
        "--evaluation-policy",
        str(cfg.get("rashomon_evaluation_policy", "unshielded")),
        "--early-stop-eval-freq",
        str(cfg["early_stop_eval_freq"]),
        "--early-stop-eval-episodes",
        str(cfg["early_stop_eval_episodes"]),
        "--early-stop-success-rate",
        str(cfg["early_stop_success_rate"]),
        "--success-reward-threshold",
        str(cfg["success_reward_threshold"]),
        "--curve-eval-freq",
        str(cfg.get("curve_eval_freq", 0)),
        "--curve-eval-episodes",
        str(cfg.get("curve_eval_episodes", 20)),
    ]


def build_set_command(
    payload: dict[str, Any],
    *,
    setting: Setting,
    n_iters: int,
    cpu_ids: list[int],
) -> list[str]:
    cfg = payload["config"]
    set_dir = Path(payload["sweep_root"]) / setting.tag / "set"
    cmd = [
        sys.executable,
        str(SET_STAGE),
        "--shield-path",
        str(resolve_path(cfg["shield_path"])),
        "--env-id",
        str(cfg["env_id"]),
        "--env-kwargs",
        json.dumps(cfg.get("env_kwargs") or {}),
        "--state-representation",
        str(payload["state_representation"]),
        "--output-dir",
        str(set_dir.parent),
        "--run-id",
        set_dir.name,
        "--seed",
        "0",
        "--device",
        str(payload["device"]),
        "--hidden-dim",
        str(payload["hidden_dim"]),
        "--n-hidden",
        str(payload["n_hidden"]),
        "--bc-target-margin",
        str(setting.bc_target_margin),
        "--linear-init-margin",
        str(setting.bc_target_margin),
        "--bc-margin-mode",
        str(setting.bc_margin_mode),
        "--rashomon-multi-label-mode",
        str(setting.rashomon_multi_label_mode),
        "--rashomon-surrogate",
        str(setting.rashomon_surrogate),
        "--safe-region-shape",
        str(payload.get("safe_region_shape", "orthotope")),
        "--rashomon-n-iters",
        str(n_iters),
        "--rashomon-checkpoint",
        str(cfg.get("rashomon_checkpoint", 100)),
        "--rashomon-batch-size",
        str(payload["safety_demo_size"]),
        "--certificate-samples",
        str(cfg.get("certificate_samples", payload["safety_demo_size"])),
    ]
    if payload.get("zonotope_rank") is not None:
        cmd.extend(["--zonotope-rank", str(payload["zonotope_rank"])])
    return cmd


def build_precomputed_train_command(
    payload: dict[str, Any],
    *,
    setting: Setting,
    seed: int,
    cpu_ids: list[int],
) -> list[str]:
    cfg = payload["config"]
    cmd = [
        sys.executable,
        str(PRECOMPUTED_STAGE),
        "--rashomon-dir",
        str(set_dir_for(Path(payload["sweep_root"]), setting)),
        "--shield-path",
        str(resolve_path(cfg["shield_path"])),
        *common_training_args(
            cfg,
            seed=seed,
            device=str(payload["device"]),
            state_representation=str(payload["state_representation"]),
            cpu_ids=cpu_ids,
        ),
        "--output-dir",
        str(run_root_for(Path(payload["sweep_root"]), setting)),
        "--run-id",
        f"seed{seed}",
    ]
    cmd.extend(["--safe-region-shape", str(payload.get("safe_region_shape", "orthotope"))])
    if payload.get("zonotope_rank") is not None:
        cmd.extend(["--zonotope-rank", str(payload["zonotope_rank"])])
    return cmd


def build_adaptive_train_command(
    payload: dict[str, Any],
    *,
    setting: Setting,
    seed: int,
    cpu_ids: list[int],
) -> list[str]:
    cfg = payload["config"]
    cmd = [
        sys.executable,
        str(ADAPTIVE_STAGE),
        "--base-policy-path",
        str(set_dir_for(Path(payload["sweep_root"]), setting) / "base_policy.pt"),
        "--shield-path",
        str(resolve_path(cfg["shield_path"])),
        *common_training_args(
            cfg,
            seed=seed,
            device=str(payload["device"]),
            state_representation=str(payload["state_representation"]),
            cpu_ids=cpu_ids,
        ),
        "--adaptive-granularity",
        "gradient_step",
        "--unsafe-update-strategy",
        "rashomon_project",
        "--rashomon-n-iters",
        str(setting.rashomon_iters),
        "--rashomon-checkpoint",
        str(max(1, int(setting.rashomon_iters) // 10)),
        "--rashomon-batch-size",
        str(payload["safety_demo_size"]),
        "--rashomon-multi-label-mode",
        str(setting.rashomon_multi_label_mode),
        "--rashomon-surrogate",
        str(setting.rashomon_surrogate),
        "--safe-region-shape",
        str(payload.get("safe_region_shape", "orthotope")),
        "--output-dir",
        str(run_root_for(Path(payload["sweep_root"]), setting)),
        "--run-id",
        f"seed{seed}",
    ]
    if payload.get("zonotope_rank") is not None:
        cmd.extend(["--zonotope-rank", str(payload["zonotope_rank"])])
    return cmd


def terminal_for(payload: dict[str, Any], setting: Setting, seed: int | None = None) -> Path:
    if seed is None:
        return set_dir_for(Path(payload["sweep_root"]), setting) / "summary.json"
    return run_root_for(Path(payload["sweep_root"]), setting) / f"seed{seed}" / "metrics.json"


def worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
    )
    return env


def run_subcommand(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    cwd: Path = REPO,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def _flatten_numeric(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def aggregate_metric_files(seed_to_metrics: dict[int, Path]) -> dict[str, Any]:
    per_seed: dict[int, dict[str, float]] = {}
    for seed, metrics_path in seed_to_metrics.items():
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_seed[int(seed)] = _flatten_numeric(payload)
    all_keys = sorted({key for metrics in per_seed.values() for key in metrics})
    aggregate: dict[str, Any] = {}
    for key in all_keys:
        values = {seed: metrics[key] for seed, metrics in per_seed.items() if key in metrics}
        vals = list(values.values())
        mean = sum(vals) / len(vals) if vals else 0.0
        std = math.sqrt(sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        aggregate[key] = {
            "mean": mean,
            "std": std,
            "n": len(vals),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "per_seed": values,
        }
    return {"seeds": sorted(seed_to_metrics), "metrics": aggregate}


def write_aggregate_files(setting_path: Path, aggregate: dict[str, Any]) -> None:
    out_dir = setting_path / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregated_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "aggregated_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std", "n", "min", "max"])
        writer.writeheader()
        for metric, stats in sorted(aggregate.get("metrics", {}).items()):
            writer.writerow(
                {
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n": stats["n"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )


def run_setting_payload(payload: dict[str, Any]) -> int:
    setting = Setting(**payload["setting"])
    force = bool(payload["force"])
    cpu_ids = [int(value) for value in payload["core_ids"]]
    apply_cpu_affinity(cpu_ids)
    env = worker_env()
    setting_path = setting_dir(Path(payload["sweep_root"]), setting)
    logs = setting_path / "logs"

    set_terminal = terminal_for(payload, setting)
    if force or not set_terminal.exists():
        set_iters = (
            int(setting.rashomon_iters)
            if setting.method == "precomputed"
            else int(payload["adaptive_base_set_iters"])
        )
        rc = run_subcommand(
            build_set_command(payload, setting=setting, n_iters=set_iters, cpu_ids=cpu_ids),
            log_path=logs / "set.log",
            env=env,
        )
        if rc != 0:
            return rc

    for seed in payload["seeds"]:
        seed = int(seed)
        terminal = terminal_for(payload, setting, seed)
        if terminal.exists() and not force:
            continue
        if setting.method == "precomputed":
            cmd = build_precomputed_train_command(payload, setting=setting, seed=seed, cpu_ids=cpu_ids)
        else:
            cmd = build_adaptive_train_command(payload, setting=setting, seed=seed, cpu_ids=cpu_ids)
        rc = run_subcommand(cmd, log_path=logs / f"seed{seed}.log", env=env)
        if rc != 0:
            return rc

    aggregate = aggregate_metric_files(
        {
            int(seed): terminal_for(payload, setting, int(seed))
            for seed in payload["seeds"]
        }
    )
    write_aggregate_files(setting_path, aggregate)
    return 0


def encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_payload(value: str) -> dict[str, Any]:
    return json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))


def setting_payload(plan: RuntimePlan, setting: Setting, *, force: bool, core_ids: list[int]) -> dict[str, Any]:
    return {
        "setting": asdict(setting),
        "sweep_root": str(plan.sweep_root),
        "config": plan.config,
        "n_hidden": plan.n_hidden,
        "hidden_dim": plan.hidden_dim,
        "state_representation": plan.state_representation,
        "seeds": plan.seeds,
        "device": plan.device,
        "adaptive_base_set_iters": plan.adaptive_base_set_iters,
        "safe_region_shape": plan.safe_region_shape,
        "zonotope_rank": plan.zonotope_rank,
        "safety_demo_size": plan.safety_demo_size,
        "force": force,
        "core_ids": core_ids,
    }


def make_setting_job(plan: RuntimePlan, setting: Setting, *, force: bool) -> Job:
    def make_command(core_ids: list[int]) -> list[str]:
        payload = setting_payload(plan, setting, force=force, core_ids=core_ids)
        return [sys.executable, str(Path(__file__).resolve()), INTERNAL_RUN_FLAG, encode_payload(payload)]

    return Job(
        name=setting.tag,
        make_command=make_command,
        log_path=setting_log_dir(plan.sweep_root, setting) / "launcher.log",
        meta={"setting": setting},
    )


def _metric_mean(aggregate: dict[str, Any], key: str) -> float:
    return float(aggregate.get("metrics", {}).get(key, {}).get("mean", 0.0))


def _metric_n(aggregate: dict[str, Any], key: str) -> int:
    return int(aggregate.get("metrics", {}).get(key, {}).get("n", 0))


def write_global_ranking(plan: RuntimePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting in plan.settings:
        aggregate_path = setting_dir(plan.sweep_root, setting) / "aggregate" / "aggregated_metrics.json"
        if aggregate_path.exists():
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        else:
            aggregate = {"metrics": {}}
        rows.append(
            {
                "method": setting.method,
                "rashomon_iters": setting.rashomon_iters,
                "bc_target_margin": setting.bc_target_margin,
                "bc_margin_mode": setting.bc_margin_mode,
                "rashomon_multi_label_mode": setting.rashomon_multi_label_mode,
                "rashomon_surrogate": setting.rashomon_surrogate,
                "safe_region_shape": plan.safe_region_shape,
                "zonotope_rank": plan.zonotope_rank,
                "n": _metric_n(aggregate, "reward.mean_total_reward"),
                "reward_mean": _metric_mean(aggregate, "reward.mean_total_reward"),
                "reward_std": float(aggregate.get("metrics", {}).get("reward.mean_total_reward", {}).get("std", 0.0)),
                "safety_rate_mean": _metric_mean(aggregate, "safety.safety_rate"),
                "violation_count_mean": _metric_mean(aggregate, "safety.violation_count"),
                "success_rate_mean": _metric_mean(aggregate, "success.success_rate"),
                "setting_dir": str(setting_dir(plan.sweep_root, setting)),
            }
        )
    rows.sort(key=lambda row: (float(row["reward_mean"]), float(row["safety_rate_mean"]), int(row["n"])), reverse=True)
    out_path = plan.sweep_root / "ranking.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "rashomon_iters",
            "bc_target_margin",
            "bc_margin_mode",
            "rashomon_multi_label_mode",
            "rashomon_surrogate",
            "safe_region_shape",
            "zonotope_rank",
            "n",
            "reward_mean",
            "reward_std",
            "safety_rate_mean",
            "violation_count_mean",
            "success_rate_mean",
            "setting_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_plan(plan: RuntimePlan, *, dry_run: bool) -> None:
    print(f"Sweep root: {plan.sweep_root}")
    print(f"Settings: {len(plan.settings)}")
    print(f"Seeds: {plan.seeds}")
    print(f"Safety demonstration size / Rashomon batch size: {plan.safety_demo_size}")
    print(f"Architecture: n_hidden={plan.n_hidden}, hidden_dim={plan.hidden_dim}, state={plan.state_representation}")
    print(f"Safe region: shape={plan.safe_region_shape}, zonotope_rank={plan.zonotope_rank}")
    print(f"Slots: {len(plan.slots)} disjoint core slot(s)")
    for i, slot in enumerate(plan.slots):
        print(f"  slot {i}: cores {slot}")
    for i, setting in enumerate(plan.settings):
        slot = plan.slots[i % len(plan.slots)]
        print(f"  setting {setting.tag} -> planned slot cores {slot}")
    if dry_run:
        print("Dry run commands:")
        for i, setting in enumerate(plan.settings):
            slot = plan.slots[i % len(plan.slots)]
            job = make_setting_job(plan, setting, force=False)
            cmd = job.make_command(slot)
            prefix = ["taskset", "-c", ",".join(str(core) for core in slot)] if plan.use_taskset else []
            print("  $ " + " ".join(prefix + cmd))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._run_setting is not None:
        return run_setting_payload(decode_payload(args._run_setting))

    plan = build_runtime_plan(args)
    print_plan(plan, dry_run=bool(args.dry_run))
    if args.dry_run:
        print("Dry run - nothing launched.")
        return 0

    jobs = [make_setting_job(plan, setting, force=bool(args.force)) for setting in plan.settings]
    results = run_core_slot_jobs(
        jobs,
        slots=plan.slots,
        env=worker_env(),
        use_taskset=plan.use_taskset,
    )
    rows = write_global_ranking(plan)
    if rows:
        best = rows[0]
        print(
            "Best setting: method={method} iters={rashomon_iters} margin={bc_target_margin} "
            "reward={reward_mean:.4f} safety={safety_rate_mean:.4f} n={n}".format(**best)
        )
    failed = {name: rc for name, rc in results.items() if rc != 0}
    if failed:
        print(f"PSPO_HPARAM_SWEEP_DONE failed={len(failed)} ranking={plan.sweep_root / 'ranking.csv'}")
        return 1
    print(f"PSPO_HPARAM_SWEEP_DONE failed=0 ranking={plan.sweep_root / 'ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
