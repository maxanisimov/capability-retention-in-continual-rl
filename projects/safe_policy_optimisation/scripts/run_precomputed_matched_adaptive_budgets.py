#!/usr/bin/env python
"""Run precomputed PSPO with per-seed budgets measured from adaptive PSPO v2.

Each seed gets its own static safe parameter region. The region is centred on
the exact base policy used by the adaptive run and receives exactly that
seed's recorded ``rashomon_iters_spent`` optimization iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SET_STAGE = REPO / "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py"
TRAIN_STAGE = REPO / "projects/safe_policy_optimisation/stages/train_pspo_precomputed.py"


@dataclass(frozen=True)
class AdaptiveSeedSpec:
    seed: int
    rashomon_iters: int
    config: dict[str, Any]
    summary_path: Path


@dataclass(frozen=True)
class BasePolicyMetadata:
    path: Path
    sha256: str
    architecture: dict[str, Any]
    target_margin: float
    margin_mode: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cpu_ids(value: str) -> list[int]:
    """Parse comma-separated CPU ids and inclusive ranges such as ``31-40``."""

    cpu_ids: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(f"Invalid CPU range {part!r}.")
            values = range(start, end + 1)
        else:
            value_int = int(part)
            if value_int < 0:
                raise argparse.ArgumentTypeError("CPU ids must be non-negative.")
            values = [value_int]
        for cpu_id in values:
            if cpu_id not in seen:
                seen.add(cpu_id)
                cpu_ids.append(cpu_id)
    if not cpu_ids:
        raise argparse.ArgumentTypeError("At least one CPU id is required.")
    return cpu_ids


def load_adaptive_seed_spec(adaptive_run_dir: Path, seed: int) -> AdaptiveSeedSpec:
    seed_dir = adaptive_run_dir / f"seed{seed}"
    config = _read_json(seed_dir / "config.json")
    summary_path = seed_dir / "summary.json"
    summary = _read_json(summary_path)
    try:
        recorded_seed = int(config["seed"])
        budget = int(summary["adaptive_diagnostics"]["rashomon_iters_spent"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Seed {seed} is missing a valid seed or adaptive Rashomon iteration total."
        ) from exc
    if recorded_seed != int(seed):
        raise ValueError(
            f"Seed directory seed{seed} contains config seed={recorded_seed}."
        )
    if budget <= 0:
        raise ValueError(f"Seed {seed} has non-positive Rashomon budget {budget}.")
    algorithm = config.get("algorithm")
    adaptive = config.get("adaptive") or {}
    if algorithm != "adaptive_safe_ppo_v2" and not (
        algorithm == "pspo_adaptive" and not adaptive.get("verify_first", False)
    ):
        raise ValueError(
            f"Seed {seed} is not a region-first PSPO adaptive result: "
            f"algorithm={config.get('algorithm')!r}."
        )
    return AdaptiveSeedSpec(
        seed=int(seed),
        rashomon_iters=budget,
        config=config,
        summary_path=summary_path.resolve(),
    )


def _comparison_signature(config: dict[str, Any]) -> dict[str, Any]:
    adaptive = dict(config["adaptive"])
    return {
        "base_policy_path": str(_resolve_path(config["base_policy_path"])),
        "shield_path": str(_resolve_path(config["shield_path"])),
        "env_id": config["env_id"],
        "env_kwargs": config.get("env_kwargs") or {},
        "state_representation": config["base_policy_architecture"]["state_representation"],
        "base_policy_architecture": config["base_policy_architecture"],
        "cost_limit": config["cost_limit"],
        "total_timesteps": config["total_timesteps"],
        "eval_episodes": config["eval_episodes"],
        "evaluation_policy": config["evaluation_policy"],
        "training_hyperparameters": config["training_hyperparameters"],
        "rashomon_checkpoint": adaptive["rashomon_checkpoint"],
        "rashomon_batch_size": adaptive["rashomon_batch_size"],
        "certificate_samples": adaptive["certificate_samples"],
        "rashomon_multi_label_mode": adaptive["rashomon_multi_label_mode"],
        "rashomon_surrogate": adaptive["rashomon_surrogate"],
        "safe_region_shape": adaptive["safe_region_shape"],
    }


def validate_shared_settings(specs: list[AdaptiveSeedSpec]) -> None:
    if not specs:
        raise ValueError("At least one adaptive seed is required.")
    reference = _comparison_signature(specs[0].config)
    for spec in specs[1:]:
        current = _comparison_signature(spec.config)
        if current != reference:
            differing = sorted(
                key for key in set(reference) | set(current)
                if reference.get(key) != current.get(key)
            )
            raise ValueError(
                f"Adaptive seed {spec.seed} does not share the comparison settings; "
                f"differing fields={differing}."
            )


def load_base_policy_metadata(
    specs: list[AdaptiveSeedSpec],
    override: Path | None,
) -> BasePolicyMetadata:
    config = specs[0].config
    path = _resolve_path(override or config["base_policy_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing base policy: {path}")

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = dict(payload.get("architecture") or {})
    expected_architecture = dict(config["base_policy_architecture"])
    if architecture != expected_architecture:
        raise ValueError(
            "Base policy architecture does not match the adaptive run: "
            f"expected={expected_architecture}, actual={architecture}."
        )
    metrics = dict(payload.get("bc_metrics") or {})
    target_margin = float(metrics.get("target_margin", 0.0))
    if target_margin <= 0:
        raise ValueError(
            "The base policy does not record a positive bc_metrics.target_margin."
        )
    margin_mode = str(
        metrics.get("bc_margin_mode", config["adaptive"]["rashomon_multi_label_mode"])
    )
    if margin_mode not in {"any", "all"}:
        raise ValueError(f"Unsupported base-policy margin mode {margin_mode!r}.")
    return BasePolicyMetadata(
        path=path,
        sha256=_file_sha256(path),
        architecture=architecture,
        target_margin=target_margin,
        margin_mode=margin_mode,
    )


def _state_representation_for_cli(architecture: dict[str, Any]) -> str:
    representation = architecture.get("state_representation")
    mapping = {
        "one_hot": "one_hot",
        "one_hot_discrete_observation": "one_hot",
        "features": "features",
        "decoded_features": "features",
    }
    try:
        return mapping[str(representation)]
    except KeyError as exc:
        raise ValueError(f"Unsupported base-policy state representation {representation!r}.") from exc


def safe_set_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "safe_sets" / f"seed{seed}"


def run_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "runs" / f"seed{seed}"


def build_safe_set_command(
    spec: AdaptiveSeedSpec,
    *,
    base_policy: BasePolicyMetadata,
    output_dir: Path,
    python: str,
) -> list[str]:
    config = spec.config
    adaptive = config["adaptive"]
    architecture = base_policy.architecture
    command = [
        python,
        str(SET_STAGE),
        "--base-policy-path",
        str(base_policy.path),
        "--shield-path",
        str(_resolve_path(config["shield_path"])),
        "--env-id",
        str(config["env_id"]),
        "--env-kwargs",
        json.dumps(config.get("env_kwargs") or {}, sort_keys=True),
        "--state-representation",
        _state_representation_for_cli(architecture),
        "--output-dir",
        str((output_dir / "safe_sets").resolve()),
        "--run-id",
        f"seed{spec.seed}",
        "--seed",
        str(spec.seed),
        "--device",
        "cpu",
        "--hidden-dim",
        str(architecture["hidden_dim"]),
        "--n-hidden",
        str(architecture["n_hidden"]),
        "--bc-target-margin",
        str(base_policy.target_margin),
        "--linear-init-margin",
        str(base_policy.target_margin),
        "--bc-margin-mode",
        base_policy.margin_mode,
        "--rashomon-n-iters",
        str(spec.rashomon_iters),
        "--rashomon-checkpoint",
        str(adaptive["rashomon_checkpoint"]),
        "--rashomon-batch-size",
        str(adaptive["rashomon_batch_size"]),
        "--certificate-samples",
        str(adaptive["certificate_samples"]),
        "--safe-region-shape",
        str(adaptive["safe_region_shape"]),
        "--rashomon-multi-label-mode",
        str(adaptive["rashomon_multi_label_mode"]),
        "--rashomon-surrogate",
        str(adaptive["rashomon_surrogate"]),
        "--growth-method",
        "IBP",
        "--certification-method",
        "IBP",
    ]
    if config.get("risk_threshold") is not None:
        command.extend(["--risk-threshold", str(config["risk_threshold"])])
    return command


def build_train_command(
    spec: AdaptiveSeedSpec,
    *,
    base_policy: BasePolicyMetadata,
    output_dir: Path,
    python: str,
) -> list[str]:
    config = spec.config
    hp = config["training_hyperparameters"]
    command = [
        python,
        str(TRAIN_STAGE),
        "--rashomon-dir",
        str(safe_set_dir(output_dir, spec.seed).resolve()),
        "--safe-region-shape",
        str(config["adaptive"]["safe_region_shape"]),
        "--shield-path",
        str(_resolve_path(config["shield_path"])),
        "--env-id",
        str(config["env_id"]),
        "--env-kwargs",
        json.dumps(config.get("env_kwargs") or {}, sort_keys=True),
        "--state-representation",
        _state_representation_for_cli(base_policy.architecture),
        "--max-episode-steps",
        str(config["max_episode_steps"]),
        "--shield-key",
        str(config["shield_key"]),
        "--shield-source",
        str(config["shield_source"]),
        "--shield-action-storage",
        str(config["shield_action_storage"]),
        "--cost-limit",
        str(config["cost_limit"]),
        "--total-timesteps",
        str(config["total_timesteps"]),
        "--eval-episodes",
        str(config["eval_episodes"]),
        "--seed",
        str(spec.seed),
        "--learning-rate",
        str(hp["learning_rate"]),
        "--n-steps",
        str(hp["n_steps"]),
        "--batch-size",
        str(hp["batch_size"]),
        "--n-epochs",
        str(hp["n_epochs"]),
        "--gamma",
        str(hp["gamma"]),
        "--gae-lambda",
        str(hp["gae_lambda"]),
        "--clip-range",
        str(hp["clip_range"]),
        "--ent-coef",
        str(hp["ent_coef"]),
        "--vf-coef",
        str(hp["vf_coef"]),
        "--max-grad-norm",
        str(hp["max_grad_norm"]),
        "--device",
        "cpu",
        "--early-stop-eval-freq",
        str(config["early_stop_eval_freq"]),
        "--early-stop-eval-episodes",
        str(config["early_stop_eval_episodes"]),
        "--early-stop-success-rate",
        str(config["early_stop_success_rate"]),
        "--early-stop-eval-policy",
        str(config["early_stop_eval_policy"]),
        "--evaluation-policy",
        str(config["evaluation_policy"]),
        "--success-reward-threshold",
        str(config["success_reward_threshold"]),
        "--curve-eval-freq",
        str(config["curve_eval_freq"]),
        "--curve-eval-episodes",
        str(config["curve_eval_episodes"]),
        "--output-dir",
        str((output_dir / "runs").resolve()),
        "--run-id",
        f"seed{spec.seed}",
    ]
    if config.get("risk_threshold") is not None:
        command.extend(["--risk-threshold", str(config["risk_threshold"])])
    return command


def safe_set_is_reusable(
    spec: AdaptiveSeedSpec,
    *,
    base_policy: BasePolicyMetadata,
    output_dir: Path,
) -> bool:
    directory = safe_set_dir(output_dir, spec.seed)
    required = [
        directory / "summary.json",
        directory / "base_policy.pt",
        directory / "rashomon_param_bounds.pt",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = _read_json(directory / "summary.json")
        source = summary["base_policy_source"]
        rashomon = summary["rashomon"]
        adaptive = spec.config["adaptive"]
        return (
            int(rashomon["n_iters"]) == spec.rashomon_iters
            and int(rashomon["checkpoint"])
            == int(adaptive["rashomon_checkpoint"])
            and int(rashomon["batch_size"]) == int(adaptive["rashomon_batch_size"])
            and int(rashomon["certificate_samples"])
            == int(adaptive["certificate_samples"])
            and str(rashomon["safe_region_shape"]) == str(adaptive["safe_region_shape"])
            and str(rashomon["multi_label_mode"])
            == str(adaptive["rashomon_multi_label_mode"])
            and str(rashomon["surrogate"]) == str(adaptive["rashomon_surrogate"])
            and str(rashomon["growth_method"]) == "IBP"
            and str(rashomon["certification_method"]) == "IBP"
            and str(source["sha256"]) == base_policy.sha256
        )
    except (KeyError, TypeError, ValueError):
        return False


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        MPLCONFIGDIR="/tmp/matplotlib-pspo-matched",
    )
    return environment


def _run_logged(command: list[str], log_path: Path, *, core: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_command = command
    if shutil.which("taskset") is not None:
        full_command = ["taskset", "-c", str(core), *command]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(full_command) + "\n\n")
        handle.flush()
        return subprocess.run(
            full_command,
            cwd=REPO,
            env=_worker_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def _write_manifest(
    output_dir: Path,
    specs: list[AdaptiveSeedSpec],
    base_policy: BasePolicyMetadata,
) -> None:
    payload = {
        "comparison": "precomputed_pspo_matched_to_adaptive_v2_iterations",
        "adaptive_run_dir": str(specs[0].summary_path.parent.parent),
        "base_policy": {
            "path": str(base_policy.path),
            "sha256": base_policy.sha256,
            "architecture": base_policy.architecture,
            "target_margin": base_policy.target_margin,
            "margin_mode": base_policy.margin_mode,
        },
        "per_seed_rashomon_iterations": {
            str(spec.seed): spec.rashomon_iters for spec in specs
        },
        "adaptive_only_settings_not_applied": [
            "directional_rashomon_growth",
            "region_update_mode",
            "stop_when_proposal_contained",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one precomputed PSPO safe region per seed using the exact "
            "Rashomon iteration total recorded by an adaptive PSPO v2 run."
        )
    )
    parser.add_argument("--adaptive-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-policy-path", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--cpu-ids",
        type=parse_cpu_ids,
        required=True,
        help="Comma-separated CPU ids/ranges, for example 31-40.",
    )
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adaptive_run_dir = _resolve_path(args.adaptive_run_dir)
    output_dir = _resolve_path(args.output_dir)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if any(seed < 0 for seed in seeds):
        raise SystemExit("Seeds must be non-negative.")
    specs = [load_adaptive_seed_spec(adaptive_run_dir, seed) for seed in seeds]
    validate_shared_settings(specs)
    base_policy = load_base_policy_metadata(specs, args.base_policy_path)

    max_parallel = len(args.cpu_ids) if args.max_parallel is None else int(args.max_parallel)
    if max_parallel <= 0:
        raise SystemExit("--max-parallel must be positive.")
    max_parallel = min(max_parallel, len(args.cpu_ids), len(specs))

    print(f"Adaptive source: {adaptive_run_dir}")
    print(f"Output: {output_dir}")
    print(f"Base policy: {base_policy.path}")
    print(f"Base policy SHA-256: {base_policy.sha256}")
    print(f"CPU ids: {args.cpu_ids}; max parallel: {max_parallel}")
    print("Per-seed Rashomon iteration budgets:")
    for spec in specs:
        print(f"  seed{spec.seed}: {spec.rashomon_iters}")

    if args.dry_run:
        for index, spec in enumerate(specs):
            core = args.cpu_ids[index % max_parallel]
            set_command = build_safe_set_command(
                spec,
                base_policy=base_policy,
                output_dir=output_dir,
                python=sys.executable,
            )
            train_command = build_train_command(
                spec,
                base_policy=base_policy,
                output_dir=output_dir,
                python=sys.executable,
            )
            print(f"dry-run seed{spec.seed} core={core} safe-set: {shlex.join(set_command)}")
            print(f"dry-run seed{spec.seed} core={core} train: {shlex.join(train_command)}")
        return 0

    _write_manifest(output_dir, specs, base_policy)
    core_queue: queue.Queue[int] = queue.Queue()
    for core in args.cpu_ids[:max_parallel]:
        core_queue.put(core)

    def run_seed(spec: AdaptiveSeedSpec) -> tuple[int, int, int, float]:
        core = core_queue.get()
        started = time.perf_counter()
        try:
            safe_set_recomputed = False
            if args.force or not safe_set_is_reusable(
                spec,
                base_policy=base_policy,
                output_dir=output_dir,
            ):
                set_command = build_safe_set_command(
                    spec,
                    base_policy=base_policy,
                    output_dir=output_dir,
                    python=sys.executable,
                )
                rc = _run_logged(
                    set_command,
                    output_dir / "logs" / f"seed{spec.seed}_safe_set.log",
                    core=core,
                )
                if rc != 0:
                    return spec.seed, core, rc, time.perf_counter() - started
                safe_set_recomputed = True

            metrics_path = run_dir(output_dir, spec.seed) / "metrics.json"
            if args.force or safe_set_recomputed or not metrics_path.is_file():
                train_command = build_train_command(
                    spec,
                    base_policy=base_policy,
                    output_dir=output_dir,
                    python=sys.executable,
                )
                rc = _run_logged(
                    train_command,
                    output_dir / "logs" / f"seed{spec.seed}_train.log",
                    core=core,
                )
                if rc != 0:
                    return spec.seed, core, rc, time.perf_counter() - started
            return spec.seed, core, 0, time.perf_counter() - started
        finally:
            core_queue.put(core)

    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(run_seed, spec) for spec in specs]
        for future in as_completed(futures):
            seed, core, returncode, elapsed = future.result()
            status = "ok" if returncode == 0 else f"FAIL rc={returncode}"
            print(f"{status} seed{seed} core={core} {elapsed:.0f}s", flush=True)
            if returncode != 0:
                failures.append(seed)

    if failures:
        print(f"Failed seeds: {sorted(failures)}", file=sys.stderr)
        return 1
    print("All matched-budget precomputed PSPO seeds completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
