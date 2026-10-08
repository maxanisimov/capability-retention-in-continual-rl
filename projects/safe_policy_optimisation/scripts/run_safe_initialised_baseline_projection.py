#!/usr/bin/env python
"""Train safe-initialised baselines and conditionally project their final actors.

Each job is one ``(environment, method, seed)`` tuple.  Its actor is
warm-started from the exact base policy stored beside a fixed, certified LID.
After ordinary baseline training, the nominal greedy actor is exhaustively
audited against the tabular shield.  Unsafe actors are projected once into the
fixed orthotope and audited again; already-safe actors are left unchanged.

The default command is a local wave scheduler.  It measures per-logical-CPU
idle time with ``mpstat``, pins each live job to a distinct admitted CPU, and
runs one environment at a time.  ``--worker`` is the internal single-job entry
point and is also useful for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-matplotlib-cache")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(REPO))

from provably_safe_policy_optimisation.projection import (  # noqa: E402
    project_to_interval_union,
    validate_and_prepare_param_interval_bounds,
)

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_baselines_pspo_init_ablation as baseline_suite,
)
from projects.safe_policy_optimisation.scripts.compare_pspo_initial_final_rewards import (  # noqa: E402
    BasePolicyPredictor,
    _evaluate,
    exhaustive_shield_alignment,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (  # noqa: E402
    build_base_policy,
)
from projects.safe_policy_optimisation.stages.train_pspo_precomputed import (  # noqa: E402
    _base_parameter_names,
    _base_to_ppo_actor_name_map,
)
from projects.safe_policy_optimisation.utils.shield import (  # noqa: E402
    load_shield_mask,
)

RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_LID_ROOT = RUNS / "static_lid_ablation_masa_matched"
DEFAULT_OUTPUT_ROOT = RUNS / "safe_initialised_baseline_projection"
ENVIRONMENTS = baseline_suite.ENVIRONMENTS
METHODS = baseline_suite.DEFAULT_METHODS
METHOD_PRIORITY = (
    "ppo",
    "ppo_lagrangian",
    "ppo_pid_lagrangian",
    "cpo",
    "ppo_shield",
)
CUSTOM_METHODS = {"ppo_lagrangian", "ppo_pid_lagrangian", "cpo"}
SB3_METHODS = {"ppo", "ppo_shield"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}
SOURCE_FILES = (
    Path(__file__).relative_to(REPO),
    Path("projects/safe_policy_optimisation/scripts/run_baselines_pspo_init_ablation.py"),
    Path("projects/safe_policy_optimisation/scripts/compare_pspo_initial_final_rewards.py"),
    Path("projects/safe_policy_optimisation/utils/warm_start.py"),
    Path("projects/safe_policy_optimisation/stages/train_ppo.py"),
    Path("projects/safe_policy_optimisation/stages/train_ppo_lagrangian.py"),
    Path("projects/safe_policy_optimisation/stages/train_cpo.py"),
    Path("projects/safe_policy_optimisation/stages/train_ppo_shield.py"),
    Path("core/provably_safe_policy_optimisation/projection.py"),
)


@dataclass(frozen=True)
class LidRecord:
    environment: str
    seed: int
    directory: Path
    base_policy_path: Path
    bounds_path: Path
    shield_path: Path
    architecture: dict[str, Any]
    iterations: int
    base_sha256: str
    bounds_sha256: str
    shield_sha256: str


@dataclass(frozen=True)
class Job:
    environment: str
    method: str
    seed: int
    cpu: int
    run_dir: Path
    log_path: Path
    command: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _manifest_iteration(manifest: dict[str, Any], seed: int) -> int:
    iterations = manifest.get("per_seed_rashomon_iterations")
    if not isinstance(iterations, dict) or str(seed) not in iterations:
        raise KeyError(f"Manifest has no LID iteration budget for seed {seed}.")
    return int(iterations[str(seed)])


def load_lid_record(lid_root: Path, environment: str, seed: int) -> LidRecord:
    environment_dir = lid_root / environment
    directory = environment_dir / "safe_sets" / f"seed{seed}"
    manifest_path = environment_dir / "comparison_manifest.json"
    summary_path = directory / "summary.json"
    base_path = directory / "base_policy.pt"
    bounds_path = directory / "rashomon_param_bounds.pt"
    for required in (manifest_path, summary_path, base_path, bounds_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    if manifest.get("comparison") != "precomputed_pspo_matched_to_adaptive_v2_iterations":
        raise ValueError(f"Unexpected matched-budget manifest in {manifest_path}.")
    iterations = _manifest_iteration(manifest, seed)
    rashomon = dict(summary.get("rashomon") or {})
    if int(rashomon.get("iterations_run", -1)) != iterations:
        raise ValueError(
            f"LID iteration mismatch for {environment}/seed{seed}: "
            f"manifest={iterations}, summary={rashomon.get('iterations_run')}."
        )
    if rashomon.get("safe_region_shape") != "orthotope":
        raise ValueError(f"The fixed LID is not an orthotope: {directory}.")
    if float(rashomon.get("selected_certificate", 0.0)) < 1.0:
        raise ValueError(f"The selected LID is not fully certified: {directory}.")
    if rashomon.get("multi_label_mode") != "all":
        raise ValueError(f"Expected all-safe-action certification: {directory}.")
    if float(rashomon.get("min_all_safe_margin", -math.inf)) < 0.0:
        raise ValueError(f"The selected LID has a negative safety margin: {directory}.")

    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    architecture = dict(base_payload.get("architecture") or {})
    if architecture != dict(manifest["base_policy"]["architecture"]):
        raise ValueError(f"Base-policy architecture mismatch in {directory}.")
    base_hash = sha256(base_path)
    expected_base_hash = str(manifest["base_policy"]["sha256"])
    source_hash = str((summary.get("base_policy_source") or {}).get("sha256", ""))
    source_path = resolve_recorded_path(manifest["base_policy"]["path"])
    if not source_path.is_file() or sha256(source_path) != expected_base_hash:
        raise ValueError(f"Canonical base-policy hash mismatch for {directory}.")
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = dict(source_payload.get("state_dict") or {})
    copied_state = dict(base_payload.get("state_dict") or {})
    state_matches = source_state.keys() == copied_state.keys() and all(
        torch.equal(source_state[name], copied_state[name]) for name in source_state
    )
    if (
        source_hash != expected_base_hash
        or dict(source_payload.get("architecture") or {}) != architecture
        or not state_matches
    ):
        raise ValueError(
            f"Seed-local base policy is not tensor-identical to the canonical policy in {directory}."
        )

    shield_path = resolve_recorded_path(summary["shield_path"])
    if not shield_path.is_file():
        raise FileNotFoundError(shield_path)
    shield_hash = sha256(shield_path)
    if shield_hash != str(summary["shield_sha256"]):
        raise ValueError(f"Shield hash mismatch for {directory}.")

    return LidRecord(
        environment=environment,
        seed=int(seed),
        directory=directory.resolve(),
        base_policy_path=base_path.resolve(),
        bounds_path=bounds_path.resolve(),
        shield_path=shield_path.resolve(),
        architecture=architecture,
        iterations=iterations,
        base_sha256=base_hash,
        bounds_sha256=sha256(bounds_path),
        shield_sha256=shield_hash,
    )


def make_policy(architecture: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    policy = build_base_policy(
        int(architecture["input_dim"]),
        int(architecture["n_actions"]),
        hidden_dim=int(architecture["hidden_dim"]),
        n_hidden=int(architecture["n_hidden"]),
    )
    policy.load_state_dict(state_dict, strict=True)
    return policy.eval()


def checkpoint_path(method: str, run_dir: Path) -> Path:
    if method in SB3_METHODS:
        return run_dir / "model.zip"
    if method in CUSTOM_METHODS:
        return run_dir / f"{method}.pt"
    raise ValueError(f"Unknown method {method!r}.")


def load_actor_state(
    method: str,
    checkpoint: Path,
    architecture: dict[str, Any],
) -> dict[str, torch.Tensor]:
    names = _base_parameter_names(architecture)
    if method in CUSTOM_METHODS:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_state = dict(payload["actor_state_dict"])
        missing = sorted(set(names) - set(raw_state))
        if missing:
            raise KeyError(f"Custom actor is missing parameters: {missing}.")
        state = {name: raw_state[name].detach().cpu().clone() for name in names}
    elif method in SB3_METHODS:
        mapping = _base_to_ppo_actor_name_map(architecture)
        with zipfile.ZipFile(checkpoint) as archive:
            with archive.open("policy.pth") as handle:
                raw_state = torch.load(handle, map_location="cpu", weights_only=False)
        missing = sorted(set(mapping.values()) - set(raw_state))
        if missing:
            raise KeyError(f"SB3 actor is missing parameters: {missing}.")
        state = {
            base_name: raw_state[sb3_name].detach().cpu().clone()
            for base_name, sb3_name in mapping.items()
        }
    else:
        raise ValueError(f"Unknown method {method!r}.")
    # A strict reconstruction catches both shape and unexpected-layout errors.
    make_policy(architecture, state)
    return state


def _inside_box(
    parameters: Sequence[torch.nn.Parameter],
    lower: Sequence[torch.Tensor],
    upper: Sequence[torch.Tensor],
) -> bool:
    return all(
        bool(torch.all(parameter.detach().cpu() >= lb.detach().cpu()).item())
        and bool(torch.all(parameter.detach().cpu() <= ub.detach().cpu()).item())
        for parameter, lb, ub in zip(parameters, lower, upper, strict=True)
    )


def audit_and_project(
    policy: torch.nn.Module,
    architecture: dict[str, Any],
    bounds_payload: dict[str, Any],
    shield_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    predictor = BasePolicyPredictor(policy, input_dim=int(architecture["input_dim"]))
    before = exhaustive_shield_alignment(
        predictor, shield_mask, input_dim=int(architecture["input_dim"])
    )
    parameters = list(policy.parameters())
    lower_sets, upper_sets = validate_and_prepare_param_interval_bounds(
        actor_params=parameters,
        actor_param_bounds_l=bounds_payload["param_bounds_l"],
        actor_param_bounds_u=bounds_payload["param_bounds_u"],
        device=torch.device("cpu"),
    )
    inside_before = any(
        _inside_box(parameters, lower, upper)
        for lower, upper in zip(lower_sets, upper_sets, strict=True)
    )
    applied = int(before["unsafe_states"]) > 0
    if applied:
        projection = asdict(project_to_interval_union(parameters, lower_sets, upper_sets))
    else:
        projection = {
            "n_projected": 0,
            "n_boundary": 0,
            "selected_set_index": None,
            "displacement_l2": 0.0,
            "displacement_linf": 0.0,
        }
    after = exhaustive_shield_alignment(
        predictor, shield_mask, input_dim=int(architecture["input_dim"])
    )
    inside_after = any(
        _inside_box(parameters, lower, upper)
        for lower, upper in zip(lower_sets, upper_sets, strict=True)
    )
    if int(after["unsafe_states"]) != 0:
        raise RuntimeError(f"Deployed actor remains unsafe after conditional projection: {after}.")
    if applied and not inside_after:
        raise RuntimeError("Projected actor is not a member of the fixed LID.")
    diagnostics = {
        "applied": applied,
        "reason": "unsafe_exact_actor" if applied else "already_safe_exact_actor",
        "inside_lid_before": inside_before,
        "inside_lid_after": inside_after,
        **projection,
    }
    return before, after, diagnostics


def _mean_reward(evaluation: dict[str, Any]) -> float:
    rewards = [float(value) for value in evaluation.get("rewards", [])]
    if not rewards:
        raise ValueError("Evaluation returned no episode rewards.")
    return float(statistics.fmean(rewards))


def training_command(
    environment: str,
    method: str,
    seed: int,
    *,
    output_root: Path,
    lid: LidRecord,
    smoke: bool,
) -> tuple[list[str], Path]:
    config = baseline_suite.reference_config(environment, seed)
    hp = config["training_hyperparameters"]
    architecture = config["base_policy_architecture"]
    if dict(architecture) != lid.architecture:
        raise ValueError(
            f"Training/LID architecture mismatch for {environment}/seed{seed}."
        )
    script, extra = baseline_suite.METHODS[method]
    run_dir = output_root / environment / method / f"seed{seed}"
    command = [
        str(REPO / ".venv/bin/python"),
        str(baseline_suite.STAGES / script),
        "--env-id", str(config["env_id"]),
        "--env-kwargs", json.dumps(config.get("env_kwargs") or {}, sort_keys=True),
        "--max-episode-steps", str(config["max_episode_steps"]),
        "--cost-limit", str(config["cost_limit"]),
        "--total-timesteps", str(2048 if smoke else int(config["total_timesteps"])),
        "--eval-episodes", str(2 if smoke else int(config["eval_episodes"])),
        "--seed", str(seed),
        "--learning-rate", str(hp["learning_rate"]),
        "--n-steps", str(8 if smoke else hp["n_steps"]),
        "--batch-size", str(8 if smoke else hp["batch_size"]),
        "--n-epochs", str(1 if smoke else hp["n_epochs"]),
        "--gamma", str(hp["gamma"]),
        "--gae-lambda", str(hp["gae_lambda"]),
        "--clip-range", str(hp["clip_range"]),
        "--ent-coef", str(hp["ent_coef"]),
        "--vf-coef", str(hp["vf_coef"]),
        "--max-grad-norm", str(hp["max_grad_norm"]),
        "--n-hidden", str(architecture["n_hidden"]),
        "--hidden-dim", str(architecture["hidden_dim"]),
        "--device", "cpu",
        "--success-reward-threshold", str(config["success_reward_threshold"]),
        "--output-dir", str(run_dir.parent),
        "--run-id", run_dir.name,
        "--init-policy-path", str(lid.base_policy_path),
        *extra,
    ]
    if method == "ppo_shield":
        command.extend(("--shield-path", str(lid.shield_path), "--evaluation-policy", "unshielded"))
    return command, run_dir


def validate_training_provenance(run_dir: Path, method: str, lid: LidRecord) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    recorded = config.get("init_policy_path")
    if recorded is None:
        # The SB3 stages validate and apply --init-policy-path but currently do
        # not duplicate that field in config.json.  The immutable job command
        # remains the source of provenance for these two methods.
        if method not in SB3_METHODS:
            raise ValueError(f"Training config does not record its initial actor: {run_dir}.")
    elif Path(str(recorded)).resolve() != lid.base_policy_path:
        raise ValueError(
            f"Training used the wrong initial actor: {recorded} != {lid.base_policy_path}."
        )
    return config


def run_worker(args: argparse.Namespace) -> int:
    lid = load_lid_record(args.lid_root, args.environment, args.seed)
    command, run_dir = training_command(
        args.environment,
        args.method,
        args.seed,
        output_root=args.output_root,
        lid=lid,
        smoke=args.smoke,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    completion = run_dir / "postprocess_metrics.json"
    if completion.is_file() and not args.force:
        print(f"complete: {completion}", flush=True)
        return 0

    checkpoint = checkpoint_path(args.method, run_dir)
    training_spec = {
        "environment": args.environment,
        "method": args.method,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "command": command,
        "safe_initialisation": {
            "path": str(lid.base_policy_path),
            "sha256": lid.base_sha256,
            "scope": "actor_only",
        },
        "fixed_lid": {
            "directory": str(lid.directory),
            "bounds_path": str(lid.bounds_path),
            "bounds_sha256": lid.bounds_sha256,
            "matched_pspo_iterations": lid.iterations,
        },
        "shield": {"path": str(lid.shield_path), "sha256": lid.shield_sha256},
        "created_utc": utc_now(),
    }
    atomic_json(run_dir / "experiment_job.json", training_spec)

    if args.force or not checkpoint.is_file():
        env = dict(os.environ)
        env.update(THREAD_ENV)
        env["PYTHONPATH"] = f"{REPO / 'core'}:{REPO}:{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        print("training: " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO, env=env, check=False)
        if completed.returncode:
            return int(completed.returncode)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce {checkpoint}.")

    training_config = validate_training_provenance(run_dir, args.method, lid)
    actor_state = load_actor_state(args.method, checkpoint, lid.architecture)
    policy = make_policy(lid.architecture, actor_state)
    shield_mask = load_shield_mask(lid.shield_path)
    base_payload = torch.load(lid.base_policy_path, map_location="cpu", weights_only=False)
    base_policy = make_policy(lid.architecture, dict(base_payload["state_dict"]))
    base_exact = exhaustive_shield_alignment(
        BasePolicyPredictor(base_policy, input_dim=int(lid.architecture["input_dim"])),
        shield_mask,
        input_dim=int(lid.architecture["input_dim"]),
    )
    if int(base_exact["unsafe_states"]) != 0:
        raise RuntimeError(f"The claimed safe initial actor fails exact audit: {base_exact}.")

    ref_config = baseline_suite.reference_config(args.environment, args.seed)
    episodes = 2 if args.smoke else int(ref_config["eval_episodes"])
    eval_seed = int(args.seed) + 10_000
    predictor = BasePolicyPredictor(policy, input_dim=int(lid.architecture["input_dim"]))
    raw_evaluation = _evaluate(
        predictor, ref_config, shield_mask, episodes=episodes, eval_seed=eval_seed
    )
    raw_mean_reward = _mean_reward(raw_evaluation)

    bounds_payload = torch.load(lid.bounds_path, map_location="cpu", weights_only=False)
    raw_exact, deployed_exact, projection = audit_and_project(
        policy, lid.architecture, bounds_payload, shield_mask
    )
    deployed_predictor = BasePolicyPredictor(
        policy, input_dim=int(lid.architecture["input_dim"])
    )
    deployed_evaluation = _evaluate(
        deployed_predictor, ref_config, shield_mask, episodes=episodes, eval_seed=eval_seed
    )
    deployed_mean_reward = _mean_reward(deployed_evaluation)

    deployed_path = run_dir / "deployed_actor.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "architecture": lid.architecture,
            "environment": args.environment,
            "method": args.method,
            "seed": int(args.seed),
            "source_checkpoint": str(checkpoint.resolve()),
            "projection": projection,
            "fixed_lid": training_spec["fixed_lid"],
        },
        deployed_path,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": utc_now(),
        "environment": args.environment,
        "method": args.method,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "episodes": episodes,
        "eval_seed_start": eval_seed,
        "evaluation_policy": "nominal_unshielded_deterministic",
        "safe_initialisation": training_spec["safe_initialisation"],
        "base_exact_safety": base_exact,
        "fixed_lid": training_spec["fixed_lid"],
        "shield": training_spec["shield"],
        "training": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "config": str((run_dir / "config.json").resolve()),
            "config_sha256": sha256(run_dir / "config.json"),
            "recorded_init_policy_path": training_config.get("init_policy_path"),
        },
        "raw": {
            "mean_total_reward": raw_mean_reward,
            "safe_trajectory_rate": float(raw_evaluation["safe_trajectory_rate"]),
            "evaluation": raw_evaluation,
            "exact_safety": raw_exact,
        },
        "projection": projection,
        "deployed": {
            "actor_path": str(deployed_path.resolve()),
            "mean_total_reward": deployed_mean_reward,
            "safe_trajectory_rate": float(deployed_evaluation["safe_trajectory_rate"]),
            "evaluation": deployed_evaluation,
            "exact_safety": deployed_exact,
        },
        "delta": {
            "mean_total_reward": deployed_mean_reward - raw_mean_reward,
            "safe_trajectory_rate": (
                float(deployed_evaluation["safe_trajectory_rate"])
                - float(raw_evaluation["safe_trajectory_rate"])
            ),
        },
    }
    atomic_json(completion, result)
    print(
        f"complete {args.environment}/{args.method}/seed{args.seed}: "
        f"projected={projection['applied']} reward={deployed_mean_reward:.6g} "
        f"safety={deployed_evaluation['safe_trajectory_rate']:.3f}",
        flush=True,
    )
    return 0


def parse_mpstat_idle(output: str) -> dict[int, float]:
    idle: dict[int, float] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "Average:" and fields[1].isdigit():
            idle[int(fields[1])] = float(fields[-1])
    if not idle:
        raise ValueError("mpstat output contains no average per-CPU samples.")
    return idle


def probe_free_cpus(*, samples: int, minimum_idle: float, reserve: int) -> tuple[list[int], dict[int, float]]:
    if shutil.which("mpstat") is None:
        raise RuntimeError("mpstat is required for per-core admission control.")
    completed = subprocess.run(
        ["mpstat", "-P", "ALL", "1", str(samples)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    idle = parse_mpstat_idle(completed.stdout)
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(idle)
    free = sorted(
        cpu for cpu, percentage in idle.items()
        if cpu >= reserve and cpu in affinity and percentage >= minimum_idle
    )
    return free, idle


def select_methods_for_capacity(
    free_core_count: int,
    *,
    seeds_per_method: int,
    requested_methods: Sequence[str] | None,
) -> tuple[str, ...]:
    if seeds_per_method <= 0:
        raise ValueError("At least one seed is required.")
    if requested_methods is not None:
        methods = tuple(requested_methods)
        needed = len(methods) * seeds_per_method
        if free_core_count < needed:
            raise RuntimeError(
                f"Requested {len(methods)} complete methods need {needed} free cores; "
                f"only {free_core_count} passed admission."
            )
        return methods
    count = min(len(METHOD_PRIORITY), free_core_count // seeds_per_method)
    return METHOD_PRIORITY[:count]


def worker_command(args: argparse.Namespace, environment: str, method: str, seed: int) -> tuple[str, ...]:
    command = [
        str(REPO / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--worker",
        "--environment", environment,
        "--method", method,
        "--seed", str(seed),
        "--lid-root", str(args.lid_root),
        "--output-root", str(args.output_root),
    ]
    if args.smoke:
        command.append("--smoke")
    if args.force:
        command.append("--force")
    return tuple(command)


def make_wave_jobs(
    args: argparse.Namespace,
    environment: str,
    methods: Sequence[str],
    cpus: Sequence[int],
) -> list[Job]:
    pairs = [(method, seed) for method in methods for seed in args.seeds]
    if len(cpus) < len(pairs):
        raise RuntimeError(f"Wave needs {len(pairs)} distinct CPUs, got {len(cpus)}.")
    jobs: list[Job] = []
    for cpu, (method, seed) in zip(cpus, pairs, strict=False):
        run_dir = args.output_root / environment / method / f"seed{seed}"
        jobs.append(
            Job(
                environment=environment,
                method=method,
                seed=int(seed),
                cpu=int(cpu),
                run_dir=run_dir,
                log_path=args.output_root / "_logs" / f"{environment}_{method}_seed{seed}.log",
                command=worker_command(args, environment, method, int(seed)),
            )
        )
    return jobs


def _run_pinned_job(job: Job) -> tuple[Job, int, float]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env["PYTHONPATH"] = f"{REPO / 'core'}:{REPO}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with job.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] cpu={job.cpu} command={' '.join(job.command)}\n")
        handle.flush()
        rc = subprocess.run(
            ["taskset", "-c", str(job.cpu), *job.command],
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    return job, int(rc), time.monotonic() - started


def _source_provenance() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        sources[str(relative)] = {
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
        }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return {"git_head": git_head, "git_status": git_status, "source_files": sources}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sem(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def collect_results(output_root: Path, *, smoke: bool | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/*/seed*/postprocess_metrics.json")):
        result = read_json(path)
        if result.get("status") != "complete":
            continue
        if smoke is not None and bool(result.get("smoke")) != smoke:
            continue
        results.append(result)
    return results


def write_report(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda r: (r["environment"], r["method"], r["seed"])):
        per_seed.append(
            {
                "environment": result["environment"],
                "method": result["method"],
                "seed": result["seed"],
                "episodes": result["episodes"],
                "lid_iterations": result["fixed_lid"]["matched_pspo_iterations"],
                "projection_applied": result["projection"]["applied"],
                "raw_mean_total_reward": result["raw"]["mean_total_reward"],
                "raw_safe_trajectory_rate": result["raw"]["safe_trajectory_rate"],
                "raw_exact_alignment_rate": result["raw"]["exact_safety"]["alignment_rate"],
                "raw_exact_unsafe_states": result["raw"]["exact_safety"]["unsafe_states"],
                "deployed_mean_total_reward": result["deployed"]["mean_total_reward"],
                "deployed_safe_trajectory_rate": result["deployed"]["safe_trajectory_rate"],
                "deployed_exact_alignment_rate": result["deployed"]["exact_safety"]["alignment_rate"],
                "deployed_exact_unsafe_states": result["deployed"]["exact_safety"]["unsafe_states"],
                "reward_delta": result["delta"]["mean_total_reward"],
                "safety_rate_delta": result["delta"]["safe_trajectory_rate"],
                "inside_lid_before": result["projection"]["inside_lid_before"],
                "inside_lid_after": result["projection"]["inside_lid_after"],
                "projected_parameter_count": result["projection"]["n_projected"],
                "boundary_parameter_count": result["projection"]["n_boundary"],
                "projection_l2": result["projection"]["displacement_l2"],
                "projection_linf": result["projection"]["displacement_linf"],
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault((str(row["environment"]), str(row["method"])), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (environment, method), group in sorted(grouped.items()):
        deployed_rewards = [float(row["deployed_mean_total_reward"]) for row in group]
        deployed_safety = [float(row["deployed_safe_trajectory_rate"]) for row in group]
        raw_rewards = [float(row["raw_mean_total_reward"]) for row in group]
        raw_safety = [float(row["raw_safe_trajectory_rate"]) for row in group]
        reward_delta = [float(row["reward_delta"]) for row in group]
        safety_delta = [float(row["safety_rate_delta"]) for row in group]
        aggregate.append(
            {
                "environment": environment,
                "method": method,
                "n_seeds": len(group),
                "deployed_mean_total_reward": statistics.fmean(deployed_rewards),
                "deployed_reward_2se": 2.0 * _sem(deployed_rewards),
                "deployed_mean_safe_trajectory_rate": statistics.fmean(deployed_safety),
                "deployed_safety_rate_2se": 2.0 * _sem(deployed_safety),
                "raw_mean_total_reward": statistics.fmean(raw_rewards),
                "raw_reward_2se": 2.0 * _sem(raw_rewards),
                "raw_mean_safe_trajectory_rate": statistics.fmean(raw_safety),
                "raw_safety_rate_2se": 2.0 * _sem(raw_safety),
                "paired_mean_reward_delta": statistics.fmean(reward_delta),
                "paired_reward_delta_2se": 2.0 * _sem(reward_delta),
                "paired_mean_safety_rate_delta": statistics.fmean(safety_delta),
                "paired_safety_rate_delta_2se": 2.0 * _sem(safety_delta),
                "projection_fraction": statistics.fmean(
                    float(bool(row["projection_applied"])) for row in group
                ),
                "deployed_exact_alignment_min": min(
                    float(row["deployed_exact_alignment_rate"]) for row in group
                ),
                "deployed_exact_unsafe_states": sum(
                    int(row["deployed_exact_unsafe_states"]) for row in group
                ),
                "mean_projection_l2": statistics.fmean(
                    float(row["projection_l2"]) for row in group
                ),
            }
        )
    atomic_json(output_root / "per_seed.json", per_seed)
    atomic_json(output_root / "aggregate.json", aggregate)
    _write_csv(output_root / "per_seed.csv", per_seed)
    _write_csv(output_root / "aggregate.csv", aggregate)

    lines = [
        "# Safe-initialised baseline projection results",
        "",
        "Policies are evaluated nominally (without a runtime shield). Reward and empirical safety "
        "are mean ± 2SE across training seeds. Exact alignment is an exhaustive greedy-action "
        "audit over every shield state with at least one safe action; empirical trajectory safety "
        "can still be below one under stochastic transitions.",
        "",
        "| Environment | Method | Seeds | Deployed reward | Deployed safety | Projected | Exact alignment |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {environment} | {method} | {n_seeds} | "
            "{deployed_mean_total_reward:.4f} ± {deployed_reward_2se:.4f} | "
            "{deployed_mean_safe_trajectory_rate:.4f} ± {deployed_safety_rate_2se:.4f} | "
            "{projection_fraction:.1%} | {deployed_exact_alignment_min:.4f} |".format(**row)
        )
    atomic_text(output_root / "REPORT.md", "\n".join(lines) + "\n")


def _validate_all_lids(lid_root: Path, environments: Iterable[str], seeds: Iterable[int]) -> None:
    for environment in environments:
        for seed in seeds:
            load_lid_record(lid_root, environment, seed)


def run_scheduler(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    _validate_all_lids(args.lid_root, args.envs, args.seeds)
    free, idle = probe_free_cpus(
        samples=args.probe_samples,
        minimum_idle=args.minimum_idle,
        reserve=args.reserve_low_cpus,
    )
    methods = select_methods_for_capacity(
        len(free), seeds_per_method=len(args.seeds), requested_methods=args.methods
    )
    if not methods:
        print(
            f"No complete baseline cohort can launch: {len(free)} admitted cores for "
            f"{len(args.seeds)} seeds.",
            file=sys.stderr,
        )
        return 2
    wave_size = len(methods) * len(args.seeds)
    admitted = free[:wave_size]
    study_manifest = {
        "schema_version": 1,
        "intent": "safe-initialised baselines with conditional fixed-LID projection",
        "created_utc": utc_now(),
        "environments": list(args.envs),
        "methods": list(methods),
        "method_priority": list(METHOD_PRIORITY),
        "seeds": list(args.seeds),
        "smoke": bool(args.smoke),
        "evaluation_policy": "nominal_unshielded_deterministic",
        "cpu_policy": {
            "probe_samples": int(args.probe_samples),
            "minimum_average_idle": float(args.minimum_idle),
            "reserve_low_cpus": int(args.reserve_low_cpus),
            "wave_size": wave_size,
        },
        "lid_root": str(args.lid_root.resolve()),
        "source_provenance": _source_provenance(),
    }
    manifest_path = args.output_root / "study_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        for key in ("environments", "methods", "seeds", "smoke", "lid_root"):
            if existing.get(key) != study_manifest.get(key):
                raise RuntimeError(
                    f"Existing immutable study manifest disagrees on {key}: "
                    f"{existing.get(key)!r} != {study_manifest.get(key)!r}."
                )
        existing_sources = (existing.get("source_provenance") or {}).get("source_files")
        current_sources = study_manifest["source_provenance"]["source_files"]
        if existing_sources != current_sources:
            raise RuntimeError(
                "Relevant source hashes differ from the immutable study manifest; "
                "use a new output root for the changed implementation."
            )
    else:
        atomic_json(manifest_path, study_manifest)

    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = args.output_root / "_launches"
    launch = {
        "launch_id": launch_id,
        "started_utc": utc_now(),
        "initial_admitted_cpu_ids": admitted,
        "initial_idle_percentages": {str(cpu): idle[cpu] for cpu in admitted},
        "waves": [],
    }
    launch_path = launch_dir / f"{launch_id}.json"
    atomic_json(launch_path, launch)

    if args.dry_run:
        print(f"selected methods: {', '.join(methods)}")
        print(f"admitted CPUs: {','.join(map(str, admitted))}")
        for environment in args.envs:
            for job in make_wave_jobs(args, environment, methods, admitted):
                print(f"cpu={job.cpu} {' '.join(job.command)}")
        return 0

    failures = 0
    for environment in args.envs:
        wave_free, wave_idle = probe_free_cpus(
            samples=args.probe_samples,
            minimum_idle=args.minimum_idle,
            reserve=args.reserve_low_cpus,
        )
        wave_cpus = wave_free[:wave_size]
        if len(wave_cpus) < wave_size:
            raise RuntimeError(
                f"Capacity fell before {environment}: need {wave_size}, admitted {len(wave_cpus)}."
            )
        jobs = make_wave_jobs(args, environment, methods, wave_cpus)
        pending = [
            job for job in jobs
            if args.force or not (job.run_dir / "postprocess_metrics.json").is_file()
        ]
        wave_record: dict[str, Any] = {
            "environment": environment,
            "started_utc": utc_now(),
            "assigned_cpu_ids": wave_cpus,
            "idle_percentages": {str(cpu): wave_idle[cpu] for cpu in wave_cpus},
            "pending_jobs": len(pending),
            "jobs": [asdict(job) | {"run_dir": str(job.run_dir), "log_path": str(job.log_path), "command": list(job.command)} for job in pending],
        }
        launch["waves"].append(wave_record)
        atomic_json(launch_path, launch)
        print(
            f"wave {environment}: {len(pending)} pending jobs on {len(wave_cpus)} distinct CPUs",
            flush=True,
        )
        wave_failures = 0
        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            futures = [pool.submit(_run_pinned_job, job) for job in pending]
            for future in as_completed(futures):
                job, rc, seconds = future.result()
                status = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(
                    f"{status} {job.environment}/{job.method}/seed{job.seed} "
                    f"cpu={job.cpu} {seconds:.0f}s -> {job.log_path}",
                    flush=True,
                )
                wave_failures += int(rc != 0)
        failures += wave_failures
        wave_record["completed_utc"] = utc_now()
        wave_record["failures"] = wave_failures
        write_report(args.output_root, collect_results(args.output_root, smoke=bool(args.smoke)))
        atomic_json(launch_path, launch)
        if wave_failures:
            print(f"Stopping after {environment} because {wave_failures} jobs failed.", file=sys.stderr)
            break
    launch["completed_utc"] = utc_now()
    launch["failures"] = failures
    atomic_json(launch_path, launch)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--environment", choices=ENVIRONMENTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--lid-root", type=Path, default=DEFAULT_LID_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--probe-samples", type=int, default=5)
    parser.add_argument("--minimum-idle", type=float, default=90.0)
    parser.add_argument("--reserve-low-cpus", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.lid_root = args.lid_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.report_only:
        write_report(args.output_root, collect_results(args.output_root, smoke=None))
        return 0
    if args.worker:
        if args.environment is None or args.method is None or args.seed is None:
            raise SystemExit("--worker requires --environment, --method, and --seed.")
        return run_worker(args)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("--seeds must be non-empty and unique.")
    return run_scheduler(args)


if __name__ == "__main__":
    raise SystemExit(main())
