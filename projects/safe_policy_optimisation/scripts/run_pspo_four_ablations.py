"""Resumable launcher for the four controlled PSPO ablation studies.

The launcher deliberately keeps LID refresh separate from enforcement cadence.
It reuses validated canonical controls, extracts seed-specific fixed-LID budgets,
pins one CPU core per seed process, and writes a source/configuration manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projects.safe_policy_optimisation.utils.pspo_defaults import environment_defaults
from projects.safe_policy_optimisation.utils.pspo_launcher import parse_cpu_ids

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
OUTPUT_ROOT = REPO / "artifacts/ablation_studies/pspo_four_ablations"
ONE_ENV = REPO / "projects/safe_policy_optimisation/scripts/run_pspo_one_env.sh"
ENVIRONMENTS = (
    "media_streaming",
    "colour_bomb",
    "colour_bomb_v2",
    "bridge_crossing",
    "bridge_crossing_v2",
    "mini_pacman",
)
CONTROL_COHORTS = {
    "media_streaming": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "colour_bomb": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "colour_bomb_v2": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "bridge_crossing": "pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1",
    "bridge_crossing_v2": (
        "pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base"
    ),
    "mini_pacman": (
        "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base"
    ),
}
VARIANTS = (
    "control",
    "no_entropy",
    "ce_only",
    "fixed_lid",
    "no_gradient",
    "region_first_instrumented",
    "verify_first",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def source_dir(environment: str, architecture: str) -> Path:
    return RUNS / CONTROL_COHORTS[environment] / architecture / environment


def _canonical_signature(config: dict[str, Any]) -> dict[str, Any]:
    adaptive = config.get("adaptive")
    training = config.get("training_hyperparameters")
    if not isinstance(adaptive, dict) or not isinstance(training, dict):
        raise ValueError("canonical config lacks adaptive/training settings")
    return {
        "env_id": config.get("env_id"),
        "env_kwargs": config.get("env_kwargs"),
        "max_episode_steps": config.get("max_episode_steps"),
        "shield_sha256": config.get("shield_sha256"),
        "base_policy_architecture": config.get("base_policy_architecture"),
        "total_timesteps": config.get("total_timesteps"),
        "evaluation_policy": config.get("evaluation_policy"),
        "training_hyperparameters": training,
        "granularity": adaptive.get("granularity"),
        "rollout_interval": adaptive.get("rollout_interval", adaptive.get("frequency")),
        "rashomon_n_iters": adaptive.get("rashomon_n_iters"),
        "rashomon_multi_label_mode": adaptive.get("rashomon_multi_label_mode"),
        "rashomon_surrogate": adaptive.get("rashomon_surrogate"),
        "rashomon_objective": adaptive.get("rashomon_objective"),
        "region_update_mode": adaptive.get("region_update_mode"),
        "directional_rashomon_growth": adaptive.get("directional_rashomon_growth"),
        "stop_when_proposal_contained": adaptive.get("stop_when_proposal_contained"),
        "verify_first": adaptive.get("verify_first"),
    }


def validate_control(
    environment: str,
    *,
    architecture: str,
    seeds: list[int],
) -> dict[str, Any]:
    root = source_dir(environment, architecture)
    if not root.exists():
        raise FileNotFoundError(f"canonical source directory is missing: {root}")
    base_policy: Path | None = None
    base_summary: Path | None = None
    base_hash: str | None = None
    budgets: dict[str, int] = {}
    signature: dict[str, Any] | None = None
    source_seeds: dict[str, Any] = {}
    for seed in seeds:
        seed_dir = root / f"seed{seed}"
        required = tuple(seed_dir / name for name in ("summary.json", "config.json", "metrics.json"))
        absent = [path for path in required if not path.exists()]
        if absent:
            raise FileNotFoundError(
                f"canonical {environment} seed{seed} is incomplete: {absent}"
            )
        summary, config = json_file(required[0]), json_file(required[1])
        configured_base = Path(str(config.get("base_policy_path", "")))
        if not configured_base.exists():
            raise FileNotFoundError(
                f"configured canonical base policy is missing for {environment} "
                f"seed{seed}: {configured_base}"
            )
        configured_hash = sha256(configured_base)
        if base_policy is None:
            base_policy = configured_base.resolve()
            base_hash = configured_hash
            base_summary = base_policy.parent / "summary.json"
            if not base_summary.exists():
                raise FileNotFoundError(
                    f"canonical base-policy summary is missing: {base_summary}"
                )
        elif configured_hash != base_hash:
            raise ValueError(
                f"canonical base-policy hash differs for {environment} seed{seed}"
            )
        diagnostics = summary.get("adaptive_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"missing adaptive diagnostics: {required[0]}")
        try:
            budget = int(diagnostics["rashomon_iters_spent"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing rashomon_iters_spent: {required[0]}") from exc
        if budget <= 0:
            raise ValueError(f"non-positive fixed-LID budget in {required[0]}")
        candidate_signature = _canonical_signature(config)
        if signature is None:
            signature = candidate_signature
        elif candidate_signature != signature:
            raise ValueError(
                f"canonical non-ablated configuration differs for {environment} seed{seed}"
            )
        budgets[str(seed)] = budget
        source_seeds[str(seed)] = {
            "directory": str(seed_dir.resolve()),
            "summary_sha256": sha256(required[0]),
            "config_sha256": sha256(required[1]),
            "metrics_sha256": sha256(required[2]),
        }
    assert signature is not None and base_policy is not None
    assert base_summary is not None and base_hash is not None
    expected = environment_defaults(environment)
    if signature["total_timesteps"] != expected.total_timesteps:
        raise ValueError(f"canonical timestep mismatch for {environment}: {signature}")
    if str(signature["rollout_interval"]) != str(expected.frequency):
        raise ValueError(f"canonical enforcement-frequency mismatch for {environment}: {signature}")
    for key, expected_value in {
        "granularity": "train_phase",
        "rashomon_n_iters": 200,
        "rashomon_multi_label_mode": "all",
        "rashomon_surrogate": "logsumexp",
        "rashomon_objective": "weighted_width",
        "region_update_mode": "replace",
        "directional_rashomon_growth": True,
        "stop_when_proposal_contained": True,
        "verify_first": False,
    }.items():
        if signature.get(key) != expected_value:
            raise ValueError(
                f"canonical {environment} has {key}={signature.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    return {
        "root": str(root.resolve()),
        "base_policy_path": str(base_policy.resolve()),
        "base_policy_sha256": base_hash,
        "base_summary_sha256": sha256(base_summary),
        "signature": signature,
        "seed_budgets": budgets,
        "seeds": source_seeds,
    }


def repository_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "commit": head,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status": status.splitlines(),
    }


@dataclass(frozen=True)
class Job:
    variant: str
    environment: str
    seed: int
    cpu: int
    output_dir: Path
    env: dict[str, str]


def variant_environment(
    variant: str,
    environment: str,
    seed: int,
    *,
    cpu: int,
    architecture: str,
    source: dict[str, Any] | None,
    output_root: Path,
    dry_run: bool,
    total_timesteps_override: int | None,
    smoke: bool = False,
) -> Job:
    output_dir = output_root / variant / architecture / environment
    settings = {
        "BC_MARGIN_LOSS_WEIGHT": "1.0",
        "BC_SAFE_ACTION_ENTROPY_WEIGHT": "1.0",
        "REGION_REFRESH": "adaptive",
        "DIRECTIONAL_RASHOMON_GROWTH": "1",
        "STOP_WHEN_PROPOSAL_CONTAINED": "1",
        "VERIFY_FIRST": "false",
        "AUDIT_CANDIDATES_EXACTLY": "0",
        "RASHOMON_N_ITERS": "200",
    }
    if variant == "no_entropy":
        settings["BC_SAFE_ACTION_ENTROPY_WEIGHT"] = "0.0"
    elif variant == "ce_only":
        settings["BC_SAFE_ACTION_ENTROPY_WEIGHT"] = "0.0"
        settings["BC_MARGIN_LOSS_WEIGHT"] = "0.0"
        # Cross-entropy is a logsumexp over the safe set, so it is governed by
        # the best safe logit. Grading it in "all" mode (worst safe logit) tests
        # a quantity the loss never optimises. "any" aligns the criterion with
        # the objective, and is applied to the Rashomon set too so the
        # initialisation stays valid downstream.
        settings["RASHOMON_MULTI_LABEL_MODE"] = "any"
    elif variant == "fixed_lid":
        assert source is not None
        settings.update(
            REGION_REFRESH="fixed",
            DIRECTIONAL_RASHOMON_GROWTH="0",
            STOP_WHEN_PROPOSAL_CONTAINED="0",
            RASHOMON_N_ITERS=str(source["seed_budgets"][str(seed)]),
        )
    elif variant == "no_gradient":
        settings.update(
            DIRECTIONAL_RASHOMON_GROWTH="0",
            STOP_WHEN_PROPOSAL_CONTAINED="0",
        )
    elif variant == "region_first_instrumented":
        settings["AUDIT_CANDIDATES_EXACTLY"] = "1"
    elif variant == "verify_first":
        settings["VERIFY_FIRST"] = "true"
    elif variant != "control":
        raise ValueError(f"unknown variant: {variant}")
    if smoke:
        settings["RASHOMON_N_ITERS"] = "1"
    if source is not None and variant not in {"no_entropy", "ce_only"}:
        settings["BASE_POLICY_PATH"] = source["base_policy_path"]
    env = {
        **os.environ,
        **settings,
        "ENV_NAME": environment,
        "SEEDS": str(seed),
        "CPU_IDS": str(cpu),
        "ARCHITECTURE": architecture,
        "RUN_NAME": f"pspo_four_ablations_{variant}",
        "OUTPUT_BASE": str(output_dir),
        "ADAPTIVE_FREQ": environment_defaults(environment).frequency,
        "TOTAL_TIMESTEPS": str(
            total_timesteps_override
            if total_timesteps_override is not None
            else environment_defaults(environment).total_timesteps
        ),
        "SKIP_EXISTING": "1",
        "DRY_RUN": "1" if dry_run else "0",
        "PSPO_SMOKE": "1" if smoke else "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    return Job(variant, environment, seed, cpu, output_dir, env)


def validate_job_output(job: Job) -> None:
    """Enforce ablation invariants against the completed seed diagnostics."""

    summary_path = job.output_dir / f"seed{job.seed}" / "summary.json"
    summary = json_file(summary_path)
    diagnostics = summary.get("adaptive_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"completed run lacks adaptive diagnostics: {summary_path}")
    if job.variant in {"fixed_lid", "no_gradient"}:
        if diagnostics.get("proposal_information_used") is not False:
            raise ValueError(
                f"{job.variant} seed{job.seed} used proposal information during LID growth"
            )
    if job.variant == "fixed_lid":
        if int(diagnostics.get("initial_region_computations", -1)) != 1:
            raise ValueError(
                f"fixed_lid seed{job.seed} did not compute exactly one initial LID"
            )
        if int(diagnostics.get("rashomon_computations", -1)) != 1:
            raise ValueError(
                f"fixed_lid seed{job.seed} recomputed its supposedly fixed LID"
            )
        if diagnostics.get("region_refresh") != "fixed":
            raise ValueError(f"fixed_lid seed{job.seed} lacks fixed refresh diagnostics")
    if job.variant in {"region_first_instrumented", "verify_first"}:
        events = diagnostics.get("safety_update_events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"{job.variant} seed{job.seed} recorded no update events")


def run_job(job: Job) -> tuple[Job, int]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = job.output_dir / "_launch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"seed{job.seed}.log").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            ["bash", str(ONE_ENV)], cwd=REPO, env=job.env,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
        return_code = int(completed.returncode)
        if return_code == 0:
            try:
                validate_job_output(job)
            except (FileNotFoundError, ValueError, OSError) as exc:
                handle.write(f"\nAblation postcondition failed: {exc}\n")
                return_code = 2
    return job, return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--architecture", choices=("two_hidden",), default="two_hidden")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cpu-ids", default=None, help="Comma/range CPU list; defaults to affinity.")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Smoke-test override only; omitted for the controlled suite.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one LID iteration and eight training steps; never use for results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.variants)) != len(args.variants) or len(set(args.envs)) != len(args.envs):
        raise SystemExit("variants and environments must be distinct")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be a non-empty distinct list")
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("seeds must be non-negative")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive")
    if args.smoke and args.total_timesteps is not None:
        raise SystemExit("--smoke cannot be combined with --total-timesteps")
    state = repository_state()
    if state["dirty"] and not args.allow_dirty and not args.dry_run:
        raise SystemExit("repository is dirty; commit changes or pass --allow-dirty")

    source_errors: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    for environment in args.envs:
        try:
            sources[environment] = validate_control(
                environment, architecture=args.architecture, seeds=args.seeds
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            source_errors[environment] = str(exc)
    source_required = any(
        variant in {"control", "fixed_lid", "no_gradient", "region_first_instrumented", "verify_first"}
        for variant in args.variants
    )
    if source_errors and source_required and (not args.dry_run or args.strict_dry_run):
        raise SystemExit("canonical source validation failed:\n" + json.dumps(source_errors, indent=2))

    allowed = sorted(os.sched_getaffinity(0))
    cpus = parse_cpu_ids(args.cpu_ids) if args.cpu_ids else allowed
    unavailable = sorted(set(cpus) - set(allowed))
    if unavailable:
        raise SystemExit(f"CPUs outside process affinity: {unavailable}")
    max_parallel = args.max_parallel or len(cpus)
    if max_parallel <= 0 or not cpus:
        raise SystemExit("at least one CPU is required")
    max_parallel = min(max_parallel, len(cpus))

    jobs: list[Job] = []
    index = 0
    for variant in args.variants:
        if variant == "control":
            continue
        for environment in args.envs:
            source = sources.get(environment)
            for seed in args.seeds:
                if variant == "fixed_lid" and source is None:
                    continue
                jobs.append(
                    variant_environment(
                        variant, environment, seed,
                        cpu=cpus[index % len(cpus)],
                        architecture=args.architecture,
                        source=source,
                        output_root=args.output_root,
                        dry_run=args.dry_run,
                        total_timesteps_override=(8 if args.smoke else args.total_timesteps),
                        smoke=args.smoke,
                    )
                )
                index += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": state,
        "architecture": args.architecture,
        "variants": args.variants,
        "environments": args.envs,
        "seeds": args.seeds,
        "canonical_sources": sources,
        "source_validation_errors": source_errors,
        "resolved_settings": {
            "timesteps": {
                environment: (
                    8
                    if args.smoke
                    else args.total_timesteps
                    or environment_defaults(environment).total_timesteps
                )
                for environment in args.envs
            },
            "enforcement_frequency": {
                environment: environment_defaults(environment).frequency
                for environment in args.envs
            },
            "lid_iterations": 200,
            "full_state_lid_batches": True,
            "weighted_width": True,
            "pinned_single_cpu_per_job": True,
        },
        "cpu_ids": cpus,
        "max_parallel": max_parallel,
        "smoke": bool(args.smoke),
        "jobs": [
            {
                "variant": job.variant,
                "environment": job.environment,
                "seed": job.seed,
                "cpu": job.cpu,
                "output_dir": str(job.output_dir),
                "settings": {
                    key: job.env[key]
                    for key in (
                        "REGION_REFRESH", "ADAPTIVE_FREQ", "VERIFY_FIRST",
                        "DIRECTIONAL_RASHOMON_GROWTH", "STOP_WHEN_PROPOSAL_CONTAINED",
                        "AUDIT_CANDIDATES_EXACTLY", "RASHOMON_N_ITERS",
                        "BC_MARGIN_LOSS_WEIGHT", "BC_SAFE_ACTION_ENTROPY_WEIGHT",
                        "TOTAL_TIMESTEPS", "BASE_POLICY_PATH",
                    )
                    if key in job.env
                },
            }
            for job in jobs
        ],
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Initialisation variants share one per-environment base policy. Run their
    # first requested seed before parallelising the remaining seed jobs.
    failed: list[dict[str, Any]] = []
    deferred = list(jobs)
    for variant in ("no_entropy", "ce_only"):
        for environment in args.envs:
            matches = [
                job for job in deferred
                if job.variant == variant and job.environment == environment
            ]
            if not matches:
                continue
            first = matches[0]
            _, rc = run_job(first)
            deferred.remove(first)
            if rc:
                failed.append({"variant": variant, "environment": environment, "seed": first.seed, "rc": rc})
                deferred = [
                    job for job in deferred
                    if not (job.variant == variant and job.environment == environment)
                ]
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_job, job): job for job in deferred}
        for future in as_completed(futures):
            job, rc = future.result()
            print(f"finished {job.variant}/{job.environment}/seed{job.seed}: rc={rc}", flush=True)
            if rc:
                failed.append({
                    "variant": job.variant, "environment": job.environment,
                    "seed": job.seed, "rc": rc,
                })
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
