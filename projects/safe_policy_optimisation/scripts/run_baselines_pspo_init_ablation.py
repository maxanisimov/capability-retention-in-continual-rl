#!/usr/bin/env python
"""Warm-start every safe-RL baseline from PSPO's safe policy initialisation.

The PSPO ablations so far only ever moved PSPO's own components. This one asks
the complementary question: is the safe initialisation useful *on its own*, to
methods that have no certified region and no projection? Each baseline is
trained twice-over -- once as published (cold start, already on disk) and once
with its actor initialised from the very ``base_policy.pt`` the matching PSPO
run used -- so the paired difference isolates the initialisation.

Only the actor is warm-started. Critics, Lagrange multipliers, and optimizer
state start fresh, and no baseline gains a certified region, so any safety
improvement here is *empirical*, not guaranteed -- unlike PSPO, nothing in
these runs constrains the policy after initialisation.

Methods
-------
``ppo``                 plain PPO (SB3)
``ppo_lagrangian``      PPO-Lagrangian
``ppo_pid_lagrangian``  PPO-PID-Lagrangian
``cpo``                 CPO
``ppo_shield``          PPO with the runtime shield attached

Usage
-----
    # inspect the full matrix without running anything
    .venv/bin/python projects/safe_policy_optimisation/scripts/run_baselines_pspo_init_ablation.py \\
        --envs colour_bomb media_streaming --dry-run

    # launch, one CPU per seed-job
    .venv/bin/python projects/safe_policy_optimisation/scripts/run_baselines_pspo_init_ablation.py \\
        --envs colour_bomb --methods ppo cpo --cpu-ids 30-39
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
STAGES = REPO / "projects/safe_policy_optimisation/stages"
RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"

# The PSPO cohort whose base policy and settings each environment inherits.
# These are the same control cohorts the PSPO ablation suite compares against,
# so the warm start is the exact initialisation PSPO itself used.
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
ENVIRONMENTS = tuple(CONTROL_COHORTS)

# method -> (stage script, extra CLI args). CPO and the two Lagrangian variants
# share one stage, selected by --algorithms.
METHODS: dict[str, tuple[str, list[str]]] = {
    "ppo": ("train_ppo.py", []),
    "ppo_lagrangian": ("train_ppo_lagrangian.py", ["--algorithms", "ppo_lagrangian"]),
    "ppo_pid_lagrangian": (
        "train_ppo_lagrangian.py",
        ["--algorithms", "ppo_pid_lagrangian"],
    ),
    "cpo": ("train_cpo.py", []),
    "ppo_shield": ("train_ppo_shield.py", []),
    # RL-SGF (Mestres et al. 2025) with its single untuned setting; opt-in only.
    "rl_sgf": (
        "train_rl_sgf.py",
        [
            "--rl-sgf-step-size", "0.1",
            "--rl-sgf-alpha", "1.0",
            "--rl-sgf-episodes-per-iter", "20",
        ],
    ),
}
# rl_sgf must be requested explicitly so existing default sweeps are unchanged.
DEFAULT_METHODS = tuple(method for method in METHODS if method != "rl_sgf")


@dataclass(frozen=True)
class Job:
    environment: str
    method: str
    seed: int
    cpu: int
    command: list[str]
    run_dir: Path
    log_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO / path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cpu_ids(value: str) -> list[int]:
    """Parse ``31-40,45`` style CPU lists."""
    out: list[int] = []
    seen: set[int] = set()
    for part in (p.strip() for p in value.split(",")):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo < 0 or hi < lo:
                raise argparse.ArgumentTypeError(f"Invalid CPU range {part!r}.")
            values: Any = range(lo, hi + 1)
        else:
            if int(part) < 0:
                raise argparse.ArgumentTypeError("CPU ids must be non-negative.")
            values = [int(part)]
        for cpu in values:
            if cpu not in seen:
                seen.add(cpu)
                out.append(cpu)
    if not out:
        raise argparse.ArgumentTypeError("No CPU ids parsed.")
    return out


def reference_config(environment: str, seed: int) -> dict[str, Any]:
    """The PSPO control seed whose settings this ablation mirrors."""
    cohort = CONTROL_COHORTS[environment]
    return _read_json(RUNS / cohort / "two_hidden" / environment / f"seed{seed}" / "config.json")


def base_policy_path(environment: str) -> Path:
    cohort = CONTROL_COHORTS[environment]
    path = RUNS / cohort / "two_hidden" / environment / "initial_base_policy" / "base_policy.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing PSPO base policy for {environment}: {path}")
    return path


def build_command(
    environment: str,
    method: str,
    seed: int,
    *,
    output_root: Path,
    warm_start: bool,
    smoke: bool,
) -> tuple[list[str], Path]:
    """Build one stage invocation, mirroring the PSPO control's settings."""
    config = reference_config(environment, seed)
    hp = config["training_hyperparameters"]
    architecture = config["base_policy_architecture"]
    script, extra = METHODS[method]

    arm = "pspo_init" if warm_start else "cold_start"
    run_dir = output_root / arm / environment / method / f"seed{seed}"

    total_timesteps = 2048 if smoke else int(config["total_timesteps"])
    eval_episodes = 2 if smoke else int(config["eval_episodes"])

    command = [
        str(REPO / ".venv/bin/python"),
        str(STAGES / script),
        "--env-id", str(config["env_id"]),
        "--env-kwargs", json.dumps(config.get("env_kwargs") or {}, sort_keys=True),
        "--max-episode-steps", str(config["max_episode_steps"]),
        "--cost-limit", str(config["cost_limit"]),
        "--total-timesteps", str(total_timesteps),
        "--eval-episodes", str(eval_episodes),
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
        # The warm start only fits an actor of the same shape, so every arm --
        # including the cold-start control -- uses PSPO's architecture.
        "--n-hidden", str(architecture["n_hidden"]),
        "--hidden-dim", str(architecture["hidden_dim"]),
        "--device", str(config.get("device", "cpu")),
        "--success-reward-threshold", str(config["success_reward_threshold"]),
        "--evaluation-policy", "unshielded",
        "--output-dir", str(run_dir.parent),
        "--run-id", f"seed{seed}",
        *extra,
    ]
    # PPO-Shield needs the shield; the unshielded baselines must not get one.
    if method == "ppo_shield":
        command += ["--shield-path", str(_resolve(config["shield_path"]))]
    if warm_start:
        command += ["--init-policy-path", str(base_policy_path(environment))]
    return command, run_dir


def plan_jobs(args: argparse.Namespace) -> list[Job]:
    arms = ["pspo_init"] + ([] if args.no_cold_start else ["cold_start"])
    cpus = args.cpu_ids or list(range(len(args.seeds)))
    jobs: list[Job] = []
    index = 0
    for environment in args.envs:
        for method in args.methods:
            for arm in arms:
                for seed in args.seeds:
                    command, run_dir = build_command(
                        environment,
                        method,
                        seed,
                        output_root=args.output_root,
                        warm_start=(arm == "pspo_init"),
                        smoke=args.smoke,
                    )
                    if args.skip_existing and (run_dir / "metrics.json").is_file():
                        continue
                    log_path = args.output_root / "_logs" / f"{arm}_{environment}_{method}_seed{seed}.log"
                    jobs.append(
                        Job(environment, method, seed, cpus[index % len(cpus)],
                            command, run_dir, log_path)
                    )
                    index += 1
    return jobs


def write_manifest(args: argparse.Namespace, jobs: list[Job]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "intent": (
            "Warm-start safe-RL baselines from PSPO's safe policy initialisation, "
            "paired against their cold-start controls."
        ),
        "environments": list(args.envs),
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "arms": ["pspo_init"] + ([] if args.no_cold_start else ["cold_start"]),
        "warm_start_scope": "actor only; critics, multipliers and optimizer state start fresh",
        "safety_caveat": (
            "No baseline receives a certified region or projection, so any safety "
            "change here is empirical and carries no guarantee."
        ),
        "base_policies": {
            environment: {
                "path": str(base_policy_path(environment)),
                "sha256": _sha256(base_policy_path(environment)),
                "cohort": CONTROL_COHORTS[environment],
            }
            for environment in args.envs
        },
        "n_jobs": len(jobs),
        "smoke": bool(args.smoke),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_job(job: Job) -> tuple[Job, int, float]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.run_dir.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{REPO / 'core'}:{REPO}:{env.get('PYTHONPATH', '')}",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    started = time.time()
    with job.log_path.open("w", encoding="utf-8") as handle:
        rc = subprocess.run(
            ["taskset", "-c", str(job.cpu), *job.command],
            cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    return job, rc, time.time() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS), choices=list(ENVIRONMENTS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=None,
                        help="Comma-separated ids/ranges, e.g. 30-39. Defaults to 0..len(seeds)-1.")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument(
        "--output-root", type=Path,
        default=RUNS / "baselines_pspo_init_ablation",
    )
    parser.add_argument("--no-cold-start", action="store_true",
                        help="Only run the warm-started arm (their cold-start controls already exist elsewhere).")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", dest="skip_existing", action="store_false")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny horizon/rollout for a wiring check. Not a study result.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = plan_jobs(args)

    print(f"environments : {', '.join(args.envs)}")
    print(f"methods      : {', '.join(args.methods)}")
    print(f"seeds        : {args.seeds}")
    print(f"output root  : {args.output_root}")
    print(f"jobs         : {len(jobs)}")
    for environment in args.envs:
        print(f"  base policy [{environment}]: {base_policy_path(environment)}")

    if args.dry_run:
        for job in jobs:
            print(f"\ndry-run {job.environment}/{job.method}/seed{job.seed} cpu={job.cpu}")
            print("  " + " ".join(job.command))
        return 0

    if not jobs:
        print("nothing to do (all requested runs already have metrics.json)")
        return 0

    write_manifest(args, jobs)
    workers = args.max_parallel or len(args.cpu_ids or []) or len(args.seeds)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            job, rc, seconds = future.result()
            status = "ok" if rc == 0 else f"FAIL rc={rc}"
            failures += int(rc != 0)
            print(f"{status} {job.environment}/{job.method}/seed{job.seed} "
                  f"cpu={job.cpu} {seconds:.0f}s -> {job.log_path}", flush=True)
    print(f"\ncompleted with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
