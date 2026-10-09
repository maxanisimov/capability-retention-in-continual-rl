#!/usr/bin/env python3
"""Prepare and supervise shaped FrozenLake PPO, PSPO and safe-RL sweeps."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-shaping-matplotlib")
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import stable_baselines3 as sb3  # noqa: E402
import torch  # noqa: E402
from provably_safe_policy_optimisation import ProvablySafePPO  # noqa: E402

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_ppo_shaping as ppo,
)
from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_pspo_shaping as pspo,
)
from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_safe_baseline_shaping as safe_baseline,
)
from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_stochastic_frozenlake_baselines as baseline,
)
from projects.safe_policy_optimisation.utils.cli import net_arch_from_args  # noqa: E402
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
    synthesise_shield,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (  # noqa: E402
    FrozenLakePotentialReward,
)
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    evaluate_shielded_total_rewards,
    evaluate_unshielded_total_rewards,
)

METHODS = ("pspo", "ppo_shield", "ppo")
SAFE_BASELINES = ("ppo_lagrangian", "ppo_pid_lagrangian", "cpo", "ppo_shield")
SUPPORTED_METHODS = (*METHODS, *SAFE_BASELINES[:-1])
THREAD_ENV = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    )
}
THREAD_ENV["MPLCONFIGDIR"] = "/tmp/ccl-frozenlake-shaping-matplotlib"
HYPER_KEYS = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def policy_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def all_source_paths() -> list[Path]:
    paths = set(pspo.source_paths())
    paths.update(safe_baseline.source_paths())
    paths.update((REPO / "core/abstract_gradient_training").rglob("*.py"))
    paths.update(Path(module.__file__).resolve() for module in (ppo, pspo, baseline))
    paths.add(Path(__file__).resolve())
    return sorted(paths)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_shield_model(
    model, factory, *, mask, episodes: int, seed: int, shield_on: bool
):
    evaluate = (
        evaluate_shielded_total_rewards
        if shield_on
        else evaluate_unshielded_total_rewards
    )
    rows = evaluate(
        model,
        factory,
        episodes=episodes,
        seed=seed,
        reward_threshold=0,
        shield_mask=mask,
    )
    metrics = {
        "algorithm": "ppo_shield" if shield_on else "ppo_shield_nominal",
        "eval_episodes": len(rows),
        "reward": {
            "mean_total_reward": float(np.mean([r["total_reward"] for r in rows]))
        },
        "safety": {"safety_rate": float(np.mean([r["safe_trajectory"] for r in rows]))},
        "success": {
            "success_mode": "goal_reached",
            "success_count": sum(r["success"] for r in rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        },
        "mean_episode_length": float(np.mean([r["length"] for r in rows])),
        "evaluation_policy": "greedy_runtime_shield"
        if shield_on
        else "greedy_unshielded",
        "runtime_shield": shield_on,
        "reward_shaping_enabled": False,
    }
    return rows, metrics


def run_shield_worker(args: argparse.Namespace) -> dict:
    if args.size < 16 or args.total_timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Require size >= 16 and positive budgets")
    if args.curve_eval_freq < 0 or args.curve_eval_episodes <= 0:
        raise ValueError("Invalid curve-evaluation settings")
    torch.set_num_threads(1)
    root = args.output_dir / (
        args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    cap = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else 5000 * args.size // 128
    )
    if cap <= 0:
        raise ValueError("Episode cap must be positive")
    kwargs = {
        "size": args.size,
        "is_slippery": True,
        "success_rate": args.success_rate,
        "step_penalty": args.step_penalty,
    }

    def raw_factory():
        return gym.make(ENV_ID, max_episode_steps=cap, **kwargs)

    raw = raw_factory()
    started = time.perf_counter()
    mask, winning, _, _ = synthesise_shield(raw.unwrapped)
    shield_seconds = time.perf_counter() - started
    if not winning[0]:
        raise ValueError("Start state is not safety-winning")
    torch.save(
        {"shield": torch.as_tensor(mask), "winning_states": torch.as_tensor(winning)},
        root / "shield_q.pt",
    )
    logger = LearningCurveLogger(
        curve_dir=root / "learning_curves", tensorboard_log_dir=root / "tensorboard"
    )
    training = ppo.TrainingRewardLogger(
        root / "training_episodes.csv", gamma=args.gamma, scale=args.shaping_scale
    )
    curves = [
        ppo.TimedRewardCurve(
            env_factory=raw_factory,
            curve_logger=logger,
            eval_freq=args.curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30000,
            reward_threshold=0,
            shield_mask=mask,
            apply_shield=shield_on,
        )
        for shield_on in (False, True)
    ]
    hyper = {key: getattr(args, key) for key in HYPER_KEYS}
    env = None
    try:
        started = time.perf_counter()
        model = ProvablySafePPO(
            "MlpPolicy",
            raw,
            shield=mask,
            obs_to_state=raw.unwrapped.make_obs_to_state(),
            shield_seed=args.seed,
            shield_action_storage="proposed",
            **hyper,
            policy_kwargs={"net_arch": net_arch_from_args(args)},
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
        model_seconds = time.perf_counter() - started
        initial_hash = policy_hash(model)
        env = FrozenLakePotentialReward(
            raw,
            gamma=args.gamma,
            scale=args.shaping_scale,
            timeout_mode=args.timeout_mode,
        )
        model.set_env(env)
        model.get_env().seed(args.seed)
        assert policy_hash(model) == initial_hash
        paths = all_source_paths()
        config = {
            "algorithm": "ppo_shield",
            "seed": args.seed,
            "env_id": ENV_ID,
            "env_kwargs": kwargs,
            "max_episode_steps": cap,
            "requested_timesteps": args.total_timesteps,
            "eval_episodes": args.eval_episodes,
            "curve_eval_freq": args.curve_eval_freq,
            "curve_eval_episodes": args.curve_eval_episodes,
            "device": args.device,
            "architecture": {"hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden},
            "training_hyperparameters": hyper,
            "shaping": {
                "enabled": args.shaping_scale != 0,
                "scale": args.shaping_scale,
                "gamma": args.gamma,
                "timeout_mode": args.timeout_mode,
                "training_only": True,
                "formula": "r + scale * (gamma * Phi(next_state) - Phi(state))",
                "distance_graph_uses_shield": False,
            },
            "initialization": {
                "random_ppo": True,
                "warm_start": False,
                "goal_information_used": False,
                "reward_information_used": False,
                "policy_sha256": initial_hash,
                "actor_and_critic_unchanged_by_shaping": True,
            },
            "shield": {
                "training": True,
                "primary_evaluation": True,
                "nominal_evaluation": False,
                "action_storage": "proposed",
                "safety_winning_states": int(winning.sum()),
            },
            "source_sha256": {
                str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "versions": {
                "python": platform.python_version(),
                "gymnasium": gym.__version__,
                "stable_baselines3": sb3.__version__,
                "torch": torch.__version__,
            },
        }
        atomic_json(root / "config.json", config)
        with zipfile.ZipFile(
            root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
        ) as z:
            for p in paths:
                z.write(p, str(p.relative_to(REPO)))
        np.savez_compressed(
            root / "potential.npz", distances=env.distances, potential=env.potential
        )
        for shield_on, name in (
            (True, "initial_metrics.json"),
            (False, "initial_metrics_nominal.json"),
        ):
            _, metrics = evaluate_shield_model(
                model,
                raw_factory,
                mask=mask,
                episodes=args.curve_eval_episodes,
                seed=args.seed + 10000,
                shield_on=shield_on,
            )
            atomic_json(root / name, metrics)
        model.set_exploration_unsafe_action_callback(logger.log_exploration_unsafe)
        logger.start_timing()
        started = time.perf_counter()
        model.learn(total_timesteps=args.total_timesteps, callback=[training, *curves])
        learn_seconds = time.perf_counter() - started
        model.save(root / "model.zip")
        started = time.perf_counter()
        rows, metrics = evaluate_shield_model(
            model,
            raw_factory,
            mask=mask,
            episodes=args.eval_episodes,
            seed=args.seed + 10000,
            shield_on=True,
        )
        shield_eval_seconds = time.perf_counter() - started
        atomic_json(root / "metrics.json", metrics)
        save_csv(root / "episodes.csv", rows)
        started = time.perf_counter()
        rows, nominal = evaluate_shield_model(
            model,
            raw_factory,
            mask=mask,
            episodes=args.eval_episodes,
            seed=args.seed + 10000,
            shield_on=False,
        )
        nominal_eval_seconds = time.perf_counter() - started
        atomic_json(root / "metrics_nominal.json", nominal)
        save_csv(root / "episodes_nominal.csv", rows)
        curve_seconds = sum(curve.evaluation_seconds for curve in curves)
        summary = {
            "algorithm": "ppo_shield",
            "final_timesteps": model.num_timesteps,
            "completed_training_episodes": training.episode,
            "learn_wall_seconds_including_curve_evaluation": learn_seconds,
            "curve_evaluation_seconds": curve_seconds,
            "training_seconds_excluding_curve_evaluation": learn_seconds
            - curve_seconds,
            "final_evaluation_seconds": shield_eval_seconds,
            "final_nominal_evaluation_seconds": nominal_eval_seconds,
            "shield_synthesis_seconds": shield_seconds,
            "model_initialisation_seconds": model_seconds,
            "initial_policy_sha256": initial_hash,
            "max_discounted_telescoping_error": training.max_telescoping_error,
            "training_shield_diagnostics": model.shield_diagnostics(),
            "metrics": metrics,
            "nominal_metrics": nominal,
        }
        atomic_json(root / "summary.json", summary)
        print(json.dumps({"run_dir": str(root), **summary}, indent=2), flush=True)
        return summary
    finally:
        (raw if env is None else env).close()
        training.close()
        logger.close()


def build_manifest(args: argparse.Namespace, cpus: list[int]) -> dict:
    methods = tuple(getattr(args, "methods", METHODS))
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - set(SUPPORTED_METHODS)
    ):
        raise ValueError("Select distinct supported methods")
    seeds = [0] if args.smoke else list(range(10))
    required = len(methods) * len(seeds)
    if len(cpus) < required or len(set(cpus[:required])) != required:
        raise ValueError(f"Need {required} distinct physical cores")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite or resume {output}")
    jobs = []
    for seed in seeds:
        for method in methods:
            script = {
                "ppo": Path(ppo.__file__),
                "pspo": Path(pspo.__file__),
                "ppo_shield": Path(__file__),
                **{m: Path(safe_baseline.__file__) for m in SAFE_BASELINES[:-1]},
            }[method].resolve()
            command = [
                "taskset",
                "-c",
                str(cpus[len(jobs)]),
                str(REPO / ".venv/bin/python"),
                "-u",
                str(script),
            ]
            if method == "ppo_shield":
                command.append("--worker")
            elif method in SAFE_BASELINES[:-1]:
                command += ["--algorithm", method, "--cost-limit", "0"]
            common = [
                "--size",
                str(args.size),
                "--seed",
                str(seed),
                "--shaping-scale",
                "1",
                "--gamma",
                "0.999",
                "--timeout-mode",
                "bootstrap",
                "--total-timesteps",
                str(64 if args.smoke else args.total_timesteps),
                "--max-episode-steps",
                str(16 if args.smoke else 5000 * args.size // 128),
                "--eval-episodes",
                str(2 if args.smoke else args.eval_episodes),
                "--curve-eval-episodes",
                str(2 if args.smoke else 10),
                "--curve-eval-freq",
                str(32 if args.smoke else 20000),
                "--device",
                "cpu",
                "--output-dir",
                str(output / method),
                "--run-id",
                f"seed{seed}",
            ]
            if args.smoke:
                common += [
                    "--n-steps",
                    "32",
                    "--batch-size",
                    "16",
                    "--n-epochs",
                    "1",
                    "--verbose",
                    "0",
                ]
            if method == "pspo":
                common += [
                    "--safety-frequency",
                    "100",
                    "--lid-iters",
                    str(1 if args.smoke else 200),
                    "--lid-checkpoint",
                    str(1 if args.smoke else 100),
                    "--lid-batch-size",
                    str(64 if args.smoke else 256),
                    "--state-representation",
                    "one_hot",
                ]
            command += common
            jobs.append(
                {
                    "id": f"{method}/seed{seed}",
                    "method": method,
                    "seed": seed,
                    "cpu": cpus[len(jobs)],
                    "directory": str(output / method / f"seed{seed}"),
                    "command": command,
                }
            )
    paths = all_source_paths()
    manifest = {
        "created_utc": utc_now(),
        "hostname": socket.gethostname(),
        "output_dir": str(output),
        "screen_session": args.screen_session,
        "size": args.size,
        "smoke": args.smoke,
        "seeds": seeds,
        "methods": list(methods),
        "requested_timesteps_per_run": 64 if args.smoke else args.total_timesteps,
        "eval_episodes_per_run": 2 if args.smoke else args.eval_episodes,
        "training_reward_shaping": True,
        "shaping_scale": 1,
        "gamma": 0.999,
        "runtime_evaluation_shield": {
            method: method == "ppo_shield" for method in methods
        },
        "pspo_initialization": "safety mask only; no reward, goal distances or witnesses",
        "pspo_safety_frequency_rollouts": 100,
        "state_representation": "one_hot",
        "threads_per_job": 1,
        "jobs": jobs,
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ("_logs", "_process", "_orchestrator"):
        (output / subdir).mkdir()
    with zipfile.ZipFile(
        output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in paths:
            z.write(p, str(p.relative_to(REPO)))
    return manifest


def run_job(job: dict, output: Path) -> dict:
    path = output / "_process" / (job["id"].replace("/", "_") + ".json")
    log_path = output / "_logs" / (job["id"].replace("/", "_") + ".log")
    started = time.perf_counter()
    record = {
        "job_id": job["id"],
        "method": job["method"],
        "seed": job["seed"],
        "cpu": job["cpu"],
        "started_utc": utc_now(),
        "status": "starting",
        "log_path": str(log_path),
    }
    atomic_json(path, record)
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                job["command"],
                cwd=REPO,
                env={
                    **os.environ,
                    **THREAD_ENV,
                    "PYTHONPATH": f"{REPO / 'core'}:{REPO}",
                    "PYTHONUNBUFFERED": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record.update(pid=process.pid, status="running")
            atomic_json(path, record)
            returncode = process.wait()
        directory = Path(job["directory"])
        complete = (
            returncode == 0
            and (directory / "summary.json").is_file()
            and (directory / "metrics.json").is_file()
            and (
                directory
                / ("model.pt" if job["method"] in SAFE_BASELINES[:-1] else "model.zip")
            ).is_file()
        )
        record.update(
            returncode=returncode, status="complete" if complete else "failed"
        )
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
    record.update(
        finished_utc=utc_now(), process_wall_seconds=time.perf_counter() - started
    )
    atomic_json(path, record)
    return record


def write_report(output: Path, manifest: dict, results: list[dict]) -> None:
    rows = []
    for job in manifest["jobs"]:
        root = Path(job["directory"])
        record = next((r for r in results if r["job_id"] == job["id"]), None)
        if record is None or record["status"] != "complete":
            continue
        summary = read_json(root / "summary.json")
        for label, name in (
            (job["method"], "metrics.json"),
            ("ppo_shield_nominal", "metrics_nominal.json"),
        ):
            if name == "metrics_nominal.json" and job["method"] != "ppo_shield":
                continue
            metrics = read_json(root / name)
            rows.append(
                {
                    "method": label,
                    "seed": job["seed"],
                    "total_reward": metrics["reward"]["mean_total_reward"],
                    "safety_rate": metrics["safety"]["safety_rate"],
                    "goal_success": metrics["success"]["success_rate"],
                    "training_seconds": summary[
                        "training_seconds_excluding_curve_evaluation"
                    ],
                    "lid_seconds": summary.get("lid_computation_seconds", 0),
                }
            )
    if rows:
        save_csv(output / "seed_results.csv", rows)
    aggregate = {}
    for method in (*SUPPORTED_METHODS, "ppo_shield_nominal"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        aggregate[method] = {"seed_count": len(group)}
        for key in (
            "total_reward",
            "safety_rate",
            "goal_success",
            "training_seconds",
            "lid_seconds",
        ):
            values = np.asarray([row[key] for row in group], dtype=float)
            aggregate[method][key] = {
                "mean": float(values.mean()),
                "two_standard_errors": float(
                    2 * values.std(ddof=1) / np.sqrt(len(values))
                )
                if len(values) > 1
                else None,
            }
    atomic_json(
        output / "aggregate.json",
        {
            "uncertainty": "mean +/- two standard errors over training seeds; incomplete methods are explicitly counted",
            "methods": aggregate,
        },
    )
    complete = sum(r["status"] == "complete" for r in results)
    failed = sum(r["status"] == "failed" for r in results)
    atomic_json(
        output / "status.json",
        {
            "updated_utc": utc_now(),
            "total": len(manifest["jobs"]),
            "complete": complete,
            "failed": failed,
            "running_or_starting": len(manifest["jobs"]) - len(results),
            "complete_jobs": [
                r["job_id"] for r in results if r["status"] == "complete"
            ],
            "failed_jobs": [r["job_id"] for r in results if r["status"] == "failed"],
        },
    )


def supervise(path: Path) -> int:
    manifest = read_json(path)
    output = Path(manifest["output_dir"])
    with (output / "_orchestrator/supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "_orchestrator/supervisor.json").exists():
            raise FileExistsError(
                "This manifest was already started; refusing duplicate jobs"
            )
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed after preparation: {name}")
        atomic_json(
            output / "_orchestrator/supervisor.json",
            {
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "screen_session": manifest["screen_session"],
            },
        )
        results = []
        write_report(output, manifest, results)
        print(
            f"Launching {len(manifest['jobs'])} jobs concurrently on distinct physical cores",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=len(manifest["jobs"])) as executor:
            futures = [
                executor.submit(run_job, job, output) for job in manifest["jobs"]
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"{result['status']} {result['job_id']} {result['process_wall_seconds']:.1f}s",
                    flush=True,
                )
                write_report(output, manifest, results)
        atomic_json(
            output / "_orchestrator/completion.json",
            {
                "finished_utc": utc_now(),
                "complete": sum(r["status"] == "complete" for r in results),
                "failed": sum(r["status"] == "failed" for r in results),
            },
        )
        return int(any(r["status"] == "failed" for r in results))


def main() -> int:
    if "--worker" in sys.argv:
        parser = ppo.build_parser()
        parser.description = (
            "PPO-Shield FrozenLake shaping worker, random initialization"
        )
        parser.add_argument("--worker", action="store_true")
        run_shield_worker(parser.parse_args())
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--methods", nargs="+", choices=SUPPORTED_METHODS, default=METHODS
    )
    parser.add_argument("--screen-session", default="frozenlake32-shaping")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise)
    if (
        args.output_dir is None
        or args.size < 16
        or args.total_timesteps <= 0
        or args.eval_episodes <= 0
    ):
        parser.error("Require --output-dir, size >= 16, and positive budgets")
    cpus, evidence = baseline.free_physical_cpus(95.0)
    cpus = sorted(cpus, key=lambda cpu: (cpu < 32, cpu))
    manifest = build_manifest(args, cpus)
    manifest["cpu_availability_evidence"] = evidence
    path = args.output_dir.resolve() / "_orchestrator/launch_manifest.json"
    atomic_json(path, manifest)
    print(f"Prepared {len(manifest['jobs'])} jobs: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
