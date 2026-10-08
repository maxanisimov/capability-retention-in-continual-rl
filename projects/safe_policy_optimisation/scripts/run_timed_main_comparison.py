#!/usr/bin/env python3
"""Detached, CPU-pinned timing sweep; only lookup PSPO persists reward/safety.

Reuse the main comparison's configurations and budgets, freshly fit reward-free
shared PSPO initialisations, and run each baseline independently. Periodic/final
evaluations still execute for matched workloads, but all ordinary stage outputs
(CSV, TensorBoard, JSON, checkpoints, stdout) are discarded in memory. Only the
explicit timing result and optional lookup final metrics are persisted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core"), str(Path(__file__).resolve().parent)]
STAGES = {
    "ppo_policy": "train_ppo",
    "ppo_shield": "train_ppo_shield",
    "ppo_lagrangian": "train_ppo_lagrangian",
    "ppo_pid_lagrangian": "train_ppo_lagrangian",
    "cpo": "train_cpo",
    "pspo": "train_pspo",
    "pspo_lookup": "train_pspo",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


class NullTextIO(io.TextIOBase):
    """Non-buffering sink: millions of CSV rows must not accumulate in RAM."""

    def writable(self):
        return True

    def write(self, value):
        return len(value)


class NullWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


@contextlib.contextmanager
def discard_stage_outputs(root: Path, captured: dict):
    """Scoped suppression, not deleting metrics after storing them.

    Input reads remain real. Existing project files are untouched. Model saves
    are unnecessary for this benchmark. Final lookup metrics are captured in
    memory and explicitly whitelisted after leaving this context.
    """
    from stable_baselines3.common.base_class import BaseAlgorithm

    from projects.safe_policy_optimisation.stages import train_ppo_lagrangian
    from projects.safe_policy_optimisation.utils import learning_curves

    original_open = Path.open

    def open_path(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+") and path.is_relative_to(root):
            if "b" in mode:
                raise RuntimeError(f"Unexpected binary stage output: {path}")
            return NullTextIO()
        return original_open(path, mode, *args, **kwargs)

    def capture(path, payload):
        if Path(path).name == "metrics.json":
            captured["metrics"] = payload
        elif Path(path).name == "summary.json":
            captured["summary"] = payload

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(Path, "open", open_path))
        stack.enter_context(patch.object(learning_curves, "SummaryWriter", NullWriter))
        stack.enter_context(patch.object(BaseAlgorithm, "save", lambda *a, **k: None))
        stack.enter_context(
            patch.object(train_ppo_lagrangian, "save_checkpoint", lambda *a, **k: None)
        )
        for module_name in set(STAGES.values()):
            module = importlib.import_module(
                "projects.safe_policy_optimisation.stages." + module_name
            )
            if hasattr(module, "write_json"):
                stack.enter_context(patch.object(module, "write_json", capture))
        with (
            contextlib.redirect_stdout(NullTextIO()),
            contextlib.redirect_stderr(NullTextIO()),
        ):
            yield


def cli_arguments(parser: argparse.ArgumentParser, values: dict) -> list[str]:
    result = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        value = values.get(action.dest)
        if value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                result.append(option)
        elif action.nargs in ("+", "*"):
            result.extend([option, *map(str, value)])
        else:
            text = (
                json.dumps(value)
                if isinstance(value, dict)
                else (str(value).lower() if isinstance(value, bool) else str(value))
            )
            result.extend([option, text])
    return result


def training_values(job: dict, run_dir: Path, cpu: int, smoke: bool) -> dict:
    config = job["config"]
    values = {
        **config,
        **config.get("training_hyperparameters", {}),
        **config.get("baseline_hyperparameters", {}),
        **config.get("adaptive", {}),
    }
    values.update(
        output_dir=run_dir.parent,
        run_id=run_dir.name,
        total_timesteps=job["budget"],
        seed=job["seed"],
        hidden_dim=64,
        n_hidden=2,
        device="cpu",
        tensorboard_log_dir=None,
        jobs=1,
        torch_num_threads=1,
        cpu_ids=str(cpu),
        init_policy_path=None,
        algorithms=[job["method"]],
    )
    if job["method"].startswith("pspo"):
        adaptive = config["adaptive"]
        values.update(
            base_policy_path=job["base_policy_path"],
            state_representation=job["representation"],
            freq=adaptive["frequency"],
            directional=adaptive["directional_rashomon_growth"],
            region_mode=adaptive["region_update_mode"],
            region_refresh="adaptive",
            adaptive_granularity=None,
        )
    if smoke:
        values.update(
            total_timesteps=8,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            eval_episodes=2,
            curve_eval_freq=8,
            curve_eval_episodes=2,
            rashomon_n_iters=2,
            rashomon_initial_n_iters=2,
            rashomon_recompute_n_iters=2,
            rashomon_checkpoint=1,
        )
    return values


def run_initialisation(job: dict, run_dir: Path) -> dict:
    import torch

    from projects.safe_policy_optimisation.stages import (
        compute_shield_rashomon_set as stage,
    )

    config = job["config"]
    values = {
        "output_dir": run_dir.parent,
        "run_id": run_dir.name,
        "base_policy_only": True,
        "shield_path": config["shield_path"],
        "env_id": config["env_id"],
        "env_kwargs": config["env_kwargs"],
        "state_representation": job["representation"],
        "seed": 0,
        "device": "cpu",
        "hidden_dim": 64,
        "n_hidden": 2,
        "bc_target_margin": 2.0,
        "linear_init_margin": 2.0,
        "bc_margin_loss_weight": 1.0,
        "bc_margin_mode": "all",
        "bc_initialisation_objective": "margin",
        "bc_safe_action_entropy_weight": 1.0,
        "bc_min_safe_action_entropy": 0.95,
        "rashomon_batch_size": "auto",
        "certificate_samples": config["adaptive"]["certificate_samples"],
    }
    args = stage.build_parser().parse_args(cli_arguments(stage.build_parser(), values))
    original_save = torch.save
    original_fit = stage.fit_base_policy
    fit_seconds = 0.0

    def timed_fit(*args, **kwargs):
        nonlocal fit_seconds
        started = time.perf_counter()
        try:
            return original_fit(*args, **kwargs)
        finally:
            fit_seconds += time.perf_counter() - started

    def save_base(payload, path, *args, **kwargs):
        if Path(path).name == "base_policy.pt":
            # Keep only parameters needed by RL; no initialisation outcome metrics.
            original_save(
                {
                    "architecture": payload["architecture"],
                    "state_dict": payload["state_dict"],
                },
                path,
                *args,
                **kwargs,
            )

    with (
        patch.object(torch, "save", save_base),
        patch.object(stage, "fit_base_policy", timed_fit),
        patch.object(stage, "write_json", lambda *a, **k: None),
        contextlib.redirect_stdout(NullTextIO()),
        contextlib.redirect_stderr(NullTextIO()),
    ):
        torch.manual_seed(0)
        summary = stage.run(args)
    if not summary["base_policy"]["reached_target"]:
        raise RuntimeError(
            "Reward-free PSPO initialisation did not reach its safety/entropy target"
        )
    result = {
        "policy_initialisation_s": summary["timing"][
            "policy_initialisation_wall_time_s"
        ],
        "behaviour_cloning_fit_s": fit_seconds,
        "shared_across_seeds": True,
        "initialisation_seed": 0,
        "reward_signal_available": False,
        "architecture": summary["architecture"],
    }
    write_json(
        run_dir / "summary.json",
        {
            "timing": {
                "policy_initialisation_wall_time_s": result["policy_initialisation_s"]
            },
            "architecture": summary["architecture"],
            "reward_signal_available": False,
        },
    )
    return result


def run_training(job: dict, run_dir: Path, cpu: int, smoke: bool) -> dict:
    from provably_safe_policy_optimisation.adaptive_safe_ppo_v2 import AdaptiveSafePPOV2
    from safe_rl_baselines import CPO, PPOLagrangian
    from stable_baselines3 import PPO

    module = importlib.import_module(
        "projects.safe_policy_optimisation.stages." + STAGES[job["method"]]
    )
    parser = module.build_parser()
    argv = cli_arguments(parser, training_values(job, run_dir, cpu, smoke))
    args = (
        module.parse_args(argv)
        if job["method"].startswith("pspo")
        else parser.parse_args(argv)
    )
    timers = {"learn_s": 0.0, "finalise_s": 0.0}
    captured = {}

    def timed_method(original, key):
        def wrapped(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                timers[key] += time.perf_counter() - started

        return wrapped

    with contextlib.ExitStack() as stack:
        for cls in (PPO, PPOLagrangian, CPO):
            stack.enter_context(
                patch.object(cls, "learn", timed_method(cls.learn, "learn_s"))
            )
        stack.enter_context(
            patch.object(
                AdaptiveSafePPOV2,
                "finalize_adaptive_update",
                timed_method(AdaptiveSafePPOV2.finalize_adaptive_update, "finalise_s"),
            )
        )
        stack.enter_context(discard_stage_outputs(run_dir, captured))
        started = time.perf_counter()
        summary = module.run(args)
        stage_s = time.perf_counter() - started
    section = summary.get(job["method"], summary)
    result = {
        "training_loop_s": timers["learn_s"],
        "adaptive_finalisation_s": timers["finalise_s"],
        "rl_stage_s": stage_s,
        "actual_timesteps": int(section["final_timesteps"]),
        "periodic_evaluations_enabled": True,
        "model_and_curve_outputs_saved": False,
    }
    if job["method"].startswith("pspo"):
        diagnostics = summary["adaptive_diagnostics"]
        result["lid_computation_count"] = diagnostics.get("rashomon_computations")
        for label, field in {
            "lid_s": "rashomon_wall_time_total_s",
            "verification_s": "exact_verification_wall_time_total_s",
            "projection_s": "projection_wall_time_total_s",
            "safety_enforcement_s": "safety_enforcement_wall_time_total_s",
            "lid_engine_s": "rashomon_engine_wall_time_total_s",
            "lid_calibration_s": "rashomon_calibration_wall_time_total_s",
        }.items():
            result[label] = diagnostics.get(field)
        base_summary = read_json(Path(job["base_policy_path"]).parent / "summary.json")
        result.update(
            policy_initialisation_s=base_summary["timing"][
                "policy_initialisation_wall_time_s"
            ],
            initialisation_shared_across_seeds=True,
            initialisation_source=str(
                Path(job["base_policy_path"]).parent / "training_time.json"
            ),
        )
    if job["method"] == "pspo_lookup":
        metrics = captured["metrics"]
        write_json(
            run_dir / "metrics.json",
            {
                "eval_episodes": metrics["eval_episodes"],
                "total_reward": metrics["reward"]["mean_total_reward"],
                "safety_rate": metrics["safety"]["safety_rate"],
            },
        )
    return result


def worker(manifest: dict, job: dict, cpu: int) -> int:
    started = time.perf_counter()
    os.sched_setaffinity(0, {cpu})
    os.environ.update(THREAD_ENV)
    os.environ["MPLCONFIGDIR"] = f"/tmp/pspo-timing-{os.getpid()}"
    run_dir = Path(manifest["output_dir"]) / job["relative_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": job["environment"],
        "method": job["method"],
        "seed": job.get("seed"),
        "nominal_timesteps": job.get("budget"),
        "representation": job.get("representation", "one_hot"),
        "hostname": socket.gethostname(),
        "cpu_id": cpu,
        "threads": 1,
        "job_id": job["id"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result.update(
            run_initialisation(job, run_dir)
            if job["kind"] == "initialisation"
            else run_training(job, run_dir, cpu, manifest["smoke"])
        )
        result["status"] = "complete"
        exit_code = 0
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        exit_code = 1
    result.update(
        worker_elapsed_s=time.perf_counter() - started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(run_dir / "training_time.json", result)
    return exit_code


def build_manifest(
    output: Path, seeds: list[int], environments: list[str] | None, smoke: bool
) -> dict:
    from generate_final_policy_table import METHODS
    from plot_extended_budget_learning_curves import ENVIRONMENTS

    jobs = []
    if environments is not None and set(environments) - {
        env.key for env in ENVIRONMENTS
    }:
        raise ValueError("Unknown environment in requested timing sweep")
    selected = [
        env for env in ENVIRONMENTS if environments is None or env.key in environments
    ]
    for env in selected:
        config = read_json(env.adaptive_root / "seed0/config.json")
        for representation in ("one_hot", "state_id_lookup"):
            name = "pspo" if representation == "one_hot" else "pspo_lookup"
            jobs.append(
                {
                    "id": f"init/{env.key}/{name}",
                    "kind": "initialisation",
                    "environment": env.key,
                    "method": name,
                    "representation": representation,
                    "relative_dir": f"initialisation/{env.key}/{name}",
                    "config": config,
                }
            )
    for env in selected:
        for seed in seeds:
            for method in METHODS:
                root = env.adaptive_root if method.adaptive else env.baseline_root
                path = (
                    root
                    / f"seed{seed}"
                    / Path(method.relative_path).parent
                    / "config.json"
                )
                config = read_json(path)
                names = ("pspo", "pspo_lookup") if method.adaptive else (method.key,)
                for name in names:
                    job = {
                        "id": f"run/{env.key}/seed{seed}/{name}",
                        "kind": "training",
                        "environment": env.key,
                        "method": name,
                        "seed": seed,
                        "budget": env.nominal_budget,
                        "config": config,
                        "configuration_source": str(path),
                        "relative_dir": f"runs/{env.key}/seed{seed}/{name}",
                    }
                    if method.adaptive:
                        job.update(
                            dependency=f"init/{env.key}/{name}",
                            representation="state_id_lookup"
                            if name == "pspo_lookup"
                            else "one_hot",
                            base_policy_path=str(
                                output
                                / f"initialisation/{env.key}/{name}/base_policy.pt"
                            ),
                        )
                    jobs.append(job)
    return {
        "output_dir": str(output),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "seeds": seeds,
        "jobs": jobs,
        "hostname": socket.gethostname(),
        "output_policy": "Timing only for standard methods; final reward and safety only for lookup PSPO",
        "timing_notes": {
            "training_loop_s": "learn(), including periodic evaluation and nested LID work",
            "rl_stage_s": "setup, training, evaluations and in-memory output finalisation",
            "process_wall_time_s": "full worker process, measured by supervisor; excludes scheduling wait",
            "policy_initialisation_s": "reward-free shared BC, once per environment/representation",
            "lid_s": "nested LID construction; already included in training/stage/process, do not add again",
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def supervise(path: Path) -> int:
    manifest = read_json(path)
    root = Path(manifest["output_dir"])
    pending = list(manifest["jobs"])
    running = {}
    completed, failed = set(), set()
    cpu_pool = list(manifest["cpu_ids"])
    state_path = root / "status.json"
    write_json(
        root / "supervisor.json", {"pid": os.getpid(), "hostname": socket.gethostname()}
    )
    while pending or running:
        for job_id, (process, job, cpu, elapsed) in list(running.items()):
            if "seconds" not in elapsed:
                continue
            returncode = process.returncode
            result_path = root / job["relative_dir"] / "training_time.json"
            result = (
                read_json(result_path)
                if result_path.exists()
                else {
                    "status": "failed",
                    "error": "Worker exited without a timing result",
                }
            )
            result.update(
                process_wall_time_s=elapsed["seconds"],
                worker_returncode=returncode,
            )
            if job["kind"] == "training" and result["status"] == "complete":
                init_process_s = 0.0
                if job["method"].startswith("pspo"):
                    initialisation = read_json(
                        Path(job["base_policy_path"]).parent / "training_time.json"
                    )
                    init_process_s = initialisation["process_wall_time_s"]
                    result["initialisation_process_s"] = init_process_s
                result["cold_single_seed_process_s"] = (
                    init_process_s + elapsed["seconds"]
                )
            write_json(result_path, result)
            (
                completed
                if returncode == 0 and result["status"] == "complete"
                else failed
            ).add(job_id)
            cpu_pool.append(cpu)
            del running[job_id]
            print(
                f"{result['status']} {job_id} {result['process_wall_time_s']:.2f}s",
                flush=True,
            )
        for job in list(pending):
            if job.get("dependency") in failed:
                write_json(
                    root / job["relative_dir"] / "training_time.json",
                    {
                        "status": "blocked",
                        "job_id": job["id"],
                        "dependency": job["dependency"],
                    },
                )
                failed.add(job["id"])
                pending.remove(job)
                continue
            if len(running) >= manifest["max_parallel"] or not cpu_pool:
                break
            if job.get("dependency") and job["dependency"] not in completed:
                continue
            cpu = cpu_pool.pop(0)
            started = time.perf_counter()
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--manifest",
                    str(path),
                    "--job-id",
                    job["id"],
                    "--cpu",
                    str(cpu),
                ],
                cwd=REPO,
                env={**os.environ, **THREAD_ENV},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            elapsed = {}

            def await_exit(child, launched_at, result):
                child.wait()
                result["seconds"] = time.perf_counter() - launched_at

            threading.Thread(
                target=await_exit, args=(process, started, elapsed), daemon=True
            ).start()
            running[job["id"]] = (process, job, cpu, elapsed)
            pending.remove(job)
        write_json(
            state_path,
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total": len(manifest["jobs"]),
                "complete": len(completed),
                "failed_or_blocked": len(failed),
                "running": len(running),
                "pending": len(pending),
                "running_jobs": {
                    key: {"pid": entry[0].pid, "cpu": entry[2]}
                    for key, entry in running.items()
                },
                "complete_jobs": sorted(completed),
                "failed_jobs": sorted(failed),
            },
        )
        if running:
            time.sleep(2)
    return int(bool(failed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--environments", nargs="+")
    parser.add_argument("--max-parallel", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.worker:
        manifest = read_json(args.manifest)
        job = next(job for job in manifest["jobs"] if job["id"] == args.job_id)
        return worker(manifest, job, args.cpu)
    if args.supervise:
        return supervise(args.manifest)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(
            "Use a new output directory; this launcher does not overwrite or resume experiments"
        )
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed not in range(10) for seed in seeds)
    ):
        parser.error("Seeds must be unique members of 0..9")
    manifest = build_manifest(output, seeds, args.environments, args.smoke)
    cpu_ids = sorted(os.sched_getaffinity(0))
    manifest.update(
        max_parallel=min(args.max_parallel, max(1, len(cpu_ids) - 8)),
        cpu_ids=cpu_ids[-min(args.max_parallel, max(1, len(cpu_ids) - 8)) :],
    )
    path = output / "manifest.json"
    write_json(path, manifest)
    print(
        f"Prepared {len(manifest['jobs'])} jobs, up to {manifest['max_parallel']} concurrent single-core workers."
    )
    if args.launch:
        with (output / "supervisor.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--supervise",
                    "--manifest",
                    str(path),
                ],
                cwd=REPO,
                env={**os.environ, **THREAD_ENV},
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"Launched supervisor PID {process.pid}; manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
