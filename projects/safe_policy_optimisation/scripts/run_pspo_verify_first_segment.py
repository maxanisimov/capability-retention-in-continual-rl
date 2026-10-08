#!/usr/bin/env python3
"""Prepare and supervise a matched verify-first + certified-segment PSPO cohort."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core"), str(Path(__file__).resolve().parent)]

from run_timed_main_comparison import (  # noqa: E402
    THREAD_ENV,
    cli_arguments,
    read_json,
    training_values,
    write_json,
)

RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
ENVIRONMENTS = (
    "media_streaming",
    "colour_bomb",
    "colour_bomb_v2",
    "bridge_crossing",
    "bridge_crossing_v2",
    "mini_pacman",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stage_arguments(config: dict, directory: Path, cpu: int, smoke: bool) -> list[str]:
    from projects.safe_policy_optimisation.stages import train_pspo

    job = {
        "method": "pspo",
        "config": config,
        "budget": config["total_timesteps"],
        "seed": config["seed"],
        "base_policy_path": config["base_policy_path"],
        "representation": "one_hot",
    }
    values = training_values(job, directory, cpu, smoke)
    values.update(verify_first=True, safe_region_shape="segment")
    # Leave normal learning curves/checkpoints/metrics enabled; this is not timing-only.
    values["tensorboard_log_dir"] = directory / "tensorboard"
    parser = train_pspo.build_parser()
    argv = cli_arguments(parser, values)
    args = train_pspo.parse_args(argv)
    assert args.verify_first and args.safe_region_shape == "segment"
    assert args.base_policy_path.resolve() == Path(config["base_policy_path"]).resolve()
    assert args.total_timesteps == (8 if smoke else config["total_timesteps"])
    assert args.eval_episodes == (2 if smoke else config["eval_episodes"])
    return argv


def build_manifest(output: Path, cpus: list[int], *, smoke: bool = False) -> dict:
    seeds = [0] if smoke else list(range(10))
    if len(cpus) != len(ENVIRONMENTS) * len(seeds) or len(set(cpus)) != len(cpus):
        raise ValueError("Need one distinct CPU per environment/seed")
    jobs = []
    bases = {}
    for environment in ENVIRONMENTS:
        for seed in seeds:
            source = (
                RUNS
                / "segment_lid/two_hidden"
                / environment
                / f"seed{seed}/config.json"
            )
            config = read_json(source)
            adaptive = config["adaptive"]
            if adaptive["verify_first"] or adaptive["safe_region_shape"] != "segment":
                raise ValueError(f"Not a segment-only reference: {source}")
            if (
                config["evaluation_policy"] != "unshielded"
                or config["eval_episodes"] != 100
            ):
                raise ValueError(f"Unexpected evaluation settings: {source}")
            base = Path(config["base_policy_path"])
            base_hash = sha256(base)
            if environment in bases and bases[environment] != base_hash:
                raise ValueError(
                    f"Reference seeds use different initial policies: {environment}"
                )
            bases[environment] = base_hash
            cpu = cpus[len(jobs)]
            directory = output / "two_hidden" / environment / f"seed{seed}"
            argv = stage_arguments(config, directory, cpu, smoke)
            jobs.append(
                {
                    "id": f"{environment}/seed{seed}",
                    "environment": environment,
                    "seed": seed,
                    "cpu": cpu,
                    "directory": str(directory),
                    "source_config": str(source),
                    "source_config_sha256": sha256(source),
                    "base_policy_path": str(base),
                    "base_policy_sha256": base_hash,
                    "shield_path": config["shield_path"],
                    "shield_sha256": sha256(Path(config["shield_path"])),
                    "nominal_timesteps": 8 if smoke else config["total_timesteps"],
                    "command": [
                        "taskset",
                        "-c",
                        str(cpu),
                        sys.executable,
                        str(
                            REPO
                            / "projects/safe_policy_optimisation/stages/train_pspo.py"
                        ),
                        *argv,
                    ],
                }
            )
    sources = [
        Path(__file__),
        REPO / "projects/safe_policy_optimisation/stages/train_pspo.py",
        REPO / "core/provably_safe_policy_optimisation/adaptive_safe_ppo.py",
        REPO / "core/src/segment_rashomon.py",
    ]
    return {
        "created_at_utc": utc_now(),
        "output_dir": str(output),
        "smoke": smoke,
        "hostname": socket.gethostname(),
        "jobs": jobs,
        "settings": {
            "verify_first": True,
            "safe_region_shape": "segment",
            "architecture": "two_hidden",
            "state_representation": "one_hot",
            "seeds": seeds,
            "environments": list(ENVIRONMENTS),
            "parallel_jobs": len(jobs),
            "threads_per_worker": 1,
            "initialisation": "Reuse exact per-environment segment-only initial policy",
            "comparison": "Copy each segment-only seed configuration; enable verify-first",
            "runtime_boundary": "Entire RL worker, including setup/evaluations/output; shared initialisation excluded",
        },
        "source_sha256": {str(path): sha256(path) for path in sources},
    }


def run_job(job: dict, logs: Path) -> dict:
    log_path = logs / f"{job['environment']}_seed{job['seed']}.log"
    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            job["command"],
            cwd=REPO,
            env={
                **os.environ,
                **THREAD_ENV,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": f"{REPO / 'core'}:{REPO}:{os.environ.get('PYTHONPATH', '')}",
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        write_json(
            Path(job["directory"]) / "process.json",
            {
                "pid": process.pid,
                "cpu": job["cpu"],
                "job_id": job["id"],
                "started_at_utc": utc_now(),
            },
        )
        code = process.wait()
    directory = Path(job["directory"])
    complete = code == 0 and (directory / "metrics.json").is_file()
    result = {
        "job_id": job["id"],
        "environment": job["environment"],
        "seed": job["seed"],
        "cpu": job["cpu"],
        "worker_returncode": code,
        "status": "complete" if complete else "failed",
        "rl_stage_wall_time_s": time.perf_counter() - started,
        "finished_at_utc": utc_now(),
        "log_path": str(log_path),
    }
    write_json(directory / "runtime.json", result)
    return result


def supervise(path: Path) -> int:
    manifest = read_json(path)
    output = Path(manifest["output_dir"])
    logs = output / "_orchestrator"
    complete, failed = [], []
    write_json(
        logs / "supervisor.json", {"pid": os.getpid(), "hostname": socket.gethostname()}
    )

    def status():
        write_json(
            output / "status.json",
            {
                "updated_at_utc": utc_now(),
                "total": len(manifest["jobs"]),
                "complete": len(complete),
                "failed": len(failed),
                "running_or_starting": len(manifest["jobs"])
                - len(complete)
                - len(failed),
                "complete_jobs": complete,
                "failed_jobs": failed,
            },
        )

    status()
    print(
        f"Launching {len(manifest['jobs'])} parallel CPU-pinned PSPO verify-first segment jobs",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(manifest["jobs"])) as executor:
        futures = [executor.submit(run_job, job, logs) for job in manifest["jobs"]]
        for future in as_completed(futures):
            result = future.result()
            (complete if result["status"] == "complete" else failed).append(
                result["job_id"]
            )
            line = (
                f"{'ok' if result['status'] == 'complete' else 'FAIL'} "
                f"seed{result['seed']} core={result['cpu']} "
                f"{result['rl_stage_wall_time_s']:.0f}s"
            )
            with (logs / f"{result['environment']}.log").open("a") as log:
                log.write(line + "\n")
            print(result["environment"], line, flush=True)
            status()
    return int(bool(failed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise)
    if not args.run_name or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must be a simple new directory name")
    output = RUNS / args.run_name
    if output.exists():
        parser.error(
            "Use a new run name; this launcher never overwrites or resumes results"
        )
    from projects.safe_policy_optimisation.scripts.launch_pspo_multi_env import (
        sample_cpu_idle,
        select_idle_cpus,
    )

    idle = sample_cpu_idle(5)
    count = 6 if args.smoke else 60
    cpus = select_idle_cpus(
        idle,
        required=count,
        minimum_idle=90.0,
        allowed_cpus={cpu for cpu in os.sched_getaffinity(0) if cpu >= 8},
    )
    manifest = build_manifest(output, cpus, smoke=args.smoke)
    manifest["cpu_idle_percent_at_prepare"] = {str(cpu): idle[cpu] for cpu in cpus}
    path = output / "_orchestrator/launch_manifest.json"
    write_json(path, manifest)
    print(f"Prepared {len(manifest['jobs'])} jobs: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
