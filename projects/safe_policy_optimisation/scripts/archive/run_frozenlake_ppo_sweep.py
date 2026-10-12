#!/usr/bin/env python3
"""Launch plain PPO on FrozenLake 128/256: ten seeds each, one screen supervisor."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_stochastic_frozenlake_baselines as baseline,
)

RUNS = baseline.RUNS_ROOT
SIZES = (128, 256)
SOURCE_FILES = (
    Path(__file__).relative_to(REPO),
    Path(baseline.__file__).relative_to(REPO),
    Path("projects/safe_policy_optimisation/stages/train_ppo.py"),
    Path("projects/safe_policy_optimisation/utils/frozen_lake_experiment.py"),
)
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


def reference_runs() -> dict[int, Path]:
    return {
        size: RUNS / f"pspo_stochastic_frozenlake{size}_safety_only_state_id_lookup"
        for size in SIZES
    }


def build_manifest(
    output: Path,
    cpus: list[int],
    *,
    smoke: bool = False,
    sources: dict[int, Path] | None = None,
) -> dict:
    seeds = (0,) if smoke else baseline.SEEDS
    required = len(SIZES) * len(seeds)
    if len(cpus) < required or len(set(cpus[:required])) != required:
        raise ValueError(f"Need {required} distinct CPUs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite or resume {output}")
    sources = sources or reference_runs()
    records = {size: baseline.verify_inputs(sources[size]) for size in SIZES}
    for size, record in records.items():
        if record["settings"]["size"] != size:
            raise ValueError(f"Reference size mismatch: {size}")
    jobs = []
    inputs = {}
    for size in SIZES:
        record = records[size]
        source = sources[size].resolve()
        root = output / f"frozenlake{size}"
        settings = record["settings"]
        inputs[str(size)] = {
            "reference_run": str(source),
            "settings": settings,
            "input_sha256": record["input_sha256"],
        }
        for seed in seeds:
            cpu = cpus[len(jobs)]
            directory = (root / "_smoke" if smoke else root) / "ppo" / f"seed{seed}"
            argv = baseline.training_arguments(
                "ppo", source, settings, seed, directory, smoke=smoke
            )
            command = [
                "taskset",
                "-c",
                str(cpu),
                str(REPO / ".venv/bin/python"),
                "-u",
                str(Path(baseline.__file__).resolve()),
                "--worker",
                "--method",
                "ppo",
                "--size",
                str(size),
                "--seed",
                str(seed),
                "--output-root",
                str(root),
                "--pspo-root",
                str(source),
            ]
            if smoke:
                command.append("--smoke")
            jobs.append(
                {
                    "id": f"frozenlake{size}/ppo/seed{seed}",
                    "size": size,
                    "seed": seed,
                    "cpu": cpu,
                    "directory": str(directory),
                    "command": command,
                    "stage_arguments": argv,
                }
            )
    manifest = {
        "created_utc": baseline.utc_now(),
        "hostname": socket.gethostname(),
        "output_dir": str(output),
        "smoke": smoke,
        "method": "ppo",
        "initialisation": "standard SB3 random actor and critic; no warm start",
        "state_representation": "native discrete observations with SB3 one-hot preprocessing",
        "shield_usage": "audit only; no masking during training or evaluation",
        "inputs": inputs,
        "jobs": jobs,
        "source_sha256": {
            str(path): baseline.sha256(REPO / path) for path in SOURCE_FILES
        },
    }
    output.mkdir(parents=True)
    logs = output / "_orchestrator"
    logs.mkdir()
    for size in SIZES:
        root = output / f"frozenlake{size}"
        root.mkdir()
        baseline.write_json(
            root / "experiment.json",
            {
                "method": "ppo",
                "seeds": list(seeds),
                "smoke": smoke,
                "reference_run": inputs[str(size)]["reference_run"],
                "reference_settings": inputs[str(size)]["settings"],
                "input_sha256": inputs[str(size)]["input_sha256"],
                "initialisation": manifest["initialisation"],
                "state_representation": manifest["state_representation"],
                "shield_usage": manifest["shield_usage"],
            },
        )
    with zipfile.ZipFile(
        output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in SOURCE_FILES:
            archive.write(REPO / path, arcname=str(path))
    return manifest


def run_job(job: dict, logs: Path) -> dict:
    directory = Path(job["directory"])
    directory.mkdir(parents=True)
    log_path = logs / f"frozenlake{job['size']}_seed{job['seed']}.log"
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
        baseline.write_json(
            directory / "process.json",
            {
                "pid": process.pid,
                "cpu": job["cpu"],
                "started_utc": baseline.utc_now(),
            },
        )
        returncode = process.wait()
    status_path = directory / "status.json"
    complete = (
        returncode == 0
        and status_path.is_file()
        and baseline.read_json(status_path).get("status") == "complete"
    )
    result = {
        "job_id": job["id"],
        "size": job["size"],
        "seed": job["seed"],
        "cpu": job["cpu"],
        "returncode": returncode,
        "status": "complete" if complete else "failed",
        "rl_stage_wall_time_s": time.perf_counter() - started,
        "finished_utc": baseline.utc_now(),
        "log_path": str(log_path),
    }
    baseline.write_json(directory / "runtime.json", result)
    return result


def supervise(path: Path) -> int:
    manifest = baseline.read_json(path)
    output = Path(manifest["output_dir"])
    logs = output / "_orchestrator"
    lock = (logs / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in manifest["source_sha256"].items():
        if baseline.sha256(REPO / name) != digest:
            raise ValueError(f"Source changed after preparation: {name}")
    for source in manifest["inputs"].values():
        baseline.verify_inputs(Path(source["reference_run"]))
    baseline.write_json(
        logs / "supervisor.json",
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_utc": baseline.utc_now(),
        },
    )
    completed, failed = [], []

    def report() -> None:
        baseline.write_json(
            output / "status.json",
            {
                "updated_utc": baseline.utc_now(),
                "total": len(manifest["jobs"]),
                "complete": len(completed),
                "failed": len(failed),
                "running_or_starting": len(manifest["jobs"])
                - len(completed)
                - len(failed),
                "complete_jobs": completed,
                "failed_jobs": failed,
            },
        )
        if not manifest["smoke"]:
            for size in SIZES:
                baseline.write_report(
                    output / f"frozenlake{size}",
                    manifest["inputs"][str(size)]["settings"]["step_penalty"],
                    ("ppo",),
                )

    report()
    print(
        f"Launching {len(manifest['jobs'])} plain PPO jobs on distinct physical cores",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(manifest["jobs"])) as executor:
        futures = [executor.submit(run_job, job, logs) for job in manifest["jobs"]]
        for future in as_completed(futures):
            result = future.result()
            (completed if result["status"] == "complete" else failed).append(
                result["job_id"]
            )
            print(
                f"{result['status']} {result['job_id']} cpu={result['cpu']} "
                f"{result['rl_stage_wall_time_s']:.1f}s",
                flush=True,
            )
            report()
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
    cpus, evidence = baseline.free_physical_cpus(95.0)
    cpus = [cpu for cpu in cpus if cpu >= 8]
    output = RUNS / args.run_name
    manifest = build_manifest(output, cpus, smoke=args.smoke)
    manifest["cpu_availability_evidence"] = evidence
    path = output / "_orchestrator/launch_manifest.json"
    baseline.write_json(path, manifest)
    print(f"Prepared {len(manifest['jobs'])} PPO jobs: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
