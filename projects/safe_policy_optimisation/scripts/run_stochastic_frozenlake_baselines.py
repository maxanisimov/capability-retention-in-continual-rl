#!/usr/bin/env python3
"""Run safe-RL baselines on structured slippery FrozenLake.

This is the baseline counterpart to ``run_stochastic_frozenlake_pspo.py``. It
deliberately does not prepare its own inputs: it reuses the layout, shield and
safe base policy that the PSPO run already produced, verifying them against the
hashes recorded in that run's ``experiment.json``. Any comparison against PSPO
is therefore on an identical environment and an identical shield.

Every optimisation flag that both stages share is copied from the PSPO run's
recorded settings, so the baselines differ from PSPO only in the training rule.

The detached controller pins concurrent workers to distinct idle physical cores
and re-polls for idle cores between waves, so the sweep grows into capacity as
other jobs on a shared machine finish.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-matplotlib")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

import torch  # noqa: E402

from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
)

DEFAULT_SIZE = 128
RUNS_ROOT = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
SEEDS = tuple(range(10))
DEFAULT_METHODS = ("ppo_shield", "cpo")
METHODS = DEFAULT_METHODS + ("ppo_lagrangian", "ppo_pid_lagrangian", "ppo")


# The inputs and the recorded settings both come from this run, so the baseline
# numbers land on exactly the layout, shield and budget PSPO was measured on.
def pspo_root(size: int) -> Path:
    return RUNS_ROOT / f"pspo_stochastic_frozenlake{size}"


def default_output(size: int) -> Path:
    return RUNS_ROOT / f"baselines_stochastic_frozenlake{size}"


def default_screen_name(size: int) -> str:
    return f"baselines-frozenlake{size}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(source: Path) -> dict:
    """Fail loudly if the PSPO inputs this sweep borrows have changed."""

    record = read_json(source / "experiment.json")
    for name, digest in record["input_sha256"].items():
        if sha256(source / "_inputs" / name) != digest:
            raise ValueError(f"Changed PSPO input: {name}")
    return record


def env_kwargs(source: Path, settings: dict) -> str:
    return json.dumps(
        {
            "layout_path": str(source / "_inputs" / "layout.txt"),
            "is_slippery": True,
            "success_rate": settings["success_rate"],
            "step_penalty": settings["step_penalty"],
        }
    )


def shared_arguments(
    source: Path, settings: dict, seed: int, *, smoke: bool
) -> list[str]:
    """Flags common to both baselines, matched to the PSPO run's settings."""

    return [
        "--env-id",
        ENV_ID,
        "--env-kwargs",
        env_kwargs(source, settings),
        "--shield-path",
        str(source / "_inputs" / "shield_q.pt"),
        "--max-episode-steps",
        str(settings["max_episode_steps"]),
        "--cost-limit",
        "0",
        "--total-timesteps",
        str(2048 if smoke else settings["total_timesteps"]),
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--learning-rate",
        str(settings["learning_rate"]),
        "--n-steps",
        str(settings["n_steps"]),
        "--batch-size",
        str(settings["batch_size"]),
        "--n-epochs",
        "1" if smoke else str(settings["n_epochs"]),
        "--gamma",
        str(settings["gamma"]),
        "--gae-lambda",
        "0.95",
        "--clip-range",
        "0.2",
        "--ent-coef",
        "0",
        "--vf-coef",
        "0.5",
        "--max-grad-norm",
        "0.5",
        "--n-hidden",
        "2",
        "--hidden-dim",
        "64",
        "--eval-episodes",
        "2" if smoke else str(settings["evaluation_episodes"]),
        "--early-stop-eval-freq",
        "0",
        "--curve-eval-freq",
        "0" if smoke else "100000",
        "--curve-eval-episodes",
        "2" if smoke else "10",
    ]


def training_arguments(
    method: str, source: Path, settings: dict, seed: int, run_dir: Path, *, smoke: bool
) -> list[str]:
    arguments = shared_arguments(source, settings, seed, smoke=smoke)
    arguments += ["--output-dir", str(run_dir.parent), "--run-id", run_dir.name]
    if method == "ppo":
        # Plain PPO never masks an action or warm-starts from the safe actor.
        # --shield-path is audit-only in train_ppo.
        if smoke:
            for flag, value in (
                ("--total-timesteps", "64"),
                ("--n-steps", "64"),
                ("--max-episode-steps", "128"),
            ):
                arguments[arguments.index(flag) + 1] = value
        return arguments
    if method == "ppo_shield":
        # PSPO is scored by its nominal actor, so score the shielded baseline the
        # same way; the stage reports the shielded evaluation alongside it.
        return arguments + ["--evaluation-policy", "unshielded"]
    if method in {"ppo_lagrangian", "ppo_pid_lagrangian"}:
        return arguments + ["--jobs", "1", "--algorithms", method]
    # The shield is audit-only for CPO: it never gates an action, it only counts
    # how many of CPO's proposals would have been unsafe.
    return arguments + ["--jobs", "1"]


def run_stage(method: str, arguments: list[str]) -> dict:
    if method == "ppo_shield":
        from projects.safe_policy_optimisation.stages import train_ppo_shield as stage
    elif method == "ppo":
        from projects.safe_policy_optimisation.stages import train_ppo as stage
    elif method == "cpo":
        from projects.safe_policy_optimisation.stages import train_cpo as stage
    else:
        from projects.safe_policy_optimisation.stages import (
            train_ppo_lagrangian as stage,
        )
    return stage.run(stage.build_parser().parse_args(arguments))


def goal_success_rate(rows: list[dict], step_penalty: float) -> float:
    """Recover the goal indicator from return and length.

    The shared stages call an episode successful when its return is positive,
    but a slow goal-reaching episode pays enough step penalty to finish
    negative. Return = goal_indicator - step_penalty * length, so adding the
    penalty back recovers the indicator exactly.
    """

    if not rows:
        return float("nan")
    reached = [
        float(row["reward"]) + step_penalty * int(row["length"]) > 0.5 for row in rows
    ]
    return statistics.fmean(float(value) for value in reached)


def episode_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def normalise_plain_ppo_goal_metrics(run_dir: Path, step_penalty: float) -> None:
    """Preserve stage metrics, then report goal attainment rather than positive return."""
    rows = episode_rows(run_dir / "episodes.csv")
    if not rows:
        raise ValueError(f"Missing PPO evaluation episodes: {run_dir}")
    metrics = read_json(run_dir / "metrics.json")
    original = run_dir / "metrics_reward_threshold.json"
    if not original.exists():
        write_json(original, metrics)
    rate = goal_success_rate(rows, step_penalty)
    metrics["success"] = {
        "success_mode": "goal_reached",
        "definition": "goal indicator recovered exactly from return plus step penalty times length",
        "success_count": int(round(rate * len(rows))),
        "success_rate": rate,
    }
    write_json(run_dir / "metrics.json", metrics)


def seed_row(method: str, run_dir: Path, step_penalty: float) -> dict | None:
    """Summarise one finished run, preferring the unshielded/nominal policy."""

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    summary = read_json(run_dir / "summary.json")
    if method == "ppo_shield":
        block = metrics["nominal"]
        rows = episode_rows(run_dir / "episodes_nominal.csv")
        shielded = metrics["shielded"]
        shielded_rows = episode_rows(run_dir / "episodes_shielded.csv")
        training = {**summary, **summary.get("training", {})}
    elif method == "ppo":
        block = metrics
        rows = episode_rows(run_dir / "episodes.csv")
        shielded = None
        shielded_rows = []
        training = {**summary, **summary.get("training", {})}
    else:
        block = metrics[method]
        rows = episode_rows(run_dir / "episodes.csv")
        shielded = None
        shielded_rows = []
        training = summary[method]
    # How often the learner *proposed* an unsafe action while exploring. For
    # PPO-Shield the shield then blocks it, so this is the honest measure of how
    # unsafe the learned policy itself is; for CPO nothing blocks it.
    explored = float(training.get("total_exploration_steps") or 0.0)
    unsafe_proposals = float(
        training.get("unsafe_proposed_actions_during_exploration") or 0.0
    )
    row = {
        "method": method,
        "mean_total_reward": block["reward"]["mean_total_reward"],
        "safety_rate": block["safety"]["safety_rate"],
        "success_rate": goal_success_rate(rows, step_penalty),
        "unsafe_proposed_action_percentage": (
            100.0 * unsafe_proposals / explored if explored else ""
        ),
        "training_safety_rate": training.get("training_safety_rate", ""),
    }
    if shielded is not None:
        row["shielded_mean_total_reward"] = shielded["reward"]["mean_total_reward"]
        row["shielded_safety_rate"] = shielded["safety"]["safety_rate"]
        row["shielded_success_rate"] = goal_success_rate(shielded_rows, step_penalty)
    return row


REPORT_FIELDS = [
    "method",
    "seed",
    "mean_total_reward",
    "safety_rate",
    "success_rate",
    "unsafe_proposed_action_percentage",
    "training_safety_rate",
    "shielded_mean_total_reward",
    "shielded_safety_rate",
    "shielded_success_rate",
]
AGGREGATE_KEYS = (
    "mean_total_reward",
    "safety_rate",
    "success_rate",
    "shielded_mean_total_reward",
    "shielded_safety_rate",
    "shielded_success_rate",
)


def write_report(root: Path, step_penalty: float, methods: tuple[str, ...]) -> None:
    rows = []
    for method in methods:
        for seed in SEEDS:
            run_dir = root / method / f"seed{seed}"
            status = run_dir / "status.json"
            if not status.exists() or read_json(status).get("status") != "complete":
                continue
            row = seed_row(method, run_dir, step_penalty)
            if row is None:
                continue
            row["seed"] = seed
            rows.append(row)
    with (root / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict = {
        "updated_utc": utc_now(),
        "requested_seeds": len(SEEDS),
        "methods": {},
    }
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        block: dict = {"completed_seeds": len(method_rows)}
        for key in AGGREGATE_KEYS:
            values = [
                float(row[key]) for row in method_rows if row.get(key) not in (None, "")
            ]
            if not values:
                continue
            block[key] = statistics.fmean(values)
            block[key + "_2se"] = (
                2 * statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else None
            )
        summary["methods"][method] = block
    write_json(root / "aggregate.json", summary)


def free_physical_cpus(min_idle: float, *, seconds: int = 5) -> tuple[list[int], dict]:
    """One CPU id per physical core whose every sibling is idle enough.

    Both siblings must be idle: a hyperthread sharing a core with a saturated
    neighbour delivers a fraction of a core, which would silently stretch the
    sweep rather than fail.
    """

    sample = json.loads(
        subprocess.check_output(
            ["mpstat", "-o", "JSON", "-P", "ALL", "1", str(seconds)],
            text=True,
        )
    )
    observations: dict[int, list[float]] = {}
    for frame in sample["sysstat"]["hosts"][0]["statistics"]:
        for cpu in frame["cpu-load"]:
            if cpu["cpu"] != "all":
                observations.setdefault(int(cpu["cpu"]), []).append(float(cpu["idle"]))
    idle = {cpu: statistics.fmean(values) for cpu, values in observations.items()}
    topology = subprocess.check_output(
        ["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True
    )
    physical: dict[tuple[int, int], list[int]] = {}
    allowed = os.sched_getaffinity(0)
    for line in topology.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, online = line.split(",")
        if online == "Y":
            physical.setdefault((int(socket), int(core)), []).append(int(cpu))
    admitted = []
    for siblings in physical.values():
        if any(cpu in (0, 1) or idle.get(cpu, 0) < min_idle for cpu in siblings):
            continue
        choices = sorted(set(siblings) & allowed)
        if choices:
            admitted.append(choices[0])
    return sorted(admitted), {
        "mean_idle_percent": idle,
        "physical_core_siblings": list(physical.values()),
        "min_idle_percent_required": min_idle,
    }


def pending_jobs(root: Path, methods: tuple[str, ...]) -> list[tuple[str, int]]:
    jobs = []
    for method in methods:
        for seed in SEEDS:
            status = root / method / f"seed{seed}" / "status.json"
            if not status.exists() or read_json(status).get("status") != "complete":
                jobs.append((method, seed))
    return jobs


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    root = args.output_root
    source = args.pspo_root
    settings = verify_inputs(source)["settings"]
    run_dir = (
        (root / "_smoke" if args.smoke else root) / args.method / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "complete":
        print(f"Already complete: {run_dir}", flush=True)
        return
    status = {
        "status": "running",
        "method": args.method,
        "seed": args.seed,
        "smoke": args.smoke,
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "started_utc": utc_now(),
    }
    write_json(status_path, status)
    started = time.monotonic()
    try:
        arguments = training_arguments(
            args.method, source, settings, args.seed, run_dir, smoke=args.smoke
        )
        run_stage(args.method, arguments)
        if args.method == "ppo":
            normalise_plain_ppo_goal_metrics(run_dir, float(settings["step_penalty"]))
        rows = seed_row(args.method, run_dir, float(settings["step_penalty"])) or {}
        status.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "wall_seconds": time.monotonic() - started,
                "result": rows,
            }
        )
    except Exception as error:  # noqa: BLE001 - recorded for the controller
        status.update(
            {
                "status": "failed",
                "finished_utc": utc_now(),
                "wall_seconds": time.monotonic() - started,
                "error": repr(error),
            }
        )
        write_json(status_path, status)
        raise
    write_json(status_path, status)


def controller(args: argparse.Namespace) -> None:
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = args.pspo_root
    record = verify_inputs(source)
    settings = record["settings"]
    methods = tuple(args.methods)
    write_json(
        root / "experiment.json",
        {
            "created_utc": utc_now(),
            "env_id": ENV_ID,
            "methods": list(methods),
            "seeds": list(SEEDS),
            "settings": settings,
            "pspo_reference_run": str(source),
            "input_sha256": record["input_sha256"],
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)
            ),
        },
    )
    (root / "_logs").mkdir(exist_ok=True)
    pending = pending_jobs(root, methods)
    cpus, cpu_evidence = free_physical_cpus(args.min_idle)
    # Starting with nothing free is not an error here: the rescan loop below
    # claims cores as other jobs release them, so the sweep queues rather than
    # either failing or muscling in on someone else's core.
    if not cpus and pending:
        print(
            f"[{utc_now()}] no idle physical cores yet; waiting for capacity",
            flush=True,
        )
    launch = {
        "started_utc": utc_now(),
        "screen_name": args.screen_name,
        "max_concurrent": args.max_concurrent,
        "queued_jobs": [list(job) for job in pending],
        "initial_admitted_cpus": list(cpus),
        "initial_cpu_evidence": cpu_evidence,
        "jobs": [],
    }
    write_json(root / "launch_manifest.json", launch)
    write_report(root, float(settings["step_penalty"]), methods)
    available = cpus[: args.max_concurrent]
    active: dict[tuple[str, int], tuple] = {}
    failures = []
    last_poll = time.monotonic()
    while pending or active:
        while pending and available and len(active) < args.max_concurrent:
            method, seed = pending.pop(0)
            cpu = available.pop(0)
            log_path = root / "_logs" / f"{method}_seed{seed}.log"
            command = [
                "taskset",
                "-c",
                str(cpu),
                str(REPO / ".venv/bin/python"),
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--method",
                method,
                "--seed",
                str(seed),
                "--output-root",
                str(root),
                "--pspo-root",
                str(source),
            ]
            log = log_path.open("a")
            process = subprocess.Popen(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT
            )
            active[(method, seed)] = (process, cpu, log)
            launch["jobs"].append(
                {
                    "method": method,
                    "seed": seed,
                    "cpu": cpu,
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                    "started_utc": utc_now(),
                }
            )
            write_json(root / "launch_manifest.json", launch)
            print(
                f"[{utc_now()}] launched {method}/seed{seed} pid={process.pid} cpu={cpu}",
                flush=True,
            )
        for key, (process, cpu, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[key]
            available.append(cpu)
            if code != 0:
                failures.append({"job": list(key), "returncode": code})
            print(f"[{utc_now()}] finished {key[0]}/seed{key[1]} rc={code}", flush=True)
            write_report(root, float(settings["step_penalty"]), methods)
        # Other users' jobs come and go on a shared machine. Re-polling lets the
        # sweep claim cores that freed up instead of staying at the width it
        # happened to start with.
        if pending and time.monotonic() - last_poll > args.poll_seconds:
            last_poll = time.monotonic()
            fresh, evidence = free_physical_cpus(args.min_idle)
            claimed = {cpu for _process, cpu, _log in active.values()} | set(available)
            gained = [cpu for cpu in fresh if cpu not in claimed]
            if gained:
                available.extend(gained)
                launch["cpu_rescans"] = launch.get("cpu_rescans", []) + [
                    {
                        "utc": utc_now(),
                        "gained_cpus": gained,
                        "min_idle_percent_required": evidence[
                            "min_idle_percent_required"
                        ],
                    }
                ]
                write_json(root / "launch_manifest.json", launch)
                print(f"[{utc_now()}] rescan admitted cpus {gained}", flush=True)
        time.sleep(5)
    launch["finished_utc"] = utc_now()
    launch["failures"] = failures
    write_json(root / "launch_manifest.json", launch)
    write_report(root, float(settings["step_penalty"]), methods)
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--pspo-root", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
        help="Baseline methods to run. Defaults to PPO-Shield and CPO.",
    )
    parser.add_argument(
        "--screen-name",
        default=None,
        help="Detached controller screen name; defaults to baselines-frozenlake<SIZE>.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Upper bound on concurrent workers, each pinned to its own physical core.",
    )
    parser.add_argument(
        "--min-idle",
        type=float,
        default=95.0,
        help="Minimum mean idle percent required of both siblings of a physical core.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=600.0,
        help="How often the controller rescans for newly idle cores while work is queued.",
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_idle <= 100:
        parser.error("--min-idle must be in (0, 100]")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods may not contain duplicates")
    if args.max_concurrent is None:
        args.max_concurrent = len(SEEDS) * len(args.methods)
    if not 1 <= args.max_concurrent <= len(SEEDS) * len(args.methods):
        parser.error(f"--max-concurrent must be in 1..{len(SEEDS) * len(args.methods)}")
    args.output_root = args.output_root or default_output(args.size)
    args.pspo_root = args.pspo_root or pspo_root(args.size)
    args.screen_name = args.screen_name or default_screen_name(args.size)
    if args.worker and (args.method is None or args.seed is None):
        parser.error("--worker requires --method and --seed")

    if args.report_only:
        settings = read_json(args.pspo_root / "experiment.json")["settings"]
        write_report(
            args.output_root, float(settings["step_penalty"]), tuple(args.methods)
        )
    elif args.worker:
        worker(args)
    elif args.controller:
        controller(args)
    elif args.launch:
        args.output_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "screen",
                "-L",
                "-Logfile",
                str(args.output_root / "controller.log"),
                "-dmS",
                args.screen_name,
                str(REPO / ".venv/bin/python"),
                "-u",
                str(Path(__file__).resolve()),
                "--controller",
                "--size",
                str(args.size),
                "--methods",
                *args.methods,
                "--screen-name",
                args.screen_name,
                "--max-concurrent",
                str(args.max_concurrent),
                "--min-idle",
                str(args.min_idle),
                "--poll-seconds",
                str(args.poll_seconds),
                "--output-root",
                str(args.output_root),
                "--pspo-root",
                str(args.pspo_root),
            ],
            cwd=REPO,
            check=True,
        )
        print(f"Detached controller in screen session {args.screen_name}")
    else:
        parser.error("Choose --launch, --controller, --worker, or --report-only")


if __name__ == "__main__":
    main()
