#!/usr/bin/env python3
"""Run separately initialised MountainCar PSPO interval/seed pairs in screen.

Each worker inherits a dedicated physical-core affinity for both initialisation
and RL. Excess jobs wait for idle cores. The controller writes final-only reports
after each completion, and refuses to overwrite any existing pair's artifacts.
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
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-mountaincar-sweep-matplotlib")

REPO = Path(__file__).resolve().parents[3]
PYTHON = REPO / ".venv/bin/python"
STAGES = REPO / "projects/safe_policy_optimisation/stages"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO / "core"), str(REPO), os.environ.get("PYTHONPATH", "")]
)
UPPER_BOUNDS = (-1.15, -1.10, -1.05, -1.00, -0.95, -0.90)
SEEDS = tuple(range(10))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_name(upper: float) -> str:
    return "x_" + f"{abs(upper):.2f}".replace(".", "p")


def shield_config(upper: float) -> dict:
    if upper not in UPPER_BOUNDS:
        raise ValueError(f"Unsupported upper bound: {upper}")
    return {"critical_min_position": -1.2, "critical_max_position": upper}


def pair_specs(smoke: bool = False) -> list[dict]:
    bounds, seeds = ((UPPER_BOUNDS[0], UPPER_BOUNDS[-1]), (0,)) if smoke else (UPPER_BOUNDS, SEEDS)
    return [
        {"id": f"{interval_name(upper)}/seed_{seed}", "upper_bound": upper, "seed": seed}
        for seed in seeds for upper in bounds
    ]


def source_paths() -> list[Path]:
    paths = {Path(__file__).resolve()}
    for directory in (
        "core/continuous_state_shields", "core/provably_safe_policy_optimisation",
        "core/src", "projects/safe_policy_optimisation/stages",
        "projects/safe_policy_optimisation/utils",
        "projects/safe_crl/pipelines/envs/mountaincar",
    ):
        paths.update((REPO / directory).rglob("*.py"))
    return sorted(paths)


def prepare(root: Path, smoke: bool) -> dict:
    import torch
    from continuous_state_shields import MountainCarShield, MountainCarShieldConfig

    from projects.safe_policy_optimisation.stages.train_mountaincar_pspo_initialisation import (
        critical_interval_tensors,
    )
    from projects.safe_policy_optimisation.stages.train_pspo_continuous import (
        validate_mountaincar_certificate,
    )

    # Creating a root is intentional; an existing root must be explicitly resumed.
    root.mkdir(parents=True, exist_ok=False)
    (root / "_logs").mkdir()
    pairs = pair_specs(smoke)
    bounds = sorted({pair["upper_bound"] for pair in pairs})
    inputs = {}
    for upper in bounds:
        directory = root / "_inputs" / interval_name(upper)
        directory.mkdir(parents=True)
        tensors = critical_interval_tensors(
            critical_min_position=-1.2, critical_max_position=upper,
            min_velocity=-0.07, max_velocity=0.0, device=torch.device("cpu"),
        )
        dataset = torch.utils.data.TensorDataset(*tensors)
        validate_mountaincar_certificate(dataset, MountainCarShield(MountainCarShieldConfig(**shield_config(upper))))
        path = directory / "critical_interval_dataset.pt"
        torch.save(dataset, path)
        inputs[str(path.relative_to(root))] = sha256(path)
        write_json(directory / "shield_config.json", shield_config(upper))
        inputs[str((directory / "shield_config.json").relative_to(root))] = sha256(directory / "shield_config.json")
    sources = {str(path.relative_to(REPO)): sha256(path) for path in source_paths()}
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO / name, name)
    record = {
        "created_utc": utc_now(), "smoke": smoke, "pairs": pairs,
        "screen_name": "pspo-mc-interval-" + root.name,
        "input_sha256": inputs, "source_sha256": sources,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "settings": {
            "env_id": "MountainCar-v0", "max_episode_steps": 200,
            "separate_safe_initialisation_per_pair": True,
            "total_timesteps": 1024 if smoke else 400_000,
            "n_steps": 1024, "batch_size": 128, "n_epochs": 1 if smoke else 10,
            "learning_rate": 0.001, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
            "mountaincar_shaped_reward": True, "evaluation_reward": "native unshaped",
            "eval_episodes": 2 if smoke else 100, "evaluation_policy": "unshielded",
            "evaluation_seed_offset": 10_000, "early_stopping": False,
            "hidden_dim": 64, "n_hidden": 2, "bc_samples": 4096, "bc_epochs": 200,
            "init_max_epochs": 2000, "init_target_margin": 2.0, "init_learning_rate": 0.01,
            "growth_method": "IBP", "certification_method": "IBP",
            "verify_first": False, "directional": True, "region_mode": "replace",
            "freq": "1", "safe_region_shape": "orthotope", "surrogate": "logsumexp",
            "rashomon_n_iters": 2 if smoke else 200, "rashomon_objective": "weighted_width",
            "rashomon_multi_label_mode": "all", "rashomon_batch_size": "auto",
            "rashomon_checkpoint": 100,
            "safety_rate": "fraction of episodes without left-wall contact",
            "action_compliance": "fraction of nominal greedy actions admitted by the shield",
        },
    }
    write_json(root / "experiment.json", record)
    write_report(root)
    return record


def verify_provenance(root: Path, record: dict) -> None:
    for name, digest in record["input_sha256"].items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Experiment input changed: {name}")
    for name, digest in record["source_sha256"].items():
        if sha256(REPO / name) != digest:
            raise RuntimeError(f"Experiment source changed: {name}; use a fresh output root")


def build_commands(root: Path, pair: dict, settings: dict) -> tuple[list[str], list[str]]:
    run = root / pair["id"]
    certificate = root / "_inputs" / interval_name(pair["upper_bound"]) / "critical_interval_dataset.pt"
    common = ["--seed", str(pair["seed"]), "--device", "cpu"]
    initialisation = [
        str(PYTHON), "-u", str(STAGES / "train_mountaincar_pspo_initialisation.py"),
        "--output-dir", str(run), "--run-id", "initialisation",
        "--certificate-dataset", str(certificate), "--obs-dim", "2", "--n-actions", "3",
        "--hidden-dim", str(settings["hidden_dim"]), "--n-hidden", str(settings["n_hidden"]),
        "--bc-samples", str(settings["bc_samples"]), "--bc-epochs", str(settings["bc_epochs"]),
        "--max-epochs", str(settings["init_max_epochs"]),
        "--target-margin", str(settings["init_target_margin"]),
        "--learning-rate", str(settings["init_learning_rate"]),
        "--certification-method", "IBP", *common,
    ]
    training = [
        str(PYTHON), "-u", str(STAGES / "train_pspo_continuous.py"),
        "--env-id", "MountainCar-v0", "--continuous-shield", "mountaincar",
        "--continuous-shield-config", json.dumps(shield_config(pair["upper_bound"])),
        "--base-policy-path", str(run / "initialisation/base_policy.pt"),
        "--certificate-dataset", str(certificate), "--max-episode-steps", "200",
        "--mountaincar-shaped-reward", "true", "--verify-first", "false",
        "--directional", "true", "--region-mode", "replace", "--freq", "1",
        "--safe-region-shape", "orthotope", "--growth-method", "IBP",
        "--certification-method", "IBP", "--surrogate", "logsumexp",
        "--rashomon-objective", "weighted_width", "--rashomon-multi-label-mode", "all",
        "--rashomon-checkpoint", "100", "--rashomon-batch-size", "auto",
        "--n-iters", str(settings["rashomon_n_iters"]),
        "--total-timesteps", str(settings["total_timesteps"]),
        "--learning-rate", str(settings["learning_rate"]), "--n-steps", str(settings["n_steps"]),
        "--batch-size", str(settings["batch_size"]), "--n-epochs", str(settings["n_epochs"]),
        "--gamma", "0.99", "--gae-lambda", "0.95", "--clip-range", "0.2",
        "--ent-coef", "0.0", "--vf-coef", "0.5", "--max-grad-norm", "0.5",
        "--eval-episodes", str(settings["eval_episodes"]), "--evaluation-policy", "unshielded",
        "--success-reward-threshold", "-110", "--early-stop-eval-policy", "unshielded",
        "--early-stop-eval-freq", "0", "--early-stop-success-rate", "1.1",
        "--curve-eval-freq", "0" if settings["total_timesteps"] == 1024 else "25000",
        "--curve-eval-episodes", "2" if settings["total_timesteps"] == 1024 else "20",
        "--output-dir", str(run), "--run-id", "training", *common,
    ]
    return initialisation, training


def validate_result(run: Path, pair: dict, settings: dict) -> dict:
    metrics = read_json(run / "training/metrics.json")
    summary = read_json(run / "training/summary.json")
    config = read_json(run / "training/config.json")
    initial = read_json(run / "initialisation/summary.json")
    if not initial["final_verification"]["all_certified"] or not summary["final_interval_certified"]:
        raise RuntimeError("Initial or final full-box certificate failed")
    if summary["evaluation_policy"] != "unshielded" or metrics["eval_episodes"] != settings["eval_episodes"]:
        raise RuntimeError("Evaluation policy or episode count differs from the experiment")
    expected_steps = math.ceil(settings["total_timesteps"] / settings["n_steps"]) * settings["n_steps"]
    if summary["final_timesteps"] != expected_steps or summary["early_stop_triggered"]:
        raise RuntimeError("Training did not finish its fixed budget")
    for key, value in shield_config(pair["upper_bound"]).items():
        if config["continuous_shield_config"][key] != value:
            raise RuntimeError(f"Wrong trained shield setting: {key}")
    if config["base_policy_sha256"] != sha256(run / "initialisation/base_policy.pt"):
        raise RuntimeError("Training base-policy hash mismatch")
    safety = float(metrics["safety"]["safety_rate"])
    if not math.isclose(safety, summary["trajectory_safety"]["safe_trajectory_rate"], abs_tol=1e-12):
        raise RuntimeError("Independent wall-contact audits disagree")
    audit = summary["evaluation_proposed_action_safety"]
    if not audit["proposed_action_checks"]:
        raise RuntimeError("Empty final action audit")
    result = {
        **pair, "mean_total_reward": float(metrics["reward"]["mean_total_reward"]),
        "safety_rate": safety, "eval_episodes": metrics["eval_episodes"],
        "action_compliance": 1 - audit["unsafe_proposed_action_count"] / audit["proposed_action_checks"],
        "unsafe_proposed_action_count": audit["unsafe_proposed_action_count"],
        "final_interval_certified": True, "final_timesteps": summary["final_timesteps"],
        "training_wall_time_s": summary["timing"]["training_wall_time_s"],
        "base_policy_sha256": config["base_policy_sha256"],
    }
    if not all(math.isfinite(result[key]) for key in ("mean_total_reward", "safety_rate", "action_compliance")):
        raise RuntimeError("Nonfinite final metrics")
    return result


def worker(root: Path, pair_id: str) -> None:
    record = read_json(root / "experiment.json")
    pair = next(pair for pair in record["pairs"] if pair["id"] == pair_id)
    run = root / pair_id
    run.mkdir(parents=True, exist_ok=False)
    status = {**pair, "status": "running", "phase": "preflight", "pid": os.getpid(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "started_utc": utc_now()}
    write_json(run / "status.json", status)
    try:
        if len(status["cpu_affinity"]) != 1:
            raise RuntimeError("Worker must be pinned to exactly one CPU")
        verify_provenance(root, record)
        commands = build_commands(root, pair, record["settings"])
        write_json(run / "commands.json", dict(zip(("initialisation", "training"), commands)))
        for phase, command in zip(("initialisation", "training"), commands):
            print(f"[{utc_now()}] {pair_id}: {phase}", flush=True)
            process = subprocess.Popen(command, cwd=REPO)
            status.update({"phase": phase, "stage_pid": process.pid})
            write_json(run / "status.json", status)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"{phase} exited {returncode}")
        write_json(run / "result.json", validate_result(run, pair, record["settings"]))
        status.update({"status": "complete", "phase": "complete", "completed_utc": utc_now()})
        write_json(run / "status.json", status)
    except BaseException as error:
        status.update({"status": "failed", "error": str(error), "completed_utc": utc_now()})
        write_json(run / "status.json", status)
        traceback.print_exc()
        raise


def select_idle_physical_cpus(topology: str, idle: dict[int, float], allowed: set[int],
                              busy: set[int], min_idle: float) -> list[int]:
    physical = {}
    for line in topology.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, online = line.split(",")
        if online == "Y":
            physical.setdefault((int(socket), int(core)), []).append(int(cpu))
    admitted = []
    for siblings in physical.values():
        if any(cpu in busy or cpu in (0, 1) or idle.get(cpu, 0) < min_idle for cpu in siblings):
            continue
        choices = sorted(set(siblings) & allowed)
        if choices:
            admitted.append(choices[0])
    return sorted(admitted)


def free_physical_cpus(busy: set[int], min_idle: float) -> tuple[list[int], dict]:
    sample = json.loads(subprocess.check_output(["mpstat", "-o", "JSON", "-P", "ALL", "1", "5"], text=True))
    observations = {}
    for frame in sample["sysstat"]["hosts"][0]["statistics"]:
        for cpu in frame["cpu-load"]:
            if cpu["cpu"] != "all":
                observations.setdefault(int(cpu["cpu"]), []).append(float(cpu["idle"]))
    idle = {cpu: statistics.fmean(values) for cpu, values in observations.items()}
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True)
    admitted = select_idle_physical_cpus(topology, idle, set(os.sched_getaffinity(0)), busy, min_idle)
    return admitted, {"sampled_utc": utc_now(), "mean_idle_percent": idle,
                      "topology": topology, "min_idle": min_idle, "admitted_cpus": admitted}


def write_report(root: Path) -> dict:
    record = read_json(root / "experiment.json")
    rows, statuses = [], {}
    for pair in record["pairs"]:
        run = root / pair["id"]
        path = run / "status.json"
        status = read_json(path)["status"] if path.exists() else "queued"
        statuses[pair["id"]] = status
        if status == "complete":
            result = validate_result(run, pair, record["settings"])
            if result != read_json(run / "result.json"):
                raise RuntimeError(f"Completed result changed: {pair['id']}")
            rows.append(result)
    fields = list(rows[0]) if rows else ["id", "upper_bound", "seed", "mean_total_reward", "safety_rate", "action_compliance"]
    temporary = root / "per_seed.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "per_seed.csv")
    aggregate = {"updated_utc": utc_now(), "completed_pairs": len(rows), "requested_pairs": len(statuses),
                 "failed_pairs": [key for key, value in statuses.items() if value == "failed"],
                 "statuses": statuses, "intervals": []}
    report = ["# MountainCar PSPO shield-interval sweep", "",
              f"Completed: {len(rows)}/{len(statuses)}; failed: {len(aggregate['failed_pairs'])}.", "",
              "Native unshaped reward; unshielded greedy evaluation. Safety = episodes without left-wall contact.",
              "Separate safe initialisation per interval/seed. Uncertainty is ±2 SE across completed seeds.", "",
              "| Position interval (v < 0) | Seeds | Total reward | Safety (%) | Action compliance (%) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for upper in sorted({pair["upper_bound"] for pair in record["pairs"]}):
        selected = [row for row in rows if row["upper_bound"] == upper]
        item = {"upper_bound": upper, "completed_seeds": len(selected),
                "requested_seeds": sum(pair["upper_bound"] == upper for pair in record["pairs"])}
        displays = []
        for key, scale in (("mean_total_reward", 1), ("safety_rate", 100), ("action_compliance", 100)):
            values = [row[key] for row in selected]
            item[key] = statistics.fmean(values) if values else None
            item[key + "_2se"] = 2 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
            displays.append("pending" if not values else f"{scale * item[key]:.2f}" +
                            (f" ± {scale * item[key + '_2se']:.2f}" if len(values) > 1 else " (one seed)"))
        aggregate["intervals"].append(item)
        report.append(f"| [-1.2, {upper:.2f}] | {len(selected)}/{item['requested_seeds']} | " + " | ".join(displays) + " |")
    if aggregate["failed_pairs"]:
        report += ["", "Failed pairs: " + ", ".join(aggregate["failed_pairs"])]
    write_json(root / "aggregate.json", aggregate)
    (root / "report.md").write_text("\n".join(report) + "\n")
    return aggregate


def pending_pairs(root: Path, pairs: list[dict]) -> list[dict]:
    # Interrupted/failed pair directories are never reused or overwritten.
    return [pair for pair in pairs if not (root / pair["id"]).exists()]


def controller(root: Path, min_idle: float, max_concurrent: int,
               *, entrypoint: Path | None = None, report_fn=None) -> None:
    # Other continuous-state sweeps reuse scheduling without changing the
    # MountainCar worker or report format.
    entrypoint = entrypoint or Path(__file__).resolve()
    report_fn = report_fn or write_report
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read_json(root / "experiment.json")
        verify_provenance(root, record)
        manifest_path = root / "launch_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {
            "started_utc": utc_now(), "screen_name": record["screen_name"], "jobs": [], "cpu_samples": [],
        }
        # Never start a second controller alongside workers orphaned by a crash.
        for pair in record["pairs"]:
            path = root / pair["id"] / "status.json"
            if path.exists() and read_json(path)["status"] == "running":
                raise RuntimeError(f"Existing running/interrupted worker {pair['id']}; inspect before resuming")
        pending, active = pending_pairs(root, record["pairs"]), {}
        manifest.update({"controller_pid": os.getpid(), "max_concurrent": max_concurrent, "min_idle": min_idle})
        write_json(manifest_path, manifest)
        report_fn(root)
        while pending or active:
            for pair_id, (process, cpu, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[pair_id]
                status_path = root / pair_id / "status.json"
                status = read_json(status_path) if status_path.exists() else {"id": pair_id}
                if code or status.get("status") != "complete":
                    status.update({"status": "failed", "worker_exit_code": code, "completed_utc": utc_now()})
                    write_json(status_path, status)
                job = next(job for job in reversed(manifest["jobs"]) if job["id"] == pair_id)
                job.update({"exit_code": code, "completed_utc": utc_now()})
                print(f"[{utc_now()}] {pair_id}: exit={code} cpu={cpu}", flush=True)
                report_fn(root)
            capacity = max_concurrent - len(active)
            if pending and capacity:
                cpus, evidence = free_physical_cpus({value[1] for value in active.values()}, min_idle)
                manifest["cpu_samples"].append(evidence)
                for cpu in cpus[:min(capacity, len(pending))]:
                    pair = pending.pop(0)
                    log_path = root / "_logs" / (pair["id"].replace("/", "_") + ".log")
                    log = log_path.open("a")
                    command = ["taskset", "-c", str(cpu), str(PYTHON), "-u", str(entrypoint),
                               "--worker", pair["id"], "--output-root", str(root)]
                    process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                    active[pair["id"]] = (process, cpu, log)
                    manifest["jobs"].append({**pair, "cpu": cpu, "pid": process.pid,
                                             "started_utc": utc_now(), "command": command, "log_path": str(log_path)})
                    print(f"[{utc_now()}] launched {pair['id']} pid={process.pid} cpu={cpu}", flush=True)
            manifest.update({"queued_pairs": [pair["id"] for pair in pending], "active_pairs": list(active)})
            write_json(manifest_path, manifest)
            if pending or active:
                time.sleep(5)
        aggregate = report_fn(root)
        manifest.update({"completed_utc": utc_now(), "failed_pairs": aggregate["failed_pairs"]})
        write_json(manifest_path, manifest)
        print(f"[{utc_now()}] finished; report={root / 'report.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--controller", action="store_true")
    mode.add_argument("--worker")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume only untouched pairs, never overwrite an existing pair")
    parser.add_argument("--min-idle", type=float, default=95)
    parser.add_argument("--max-concurrent", type=int, default=60)
    args = parser.parse_args()
    if not 0 < args.min_idle <= 100 or not 1 <= args.max_concurrent <= 60:
        parser.error("--min-idle must be in (0,100]; --max-concurrent must be in 1..60")
    if args.output_root is None:
        if not args.launch:
            parser.error("--output-root is required except for a fresh launch")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_root = REPO / "outputs/continuous_state_shields/pspo" / (stamp + "_mountaincar_interval_sweep" + ("_smoke" if args.smoke else ""))
    root = args.output_root.resolve()
    if args.worker:
        worker(root, args.worker)
    elif args.controller:
        controller(root, args.min_idle, args.max_concurrent)
    elif args.report_only:
        write_report(root)
    else:
        if args.resume:
            record = read_json(root / "experiment.json")
            verify_provenance(root, record)
        else:
            record = prepare(root, args.smoke)
        screens = subprocess.run(["screen", "-ls"], text=True, capture_output=True).stdout
        if f".{record['screen_name']}\t" in screens:
            raise RuntimeError("This experiment's screen session is already running")
        subprocess.run([
            "screen", "-L", "-Logfile", str(root / "_logs/screen.log"), "-dmS", record["screen_name"],
            str(PYTHON), "-u", str(Path(__file__).resolve()), "--controller", "--output-root", str(root),
            "--min-idle", str(args.min_idle), "--max-concurrent", str(args.max_concurrent),
        ], cwd=REPO, check=True)
        print(f"Detached screen: {record['screen_name']}\nOutput root: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
