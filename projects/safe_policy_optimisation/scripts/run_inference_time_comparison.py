#!/usr/bin/env python3
"""Time deployed PSPO and shield-on PPO decisions during actual environment steps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core"), str(Path(__file__).resolve().parent)]

from run_timed_main_comparison import THREAD_ENV, read_json, write_json  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(
    output: Path,
    seeds: list[int],
    environments: list[str] | None,
    steps: int,
    warmup: int,
    pspo_variant: str = "orthotope",
) -> dict:
    from plot_extended_budget_learning_curves import ENVIRONMENTS, RUNS

    if pspo_variant not in {"orthotope", "segment"}:
        raise ValueError("Unknown PSPO variant")
    if environments and set(environments) - {env.key for env in ENVIRONMENTS}:
        raise ValueError("Unknown environment")
    jobs = []
    for env in ENVIRONMENTS:
        if environments and env.key not in environments:
            continue
        for seed in seeds:
            for method, root in (
                ("pspo", RUNS / "segment_lid/two_hidden" / env.key
                 if pspo_variant == "segment" else env.adaptive_root),
                ("ppo_shield", env.baseline_root),
            ):
                source = root / f"seed{seed}"
                if method == "ppo_shield":
                    source /= "ppo_shield"
                model = source / "model.zip"
                config = source / "config.json"
                if not model.is_file() or not config.is_file():
                    raise FileNotFoundError(f"Missing checkpoint or config in {source}")
                values = read_json(config)
                if method == "pspo" and pspo_variant == "segment":
                    adaptive = values["adaptive"]
                    if adaptive["safe_region_shape"] != "segment" or adaptive["verify_first"]:
                        raise ValueError(f"Not region-first PSPO-LS: {config}")
                if method == "ppo_shield" and not Path(values["shield_path"]).is_file():
                    raise FileNotFoundError(values["shield_path"])
                jobs.append(
                    {
                        "id": f"{env.key}/seed{seed}/{method}",
                        "environment": env.key,
                        "environment_label": env.label,
                        "seed": seed,
                        "method": method,
                        "model_path": str(model),
                        "configuration_source": str(config),
                        "config": values,
                        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                        "steps": steps,
                        "warmup_steps": warmup,
                    }
                )
    return {
        "output_dir": str(output),
        "created_at_utc": utc_now(),
        "pspo_variant": pspo_variant,
        "jobs": jobs,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": {
            "deterministic": True,
            "batch_size": 1,
            "device": "cpu",
            "torch_threads": 1,
            "pspo": ("Unshielded saved region-first PSPO-LS actor" if pspo_variant == "segment"
                     else "Unshielded saved default orthotope PSPO actor"),
            "ppo_shield": "Saved actor plus canonical Shield override (shield on)",
            "inference_s": "Sum of per-step wall times: predict, action conversion, and runtime shield if applicable; excludes env.step/reset",
            "rollout_s": "Full measured rollout including decisions, env.step/reset, and loop overhead; excludes setup, warmup, and progress-file writes",
            "reset_seed": "10000 + training seed + completed episode number",
            "saved_metrics": "Timing and progress only; no rewards or safety rates",
            "error_bar": "2 * sample standard deviation (ddof=1) / sqrt(seed count)",
            "checkpoint_cohort": "Existing main-performance cohorts, not timing-only retraining",
            "latency_table_error_bar": "sample standard deviation (ddof=1) / sqrt(10); reductions computed within each seed pair",
        },
    }


def measure_rollout(
    model, env, shield, *, steps: int, seed: int, progress=None, interval: int = 10000
) -> dict:
    """Execute exactly steps transitions, resetting on termination OR truncation."""
    import numpy as np

    from projects.safe_policy_optimisation.stages.train_ppo_shield import _override_one

    episodes = 0
    obs, _ = env.reset(seed=seed)
    inference_s = prediction_s = shield_s = rollout_s = 0.0
    block_started = time.perf_counter()
    for step in range(1, steps + 1):
        started = time.perf_counter()
        proposed, _ = model.predict(obs, deterministic=True)
        predicted = time.perf_counter()
        action = int(np.asarray(proposed).item())
        converted = time.perf_counter()
        if shield is not None:
            action = _override_one(shield, obs, action)
        finished = time.perf_counter()
        inference_s += finished - started
        prediction_s += predicted - started
        if shield is not None:
            shield_s += finished - converted
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset(seed=seed + episodes)
        if step % interval == 0 or step == steps:
            rollout_s += time.perf_counter() - block_started
            if progress is not None:
                progress(
                    {
                        "steps_completed": step,
                        "target_steps": steps,
                        "inference_s": inference_s,
                        "rollout_s": rollout_s,
                        "updated_at_utc": utc_now(),
                    }
                )
            block_started = time.perf_counter()
    return {
        "environment_steps": steps,
        "completed_episodes": episodes,
        "inference_s": inference_s,
        "prediction_s": prediction_s,
        "shield_s": shield_s,
        "rollout_s": rollout_s,
        "mean_inference_us_per_step": inference_s / steps * 1e6,
    }


def worker(manifest: dict, job: dict, cpu: int) -> int:
    os.sched_setaffinity(0, {cpu})
    import torch
    from provably_safe_policy_optimisation import Shield
    from stable_baselines3 import PPO

    from projects.safe_policy_optimisation.stages.train_ppo_shield import (
        make_unshielded_env,
    )
    from projects.safe_policy_optimisation.utils.shield import load_shield_mask

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = Path(manifest["output_dir"]) / "runs" / job["id"]
    result = {
        "status": "running",
        "job_id": job["id"],
        "environment": job["environment"],
        "method": job["method"],
        "seed": job["seed"],
        "hostname": socket.gethostname(),
        "cpu_id": cpu,
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "model_path": job["model_path"],
        "warmup_steps": job["warmup_steps"],
    }
    env = None
    try:
        config = job["config"]
        # Load only the unchanged SB3 policy, avoiding PSPO's training/certification hooks.
        model = PPO.load(job["model_path"], device="cpu")
        model.policy.set_training_mode(False)
        env = make_unshielded_env(
            config["env_id"],
            env_kwargs=config["env_kwargs"],
            max_episode_steps=config.get("max_episode_steps"),
            cost_limit=config.get("cost_limit", 0.0),
            record_episodes=False,
        )
        shield = None
        if job["method"] == "ppo_shield":
            mask = load_shield_mask(
                Path(config["shield_path"]),
                shield_key=config.get("shield_key", "shield"),
                source=config.get("shield_source", "shield"),
                risk_threshold=config.get("risk_threshold"),
            )
            shield = Shield(mask, seed=job["seed"])
        with torch.no_grad():
            if job["warmup_steps"]:
                measure_rollout(
                    model,
                    env,
                    shield,
                    steps=job["warmup_steps"],
                    seed=job["seed"] + 1000000,
                )
            if shield is not None:
                shield = Shield(mask, seed=job["seed"])
            result.update(
                measure_rollout(
                    model,
                    env,
                    shield,
                    steps=job["steps"],
                    seed=job["seed"] + 10000,
                    progress=lambda values: write_json(
                        directory / "progress.json", values
                    ),
                )
            )
        result["status"] = "complete"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        result["finished_at_utc"] = utc_now()
        write_json(directory / "inference_time.json", result)
    return int(result["status"] != "complete")


def mean_two_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, 2 * math.sqrt(variance / len(values))


def write_latency_table(manifest: dict, groups: dict) -> None:
    """Export the paper table only after ten complete paired million-step runs."""
    if manifest["pspo_variant"] != "segment":
        return
    labels = dict((job["environment"], job["environment_label"]) for job in manifest["jobs"])
    rows, pairs = [], []
    for environment, label in labels.items():
        by_method = {
            method: {result["seed"]: result for result in groups.get((environment, method), [])}
            for method in ("pspo", "ppo_shield")
        }
        if any(set(results) != set(range(10)) for results in by_method.values()):
            return
        times, reductions, percentages = [], [], []
        for seed in range(10):
            pspo, shield = (by_method[method][seed] for method in ("pspo", "ppo_shield"))
            if pspo["environment_steps"] != 1_000_000 or shield["environment_steps"] != 1_000_000:
                return
            pspo_time, shield_time = pspo["inference_s"], shield["inference_s"]
            if not (math.isfinite(pspo_time) and math.isfinite(shield_time)
                    and pspo_time > 0 and shield_time > 0):
                raise ValueError("Invalid paired inference timing")
            delta = shield_time - pspo_time
            percent = 100 * delta / shield_time
            times.append(pspo_time)
            reductions.append(delta)
            percentages.append(percent)
            pairs.append({"environment": environment, "seed": seed,
                          "pspo_ls_inference_s": pspo_time, "ppo_shield_inference_s": shield_time,
                          "latency_reduction_s": delta, "latency_reduction_percent": percent})
        row = {"environment": environment, "label": label, "n": 10}
        for field, values in (("pspo_inference_s", times), ("reduction_s", reductions),
                              ("reduction_percent", percentages)):
            row.update({f"{field}_mean": statistics.mean(values),
                        f"{field}_se": statistics.stdev(values) / math.sqrt(10)})
        rows.append(row)
    root = Path(manifest["output_dir"])
    for name, records in (("latency_paired_seeds.csv", pairs), ("latency_summary.csv", rows)):
        with (root / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        r"\begin{table}[h]", r"  \centering",
        r"  \caption{PSPO inference time per 1 million environment steps (in seconds) compared to PPO-Shield (mean $\pm$ standard error over 10 paired seeds). PSPO uses PSPO-LS checkpoints without runtime shielding.}",
        r"  \label{tab:pspo-latency}", "", r"  \small",
        r"  \setlength{\tabcolsep}{3pt}", r"  \renewcommand{\arraystretch}{1.08}", "",
        r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule",
        r"    Environment &",
        r"    \makecell{PSPO\\inference (s)} &",
        r"    \makecell{Latency reduction\\vs PPO-Shield (s)} &",
        r"    \makecell{Latency reduction\\vs PPO-Shield (\%)} \\",
        r"    \midrule",
    ]
    for row in rows:
        cells = [f"${row[field + '_mean']:.2f} \\pm {row[field + '_se']:.2f}$"
                 for field in ("pspo_inference_s", "reduction_s", "reduction_percent")]
        lines.append("    " + row["label"] + " & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (root / "latency_table.tex").write_text("\n".join(lines) + "\n")


def aggregate(manifest: dict) -> None:
    root = Path(manifest["output_dir"])
    groups = {}
    for job in manifest["jobs"]:
        path = root / "runs" / job["id"] / "inference_time.json"
        if path.is_file():
            result = read_json(path)
            if result["status"] == "complete":
                groups.setdefault((job["environment"], job["method"]), []).append(
                    result
                )
    rows = []
    for (environment, method), results in groups.items():
        row = {
            "environment": environment,
            "method": method,
            "n": len(results),
            "steps_per_seed": results[0]["environment_steps"],
        }
        for field in (
            "inference_s",
            "prediction_s",
            "shield_s",
            "rollout_s",
            "mean_inference_us_per_step",
        ):
            mean, error = mean_two_se([result[field] for result in results])
            row.update({f"{field}_mean": mean, f"{field}_two_se": error})
        rows.append(row)
    write_json(
        root / "aggregate.json",
        {
            "expected_jobs": len(manifest["jobs"]),
            "completed_jobs": sum(row["n"] for row in rows),
            "rows": rows,
        },
    )
    if rows:
        with (root / "aggregate.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "Inference timing (seconds per fixed environment-step budget).",
        "Mean +/- two standard errors across completed seeds; n=10 required for final results.",
        "PSPO is unshielded; PPO-Shield includes shield checking/overrides.",
        "",
        "| Environment | Method | Seeds | Inference (s) | Full rollout (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | {row['method']} | {row['n']} | "
            f"{row['inference_s_mean']:.3f} +/- {row['inference_s_two_se']:.3f} | "
            f"{row['rollout_s_mean']:.3f} +/- {row['rollout_s_two_se']:.3f} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if manifest.get("pspo_variant") == "segment":
        write_latency_table(manifest, groups)


def supervise(path: Path) -> int:
    manifest = read_json(path)
    root = Path(manifest["output_dir"])
    pending, running, complete, failed = list(manifest["jobs"]), {}, [], []
    cpus = list(manifest["cpu_ids"])
    write_json(
        root / "supervisor.json", {"pid": os.getpid(), "hostname": socket.gethostname()}
    )
    while pending or running:
        for job_id, (process, cpu, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = root / "runs" / job_id / "inference_time.json"
            if (
                code == 0
                and result.is_file()
                and read_json(result)["status"] == "complete"
            ):
                complete.append(job_id)
            else:
                failed.append(job_id)
            cpus.append(cpu)
            del running[job_id]
            print(f"exit={code} {job_id}", flush=True)
            aggregate(manifest)
        while pending and cpus and len(running) < manifest["max_parallel"]:
            job, cpu = pending.pop(0), cpus.pop(0)
            directory = root / "runs" / job["id"]
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "worker.log").open("w")
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
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[job["id"]] = (process, cpu, log)
        write_json(
            root / "status.json",
            {
                "updated_at_utc": utc_now(),
                "total": len(manifest["jobs"]),
                "complete": len(complete),
                "failed": len(failed),
                "pending": len(pending),
                "running": len(running),
                "complete_jobs": complete,
                "failed_jobs": failed,
                "running_jobs": {
                    key: {"pid": value[0].pid, "cpu": value[1]}
                    for key, value in running.items()
                },
            },
        )
        if running:
            time.sleep(2)
    aggregate(manifest)
    return int(bool(failed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--environments", nargs="+")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-parallel", type=int, default=120)
    parser.add_argument("--pspo-variant", choices=("orthotope", "segment"), default="orthotope")
    parser.add_argument("--cpu-ids", help="Comma-separated host CPU IDs to use (default: affinity minus eight reserved CPUs).")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.worker:
        manifest = read_json(args.manifest)
        return worker(
            manifest,
            next(job for job in manifest["jobs"] if job["id"] == args.job_id),
            args.cpu,
        )
    if args.supervise:
        return supervise(args.manifest)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(
            "Output must be a new directory; existing results are never overwritten"
        )
    seeds = [int(value) for value in args.seeds.split(",")]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed not in range(10) for seed in seeds)
        or args.steps < 1
        or args.warmup_steps < 0
        or args.max_parallel < 1
    ):
        parser.error("Invalid seeds, steps, warmup, or parallelism")
    manifest = build_manifest(
        output, seeds, args.environments, args.steps, args.warmup_steps, args.pspo_variant
    )
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_ids:
        try:
            cpus = [int(value) for value in args.cpu_ids.split(",")]
        except ValueError:
            parser.error("--cpu-ids must contain comma-separated integers")
        if not cpus or len(cpus) != len(set(cpus)) or set(cpus) - set(available):
            parser.error("--cpu-ids must contain unique CPUs in the process affinity")
        parallel = min(args.max_parallel, len(cpus))
    else:
        cpus = available
        parallel = min(args.max_parallel, max(1, len(cpus) - 8))
    manifest.update(max_parallel=parallel, cpu_ids=cpus[-parallel:])
    path = output / "manifest.json"
    write_json(path, manifest)
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
        print(
            f"Launched {len(manifest['jobs'])} jobs; supervisor PID {process.pid}; {path}"
        )
    else:
        print(f"Prepared {len(manifest['jobs'])} jobs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
