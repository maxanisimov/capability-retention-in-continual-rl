#!/usr/bin/env python3
"""Launch PSPO's six LunarLander viewport shields × ten seeds in screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_mountaincar_shield_interval_sweep as scheduler,
)

read_json, write_json, sha256, utc_now = scheduler.read_json, scheduler.write_json, scheduler.sha256, scheduler.utc_now
MS = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
PYTHON, STAGES = scheduler.PYTHON, scheduler.STAGES


def band_name(m: float) -> str:
    return "m_" + f"{m:.2f}".replace(".", "p")


def settings(smoke: bool) -> dict:
    return {
        "env_id": "LunarLander-v3", "env_kwargs": {"continuous": False, "enable_wind": False},
        "max_episode_steps": 1000, "total_timesteps": 2048 if smoke else 500_000,
        "learning_rate": 0.0003, "n_steps": 2048, "batch_size": 256,
        "n_epochs": 1 if smoke else 10, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "eval_episodes": 2 if smoke else 100, "evaluation_policy": "unshielded greedy",
        "evaluation_seed_offset": 10_000, "early_stopping": False,
        "separate_safe_initialisation_per_pair": True, "hidden_dim": 64, "n_hidden": 2,
        "bc_samples": 4096, "bc_epochs": 200, "init_max_epochs": 2000,
        "init_learning_rate": 0.01, "init_target_margin": 2.0,
        "verify_first": False, "directional": True, "region_mode": "replace", "freq": "1",
        "safe_region_shape": "orthotope", "growth_method": "IBP", "certification_method": "IBP",
        "rashomon_n_iters": 2 if smoke else 200, "rashomon_objective": "weighted_width",
        "surrogate": "logsumexp", "rashomon_multi_label_mode": "all", "rashomon_batch_size": "auto",
        "reward": "native unshaped", "safety_rate": "episodes never reaching abs(raw normalised x) >= 1",
        "push_left_action": 1, "push_right_action": 3,
        "note": "Action constraint only; inward impulse assumes upright orientation; no trajectory guarantee.",
    }


def prepare(root: Path, smoke: bool) -> dict:
    import gymnasium as gym
    import torch
    from continuous_state_shields import (
        LunarLanderViewportShield,
        LunarLanderViewportShieldConfig,
    )
    from continuous_state_shields.lunar_lander_viewport import (
        viewport_certificate_arrays,
    )

    from projects.safe_policy_optimisation.stages.train_pspo_continuous import (
        validate_lunarlander_certificate,
    )

    root.mkdir(parents=True, exist_ok=False)
    (root / "_logs").mkdir()
    bounds, seeds = ((MS[0], MS[-1]), (0,)) if smoke else (MS, range(10))
    pairs = [{"id": f"{band_name(m)}/seed_{seed}", "m": m, "seed": seed} for seed in seeds for m in bounds]
    env = gym.make("LunarLander-v3", continuous=False, enable_wind=False)
    try:
        low, high = env.observation_space.low.copy(), env.observation_space.high.copy()
    finally:
        env.close()
    inputs = {}
    for m in bounds:
        directory = root / "_inputs" / band_name(m)
        directory.mkdir(parents=True)
        arrays = viewport_certificate_arrays(m, low, high)
        dataset = torch.utils.data.TensorDataset(*(torch.as_tensor(array) for array in arrays))
        validate_lunarlander_certificate(dataset, LunarLanderViewportShield(LunarLanderViewportShieldConfig(m=m)))
        path = directory / "critical_interval_dataset.pt"
        torch.save(dataset, path)
        inputs[str(path.relative_to(root))] = sha256(path)
        config = directory / "shield_config.json"
        write_json(config, {"m": m})
        inputs[str(config.relative_to(root))] = sha256(config)
    paths = set(scheduler.source_paths()) | {Path(__file__).resolve()}
    sources = {str(path.relative_to(REPO)): sha256(path) for path in sorted(paths)}
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO / name, name)
    record = {
        "created_utc": utc_now(), "smoke": smoke, "pairs": pairs, "settings": settings(smoke),
        "screen_name": "pspo-ll-viewport-" + root.name,
        "input_sha256": inputs, "source_sha256": sources,
        "observation_low": low.tolist(), "observation_high": high.tolist(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "gymnasium_version": gym.__version__, "torch_version": torch.__version__,
    }
    write_json(root / "experiment.json", record)
    write_report(root)
    return record


def build_commands(root: Path, pair: dict, config: dict) -> tuple[list[str], list[str]]:
    run = root / pair["id"]
    certificate = root / "_inputs" / band_name(pair["m"]) / "critical_interval_dataset.pt"
    common = ["--seed", str(pair["seed"]), "--device", "cpu"]
    initialisation = [
        str(PYTHON), "-u", str(STAGES / "train_mountaincar_pspo_initialisation.py"),
        "--output-dir", str(run), "--run-id", "initialisation",
        "--certificate-dataset", str(certificate), "--obs-dim", "8", "--n-actions", "4",
        "--hidden-dim", "64", "--n-hidden", "2", "--bc-samples", "4096", "--bc-epochs", "200",
        "--max-epochs", "2000", "--target-margin", "2.0", "--learning-rate", "0.01",
        "--certification-method", "IBP", *common,
    ]
    training = [
        str(PYTHON), "-u", str(STAGES / "train_pspo_continuous.py"),
        "--env-id", "LunarLander-v3", "--env-kwargs", json.dumps(config["env_kwargs"]),
        "--continuous-shield", "lunarlander-viewport", "--continuous-shield-config", json.dumps({"m": pair["m"]}),
        "--base-policy-path", str(run / "initialisation/base_policy.pt"),
        "--certificate-dataset", str(certificate), "--max-episode-steps", "1000",
        "--mountaincar-shaped-reward", "false", "--verify-first", "false", "--directional", "true",
        "--region-mode", "replace", "--freq", "1", "--safe-region-shape", "orthotope",
        "--growth-method", "IBP", "--certification-method", "IBP", "--surrogate", "logsumexp",
        "--rashomon-objective", "weighted_width", "--rashomon-multi-label-mode", "all",
        "--rashomon-batch-size", "auto", "--rashomon-checkpoint", "100",
        "--n-iters", str(config["rashomon_n_iters"]), "--total-timesteps", str(config["total_timesteps"]),
        "--learning-rate", "0.0003", "--n-steps", "2048", "--batch-size", "256",
        "--n-epochs", str(config["n_epochs"]), "--gamma", "0.99", "--gae-lambda", "0.95",
        "--clip-range", "0.2", "--ent-coef", "0.01", "--vf-coef", "0.5", "--max-grad-norm", "0.5",
        "--eval-episodes", str(config["eval_episodes"]), "--evaluation-policy", "unshielded",
        "--success-reward-threshold", "200", "--early-stop-eval-policy", "unshielded",
        "--early-stop-eval-freq", "0", "--early-stop-success-rate", "1.1",
        "--curve-eval-freq", "0" if config["total_timesteps"] == 2048 else "25000",
        "--curve-eval-episodes", "2" if config["total_timesteps"] == 2048 else "20",
        "--output-dir", str(run), "--run-id", "training", *common,
    ]
    return initialisation, training


def validate_result(run: Path, pair: dict, config: dict) -> dict:
    metrics = read_json(run / "training/metrics.json")
    summary = read_json(run / "training/summary.json")
    trained_config = read_json(run / "training/config.json")
    initial = read_json(run / "initialisation/summary.json")
    if not initial["final_verification"]["all_certified"] or not summary["final_interval_certified"]:
        raise RuntimeError("Initial or final full-box certificate failed")
    if summary["evaluation_policy"] != "unshielded" or metrics["eval_episodes"] != config["eval_episodes"]:
        raise RuntimeError("Wrong evaluation policy or episode count")
    steps = math.ceil(config["total_timesteps"] / config["n_steps"]) * config["n_steps"]
    if summary["final_timesteps"] != steps or summary["early_stop_triggered"]:
        raise RuntimeError("Training did not finish its fixed budget")
    if trained_config["continuous_shield"] != "lunarlander-viewport" or trained_config["continuous_shield_config"]["m"] != pair["m"]:
        raise RuntimeError("Wrong shield in trained policy")
    for key in ("push_left_action", "push_right_action"):
        if trained_config["continuous_shield_config"][key] != config[key]:
            raise RuntimeError(f"Wrong shield action direction: {key}")
    if trained_config["base_policy_sha256"] != sha256(run / "initialisation/base_policy.pt"):
        raise RuntimeError("Initial-policy hash mismatch")
    trajectory = summary["trajectory_safety"]
    if not math.isclose(metrics["safety"]["safety_rate"], trajectory["safe_trajectory_rate"], abs_tol=1e-12):
        raise RuntimeError("Independent out-of-view audits disagree")
    with (run / "training/episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    audited = trajectory["per_episode"]
    if len(episodes) != config["eval_episodes"] or len(audited) != len(episodes):
        raise RuntimeError("Incomplete final episode audit")
    for row, audit in zip(episodes, audited):
        if int(row["length"]) != audit["length"] or not math.isclose(float(row["reward"]), audit["reward"], abs_tol=1e-5):
            raise RuntimeError("Independent final evaluation rewards/lengths disagree")
    action = summary["evaluation_proposed_action_safety"]
    if not action["proposed_action_checks"]:
        raise RuntimeError("Empty nominal action-compliance audit")
    return {
        **pair, "mean_total_reward": metrics["reward"]["mean_total_reward"],
        "safety_rate": metrics["safety"]["safety_rate"],
        "action_compliance": 1 - action["unsafe_proposed_action_count"] / action["proposed_action_checks"],
        "unsafe_proposed_action_count": action["unsafe_proposed_action_count"],
        "no_truncation_rate": trajectory["no_truncation_rate"],
        "successful_landing_rate": trajectory["successful_landing_rate"],
        "reward_threshold_success_rate": metrics["success"]["success_rate"],
        "eval_episodes": metrics["eval_episodes"], "final_interval_certified": True,
        "final_timesteps": summary["final_timesteps"],
        "training_wall_time_s": summary["timing"]["training_wall_time_s"],
        "base_policy_sha256": trained_config["base_policy_sha256"],
    }


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
        scheduler.verify_provenance(root, record)
        commands = build_commands(root, pair, record["settings"])
        write_json(run / "commands.json", dict(zip(("initialisation", "training"), commands)))
        for phase, command in zip(("initialisation", "training"), commands):
            print(f"[{utc_now()}] {pair_id}: {phase}", flush=True)
            process = subprocess.Popen(command, cwd=REPO)
            status.update({"phase": phase, "stage_pid": process.pid})
            write_json(run / "status.json", status)
            code = process.wait()
            if code:
                raise RuntimeError(f"{phase} exited {code}")
        result = validate_result(run, pair, record["settings"])
        write_json(run / "result.json", result)
        status.update({"status": "complete", "phase": "complete", "completed_utc": utc_now()})
        write_json(run / "status.json", status)
    except BaseException as error:
        status.update({"status": "failed", "error": str(error), "completed_utc": utc_now()})
        write_json(run / "status.json", status)
        traceback.print_exc()
        raise


def write_report(root: Path) -> dict:
    record = read_json(root / "experiment.json")
    rows, statuses = [], {}
    for pair in record["pairs"]:
        path = root / pair["id"] / "status.json"
        status = read_json(path)["status"] if path.exists() else "queued"
        statuses[pair["id"]] = status
        if status == "complete":
            result = validate_result(root / pair["id"], pair, record["settings"])
            if result != read_json(root / pair["id"] / "result.json"):
                raise RuntimeError(f"Completed result changed: {pair['id']}")
            rows.append(result)
    temporary = root / "per_seed.csv.tmp"
    with temporary.open("w", newline="") as handle:
        fields = list(rows[0]) if rows else ["id", "m", "seed", "mean_total_reward", "safety_rate"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "per_seed.csv")
    aggregate = {"updated_utc": utc_now(), "completed_pairs": len(rows), "requested_pairs": len(statuses),
                 "failed_pairs": [key for key, status in statuses.items() if status == "failed"], "statuses": statuses, "intervals": []}
    report = ["# LunarLander PSPO viewport-shield sweep", "",
              f"Completed: {len(rows)}/{len(statuses)}; failed: {len(aggregate['failed_pairs'])}.", "",
              "Native reward; final unshielded greedy evaluation, 100 episodes/seed in production. Mean ±2 SE across seeds.",
              "Safety = never abs(raw normalised x) >= 1. Rule: action 1 for [m,1], action 3 for [-1,-m].",
              "Separate certified initialisation per pair. The action rule does not guarantee viewport containment.",
              "No truncation includes crashes and viewport exits; it is not equivalent to a successful landing.", "",
              "| m | Seeds | Total reward | Safety (%) | Action compliance (%) | No truncation (%) | Landed (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in sorted({pair["m"] for pair in record["pairs"]}, reverse=True):
        selected = [row for row in rows if row["m"] == m]
        item = {"m": m, "completed_seeds": len(selected), "requested_seeds": sum(pair["m"] == m for pair in record["pairs"])}
        displays = []
        for key in ("mean_total_reward", "safety_rate", "action_compliance", "no_truncation_rate", "successful_landing_rate", "reward_threshold_success_rate"):
            values = [row[key] for row in selected]
            item[key] = statistics.fmean(values) if values else None
            item[key + "_2se"] = 2 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
            scale = 1 if key == "mean_total_reward" else 100
            displays.append("pending" if not values else f"{scale * item[key]:.2f}" +
                            (f" ± {scale * item[key + '_2se']:.2f}" if len(values) > 1 else " (one seed)"))
        aggregate["intervals"].append(item)
        report.append(f"| {m:.2f} | {len(selected)}/{item['requested_seeds']} | " + " | ".join(displays[:5]) + " |")
    if aggregate["failed_pairs"]:
        report += ["", "Failed pairs: " + ", ".join(aggregate["failed_pairs"])]
    write_json(root / "aggregate.json", aggregate)
    (root / "report.md").write_text("\n".join(report) + "\n")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--controller", action="store_true")
    mode.add_argument("--worker")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Dispatch untouched pairs only; never overwrite artifacts")
    parser.add_argument("--min-idle", type=float, default=95)
    parser.add_argument("--max-concurrent", type=int, default=60)
    args = parser.parse_args()
    if not 0 < args.min_idle <= 100 or not 1 <= args.max_concurrent <= 60:
        parser.error("--min-idle must be in (0,100]; --max-concurrent must be in 1..60")
    if args.output_root is None:
        if not args.launch:
            parser.error("--output-root is required except for a fresh launch")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_root = REPO / "outputs/continuous_state_shields/pspo" / (stamp + "_lunarlander_viewport_sweep" + ("_smoke" if args.smoke else ""))
    root = args.output_root.resolve()
    if args.worker:
        worker(root, args.worker)
    elif args.controller:
        scheduler.controller(root, args.min_idle, args.max_concurrent, entrypoint=Path(__file__).resolve(), report_fn=write_report)
    elif args.report_only:
        write_report(root)
    else:
        record = read_json(root / "experiment.json") if args.resume else prepare(root, args.smoke)
        scheduler.verify_provenance(root, record)
        screens = subprocess.run(["screen", "-ls"], text=True, capture_output=True).stdout
        if f".{record['screen_name']}\t" in screens:
            raise RuntimeError("This experiment's screen is already running")
        subprocess.run([
            "screen", "-L", "-Logfile", str(root / "_logs/screen.log"), "-dmS", record["screen_name"],
            str(PYTHON), "-u", str(Path(__file__).resolve()), "--controller", "--output-root", str(root),
            "--min-idle", str(args.min_idle), "--max-concurrent", str(args.max_concurrent),
        ], cwd=REPO, check=True)
        print(f"Detached screen: {record['screen_name']}\nOutput root: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
