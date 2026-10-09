#!/usr/bin/env python3
"""Train ten vanilla LunarLander PPO policies and audit viewport safety.

Each seed is trained without a shield or safe initialisation.  After the full
training budget, the deterministic policy is replayed for 100 episodes to
measure raw viewport exits and its actor is soundly verified (IBP) against the
two complete observation boxes induced by each viewport threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_mountaincar_shield_interval_sweep as scheduler,
)

MS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
PYTHON, STAGES = scheduler.PYTHON, scheduler.STAGES
read_json, write_json = scheduler.read_json, scheduler.write_json
sha256, utc_now = scheduler.sha256, scheduler.utc_now


def m_key(m: float) -> str:
    return f"{m:.2f}"


def settings(smoke: bool, init_policy_root: Path | None = None) -> dict[str, Any]:
    return {
        "env_id": "LunarLander-v3",
        "env_kwargs": {"continuous": False, "enable_wind": False},
        "max_episode_steps": 1000,
        "total_timesteps": 2048 if smoke else 500_000,
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 1 if smoke else 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "hidden_dim": 64,
        "n_hidden": 2,
        "eval_episodes": 2 if smoke else 100,
        "evaluation_policy": "unshielded deterministic greedy",
        "evaluation_seed_offset": 10_000,
        "early_stopping": False,
        "success_reward_threshold": 200,
        "viewport_thresholds": list(MS),
        "viewport_exit": "episode reaches abs(raw normalised Box2D x) >= 1, including terminal state",
        "verification": (
            "Sound IBP verification of the greedy actor over both complete 8-D "
            "shield boxes; pass requires both boxes certified"
        ),
        "training_initialisation": (
            "matching seed's certified m=0.95 PSPO actor; critic and optimizer start fresh"
            if init_policy_root is not None
            else "standard independent SB3 PPO random initialisation per seed"
        ),
        "warm_start": init_policy_root is not None,
        "init_policy_source_root": (
            str(init_policy_root.resolve()) if init_policy_root is not None else None
        ),
        "shield_during_training_or_evaluation": False,
    }


def prepare(root: Path, smoke: bool, init_policy_root: Path | None = None) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    from continuous_state_shields.lunar_lander_viewport import viewport_certificate_arrays

    root.mkdir(parents=True, exist_ok=False)
    (root / "_logs").mkdir()
    (root / "_inputs").mkdir()
    seeds = (0,) if smoke else tuple(range(10))
    pairs = [{"id": f"seed_{seed}", "seed": seed} for seed in seeds]

    env = gym.make("LunarLander-v3", continuous=False, enable_wind=False)
    try:
        low = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
    finally:
        env.close()

    inputs: dict[str, str] = {}
    for m in MS:
        arrays = viewport_certificate_arrays(m, low, high)
        dataset = torch.utils.data.TensorDataset(
            *(torch.as_tensor(array) for array in arrays)
        )
        path = root / "_inputs" / f"viewport_m_{m:.2f}.pt"
        torch.save(dataset, path)
        inputs[str(path.relative_to(root))] = sha256(path)

    initialisations: dict[str, Any] = {}
    if init_policy_root is not None:
        init_policy_root = init_policy_root.resolve()
        for seed in seeds:
            source_dir = init_policy_root / f"seed_{seed}" / "initialisation"
            source = source_dir / "base_policy.pt"
            source_summary = source_dir / "summary.json"
            if not source.is_file() or not source_summary.is_file():
                raise FileNotFoundError(
                    f"Missing certified seed-{seed} initialisation under {source_dir}"
                )
            summary = read_json(source_summary)
            final = dict(summary.get("final_verification") or {})
            if not final.get("all_certified") or float(final.get("certified_fraction", 0)) != 1.0:
                raise ValueError(f"Seed {seed} source initialisation is not fully certified")
            destination_dir = root / "_inputs" / "safe_initialisation" / f"seed_{seed}"
            destination_dir.mkdir(parents=True)
            destination = destination_dir / "base_policy.pt"
            shutil.copy2(source, destination)
            relative = str(destination.relative_to(root))
            inputs[relative] = sha256(destination)
            initialisations[str(seed)] = {
                "source_path": str(source),
                "source_sha256": sha256(source),
                "copied_path": relative,
                "copied_sha256": inputs[relative],
                "source_final_verification": final,
            }

    paths = set(scheduler.source_paths()) | {Path(__file__).resolve()}
    sources = {str(path.relative_to(REPO)): sha256(path) for path in sorted(paths)}
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO / name, name)

    record = {
        "created_utc": utc_now(),
        "smoke": smoke,
        "pairs": pairs,
        "settings": settings(smoke, init_policy_root),
        "screen_name": "ppo-ll-viewport-" + root.name,
        "input_sha256": inputs,
        "source_sha256": sources,
        "observation_low": low.tolist(),
        "observation_high": high.tolist(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ),
        "gymnasium_version": gym.__version__,
        "torch_version": torch.__version__,
        "safe_initialisations": initialisations,
    }
    write_json(root / "experiment.json", record)
    write_report(root)
    return record


def build_training_command(root: Path, pair: dict[str, Any], config: dict[str, Any]) -> list[str]:
    run = root / pair["id"]
    command = [
        str(PYTHON), "-u", str(STAGES / "train_ppo.py"),
        "--env-id", config["env_id"],
        "--env-kwargs", json.dumps(config["env_kwargs"]),
        "--max-episode-steps", str(config["max_episode_steps"]),
        "--total-timesteps", str(config["total_timesteps"]),
        "--eval-episodes", str(config["eval_episodes"]),
        "--learning-rate", str(config["learning_rate"]),
        "--n-steps", str(config["n_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--n-epochs", str(config["n_epochs"]),
        "--gamma", str(config["gamma"]),
        "--gae-lambda", str(config["gae_lambda"]),
        "--clip-range", str(config["clip_range"]),
        "--ent-coef", str(config["ent_coef"]),
        "--vf-coef", str(config["vf_coef"]),
        "--max-grad-norm", str(config["max_grad_norm"]),
        "--n-hidden", str(config["n_hidden"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--success-reward-threshold", str(config["success_reward_threshold"]),
        "--early-stop-eval-freq", "0",
        "--curve-eval-freq", "0" if config["total_timesteps"] == 2048 else "25000",
        "--curve-eval-episodes", "2" if config["total_timesteps"] == 2048 else "20",
        "--seed", str(pair["seed"]),
        "--device", "cpu",
        "--output-dir", str(run),
        "--run-id", "training",
    ]
    if config.get("warm_start", False):
        command += [
            "--init-policy-path",
            str(root / "_inputs" / "safe_initialisation" / pair["id"] / "base_policy.pt"),
        ]
    return command


def verify_initial_policy(root: Path, pair: dict[str, Any], config: dict[str, Any]) -> None:
    """Verify the exact SB3 actor after the base-policy parameters are mapped."""
    if not config.get("warm_start", False):
        return

    import torch
    from stable_baselines3 import PPO

    from provably_safe_policy_optimisation import extract_feature_actor_parameters_and_network
    from provably_safe_policy_optimisation.safe_init import certify_with_verifier
    from projects.safe_policy_optimisation.stages.train_ppo_shield import make_unshielded_env
    from projects.safe_policy_optimisation.utils.warm_start import warm_start_actor

    init_path = root / "_inputs" / "safe_initialisation" / pair["id"] / "base_policy.pt"
    env = make_unshielded_env(
        config["env_id"],
        env_kwargs=config["env_kwargs"],
        max_episode_steps=config["max_episode_steps"],
        cost_limit=0.0,
        record_episodes=False,
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs={"net_arch": [config["hidden_dim"]] * config["n_hidden"]},
            seed=pair["seed"],
            device="cpu",
            verbose=0,
        )
        warm_start = warm_start_actor(
            model,
            init_path,
            hidden_dim=config["hidden_dim"],
            n_hidden=config["n_hidden"],
        )
        _parameters, actor = extract_feature_actor_parameters_and_network(model)
        dataset = torch.load(
            root / "_inputs" / "viewport_m_0.95.pt",
            map_location="cpu",
            weights_only=False,
        )
        x_l, x_u, mask = dataset.tensors
        fraction, passed = certify_with_verifier(
            actor, x_l, x_u, mask.bool(), method="IBP"
        )
    finally:
        env.close()
    report = {
        "m": 0.95,
        "method": "IBP",
        "certified_fraction": float(fraction),
        "passed": bool(passed),
        "base_policy_sha256": sha256(init_path),
        "warm_start": warm_start,
    }
    if not report["passed"] or report["certified_fraction"] != 1.0:
        raise RuntimeError("The exact warm-started SB3 actor did not pass m=0.95 verification")
    write_json(root / pair["id"] / "initial_verification.json", report)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)


def audit_policy(root: Path, pair: dict[str, Any], config: dict[str, Any]) -> None:
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from continuous_state_shields import (
        LunarLanderViewportShield,
        LunarLanderViewportShieldConfig,
    )
    from provably_safe_policy_optimisation import (
        extract_feature_actor_parameters_and_network,
    )
    from provably_safe_policy_optimisation.safe_init import certify_with_verifier
    from projects.safe_policy_optimisation.stages.train_ppo_shield import make_unshielded_env
    from projects.safe_policy_optimisation.archive.stages.train_pspo_continuous import (
        evaluate_viewport_safety,
    )

    run = root / pair["id"]
    model = PPO.load(run / "training/model.zip", device="cpu")
    _parameters, actor = extract_feature_actor_parameters_and_network(model)

    # Guard against certifying a network that is not exactly the SB3 greedy actor.
    rng = np.random.default_rng(pair["seed"] + 91_000)
    experiment = read_json(root / "experiment.json")
    low = np.asarray(experiment["observation_low"])
    high = np.asarray(experiment["observation_high"])
    probes = rng.uniform(low, high, size=(128, low.size)).astype(np.float32)
    with torch.no_grad():
        extracted_actions = actor(torch.as_tensor(probes)).argmax(dim=1).cpu().numpy()
    sb3_actions, _ = model.predict(probes, deterministic=True)
    if not np.array_equal(extracted_actions, np.asarray(sb3_actions).reshape(-1)):
        raise RuntimeError("Extracted actor does not reproduce SB3 deterministic actions")

    env_factory = lambda: make_unshielded_env(  # noqa: E731
        config["env_id"],
        env_kwargs=config["env_kwargs"],
        max_episode_steps=config["max_episode_steps"],
        cost_limit=0.0,
        record_episodes=False,
    )
    viewport = evaluate_viewport_safety(
        model,
        env_factory,
        LunarLanderViewportShield(LunarLanderViewportShieldConfig(m=MS[0])),
        apply_shield=False,
        episodes=config["eval_episodes"],
        seed=pair["seed"] + config["evaluation_seed_offset"],
    )
    viewport["out_of_view_rate"] = (
        viewport["out_of_view_episodes"] / viewport["episodes"]
        if viewport["episodes"] else 0.0
    )
    viewport["actor_equivalence_probe_count"] = len(probes)
    _write_rows(run / "viewport_audit_episodes.csv", viewport["per_episode"])
    write_json(run / "viewport_audit.json", viewport)

    thresholds: dict[str, Any] = {}
    for m in MS:
        dataset = torch.load(
            root / "_inputs" / f"viewport_m_{m:.2f}.pt",
            map_location="cpu",
            weights_only=False,
        )
        x_l, x_u, mask = (tensor.cpu() for tensor in dataset.tensors)
        fraction, passed = certify_with_verifier(actor, x_l, x_u, mask.bool(), method="IBP")
        sides = {}
        for index, side in enumerate(("left", "right")):
            side_fraction, side_passed = certify_with_verifier(
                actor,
                x_l[index:index + 1],
                x_u[index:index + 1],
                mask[index:index + 1].bool(),
                method="IBP",
            )
            sides[side] = {
                "certified_fraction": float(side_fraction),
                "passed": bool(side_passed),
            }
        thresholds[m_key(m)] = {
            "m": m,
            "certified_fraction": float(fraction),
            "passed": bool(passed),
            "left_band": sides["left"],
            "right_band": sides["right"],
        }
    verification = {
        "method": "IBP",
        "actor": "SB3 features extractor + policy MLP + action logits",
        "decision_rule": "greedy argmax",
        "pass_condition": "both complete 8-D shield boxes are certified",
        "thresholds": thresholds,
    }
    write_json(run / "viewport_verification.json", verification)


def validate_result(root: Path, pair: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    run = root / pair["id"]
    metrics = read_json(run / "training/metrics.json")
    summary = read_json(run / "training/summary.json")
    trained = read_json(run / "training/config.json")
    viewport = read_json(run / "viewport_audit.json")
    verification = read_json(run / "viewport_verification.json")

    expected_steps = math.ceil(config["total_timesteps"] / config["n_steps"]) * config["n_steps"]
    if summary["final_timesteps"] != expected_steps or summary["early_stop_triggered"]:
        raise RuntimeError("Vanilla PPO did not finish its fixed training budget")
    if trained["algorithm"] != "plain_ppo":
        raise RuntimeError("Run was not vanilla PPO")
    if config.get("warm_start", False):
        initial = read_json(run / "initial_verification.json")
        expected_init = root / "_inputs" / "safe_initialisation" / pair["id"] / "base_policy.pt"
        if not initial["passed"] or initial["certified_fraction"] != 1.0:
            raise RuntimeError("Warm-started actor was not initially certified")
        if trained["warm_start"] is None:
            raise RuntimeError("Training did not record its requested warm start")
        if Path(trained["warm_start"]["init_policy_path"]).resolve() != expected_init.resolve():
            raise RuntimeError("Training used the wrong warm-start parameter vector")
        if initial["base_policy_sha256"] != sha256(expected_init):
            raise RuntimeError("Initial verification hash differs from the trained warm start")
    elif trained["warm_start"] is not None:
        raise RuntimeError("Cold-start control unexpectedly used a warm start")
    if trained["shield_path"] is not None or trained["env_kwargs"] != config["env_kwargs"]:
        raise RuntimeError("Training unexpectedly used a shield or wrong environment")
    if metrics["eval_episodes"] != config["eval_episodes"] or viewport["episodes"] != config["eval_episodes"]:
        raise RuntimeError("Incomplete final evaluation")
    if viewport["out_of_view_episodes"] + round(viewport["safe_trajectory_rate"] * viewport["episodes"]) != viewport["episodes"]:
        raise RuntimeError("Viewport exit and safety counts disagree")

    with (run / "training/episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    audited = viewport["per_episode"]
    if len(episodes) != len(audited):
        raise RuntimeError("Training and independent viewport evaluations differ in size")
    for row, audit in zip(episodes, audited):
        if int(row["length"]) != audit["length"] or not math.isclose(
            float(row["reward"]), audit["reward"], abs_tol=1e-5
        ):
            raise RuntimeError("Independent replay does not reproduce final PPO evaluation")

    expected_keys = {m_key(m) for m in MS}
    if verification["method"] != "IBP" or set(verification["thresholds"]) != expected_keys:
        raise RuntimeError("Viewport verification is incomplete")
    result = {
        **pair,
        "mean_total_reward": float(metrics["reward"]["mean_total_reward"]),
        "out_of_view_episodes": int(viewport["out_of_view_episodes"]),
        "out_of_view_rate": float(viewport["out_of_view_rate"]),
        "safety_rate": float(viewport["safe_trajectory_rate"]),
        "no_truncation_rate": float(viewport["no_truncation_rate"]),
        "successful_landing_rate": float(viewport["successful_landing_rate"]),
        "crash_rate": float(viewport["crash_rate"]),
        "eval_episodes": int(metrics["eval_episodes"]),
        "final_timesteps": int(summary["final_timesteps"]),
        "verification_pass": {
            key: bool(verification["thresholds"][key]["passed"])
            for key in sorted(expected_keys, reverse=True)
        },
        "verification_certified_fraction": {
            key: float(verification["thresholds"][key]["certified_fraction"])
            for key in sorted(expected_keys, reverse=True)
        },
    }
    if not all(math.isfinite(result[key]) for key in (
        "mean_total_reward", "out_of_view_rate", "safety_rate",
        "no_truncation_rate", "successful_landing_rate", "crash_rate",
    )):
        raise RuntimeError("Non-finite audit result")
    return result


def worker(root: Path, pair_id: str) -> None:
    record = read_json(root / "experiment.json")
    pair = next(pair for pair in record["pairs"] if pair["id"] == pair_id)
    run = root / pair_id
    run.mkdir(parents=True, exist_ok=False)
    status = {
        **pair,
        "status": "running",
        "phase": "preflight",
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "started_utc": utc_now(),
    }
    write_json(run / "status.json", status)
    try:
        if len(status["cpu_affinity"]) != 1:
            raise RuntimeError("Worker must be pinned to exactly one CPU")
        scheduler.verify_provenance(root, record)
        if record["settings"].get("warm_start", False):
            status.update({"phase": "initial_verification"})
            write_json(run / "status.json", status)
            print(f"[{utc_now()}] {pair_id}: verify warm-start actor", flush=True)
            verify_initial_policy(root, pair, record["settings"])
        command = build_training_command(root, pair, record["settings"])
        write_json(run / "commands.json", {"training": command, "audit": "in-process after training"})
        status.update({"phase": "training"})
        write_json(run / "status.json", status)
        print(f"[{utc_now()}] {pair_id}: training", flush=True)
        process = subprocess.Popen(command, cwd=REPO)
        status["stage_pid"] = process.pid
        write_json(run / "status.json", status)
        code = process.wait()
        if code:
            raise RuntimeError(f"training exited {code}")

        status.update({"phase": "viewport_audit", "stage_pid": os.getpid()})
        write_json(run / "status.json", status)
        print(f"[{utc_now()}] {pair_id}: viewport audit and verification", flush=True)
        audit_policy(root, pair, record["settings"])
        result = validate_result(root, pair, record["settings"])
        write_json(run / "result.json", result)
        status.update({"status": "complete", "phase": "complete", "completed_utc": utc_now()})
        write_json(run / "status.json", status)
    except BaseException as error:
        status.update({"status": "failed", "error": str(error), "completed_utc": utc_now()})
        write_json(run / "status.json", status)
        traceback.print_exc()
        raise


def _mean_2se(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return None, None
    return statistics.fmean(values), (
        2 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    )


def _display(mean: float | None, error: float | None, scale: float = 1.0) -> str:
    if mean is None:
        return "pending"
    text = f"{scale * mean:.2f}"
    return text + (f" ± {scale * error:.2f}" if error is not None else " (one seed)")


def write_report(root: Path) -> dict[str, Any]:
    record = read_json(root / "experiment.json")
    rows, statuses = [], {}
    for pair in record["pairs"]:
        status_path = root / pair["id"] / "status.json"
        status = read_json(status_path)["status"] if status_path.exists() else "queued"
        statuses[pair["id"]] = status
        if status == "complete":
            result = validate_result(root, pair, record["settings"])
            if result != read_json(root / pair["id"] / "result.json"):
                raise RuntimeError(f"Completed result changed: {pair['id']}")
            rows.append(result)

    csv_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for m in MS:
            key = m_key(m)
            flat[f"verification_m_{key}"] = row["verification_pass"][key]
            flat[f"certified_fraction_m_{key}"] = row["verification_certified_fraction"][key]
        csv_rows.append(flat)
    fields = list(csv_rows[0]) if csv_rows else [
        "id", "seed", "mean_total_reward", "out_of_view_episodes", "out_of_view_rate", "safety_rate"
    ]
    temporary = root / "per_seed.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    temporary.replace(root / "per_seed.csv")

    reward, reward_2se = _mean_2se(rows, "mean_total_reward")
    exit_rate, exit_2se = _mean_2se(rows, "out_of_view_rate")
    safe_rate, safe_2se = _mean_2se(rows, "safety_rate")
    total_episodes = sum(row["eval_episodes"] for row in rows)
    total_exits = sum(row["out_of_view_episodes"] for row in rows)
    verification = []
    for m in MS:
        key = m_key(m)
        verification.append({
            "m": m,
            "passed_seeds": sum(row["verification_pass"][key] for row in rows),
            "completed_seeds": len(rows),
            "mean_certified_fraction": (
                statistics.fmean(row["verification_certified_fraction"][key] for row in rows)
                if rows else None
            ),
        })
    aggregate = {
        "updated_utc": utc_now(),
        "completed_pairs": len(rows),
        "requested_pairs": len(statuses),
        "failed_pairs": [key for key, status in statuses.items() if status == "failed"],
        "statuses": statuses,
        "mean_total_reward": reward,
        "mean_total_reward_2se": reward_2se,
        "mean_out_of_view_rate": exit_rate,
        "mean_out_of_view_rate_2se": exit_2se,
        "mean_safety_rate": safe_rate,
        "mean_safety_rate_2se": safe_2se,
        "pooled_out_of_view_episodes": total_exits,
        "pooled_evaluation_episodes": total_episodes,
        "initial_m_0.95_verified_seeds": (
            len(rows) if record["settings"].get("warm_start", False) else 0
        ),
        "verification": verification,
    }
    warm_start = bool(record["settings"].get("warm_start", False))
    lines = [
        "# Safe-initialised vanilla PPO LunarLander viewport audit" if warm_start
        else "# Vanilla PPO LunarLander viewport audit",
        "",
        f"Completed: {len(rows)}/{len(statuses)}; failed: {len(aggregate['failed_pairs'])}.",
        "",
        (
            "Each actor starts from its matching seed's IBP-certified m=0.95 parameter vector; "
            "the critic and optimizer start fresh. PPO remains unconstrained after initialisation."
            if warm_start else
            "Policies use independent standard PPO initialisations, no shield, and no safe warm start."
        ),
        "Final evaluation is deterministic and unshielded; production uses 100 episodes/seed.",
        "A viewport exit means abs(raw normalised Box2D x) >= 1 at any state, including the terminal state.",
        "Uncertainty is ±2 SE across completed seeds. Verification is sound IBP over both full 8-D bands.",
        "",
        "| Seeds | Total reward | View-exit rate (%) | Viewport-safe rate (%) | Pooled exits |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {len(rows)}/{len(statuses)} | {_display(reward, reward_2se)} | "
        f"{_display(exit_rate, exit_2se, 100)} | {_display(safe_rate, safe_2se, 100)} | "
        f"{total_exits}/{total_episodes} |",
        "",
        "| Shield threshold m | Policies passing verification | Mean certified boxes (%) |",
        "| ---: | ---: | ---: |",
    ]
    for item in verification:
        fraction = "pending" if item["mean_certified_fraction"] is None else f"{100 * item['mean_certified_fraction']:.2f}"
        lines.append(
            f"| {item['m']:.2f} | {item['passed_seeds']}/{item['completed_seeds']} | {fraction} |"
        )
    if aggregate["failed_pairs"]:
        lines += ["", "Failed pairs: " + ", ".join(aggregate["failed_pairs"])]
    write_json(root / "aggregate.json", aggregate)
    (root / "report.md").write_text("\n".join(lines) + "\n")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-policy-root",
        type=Path,
        default=None,
        help=(
            "Optional m=0.95 PSPO directory containing seed_N/initialisation/base_policy.pt. "
            "The matching certified actor warm-starts each PPO seed."
        ),
    )
    parser.add_argument("--min-idle", type=float, default=95)
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.min_idle <= 100 or not 1 <= args.max_concurrent <= 10:
        parser.error("--min-idle must be in (0,100]; --max-concurrent must be in 1..10")
    if args.output_root is None:
        if not args.launch:
            parser.error("--output-root is required except for a fresh launch")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        arm = "safe_initialised_" if args.init_policy_root is not None else ""
        suffix = f"_lunarlander_{arm}vanilla_ppo_viewport_audit" + ("_smoke" if args.smoke else "")
        args.output_root = REPO / "outputs/continuous_state_shields/ppo" / (stamp + suffix)
    root = args.output_root.resolve()
    if args.worker:
        worker(root, args.worker)
    elif args.controller:
        scheduler.controller(
            root,
            args.min_idle,
            args.max_concurrent,
            entrypoint=Path(__file__).resolve(),
            report_fn=write_report,
        )
    elif args.report_only:
        write_report(root)
    else:
        record = (
            read_json(root / "experiment.json")
            if args.resume
            else prepare(root, args.smoke, args.init_policy_root)
        )
        scheduler.verify_provenance(root, record)
        screens = subprocess.run(["screen", "-ls"], text=True, capture_output=True).stdout
        if f".{record['screen_name']}\t" in screens:
            raise RuntimeError("This experiment's screen session is already running")
        subprocess.run([
            "screen", "-L", "-Logfile", str(root / "_logs/screen.log"),
            "-dmS", record["screen_name"],
            str(PYTHON), "-u", str(Path(__file__).resolve()),
            "--controller", "--output-root", str(root),
            "--min-idle", str(args.min_idle),
            "--max-concurrent", str(args.max_concurrent),
        ], cwd=REPO, check=True)
        print(f"Detached screen: {record['screen_name']}\nOutput root: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
