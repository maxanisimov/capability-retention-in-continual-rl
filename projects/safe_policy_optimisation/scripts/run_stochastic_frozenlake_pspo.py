#!/usr/bin/env python3
"""Prepare, validate and launch ten PSPO seeds on structured slippery FrozenLake.

The detached screen controller pins concurrent workers to distinct idle physical
cores. Each worker invokes the existing PSPO training stage; no alternate RL
implementation is used. Inputs and exact safety/reachability proofs are saved.
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

for thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-matplotlib")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
    SparseFrozenLake,
    proper_goal_policies,
    structured_layout,
    synthesise_shield,
)

DEFAULT_SIZE = 128
RUNS_ROOT = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
SEEDS = tuple(range(10))
INITIALISATION_SCHEME = "safety_only_shield_mask_v1"


def default_output(size: int, state_representation: str = "one_hot") -> Path:
    suffix = "_state_id_lookup" if state_representation == "state_id_lookup" else ""
    return RUNS_ROOT / f"pspo_stochastic_frozenlake{size}_safety_only{suffix}"


def screen_name(size: int, state_representation: str = "one_hot") -> str:
    suffix = "-ids" if state_representation == "state_id_lookup" else ""
    return f"pspo-frozenlake{size}-safety-only{suffix}"


def episode_cap(size: int) -> int:
    """Scale the horizon with goal distance, which grows linearly in size.

    The witness preflight demands 100/100 goal-reaching episodes, and the
    start-to-goal Manhattan distance is 2*(size-1); a fixed cap would make the
    larger layouts unreachable rather than merely harder.
    """
    return 5000 * size // DEFAULT_SIZE


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


def build_safe_actor(
    mask: np.ndarray,
    *,
    state_representation: str = "one_hot",
) -> tuple[dict, dict]:
    """Encode a reward-agnostic safe actor from the shield mask alone.

    Every safe action receives the same logit (2) and every unsafe action receives
    the same logit (-2).  No goal, reward, witness policy, preferred action, or
    distance-to-goal information is accepted by this function.  The one-hot
    first-layer columns are analytically inverted through both tanh layers. Unused
    hidden units have random nonzero activations, allowing PPO to recruit them
    despite their zero initial output weights.
    """
    generator = torch.Generator().manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Linear(len(mask), 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4),
    )
    logits = np.where(mask, 2.0, -2.0)
    safe_ids = np.flatnonzero(mask.any(axis=1))
    encoded = np.arctanh(np.arctanh(logits / 4.0))
    with torch.no_grad():
        network[0].weight.copy_(torch.randn(network[0].weight.shape, generator=generator) * 0.02)
        network[0].weight[:4].copy_(torch.tensor(encoded.T, dtype=torch.float32))
        network[0].bias.zero_()
        network[2].weight.copy_(torch.eye(64))
        network[2].bias.zero_()
        network[4].weight.zero_()
        network[4].weight[:, :4].copy_(4 * torch.eye(4))
        network[4].bias.zero_()
        # Exact evaluation without materialising a 16384-by-16384 identity.
        actual = network[4](network[3](network[2](network[1](
            network[0].weight.T + network[0].bias
        )))).numpy()
    greedy = actual.argmax(axis=1)
    if not np.all(mask[safe_ids, greedy[safe_ids]]):
        raise AssertionError("Analytic safe actor failed greedy audit.")
    if not np.allclose(actual, logits, atol=1e-5):
        raise AssertionError("Analytic logit encoding failed.")
    architecture = {
        "activation": "Tanh", "hidden_dim": 64, "input_dim": len(mask),
        "n_actions": 4, "n_hidden": 2,
        "state_representation": (
            "state_id_lookup_discrete_observation"
            if state_representation == "state_id_lookup"
            else "one_hot_discrete_observation"
        ),
    }
    return {"architecture": architecture, "state_dict": network.state_dict()}, {
        "initialisation": INITIALISATION_SCHEME,
        "initialisation_inputs": ["shield_action_mask"],
        "reward_information_used": False,
        "goal_information_used": False,
        "witness_policy_used": False,
        "reward_training_used": False, "all_winning_state_greedy_safety": 1.0,
        "safe_logit": 2.0, "unsafe_logit": -2.0,
        "safe_action_logits_equal": True,
        "minimum_all_safe_vs_unsafe_logit_margin": 4.0,
        "same_actor_for_all_training_seeds": True,
    }


def evaluate_witnesses(env: SparseFrozenLake, policies: list[np.ndarray], horizon: int) -> list[dict]:
    results = []
    for index, policy in enumerate(policies):
        rewards, lengths, successes, unsafe = [], [], 0, 0
        for episode in range(100):
            state, _ = env.reset(seed=50_000 + episode)
            total = 0.0
            for step in range(horizon):
                state, reward, terminated, _truncated, info = env.step(int(policy[state]))
                total += reward
                unsafe += int(info["cost"] > 0)
                if terminated:
                    successes += int(info["success"])
                    break
            rewards.append(total)
            lengths.append(step + 1)
        results.append({
            "policy": index, "episodes": 100, "evaluation_seed_start": 50_000,
            "mean_total_reward": statistics.fmean(rewards),
            "mean_episode_length": statistics.fmean(lengths),
            "goal_success_rate": successes / 100, "unsafe_visits": unsafe,
        })
        if unsafe or successes != 100:
            raise ValueError(f"Witness failed the finite-horizon preflight: {results[-1]}")
    return results


def source_paths() -> list[Path]:
    paths = {Path(__file__).resolve(), REPO / "projects/safe_policy_optimisation/utils/frozen_lake_experiment.py"}
    for directory in (
        "core/provably_safe_policy_optimisation", "core/src",
        "projects/safe_policy_optimisation/stages", "projects/safe_policy_optimisation/utils",
    ):
        paths.update((REPO / directory).rglob("*.py"))
    return sorted(paths)


def refresh_source_record(root: Path, record: dict) -> None:
    paths = source_paths()
    record["source_sha256"] = {str(p.relative_to(REPO)): sha256(p) for p in paths}
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(REPO))
    write_json(root / "experiment.json", record)


def prepare(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    root = args.output_root
    inputs = root / "_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (root / "_logs").mkdir(exist_ok=True)
    experiment_path = root / "experiment.json"
    expected = {
        "size": args.size, "seeds": list(SEEDS), "success_rate": 0.8,
        "perpendicular_slip_probability_each": 0.1, "step_penalty": 0.001,
        "max_episode_steps": episode_cap(args.size),
        "total_timesteps": args.total_timesteps,
        "state_representation": args.state_representation,
        "initialisation": INITIALISATION_SCHEME,
        "pspo_variant": "adaptive directional orthotope, region-first, every 100 rollouts",
        "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.999,
        "learning_rate": 0.0003, "rashomon_n_iters": 200,
        "rashomon_batch_size": 256, "certificate": "exhaustive over all safety-winning states",
        "evaluation_episodes": 100, "evaluation_policy": "nominal greedy, no runtime shield",
    }
    if experiment_path.exists():
        record = read_json(experiment_path)
        if record["settings"] != expected:
            raise ValueError("Existing experiment settings differ; use another output root.")
        for name, digest in record["input_sha256"].items():
            if sha256(inputs / name) != digest:
                raise ValueError(f"Input provenance mismatch: {name}")
        current_sources = {str(p.relative_to(REPO)): sha256(p) for p in source_paths()}
        if current_sources != record["source_sha256"]:
            if list(root.glob("seed*/status.json")):
                raise ValueError("Production runs already exist; do not change their source provenance.")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            experiment_path.rename(root / f"experiment_preflight_{stamp}.json")
            (root / "source_snapshot.zip").rename(root / f"source_snapshot_preflight_{stamp}.zip")
            refresh_source_record(root, record)
        return record
    layout_path = inputs / "layout.txt"
    layout_path.write_text("\n".join(structured_layout(args.size)) + "\n")
    env = SparseFrozenLake(layout_path=str(layout_path))
    mask, winning, successors, probabilities = synthesise_shield(env)
    # Construct the actor before computing any goal-aware diagnostics.  Its API
    # accepts only the shield action mask, making reward/goal leakage impossible.
    actor, actor_analysis = build_safe_actor(
        mask,
        state_representation=args.state_representation,
    )
    goal = env._n_states - 1
    if not winning[0]:
        raise ValueError("Initial state is not safety-winning.")
    distances, policies, proof = proper_goal_policies(mask, winning, successors, probabilities, goal)
    empirical = evaluate_witnesses(env, policies, expected["max_episode_steps"])
    actor.update({"environment_id": ENV_ID, "safe_initialisation": actor_analysis})
    torch.save(actor, inputs / "base_policy.pt")
    torch.save({
        "shield": torch.tensor(mask), "winning_states": torch.tensor(winning),
        "env_id": ENV_ID, "layout_sha256": sha256(layout_path),
        "risk_threshold": 0.0, "safety_requirement": "never enter a hole",
        "synthesis": "greatest fixed point over all positive-probability successors",
    }, inputs / "shield_q.pt")
    np.savez_compressed(inputs / "witness_policies.npz", policies=np.stack(policies), distances=distances)
    validation = {
        "nominal_state_count": env._n_states, "action_count": env._n_actions,
        "hole_states": int(np.count_nonzero(env.desc == b"H")),
        "safety_winning_states": int(winning.sum()), "safe_state_action_pairs": int(mask.sum()),
        "states_with_multiple_safe_actions": int(np.count_nonzero(mask.sum(axis=1) >= 2)),
        "start_safety_winning": True, "goal_reachability": proof,
        "witness_evaluation": empirical, "witnesses_used_for_initialisation": False,
        "initial_actor": actor_analysis,
        "reward_definition": "1 on goal minus 0.001 per step; holes are safety violations",
    }
    write_json(inputs / "layout_validation.json", validation)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.imshow(env.desc == b"H", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.scatter([0], [0], c="green", s=30, label="Start")
    ax.scatter([args.size - 1], [args.size - 1], c="red", marker="*", s=60, label="Goal")
    ax.set_title(f"{args.size}×{args.size} slippery FrozenLake — fixed layout")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right")
    fig.savefig(inputs / "layout.png", dpi=200)
    plt.close(fig)
    files = ["layout.txt", "base_policy.pt", "shield_q.pt", "layout_validation.json", "witness_policies.npz"]
    record = {
        "created_utc": utc_now(), "settings": expected, "env_id": ENV_ID,
        "input_sha256": {name: sha256(inputs / name) for name in files},
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in source_paths()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
    }
    refresh_source_record(root, record)
    env.close()
    print(json.dumps(validation, indent=2), flush=True)
    return record


def training_arguments(root: Path, seed: int, *, smoke: bool = False) -> list[str]:
    record = read_json(root / "experiment.json")
    settings = record["settings"]
    inputs = root / "_inputs"
    return [
        "--env-id", ENV_ID, "--env-kwargs", json.dumps({
            "layout_path": str(inputs / "layout.txt"), "is_slippery": True,
            "success_rate": settings["success_rate"], "step_penalty": settings["step_penalty"],
        }),
        "--base-policy-path", str(inputs / "base_policy.pt"),
        "--shield-path", str(inputs / "shield_q.pt"),
        "--state-representation", settings["state_representation"],
        "--max-episode-steps", str(settings["max_episode_steps"]), "--cost-limit", "0",
        "--total-timesteps", str(2048 if smoke else settings["total_timesteps"]),
        "--seed", str(seed), "--device", "cpu", "--evaluation-policy", "unshielded",
        "--learning-rate", "0.0003", "--n-steps", "2048", "--batch-size", "64",
        "--n-epochs", "1" if smoke else "10", "--gamma", "0.999", "--gae-lambda", "0.95",
        "--clip-range", "0.2", "--ent-coef", "0", "--vf-coef", "0.5", "--max-grad-norm", "0.5",
        "--freq", "100", "--verify-first", "false", "--region-refresh", "adaptive",
        "--directional", "true", "--region-mode", "replace", "--unsafe-update-strategy", "rashomon_project",
        "--n-iters", "2" if smoke else "200", "--rashomon-checkpoint", "2" if smoke else "100",
        "--rashomon-batch-size", "256", "--rashomon-inverse-temp", "1",
        "--rashomon-multi-label-mode", "all", "--surrogate", "logsumexp",
        "--rashomon-objective", "weighted_width", "--safe-region-shape", "orthotope",
        "--eval-episodes", "2" if smoke else "100", "--early-stop-eval-freq", "0",
        "--curve-eval-freq", "0" if smoke else "100000", "--curve-eval-episodes", "2" if smoke else "10",
        "--output-dir", str(root / "_smoke" if smoke else root), "--run-id", f"seed{seed}",
    ]


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    root = args.output_root
    run_dir = (root / "_smoke" if args.smoke else root) / f"seed{args.worker}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "complete":
        print(f"Already complete: {run_dir}", flush=True)
        return
    record = read_json(root / "experiment.json")
    for name, digest in record["input_sha256"].items():
        if sha256(root / "_inputs" / name) != digest:
            raise ValueError(f"Changed input: {name}")
    for relative, digest in record["source_sha256"].items():
        if sha256(REPO / relative) != digest:
            raise ValueError(f"Source changed since preparation: {relative}")
    status = {
        "status": "running", "seed": args.worker, "smoke": args.smoke,
        "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "started_utc": utc_now(),
    }
    write_json(status_path, status)
    try:
        from projects.safe_policy_optimisation.stages import train_pspo
        summary = train_pspo.run(train_pspo.parse_args(training_arguments(root, args.worker, smoke=args.smoke)))
        if float(summary["final_exact_all_state_alignment"]) < 1.0:
            raise RuntimeError("Final nominal actor failed exhaustive safety audit.")
        # The shared stage defines success by positive return. With step costs,
        # a slow goal-reaching episode may have negative return. Recover the
        # exact goal indicator from return + penalty * length instead.
        with (run_dir / "episodes.csv").open() as handle:
            episodes = list(csv.DictReader(handle))
        penalty = record["settings"]["step_penalty"]
        successes = sum(goal_indicator(float(row["reward"]), int(row["length"]), penalty) for row in episodes)
        metrics_path = run_dir / "metrics.json"
        metrics = read_json(metrics_path)
        write_json(run_dir / "metrics_reward_threshold.json", metrics)
        metrics["success"] = {
            "success_mode": "goal_reached", "success_count": successes,
            "success_rate": successes / len(episodes),
            "definition": "goal indicator recovered exactly from return plus step penalty times length",
        }
        write_json(metrics_path, metrics)
        status.update({"status": "complete", "completed_utc": utc_now(), "summary_path": str(run_dir / "summary.json")})
        write_json(status_path, status)
        print(f"COMPLETED seed={args.worker}: {summary}", flush=True)
    except Exception:
        status.update({"status": "failed", "failed_utc": utc_now(), "traceback": traceback.format_exc()})
        write_json(status_path, status)
        raise


def goal_indicator(reward: float, length: int, penalty: float) -> int:
    indicator = reward + penalty * length
    if not (math.isclose(indicator, 0, abs_tol=1e-7) or math.isclose(indicator, 1, abs_tol=1e-7)):
        raise ValueError(f"Unexpected episode reward semantics: {indicator}")
    return int(round(indicator))


def free_physical_cpus(min_idle: float = 95.0) -> tuple[list[int], dict]:
    """Admit physical cores only when every SMT sibling averaged >=min_idle idle.

    The default is deliberately strict. A shared machine carrying light load
    spread across every core can leave nothing at all above 95%, in which case
    the caller may lower the bar rather than run a ten-seed sweep serially.
    """
    sample = json.loads(subprocess.check_output(
        ["mpstat", "-o", "JSON", "-P", "ALL", "1", "5"], text=True,
    ))
    observations: dict[int, list[float]] = {}
    for frame in sample["sysstat"]["hosts"][0]["statistics"]:
        for cpu in frame["cpu-load"]:
            if cpu["cpu"] != "all":
                observations.setdefault(int(cpu["cpu"]), []).append(float(cpu["idle"]))
    idle = {cpu: statistics.fmean(values) for cpu, values in observations.items()}
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True)
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
        "mean_idle_percent": idle, "physical_core_siblings": list(physical.values()),
        "min_idle_percent_required": min_idle,
    }


def write_report(root: Path) -> None:
    rows = []
    for seed in SEEDS:
        run = root / f"seed{seed}"
        if not (run / "status.json").exists() or read_json(run / "status.json")["status"] != "complete":
            continue
        metrics = read_json(run / "metrics.json")
        rows.append({
            "seed": seed, "mean_total_reward": metrics["reward"]["mean_total_reward"],
            "safety_rate": metrics["safety"]["safety_rate"],
            "success_rate": metrics["success"]["success_rate"],
        })
    with (root / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "mean_total_reward", "safety_rate", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"completed_seeds": len(rows), "requested_seeds": len(SEEDS), "updated_utc": utc_now()}
    if rows:
        for key in ("mean_total_reward", "safety_rate", "success_rate"):
            values = [float(row[key]) for row in rows]
            summary[key] = statistics.fmean(values)
            summary[key + "_2se"] = 2 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    write_json(root / "aggregate.json", summary)


def controller(args: argparse.Namespace) -> None:
    root = args.output_root
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus, cpu_evidence = free_physical_cpus(args.min_idle)
    pending = [seed for seed in SEEDS if not (root / f"seed{seed}" / "status.json").exists()
               or read_json(root / f"seed{seed}" / "status.json").get("status") != "complete"]
    if not cpus and pending:
        raise RuntimeError("No sufficiently idle physical cores; no experiments launched.")
    # Exhaustive one-hot certification is memory-bound well before it is
    # core-bound: peak RSS per worker is ~8 GiB at size 128 but ~100 GiB at 256.
    # Admitting one worker per idle core would exhaust a shared machine, so the
    # concurrency cap is the binding constraint above size 128.
    available = cpus[:min(len(pending), args.max_concurrent)]
    launch = {"started_utc": utc_now(), "screen_name": screen_name(args.size, args.state_representation),
              "max_concurrent": args.max_concurrent, "queued_seeds": list(pending),
              "admitted_cpus": list(available), "cpu_evidence": cpu_evidence, "jobs": []}
    write_json(root / "launch_manifest.json", launch)
    write_report(root)
    active = {}
    failures = []
    while pending or active:
        while pending and available:
            seed, cpu = pending.pop(0), available.pop(0)
            log_path = root / "_logs" / f"seed{seed}.log"
            command = ["taskset", "-c", str(cpu), str(REPO / ".venv/bin/python"), "-u",
                       str(Path(__file__).resolve()), "--worker", str(seed), "--output-root", str(root),
                       "--state-representation", args.state_representation]
            log = log_path.open("a")
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            active[seed] = (process, cpu, log)
            launch["jobs"].append({"seed": seed, "cpu": cpu, "pid": process.pid,
                                   "command": command, "log_path": str(log_path), "started_utc": utc_now()})
            write_json(root / "launch_manifest.json", launch)
            print(f"[{utc_now()}] launched seed{seed} pid={process.pid} cpu={cpu}", flush=True)
        for seed, (process, cpu, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[seed]
            available.append(cpu)
            if code:
                failures.append(seed)
            print(f"[{utc_now()}] seed{seed} exit={code}", flush=True)
            write_report(root)
        if active:
            time.sleep(5)
    write_report(root)
    launch.update({"completed_utc": utc_now(), "failed_seeds": failures})
    write_json(root / "launch_manifest.json", launch)
    if failures:
        raise RuntimeError(f"Failed seeds: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--max-concurrent", type=int, default=len(SEEDS),
        help="Workers to run at once. Memory-bound above size 128; see controller().",
    )
    parser.add_argument(
        "--min-idle", type=float, default=95.0,
        help="Minimum mean idle %% required of every SMT sibling to admit a core.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument(
        "--state-representation",
        choices=("one_hot", "state_id_lookup"),
        default="one_hot",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--worker", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1):
        parser.error("--size must be a power of two and at least 16")
    if not 0 < args.min_idle <= 100:
        parser.error("--min-idle must be in (0, 100]")
    if not 1 <= args.max_concurrent <= len(SEEDS):
        parser.error(f"--max-concurrent must be in 1..{len(SEEDS)}")
    if args.output_root is None:
        args.output_root = default_output(args.size, args.state_representation)
    args.output_root = args.output_root.resolve()
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.worker is not None:
        worker(args)
    elif args.controller:
        controller(args)
    elif args.report_only:
        write_report(args.output_root)
    elif args.launch:
        prepare(args)
        subprocess.run([
            "screen", "-L", "-Logfile", str(args.output_root / "_logs" / "screen.log"),
            "-dmS", screen_name(args.size, args.state_representation), str(REPO / ".venv/bin/python"), "-u",
            str(Path(__file__).resolve()), "--controller", "--size", str(args.size),
            "--max-concurrent", str(args.max_concurrent),
            "--min-idle", str(args.min_idle),
            "--output-root", str(args.output_root),
            "--state-representation", args.state_representation,
        ], cwd=REPO, check=True)
        print(
            "Requested detached screen session: "
            f"{screen_name(args.size, args.state_representation)}"
        )
    elif args.prepare:
        prepare(args)
    else:
        parser.error("Choose --prepare, --launch, --worker, --controller, or --report-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
