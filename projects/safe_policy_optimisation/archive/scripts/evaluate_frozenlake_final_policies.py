#!/usr/bin/env python3
"""Measure PSPO, PPO-Shield and CPO by empirical rollouts of the final policy.

Every method's stored metrics already come from final-policy rollouts, but they
are produced by three different stage code paths. This script re-measures all
three through one harness so the comparison cannot turn on an evaluation
detail: same unshielded environment, same deterministic action rule, same
episode seeds, same success definition.

Using one shared list of episode seeds also makes the comparison paired - every
policy meets the same slip realisations - so differences between methods are
not confounded by evaluation noise.

PPO-Shield is measured twice. Bare, it answers whether the learned policy is
safe by itself; with its runtime shield attached, it answers how the method
performs as actually deployed. PSPO is measured bare only, since its claim is
that no runtime shield is needed, and CPO has no shield to attach.
"""

from __future__ import annotations

import argparse
import csv
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
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-matplotlib")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from projects.safe_policy_optimisation.utils.frozen_lake_experiment import ENV_ID  # noqa: E402

RUNS_ROOT = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
SEEDS = tuple(range(10))
# ``ppo_shield_shielded`` reuses the ppo_shield policy but keeps the runtime
# shield attached, which is how that method is actually deployed. PSPO is only
# measured bare: its claim is that the learned policy needs no runtime shield.
METHODS = ("pspo", "ppo_shield", "ppo_shield_shielded", "cpo")
SHIELDED_METHODS = ("ppo_shield_shielded",)
# Fixed and shared by every (method, seed): episode e of every rollout set uses
# EVAL_SEED_BASE + e, so all policies face identical slip realisations.
EVAL_SEED_BASE = 1_000_000
# The shield picks among safe actions when it overrides, so it carries its own
# randomness. Fixing it across seeds keeps the policy the only thing that varies.
SHIELD_SEED = 0
DEFAULT_EPISODES = 200


def default_output(size: int) -> Path:
    return RUNS_ROOT / f"final_policy_rollouts_frozenlake{size}"


def pspo_root(size: int) -> Path:
    return RUNS_ROOT / f"pspo_stochastic_frozenlake{size}"


def baselines_root(size: int) -> Path:
    return RUNS_ROOT / f"baselines_stochastic_frozenlake{size}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def model_path(method: str, seed: int, size: int) -> Path:
    if method == "pspo":
        return pspo_root(size) / f"seed{seed}" / "model.zip"
    if method in ("ppo_shield", "ppo_shield_shielded"):
        return baselines_root(size) / "ppo_shield" / f"seed{seed}" / "model.zip"
    return baselines_root(size) / "cpo" / f"seed{seed}" / "cpo.pt"


def make_shield(size: int, env: gym.Env):
    """The same tabular shield the PSPO run was built and trained against."""

    from provably_safe_policy_optimisation import Shield

    from projects.safe_policy_optimisation.utils.shield import load_shield_mask

    mask = load_shield_mask(
        pspo_root(size) / "_inputs" / "shield_q.pt",
        shield_key="shield", source="shield", risk_threshold=None,
    )
    return Shield(mask, obs_to_state=env.unwrapped.make_obs_to_state(), seed=SHIELD_SEED)


def make_env(size: int, settings: dict) -> gym.Env:
    """The plain environment: no shield wrapper, no episode recorder."""

    return gym.make(
        ENV_ID,
        max_episode_steps=int(settings["max_episode_steps"]),
        layout_path=str(pspo_root(size) / "_inputs" / "layout.txt"),
        is_slippery=True,
        success_rate=settings["success_rate"],
        step_penalty=settings["step_penalty"],
    )


def load_model(method: str, seed: int, size: int, env: gym.Env):
    path = model_path(method, seed, size)
    if not path.exists():
        raise FileNotFoundError(f"No final policy for {method} seed{seed}: {path}")
    if method == "cpo":
        from projects.safe_policy_optimisation.utils.safe_rl import load_checkpoint_model
        model, _checkpoint = load_checkpoint_model(path, env=env, device="cpu")
        return model
    # Loading without a shield is what we want: ProvablySafePPO then applies no
    # shielding, so PSPO and PPO-Shield are both measured bare.
    from provably_safe_policy_optimisation import ProvablySafePPO
    return ProvablySafePPO.load(path, device="cpu")


def rollout(
    model, env: gym.Env, *, episodes: int, step_penalty: float, shield=None
) -> list[dict]:
    """Deterministic rollouts, recording reward, hole visits and goal arrival.

    With ``shield`` attached the policy's action is treated as a proposal and the
    shield may override it, which is how a shielded method runs in deployment.
    """

    rows = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=EVAL_SEED_BASE + episode)
        done = False
        total_reward = 0.0
        unsafe_visits = 0
        length = 0
        overrides = 0
        reached_goal = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            proposed = int(np.asarray(action).item())
            if shield is not None:
                state = shield.obs_to_state(np.asarray([obs]))
                action = int(shield.override(state, np.asarray([proposed]))[0])
                overrides += int(action != proposed)
            else:
                action = proposed
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += float(reward)
            unsafe_visits += int(float(info.get("cost", 0.0)) > 0.0)
            reached_goal = bool(info.get("success", False))
            length += 1
            done = bool(terminated or truncated)
        # Cross-check the env's own goal flag against the return identity the
        # step penalty makes available; they must agree.
        recovered = total_reward + step_penalty * length > 0.5
        if recovered != reached_goal:
            raise AssertionError(
                f"Goal indicator disagreement on episode {episode}: "
                f"env flag {reached_goal}, recovered {recovered}"
            )
        rows.append({
            "episode": episode, "episode_seed": EVAL_SEED_BASE + episode,
            "reward": total_reward, "length": length,
            "unsafe_state_visits": unsafe_visits,
            "safe_trajectory": unsafe_visits == 0,
            "reached_goal": reached_goal,
            "shield_overrides": overrides,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "mean_total_reward": statistics.fmean(row["reward"] for row in rows),
        "safety_rate": statistics.fmean(float(row["safe_trajectory"]) for row in rows),
        "success_rate": statistics.fmean(float(row["reached_goal"]) for row in rows),
        "mean_episode_length": statistics.fmean(float(row["length"]) for row in rows),
        "total_unsafe_state_visits": sum(row["unsafe_state_visits"] for row in rows),
        "shield_override_rate": (
            sum(row["shield_overrides"] for row in rows)
            / max(1, sum(row["length"] for row in rows))
        ),
    }


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    settings = read_json(pspo_root(args.size) / "experiment.json")["settings"]
    out_dir = args.output_root / args.method / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.size, settings)
    model = load_model(args.method, args.seed, args.size, env)
    shielded = args.method in SHIELDED_METHODS
    shield = make_shield(args.size, env) if shielded else None
    started = time.monotonic()
    rows = rollout(
        model, env, episodes=args.episodes,
        step_penalty=float(settings["step_penalty"]), shield=shield,
    )
    env.close()
    with (out_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": args.method, "seed": args.seed, "size": args.size,
        "model_path": str(model_path(args.method, args.seed, args.size)),
        "eval_seed_base": EVAL_SEED_BASE, "shielded": shielded,
        "shield_seed": SHIELD_SEED if shielded else None,
        "deterministic": True, "wall_seconds": time.monotonic() - started,
        "finished_utc": utc_now(), **summarise(rows),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


AGGREGATE_KEYS = (
    "mean_total_reward", "safety_rate", "success_rate", "mean_episode_length",
    "shield_override_rate",
)


def write_report(root: Path) -> None:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = root / method / f"seed{seed}" / "summary.json"
            if path.exists():
                rows.append(read_json(path))
    fields = ["method", "seed", "episodes", "mean_total_reward", "safety_rate",
              "success_rate", "mean_episode_length", "total_unsafe_state_visits",
              "shielded", "shield_override_rate"]
    with (root / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict = {
        "updated_utc": utc_now(), "eval_seed_base": EVAL_SEED_BASE,
        "deterministic": True, "methods": {},
    }
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        block = {
            "completed_seeds": len(method_rows),
            "episodes_per_seed": method_rows[0]["episodes"],
            "shielded": bool(method_rows[0].get("shielded", False)),
        }
        for key in AGGREGATE_KEYS:
            # Not every key applies to every method, and summaries written by an
            # earlier version of this script may predate a key entirely.
            values = [float(row[key]) for row in method_rows if key in row]
            if len(values) != len(method_rows):
                continue
            block[key] = statistics.fmean(values)
            block[key + "_2se"] = (
                2 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
            )
        aggregate["methods"][method] = block
    write_json(root / "aggregate.json", aggregate)


def run_all(args: argparse.Namespace) -> None:
    from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_baselines import (
        free_physical_cpus,
    )

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "_logs").mkdir(exist_ok=True)
    pending = [
        (method, seed) for method in METHODS for seed in SEEDS
        if not (root / method / f"seed{seed}" / "summary.json").exists()
    ]
    cpus, _evidence = free_physical_cpus(args.min_idle)
    if not cpus:
        raise RuntimeError("No sufficiently idle physical cores for the rollout sweep.")
    # Rollouts are deterministic, so sharing a core costs wall-clock but cannot
    # change a measurement. The cap keeps the sweep's footprint small on a busy
    # machine rather than claiming every core that clears the bar.
    available = list(cpus)[:args.max_concurrent]
    active: dict[tuple[str, int], tuple] = {}
    failures = []
    print(f"[{utc_now()}] {len(pending)} rollout jobs over {len(available)} cores", flush=True)
    while pending or active:
        while pending and available:
            method, seed = pending.pop(0)
            cpu = available.pop(0)
            log = (root / "_logs" / f"{method}_seed{seed}.log").open("a")
            command = ["taskset", "-c", str(cpu), str(REPO / ".venv/bin/python"), "-u",
                       str(Path(__file__).resolve()), "--worker",
                       "--method", method, "--seed", str(seed),
                       "--size", str(args.size), "--episodes", str(args.episodes),
                       "--output-root", str(root)]
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            active[(method, seed)] = (process, cpu, log)
        for key, (process, cpu, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[key]
            available.append(cpu)
            if code != 0:
                failures.append({"job": list(key), "returncode": code})
            print(f"[{utc_now()}] {key[0]}/seed{key[1]} rc={code}", flush=True)
        time.sleep(2)
    write_report(root)
    if failures:
        raise RuntimeError(f"Failed rollout jobs: {failures}")
    print(f"[{utc_now()}] complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-idle", type=float, default=95.0)
    parser.add_argument("--max-concurrent", type=int, default=len(SEEDS) * len(METHODS))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root or default_output(args.size)
    if args.worker:
        if args.method is None or args.seed is None:
            parser.error("--worker requires --method and --seed")
        worker(args)
    elif args.report_only:
        write_report(args.output_root)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
