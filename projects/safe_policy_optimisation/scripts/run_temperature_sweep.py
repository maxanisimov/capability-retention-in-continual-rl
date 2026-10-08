#!/usr/bin/env python3
"""Evaluate every trained policy under temperature-scaled stochastic deployment.

Every other result in this project evaluates the final *greedy* policy, and
PSPO's safety argument is stated for greedy execution: the safe-parameter-region
verification certifies that the argmax action is safe in every state. This
script asks what survives when the policy is deployed stochastically instead --
sampling from ``softmax(logits / T)`` -- sweeping ``T`` from 0 (exact argmax,
the certified case) up to a value large enough to be near-uniform.

Because a fixed ``T`` is not equally stochastic across methods (logit scales
differ), each cell also records the achieved policy entropy normalised by
``log |A|``. That makes "approaching uniform" a measured fact and gives a
method-fair second x-axis.

Cost shape, measured on this machine: importing torch/SB3 off the NFS home
costs ~170 s per process and loading one checkpoint ~60 s, while rollout runs
at ~2k steps/s. So the unit of work is one (environment, seed, variant) -- the
checkpoint is loaded once and every temperature is swept inside it. Reloading
per temperature would add hours of pure loading.

Results are written one JSON per unit and completed units are skipped, so the
sweep is resumable and partial output is usable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_TEMPERATURES = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671"
    / "analysis/temperature_sweep"
)


@dataclass(frozen=True)
class VariantSpec:
    """One evaluable policy: where its checkpoint is and how to run it."""

    key: str
    label: str
    # Directory under seed<k>/ holding config.json and the checkpoint.
    stage_dir: str
    # Checkpoint filename within stage_dir.
    checkpoint: str
    # "sb3" (model.zip via a stable-baselines3 subclass) or "baselines"
    # (a .pt state dict rebuilt by utils.safe_rl.load_checkpoint_model).
    family: str
    # True for the PSPO runs, which live under a different root.
    adaptive: bool = False
    # Apply the runtime shield to the sampled action before stepping.
    shielded: bool = False


VARIANTS = (
    VariantSpec("ppo_policy", "PPO", "ppo_policy", "model.zip", "sb3"),
    VariantSpec(
        "ppo_lagrangian", "PPO-Lagrangian", "ppo_lagrangian",
        "ppo_lagrangian.pt", "baselines",
    ),
    VariantSpec(
        "ppo_pid_lagrangian", "PPO-PID-Lagrangian", "ppo_lagrangian",
        "ppo_pid_lagrangian.pt", "baselines",
    ),
    VariantSpec("cpo", "CPO", "cpo", "cpo.pt", "baselines"),
    VariantSpec(
        "ppo_shield", "PPO-Shield", "ppo_shield", "model.zip", "sb3",
        shielded=True,
    ),
    VariantSpec(
        "ppo_shield_nominal", "PPO-Shield (shield-free)", "ppo_shield",
        "model.zip", "sb3",
    ),
    VariantSpec("pspo", "PSPO", ".", "model.zip", "sb3", adaptive=True),
)
VARIANT_BY_KEY = {spec.key: spec for spec in VARIANTS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", action="append",
        help="Environment key; repeat as needed (default: all six).",
    )
    parser.add_argument(
        "--variant", action="append", choices=tuple(VARIANT_BY_KEY),
        help="Variant to evaluate; repeat as needed (default: all seven).",
    )
    parser.add_argument("--seed", action="append", type=int, help="Seeds (default: 0-9).")
    parser.add_argument(
        "--temperature", action="append", type=float,
        help=f"Temperature; repeat. Default: {DEFAULT_TEMPERATURES}.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="Parallel worker processes (default: min(4, cpu_count)).",
    )
    parser.add_argument(
        "--reset-seed-mode",
        choices=("uniform", "stored"),
        default="uniform",
        help=(
            "uniform (default): every method resets on seed+10000, so all "
            "methods see the same initial states. stored: reproduce each "
            "training stage's own offset, so T=0 matches the shipped "
            "metrics.json exactly (use for the correctness check)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute units whose result file already exists.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the pending unit list and exit without evaluating.",
    )
    return parser.parse_args(argv)


def _unit_path(output_dir: Path, env_key: str, seed: int, variant: str) -> Path:
    return output_dir / env_key / f"seed{seed}" / f"{variant}.json"


def _logits_fn(model, family: str):
    """Return ``obs -> 1-D logit tensor`` for either checkpoint family.

    Both families end at a discrete Categorical, but reach it differently: the
    SB3 policies expose a distribution object, while the safe_rl_baselines
    actors return the logit vector straight from ``forward`` (see
    ``core/safe_rl_baselines/ppo_lagrangian.py`` ``predict``). CPO's actor
    wraps it in a dict on some paths, so unwrap that.
    """
    import torch

    if family == "sb3":
        policy = model.policy

        def logits(obs):
            tensor, _ = policy.obs_to_tensor(obs)
            with torch.no_grad():
                return policy.get_distribution(tensor).distribution.logits.reshape(-1)

        return logits

    # _preprocess one-hots Discrete observations; the actor's first layer is
    # (hidden, n_states), so feeding the raw integer would be a shape error.
    preprocess = model._preprocess
    actor = model.actor

    def logits(obs):
        with torch.no_grad():
            out = actor(preprocess(np.asarray(obs)))
        if isinstance(out, dict):
            out = out["logits"]
        return out.reshape(-1)

    return logits


def _load(spec: VariantSpec, run_dir: Path, env, config: dict):
    """Load one checkpoint, returning the model object."""
    path = run_dir / spec.checkpoint
    if spec.family == "sb3":
        if spec.adaptive:
            from core.provably_safe_policy_optimisation.adaptive_safe_ppo_v2 import (
                AdaptiveSafePPOV2,
            )

            return AdaptiveSafePPOV2.load(path, env=env, device="cpu")
        from core.provably_safe_policy_optimisation.provably_safe_ppo import (
            ProvablySafePPO,
        )

        return ProvablySafePPO.load(path, env=env, device="cpu")
    # Deliberately NOT utils.safe_rl.load_checkpoint_model: it rebuilds via
    # build_safe_rl_baseline without passing net_arch, so it silently assumes
    # the (64, 64) default and raises a state-dict shape error on any run with
    # a different architecture. net_arch is not in the checkpoint metadata, so
    # take it from the stage config, and load only the actor -- the critics are
    # not needed for a rollout.
    import torch

    from projects.safe_policy_optimisation.utils.safe_rl import build_safe_rl_baseline

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    net_arch = tuple(
        [int(config["hidden_dim"])] * int(config["n_hidden"])
    ) if "hidden_dim" in config and "n_hidden" in config else (64, 64)
    model = build_safe_rl_baseline(
        checkpoint["algorithm"],
        env,
        cost_limit=float(metadata.get("cost_limit", config.get("cost_limit", 0.0))),
        seed=int(metadata.get("seed", config.get("seed", 0))),
        device="cpu",
        net_arch=net_arch,
    )
    model.actor.load_state_dict(checkpoint["actor_state_dict"])
    model.actor.eval()
    return model


def evaluate_unit(
    env_key: str,
    seed: int,
    variant_key: str,
    *,
    baseline_root: str,
    adaptive_root: str,
    temperatures: tuple[float, ...],
    episodes: int,
    output_path: str,
    reset_seed_mode: str = "uniform",
) -> dict:
    """Evaluate one (environment, seed, variant) across every temperature.

    Runs inside a worker process: imports, env construction and checkpoint
    loading all happen here, once, and are then amortised over the whole
    temperature grid.
    """
    import torch

    # Four workers each spawning intra-op threads would oversubscribe 4 cores.
    torch.set_num_threads(1)

    from core.provably_safe_policy_optimisation.shield import Shield
    from projects.safe_policy_optimisation.stages.train_ppo_shield import (
        make_unshielded_env,
    )
    from projects.safe_policy_optimisation.utils.metrics import (
        success_mode_for_env,
        summarise_evaluation,
    )
    from projects.safe_policy_optimisation.utils.safe_rl import obs_state_id, state_cost
    from projects.safe_policy_optimisation.utils.shield import load_shield_mask

    spec = VARIANT_BY_KEY[variant_key]
    root = Path(adaptive_root if spec.adaptive else baseline_root)
    run_dir = (root / f"seed{seed}" / spec.stage_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())

    started = time.perf_counter()
    env = make_unshielded_env(
        config["env_id"],
        env_kwargs=config.get("env_kwargs", {}),
        max_episode_steps=config.get("max_episode_steps"),
        cost_limit=float(config.get("cost_limit", 0.0)),
        record_episodes=False,
    )
    mask = load_shield_mask(
        Path(config["shield_path"]),
        shield_key=config.get("shield_key", "shield"),
        source=config.get("shield_source", "shield"),
        risk_threshold=config.get("risk_threshold"),
    )
    model = _load(spec, run_dir, env, config)
    logits_of = _logits_fn(model, spec.family)
    load_seconds = time.perf_counter() - started

    # Reset seeds. The SB3 stages all used seed+10_000, but the safe-RL
    # baselines used seed + 10_000 + offset*1_000 where offset is the
    # algorithm's index in that stage's --algorithms list
    # (stages/train_ppo_lagrangian.py:470) -- so PPO-PID-Lagrangian's stored
    # evaluation sits at +11_000.
    #
    # "uniform" (default) puts every method on the same initial states, which
    # is what makes a cross-method comparison fair. "stored" reproduces each
    # stage's own offset, which is what makes T=0 reproduce the shipped
    # metrics.json exactly -- used for the correctness check.
    base_seed = int(config.get("seed", seed)) + 10_000
    if reset_seed_mode == "stored":
        algorithms = config.get("algorithms")
        if algorithms and spec.key in algorithms:
            base_seed += 1_000 * list(algorithms).index(spec.key)
    success_mode = success_mode_for_env(config["env_id"])
    threshold = float(config.get("success_reward_threshold", 0.0))
    cost_limit = float(config.get("cost_limit", 0.0))
    shield = (
        Shield(mask, obs_to_state=env.unwrapped.make_obs_to_state(), seed=base_seed)
        if spec.shielded
        else None
    )

    results: dict[str, dict] = {}
    for temperature in temperatures:
        generator = torch.Generator().manual_seed(base_seed + 7919)
        records: list[dict] = []
        entropy_total = 0.0
        entropy_steps = 0
        proposals = 0
        unsafe_proposals = 0
        for episode in range(episodes):
            obs, _ = env.reset(seed=base_seed + episode)
            done = False
            total_reward = 0.0
            initial_cost = state_cost(env, obs)
            total_cost = initial_cost
            unsafe_visits = int(initial_cost > 0.0)
            length = 0
            while not done:
                logits = logits_of(obs)
                if temperature == 0.0:
                    # Exact argmax. Never approximate greedy with a tiny
                    # temperature -- dividing by ~0 overflows to NaN.
                    action = int(torch.argmax(logits).item())
                    probabilities = torch.zeros_like(logits)
                    probabilities[action] = 1.0
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    action = int(
                        torch.multinomial(probabilities, 1, generator=generator).item()
                    )
                nonzero = probabilities[probabilities > 0]
                entropy_total += float(-(nonzero * nonzero.log()).sum())
                entropy_steps += 1

                state = obs_state_id(env, obs)
                proposals += 1
                unsafe_proposals += int(not bool(mask[state, action]))
                if shield is not None:
                    action = int(
                        shield.override(
                            np.asarray([state]), np.asarray([action])
                        )[0]
                    )

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_cost = state_cost(env, obs, info)
                total_cost += step_cost
                unsafe_visits += int(step_cost > 0.0)
                length += 1
                done = bool(terminated or truncated)
            # Key names match what summarise_evaluation reads ("reward", not
            # "total_reward"); safe_trajectory follows utils.safe_rl's
            # definition of an episode that never entered an unsafe state.
            records.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "cost": total_cost,
                    "length": length,
                    "violated": total_cost > cost_limit,
                    "unsafe_state_visit_count": unsafe_visits,
                    "safe_trajectory": unsafe_visits == 0,
                }
            )

        summary = summarise_evaluation(
            records,
            success_reward_threshold=threshold,
            cost_limit=cost_limit,
            algorithm=spec.key,
            success_mode=success_mode,
        )
        action_count = int(mask.shape[1])
        summary["entropy"] = {
            "mean_nats": entropy_total / entropy_steps if entropy_steps else 0.0,
            "mean_normalised": (
                (entropy_total / entropy_steps) / math.log(action_count)
                if entropy_steps and action_count > 1
                else 0.0
            ),
            "action_count": action_count,
        }
        summary["proposed_action_safety"] = {
            "proposed_action_checks": proposals,
            "unsafe_proposed_action_count": unsafe_proposals,
            "unsafe_proposed_action_rate": (
                unsafe_proposals / proposals if proposals else 0.0
            ),
        }
        results[f"{temperature:g}"] = summary

    env.close()
    payload = {
        "environment_key": env_key,
        "seed": seed,
        "variant": spec.key,
        "variant_label": spec.label,
        "shielded": spec.shielded,
        "episodes": episodes,
        "temperatures": list(temperatures),
        "reset_seed_base": base_seed,
        "reset_seed_mode": reset_seed_mode,
        "run_dir": str(run_dir),
        "load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "environment_key": env_key,
        "seed": seed,
        "variant": spec.key,
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    from plot_extended_budget_learning_curves import (
        ENVIRONMENT_BY_KEY,
        ENVIRONMENTS,
        EXPECTED_SEEDS,
    )

    env_keys = args.environment or [e.key for e in ENVIRONMENTS]
    unknown = sorted(set(env_keys) - set(ENVIRONMENT_BY_KEY))
    if unknown:
        raise SystemExit(f"unknown environment(s): {unknown}")
    environments = [ENVIRONMENT_BY_KEY[key] for key in dict.fromkeys(env_keys)]
    variants = [
        VARIANT_BY_KEY[key]
        for key in dict.fromkeys(args.variant or list(VARIANT_BY_KEY))
    ]
    seeds = list(dict.fromkeys(args.seed or list(EXPECTED_SEEDS)))
    temperatures = tuple(
        dict.fromkeys(args.temperature or list(DEFAULT_TEMPERATURES))
    )
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if any(value < 0 for value in temperatures):
        raise SystemExit("temperatures must be non-negative")

    pending = []
    done = 0
    for environment in environments:
        for seed in seeds:
            for spec in variants:
                path = _unit_path(args.output_dir, environment.key, seed, spec.key)
                if path.is_file() and not args.overwrite:
                    done += 1
                    continue
                pending.append(
                    {
                        "env_key": environment.key,
                        "seed": seed,
                        "variant_key": spec.key,
                        "baseline_root": str(environment.baseline_root),
                        "adaptive_root": str(environment.adaptive_root),
                        "temperatures": temperatures,
                        "episodes": args.episodes,
                        "output_path": str(path),
                        "reset_seed_mode": args.reset_seed_mode,
                    }
                )

    total = done + len(pending)
    print(
        f"{total} units ({len(environments)} envs x {len(seeds)} seeds x "
        f"{len(variants)} variants); {done} already done, {len(pending)} pending"
    )
    print(f"temperatures: {', '.join(f'{t:g}' for t in temperatures)}")
    print(f"episodes per cell: {args.episodes}")
    if args.list or not pending:
        for job in pending:
            print(f"  {job['env_key']}/seed{job['seed']}/{job['variant_key']}")
        return 0

    workers = max(1, min(args.workers, len(pending)))
    print(f"running on {workers} worker(s); first result lands after the "
          f"~3 min per-process import\n", flush=True)
    completed = 0
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_unit,
                job["env_key"], job["seed"], job["variant_key"],
                baseline_root=job["baseline_root"],
                adaptive_root=job["adaptive_root"],
                temperatures=job["temperatures"],
                episodes=job["episodes"],
                output_path=job["output_path"],
                reset_seed_mode=job["reset_seed_mode"],
            ): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            label = f"{job['env_key']}/seed{job['seed']}/{job['variant_key']}"
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append(f"{label}: {type(error).__name__}: {error}")
                print(f"FAILED {label}: {type(error).__name__}: {error}", flush=True)
                continue
            completed += 1
            elapsed = time.perf_counter() - started
            rate = elapsed / completed
            remaining = (len(pending) - completed) * rate / workers
            print(
                f"[{completed}/{len(pending)}] {label} "
                f"({result['elapsed_seconds']:.0f}s) "
                f"~{remaining / 60:.0f} min left",
                flush=True,
            )

    if failures:
        print(f"\n{len(failures)} unit(s) failed:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"\nAll {completed} unit(s) complete in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
