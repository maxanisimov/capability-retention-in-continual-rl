#!/usr/bin/env python3
"""Run PSPO on structured slippery FrozenLake with (row, col) feature observations.

This is the compute-efficiency counterpart to ``run_stochastic_frozenlake_pspo.py``.
Environment, layout, dynamics, rewards, shield and witness proofs are identical;
only the actor's observation changes, from a one-hot state id to the normalised
``(row, col)`` pair. That collapses the exhaustive certificate dataset from a
dense ``|W| x |S|`` matrix to ``|W| x 2``.

The one-hot launcher builds its initial actor analytically by inverting the
one-hot first-layer columns. With two inputs that construction does not exist,
so the initial actor is fitted here instead, using only the shield action mask
and the *same* certification objective PSPO uses (all-safe-vs-unsafe logsumexp
margin, tau = 1). It is rejected unless it is exactly safe on every
safety-winning state. No goal, reward, witness, or goal-distance information is
available to the actor fit.

Every PPO and Rashomon hyperparameter is copied from the one-hot 128 run so the
two are directly comparable on wall time and memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
import traceback
import zipfile
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

from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_pspo import (  # noqa: E402
    INITIALISATION_SCHEME,
    RUNS_ROOT,
    episode_cap,
    evaluate_witnesses,
    goal_indicator,
    read_json,
    sha256,
    utc_now,
    write_json,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    proper_goal_policies,
    structured_layout,
    synthesise_shield,
)
from projects.safe_policy_optimisation.utils.frozen_lake_features_experiment import (  # noqa: E402
    ENV_ID,
    FeatureFrozenLake,
    feature_matrix,
)

DEFAULT_SIZE = 128
# Matches the analytic one-hot initialisation: all safe logits are equal, unsafe
# logits are lower, and the all-safe-vs-unsafe margin target is 4.
BC_MARGIN_TARGET = 4.0


def default_output(size: int) -> Path:
    return RUNS_ROOT / f"pspo_features_frozenlake{size}_safety_only"


def certification_margin(logits: torch.Tensor, targets: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """The ``mode="all"`` logsumexp margin PSPO certifies, on exact logits.

    Mirrors ``core/src/verification/verify.bound_multi_label_logsumexp_accuracy_margin``
    with a degenerate (point) interval, so a positive value here is the same
    quantity the Rashomon engine must bound below zero-free.
    """
    valid = targets.bool()
    invalid = ~valid
    neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype)
    has_invalid = invalid.any(dim=1)
    invalid_terms = (logits / tau).masked_fill(~invalid, neg_inf)
    invalid_terms = torch.where(
        has_invalid.unsqueeze(1), invalid_terms, torch.zeros_like(invalid_terms)
    )
    invalid_lse = tau * torch.logsumexp(invalid_terms, dim=1)
    valid_terms = (-logits / tau).masked_fill(~valid, neg_inf)
    valid_softmin = -tau * torch.logsumexp(valid_terms, dim=1)
    margins = valid_softmin - invalid_lse
    return torch.where(has_invalid, margins, torch.full_like(margins, BC_MARGIN_TARGET))


def build_safe_actor_features(
    env: FeatureFrozenLake, mask: np.ndarray,
    *, epochs: int = 8000, seed: int = 0,
) -> tuple[dict, dict]:
    """Fit a reward-agnostic actor on ``(row, col) -> safe-action mask``.

    The loss is solely a hinge on the exact certification margin. No goal,
    reward, witness policy, preferred action, or distance-to-goal information is
    accepted. Raises unless the fitted actor is exactly safe on every winning
    state.
    """
    features = feature_matrix(env)
    ids = np.flatnonzero(mask.sum(axis=1) > 0)
    x = torch.tensor(features[ids])
    y = torch.tensor(mask[ids].astype(np.float32))

    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(2, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4),
    )
    optimiser = torch.optim.Adam(network.parameters(), lr=3e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        optimiser.zero_grad()
        logits = network(x)
        margins = certification_margin(logits, y)
        safety_loss = torch.relu(BC_MARGIN_TARGET - margins).mean()
        loss = safety_loss
        loss.backward()
        optimiser.step()
        schedule.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                m = certification_margin(network(x), y)
            history.append({
                "epoch": epoch, "min_margin": float(m.min()),
                "safety_loss": float(safety_loss.detach()),
            })
    fit_seconds = time.perf_counter() - started

    with torch.no_grad():
        logits = network(x)
        margins = certification_margin(logits, y)
        greedy = logits.argmax(dim=1).numpy()
    greedy_safety = float(mask[ids, greedy].mean())
    if greedy_safety < 1.0:
        raise AssertionError(
            f"Fitted feature actor is unsafe on {int((1 - greedy_safety) * len(ids))} winning states."
        )
    if float(margins.min()) <= 0.0:
        raise AssertionError(
            f"Fitted feature actor has a non-positive certification margin: {float(margins.min())}."
        )
    architecture = {
        "activation": "Tanh", "hidden_dim": 64, "input_dim": 2,
        "n_actions": 4, "n_hidden": 2,
        "state_representation": "row_col_features",
    }
    analysis = {
        "initialisation": INITIALISATION_SCHEME,
        "initialisation_inputs": ["normalised_row_col_observations", "shield_action_mask"],
        "reward_information_used": False,
        "goal_information_used": False,
        "witness_policy_used": False,
        "reward_training_used": False,
        "all_winning_state_greedy_safety": greedy_safety,
        "minimum_all_safe_vs_unsafe_logit_margin": float(margins.min()),
        "mean_all_safe_vs_unsafe_logit_margin": float(margins.mean()),
        "bc_margin_target": BC_MARGIN_TARGET,
        "bc_epochs": epochs, "bc_fit_seconds": fit_seconds,
        "bc_history": history,
        "same_actor_for_all_training_seeds": True,
    }
    return {"architecture": architecture, "state_dict": network.state_dict()}, analysis


def source_paths() -> list[Path]:
    paths = {
        Path(__file__).resolve(),
        REPO / "projects/safe_policy_optimisation/utils/frozen_lake_experiment.py",
        REPO / "projects/safe_policy_optimisation/utils/frozen_lake_features_experiment.py",
        REPO / "projects/safe_policy_optimisation/scripts/run_stochastic_frozenlake_pspo.py",
    }
    for directory in (
        "core/provably_safe_policy_optimisation", "core/src",
        "projects/safe_policy_optimisation/stages", "projects/safe_policy_optimisation/utils",
    ):
        paths.update((REPO / directory).rglob("*.py"))
    return sorted(paths)


def settings_for(args: argparse.Namespace) -> dict:
    """Identical to the one-hot 128 run apart from the observation."""
    return {
        "size": args.size, "seeds": [args.seed], "success_rate": 0.8,
        "perpendicular_slip_probability_each": 0.1, "step_penalty": 0.001,
        "max_episode_steps": episode_cap(args.size),
        "total_timesteps": args.total_timesteps,
        "state_representation": "features",
        "initialisation": INITIALISATION_SCHEME,
        "observation": "normalised (row, col), Box(0, 1, (2,))",
        "pspo_variant": "adaptive directional orthotope, region-first, every 100 rollouts",
        "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.999,
        "learning_rate": 0.0003, "rashomon_n_iters": 200,
        "rashomon_batch_size": 256, "certificate": "exhaustive over all safety-winning states",
        "evaluation_episodes": 100, "evaluation_policy": "nominal greedy, no runtime shield",
        "comparison_baseline": "pspo_stochastic_frozenlake128_safety_only (one-hot), same hyperparameters",
    }


def prepare(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    root = args.output_root
    inputs = root / "_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (root / "_logs").mkdir(exist_ok=True)
    experiment_path = root / "experiment.json"
    expected = settings_for(args)
    if experiment_path.exists():
        record = read_json(experiment_path)
        if record["settings"] != expected:
            raise ValueError("Existing experiment settings differ; use another output root.")
        for name, digest in record["input_sha256"].items():
            if sha256(inputs / name) != digest:
                raise ValueError(f"Input provenance mismatch: {name}")
        return record

    layout_path = inputs / "layout.txt"
    layout_path.write_text("\n".join(structured_layout(args.size)) + "\n")
    env = FeatureFrozenLake(layout_path=str(layout_path))
    mask, winning, successors, probabilities = synthesise_shield(env)
    # Fit before computing goal-aware diagnostics. The fit receives only
    # observations and the safety mask.
    actor, actor_analysis = build_safe_actor_features(env, mask, epochs=args.bc_epochs)
    goal = env._n_states - 1
    if not winning[0]:
        raise ValueError("Initial state is not safety-winning.")
    distances, policies, proof = proper_goal_policies(
        mask, winning, successors, probabilities, goal
    )
    # The shared witness preflight treats the observation as a state id, so run
    # it against an index-mode instance of this same env (identical dynamics).
    index_env = FeatureFrozenLake(layout_path=str(layout_path), observation_mode="index")
    empirical = evaluate_witnesses(index_env, policies, expected["max_episode_steps"])
    index_env.close()
    actor.update({"environment_id": ENV_ID, "safe_initialisation": actor_analysis})
    torch.save(actor, inputs / "base_policy.pt")
    torch.save({
        "shield": torch.tensor(mask), "winning_states": torch.tensor(winning),
        "env_id": ENV_ID, "layout_sha256": sha256(layout_path),
        "risk_threshold": 0.0, "safety_requirement": "never enter a hole",
        "synthesis": "greatest fixed point over all positive-probability successors",
    }, inputs / "shield_q.pt")
    np.savez_compressed(
        inputs / "witness_policies.npz", policies=np.stack(policies), distances=distances
    )
    n_winning = int(winning.sum())
    validation = {
        "nominal_state_count": env._n_states, "action_count": env._n_actions,
        "observation_space": str(env.observation_space),
        "safety_winning_states": n_winning,
        "safe_state_action_pairs": int(mask.sum()),
        "certificate_dataset_bytes": n_winning * 2 * 4,
        "one_hot_certificate_dataset_bytes": n_winning * env._n_states * 4,
        "start_safety_winning": True, "goal_reachability": proof,
        "witness_evaluation": empirical, "witnesses_used_for_initialisation": False,
        "initial_actor": actor_analysis,
    }
    write_json(inputs / "layout_validation.json", validation)
    files = ["layout.txt", "base_policy.pt", "shield_q.pt", "layout_validation.json",
             "witness_policies.npz"]
    record = {
        "created_utc": utc_now(), "settings": expected, "env_id": ENV_ID,
        "input_sha256": {name: sha256(inputs / name) for name in files},
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in source_paths()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
    }
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            archive.write(path, path.relative_to(REPO))
    write_json(experiment_path, record)
    env.close()
    print(json.dumps({k: v for k, v in validation.items() if k != "goal_reachability"}, indent=2), flush=True)
    return record


def training_arguments(root: Path, seed: int, *, smoke: bool = False) -> list[str]:
    settings = read_json(root / "experiment.json")["settings"]
    inputs = root / "_inputs"
    return [
        "--env-id", ENV_ID, "--env-kwargs", json.dumps({
            "layout_path": str(inputs / "layout.txt"), "is_slippery": True,
            "success_rate": settings["success_rate"], "step_penalty": settings["step_penalty"],
            "observation_mode": "features",
        }),
        "--base-policy-path", str(inputs / "base_policy.pt"),
        "--shield-path", str(inputs / "shield_q.pt"), "--state-representation", "features",
        "--max-episode-steps", str(settings["max_episode_steps"]), "--cost-limit", "0",
        "--total-timesteps", str(2048 if smoke else settings["total_timesteps"]),
        "--seed", str(seed), "--device", "cpu", "--evaluation-policy", "unshielded",
        "--learning-rate", "0.0003", "--n-steps", "2048", "--batch-size", "64",
        "--n-epochs", "1" if smoke else "10", "--gamma", "0.999", "--gae-lambda", "0.95",
        "--clip-range", "0.2", "--ent-coef", "0", "--vf-coef", "0.5", "--max-grad-norm", "0.5",
        "--freq", "100", "--verify-first", "false", "--region-refresh", "adaptive",
        "--directional", "true", "--region-mode", "replace",
        "--unsafe-update-strategy", "rashomon_project",
        "--n-iters", "2" if smoke else "200", "--rashomon-checkpoint", "2" if smoke else "100",
        "--rashomon-batch-size", "256", "--rashomon-inverse-temp", "1",
        "--rashomon-multi-label-mode", "all", "--surrogate", "logsumexp",
        "--rashomon-objective", "weighted_width", "--safe-region-shape", "orthotope",
        "--eval-episodes", "2" if smoke else "100", "--early-stop-eval-freq", "0",
        "--curve-eval-freq", "0" if smoke else "100000", "--curve-eval-episodes", "2" if smoke else "10",
        "--output-dir", str(root / "_smoke" if smoke else root), "--run-id", f"seed{seed}",
    ]


# train_pspo.run builds the unshielded-evaluation audit shield as
# ``Shield(mask, seed=...)`` with no obs_to_state, so it falls back to the
# identity "the observation *is* the state id" decode. That holds in one-hot
# mode (the observation is an int) but raises on a 2-vector feature
# observation. Features + --evaluation-policy unshielded was never exercised
# together, so this is a pre-existing bug in a file this script must not edit
# (its hash is pinned by the running 256 sweep's provenance check). Patch the
# decode in at runtime instead; it is a no-op in index mode.
UNSHIELDED_EVAL_PATCH = (
    "train_pspo.evaluate_unshielded_policy wrapped to install the env's exact "
    "feature->state-id inverse on the audit Shield (train_pspo.py builds it "
    "without an obs_to_state, which only works for index observations)"
)


def patch_unshielded_eval_decode(module) -> None:
    original = module.evaluate_unshielded_policy

    def patched(model, env, shield, *, episodes, seed):
        unwrapped = env.unwrapped
        if getattr(unwrapped, "_observation_mode", "index") == "features":
            shield.obs_to_state = unwrapped.make_obs_to_state()
        return original(model, env, shield, episodes=episodes, seed=seed)

    module.evaluate_unshielded_policy = patched


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    root = args.output_root
    run_dir = (root / "_smoke" if args.smoke else root) / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    record = read_json(root / "experiment.json")
    for name, digest in record["input_sha256"].items():
        if sha256(root / "_inputs" / name) != digest:
            raise ValueError(f"Changed input: {name}")
    status = {
        "status": "running", "seed": args.seed, "smoke": args.smoke, "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "started_utc": utc_now(),
    }
    write_json(status_path, status)
    started = time.perf_counter()
    try:
        from projects.safe_policy_optimisation.stages import train_pspo
        patch_unshielded_eval_decode(train_pspo)
        summary = train_pspo.run(
            train_pspo.parse_args(training_arguments(root, args.seed, smoke=args.smoke))
        )
        wall = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if float(summary["final_exact_all_state_alignment"]) < 1.0:
            raise RuntimeError("Final nominal actor failed exhaustive safety audit.")
        with (run_dir / "episodes.csv").open() as handle:
            episodes = list(csv.DictReader(handle))
        penalty = record["settings"]["step_penalty"]
        successes = sum(
            goal_indicator(float(row["reward"]), int(row["length"]), penalty) for row in episodes
        )
        metrics_path = run_dir / "metrics.json"
        metrics = read_json(metrics_path)
        write_json(run_dir / "metrics_reward_threshold.json", metrics)
        metrics["success"] = {
            "success_mode": "goal_reached", "success_count": successes,
            "success_rate": successes / len(episodes),
            "definition": "goal indicator recovered exactly from return plus step penalty times length",
        }
        write_json(metrics_path, metrics)
        write_json(run_dir / "efficiency.json", {
            "state_representation": "features", "wall_time_s": wall,
            "runtime_patches": [UNSHIELDED_EVAL_PATCH],
            "peak_rss_bytes": peak_rss,
            "timing": summary.get("timing"),
            "certificate_regions": summary.get("adaptive_diagnostics", {}).get("certificate_regions"),
        })
        status.update({
            "status": "complete", "completed_utc": utc_now(),
            "wall_time_s": wall, "peak_rss_bytes": peak_rss,
        })
        write_json(status_path, status)
        print(f"COMPLETED seed={args.seed} wall={wall:.1f}s peak_rss={peak_rss/2**30:.2f}GiB", flush=True)
    except Exception:
        status.update({
            "status": "failed", "failed_utc": utc_now(), "traceback": traceback.format_exc(),
        })
        write_json(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--bc-epochs", type=int, default=8000)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1):
        parser.error("--size must be a power of two and at least 16")
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.output_root is None:
        args.output_root = default_output(args.size)
    args.output_root = args.output_root.resolve()
    if args.run:
        prepare(args)
        worker(args)
    elif args.prepare:
        prepare(args)
    else:
        parser.error("Choose --prepare or --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
