#!/usr/bin/env python3
"""One-seed PSPO shaping pilot, without changing source-locked live stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-shaping-matplotlib")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from provably_safe_policy_optimisation import AdaptiveSafePPOV2  # noqa: E402

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_ppo_shaping as ppo,
)
from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_pspo import (  # noqa: E402
    build_safe_actor,
    source_paths,
)
from projects.safe_policy_optimisation.stages.train_pspo_precomputed import (  # noqa: E402
    base_state_dict_to_ppo_actor,
    policy_kwargs_from_base_architecture,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
    SparseFrozenLake,
    synthesise_shield,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (  # noqa: E402
    FrozenLakePotentialReward,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
)


class FinalizedRewardCurve(ppo.TimedRewardCurve):
    """Do not record the final curve point before PSPO's mandatory final flush."""

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.model._total_timesteps:
            return True
        return super()._on_step()

    def _on_training_end(self) -> None:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = ppo.build_parser()
    parser.description = __doc__
    parser.set_defaults(size=64)
    parser.add_argument(
        "--safety-frequency",
        type=int,
        default=100,
        help="FrozenLake PSPO cadence in rollouts; pending updates are always finalized.",
    )
    parser.add_argument("--lid-iters", type=int, default=200)
    parser.add_argument("--lid-checkpoint", type=int, default=100)
    parser.add_argument("--lid-batch-size", type=int, default=256)
    parser.add_argument(
        "--state-representation",
        choices=("one_hot", "state_id_lookup"),
        default="one_hot",
    )
    return parser


def prepare_safety_only_actor(
    args: argparse.Namespace, inputs: Path
) -> tuple[dict, np.ndarray, dict]:
    """Build the initial actor from the safety mask, before any potential exists."""
    inputs.mkdir(parents=True, exist_ok=False)
    env = SparseFrozenLake(
        size=args.size, success_rate=args.success_rate, step_penalty=args.step_penalty
    )
    try:
        started = time.perf_counter()
        mask, winning, _, _ = synthesise_shield(env)
        shield_seconds = time.perf_counter() - started
        if not winning[0]:
            raise ValueError("The start state must be safety-winning.")
        started = time.perf_counter()
        payload, audit = build_safe_actor(
            mask, state_representation=args.state_representation
        )
        actor_seconds = time.perf_counter() - started
        assert audit["initialisation_inputs"] == ["shield_action_mask"]
        assert not any(
            audit[key]
            for key in (
                "reward_information_used",
                "goal_information_used",
                "witness_policy_used",
            )
        )
        payload["safe_initialisation"] = audit
        torch.save(payload, inputs / "base_policy.pt")
        torch.save(
            {
                "shield": torch.as_tensor(mask),
                "winning_states": torch.as_tensor(winning),
            },
            inputs / "shield_q.pt",
        )
        (inputs / "layout.txt").write_text(
            "\n".join("".join(cell.decode() for cell in row) for row in env.desc) + "\n"
        )
        record = {
            "initial_actor": audit,
            "safety_winning_states": int(winning.sum()),
            "safe_state_action_pairs": int(mask.sum()),
            "shield_synthesis_seconds": shield_seconds,
            "policy_initialisation_seconds": actor_seconds,
            "initial_actor_built_before_shaping": True,
        }
        write_json(inputs / "initialisation_audit.json", record)
        return payload, mask, record
    finally:
        env.close()


def actor_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        if name.startswith("mlp_extractor.policy_net.") or name.startswith(
            "action_net."
        ):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def attach_training_shaping(
    model: Any, raw_env: gym.Env, args: argparse.Namespace
) -> FrozenLakePotentialReward:
    """Attach shaping only after the safety-only actor has been initialized."""
    before = actor_hash(model)
    train_env = FrozenLakePotentialReward(
        raw_env,
        gamma=args.gamma,
        scale=args.shaping_scale,
        timeout_mode=args.timeout_mode,
    )
    model.set_env(train_env)
    model.get_env().seed(args.seed)
    if actor_hash(model) != before:
        raise AssertionError(
            "Attaching reward shaping must not alter initial actor parameters."
        )
    return train_env


def evaluate(
    model: Any, factory: Any, *, episodes: int, seed: int
) -> tuple[list[dict], dict]:
    rows, metrics = ppo.evaluate(model, factory, episodes=episodes, seed=seed)
    metrics["algorithm"] = "pspo"
    return rows, metrics


def run(args: argparse.Namespace) -> dict:
    if args.size < 16 or args.total_timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Require size >= 16 and positive training/evaluation budgets.")
    if (args.hidden_dim, args.n_hidden) != (64, 2):
        raise ValueError(
            "The exact FrozenLake safety-only initializer uses two 64-unit layers."
        )
    if (
        min(
            args.safety_frequency,
            args.lid_iters,
            args.lid_checkpoint,
            args.lid_batch_size,
        )
        <= 0
    ):
        raise ValueError("Safety cadence and LID settings must be positive.")
    if args.curve_eval_freq < 0 or args.curve_eval_episodes <= 0:
        raise ValueError("Invalid curve-evaluation settings.")
    torch.set_num_threads(1)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    payload, mask, audit = prepare_safety_only_actor(args, root / "_inputs")
    cap = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else 5000 * args.size // 128
    )
    if cap <= 0:
        raise ValueError("Episode cap must be positive.")
    kwargs = {
        "size": args.size,
        "is_slippery": True,
        "success_rate": args.success_rate,
        "step_penalty": args.step_penalty,
    }

    def raw_factory():
        return gym.make(ENV_ID, max_episode_steps=cap, **kwargs)

    raw_train_env = raw_factory()
    curve_logger = LearningCurveLogger(
        curve_dir=root / "learning_curves", tensorboard_log_dir=root / "tensorboard"
    )
    training_logger = ppo.TrainingRewardLogger(
        root / "training_episodes.csv", gamma=args.gamma, scale=args.shaping_scale
    )
    curve = FinalizedRewardCurve(
        env_factory=raw_factory,
        curve_logger=curve_logger,
        eval_freq=args.curve_eval_freq,
        eval_episodes=args.curve_eval_episodes,
        seed=args.seed + 30000,
        reward_threshold=0,
        shield_mask=mask,
    )
    hyper = {
        key: getattr(args, key)
        for key in (
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        )
    }
    actor_state = base_state_dict_to_ppo_actor(
        payload["architecture"], payload["state_dict"]
    )
    started = time.perf_counter()
    train_env = None
    try:
        model = AdaptiveSafePPOV2(
            "MlpPolicy",
            raw_train_env,
            shield=mask,
            obs_to_state=raw_train_env.unwrapped.make_obs_to_state(),
            discrete_state_representation=args.state_representation,
            shield_seed=args.seed,
            shield_action_storage="proposed",
            base_policy_state_dict=actor_state,
            adaptive_granularity="train_phase",
            adaptive_frequency=args.safety_frequency,
            unsafe_update_strategy="rashomon_project",
            region_update_mode="replace",
            rashomon_n_iters=args.lid_iters,
            rashomon_checkpoint=args.lid_checkpoint,
            rashomon_batch_size=args.lid_batch_size,
            rashomon_certificate_samples=None,
            rashomon_inverse_temperature=1,
            rashomon_multi_label_mode="all",
            rashomon_surrogate="logsumexp",
            rashomon_objective="weighted_width",
            safe_region_shape="orthotope",
            rashomon_seed=args.seed,
            directional_rashomon_growth=True,
            stop_when_proposal_contained=True,
            policy_kwargs=policy_kwargs_from_base_architecture(payload["architecture"]),
            **hyper,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
        model_initialisation_seconds = time.perf_counter() - started
        for name, value in actor_state.items():
            torch.testing.assert_close(
                model.policy.state_dict()[name].cpu(), value, rtol=0, atol=0
            )
        initial_hash = actor_hash(model)
        assert model._greedy_safe_rate_now() == 1
        train_env = attach_training_shaping(model, raw_train_env, args)
        paths = set(source_paths()) | {
            Path(__file__).resolve(),
            Path(ppo.__file__).resolve(),
        }
        paths.update((REPO / "core/abstract_gradient_training").rglob("*.py"))
        config = {
            "algorithm": "pspo",
            "seed": args.seed,
            "env_id": ENV_ID,
            "env_kwargs": kwargs,
            "max_episode_steps": cap,
            "requested_timesteps": args.total_timesteps,
            "eval_episodes": args.eval_episodes,
            "curve_eval_freq": args.curve_eval_freq,
            "curve_eval_episodes": args.curve_eval_episodes,
            "state_representation": args.state_representation,
            "architecture": payload["architecture"],
            "training_hyperparameters": hyper,
            "shaping": {
                "enabled": args.shaping_scale != 0,
                "scale": args.shaping_scale,
                "gamma": args.gamma,
                "timeout_mode": args.timeout_mode,
                "training_only": True,
                "formula": "r + scale * (gamma * Phi(next_state) - Phi(state))",
            },
            "initialization": {
                **audit,
                "actor_sha256_before_shaping": initial_hash,
                "actor_sha256_after_shaping": actor_hash(model),
            },
            "safety_enforcement": {
                "variant": "orthotope_region_first_directional",
                "frequency_rollouts": args.safety_frequency,
                "lid_iterations": args.lid_iters,
                "lid_batch_size": args.lid_batch_size,
                "certificate_samples": None,
                "mandatory_final_flush": True,
                "runtime_training_shield": True,
                "runtime_evaluation_shield": False,
            },
            "source_sha256": {
                str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)
            },
        }
        write_json(root / "config.json", config)
        with zipfile.ZipFile(
            root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
        ) as snapshot:
            for p in sorted(paths):
                snapshot.write(p, str(p.relative_to(REPO)))
        np.savez_compressed(
            root / "potential.npz",
            distances=train_env.distances,
            potential=train_env.potential,
        )
        _, initial_metrics = evaluate(
            model,
            raw_factory,
            episodes=args.curve_eval_episodes,
            seed=args.seed + 10000,
        )
        write_json(root / "initial_metrics.json", initial_metrics)
        model.set_exploration_unsafe_action_callback(
            curve_logger.log_exploration_unsafe
        )
        curve_logger.start_timing(
            lambda: float(getattr(model, "_safety_enforcement_wall_time_s", 0))
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps, callback=[training_logger, curve]
        )
        learn_seconds = time.perf_counter() - started
        curve_learning_seconds = curve.evaluation_seconds
        _, pre_flush_metrics = evaluate(
            model,
            raw_factory,
            episodes=args.curve_eval_episodes,
            seed=args.seed + 10000,
        )
        write_json(root / "pre_finalization_metrics.json", pre_flush_metrics)
        started = time.perf_counter()
        model.finalize_adaptive_update()
        finalization_seconds = time.perf_counter() - started
        alignment = float(model._greedy_safe_rate_now())
        if alignment != 1:
            raise AssertionError(
                "The final nominal policy must pass exhaustive greedy safety verification."
            )
        curve.record_final_evaluation()
        model.save(root / "model.zip")
        started = time.perf_counter()
        rows, metrics = evaluate(
            model, raw_factory, episodes=args.eval_episodes, seed=args.seed + 10000
        )
        final_evaluation_seconds = time.perf_counter() - started
        write_json(root / "metrics.json", metrics)
        with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        diagnostics = model.adaptive_diagnostics()
        summary = {
            "algorithm": "pspo",
            "final_timesteps": model.num_timesteps,
            "completed_training_episodes": training_logger.episode,
            "training_seconds_excluding_curve_evaluation": learn_seconds
            - curve_learning_seconds
            + finalization_seconds,
            "learn_wall_seconds_including_curve_evaluation": learn_seconds,
            "curve_evaluation_seconds_during_learning": curve_learning_seconds,
            "finalization_seconds": finalization_seconds,
            "final_evaluation_seconds": final_evaluation_seconds,
            "shield_synthesis_seconds": audit["shield_synthesis_seconds"],
            "policy_initialisation_seconds": audit["policy_initialisation_seconds"],
            "model_initialisation_seconds": model_initialisation_seconds,
            "lid_computation_seconds": diagnostics["rashomon_wall_time_total_s"],
            "safety_enforcement_seconds": diagnostics[
                "safety_enforcement_wall_time_total_s"
            ],
            "max_discounted_telescoping_error": training_logger.max_telescoping_error,
            "adaptive_diagnostics": diagnostics,
            "training_shield_diagnostics": model.shield_diagnostics(),
            "final_exact_all_state_alignment": alignment,
            "initial_actor_sha256": initial_hash,
            "metrics": metrics,
        }
        write_json(root / "summary.json", summary)
        write_json(
            root / "certificate.json",
            {
                "greedy_alignment": alignment,
                "safety_winning_states_checked": audit["safety_winning_states"],
                "certificate_sampling": False,
                "nominal_policy_unshielded": True,
                "scope": "greedy actions on every safety-winning state under all positive-probability transitions",
            },
        )
        print(json.dumps({"run_dir": str(root), **summary}, indent=2), flush=True)
        return summary
    finally:
        (raw_train_env if train_env is None else train_env).close()
        training_logger.close()
        curve_logger.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
