"""Train AdaptiveSafePPO: shielded exploration + adaptive verify-then-project updates.

Unlike ``train_pspo_precomputed.py`` (which loads a *precomputed*
Rashomon set and projects every gradient step onto it), this stage loads only a
safe base policy (``base_policy.pt`` from ``compute_shield_rashomon_set.py``)
and lets :class:`AdaptiveSafePPO` verify each policy update, computing a
Rashomon set on demand around the last safe iterate only when a candidate
update is unsafe.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation import AdaptiveSafePPO, Shield  # noqa: E402

from projects.safe_policy_optimisation.stages.train_ppo_shield import (  # noqa: E402
    _episode_rows,
    _records_to_metrics,
    _training_rows,
    evaluate_shielded_policy,
    evaluate_unshielded_policy,
    load_shield_mask,
    make_unshielded_env,
    validate_shield_for_env,
)
from projects.safe_policy_optimisation.stages.train_pspo_precomputed import (  # noqa: E402
    EarlyStopOnSuccessCallback,
    _base_to_ppo_actor_name_map,
    _make_env_factory,
    _resolve_curve_eval_freq,
    _write_csv,
    policy_kwargs_from_base_architecture,
)
from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs  # noqa: E402
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.metrics import (  # noqa: E402
    success_mode_for_env,
    summarise_evaluation,
)
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    aggregate_training_violations,
    aggregate_violations,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

ALGORITHM_NAME = "adaptive_safe_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "adaptive_safe_policy"
)


def load_base_policy_payload(base_policy_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load ``(architecture, state_dict)`` from a saved ``base_policy.pt``."""

    payload = torch.load(base_policy_path, map_location="cpu", weights_only=False)
    for key in ("architecture", "state_dict"):
        if key not in payload:
            raise KeyError(
                f"Base policy file must contain {key!r}; keys={sorted(payload.keys())}."
            )
    architecture = dict(payload["architecture"])
    required = {"input_dim", "n_actions", "hidden_dim", "n_hidden", "activation"}
    missing = sorted(required.difference(architecture))
    if missing:
        raise KeyError(f"Base policy architecture is missing keys: {missing}")
    return architecture, dict(payload["state_dict"])


def base_state_dict_to_ppo_actor(
    architecture: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename saved Sequential parameters to SB3 PPO actor parameter names."""

    name_map = _base_to_ppo_actor_name_map(architecture)
    missing = sorted(set(name_map) - set(state_dict))
    if missing:
        raise KeyError(f"Base policy state_dict is missing parameters: {missing}")
    return {ppo_name: state_dict[base_name] for base_name, ppo_name in name_map.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AdaptiveSafePPO from a saved shield and safe base policy.",
    )
    parser.add_argument("--base-policy-path", type=Path, required=True)
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to gym.make.")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--shield-key", default="shield")
    parser.add_argument("--shield-source", choices=("shield", "action_risk"), default="shield")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--shield-action-storage", choices=("proposed", "executed"), default="proposed")
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adaptive-granularity",
        choices=("gradient_step", "train_phase"),
        default="gradient_step",
        help="What counts as one policy-update candidate for the adaptive scheme.",
    )
    parser.add_argument(
        "--unsafe-update-strategy",
        choices=("rashomon_project", "none"),
        default="rashomon_project",
        help=(
            "Fallback applied when a candidate policy update fails verification. "
            "'none' is the monitor-only ablation: verify and record, never "
            "correct -- reports safe_update_fraction but gives no safety "
            "guarantee."
        ),
    )
    parser.add_argument(
        "--rashomon-n-iters",
        type=int,
        default=100,
        help="Optimization budget of each on-demand Rashomon-set computation.",
    )
    parser.add_argument(
        "--rashomon-checkpoint",
        type=int,
        default=None,
        help="Engine checkpoint cadence. Defaults to max(1, rashomon-n-iters // 10).",
    )
    parser.add_argument("--rashomon-batch-size", type=int, default=500)
    parser.add_argument(
        "--certificate-samples",
        type=int,
        default=None,
        help="Certificate batch size. Defaults to all states with a safe action (exhaustive).",
    )
    parser.add_argument(
        "--rashomon-inverse-temp",
        type=int,
        default=None,
        help="Fixed inverse temperature for the Rashomon surrogate. Defaults to per-call calibration.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early-stop-eval-freq", type=int, default=5_000)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory for learning curves. Defaults to <run-dir>/tensorboard.",
    )
    parser.add_argument(
        "--curve-eval-freq",
        type=int,
        default=None,
        help=(
            "Evaluate and log unshielded total reward every N timesteps. "
            "Defaults to --early-stop-eval-freq when positive, otherwise --n-steps. Use 0 to disable."
        ),
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--evaluation-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help=(
            "Policy used for the final evaluation rollout. 'unshielded' executes the raw greedy "
            "policy and audits whether its proposed actions are shield-safe; 'shielded' applies "
            "the shield before stepping the environment."
        ),
    )
    parser.add_argument(
        "--early-stop-eval-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help=(
            "Policy evaluated by the early-stopping callback. 'unshielded' evaluates the "
            "raw model action without applying the shield; 'shielded' evaluates deployment "
            "with shield overrides."
        ),
    )
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id is None:
        raise ValueError("--env-id is required for adaptive safe policy training.")
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    mask = load_shield_mask(
        args.shield_path,
        shield_key=args.shield_key,
        source=args.shield_source,
        risk_threshold=args.risk_threshold,
    )
    architecture, base_state_dict = load_base_policy_payload(args.base_policy_path)
    base_policy_state_dict = base_state_dict_to_ppo_actor(architecture, base_state_dict)
    policy_kwargs = policy_kwargs_from_base_architecture(architecture)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=args.tensorboard_log_dir or run_dir / "tensorboard",
    )
    curve_eval_freq = _resolve_curve_eval_freq(args)

    env_factory = _make_env_factory(args, env_kwargs, mask)
    train_env = env_factory(True)
    validate_shield_for_env(mask, train_env)
    try:
        from stable_baselines3.common.preprocessing import get_flattened_obs_dim

        expected_input_dim = int(get_flattened_obs_dim(train_env.observation_space))
        expected_n_actions = int(train_env.action_space.n)
        # Features mode: the exact verifier enumerates states via the env's
        # forward feature map; the shield needs the inverse.
        feature_mode = getattr(train_env.unwrapped, "_observation_mode", "index") == "features"
        state_to_features = train_env.unwrapped.state_to_features if feature_mode else None
        if int(architecture["input_dim"]) != expected_input_dim:
            raise ValueError(
                "Base policy input_dim does not match environment observation space: "
                f"architecture={architecture['input_dim']}, env={expected_input_dim}."
            )
        if int(architecture["n_actions"]) != expected_n_actions:
            raise ValueError(
                "Base policy n_actions does not match environment action space: "
                f"architecture={architecture['n_actions']}, env={expected_n_actions}."
            )

        model = AdaptiveSafePPO(
            "MlpPolicy",
            train_env,
            shield=mask,
            obs_to_state=train_env.unwrapped.make_obs_to_state(),
            state_to_features=state_to_features,
            shield_seed=args.seed,
            shield_action_storage=args.shield_action_storage,
            base_policy_state_dict=base_policy_state_dict,
            adaptive_granularity=args.adaptive_granularity,
            unsafe_update_strategy=args.unsafe_update_strategy,
            rashomon_n_iters=args.rashomon_n_iters,
            rashomon_checkpoint=args.rashomon_checkpoint,
            rashomon_batch_size=args.rashomon_batch_size,
            rashomon_certificate_samples=args.certificate_samples,
            rashomon_inverse_temperature=args.rashomon_inverse_temp,
            rashomon_seed=args.seed,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        model.set_exploration_unsafe_action_callback(curve_logger.log_exploration_unsafe)
        reward_curve = UnshieldedRewardCurveCallback(
            env_factory=lambda: env_factory(False),
            curve_logger=curve_logger,
            eval_freq=curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30_000,
            reward_threshold=args.success_reward_threshold,
            shield_mask=mask,
        )
        early_stop = EarlyStopOnSuccessCallback(
            env_factory=env_factory,
            shield_mask=mask,
            eval_freq=args.early_stop_eval_freq,
            eval_episodes=args.early_stop_eval_episodes,
            success_rate=args.early_stop_success_rate,
            seed=args.seed + 20_000,
            reward_threshold=args.success_reward_threshold,
            eval_policy=args.early_stop_eval_policy,
        )
        log_info(f"[{ALGORITHM_NAME}] training for up to {args.total_timesteps} timesteps")
        model.learn(total_timesteps=args.total_timesteps, callback=[reward_curve, early_stop])
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(train_env.episodes)
        executed_action_diagnostics = train_env.diagnostics()
        executed_action_records = list(train_env.records)
        training_shield_diagnostics = model.shield_diagnostics()
        adaptive_diagnostics = model.adaptive_diagnostics()
        model.save(run_dir / "model.zip")
    finally:
        train_env.close()
        curve_logger.close()

    eval_env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        if args.evaluation_policy == "shielded":
            eval_shield = Shield(mask, seed=args.seed)
            eval_records = evaluate_shielded_policy(
                model,
                eval_env,
                eval_shield,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = eval_shield.diagnostics()
            eval_action_safety = {
                "proposed_action_checks": int(eval_shield_diagnostics["checked"]),
                "unsafe_proposed_action_count": int(eval_shield_diagnostics["overridden"]),
                "unsafe_proposed_action_percentage": (
                    100.0 * float(eval_shield_diagnostics["overridden"]) / float(eval_shield_diagnostics["checked"])
                    if int(eval_shield_diagnostics["checked"])
                    else 0.0
                ),
            }
        else:
            eval_records, eval_action_safety = evaluate_unshielded_policy(
                model,
                eval_env,
                mask,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = None
    finally:
        eval_env.close()

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "max_episode_steps": args.max_episode_steps,
        "shield_path": str(args.shield_path),
        "shield_source": args.shield_source,
        "shield_key": args.shield_key,
        "risk_threshold": args.risk_threshold,
        "shield_action_storage": args.shield_action_storage,
        "shield_shape": list(mask.shape),
        "base_policy_path": str(args.base_policy_path),
        "base_policy_architecture": architecture,
        "policy_kwargs": {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": "Tanh",
        },
        "adaptive": {
            "granularity": args.adaptive_granularity,
            "unsafe_update_strategy": args.unsafe_update_strategy,
            "rashomon_n_iters": int(args.rashomon_n_iters),
            "rashomon_checkpoint": args.rashomon_checkpoint,
            "rashomon_batch_size": int(args.rashomon_batch_size),
            "certificate_samples": args.certificate_samples,
            "rashomon_inverse_temperature": args.rashomon_inverse_temp,
        },
        "cost_limit": float(args.cost_limit),
        "total_timesteps": int(args.total_timesteps),
        "training_hyperparameters": {
            "learning_rate": float(args.learning_rate),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
        },
        "eval_episodes": int(args.eval_episodes),
        "evaluation_policy": args.evaluation_policy,
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "early_stop_eval_policy": args.early_stop_eval_policy,
        "success_reward_threshold": float(args.success_reward_threshold),
        "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
        "learning_curve_dir": str(curve_logger.curve_dir),
        "curve_eval_freq": int(curve_eval_freq),
        "curve_eval_episodes": int(args.curve_eval_episodes),
        "seed": int(args.seed),
    }
    write_json(run_dir / "config.json", config)

    _write_csv(
        run_dir / "training_episodes.csv",
        _training_rows(training_records),
        [
            "algorithm",
            "episode",
            "end_timestep",
            "reward",
            "cost",
            "length",
            "violated",
            "unsafe_state_visit_count",
            "safe_trajectory",
        ],
    )
    _write_csv(
        run_dir / "episodes.csv",
        _episode_rows(eval_records),
        [
            "algorithm",
            "episode",
            "reward",
            "cost",
            "length",
            "violated",
            "unsafe_state_visit_count",
            "safe_trajectory",
        ],
    )
    _write_csv(
        run_dir / "executed_unsafe_actions.csv",
        executed_action_records,
        ["episode", "episode_step", "global_step", "state", "executed_action", "unsafe_executed_action"],
    )
    _write_csv(
        run_dir / "early_stop_evaluations.csv",
        early_stop.evaluations,
        ["timesteps", "episodes", "success_count", "success_rate", "mean_reward", "eval_policy"],
    )

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "final_timesteps": int(model.num_timesteps),
        "total_exploration_steps": int(model.num_timesteps),
        "unsafe_proposed_actions_during_exploration": int(curve_logger.cumulative_unsafe),
        "unshielded_eval_unsafe_action_count": (
            0 if final_curve_evaluation is None else int(final_curve_evaluation.get("unsafe_proposed_action_count", 0))
        ),
        "unshielded_eval_safety_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("safety_rate", 0.0))
        ),
        "unshielded_eval_success_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("success_rate", 0.0))
        ),
        "unshielded_eval_mean_total_reward": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("mean_total_reward", 0.0))
        ),
        "early_stop_triggered": bool(early_stop.stop_triggered),
        "last_early_stop_evaluation": early_stop.evaluations[-1] if early_stop.evaluations else None,
        "training": aggregate_training_violations(training_records),
        "evaluation": aggregate_violations(_records_to_metrics(eval_records)),
        "evaluation_policy": args.evaluation_policy,
        "evaluation_proposed_action_safety": eval_action_safety,
        "executed_action_safety": executed_action_diagnostics,
        "training_shield_diagnostics": training_shield_diagnostics,
        "evaluation_shield_diagnostics": eval_shield_diagnostics,
        "adaptive_diagnostics": adaptive_diagnostics,
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "unshielded_reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "metrics.json",
        summarise_evaluation(
            eval_records,
            success_reward_threshold=float(args.success_reward_threshold),
            cost_limit=float(args.cost_limit),
            algorithm=ALGORITHM_NAME,
            success_mode=success_mode_for_env(getattr(args, "env_id", None)),
        ),
    )
    log_info(
        "[{algorithm}] executed unsafe actions: {unsafe}/{checked} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            unsafe=executed_action_diagnostics["executed_unsafe_action_count"],
            checked=executed_action_diagnostics["executed_action_checks"],
            pct=executed_action_diagnostics["executed_unsafe_action_percentage"],
        )
    )
    log_info(
        "[{algorithm}] adaptive updates: {accepted}/{checked} accepted without Rashomon, "
        "{rashomon} Rashomon computations, {projections} projections, {reverts} reverts".format(
            algorithm=ALGORITHM_NAME,
            accepted=adaptive_diagnostics["accepted_without_rashomon"],
            checked=adaptive_diagnostics["verifications_run"],
            rashomon=adaptive_diagnostics["rashomon_computations"],
            projections=adaptive_diagnostics["projections_applied"],
            reverts=adaptive_diagnostics["fallback_reverts"],
        )
    )
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
