"""Train PSPO with configurable adaptive safety-enforcement behavior.

``--verify-first true`` selects verify-then-project behavior; the default
region-first behavior computes a certified parameter region before accepting
or projecting each enforced candidate. Enforcement can happen after every
optimizer update, after every N PPO rollouts, or against one fixed initial
region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation import (  # noqa: E402
    AdaptiveSafePPO,
    AdaptiveSafePPOV2,
    Shield,
)

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
    env_kwargs_with_state_representation,
    policy_kwargs_from_base_architecture,
)
from projects.safe_policy_optimisation.utils.cli import (  # noqa: E402
    add_ppo_hyperparameter_args,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402
from projects.safe_policy_optimisation.utils.metrics import (  # noqa: E402
    success_mode_for_env,
    summarise_evaluation,
)
from projects.safe_policy_optimisation.utils.pspo_defaults import (  # noqa: E402
    RASHOMON_CHECKPOINT,
    RASHOMON_N_ITERS,
    environment_defaults,
)
from projects.safe_policy_optimisation.utils.rashomon import (  # noqa: E402
    parse_rashomon_batch_size,
    resolve_rashomon_batch_size,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    aggregate_training_violations,
    aggregate_violations,
)

ALGORITHM_NAME = "pspo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "pspo_policy"
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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_frequency(value: str) -> tuple[str, int, bool]:
    normalized = value.strip().lower()
    if normalized in {"update", "gradient_step"}:
        return "gradient_step", 1, False
    if normalized in {"rollout", "train_phase"}:
        return "train_phase", 1, False
    if normalized == "once":
        return "gradient_step", 1, True
    try:
        interval = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--freq must be update, rollout, once, or a positive rollout count"
        ) from exc
    if interval <= 0:
        raise argparse.ArgumentTypeError("numeric --freq must be positive")
    return "train_phase", interval, False


def _policy_initialisation_seconds(base_policy_path: Path) -> tuple[float | None, str]:
    """Return the wall time spent fitting the base policy, and where it came from.

    The base policy is fitted once per environment and reused by every seed, so
    this number is shared across a seed sweep rather than incurred per seed.
    Runs whose base policy predates the timing instrumentation return ``None``.
    """

    summary_path = Path(base_policy_path).parent / "summary.json"
    if not summary_path.exists():
        return None, "missing"
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return None, "unreadable"
    seconds = (payload.get("timing") or {}).get("policy_initialisation_wall_time_s")
    if not isinstance(seconds, (int, float)):
        return None, "not_recorded"
    return float(seconds), str(summary_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PSPO from a saved shield and safe base policy.",
    )
    parser.add_argument("--base-policy-path", type=Path, required=True)
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to gym.make.")
    parser.add_argument(
        "--state-representation",
        choices=("one_hot", "features", "state_id_lookup"),
        default="one_hot",
        help=(
            "Observation representation used by the policy. Defaults to one-hot "
            "encoding of discrete state indices."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--shield-key", default="shield")
    parser.add_argument("--shield-source", choices=("shield", "action_risk"), default="shield")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--shield-action-storage", choices=("proposed", "executed"), default="proposed")
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=environment_defaults(None).total_timesteps,
        help="Defaults to the best-known budget for a recognised environment.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adaptive-granularity",
        choices=("gradient_step", "train_phase"),
        default="gradient_step",
        help="Deprecated alias for --freq update/rollout.",
    )
    parser.add_argument(
        "--verify-first",
        type=_parse_bool,
        default=False,
        metavar="BOOL",
        help=(
            "true selects verify-then-project behavior; false (default) computes "
            "a safe region before accepting/projecting each enforced update."
        ),
    )
    parser.add_argument(
        "--region-refresh",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "Whether to recompute LIDs adaptively or construct one fixed LID "
            "before PPO while retaining --freq as the enforcement cadence."
        ),
    )
    parser.add_argument(
        "--audit-candidates-exactly",
        action="store_true",
        help=(
            "Exactly audit region-first proposals without changing the decision; "
            "diagnostic audit time is reported separately."
        ),
    )
    parser.add_argument(
        "--freq",
        default=environment_defaults(None).frequency,
        help=(
            "Safety-enforcement frequency: update, rollout, once, or a positive "
            "integer meaning every N rollouts. Defaults to the best-known "
            "environment setting."
        ),
    )
    parser.add_argument(
        "--directional",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Grow safe regions only toward the proposed policy update (default: true).",
    )
    parser.add_argument(
        "--region-mode",
        choices=("replace", "union"),
        default="replace",
        help="Replace the previous region or retain the union of certified regions.",
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
        "--n-iters",
        "--rashomon-n-iters",
        dest="rashomon_n_iters",
        type=int,
        default=RASHOMON_N_ITERS,
        help="Maximum optimization budget for each safe-region computation.",
    )
    parser.add_argument(
        "--rashomon-total-iters",
        dest="rashomon_total_iters",
        type=int,
        default=None,
        help=(
            "Total safe-region optimization budget shared across every "
            "computation in the run. Supplying it switches the budget mode "
            "from per-computation to total."
        ),
    )
    parser.add_argument(
        "--rashomon-initial-n-iters",
        dest="rashomon_initial_n_iters",
        type=int,
        default=None,
        help=(
            "Budget for the first safe-region computation. Defaults to "
            "--n-iters, or to --rashomon-total-iters in total budget mode."
        ),
    )
    parser.add_argument(
        "--rashomon-recompute-n-iters",
        dest="rashomon_recompute_n_iters",
        type=int,
        default=None,
        help=(
            "Budget for each safe-region recomputation after the first. "
            "Defaults to --rashomon-initial-n-iters."
        ),
    )
    parser.add_argument(
        "--rashomon-checkpoint",
        type=int,
        default=RASHOMON_CHECKPOINT,
        help="Engine checkpoint cadence.",
    )
    parser.add_argument(
        "--rashomon-batch-size",
        type=parse_rashomon_batch_size,
        default="auto",
        help=(
            "Safe-behaviour optimisation batch size. 'auto' (default) uses the "
            "entire safe-behaviour demonstration dataset; a positive integer "
            "requests an explicit size."
        ),
    )
    parser.add_argument(
        "--certificate-samples",
        type=int,
        default=None,
        help=(
            "Certificate batch size. Defaults to all states with a safe action "
            "(exhaustive). Ignored by --safe-region-shape segment, which always "
            "evaluates the whole certificate set."
        ),
    )
    parser.add_argument(
        "--rashomon-inverse-temp",
        type=int,
        default=None,
        help="Fixed inverse temperature for the Rashomon surrogate. Defaults to per-call calibration.",
    )
    parser.add_argument(
        "--rashomon-multi-label-mode",
        choices=("any", "all"),
        default="all",
        help=(
            "Admissible-set certificate/surrogate used for adaptive Rashomon boxes. "
            "'any' requires at least one safe action logit to beat every unsafe "
            "action logit. 'all' requires every safe action logit to beat every "
            "unsafe action logit."
        ),
    )
    parser.add_argument(
        "--surrogate",
        "--rashomon-surrogate",
        dest="rashomon_surrogate",
        choices=("auto", "probability", "logsumexp"),
        default="logsumexp",
        help=(
            "Soft constraint used for adaptive safe regions. 'probability' and "
            "'logsumexp' both support all-safe-vs-all-unsafe semantics; 'auto' "
            "preserves the historical per-mode formula."
        ),
    )
    parser.add_argument(
        "--rashomon-objective",
        choices=("weighted_width", "projection_distance"),
        default="weighted_width",
        help=(
            "Certified-region growth objective. 'weighted_width' preserves the "
            "historical update-magnitude-weighted width objective; "
            "'projection_distance' minimizes the proposed policy's normalized "
            "squared L2 distance to the certified box."
        ),
    )
    parser.add_argument(
        "--safe-region-shape",
        choices=("orthotope", "zonotope", "segment"),
        default="orthotope",
        help=(
            "Certified region geometry. 'segment' is the rank-one zonotope "
            "spanning [last safe params, proposal]: the longest certified step "
            "towards the proposed update."
        ),
    )
    parser.add_argument(
        "--zonotope-rank",
        type=int,
        default=None,
        help="Number of learned zonotope generator directions. Defaults to min(16, n_actor_params).",
    )
    parser.add_argument(
        "--segment-tolerance",
        type=float,
        default=1e-3,
        help="Bisection tolerance on the certified step length alpha.",
    )
    parser.add_argument(
        "--segment-splits",
        type=int,
        default=4,
        help="Sub-segments per certificate evaluation (higher = tighter, linear cost).",
    )
    parser.add_argument(
        "--segment-max-splits",
        type=int,
        default=8,
        help="Split cap when refining a segment that failed to certify.",
    )
    add_ppo_hyperparameter_args(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early-stop-eval-freq", type=int, default=0)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    task_defaults = environment_defaults(args.env_id)
    explicit_total_timesteps = any(
        token == "--total-timesteps" or token.startswith("--total-timesteps=")
        for token in raw_argv
    )
    if not explicit_total_timesteps:
        args.total_timesteps = task_defaults.total_timesteps
    explicit_freq = any(
        token == "--freq" or token.startswith("--freq=") for token in raw_argv
    )
    explicit_legacy_granularity = any(
        token == "--adaptive-granularity"
        or token.startswith("--adaptive-granularity=")
        for token in raw_argv
    )
    if explicit_freq and explicit_legacy_granularity:
        parser.error("--freq and --adaptive-granularity cannot be combined.")
    frequency_value = (
        args.adaptive_granularity
        if explicit_legacy_granularity
        else str(args.freq if explicit_freq else task_defaults.frequency)
    )
    try:
        granularity, interval, compute_once = _parse_frequency(str(frequency_value))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if int(args.rashomon_n_iters) <= 0:
        parser.error("--n-iters must be positive.")
    if args.directional and args.safe_region_shape == "zonotope":
        parser.error(
            "--directional true requires --safe-region-shape orthotope or segment."
        )
    if compute_once and args.verify_first:
        parser.error("--freq once requires --verify-first false.")
    if compute_once and args.directional:
        parser.error("--freq once requires --directional false.")
    if args.region_refresh == "fixed" and args.verify_first:
        parser.error("--region-refresh fixed requires --verify-first false.")
    if args.region_refresh == "fixed" and args.directional:
        parser.error("--region-refresh fixed requires --directional false.")
    if args.audit_candidates_exactly and args.verify_first:
        parser.error("--audit-candidates-exactly is only valid for region-first runs.")
    if not args.verify_first and args.unsafe_update_strategy != "rashomon_project":
        parser.error("Region-first PSPO does not support monitor-only correction.")

    args.freq = "once" if compute_once else (
        "update" if granularity == "gradient_step" else str(interval)
    )
    args.adaptive_granularity = granularity
    args.adaptive_frequency = int(interval)
    args.compute_region_once = bool(compute_once or args.region_refresh == "fixed")
    args.directional_rashomon_growth = bool(args.directional)
    args.stop_when_proposal_contained = bool(
        args.directional and args.region_refresh == "adaptive"
    )
    args.region_update_mode = str(args.region_mode)
    args.rashomon_budget_mode = (
        "total" if args.rashomon_total_iters is not None else "per_computation"
    )
    if args.rashomon_total_iters is not None:
        args.rashomon_total_iters = int(args.rashomon_total_iters)
        if args.rashomon_total_iters <= 0:
            parser.error("--rashomon-total-iters must be positive.")
    if args.rashomon_initial_n_iters is None:
        args.rashomon_initial_n_iters = int(
            args.rashomon_total_iters
            if args.rashomon_total_iters is not None
            else args.rashomon_n_iters
        )
    else:
        args.rashomon_initial_n_iters = int(args.rashomon_initial_n_iters)
        if args.rashomon_initial_n_iters <= 0:
            parser.error("--rashomon-initial-n-iters must be positive.")
    if args.rashomon_recompute_n_iters is None:
        args.rashomon_recompute_n_iters = int(args.rashomon_initial_n_iters)
    else:
        args.rashomon_recompute_n_iters = int(args.rashomon_recompute_n_iters)
        if args.rashomon_recompute_n_iters <= 0:
            parser.error("--rashomon-recompute-n-iters must be positive.")
    if args.rashomon_total_iters is not None:
        if args.rashomon_initial_n_iters > args.rashomon_total_iters:
            parser.error(
                "--rashomon-initial-n-iters cannot exceed --rashomon-total-iters."
            )
        if args.rashomon_recompute_n_iters > args.rashomon_total_iters:
            parser.error(
                "--rashomon-recompute-n-iters cannot exceed --rashomon-total-iters."
            )
    train_phases = int(math.ceil(float(args.total_timesteps) / float(args.n_steps)))
    if granularity == "gradient_step":
        minibatches = int(math.ceil(float(args.n_steps) / float(args.batch_size)))
        args.rashomon_max_region_computations = int(
            train_phases * int(args.n_epochs) * minibatches
        )
    else:
        args.rashomon_max_region_computations = max(
            1, int(math.ceil(float(train_phases) / float(interval)))
        )
    args.algorithm_name = ALGORITHM_NAME
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_started = time.perf_counter()
    algorithm_name = getattr(args, "algorithm_name", ALGORITHM_NAME)
    verify_first = bool(
        getattr(
            args,
            "verify_first",
            getattr(args, "adaptive_version", "v1") != "v2",
        )
    )
    if args.env_id is None:
        raise ValueError("--env-id is required for adaptive safe policy training.")
    env_kwargs = env_kwargs_with_state_representation(args)
    mask = load_shield_mask(
        args.shield_path,
        shield_key=args.shield_key,
        source=args.shield_source,
        risk_threshold=args.risk_threshold,
    )
    rashomon_batch_size = resolve_rashomon_batch_size(
        args.rashomon_batch_size,
        mask,
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
        lookup_tag = "state_id_lookup_discrete_observation"
        artifact_representation = str(architecture.get("state_representation", ""))
        if (
            args.state_representation == "state_id_lookup"
            or artifact_representation == lookup_tag
        ) and artifact_representation != lookup_tag:
            raise ValueError(
                "state_id_lookup training requires a base-policy artifact built "
                f"with that representation; got {artifact_representation!r}."
            )
        if (
            artifact_representation == lookup_tag
            and args.state_representation != "state_id_lookup"
        ):
            raise ValueError(
                "A state_id_lookup base-policy artifact must be trained with "
                "--state-representation state_id_lookup."
            )

        model_cls = AdaptiveSafePPO if verify_first else AdaptiveSafePPOV2
        adaptive_kwargs: dict[str, Any] = {}
        if not verify_first:
            adaptive_kwargs.update(
                {
                    "region_update_mode": getattr(args, "region_update_mode", "replace"),
                    "rashomon_budget_mode": getattr(
                        args, "rashomon_budget_mode", "per_computation"
                    ),
                    "rashomon_total_iters": getattr(args, "rashomon_total_iters", None),
                    "rashomon_initial_n_iters": getattr(
                        args, "rashomon_initial_n_iters", args.rashomon_n_iters
                    ),
                    "rashomon_recompute_n_iters": getattr(
                        args, "rashomon_recompute_n_iters", args.rashomon_n_iters
                    ),
                    "rashomon_max_region_computations": getattr(
                        args, "rashomon_max_region_computations", None
                    ),
                    "compute_region_once": getattr(args, "compute_region_once", False),
                    "audit_candidates_exactly": getattr(
                        args, "audit_candidates_exactly", False
                    ),
                }
            )
        training_started = time.perf_counter()
        curve_logger.start_timing()
        model = model_cls(
            "MlpPolicy",
            train_env,
            shield=mask,
            obs_to_state=train_env.unwrapped.make_obs_to_state(),
            state_to_features=state_to_features,
            discrete_state_representation=(
                "state_id_lookup"
                if args.state_representation == "state_id_lookup"
                else "one_hot"
            ),
            shield_seed=args.seed,
            shield_action_storage=args.shield_action_storage,
            base_policy_state_dict=base_policy_state_dict,
            adaptive_granularity=args.adaptive_granularity,
            adaptive_frequency=getattr(args, "adaptive_frequency", 1),
            unsafe_update_strategy=args.unsafe_update_strategy,
            rashomon_n_iters=args.rashomon_n_iters,
            rashomon_checkpoint=args.rashomon_checkpoint,
            rashomon_batch_size=rashomon_batch_size,
            rashomon_certificate_samples=args.certificate_samples,
            rashomon_inverse_temperature=args.rashomon_inverse_temp,
            rashomon_multi_label_mode=args.rashomon_multi_label_mode,
            rashomon_surrogate=args.rashomon_surrogate,
            rashomon_objective=args.rashomon_objective,
            safe_region_shape=args.safe_region_shape,
            zonotope_rank=args.zonotope_rank,
            segment_tolerance=args.segment_tolerance,
            segment_splits=args.segment_splits,
            segment_max_splits=args.segment_max_splits,
            rashomon_seed=args.seed,
            directional_rashomon_growth=getattr(
                args, "directional_rashomon_growth", getattr(args, "directional", True)
            ),
            stop_when_proposal_contained=getattr(
                args, "stop_when_proposal_contained", True
            ),
            **adaptive_kwargs,
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
        log_info(f"[{algorithm_name}] training for up to {args.total_timesteps} timesteps")
        curve_logger.start_timing(
            lambda: float(getattr(model, "_safety_enforcement_wall_time_s", 0.0))
        )
        model.learn(total_timesteps=args.total_timesteps, callback=[reward_curve, early_stop])
        model.finalize_adaptive_update()
        training_wall_time_s = time.perf_counter() - training_started
        final_exact_all_state_alignment = float(model._greedy_safe_rate_now())
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(train_env.episodes)
        executed_action_diagnostics = train_env.diagnostics()
        executed_action_records = list(train_env.records)
        training_shield_diagnostics = model.shield_diagnostics()
        adaptive_diagnostics = model.adaptive_diagnostics()
        adaptive_diagnostics["method"] = ALGORITHM_NAME
        adaptive_diagnostics["verify_first"] = verify_first
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
            # Same contract as the shielded branch: the audit goes through the
            # Shield interface, not the bare mask.
            eval_records, eval_action_safety = evaluate_unshielded_policy(
                model,
                eval_env,
                Shield(mask, seed=args.seed),
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = None
    finally:
        eval_env.close()

    config = {
        "algorithm": algorithm_name,
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
        "base_policy_sha256": _file_sha256(args.base_policy_path),
        "shield_sha256": _file_sha256(args.shield_path),
        "base_policy_architecture": architecture,
        "policy_kwargs": {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": "Tanh",
        },
        "adaptive": {
            "method": ALGORITHM_NAME,
            "verify_first": verify_first,
            "frequency": getattr(args, "freq", args.adaptive_granularity),
            "granularity": args.adaptive_granularity,
            "rollout_interval": int(getattr(args, "adaptive_frequency", 1)),
            "compute_region_once": bool(getattr(args, "compute_region_once", False)),
            "region_refresh": str(getattr(args, "region_refresh", "adaptive")),
            "audit_candidates_exactly": bool(
                getattr(args, "audit_candidates_exactly", False)
            ),
            "unsafe_update_strategy": args.unsafe_update_strategy,
            "rashomon_n_iters": int(args.rashomon_n_iters),
            "rashomon_checkpoint": args.rashomon_checkpoint,
            "rashomon_batch_size_setting": args.rashomon_batch_size,
            "rashomon_batch_size": int(rashomon_batch_size),
            "certificate_samples": args.certificate_samples,
            "rashomon_inverse_temperature": args.rashomon_inverse_temp,
            "rashomon_multi_label_mode": args.rashomon_multi_label_mode,
            "rashomon_surrogate": args.rashomon_surrogate,
            "rashomon_objective": args.rashomon_objective,
            "rashomon_resolved_surrogate": adaptive_diagnostics.get(
                "rashomon_resolved_surrogate"
            ),
            "safe_region_shape": args.safe_region_shape,
            "zonotope_rank": args.zonotope_rank,
            "segment_tolerance": args.segment_tolerance,
            "segment_splits": args.segment_splits,
            "segment_max_splits": args.segment_max_splits,
            "directional_rashomon_growth": getattr(
                args, "directional_rashomon_growth", getattr(args, "directional", True)
            ),
            "stop_when_proposal_contained": getattr(
                args, "stop_when_proposal_contained", True
            ),
            **(
                {
                    "region_update_mode": getattr(args, "region_update_mode", "replace"),
                    "rashomon_budget_mode": getattr(
                        args, "rashomon_budget_mode", "per_computation"
                    ),
                    "rashomon_total_iters": getattr(args, "rashomon_total_iters", None),
                    "rashomon_initial_n_iters": getattr(
                        args, "rashomon_initial_n_iters", args.rashomon_n_iters
                    ),
                    "rashomon_recompute_n_iters": getattr(
                        args, "rashomon_recompute_n_iters", args.rashomon_n_iters
                    ),
                    "rashomon_max_region_computations": getattr(
                        args, "rashomon_max_region_computations", None
                    ),
                }
                if not verify_first
                else {}
            ),
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
        run_dir / "safety_update_events.csv",
        adaptive_diagnostics.get("safety_update_events", []),
        [
            "update",
            "timestep",
            "exact_safe",
            "decision",
            "accepted_unchanged",
            "false_negative",
            "projection_displacement_l2",
            "lid_iterations",
            "exact_verification_s",
            "lid_s",
            "projection_s",
            "safety_enforcement_s",
            "diagnostic_audit_s",
        ],
    )
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
        "algorithm": algorithm_name,
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
        "final_exact_all_state_alignment": final_exact_all_state_alignment,
        "timing": {
            "training_wall_time_s": float(training_wall_time_s),
            "exact_verification_s": float(
                adaptive_diagnostics.get("exact_verification_wall_time_total_s", 0.0)
            ),
            "lid_complete_s": float(
                adaptive_diagnostics.get("rashomon_wall_time_total_s", 0.0)
            ),
            "projection_s": float(
                adaptive_diagnostics.get("projection_wall_time_total_s", 0.0)
            ),
            "safety_enforcement_s": float(
                adaptive_diagnostics.get("safety_enforcement_wall_time_total_s", 0.0)
            ),
            "diagnostic_audit_s": float(
                adaptive_diagnostics.get("diagnostic_audit_wall_time_total_s", 0.0)
            ),
            "decision_path_s": float(
                adaptive_diagnostics.get("rashomon_wall_time_total_s", 0.0)
                + adaptive_diagnostics.get("projection_wall_time_total_s", 0.0)
                + (
                    adaptive_diagnostics.get("exact_verification_wall_time_total_s", 0.0)
                    if verify_first
                    else 0.0
                )
            ),
        },
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "unshielded_reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    policy_initialisation_s, policy_initialisation_source = _policy_initialisation_seconds(
        args.base_policy_path
    )
    rl_stage_wall_time_s = float(time.perf_counter() - stage_started)
    write_json(
        run_dir / "training_time.json",
        {
            "run_id": run_id,
            "seed": int(getattr(args, "seed", -1)),
            "algorithm": algorithm_name,
            "policy_initialisation_s": policy_initialisation_s,
            "policy_initialisation_source": policy_initialisation_source,
            "policy_initialisation_shared_across_seeds": True,
            "rl_training_s": float(training_wall_time_s),
            "rl_stage_wall_time_s": rl_stage_wall_time_s,
            "total_s": (
                None
                if policy_initialisation_s is None
                else float(policy_initialisation_s + rl_stage_wall_time_s)
            ),
            "field_notes": {
                "policy_initialisation_s": (
                    "Wall time to fit the shared base policy (behaviour cloning "
                    "against the shield). Fitted once per environment and reused "
                    "by every seed, so it is not a per-seed cost."
                ),
                "rl_training_s": "model.learn() only, excluding final evaluation.",
                "rl_stage_wall_time_s": (
                    "Whole RL stage: setup, model.learn(), and final evaluation."
                ),
                "total_s": "policy_initialisation_s + rl_stage_wall_time_s.",
            },
        },
    )
    write_json(
        run_dir / "metrics.json",
        summarise_evaluation(
            eval_records,
            success_reward_threshold=float(args.success_reward_threshold),
            cost_limit=float(args.cost_limit),
            algorithm=algorithm_name,
            success_mode=success_mode_for_env(getattr(args, "env_id", None)),
        ),
    )
    log_info(
        "[{algorithm}] executed unsafe actions: {unsafe}/{checked} ({pct:.2f}%)".format(
            algorithm=algorithm_name,
            unsafe=executed_action_diagnostics["executed_unsafe_action_count"],
            checked=executed_action_diagnostics["executed_action_checks"],
            pct=executed_action_diagnostics["executed_unsafe_action_percentage"],
        )
    )
    log_info(
        "[{algorithm}] adaptive enforcement: verify_first={verify_first}, "
        "verifications={checked}, region_computations={regions}, "
        "projections={projections}, reverts={reverts}, final_flushes={flushes}".format(
            algorithm=algorithm_name,
            verify_first=verify_first,
            checked=adaptive_diagnostics["verifications_run"],
            regions=adaptive_diagnostics["rashomon_computations"],
            projections=adaptive_diagnostics["projections_applied"],
            reverts=adaptive_diagnostics["fallback_reverts"],
            flushes=adaptive_diagnostics.get("final_flushes", 0),
        )
    )
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
