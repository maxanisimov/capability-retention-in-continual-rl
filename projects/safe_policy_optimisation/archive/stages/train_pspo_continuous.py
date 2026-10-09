"""Train interval-certified PSPO in a continuous-state environment.

Supports ``MountainCar-v0`` and ``LunarLander-v3``.  It follows
``train_pspo.py``'s adaptive update logic, but replaces exhaustive enumeration
of a finite state table with sound verification of the complete safety-critical
input box(es).  For MountainCar that is ``[-1.2, -1.0] x [-0.07, 0.0]``; for
LunarLander it is the descent band ``y <= 0.6``, ``v_y <= -0.35`` spanning the
declared range of the six remaining observation coordinates. The viewport
shield uses two boxes, ``x in [-1,-m]`` and ``x in [m,1]``, spanning all other
declared coordinates. The descent and MountainCar boxes are
closed extensions of open runtime conditions, so each certificate covers a
superset of the states its shield actually constrains.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import TensorDataset

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from continuous_state_shields import (  # noqa: E402
    LunarLanderDescentShield,
    LunarLanderViewportShield,
    MountainCarIntervalBoxShield,
)
from provably_safe_policy_optimisation import (  # noqa: E402
    AdaptiveSafePPO,
    AdaptiveSafePPOV2,
    as_action_shield,
)
from provably_safe_policy_optimisation.adaptive_safe_ppo import (  # noqa: E402
    validate_interval_certificate_dataset,
)

from projects.safe_policy_optimisation.stages.train_ppo_shield import (  # noqa: E402
    _records_to_metrics,
    evaluate_shielded_policy,
    evaluate_unshielded_policy,
    make_continuous_state_shield,
    make_unshielded_env,
    validate_shield_for_env,
)
from projects.safe_policy_optimisation.stages.train_pspo import (  # noqa: E402
    _file_sha256,
    _parse_bool,
    _parse_frequency,
    base_state_dict_to_ppo_actor,
    load_base_policy_payload,
)
from projects.safe_policy_optimisation.stages.train_pspo_precomputed import (  # noqa: E402
    _resolve_curve_eval_freq,
    _write_csv,
    policy_kwargs_from_base_architecture,
)
from projects.safe_policy_optimisation.utils import io  # noqa: E402
from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs  # noqa: E402
from projects.safe_policy_optimisation.utils.episode_recording import (  # noqa: E402
    EpisodeRecorderWrapper,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
    episode_success,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402
from projects.safe_policy_optimisation.utils.metrics import (  # noqa: E402
    success_mode_for_env,
    summarise_evaluation,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    aggregate_training_violations,
    aggregate_violations,
)

ALGORITHM_NAME = "pspo_continuous"
MOUNTAINCAR_ENV_ID = "MountainCar-v0"
LUNARLANDER_ENV_ID = "LunarLander-v3"
SUPPORTED_ENV_ID = MOUNTAINCAR_ENV_ID
SUPPORTED_ENV_IDS = (MOUNTAINCAR_ENV_ID, LUNARLANDER_ENV_ID)
# (observation dimension, discrete action count) each environment's actor takes.
_ENV_POLICY_SHAPE = {MOUNTAINCAR_ENV_ID: (2, 3), LUNARLANDER_ENV_ID: (8, 4)}
# Default episode caps; LunarLander needs its full 1000-step budget to land.
_ENV_MAX_EPISODE_STEPS = {MOUNTAINCAR_ENV_ID: 200, LUNARLANDER_ENV_ID: 1000}
LUNARLANDER_ALTITUDE_INDEX = 1
LUNARLANDER_VERTICAL_SPEED_INDEX = 3
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "pspo_continuous"
)


class MountainCarSafetyCostWrapper(gym.Wrapper):
    """Expose physical left-wall contact as a binary state cost."""

    def __init__(
        self, env: gym.Env, *, left_boundary_position: float, tolerance: float
    ) -> None:
        super().__init__(env)
        self.left_boundary_position = float(left_boundary_position)
        self.tolerance = float(tolerance)

    def _with_cost(self, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result["cost"] = float(
            float(np.asarray(observation, dtype=float)[0])
            <= self.left_boundary_position + self.tolerance
        )
        return result

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_cost(observation, info)

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._with_cost(observation, info),
        )


class LunarLanderDescentCostWrapper(gym.Wrapper):
    """Expose an unsafe descent rate near the ground as a binary state cost.

    The unsafe set is deliberately stricter than the shield's activation band:
    the shield acts from ``critical_height``/``safe_min_vertical_speed``, while a
    violation is only recorded once the lander is inside
    ``unsafe_height``/``unsafe_vertical_speed``.  The gap between the two is the
    margin the shield needs in order to act before a violation rather than after.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        unsafe_height: float,
        unsafe_vertical_speed: float,
        tolerance: float,
    ) -> None:
        super().__init__(env)
        self.unsafe_height = float(unsafe_height)
        self.unsafe_vertical_speed = float(unsafe_vertical_speed)
        self.tolerance = float(tolerance)

    def is_unsafe(self, observation: Any) -> bool:
        values = np.asarray(observation, dtype=float)
        altitude = float(values[LUNARLANDER_ALTITUDE_INDEX])
        vertical_speed = float(values[LUNARLANDER_VERTICAL_SPEED_INDEX])
        return (
            altitude < self.unsafe_height - self.tolerance
            and vertical_speed < self.unsafe_vertical_speed - self.tolerance
        )

    def _with_cost(self, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result["cost"] = float(self.is_unsafe(observation))
        return result

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_cost(observation, info)

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation, reward, terminated, truncated,
            self._with_cost(observation, info),
        )


def lunarlander_out_of_view(env: gym.Env, observation: Any) -> bool:
    """Use the same pre-float32 x calculation as Gymnasium's termination test."""
    from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_W

    lander = getattr(env.unwrapped, "lander", None)
    if lander is None:
        return abs(float(np.asarray(observation)[0])) >= 1.0
    half_width = VIEWPORT_W / SCALE / 2
    return abs((float(lander.position.x) - half_width) / half_width) >= 1.0


class LunarLanderViewportCostWrapper(gym.Wrapper):
    """Expose crossing either x viewport boundary as cost, including final state."""

    def _with_cost(self, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
        return {**info, "cost": float(lunarlander_out_of_view(self, observation))}

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return obs, self._with_cost(obs, info)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._with_cost(obs, info)


def _action(parser: argparse.ArgumentParser, destination: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == destination)


def build_parser() -> argparse.ArgumentParser:
    """Build the discrete PSPO-compatible CLI with continuous-state inputs."""

    from projects.safe_policy_optimisation.stages.train_pspo import (
        build_parser as base_parser,
    )

    parser = base_parser()
    parser.description = (
        "Train continuous-state PSPO from an interval-certified safe initialization."
    )
    _action(parser, "shield_path").required = False
    _action(
        parser, "shield_path"
    ).help = "Not used for continuous PSPO; a built-in continuous shield is used."
    _action(parser, "env_id").default = SUPPORTED_ENV_ID
    _action(parser, "env_id").choices = SUPPORTED_ENV_IDS
    _action(parser, "state_representation").default = "features"
    _action(parser, "max_episode_steps").default = None
    _action(parser, "output_dir").default = DEFAULT_OUTPUT_DIR
    parser.add_argument(
        "--certificate-dataset",
        type=Path,
        default=None,
        help=(
            "TensorDataset(X_l, X_u, safe_mask). Defaults to "
            "critical_interval_dataset.pt beside --base-policy-path."
        ),
    )
    parser.add_argument(
        "--continuous-shield",
        choices=("auto", "mountaincar", "mountaincar-boxes", "lunarlander-descent", "lunarlander-viewport"),
        default="auto",
    )
    parser.add_argument(
        "--continuous-shield-artifact",
        type=Path,
        default=None,
        help=(
            "Required for mountaincar-boxes: boxed .npz shield produced by "
            "box_mountaincar_reach_avoid_shield.py."
        ),
    )
    parser.add_argument(
        "--continuous-shield-config",
        default=None,
        help="Optional JSON object overriding the selected shield's config.",
    )
    parser.add_argument(
        "--growth-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verifier bounding the certified region while it is grown. A tighter "
            "verifier admits larger regions at more cost per iteration."
        ),
    )
    parser.add_argument(
        "--certification-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verifier deciding whether a grown region is accepted, and the one "
            "the reported final certificate is stated under."
        ),
    )
    parser.add_argument(
        "--lunarlander-unsafe-height",
        type=float,
        default=0.5,
        help=(
            "LunarLander violation altitude. Must sit below the shield's "
            "critical_height so the shield acts before a violation (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--lunarlander-unsafe-vertical-speed",
        type=float,
        default=-0.4,
        help=(
            "LunarLander violation descent rate. Must sit below the shield's "
            "safe_min_vertical_speed (default: -0.4)."
        ),
    )
    parser.add_argument(
        "--mountaincar-shaped-reward",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Use potential-based reward shaping for training only (default: true).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI while retaining ``train_pspo.py`` frequency semantics."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.shield_path is not None:
        parser.error(
            "--shield-path is not used; select the built-in continuous shield."
        )
    if args.state_representation != "features":
        parser.error("Continuous PSPO requires --state-representation features.")
    if args.continuous_shield == "auto":
        args.continuous_shield = (
            "mountaincar"
            if args.env_id == MOUNTAINCAR_ENV_ID
            else "lunarlander-descent"
        )
    if args.env_id != MOUNTAINCAR_ENV_ID and args.continuous_shield.startswith(
        "mountaincar"
    ):
        parser.error(
            f"--continuous-shield {args.continuous_shield} is MountainCar-only; "
            f"got --env-id {args.env_id}."
        )
    if args.env_id == MOUNTAINCAR_ENV_ID and args.continuous_shield.startswith(
        "lunarlander"
    ):
        parser.error(
            f"--continuous-shield {args.continuous_shield} is LunarLander-only; "
            f"got --env-id {args.env_id}."
        )
    if args.max_episode_steps is None:
        args.max_episode_steps = _ENV_MAX_EPISODE_STEPS[args.env_id]
    if args.env_id != MOUNTAINCAR_ENV_ID and args.mountaincar_shaped_reward:
        # The flag defaults to true for MountainCar; silently ignoring it
        # elsewhere would hide a misconfigured launcher.
        parser.error(
            "--mountaincar-shaped-reward is MountainCar-only; pass false for "
            f"--env-id {args.env_id}."
        )
    if args.safe_region_shape == "zonotope":
        parser.error(
            "Continuous input intervals are not supported by the learned-zonotope "
            "safe region; use --safe-region-shape orthotope or segment."
        )

    explicit_freq = any(
        token == "--freq" or token.startswith("--freq=") for token in raw_argv
    )
    explicit_legacy = any(
        token == "--adaptive-granularity" or token.startswith("--adaptive-granularity=")
        for token in raw_argv
    )
    if explicit_freq and explicit_legacy:
        parser.error("--freq and --adaptive-granularity cannot be combined.")
    frequency = args.adaptive_granularity if explicit_legacy else str(args.freq)
    try:
        granularity, interval, compute_once = _parse_frequency(frequency)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if int(args.rashomon_n_iters) <= 0:
        parser.error("--n-iters must be positive.")
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

    args.freq = (
        "once"
        if compute_once
        else ("update" if granularity == "gradient_step" else str(interval))
    )
    args.adaptive_granularity = granularity
    args.adaptive_frequency = int(interval)
    args.compute_region_once = bool(compute_once or args.region_refresh == "fixed")
    args.directional_rashomon_growth = bool(args.directional)
    args.stop_when_proposal_contained = bool(
        args.directional and args.region_refresh == "adaptive"
    )
    args.region_update_mode = str(args.region_mode)
    if args.safe_region_shape == "segment":
        # The core raises on these combinations from inside run(); catching them
        # here turns a mid-training traceback into a clean CLI error.
        if args.compute_region_once:
            parser.error(
                "--safe-region-shape segment cannot be combined with --freq once or "
                "--region-refresh fixed: a segment has no volume, so it cannot absorb "
                "a later update in a different direction."
            )
        if args.region_update_mode == "union":
            parser.error(
                "--safe-region-shape segment cannot be combined with --region-mode "
                "union: each segment points at the proposal that produced it, so "
                "accumulating stale segments would project onto an old direction."
            )
        if args.growth_method != "IBP":
            # The segment engine calls bound_forward_pass directly rather than the
            # verifier registry, so --growth-method is silently ignored on this path
            # while --certification-method still drives the acceptance audit. Left
            # unguarded, a run would grow regions under one verifier and certify them
            # under another; see docs/crown_vs_ibp_tanh_looseness.md for why those two
            # disagree in exactly this regime.
            parser.error(
                "--safe-region-shape segment supports --growth-method IBP only; "
                f"got {args.growth_method}."
            )
    args.rashomon_budget_mode = (
        "total" if args.rashomon_total_iters is not None else "per_computation"
    )
    if args.rashomon_total_iters is not None and int(args.rashomon_total_iters) <= 0:
        parser.error("--rashomon-total-iters must be positive.")
    args.rashomon_initial_n_iters = int(
        args.rashomon_initial_n_iters
        if args.rashomon_initial_n_iters is not None
        else (
            args.rashomon_total_iters
            if args.rashomon_total_iters is not None
            else args.rashomon_n_iters
        )
    )
    args.rashomon_recompute_n_iters = int(
        args.rashomon_recompute_n_iters
        if args.rashomon_recompute_n_iters is not None
        else args.rashomon_initial_n_iters
    )
    if args.rashomon_initial_n_iters <= 0 or args.rashomon_recompute_n_iters <= 0:
        parser.error("Rashomon initial and recompute budgets must be positive.")
    if args.rashomon_total_iters is not None and (
        args.rashomon_initial_n_iters > args.rashomon_total_iters
        or args.rashomon_recompute_n_iters > args.rashomon_total_iters
    ):
        parser.error(
            "Per-region Rashomon budgets cannot exceed --rashomon-total-iters."
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


def load_interval_certificate(path: Path, env_id: str) -> TensorDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, TensorDataset):
        raise ValueError("--certificate-dataset must contain a torch TensorDataset.")
    obs_dim, n_actions = _ENV_POLICY_SHAPE[env_id]
    return validate_interval_certificate_dataset(
        payload, observation_shape=(obs_dim,), n_actions=n_actions
    )


def validate_lunarlander_certificate(dataset: TensorDataset, shield: Any) -> None:
    """Require complete descent-band or viewport-edge certificate coverage.

    The runtime band is open in altitude and vertical speed, so the certified
    box must close it from above in exactly those two coordinates and span the
    environment's declared range everywhere else.  Any box the shield does not
    constrain, or any mask admitting more than the main engine, would let PSPO
    project into parameters the runtime shield never vouched for.
    """

    if isinstance(shield, LunarLanderViewportShield):
        from continuous_state_shields.lunar_lander_viewport import (
            viewport_certificate_arrays,
        )

        env = gym.make(LUNARLANDER_ENV_ID, continuous=False)
        try:
            expected = viewport_certificate_arrays(shield.config.m, env.observation_space.low, env.observation_space.high)
        finally:
            env.close()
        for actual, wanted in zip(dataset.tensors, expected):
            if actual.shape != wanted.shape or not torch.equal(actual.cpu(), torch.as_tensor(wanted, dtype=actual.dtype)):
                raise ValueError("Viewport certificate must contain both complete edge boxes and only the inward actions")
        return
    x_l, x_u, mask = dataset.tensors
    if not isinstance(shield, LunarLanderDescentShield):
        raise ValueError(
            "LunarLander PSPO requires the lunarlander-descent shield; got "
            f"{type(shield).__name__}."
        )
    cfg = shield.config
    expected_mask = torch.zeros((len(x_l), int(cfg.n_actions)), dtype=torch.bool)
    expected_mask[:, int(cfg.main_engine_action)] = True
    if mask.shape != expected_mask.shape or not torch.equal(
        mask.cpu().bool(), expected_mask
    ):
        raise ValueError(
            "Certificate safe-action mask must make the main engine "
            f"(action {int(cfg.main_engine_action)}) the only safe action in every "
            "certified box."
        )
    altitude_upper = float(x_u[:, LUNARLANDER_ALTITUDE_INDEX].max().item())
    speed_upper = float(x_u[:, LUNARLANDER_VERTICAL_SPEED_INDEX].max().item())
    if altitude_upper > float(cfg.critical_height) + 1e-6:
        raise ValueError(
            f"Certified altitude upper bound {altitude_upper} exceeds the shield's "
            f"critical height {float(cfg.critical_height)}."
        )
    if speed_upper > float(cfg.safe_min_vertical_speed) + 1e-6:
        raise ValueError(
            f"Certified vertical-speed upper bound {speed_upper} exceeds the shield's "
            f"safe minimum {float(cfg.safe_min_vertical_speed)}."
        )
    env = gym.make(LUNARLANDER_ENV_ID, continuous=False)
    try:
        low = torch.as_tensor(env.observation_space.low, dtype=x_l.dtype)
        high = torch.as_tensor(env.observation_space.high, dtype=x_u.dtype)
    finally:
        env.close()
    if bool((x_l.cpu() < low - 1e-6).any()) or bool((x_u.cpu() > high + 1e-6).any()):
        raise ValueError(
            "Certified boxes must lie inside the declared observation space."
        )
    # Coverage: the union of the boxes must contain the whole critical band.
    # Boxes are produced by tiling one band, so checking the union's corners is
    # sufficient and keeps this validation independent of the split pattern.
    band_low = low.clone()
    band_high = high.clone()
    band_high[LUNARLANDER_ALTITUDE_INDEX] = float(cfg.critical_height)
    band_high[LUNARLANDER_VERTICAL_SPEED_INDEX] = float(cfg.safe_min_vertical_speed)
    union_low = x_l.cpu().min(dim=0).values
    union_high = x_u.cpu().max(dim=0).values
    if bool((union_low > band_low + 1e-6).any()) or bool(
        (union_high < band_high - 1e-6).any()
    ):
        raise ValueError(
            "Certified boxes do not cover the complete critical descent band: "
            f"union=[{union_low.tolist()}, {union_high.tolist()}] vs "
            f"band=[{band_low.tolist()}, {band_high.tolist()}]."
        )


def validate_continuous_certificate(
    dataset: TensorDataset, shield: Any, env_id: str
) -> None:
    """Dispatch certificate validation to the environment's contract."""

    if env_id == MOUNTAINCAR_ENV_ID:
        validate_mountaincar_certificate(dataset, shield)
        return
    validate_lunarlander_certificate(dataset, shield)


def validate_mountaincar_certificate(dataset: TensorDataset, shield: Any) -> None:
    """Require the certificate to match the runtime shield exactly."""

    x_l, x_u, mask = dataset.tensors
    if isinstance(shield, MountainCarIntervalBoxShield):
        expected_l = torch.as_tensor(shield.box_lows, dtype=x_l.dtype)
        expected_u = torch.as_tensor(shield.box_highs, dtype=x_u.dtype)
        expected_mask = torch.as_tensor(shield.safe_masks, dtype=torch.bool)
        if x_l.shape != expected_l.shape or not torch.allclose(
            x_l.cpu(), expected_l, atol=1e-7, rtol=0.0
        ):
            raise ValueError(
                "Certificate lower bounds do not match the boxed runtime shield."
            )
        if x_u.shape != expected_u.shape or not torch.allclose(
            x_u.cpu(), expected_u, atol=1e-7, rtol=0.0
        ):
            raise ValueError(
                "Certificate upper bounds do not match the boxed runtime shield."
            )
        if mask.shape != expected_mask.shape or not torch.equal(
            mask.cpu().bool(), expected_mask
        ):
            raise ValueError(
                "Certificate action masks do not match the boxed runtime shield."
            )
        return
    cfg = shield.config
    expected_l = torch.tensor([[cfg.critical_min_position, -cfg.max_speed]])
    # v < 0 is open; interval verification uses its conservative closed
    # extension through v=0 so every negative velocity is covered.
    expected_u = torch.tensor([[cfg.critical_max_position, 0.0]])
    expected_mask = torch.zeros((1, 3), dtype=torch.bool)
    expected_mask[0, cfg.push_right_action] = True
    if len(dataset) != 1:
        raise ValueError("MountainCar PSPO requires exactly one complete critical box.")
    if not torch.allclose(x_l, expected_l, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"Certificate lower bound {x_l.tolist()} does not match {expected_l.tolist()}."
        )
    if not torch.allclose(x_u, expected_u, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"Certificate upper bound {x_u.tolist()} does not match {expected_u.tolist()}."
        )
    if not torch.equal(mask.bool(), expected_mask):
        raise ValueError(
            "Certificate safe-action mask must make push-right (action 2) the only "
            "safe action throughout the critical box."
        )


def validate_base_architecture(architecture: dict[str, Any], env_id: str) -> None:
    obs_dim, n_actions = _ENV_POLICY_SHAPE[env_id]
    expected = {
        "input_dim": obs_dim,
        "n_actions": n_actions,
        "state_representation": "continuous_features",
    }
    mismatches = {
        key: {"expected": value, "actual": architecture.get(key)}
        for key, value in expected.items()
        if architecture.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Base policy is incompatible with {env_id}: {mismatches}.")


def _make_env(
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    shield: Any,
    *,
    record_episodes: bool,
) -> gym.Env:
    env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=False,
    )
    if args.env_id == MOUNTAINCAR_ENV_ID:
        env = MountainCarSafetyCostWrapper(
            env,
            left_boundary_position=shield.config.min_position,
            tolerance=shield.config.tolerance,
        )
    elif isinstance(shield, LunarLanderViewportShield):
        env = LunarLanderViewportCostWrapper(env)
    else:
        env = LunarLanderDescentCostWrapper(
            env,
            unsafe_height=float(args.lunarlander_unsafe_height),
            unsafe_vertical_speed=float(args.lunarlander_unsafe_vertical_speed),
            tolerance=float(shield.config.tolerance),
        )
    if record_episodes:
        env = EpisodeRecorderWrapper(env, cost_limit=args.cost_limit)
    return env


def _evaluate_success(
    model: Any,
    env_factory: Any,
    shield: Any,
    *,
    apply_shield: bool,
    episodes: int,
    seed: int,
    reward_threshold: float,
) -> dict[str, Any]:
    runtime_shield = as_action_shield(shield)
    rewards: list[float] = []
    successes = 0
    env = env_factory()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            infos: list[dict[str, Any]] = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                if apply_shield:
                    action_int = int(runtime_shield.shield_action(obs, action_int))
                obs, reward, terminated, truncated, info = env.step(action_int)
                total_reward += float(reward)
                infos.append(dict(info))
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            successes += int(
                episode_success(total_reward, infos, reward_threshold=reward_threshold)
            )
    finally:
        env.close()
    return {
        "episodes": int(episodes),
        "success_count": int(successes),
        "success_rate": float(successes / episodes) if episodes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "eval_policy": "shielded" if apply_shield else "unshielded",
    }


class EarlyStopOnContinuousSuccessCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_factory: Any,
        shield: Any,
        eval_policy: str,
        eval_freq: int,
        eval_episodes: int,
        success_rate: float,
        seed: int,
        reward_threshold: float,
    ) -> None:
        super().__init__()
        self.env_factory = env_factory
        self.shield = shield
        self.eval_policy = str(eval_policy)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.target_success_rate = float(success_rate)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.evaluations: list[dict[str, Any]] = []
        self.stop_triggered = False

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq:
            return True
        metrics = _evaluate_success(
            self.model,
            self.env_factory,
            self.shield,
            apply_shield=self.eval_policy == "shielded",
            episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
            reward_threshold=self.reward_threshold,
        )
        self.evaluations.append({"timesteps": int(self.num_timesteps), **metrics})
        if float(metrics["success_rate"]) >= self.target_success_rate:
            self.stop_triggered = True
            return False
        return True


def evaluate_left_boundary_safety(
    model: Any,
    env_factory: Any,
    shield: Any,
    *,
    apply_shield: bool,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Measure trajectories that never touch MountainCar's physical left wall."""

    runtime_shield = as_action_shield(shield)
    reached = 0
    minimum_positions: list[float] = []
    env = env_factory()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            minimum = float(np.asarray(obs)[0])
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                if apply_shield:
                    action_int = int(runtime_shield.shield_action(obs, action_int))
                obs, _reward, terminated, truncated, _info = env.step(action_int)
                minimum = min(minimum, float(np.asarray(obs)[0]))
                done = bool(terminated or truncated)
            minimum_positions.append(minimum)
            reached += int(
                minimum <= shield.config.min_position + shield.config.tolerance
            )
    finally:
        env.close()
    return {
        "episodes": int(episodes),
        "left_boundary_reached_episodes": int(reached),
        "left_boundary_reached_rate": float(reached / episodes) if episodes else 0.0,
        "safe_trajectory_rate": float((episodes - reached) / episodes)
        if episodes
        else 0.0,
        "minimum_position": float(min(minimum_positions))
        if minimum_positions
        else None,
        "boundary_position": float(shield.config.min_position),
    }


def evaluate_descent_safety(
    model: Any,
    env_factory: Any,
    shield: Any,
    *,
    apply_shield: bool,
    episodes: int,
    seed: int,
    unsafe_height: float,
    unsafe_vertical_speed: float,
) -> dict[str, Any]:
    """Measure trajectories that never descend too fast near the ground."""

    runtime_shield = as_action_shield(shield)
    violating = 0
    worst_speeds: list[float] = []
    env = env_factory()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            unsafe = False
            worst = 0.0
            done = False
            while not done:
                values = np.asarray(obs, dtype=float)
                if (
                    float(values[LUNARLANDER_ALTITUDE_INDEX]) < unsafe_height
                    and float(values[LUNARLANDER_VERTICAL_SPEED_INDEX])
                    < unsafe_vertical_speed
                ):
                    unsafe = True
                if float(values[LUNARLANDER_ALTITUDE_INDEX]) < unsafe_height:
                    worst = min(worst, float(values[LUNARLANDER_VERTICAL_SPEED_INDEX]))
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                if apply_shield:
                    action_int = int(runtime_shield.shield_action(obs, action_int))
                obs, _reward, terminated, truncated, _info = env.step(action_int)
                done = bool(terminated or truncated)
            values = np.asarray(obs, dtype=float)
            if (
                float(values[LUNARLANDER_ALTITUDE_INDEX]) < unsafe_height
                and float(values[LUNARLANDER_VERTICAL_SPEED_INDEX])
                < unsafe_vertical_speed
            ):
                unsafe = True
            violating += int(unsafe)
            worst_speeds.append(worst)
    finally:
        env.close()
    return {
        "episodes": int(episodes),
        "unsafe_descent_episodes": int(violating),
        "unsafe_descent_rate": float(violating / episodes) if episodes else 0.0,
        "safe_trajectory_rate": float((episodes - violating) / episodes)
        if episodes
        else 0.0,
        "worst_vertical_speed_below_unsafe_height": (
            float(min(worst_speeds)) if worst_speeds else None
        ),
        "unsafe_height": float(unsafe_height),
        "unsafe_vertical_speed": float(unsafe_vertical_speed),
    }


def evaluate_viewport_safety(
    model: Any, env_factory: Any, shield: Any, *, apply_shield: bool,
    episodes: int, seed: int,
) -> dict[str, Any]:
    """Audit raw viewport escape and preserve actual episode-ending flags.

    No truncation is not synonymous with successful landing: crashes and
    viewport exits also terminate. Keep all three quantities separate.
    """
    runtime = as_action_shield(shield)
    rows = []
    env = env_factory()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            unsafe = lunarlander_out_of_view(env, obs)
            total, length, done = 0.0, 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(action).item())
                if apply_shield:
                    action = int(runtime.shield_action(obs, action))
                obs, reward, terminated, truncated, _ = env.step(action)
                unsafe = unsafe or lunarlander_out_of_view(env, obs)
                total += float(reward)
                length += 1
                done = bool(terminated or truncated)
            raw = env.unwrapped
            crashed = bool(getattr(raw, "game_over", False))
            asleep = not bool(getattr(getattr(raw, "lander", None), "awake", True))
            rows.append({
                "episode": episode, "reward": total, "length": length,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "safe_trajectory": not unsafe, "out_of_view": unsafe,
                "crashed": crashed,
                "landed": bool(terminated and not truncated and asleep and not crashed and not unsafe),
            })
    finally:
        env.close()
    return {
        "episodes": episodes,
        "out_of_view_episodes": sum(row["out_of_view"] for row in rows),
        "safe_trajectory_rate": sum(row["safe_trajectory"] for row in rows) / episodes if episodes else 0.0,
        "no_truncation_rate": sum(not row["truncated"] for row in rows) / episodes if episodes else 0.0,
        "successful_landing_rate": sum(row["landed"] for row in rows) / episodes if episodes else 0.0,
        "crash_rate": sum(row["crashed"] for row in rows) / episodes if episodes else 0.0,
        "unsafe_event": "abs(normalised raw x) >= 1", "per_episode": rows,
    }


def evaluate_trajectory_safety(
    args: argparse.Namespace,
    model: Any,
    env_factory: Any,
    shield: Any,
    *,
    apply_shield: bool,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run the environment's physical trajectory-safety audit."""

    if args.env_id == MOUNTAINCAR_ENV_ID:
        return evaluate_left_boundary_safety(
            model,
            env_factory,
            shield,
            apply_shield=apply_shield,
            episodes=episodes,
            seed=seed,
        )
    if isinstance(shield, LunarLanderViewportShield):
        return evaluate_viewport_safety(
            model, env_factory, shield, apply_shield=apply_shield,
            episodes=episodes, seed=seed,
        )
    return evaluate_descent_safety(
        model,
        env_factory,
        shield,
        apply_shield=apply_shield,
        episodes=episodes,
        seed=seed,
        unsafe_height=float(args.lunarlander_unsafe_height),
        unsafe_vertical_speed=float(args.lunarlander_unsafe_vertical_speed),
    )


def _write_early_stop_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timesteps",
                "episodes",
                "success_count",
                "success_rate",
                "mean_reward",
                "eval_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id not in SUPPORTED_ENV_IDS:
        supported = ", ".join(SUPPORTED_ENV_IDS)
        raise ValueError(f"Continuous PSPO supports only {supported}.")
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.continuous_shield == "mountaincar-boxes":
        if args.continuous_shield_artifact is None:
            raise ValueError(
                "--continuous-shield-artifact is required for mountaincar-boxes."
            )
        if args.continuous_shield_config is not None:
            raise ValueError(
                "--continuous-shield-config cannot be combined with "
                "mountaincar-boxes."
            )
        shield = MountainCarIntervalBoxShield.load(
            Path(args.continuous_shield_artifact)
        )
    else:
        if args.continuous_shield_artifact is not None:
            raise ValueError(
                "--continuous-shield-artifact is only valid for mountaincar-boxes."
            )
        shield = make_continuous_state_shield(
            args.continuous_shield, args.env_id, args.continuous_shield_config
        )
    certificate_path = Path(
        args.certificate_dataset
        if args.certificate_dataset is not None
        else Path(args.base_policy_path).parent / "critical_interval_dataset.pt"
    )
    certificate_dataset = load_interval_certificate(certificate_path, args.env_id)
    validate_continuous_certificate(certificate_dataset, shield, args.env_id)
    architecture, base_state_dict = load_base_policy_payload(
        Path(args.base_policy_path)
    )
    validate_base_architecture(architecture, args.env_id)
    base_actor = base_state_dict_to_ppo_actor(architecture, base_state_dict)
    policy_kwargs = policy_kwargs_from_base_architecture(architecture)
    rashomon_batch_size = (
        len(certificate_dataset)
        if args.rashomon_batch_size == "auto"
        else int(args.rashomon_batch_size)
    )
    if args.certificate_samples is not None and int(args.certificate_samples) != len(
        certificate_dataset
    ):
        raise ValueError(
            "Continuous PSPO must certify every interval; --certificate-samples must "
            f"be omitted or equal {len(certificate_dataset)}."
        )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=args.tensorboard_log_dir or run_dir / "tensorboard",
    )
    curve_eval_freq = _resolve_curve_eval_freq(args)
    episode_env = _make_env(args, env_kwargs, shield, record_episodes=True)
    train_env: gym.Env = episode_env
    if args.env_id == MOUNTAINCAR_ENV_ID and args.mountaincar_shaped_reward:
        from projects.safe_crl.pipelines.envs.mountaincar.mountaincar_utils import (
            MountainCarShapedReward,
        )

        train_env = MountainCarShapedReward(train_env, gamma=float(args.gamma))
    validate_shield_for_env(shield, train_env)

    verify_first = bool(args.verify_first)
    model_cls = AdaptiveSafePPO if verify_first else AdaptiveSafePPOV2
    adaptive_kwargs: dict[str, Any] = {}
    if not verify_first:
        adaptive_kwargs = {
            "region_update_mode": args.region_update_mode,
            "rashomon_budget_mode": args.rashomon_budget_mode,
            "rashomon_total_iters": args.rashomon_total_iters,
            "rashomon_initial_n_iters": args.rashomon_initial_n_iters,
            "rashomon_recompute_n_iters": args.rashomon_recompute_n_iters,
            "rashomon_max_region_computations": args.rashomon_max_region_computations,
            "compute_region_once": args.compute_region_once,
            "audit_candidates_exactly": args.audit_candidates_exactly,
        }

    def raw_env_factory() -> gym.Env:
        return _make_env(args, env_kwargs, shield, record_episodes=False)

    try:
        model = model_cls(
            "MlpPolicy",
            train_env,
            shield=shield,
            interval_certificate_dataset=certificate_dataset,
            shield_seed=args.seed,
            shield_action_storage=args.shield_action_storage,
            base_policy_state_dict=base_actor,
            adaptive_granularity=args.adaptive_granularity,
            adaptive_frequency=args.adaptive_frequency,
            unsafe_update_strategy=args.unsafe_update_strategy,
            rashomon_n_iters=args.rashomon_n_iters,
            rashomon_checkpoint=args.rashomon_checkpoint,
            rashomon_batch_size=rashomon_batch_size,
            rashomon_certificate_samples=args.certificate_samples,
            rashomon_inverse_temperature=args.rashomon_inverse_temp,
            rashomon_multi_label_mode=args.rashomon_multi_label_mode,
            rashomon_surrogate=args.rashomon_surrogate,
            rashomon_objective=args.rashomon_objective,
            rashomon_growth_method=args.growth_method,
            rashomon_certification_method=args.certification_method,
            safe_region_shape=args.safe_region_shape,
            segment_tolerance=args.segment_tolerance,
            segment_splits=args.segment_splits,
            segment_max_splits=args.segment_max_splits,
            rashomon_seed=args.seed,
            directional_rashomon_growth=args.directional_rashomon_growth,
            stop_when_proposal_contained=args.stop_when_proposal_contained,
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
        model.set_exploration_unsafe_action_callback(
            curve_logger.log_exploration_unsafe
        )
        reward_curve = UnshieldedRewardCurveCallback(
            env_factory=raw_env_factory,
            curve_logger=curve_logger,
            eval_freq=curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30_000,
            reward_threshold=args.success_reward_threshold,
            continuous_shield=shield,
        )
        early_stop = EarlyStopOnContinuousSuccessCallback(
            env_factory=raw_env_factory,
            shield=shield,
            eval_policy=args.early_stop_eval_policy,
            eval_freq=args.early_stop_eval_freq,
            eval_episodes=args.early_stop_eval_episodes,
            success_rate=args.early_stop_success_rate,
            seed=args.seed + 20_000,
            reward_threshold=args.success_reward_threshold,
        )
        started = time.perf_counter()
        curve_logger.start_timing(
            lambda: float(getattr(model, "_safety_enforcement_wall_time_s", 0.0))
        )
        log_info(
            f"[{ALGORITHM_NAME}] training for up to {args.total_timesteps} timesteps"
        )
        model.learn(
            total_timesteps=int(args.total_timesteps),
            callback=[reward_curve, early_stop],
        )
        model.finalize_adaptive_update()
        training_wall_time_s = time.perf_counter() - started
        final_interval_certified_fraction = float(model._greedy_safe_rate_now())
        if final_interval_certified_fraction < 1.0:
            raise RuntimeError(
                "Final policy failed complete interval verification; refusing to save "
                "an uncertified PSPO model."
            )
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(episode_env.episodes)
        training_shield_diagnostics = model.shield_diagnostics()
        adaptive_diagnostics = model.adaptive_diagnostics()
        adaptive_diagnostics.update(
            {"method": ALGORITHM_NAME, "verify_first": verify_first}
        )
        model.save(run_dir / "model.zip")
    finally:
        train_env.close()
        curve_logger.close()

    eval_env = _make_env(args, env_kwargs, shield, record_episodes=True)
    runtime_eval_shield = as_action_shield(shield)
    try:
        if args.evaluation_policy == "shielded":
            eval_records = evaluate_shielded_policy(
                model,
                eval_env,
                runtime_eval_shield,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            diagnostics = runtime_eval_shield.diagnostics()
            checked, unsafe = (
                int(diagnostics["checked"]),
                int(diagnostics["overridden"]),
            )
            eval_action_safety = {
                "proposed_action_checks": checked,
                "unsafe_proposed_action_count": unsafe,
                "unsafe_proposed_action_percentage": 100.0 * unsafe / checked
                if checked
                else 0.0,
            }
            eval_shield_diagnostics = diagnostics
        else:
            eval_records, eval_action_safety = evaluate_unshielded_policy(
                model,
                eval_env,
                runtime_eval_shield,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = None
    finally:
        eval_env.close()
    trajectory_safety = evaluate_trajectory_safety(
        args,
        model,
        raw_env_factory,
        shield,
        apply_shield=args.evaluation_policy == "shielded",
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
    )

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "max_episode_steps": args.max_episode_steps,
        "shield_type": "continuous_state",
        "continuous_shield": str(args.continuous_shield),
        "continuous_shield_config": asdict(shield.config),
        "continuous_shield_artifact": (
            str(Path(args.continuous_shield_artifact).resolve())
            if args.continuous_shield_artifact is not None
            else None
        ),
        "continuous_shield_artifact_sha256": (
            _file_sha256(Path(args.continuous_shield_artifact))
            if args.continuous_shield_artifact is not None
            else None
        ),
        "certificate_dataset": str(certificate_path.resolve()),
        "certificate_dataset_sha256": _file_sha256(certificate_path),
        "certificate_regions": len(certificate_dataset),
        "base_policy_path": str(Path(args.base_policy_path).resolve()),
        "base_policy_sha256": _file_sha256(Path(args.base_policy_path)),
        "base_policy_architecture": architecture,
        "mountaincar_shaped_reward": bool(args.mountaincar_shaped_reward),
        "lunarlander_unsafe_set": (
            None
            if args.env_id == MOUNTAINCAR_ENV_ID
            else {"event": "out_of_view", "abs_x_greater_equal": 1.0}
            if isinstance(shield, LunarLanderViewportShield)
            else {
                "height": float(args.lunarlander_unsafe_height),
                "vertical_speed": float(args.lunarlander_unsafe_vertical_speed),
            }
        ),
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "evaluation_policy": args.evaluation_policy,
        "eval_episodes": int(args.eval_episodes),
        "adaptive": {
            "verify_first": verify_first,
            "frequency": args.freq,
            "granularity": args.adaptive_granularity,
            "rollout_interval": args.adaptive_frequency,
            "unsafe_update_strategy": args.unsafe_update_strategy,
            "growth_method": str(args.growth_method),
            "certification_method": str(args.certification_method),
            "safe_region_shape": args.safe_region_shape,
            "segment_tolerance": args.segment_tolerance,
            "segment_splits": args.segment_splits,
            "segment_max_splits": args.segment_max_splits,
            "directional_rashomon_growth": args.directional_rashomon_growth,
            "rashomon_n_iters": args.rashomon_n_iters,
            "rashomon_checkpoint": args.rashomon_checkpoint,
            "rashomon_batch_size": rashomon_batch_size,
            "certificate_samples": args.certificate_samples,
            "rashomon_multi_label_mode": args.rashomon_multi_label_mode,
            "rashomon_surrogate": args.rashomon_surrogate,
            "rashomon_objective": args.rashomon_objective,
        },
        "training_hyperparameters": {
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
        },
    }
    write_json(run_dir / "config.json", config)
    io.write_record_csv(
        run_dir / "training_episodes.csv",
        io.record_training_rows(training_records, algorithm=ALGORITHM_NAME),
        include_end_timestep=True,
    )
    io.write_record_csv(
        run_dir / "episodes.csv",
        io.record_rows(eval_records, algorithm=ALGORITHM_NAME),
    )
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
    _write_early_stop_csv(
        run_dir / "early_stop_evaluations.csv", early_stop.evaluations
    )

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "final_timesteps": int(model.num_timesteps),
        "training": aggregate_training_violations(training_records),
        "evaluation": aggregate_violations(_records_to_metrics(eval_records)),
        "evaluation_policy": args.evaluation_policy,
        "evaluation_proposed_action_safety": eval_action_safety,
        "evaluation_shield_diagnostics": eval_shield_diagnostics,
        "trajectory_safety": trajectory_safety,
        # Retained under its historical key so the MountainCar analysis scripts
        # keep resolving; LunarLander reports the descent audit here instead.
        "left_boundary_trajectory_safety": trajectory_safety,
        "training_shield_diagnostics": training_shield_diagnostics,
        "adaptive_diagnostics": adaptive_diagnostics,
        "final_interval_certified_fraction": final_interval_certified_fraction,
        "final_interval_certified": final_interval_certified_fraction == 1.0,
        "early_stop_triggered": early_stop.stop_triggered,
        "last_early_stop_evaluation": (
            early_stop.evaluations[-1] if early_stop.evaluations else None
        ),
        "unshielded_eval_unsafe_action_count": (
            0
            if final_curve_evaluation is None
            else int(final_curve_evaluation.get("unsafe_proposed_action_count", 0))
        ),
        "unshielded_eval_safety_rate": (
            0.0
            if final_curve_evaluation is None
            else float(final_curve_evaluation.get("safety_rate", 0.0))
        ),
        "timing": {
            "training_wall_time_s": training_wall_time_s,
            "exact_verification_s": adaptive_diagnostics.get(
                "exact_verification_wall_time_total_s", 0.0
            ),
            "lid_complete_s": adaptive_diagnostics.get(
                "rashomon_wall_time_total_s", 0.0
            ),
            "projection_s": adaptive_diagnostics.get(
                "projection_wall_time_total_s", 0.0
            ),
            "safety_enforcement_s": adaptive_diagnostics.get(
                "safety_enforcement_wall_time_total_s", 0.0
            ),
        },
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "unshielded_reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    if isinstance(shield, LunarLanderViewportShield):
        _write_csv(run_dir / "trajectory_audit_episodes.csv", trajectory_safety["per_episode"],
                   ["episode", "reward", "length", "terminated", "truncated", "safe_trajectory", "landed", "crashed", "out_of_view"])
    write_json(
        run_dir / "metrics.json",
        summarise_evaluation(
            eval_records,
            success_reward_threshold=float(args.success_reward_threshold),
            cost_limit=float(args.cost_limit),
            algorithm=ALGORITHM_NAME,
            success_mode=success_mode_for_env(args.env_id),
        ),
    )
    log_info(
        f"[{ALGORITHM_NAME}] final complete-interval certificate: "
        f"{final_interval_certified_fraction:.0%}"
    )
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
