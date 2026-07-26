"""Train projected PPO in the original state-action space with env-side shielded exploration.

The agent's policy operates purely on the original (Discrete) observation and
action spaces. During exploration, each action sampled from the policy is
*proposed* to a ``ProbShieldWrapperDisc``-shielded environment, which projects
it onto the safe-action polytope so the probabilistic safety guarantee
(eventual-unsafe reachability <= ``--safety-tolerance``) holds for the executed
behaviour. The agent never observes the executed action or the safety budget
``q``: from PPO's perspective the shield is invisible env stochasticity, so the
standard rollout collection already stores (original state, proposed action,
proposed log-prob) and gradients are computed in the original space.

Policy updates use projected gradient descent onto Rashomon parameter bounds,
exactly as in ``ProvablySafePPO``. By default the bounds are derived in-process
("auto-Rashomon"): (1) exact value iteration on the env's cost signal yields
the set of safest actions per state, (2) those sets become a safety
demonstration dataset (one-hot state -> multi-hot safest actions), (3) a base
policy is behaviour-cloned to 100% allowed-action accuracy, (4) an IBP Rashomon
set is certified around it, and (5) the actor is projected onto the resulting
box after every optimizer step - so every policy iterate proposes one of the
safest actions in every state. Pass ``--rashomon-dir`` to reuse a precomputed
``compute_shield_rashomon_set.py`` run instead, or ``--no-auto-rashomon`` to
train unprojected (single-box bounds only; union-of-boxes is out of scope).

Evaluation (final and periodic) runs in the **unshielded** original environment:
its purpose is to measure whether the learned policy is intrinsically safe
without the shield's interventions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from masa.common.wrappers import ConstraintPersistentWrapper
from masa.prob_shield.prob_shield_wrapper_v2 import ProbShieldWrapperDisc
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation import ProjectedPPO  # noqa: E402

from projects.safe_crl.utils.shield_utils import (  # noqa: E402
    synthesise_probabilistic_shield,
    synthesise_shield_from_successor_dict,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (  # noqa: E402
    build_base_policy,
    calibrate_inverse_temperature,
    compute_rashomon_bounds,
    fit_base_policy,
    make_safe_behaviour_payload,
)
from projects.safe_policy_optimisation.stages.train_masa_shielded_policy import (  # noqa: E402
    EpisodeRecorderWrapper,
)
from projects.safe_policy_optimisation.stages.train_rashomon_shielded_policy import (  # noqa: E402
    align_rashomon_bounds_to_ppo_actor,
    load_base_policy_architecture,
    load_rashomon_bounds,
    policy_kwargs_from_base_architecture,
    validate_rashomon_shapes,
)
from projects.safe_policy_optimisation.utils import io  # noqa: E402
from projects.safe_policy_optimisation.utils.envs import env_kwargs_from_args  # noqa: E402
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402
from projects.safe_policy_optimisation.utils.metrics import (  # noqa: E402
    success_mode_for_env,
    summarise_evaluation,
)
from projects.safe_policy_optimisation.utils.safe_crl_bridge import (  # noqa: E402
    make_custom_masa_env,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
)


ALGORITHM_NAME = "masa_shielded_projected_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "masa_shielded_projected_policy"
)


class SingleProposalShieldAdapter(gym.Wrapper):
    """Expose a ``ProbShieldWrapperDisc`` env in the original state-action space.

    The agent proposes a single original action ``a``; the adapter forwards the
    edge-mode proposal ``(i=a, j=a, mix=0)``, so the shield executes ``a``
    exactly when it is budget-feasible and otherwise mixes it with the safest
    action just enough to respect the safety budget. The executed action and
    the budget ``q`` are never exposed to the agent: the observation is the
    original ``orig_obs`` and the budget only appears in ``info`` for logging.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict) or "orig_obs" not in env.observation_space.spaces:
            raise TypeError(
                "SingleProposalShieldAdapter expects a ProbShieldWrapperDisc-style Dict "
                f"observation space with an 'orig_obs' key, got {env.observation_space}."
            )
        if not isinstance(env.action_space, spaces.MultiDiscrete) or len(env.action_space.nvec) != 3:
            raise TypeError(
                "SingleProposalShieldAdapter expects the (i, j, mix) MultiDiscrete action "
                f"space of ProbShieldWrapperDisc, got {env.action_space}."
            )
        self.observation_space = env.observation_space.spaces["orig_obs"]
        self.action_space = spaces.Discrete(int(env.action_space.nvec[0]))

    def _step_info(self, obs: dict[str, Any], info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        state = int(np.asarray(obs["orig_obs"]).item())
        cost = float(self.unwrapped.cost_fn(self.unwrapped.label_fn(state)))
        info = dict(info)
        info["cost"] = cost
        info["violated_step"] = cost > 0.0
        info["orig_obs"] = state
        info["safety_bound"] = float(np.asarray(obs["safety_bound"]).reshape(-1)[0])
        return obs["orig_obs"], info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return obs["orig_obs"], info

    def step(self, action: Any):
        proposed = int(np.asarray(action).item())
        proposal = np.array([proposed, proposed, 0], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(proposal)
        orig_obs, info = self._step_info(obs, info)
        return orig_obs, reward, terminated, truncated, info


class UnshieldedCostInfoWrapper(gym.Wrapper):
    """Expose the label-derived cost in ``info`` for an unshielded tabular env."""

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = int(np.asarray(obs).item())
        cost = float(self.unwrapped.cost_fn(self.unwrapped.label_fn(state)))
        info = dict(info)
        info["cost"] = cost
        info["violated_step"] = cost > 0.0
        return obs, reward, terminated, truncated, info


def make_shielded_train_env(
    env_id: str,
    *,
    max_episode_steps: int,
    env_kwargs: dict[str, Any],
    safety_tolerance: float,
    theta: float,
    max_vi_steps: int,
    granularity: int,
    cost_limit: float,
    record_episodes: bool,
) -> gym.Env:
    """Build the shielded exploration env exposed in the original spaces."""

    base_env = make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
    )
    try:
        shielded = ProbShieldWrapperDisc(
            ConstraintPersistentWrapper(base_env),
            label_fn=base_env.unwrapped.label_fn,
            cost_fn=base_env.unwrapped.cost_fn,
            theta=theta,
            max_vi_steps=max_vi_steps,
            init_safety_bound=safety_tolerance,
            granularity=granularity,
        )
    except AssertionError as error:
        raise ValueError(
            f"Shield synthesis could not certify safety bound {safety_tolerance} from the "
            f"initial state of {env_id}; increase --safety-tolerance above the state's "
            f"minimum achievable risk. Original error: {error}"
        ) from error
    env: gym.Env = SingleProposalShieldAdapter(shielded)
    if record_episodes:
        env = EpisodeRecorderWrapper(env, cost_limit=cost_limit)
    return env


def make_unshielded_eval_env(
    env_id: str,
    *,
    max_episode_steps: int,
    env_kwargs: dict[str, Any],
    cost_limit: float,
) -> EpisodeRecorderWrapper:
    """Build the unshielded env used to measure intrinsic policy safety."""

    base_env = make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
    )
    return EpisodeRecorderWrapper(UnshieldedCostInfoWrapper(base_env), cost_limit=cost_limit)


def evaluate_unshielded(
    model: ProjectedPPO,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    *,
    episodes: int,
    seed_offset: int = 10_000,
) -> list[dict[str, Any]]:
    """Roll out the deterministic raw policy in the unshielded env."""

    env = make_unshielded_eval_env(
        args.env_id,
        max_episode_steps=args.max_episode_steps,
        env_kwargs=env_kwargs,
        cost_limit=args.cost_limit,
    )
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=int(args.seed) + seed_offset + episode)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(int(np.asarray(action).item()))
                done = bool(terminated or truncated)
        return list(env.episodes)
    finally:
        env.close()


class UnshieldedEvalCurveCallback(BaseCallback):
    """Periodically evaluate intrinsic safety in the unshielded env during training."""

    def __init__(
        self,
        args: argparse.Namespace,
        env_kwargs: dict[str, Any],
        *,
        eval_freq: int,
        eval_episodes: int,
    ) -> None:
        super().__init__()
        self._args = args
        self._env_kwargs = dict(env_kwargs)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        records = evaluate_unshielded(
            self.model,
            self._args,
            self._env_kwargs,
            episodes=self.eval_episodes,
            seed_offset=20_000 + self.num_timesteps,
        )
        if records:
            self.rows.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "episodes": len(records),
                    "mean_reward": float(np.mean([r["reward"] for r in records])),
                    "mean_cost": float(np.mean([r["cost"] for r in records])),
                    "violation_rate": float(np.mean([bool(r["violated"]) for r in records])),
                }
            )
        return True


def _episode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return io.record_rows(records, algorithm=ALGORITHM_NAME)


def _training_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return io.record_training_rows(records, algorithm=ALGORITHM_NAME)


def _write_episode_csv(path: Path, rows: list[dict[str, Any]], *, include_end_timestep: bool = False) -> None:
    # Reduced column set, matching the MASA-shielded stage (no unsafe-state counters).
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algorithm", "episode", "reward", "cost", "length", "violated"]
    if include_end_timestep:
        fieldnames.insert(2, "end_timestep")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_eval_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timesteps", "episodes", "mean_reward", "mean_cost", "violation_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_projection_setup(
    rashomon_dir: Path,
    *,
    n_states: int,
    n_actions: int,
) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]:
    """Load and align Rashomon parameter bounds for the PPO actor."""

    architecture = load_base_policy_architecture(rashomon_dir)
    if int(architecture["input_dim"]) != n_states:
        raise ValueError(
            f"Rashomon base policy input_dim ({architecture['input_dim']}) does not match "
            f"the env's state count ({n_states})."
        )
    if int(architecture["n_actions"]) != n_actions:
        raise ValueError(
            f"Rashomon base policy n_actions ({architecture['n_actions']}) does not match "
            f"the env's action count ({n_actions})."
        )
    param_bounds_l, param_bounds_u = load_rashomon_bounds(rashomon_dir)
    param_bounds_l, param_bounds_u = align_rashomon_bounds_to_ppo_actor(
        architecture, param_bounds_l, param_bounds_u
    )
    policy_kwargs = policy_kwargs_from_base_architecture(architecture)
    return param_bounds_l, param_bounds_u, policy_kwargs, architecture


def synthesise_safest_action_mask(
    env_id: str,
    *,
    max_episode_steps: int,
    env_kwargs: dict[str, Any],
    theta: float,
    max_vi_steps: int,
    unsafe_cost_threshold: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Step 1: exact VI on the cost signal; returns the binary safest-actions mask.

    Entry ``[s, a] = 1`` iff ``a`` minimises the eventual-unsafe risk in state ``s``
    (within ``theta``). Rows of unsafe states are all zero.
    """

    env = make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
    )
    env.reset(seed=seed)
    try:
        unwrapped = env.unwrapped
        if unwrapped.get_transition_matrix() is not None:
            mask, info = synthesise_probabilistic_shield(
                env,
                transition_matrix_fn=lambda e: e.get_transition_matrix(),
                label_fn=unwrapped.label_fn,
                cost_fn=unwrapped.cost_fn,
                theta=theta,
                max_vi_steps=max_vi_steps,
                unsafe_cost_threshold=unsafe_cost_threshold,
                use_masa_helper=False,
                return_info=True,
            )
        elif unwrapped.has_successor_states_dict:
            mask, info = synthesise_shield_from_successor_dict(
                env,
                label_fn=unwrapped.label_fn,
                cost_fn=unwrapped.cost_fn,
                shield_type="probabilistic",
                theta=theta,
                max_vi_steps=max_vi_steps,
                unsafe_cost_threshold=unsafe_cost_threshold,
                return_info=True,
            )
        else:
            raise ValueError(
                f"{env_id} exposes neither a dense transition matrix nor a successor-states "
                "dict; exact tabular VI requires one of them.",
            )
    finally:
        env.close()

    mask = np.asarray(mask, dtype=np.int64)
    set_sizes = mask.sum(axis=1)
    nonzero = set_sizes[set_sizes > 0]
    stats = {
        "n_states": int(mask.shape[0]),
        "n_actions": int(mask.shape[1]),
        "states_with_safe_action": int(nonzero.size),
        "mean_safe_set_size": float(nonzero.mean()) if nonzero.size else 0.0,
        "mean_safe_set_size_all_states": float(set_sizes.mean()),
        "min_safe_set_size": int(nonzero.min()) if nonzero.size else 0,
        "max_safe_set_size": int(nonzero.max()) if nonzero.size else 0,
        "vi_steps": int(info.vi_steps) if info.vi_steps is not None else None,
        "vi_residual": float(info.vi_residual) if info.vi_residual is not None else None,
    }
    return mask, stats


def compute_rashomon_from_mask(
    mask: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Steps 2-4: BC dataset -> 100%-accurate base policy -> certified Rashomon box.

    Saves the same artifact set as ``compute_shield_rashomon_set.py`` under
    ``out_dir`` and returns actor-aligned bounds ready for ``ProjectedPPO``.
    """

    dataset, dataset_metadata = make_safe_behaviour_payload(mask)
    n_states = int(dataset["state"].shape[1])
    n_actions = int(dataset["actions"].shape[1])

    base_model = build_base_policy(
        n_states,
        n_actions,
        hidden_dim=args.bc_hidden_dim,
        n_hidden=args.bc_n_hidden,
    )
    bc_metrics = fit_base_policy(
        base_model,
        dataset,
        lr=args.bc_lr,
        max_epochs=args.bc_max_epochs,
        batch_size=args.bc_batch_size,
        seed=args.seed,
        device=args.device,
    )
    if not bc_metrics["reached_target"]:
        raise RuntimeError(
            "Base policy did not reach 100% allowed-action accuracy: "
            f"final_accuracy={bc_metrics['final_accuracy']:.6f}.",
        )

    inverse_temp, min_valid_mass, surrogate_threshold = calibrate_inverse_temperature(
        base_model,
        dataset,
        inverse_temp_start=1,
        inverse_temp_max=1000,
        device=args.device,
    )
    param_bounds_l, param_bounds_u, bounded_model, rashomon_metadata = compute_rashomon_bounds(
        base_model,
        dataset,
        seed=args.seed,
        n_iters=args.rashomon_n_iters,
        checkpoint=args.rashomon_checkpoint,
        batch_size=args.rashomon_batch_size,
        certificate_samples=args.certificate_samples,
        inverse_temp=inverse_temp,
    )

    architecture = {
        "input_dim": n_states,
        "n_actions": n_actions,
        "hidden_dim": int(args.bc_hidden_dim),
        "n_hidden": int(args.bc_n_hidden),
        "activation": "Tanh",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_dir / "safe_behaviour_dataset.pt")
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
            "architecture": architecture,
            "bc_metrics": bc_metrics,
        },
        out_dir / "base_policy.pt",
    )
    torch.save(bounded_model, out_dir / "rashomon_bounded_model.pt")
    torch.save(
        {"param_bounds_l": param_bounds_l, "param_bounds_u": param_bounds_u},
        out_dir / "rashomon_param_bounds.pt",
    )

    aligned_l, aligned_u = align_rashomon_bounds_to_ppo_actor(
        architecture, param_bounds_l, param_bounds_u
    )
    policy_kwargs = policy_kwargs_from_base_architecture(architecture)
    metadata = {
        "dataset": dataset_metadata,
        "base_policy": bc_metrics,
        "rashomon": {
            "inverse_temperature": int(inverse_temp),
            "min_valid_mass": float(min_valid_mass),
            "surrogate_threshold": float(surrogate_threshold),
            "n_iters": int(args.rashomon_n_iters),
            **rashomon_metadata,
        },
    }
    return aligned_l, aligned_u, policy_kwargs, architecture, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train projected PPO in the original state-action space; exploration is "
            "shielded by ProbShieldWrapperDisc, evaluation runs unshielded."
        ),
    )
    parser.add_argument("--env-id", default=None, help="MASA-style Gymnasium env id to train on.")
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to the MASA env factory.")
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--ghost-rand-prob",
        type=float,
        default=0.0,
        help="Backward-compatible MiniPacman shortcut used only when --env-kwargs is omitted.",
    )
    parser.add_argument(
        "--safety-tolerance",
        type=float,
        default=0.0,
        help="MASA initial safety bound. Default 0.0 allows only zero-risk exploration.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.0,
        help="Episode cost budget used for reporting violation counts.",
    )
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-vi-steps", type=int, default=1000)
    parser.add_argument("--granularity", type=int, default=20)
    parser.add_argument(
        "--rashomon-dir",
        type=Path,
        default=None,
        help=(
            "Directory with rashomon_param_bounds.pt and base_policy.pt (from "
            "compute_shield_rashomon_set.py). When given, it overrides the in-process "
            "auto-Rashomon pipeline."
        ),
    )
    parser.add_argument(
        "--no-auto-rashomon",
        action="store_true",
        help=(
            "Disable the in-process safest-actions VI -> BC -> Rashomon pipeline and "
            "train unprojected (only relevant when --rashomon-dir is omitted)."
        ),
    )
    parser.add_argument(
        "--unsafe-cost-threshold",
        type=float,
        default=0.5,
        help="Cost threshold separating safe and unsafe states in the VI cost signal.",
    )
    parser.add_argument("--bc-hidden-dim", type=int, default=64)
    parser.add_argument("--bc-n-hidden", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-max-epochs", type=int, default=1000)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--rashomon-n-iters", type=int, default=2000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument("--rashomon-batch-size", type=int, default=500)
    parser.add_argument("--certificate-samples", type=int, default=1000)
    parser.add_argument("--projection-distance-norm", default="l2", choices=("l2", "l1", "linf"))
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Timestep period for periodic unshielded evaluation during training (0 disables).",
    )
    parser.add_argument("--eval-episodes-periodic", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id is None:
        raise ValueError("--env-id is required for env-shielded projected policy training.")
    env_kwargs = env_kwargs_from_args(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_shielded_train_env(
        args.env_id,
        max_episode_steps=args.max_episode_steps,
        env_kwargs=env_kwargs,
        safety_tolerance=args.safety_tolerance,
        theta=args.theta,
        max_vi_steps=args.max_vi_steps,
        granularity=args.granularity,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )

    param_bounds_l = None
    param_bounds_u = None
    policy_kwargs: dict[str, Any] | None = None
    architecture: dict[str, Any] | None = None
    safe_set_stats: dict[str, Any] | None = None
    rashomon_metadata: dict[str, Any] | None = None
    if args.rashomon_dir is not None:
        projection_source = "precomputed"
        param_bounds_l, param_bounds_u, policy_kwargs, architecture = load_projection_setup(
            args.rashomon_dir,
            n_states=int(train_env.observation_space.n),
            n_actions=int(train_env.action_space.n),
        )
    elif not args.no_auto_rashomon:
        projection_source = "auto"
        # Steps 1-2: exact VI on the cost signal -> per-state safest-action sets.
        mask, safe_set_stats = synthesise_safest_action_mask(
            args.env_id,
            max_episode_steps=args.max_episode_steps,
            env_kwargs=env_kwargs,
            theta=args.theta,
            max_vi_steps=args.max_vi_steps,
            unsafe_cost_threshold=args.unsafe_cost_threshold,
            seed=args.seed,
        )
        log_info(
            "[{algorithm}] safest-action VI: mean safe-set size "
            "{mean:.3f} actions/state over {n_safe}/{n_states} states with a safe action "
            "(min={min}, max={max}, VI steps={steps})".format(
                algorithm=ALGORITHM_NAME,
                mean=safe_set_stats["mean_safe_set_size"],
                n_safe=safe_set_stats["states_with_safe_action"],
                n_states=safe_set_stats["n_states"],
                min=safe_set_stats["min_safe_set_size"],
                max=safe_set_stats["max_safe_set_size"],
                steps=safe_set_stats["vi_steps"],
            )
        )
        # Steps 3-5: BC to 100% accuracy -> certified Rashomon box -> PGD bounds.
        param_bounds_l, param_bounds_u, policy_kwargs, architecture, rashomon_metadata = (
            compute_rashomon_from_mask(mask, run_dir / "rashomon", args)
        )
        log_info(
            "[{algorithm}] auto-Rashomon: BC accuracy {acc:.3f} in {epochs} epochs, "
            "certificate {cert:.3f}".format(
                algorithm=ALGORITHM_NAME,
                acc=rashomon_metadata["base_policy"]["final_accuracy"],
                epochs=rashomon_metadata["base_policy"]["epochs_run"],
                cert=rashomon_metadata["rashomon"]["selected_certificate"],
            )
        )
    else:
        projection_source = "none"
        log_info(f"[{ALGORITHM_NAME}] auto-Rashomon disabled; training without projection.")

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "wrapper": "masa.prob_shield.prob_shield_wrapper_v2.ProbShieldWrapperDisc",
        "adapter": "SingleProposalShieldAdapter",
        "safety_tolerance": float(args.safety_tolerance),
        "cost_limit": float(args.cost_limit),
        "total_timesteps": int(args.total_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "eval_policy": "unshielded",
        "eval_every": int(args.eval_every),
        "eval_episodes_periodic": int(args.eval_episodes_periodic),
        "seed": int(args.seed),
        "max_episode_steps": int(args.max_episode_steps),
        "theta": float(args.theta),
        "max_vi_steps": int(args.max_vi_steps),
        "granularity": int(args.granularity),
        "rashomon_dir": str(args.rashomon_dir) if args.rashomon_dir is not None else None,
        "projection_enabled": param_bounds_l is not None,
        "projection_source": projection_source,
        "projection_distance_norm": str(args.projection_distance_norm),
        "unsafe_cost_threshold": float(args.unsafe_cost_threshold),
        "base_policy_architecture": architecture,
        "safest_action_sets": safe_set_stats,
        "auto_rashomon": rashomon_metadata,
    }
    write_json(run_dir / "config.json", config)

    try:
        model = ProjectedPPO(
            "MlpPolicy",
            train_env,
            param_bounds_l=param_bounds_l,
            param_bounds_u=param_bounds_u,
            projection_distance_norm=args.projection_distance_norm,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        if param_bounds_l is not None:
            projection_rows = validate_rashomon_shapes(model, param_bounds_l, param_bounds_u)
            write_json(run_dir / "projection_targets.json", projection_rows)

        eval_curve = UnshieldedEvalCurveCallback(
            args,
            env_kwargs,
            eval_freq=args.eval_every,
            eval_episodes=args.eval_episodes_periodic,
        )
        log_info(f"[{ALGORITHM_NAME}] training for {args.total_timesteps} timesteps")
        model.learn(total_timesteps=args.total_timesteps, callback=eval_curve)
        model.save(run_dir / "model.zip")
        training_records = list(train_env.episodes)
    finally:
        train_env.close()

    eval_records = evaluate_unshielded(model, args, env_kwargs, episodes=args.eval_episodes)
    _write_episode_csv(
        run_dir / "training_episodes.csv", _training_rows(training_records), include_end_timestep=True
    )
    _write_episode_csv(run_dir / "episodes.csv", _episode_rows(eval_records))
    if eval_curve.rows:
        _write_eval_curve_csv(run_dir / "eval_curve.csv", eval_curve.rows)

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "training_shielded": aggregate_training_violations(training_records),
        "evaluation_unshielded": aggregate_violations(
            [
                EpisodeMetrics(
                    episode=int(record["episode"]),
                    reward=float(record["reward"]),
                    cost=float(record["cost"]),
                    length=int(record["length"]),
                    violated=bool(record["violated"]),
                )
                for record in eval_records
            ]
        ),
        "projection": {
            "enabled": param_bounds_l is not None,
            "source": projection_source,
            "is_within_bounds": bool(model.is_within_bounds()) if param_bounds_l is not None else None,
            "diagnostics": model.projection_diagnostics() if param_bounds_l is not None else {},
        },
        "safest_action_sets": safe_set_stats,
        "auto_rashomon": rashomon_metadata,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "metrics.json",
        summarise_evaluation(
            eval_records,
            success_reward_threshold=float(args.success_reward_threshold),
            success_mode=success_mode_for_env(getattr(args, "env_id", None)),
            cost_limit=float(args.cost_limit),
            algorithm=ALGORITHM_NAME,
        ),
    )
    log_info(
        "[{algorithm}] shielded training violations: {count}/{episodes} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            count=summary["training_shielded"]["training_violation_count"],
            episodes=summary["training_shielded"]["training_episode_count"],
            pct=summary["training_shielded"]["training_violation_percentage"],
        )
    )
    log_info(
        "[{algorithm}] unshielded eval violations: {count}/{episodes} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            count=summary["evaluation_unshielded"]["violation_count"],
            episodes=summary["evaluation_unshielded"]["episodes"],
            pct=summary["evaluation_unshielded"]["violation_percentage"],
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
