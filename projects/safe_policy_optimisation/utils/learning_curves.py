"""TensorBoard and CSV learning-curve logging for shielded policy experiments."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from projects.safe_policy_optimisation.utils.safe_rl import state_cost

EXPLORATION_FIELDS = [
    "timestep",
    "unsafe_this_step",
    "checked_this_step",
    "unsafe_rate",
    "cumulative_unsafe",
    "cumulative_checked",
]
EVALUATION_SUMMARY_FIELDS = [
    "eval_index",
    "timestep",
    "episodes",
    "mean_total_reward",
    "min_total_reward",
    "max_total_reward",
    "success_rate",
    "safe_trajectory_count",
    "unsafe_state_visit_count",
    "safety_rate",
    "proposed_action_checks",
    "unsafe_proposed_action_count",
    "cumulative_unsafe_proposed_action_count",
    "unsafe_proposed_action_rate",
    "shield_alignment_rate",
]
EVALUATION_EPISODE_FIELDS = [
    "eval_index",
    "timestep",
    "episode",
    "total_reward",
    "length",
    "success",
    "safe_trajectory",
    "unsafe_state_visit_count",
    "proposed_action_checks",
    "unsafe_proposed_action_count",
    "unsafe_proposed_action_rate",
    "shield_alignment_rate",
]


class LearningCurveLogger:
    """Write requested learning curves to TensorBoard and project-local CSV files."""

    def __init__(self, *, curve_dir: Path, tensorboard_log_dir: Path) -> None:
        self.curve_dir = Path(curve_dir)
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir), flush_secs=10)
        self.cumulative_unsafe = 0
        self.cumulative_checked = 0
        self.cumulative_eval_unsafe = 0
        self.eval_index = 0

        self.exploration_path = self.curve_dir / "exploration_unsafe_actions.csv"
        self.evaluation_summary_path = self.curve_dir / "evaluation_unshielded_summary.csv"
        self.evaluation_episodes_path = self.curve_dir / "evaluation_unshielded_episodes.csv"

        self._exploration_handle = self.exploration_path.open("w", newline="", encoding="utf-8")
        self._evaluation_summary_handle = self.evaluation_summary_path.open("w", newline="", encoding="utf-8")
        self._evaluation_episodes_handle = self.evaluation_episodes_path.open("w", newline="", encoding="utf-8")
        self._exploration_writer = csv.DictWriter(self._exploration_handle, fieldnames=EXPLORATION_FIELDS)
        self._evaluation_summary_writer = csv.DictWriter(
            self._evaluation_summary_handle,
            fieldnames=EVALUATION_SUMMARY_FIELDS,
        )
        self._evaluation_episodes_writer = csv.DictWriter(
            self._evaluation_episodes_handle,
            fieldnames=EVALUATION_EPISODE_FIELDS,
        )
        self._exploration_writer.writeheader()
        self._evaluation_summary_writer.writeheader()
        self._evaluation_episodes_writer.writeheader()

    def log_exploration_unsafe(
        self,
        *,
        timestep: int,
        unsafe_this_step: int,
        checked_this_step: int,
    ) -> dict[str, float | int]:
        """Record unsafe raw actions proposed by the policy during shielded exploration."""

        checked = int(checked_this_step)
        unsafe = int(unsafe_this_step)
        self.cumulative_unsafe += unsafe
        self.cumulative_checked += checked
        row = {
            "timestep": int(timestep),
            "unsafe_this_step": unsafe,
            "checked_this_step": checked,
            "unsafe_rate": float(unsafe / checked) if checked else 0.0,
            "cumulative_unsafe": int(self.cumulative_unsafe),
            "cumulative_checked": int(self.cumulative_checked),
        }
        self._exploration_writer.writerow(row)
        self._exploration_handle.flush()
        self.writer.add_scalar(
            "exploration/cumulative_proposed_unsafe_actions",
            row["cumulative_unsafe"],
            int(timestep),
        )
        self.writer.add_scalar(
            "exploration/cumulative_exploration_steps",
            row["cumulative_checked"],
            int(timestep),
        )
        self.writer.add_scalar("exploration/proposed_unsafe_actions", unsafe, int(timestep))
        self.writer.add_scalar("exploration/proposed_unsafe_action_rate", row["unsafe_rate"], int(timestep))
        return row

    def log_unshielded_evaluation(
        self,
        *,
        timestep: int,
        episode_rows: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Record total rewards from deterministic unshielded policy evaluation."""

        eval_index = self.eval_index
        self.eval_index += 1
        rewards = np.asarray([float(row["total_reward"]) for row in episode_rows], dtype=np.float64)
        success_count = sum(int(bool(row.get("success", False))) for row in episode_rows)
        safe_trajectory_count = sum(int(bool(row.get("safe_trajectory", True))) for row in episode_rows)
        unsafe_state_visits = sum(int(row.get("unsafe_state_visit_count", 0)) for row in episode_rows)
        checked = sum(int(row.get("proposed_action_checks", 0)) for row in episode_rows)
        unsafe = sum(int(row.get("unsafe_proposed_action_count", 0)) for row in episode_rows)
        self.cumulative_eval_unsafe += unsafe
        unsafe_rate = float(unsafe / checked) if checked else 0.0
        summary = {
            "eval_index": int(eval_index),
            "timestep": int(timestep),
            "episodes": int(len(episode_rows)),
            "mean_total_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            "min_total_reward": float(np.min(rewards)) if rewards.size else 0.0,
            "max_total_reward": float(np.max(rewards)) if rewards.size else 0.0,
            "success_rate": float(success_count / len(episode_rows)) if episode_rows else 0.0,
            "safe_trajectory_count": int(safe_trajectory_count),
            "unsafe_state_visit_count": int(unsafe_state_visits),
            "safety_rate": float(safe_trajectory_count / len(episode_rows)) if episode_rows else 0.0,
            "proposed_action_checks": int(checked),
            "unsafe_proposed_action_count": int(unsafe),
            "cumulative_unsafe_proposed_action_count": int(self.cumulative_eval_unsafe),
            "unsafe_proposed_action_rate": unsafe_rate,
            "shield_alignment_rate": float(1.0 - unsafe_rate) if checked else 0.0,
        }
        self._evaluation_summary_writer.writerow(summary)
        self._evaluation_summary_handle.flush()
        for row in episode_rows:
            episode_row = {
                "eval_index": int(eval_index),
                "timestep": int(timestep),
                "episode": int(row["episode"]),
                "total_reward": float(row["total_reward"]),
                "length": int(row["length"]),
                "success": bool(row["success"]),
                "safe_trajectory": bool(row.get("safe_trajectory", True)),
                "unsafe_state_visit_count": int(row.get("unsafe_state_visit_count", 0)),
                "proposed_action_checks": int(row.get("proposed_action_checks", 0)),
                "unsafe_proposed_action_count": int(row.get("unsafe_proposed_action_count", 0)),
                "unsafe_proposed_action_rate": float(row.get("unsafe_proposed_action_rate", 0.0)),
                "shield_alignment_rate": float(row.get("shield_alignment_rate", 0.0)),
            }
            self._evaluation_episodes_writer.writerow(episode_row)
        self._evaluation_episodes_handle.flush()
        self.writer.add_scalar(
            "evaluation/unshielded_total_reward_mean",
            summary["mean_total_reward"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_total_reward_min",
            summary["min_total_reward"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_total_reward_max",
            summary["max_total_reward"],
            int(timestep),
        )
        self.writer.add_scalar("evaluation/unshielded_success_rate", summary["success_rate"], int(timestep))
        self.writer.add_scalar("evaluation/unshielded_safety_rate", summary["safety_rate"], int(timestep))
        self.writer.add_scalar(
            "evaluation/unshielded_unsafe_state_visits",
            summary["unsafe_state_visit_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_proposed_action_checks",
            summary["proposed_action_checks"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_unsafe_proposed_actions",
            summary["unsafe_proposed_action_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/cumulative_unshielded_unsafe_actions",
            summary["cumulative_unsafe_proposed_action_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_unsafe_proposed_action_rate",
            summary["unsafe_proposed_action_rate"],
            int(timestep),
        )
        self.writer.add_scalar(
            "evaluation/unshielded_shield_alignment_rate",
            summary["shield_alignment_rate"],
            int(timestep),
        )
        self.writer.flush()
        return summary

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self._exploration_handle.close()
        self._evaluation_summary_handle.close()
        self._evaluation_episodes_handle.close()


def episode_success(total_reward: float, infos: list[dict[str, Any]], *, reward_threshold: float) -> bool:
    """Determine success from environment info flags, falling back to reward threshold."""

    for key in ("is_success", "success"):
        for info in reversed(infos):
            if key in info:
                return bool(info[key])
    return float(total_reward) > float(reward_threshold)


def evaluate_unshielded_total_rewards(
    model: Any,
    env_factory: Callable[[], gym.Env],
    *,
    episodes: int,
    seed: int,
    reward_threshold: float,
    shield_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | bool]]:
    """Run deterministic raw-policy episodes and return per-episode total rewards."""

    env = env_factory()
    shield = None if shield_mask is None else np.asarray(shield_mask) != 0
    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + episode)
            done = False
            total_reward = 0.0
            length = 0
            checked = 0
            unsafe = 0
            initial_cost = state_cost(env, obs)
            unsafe_state_visits = int(initial_cost > 0.0)
            infos: list[dict[str, Any]] = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                if shield is not None:
                    state = int(np.asarray(obs).item())
                    checked += 1
                    unsafe += int(not bool(shield[state, action_int]))
                obs, reward, terminated, truncated, info = env.step(action_int)
                infos.append(dict(info))
                step_cost = state_cost(env, obs, info)
                unsafe_state_visits += int(step_cost > 0.0)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            rows.append(
                {
                    "episode": int(episode),
                    "total_reward": float(total_reward),
                    "length": int(length),
                    "success": bool(
                        episode_success(
                            total_reward,
                            infos,
                            reward_threshold=reward_threshold,
                        )
                    ),
                    "safe_trajectory": bool(unsafe_state_visits == 0),
                    "unsafe_state_visit_count": int(unsafe_state_visits),
                    "proposed_action_checks": int(checked),
                    "unsafe_proposed_action_count": int(unsafe),
                    "unsafe_proposed_action_rate": float(unsafe / checked) if checked else 0.0,
                    "shield_alignment_rate": float(1.0 - (unsafe / checked)) if checked else 0.0,
                }
            )
    finally:
        env.close()
    return rows


class UnshieldedRewardCurveCallback(BaseCallback):
    """Evaluate and log unshielded policy reward curves at a fixed timestep cadence."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        curve_logger: LearningCurveLogger,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        reward_threshold: float,
        shield_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.env_factory = env_factory
        self.curve_logger = curve_logger
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.shield_mask = None if shield_mask is None else np.asarray(shield_mask) != 0
        self.evaluations: list[dict[str, float | int]] = []

    def _logged_evaluation_at(self, timestep: int) -> dict[str, float | int] | None:
        for evaluation in reversed(self.evaluations):
            if int(evaluation["timestep"]) == int(timestep):
                return evaluation
        return None

    def _evaluate_and_log(self, *, timestep: int) -> dict[str, float | int]:
        episode_rows = evaluate_unshielded_total_rewards(
            self.model,
            self.env_factory,
            episodes=self.eval_episodes,
            seed=self.seed + int(timestep),
            reward_threshold=self.reward_threshold,
            shield_mask=self.shield_mask,
        )
        summary = self.curve_logger.log_unshielded_evaluation(
            timestep=int(timestep),
            episode_rows=episode_rows,
        )
        self.evaluations.append(summary)
        return summary

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._evaluate_and_log(timestep=int(self.num_timesteps))
        return True

    def _on_training_end(self) -> None:
        self.record_final_evaluation()

    def record_final_evaluation(self, timestep: int | None = None) -> dict[str, float | int] | None:
        """Record a final evaluation point unless this timestep is already logged."""

        if self.eval_freq <= 0:
            return None
        final_timestep = int(self.num_timesteps if timestep is None else timestep)
        existing = self._logged_evaluation_at(final_timestep)
        if existing is not None:
            return existing
        return self._evaluate_and_log(timestep=final_timestep)


class CallableUnshieldedRewardCurveCallback:
    """Callable reward-curve logger for training loops that accept ``callback(model)``."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        curve_logger: LearningCurveLogger,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        reward_threshold: float,
        shield_mask: np.ndarray | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.curve_logger = curve_logger
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.shield_mask = None if shield_mask is None else np.asarray(shield_mask) != 0
        self.evaluations: list[dict[str, float | int]] = []
        self._next_eval_timestep = self.eval_freq if self.eval_freq > 0 else None

    def _logged_evaluation_at(self, timestep: int) -> dict[str, float | int] | None:
        for evaluation in reversed(self.evaluations):
            if int(evaluation["timestep"]) == int(timestep):
                return evaluation
        return None

    def _evaluate_and_log(self, model: Any, *, timestep: int) -> dict[str, float | int]:
        episode_rows = evaluate_unshielded_total_rewards(
            model,
            self.env_factory,
            episodes=self.eval_episodes,
            seed=self.seed + int(timestep),
            reward_threshold=self.reward_threshold,
            shield_mask=self.shield_mask,
        )
        summary = self.curve_logger.log_unshielded_evaluation(
            timestep=int(timestep),
            episode_rows=episode_rows,
        )
        self.evaluations.append(summary)
        return summary

    def _advance_next_eval_timestep(self, current_timestep: int) -> None:
        if self._next_eval_timestep is None:
            return
        while self._next_eval_timestep <= int(current_timestep):
            self._next_eval_timestep += self.eval_freq

    def __call__(self, model: Any) -> bool:
        if self._next_eval_timestep is None:
            return True
        current_timestep = int(model.num_timesteps)
        if current_timestep < self._next_eval_timestep:
            return True
        self._evaluate_and_log(model, timestep=current_timestep)
        self._advance_next_eval_timestep(current_timestep)
        return True

    def record_final_evaluation(
        self,
        model: Any,
        timestep: int | None = None,
    ) -> dict[str, float | int] | None:
        """Record a final evaluation point unless this timestep is already logged."""

        if self.eval_freq <= 0:
            return None
        final_timestep = int(model.num_timesteps if timestep is None else timestep)
        existing = self._logged_evaluation_at(final_timestep)
        if existing is not None:
            return existing
        summary = self._evaluate_and_log(model, timestep=final_timestep)
        self._advance_next_eval_timestep(final_timestep)
        return summary
