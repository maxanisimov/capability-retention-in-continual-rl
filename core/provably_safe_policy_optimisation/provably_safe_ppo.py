"""Provably-safe PPO: projection (retention) + shielded exploration (safety).

:class:`ProvablySafePPO` extends
:class:`provably_safe_policy_optimisation.projected_ppo.ProjectedPPO` with a
user-provided safety :class:`~provably_safe_policy_optimisation.shield.Shield`.

During rollout collection, the policy proposes an action, the shield may replace
it before the environment is stepped, and PPO can store either the proposed action
or the executed shielded action in the rollout buffer. The default is
``shield_action_storage="proposed"``: PPO optimises the action emitted by the
policy, while the environment transition comes from the shielded action. Use
``"executed"`` to store the overridden action and recompute its log-probability.

Scope: ``predict()`` is not shielded; shielding applies to training exploration
inside ``collect_rollouts``.

Usage
-----
>>> from provably_safe_policy_optimisation import ProvablySafePPO
>>> model = ProvablySafePPO("MlpPolicy", env, shield=mask)  # mask: (n_states, n_actions)
>>> model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from provably_safe_policy_optimisation._projection_logging import record_shield_window
from provably_safe_policy_optimisation.policy_introspection import (
    extract_feature_actor_parameters_and_network,
)
from provably_safe_policy_optimisation.projected_ppo import ProjectedPPO
from provably_safe_policy_optimisation.safe_init import SafeInitReport, run_safe_init
from provably_safe_policy_optimisation.shield import Shield, as_shield

ShieldActionStorage = Literal["proposed", "executed"]


class ProvablySafePPO(ProjectedPPO):
    """PPO with projected updates and shielded exploration."""

    def __init__(
        self,
        *args: Any,
        shield: Any = None,
        obs_to_state: Any = None,
        shield_seed: int | None = None,
        shield_action_storage: ShieldActionStorage = "proposed",
        **kwargs: Any,
    ) -> None:
        if shield_action_storage not in ("proposed", "executed"):
            raise ValueError(
                "shield_action_storage must be either 'proposed' or 'executed', "
                f"got {shield_action_storage!r}.",
            )
        self._shield: Shield | None = None
        self._exploration_unsafe_action_callback: Any | None = None
        self.shield_action_storage: ShieldActionStorage = shield_action_storage
        # _setup_model runs inside super().__init__; attaching the shield afterwards
        # is fine because collect_rollouts reads self._shield live.
        super().__init__(*args, **kwargs)
        if shield is not None:
            self.set_shield(shield, obs_to_state, seed=shield_seed)
        else:
            warnings.warn(
                "ProvablySafePPO constructed without a shield; no shielding will be applied "
                "until set_shield(...) is called (e.g. after load).",
                stacklevel=2,
            )

    def set_shield(self, shield: Any, obs_to_state: Any = None, *, seed: int | None = None) -> None:
        """Attach (or re-attach) the safety shield, validating it against the action space."""
        shield = as_shield(shield, obs_to_state, seed=seed)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("ProvablySafePPO requires a Discrete action space.")
        if shield.n_actions != int(self.action_space.n):
            raise ValueError(
                f"Shield n_actions ({shield.n_actions}) does not match the action "
                f"space ({int(self.action_space.n)}).",
            )
        self._shield = shield

    def set_exploration_unsafe_action_callback(self, callback: Any | None) -> None:
        """Attach a hook called with unsafe proposed-action counts during rollout collection."""

        self._exploration_unsafe_action_callback = callback

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "_exploration_unsafe_action_callback",
        ]

    def _setup_model(self) -> None:
        super()._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect PPO rollouts with shielded execution.

        ``shield_action_storage="proposed"`` (default) sends the shielded action to
        the environment but stores the policy-proposed action and its log-probability
        in the rollout buffer. ``"executed"`` stores the shielded action instead and
        recomputes its log-probability under the current policy.
        """

        if self._shield is None:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                proposed_actions, values, proposed_log_probs = self.policy(obs_tensor)

            proposed_actions_np = proposed_actions.detach().cpu().numpy()
            env_actions = proposed_actions_np
            buffer_actions = proposed_actions_np
            log_probs = proposed_log_probs

            if isinstance(self.action_space, spaces.Discrete):
                states = np.asarray(self._shield.obs_to_state(obs_tensor)).astype(np.int64).reshape(-1)
                proposed_flat = np.asarray(proposed_actions_np).astype(np.int64).reshape(-1)
                unsafe_proposed_actions = ~self._shield.mask[states, proposed_flat]
                executed_flat = self._shield.override(states, proposed_flat)
                env_actions = executed_flat
                if self.shield_action_storage == "executed":
                    executed_actions = th.as_tensor(
                        executed_flat,
                        device=proposed_actions.device,
                        dtype=proposed_actions.dtype,
                    ).reshape(proposed_actions.shape)
                    with th.no_grad():
                        log_probs = self.policy.get_distribution(obs_tensor).log_prob(executed_actions)
                    buffer_actions = executed_flat
                else:
                    buffer_actions = proposed_actions_np

            clipped_actions = env_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(env_actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            hook = self._exploration_unsafe_action_callback
            if isinstance(self.action_space, spaces.Discrete) and hook is not None:
                hook(
                    timestep=int(self.num_timesteps),
                    unsafe_this_step=int(np.count_nonzero(unsafe_proposed_actions)),
                    checked_this_step=int(unsafe_proposed_actions.size),
                )

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                buffer_actions = np.asarray(buffer_actions).reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                buffer_actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def pretrain_on_shield(self, **kwargs: Any) -> SafeInitReport:
        """Make the actor propose safe actions before ``learn()``.

        Behaviourally clones the shield's safe actions (so the greedy action is safe),
        then — for box/discrete shields — certifies greedy-safety over each region via
        IBP, refining the worst-case margin until certified. If projection bounds are
        attached, parameters are re-projected afterwards. See :func:`run_safe_init` for
        keyword arguments.
        """
        # copy_modules=False: the returned Sequential shares the live actor parameters.
        _, actor_seq = extract_feature_actor_parameters_and_network(self, copy_modules=False)
        report = run_safe_init(
            self,
            logits_fn=lambda obs_t: self.policy.get_distribution(obs_t).distribution.logits,
            certify_seq=actor_seq,
            params=list(actor_seq.parameters()),
            **kwargs,
        )
        if self.policy.optimizer.has_bounds:
            self.policy.optimizer.project_now()
        return report

    def shield_diagnostics(self) -> dict[str, float]:
        """Cumulative shield intervention diagnostics (see ``Shield.diagnostics``)."""
        return self._shield.diagnostics() if self._shield is not None else {}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        result = super().train(*args, **kwargs)
        record_shield_window(self)
        return result
