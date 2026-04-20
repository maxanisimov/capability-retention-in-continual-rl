"""Stable-Baselines3-style Discrete SAC implementation.

This module implements a discrete-action variant of SAC while preserving the
usual SB3 algorithm API:

```python
model = DiscreteSAC("MlpPolicy", env)
model.learn(total_timesteps=100_000)
action, _ = model.predict(obs, deterministic=True)
```

Notes
-----
- SB3 upstream SAC supports continuous ``Box`` actions only.
- This implementation targets ``gymnasium.spaces.Discrete`` action spaces.
- The policy keeps the familiar actor + twin-critic structure and entropy
  temperature auto-tuning.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, PyTorchObs, Schedule
from stable_baselines3.common.utils import polyak_update


def _probs_and_log_probs_from_logits(logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Compute categorical probs/log-probs in a numerically stable way."""
    log_probs = th.log_softmax(logits, dim=-1)
    probs = th.exp(log_probs)
    return probs, log_probs


class DiscreteActor(BasePolicy):
    """Categorical actor for Discrete SAC."""

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.action_dim = int(self.action_space.n)

        self.policy_net = nn.Sequential(*create_mlp(features_dim, self.action_dim, net_arch, activation_fn))

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def get_action_logits(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.policy_net(features)

    def action_probabilities(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        logits = self.get_action_logits(obs)
        return _probs_and_log_probs_from_logits(logits)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        probs, _ = self.action_probabilities(obs)
        if deterministic:
            return probs.argmax(dim=1).reshape(-1)
        return th.distributions.Categorical(probs=probs).sample().reshape(-1)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic=deterministic)


class DiscreteCritic(BaseModel):
    """Twin (or multi-head) Q critic for Discrete SAC.

    Each head predicts Q-values for all actions: ``Q_i(s, :)``.
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor
        self.action_dim = int(self.action_space.n)

        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = nn.Sequential(*create_mlp(features_dim, self.action_dim, net_arch, activation_fn))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                n_critics=self.n_critics,
                share_features_extractor=self.share_features_extractor,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: PyTorchObs) -> tuple[th.Tensor, ...]:
        # If features are shared with actor, only actor loss should update them.
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features) for q_net in self.q_networks)

    def q1_forward(self, obs: PyTorchObs) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features)


class DiscreteSACPolicy(BasePolicy):
    """Policy class (actor + twin critics) for Discrete SAC."""

    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update({"net_arch": actor_arch})
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "net_arch": critic_arch,
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )
        self.share_features_extractor = share_features_extractor
        self.use_sde = use_sde
        if self.use_sde:
            raise ValueError("DiscreteSACPolicy does not support use_sde=True.")

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_parameters = [
                param for name, param in self.critic.named_parameters() if "features_extractor" not in name
            ]
        else:
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Never share critic_target extractor with actor/critic.
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.use_sde,
                n_critics=self.critic_kwargs["n_critics"],
                share_features_extractor=self.share_features_extractor,
                lr_schedule=self._dummy_schedule,  # not needed when loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DiscreteActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DiscreteActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DiscreteCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DiscreteCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


class MlpPolicy(DiscreteSACPolicy):
    """Alias of :class:`DiscreteSACPolicy`."""


class CnnPolicy(DiscreteSACPolicy):
    """CNN policy variant for image observations."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )


class MultiInputPolicy(DiscreteSACPolicy):
    """Dict-observation policy variant."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )


SelfDiscreteSAC = TypeVar("SelfDiscreteSAC", bound="DiscreteSAC")


class DiscreteSAC(OffPolicyAlgorithm):
    """Soft Actor-Critic for discrete action spaces."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: DiscreteSACPolicy
    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        policy: Union[str, type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.ent_coef_tensor: Optional[th.Tensor] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        if isinstance(self.target_entropy, str):
            if not self.target_entropy.startswith("auto"):
                raise ValueError("target_entropy must be float or a string starting with 'auto'.")
            entropy_fraction = 0.98
            if "_" in self.target_entropy:
                entropy_fraction = float(self.target_entropy.split("_")[1])
                if entropy_fraction <= 0.0:
                    raise ValueError("target_entropy auto scaling factor must be > 0.")
            self.target_entropy = float(entropy_fraction * np.log(self.action_space.n))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                if init_value <= 0.0:
                    raise ValueError("The initial value of ent_coef must be greater than 0.")
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
            self.ent_coef_tensor = None
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
            self.log_ent_coef = None
            self.ent_coef_optimizer = None

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers: list[th.optim.Optimizer] = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                assert self.ent_coef_tensor is not None
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            with th.no_grad():
                next_probs, next_log_probs = self.actor.action_probabilities(replay_data.next_observations)
                next_q_values = th.stack(self.critic_target(replay_data.next_observations), dim=0).min(dim=0).values
                next_v = (next_probs * (next_q_values - ent_coef * next_log_probs)).sum(dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_v

            current_q_values = self.critic(replay_data.observations)
            action_indices = replay_data.actions.long()
            if action_indices.ndim == 1:
                action_indices = action_indices.unsqueeze(1)

            critic_loss = 0.5 * sum(
                F.mse_loss(q_values.gather(1, action_indices), target_q_values) for q_values in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            probs, log_probs = self.actor.action_probabilities(replay_data.observations)
            min_q_pi = th.stack(self.critic(replay_data.observations), dim=0).min(dim=0).values.detach()
            actor_loss = (probs * (ent_coef * log_probs - min_q_pi)).sum(dim=1).mean()
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                entropy = -(probs * log_probs).sum(dim=1, keepdim=True).detach()
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = (self.log_ent_coef * (entropy - self.target_entropy)).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfDiscreteSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DiscreteSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDiscreteSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


__all__ = [
    "DiscreteSAC",
    "DiscreteSACPolicy",
    "MlpPolicy",
    "CnnPolicy",
    "MultiInputPolicy",
]
