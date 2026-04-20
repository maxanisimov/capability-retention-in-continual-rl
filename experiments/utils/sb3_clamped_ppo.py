"""SB3 PPO extension with post-update parameter clamping.

This module adds a local PPO subclass that clamps selected policy parameters
immediately after each optimizer step. It is designed for local experimentation
without patching the installed stable-baselines3 package.

Example
-------
Clamp the feature extractor and actor branch to ``[-0.05, 0.05]``:

```python
import gymnasium as gym
from experiments.utils.sb3_clamped_ppo import ClampedPPO, ClampRule

env = gym.make("CartPole-v1")
model = ClampedPPO(
    "MlpPolicy",
    env,
    param_clamp_rules=[
        ClampRule(selector="feature_actor", min_value=-0.05, max_value=0.05),
    ],
)
model.learn(total_timesteps=10_000)
```

Pass per-parameter tensor bounds for the full ``feature_actor`` set:

```python
import torch as th

params = dict(model.policy.named_parameters())
feature_actor_names = [
    name
    for name in params
    if name.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net."))
]
lower_bounds = [th.full_like(params[name], -0.05) for name in feature_actor_names]
upper_bounds = [th.full_like(params[name], 0.05) for name in feature_actor_names]
model.set_feature_actor_parameter_bounds(lower_bounds, upper_bounds)
model.learn(total_timesteps=10_000)
```
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping, Sequence

import torch as th
from stable_baselines3 import PPO


ClampSelector = Literal[
    "actor_all",
    "critic_all",
    "actor_head",
    "critic_head",
    "log_std",
    "feature_actor",
    "feature_critic",
]


@dataclass(frozen=True)
class ClampRule:
    """Single parameter-clamping rule.

    Exactly one matcher must be provided among:
    - ``selector``
    - ``name_prefix``
    - ``name_regex``
    """

    min_value: float
    max_value: float
    selector: ClampSelector | None = None
    name_prefix: str | None = None
    name_regex: str | None = None


@dataclass(frozen=True)
class _ResolvedRule:
    """Validated, normalized clamp rule."""

    index: int
    min_value: float
    max_value: float
    selector: ClampSelector | None
    name_prefix: str | None
    name_regex: str | None


@dataclass(frozen=True)
class _TensorBoundTarget:
    """Per-parameter tensor bounds for clamping."""

    name: str
    parameter: th.nn.Parameter
    lower: th.Tensor
    upper: th.Tensor


_SELECTOR_VALUES: set[str] = {
    "actor_all",
    "critic_all",
    "actor_head",
    "critic_head",
    "log_std",
    "feature_actor",
    "feature_critic",
}

_FEATURE_SELECTORS: set[str] = {"feature_actor", "feature_critic"}


def _module_parameter_names(
    *,
    module: th.nn.Module | None,
    name_to_param: Mapping[str, th.nn.Parameter],
) -> set[str]:
    """Return names in ``name_to_param`` that belong to ``module`` by identity."""
    if module is None:
        return set()
    module_param_ids = {id(param) for param in module.parameters()}
    return {
        name
        for name, param in name_to_param.items()
        if id(param) in module_param_ids
    }


def _resolve_policy(model_or_policy: Any) -> Any:
    """Return an SB3 policy object from either a model or policy input.

    Args:
        model_or_policy: Either an SB3 algorithm instance with ``.policy`` or an
            SB3 policy object itself.

    Returns:
        The resolved policy object.

    Raises:
        TypeError: If the input does not look like an SB3 model/policy.
    """
    policy = getattr(model_or_policy, "policy", model_or_policy)
    required_attrs = ("named_parameters", "mlp_extractor", "action_net")
    if not all(hasattr(policy, attr) for attr in required_attrs):
        raise TypeError(
            "Expected an SB3 model (with .policy) or ActorCritic-style policy "
            "with named_parameters/mlp_extractor/action_net.",
        )
    return policy


def _resolve_feature_actor_names_for_policy(
    policy: Any,
    name_to_param: Mapping[str, th.nn.Parameter],
) -> list[str]:
    """Resolve canonical ``feature_actor`` parameter names for a given policy."""
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    policy_net = getattr(mlp_extractor, "policy_net", None)
    actor_all = (
        _module_parameter_names(module=policy_net, name_to_param=name_to_param)
        | _module_parameter_names(
            module=getattr(policy, "action_net", None),
            name_to_param=name_to_param,
        )
    )

    share_features = bool(getattr(policy, "share_features_extractor", False))
    if share_features:
        actor_feature_names = _module_parameter_names(
            module=getattr(policy, "features_extractor", None),
            name_to_param=name_to_param,
        )
    else:
        actor_feature_names = _module_parameter_names(
            module=getattr(policy, "pi_features_extractor", None),
            name_to_param=name_to_param,
        )
    return sorted(set(actor_all) | set(actor_feature_names))


def _flatten_for_extracted_sequential(
    module: th.nn.Module,
    *,
    copy_modules: bool,
    unwrap_modules_with_children: bool = False,
) -> list[th.nn.Module]:
    """Flatten modules into a list suitable for a single ``nn.Sequential``.

    This helper recursively expands nested ``nn.Sequential`` containers.
    When ``unwrap_modules_with_children=True``, any module that contains child
    modules is treated as a transparent wrapper and recursively expanded into
    those children.
    """
    if copy_modules:
        module = copy.deepcopy(module)

    if isinstance(module, th.nn.Sequential):
        flattened: list[th.nn.Module] = []
        for child in module:
            flattened.extend(
                _flatten_for_extracted_sequential(
                    child,
                    copy_modules=False,
                    unwrap_modules_with_children=unwrap_modules_with_children,
                ),
            )
        return flattened

    if unwrap_modules_with_children:
        children = list(module.children())
        if children:
            flattened: list[th.nn.Module] = []
            for child in children:
                flattened.extend(
                    _flatten_for_extracted_sequential(
                        child,
                        copy_modules=False,
                        unwrap_modules_with_children=True,
                    ),
                )
            return flattened

    return [module]


def extract_feature_actor_parameters_and_network(
    model_or_policy: Any,
    *,
    copy_modules: bool = True,
) -> tuple[dict[str, th.Tensor], th.nn.Sequential]:
    """Extract feature+actor parameters and build ``feature_extractor -> actor``.

    The returned network follows the actor path:
    ``feature_extractor -> mlp_extractor.policy_net -> action_net``.

    Args:
        model_or_policy: SB3 model (for example PPO/A2C) or policy object.
        copy_modules: If ``True`` (default), deep-copy modules into an isolated
            ``nn.Sequential``. If ``False``, the returned sequential reuses the
            original policy modules.

    Returns:
        A tuple ``(parameter_tensors, policy_network)`` where:
        - ``parameter_tensors`` maps canonical feature-actor parameter names to
          detached tensor copies.
        - ``policy_network`` is a flat ``nn.Sequential`` corresponding to
          ``feature_extractor -> actor_mlp -> action_net``.
          Nested wrappers with child modules are recursively unwrapped.
    """
    policy = _resolve_policy(model_or_policy)
    name_to_param = dict(policy.named_parameters())
    names = _resolve_feature_actor_names_for_policy(policy, name_to_param)
    parameters = {name: name_to_param[name].detach().clone() for name in names}

    share_features = bool(getattr(policy, "share_features_extractor", False))
    feature_extractor = policy.features_extractor if share_features else policy.pi_features_extractor
    actor_mlp = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    modules: list[th.nn.Module] = []
    for module in (feature_extractor, actor_mlp, action_net):
        modules.extend(
            _flatten_for_extracted_sequential(
                module,
                copy_modules=copy_modules,
                unwrap_modules_with_children=True,
            )
        )

    network = th.nn.Sequential(*modules)
    network.eval()
    return parameters, network


class ClampedPPO(PPO):
    """PPO variant with configurable post-step parameter clamping.

    The class validates user-supplied clamping rules at initialization, resolves
    each rule to concrete parameter names, and wraps the policy optimizer so
    clamping is applied right after each optimizer step.
    """

    def __init__(
        self,
        *args: Any,
        param_clamp_rules: Sequence[ClampRule | Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """Initialize PPO and configure post-update clamping.

        Args:
            *args: Positional arguments forwarded to ``stable_baselines3.PPO``.
            param_clamp_rules: Optional list of clamping rules. Each rule must
                provide bounds (``min_value``, ``max_value``) and exactly one
                matcher among ``selector``, ``name_prefix``, or ``name_regex``.
            **kwargs: Keyword arguments forwarded to ``stable_baselines3.PPO``.
        """
        self._param_clamp_rules_input = list(param_clamp_rules or [])
        self._param_clamp_targets: list[tuple[str, th.nn.Parameter, float, float, int]] = []
        self._param_clamp_step_calls = 0
        self._param_clamp_projected_elements = 0
        self._optimizer_step_wrapped = False
        self._optimizer_original_step: Any = None
        self._feature_actor_bound_targets: dict[str, _TensorBoundTarget] = {}

        super().__init__(*args, **kwargs)

        self._configure_param_clamping(self._param_clamp_rules_input)
        if self._has_any_clamp_targets():
            self._wrap_optimizer_step_for_clamping()

    @property
    def param_clamp_step_calls(self) -> int:
        """Total number of optimizer steps where clamping was applied."""
        return int(self._param_clamp_step_calls)

    @property
    def param_clamp_projected_elements(self) -> int:
        """Total number of projected (out-of-bounds) tensor elements."""
        return int(self._param_clamp_projected_elements)

    @property
    def param_clamp_target_names(self) -> tuple[str, ...]:
        """Names of parameters that are currently clamped."""
        names = {name for name, _, _, _, _ in self._param_clamp_targets}
        names.update(self._feature_actor_bound_targets.keys())
        return tuple(sorted(names))

    def set_feature_actor_parameter_bounds(
        self,
        lower_bounds: Sequence[th.Tensor] | Mapping[str, th.Tensor],
        upper_bounds: Sequence[th.Tensor] | Mapping[str, th.Tensor],
    ) -> tuple[str, ...]:
        """Set per-parameter bounds for the ``feature_actor`` parameter set.

        This method configures elementwise lower/upper bounds for all parameters
        selected by ``feature_actor`` semantics:
        - actor branch parameters (policy MLP + action head), and
        - actor-side feature extractor parameters.

        If feature extractors are shared, shared feature parameters are included.
        If feature extractors are not shared, only actor-side feature extractor
        parameters are included.

        Args:
            lower_bounds: Either
                1) sequence of tensors ordered by the method's resolved parameter
                   order (returned by this method), or
                2) mapping from parameter name to lower-bound tensor.
            upper_bounds: Same structure as ``lower_bounds`` for upper bounds.

        Returns:
            Tuple of resolved parameter names in the canonical order used for
            sequence-based bounds.

        Raises:
            TypeError: If lower/upper bounds are provided with mismatched types.
            ValueError: If keys/lengths do not match the resolved parameter set,
                shapes are incompatible, or any lower bound exceeds upper bound.
        """
        names, name_to_param = self._resolve_feature_actor_names()
        if not names:
            raise ValueError("No feature_actor parameters found in the current policy.")

        lower_is_mapping = isinstance(lower_bounds, Mapping)
        upper_is_mapping = isinstance(upper_bounds, Mapping)
        if lower_is_mapping != upper_is_mapping:
            raise TypeError(
                "lower_bounds and upper_bounds must both be mappings or both be sequences.",
            )

        ordered_bounds: list[tuple[str, th.Tensor, th.Tensor]] = []
        if lower_is_mapping:
            lower_map = dict(lower_bounds)
            upper_map = dict(upper_bounds)
            expected = set(names)
            if set(lower_map.keys()) != expected or set(upper_map.keys()) != expected:
                raise ValueError(
                    "Mapping bounds keys must exactly match the feature_actor parameter names.",
                )
            ordered_bounds = [
                (name, lower_map[name], upper_map[name])
                for name in names
            ]
        else:
            lower_seq = list(lower_bounds)
            upper_seq = list(upper_bounds)
            if len(lower_seq) != len(names) or len(upper_seq) != len(names):
                raise ValueError(
                    "Sequence bounds must have length equal to the number of "
                    f"feature_actor parameters ({len(names)}).",
                )
            ordered_bounds = list(zip(names, lower_seq, upper_seq))

        targets: dict[str, _TensorBoundTarget] = {}
        for name, lower_raw, upper_raw in ordered_bounds:
            parameter = name_to_param[name]
            lower = self._normalize_bound_tensor(
                lower_raw,
                parameter=parameter,
                parameter_name=name,
                bound_label="lower",
            )
            upper = self._normalize_bound_tensor(
                upper_raw,
                parameter=parameter,
                parameter_name=name,
                bound_label="upper",
            )
            if th.any(lower > upper):
                raise ValueError(
                    f"Invalid tensor bounds for '{name}': lower has values greater than upper.",
                )
            targets[name] = _TensorBoundTarget(
                name=name,
                parameter=parameter,
                lower=lower,
                upper=upper,
            )

        self._feature_actor_bound_targets = targets
        if self._has_any_clamp_targets():
            self._wrap_optimizer_step_for_clamping()
        return tuple(names)

    def clear_feature_actor_parameter_bounds(self) -> None:
        """Remove per-parameter ``feature_actor`` tensor bounds."""
        self._feature_actor_bound_targets.clear()

    def train(self) -> None:
        """Run PPO training and append clamp metrics to the logger."""
        steps_before = self._param_clamp_step_calls
        projected_before = self._param_clamp_projected_elements

        super().train()

        if self._has_any_clamp_targets():
            self.logger.record(
                "train/param_clamp_steps",
                float(self._param_clamp_step_calls - steps_before),
            )
            self.logger.record(
                "train/param_clamp_projected_elements",
                float(self._param_clamp_projected_elements - projected_before),
            )

    def _configure_param_clamping(
        self,
        raw_rules: Sequence[ClampRule | Mapping[str, Any]],
    ) -> None:
        """Resolve validated rules into concrete parameter clamp targets.

        This method:
        - builds selector-to-parameter mappings from the current policy,
        - validates that every rule matches at least one parameter,
        - enforces strict non-overlap rules,
        - allows only one overlap exception: shared-feature overlap between
          ``feature_actor`` and ``feature_critic`` when bounds are identical.

        Args:
            raw_rules: User-supplied rules as ``ClampRule`` objects or mappings.
        """
        if not raw_rules:
            self._param_clamp_targets = []
            return

        name_to_param = dict(self.policy.named_parameters())
        resolved_rules = [
            self._normalize_rule(rule, index=index)
            for index, rule in enumerate(raw_rules)
        ]

        mlp_extractor = getattr(self.policy, "mlp_extractor", None)
        policy_net = getattr(mlp_extractor, "policy_net", None)
        value_net = getattr(mlp_extractor, "value_net", None)

        actor_all = (
            _module_parameter_names(
                module=policy_net,
                name_to_param=name_to_param,
            )
            | _module_parameter_names(
                module=getattr(self.policy, "action_net", None),
                name_to_param=name_to_param,
            )
        )
        critic_all = (
            _module_parameter_names(
                module=value_net,
                name_to_param=name_to_param,
            )
            | _module_parameter_names(
                module=getattr(self.policy, "value_net", None),
                name_to_param=name_to_param,
            )
        )

        share_features = bool(getattr(self.policy, "share_features_extractor", False))
        shared_feature_names = _module_parameter_names(
            module=getattr(self.policy, "features_extractor", None),
            name_to_param=name_to_param,
        )
        if share_features:
            actor_feature_names = set(shared_feature_names)
            critic_feature_names = set(shared_feature_names)
        else:
            actor_feature_names = _module_parameter_names(
                module=getattr(self.policy, "pi_features_extractor", None),
                name_to_param=name_to_param,
            )
            critic_feature_names = _module_parameter_names(
                module=getattr(self.policy, "vf_features_extractor", None),
                name_to_param=name_to_param,
            )

        selector_to_names: dict[str, set[str]] = {
            "actor_all": set(actor_all),
            "critic_all": set(critic_all),
            "actor_head": _module_parameter_names(
                module=getattr(self.policy, "action_net", None),
                name_to_param=name_to_param,
            ),
            "critic_head": _module_parameter_names(
                module=getattr(self.policy, "value_net", None),
                name_to_param=name_to_param,
            ),
            "log_std": {"log_std"} if "log_std" in name_to_param else set(),
            "feature_actor": set(actor_all) | set(actor_feature_names),
            "feature_critic": set(critic_all) | set(critic_feature_names),
        }

        assigned: dict[str, tuple[float, float, _ResolvedRule]] = {}
        for rule in resolved_rules:
            matched_names = self._match_rule_names(
                rule=rule,
                all_names=name_to_param.keys(),
                selector_to_names=selector_to_names,
            )
            if not matched_names:
                raise ValueError(
                    f"Clamp rule {rule.index} matched no parameters: {rule}",
                )

            for name in sorted(matched_names):
                if name not in assigned:
                    assigned[name] = (rule.min_value, rule.max_value, rule)
                    continue

                old_min, old_max, old_rule = assigned[name]
                if self._is_allowed_shared_feature_overlap(
                    name=name,
                    shared_feature_names=shared_feature_names,
                    first_rule=old_rule,
                    second_rule=rule,
                    first_bounds=(old_min, old_max),
                    second_bounds=(rule.min_value, rule.max_value),
                ):
                    continue

                raise ValueError(
                    "Overlapping clamp rules are not allowed unless they are "
                    "feature_actor/feature_critic overlap on shared feature "
                    "parameters with identical bounds. "
                    f"Parameter '{name}' matched rule {old_rule.index} and {rule.index}.",
                )

        targets: list[tuple[str, th.nn.Parameter, float, float, int]] = []
        for name in sorted(assigned):
            min_value, max_value, rule = assigned[name]
            targets.append((name, name_to_param[name], min_value, max_value, rule.index))
        self._param_clamp_targets = targets

    def _resolve_feature_actor_names(self) -> tuple[list[str], dict[str, th.nn.Parameter]]:
        """Resolve canonical ``feature_actor`` parameter names and parameter map."""
        name_to_param = dict(self.policy.named_parameters())
        names = _resolve_feature_actor_names_for_policy(self.policy, name_to_param)
        return names, name_to_param

    def _normalize_bound_tensor(
        self,
        raw_bound: Any,
        *,
        parameter: th.nn.Parameter,
        parameter_name: str,
        bound_label: str,
    ) -> th.Tensor:
        """Convert one raw bound into a tensor broadcastable to parameter shape."""
        if isinstance(raw_bound, th.Tensor):
            bound = raw_bound.detach().to(device=parameter.device, dtype=parameter.dtype)
        else:
            bound = th.as_tensor(raw_bound, dtype=parameter.dtype, device=parameter.device)
        if bound.ndim == 0:
            return th.full_like(parameter, fill_value=float(bound.item()))
        if bound.shape == parameter.shape:
            return bound.clone()
        try:
            return th.broadcast_to(bound, parameter.shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"{bound_label.capitalize()} bound for '{parameter_name}' with shape "
                f"{tuple(bound.shape)} is not broadcastable to parameter shape "
                f"{tuple(parameter.shape)}.",
            ) from exc

    def _has_any_clamp_targets(self) -> bool:
        """Return ``True`` when any scalar or tensor clamp targets are configured."""
        return bool(self._param_clamp_targets or self._feature_actor_bound_targets)

    def _normalize_rule(
        self,
        raw_rule: ClampRule | Mapping[str, Any],
        *,
        index: int,
    ) -> _ResolvedRule:
        """Validate and normalize one user-supplied clamping rule.

        Args:
            raw_rule: Rule supplied as a ``ClampRule`` or dictionary.
            index: Rule index in the input list, used for error messages.

        Returns:
            A validated ``_ResolvedRule`` with normalized scalar bounds.

        Raises:
            ValueError: If fields are missing/invalid, bounds are inconsistent,
                matcher count is not exactly one, or regex cannot compile.
            TypeError: If ``raw_rule`` is neither ``ClampRule`` nor mapping.
        """
        if isinstance(raw_rule, ClampRule):
            min_value = float(raw_rule.min_value)
            max_value = float(raw_rule.max_value)
            selector = raw_rule.selector
            name_prefix = raw_rule.name_prefix
            name_regex = raw_rule.name_regex
        elif isinstance(raw_rule, Mapping):
            if "min_value" not in raw_rule or "max_value" not in raw_rule:
                raise ValueError(
                    f"Clamp rule {index} must contain both 'min_value' and 'max_value'.",
                )
            min_value = float(raw_rule["min_value"])
            max_value = float(raw_rule["max_value"])
            selector = raw_rule.get("selector")
            name_prefix = raw_rule.get("name_prefix")
            name_regex = raw_rule.get("name_regex")
        else:
            raise TypeError(
                f"Clamp rule {index} must be ClampRule or mapping, got {type(raw_rule)}.",
            )

        if min_value > max_value:
            raise ValueError(
                f"Clamp rule {index} has invalid bounds: min_value={min_value} > max_value={max_value}.",
            )

        matcher_count = sum(
            value is not None
            for value in (selector, name_prefix, name_regex)
        )
        if matcher_count != 1:
            raise ValueError(
                f"Clamp rule {index} must specify exactly one matcher among "
                "'selector', 'name_prefix', 'name_regex'.",
            )

        if selector is not None:
            if selector not in _SELECTOR_VALUES:
                raise ValueError(
                    f"Clamp rule {index} has unsupported selector '{selector}'. "
                    f"Allowed selectors: {sorted(_SELECTOR_VALUES)}.",
                )
            selector = selector  # narrow type

        if name_prefix is not None and not isinstance(name_prefix, str):
            raise ValueError(
                f"Clamp rule {index} has non-string name_prefix={name_prefix!r}.",
            )
        if name_regex is not None:
            if not isinstance(name_regex, str):
                raise ValueError(
                    f"Clamp rule {index} has non-string name_regex={name_regex!r}.",
                )
            try:
                re.compile(name_regex)
            except re.error as exc:
                raise ValueError(
                    f"Clamp rule {index} has invalid name_regex={name_regex!r}: {exc}",
                ) from exc

        return _ResolvedRule(
            index=index,
            min_value=min_value,
            max_value=max_value,
            selector=selector,
            name_prefix=name_prefix,
            name_regex=name_regex,
        )

    def _match_rule_names(
        self,
        *,
        rule: _ResolvedRule,
        all_names: Sequence[str],
        selector_to_names: Mapping[str, set[str]],
    ) -> set[str]:
        """Return parameter names matched by a resolved rule.

        Args:
            rule: Normalized clamping rule.
            all_names: All policy parameter names.
            selector_to_names: Precomputed selector-to-parameter-name mapping.

        Returns:
            Set of matched parameter names.
        """
        if rule.selector is not None:
            return set(selector_to_names[rule.selector])
        if rule.name_prefix is not None:
            return {name for name in all_names if name.startswith(rule.name_prefix)}
        if rule.name_regex is not None:
            regex = re.compile(rule.name_regex)
            return {name for name in all_names if regex.search(name)}
        return set()

    def _is_allowed_shared_feature_overlap(
        self,
        *,
        name: str,
        shared_feature_names: set[str],
        first_rule: _ResolvedRule,
        second_rule: _ResolvedRule,
        first_bounds: tuple[float, float],
        second_bounds: tuple[float, float],
    ) -> bool:
        """Check whether a rule overlap is allowed by shared-feature exception.

        Allowed overlap is limited to:
        - parameter belongs to shared feature extractor,
        - one rule uses ``feature_actor`` and the other ``feature_critic``,
        - both rules define identical bounds.
        """
        if name not in shared_feature_names:
            return False
        if first_rule.selector is None or second_rule.selector is None:
            return False
        if {first_rule.selector, second_rule.selector} != _FEATURE_SELECTORS:
            return False
        return first_bounds == second_bounds

    def _wrap_optimizer_step_for_clamping(self) -> None:
        """Monkey-patch optimizer ``step()`` to apply clamps post-update."""
        if self._optimizer_step_wrapped:
            return

        optimizer = self.policy.optimizer
        self._optimizer_original_step = optimizer.step
        original_step = self._optimizer_original_step

        def _step_with_clamp(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            self._apply_param_clamps()
            return result

        optimizer.step = _step_with_clamp  # type: ignore[method-assign]
        self._optimizer_step_wrapped = True

    def _apply_param_clamps(self) -> None:
        """Apply in-place clamping for all resolved targets and update counters."""
        if not self._has_any_clamp_targets():
            return

        projected = 0
        tensor_bound_names = set(self._feature_actor_bound_targets.keys())
        with th.no_grad():
            for name, parameter, min_value, max_value, _rule_index in self._param_clamp_targets:
                if name in tensor_bound_names:
                    continue
                violations = ((parameter < min_value) | (parameter > max_value)).sum().item()
                projected += int(violations)
                parameter.clamp_(min=min_value, max=max_value)
            for target in self._feature_actor_bound_targets.values():
                violations = ((target.parameter < target.lower) | (target.parameter > target.upper)).sum().item()
                projected += int(violations)
                target.parameter.copy_(th.clamp(target.parameter, min=target.lower, max=target.upper))

        self._param_clamp_step_calls += 1
        self._param_clamp_projected_elements += projected
