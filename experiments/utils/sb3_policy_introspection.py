"""Introspection helpers for SB3 ActorCritic-style policies.

These utilities identify the actor ("feature_actor") branch of an SB3 policy
and extract it as a standalone network. They are independent of any particular
training class and are reused by, e.g.,
:class:`experiments.utils.sb3_projected_ppo.ProjectedPPO`.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch as th


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


def resolve_policy(model_or_policy: Any) -> Any:
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


def resolve_feature_actor_names_for_policy(
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
            flattened = []
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
    policy = resolve_policy(model_or_policy)
    name_to_param = dict(policy.named_parameters())
    names = resolve_feature_actor_names_for_policy(policy, name_to_param)
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
