"""Registry mapping verification method names to BoundedModel backends."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from abstract_gradient_training.bounded_models import (
    BoundedModel,
    CROWNBoundedModel,
    IntervalBoundedModel,
)


@dataclasses.dataclass(frozen=True)
class VerificationMethod:
    """Describes one verification backend selectable by name."""

    name: str
    bounded_model_cls: type[BoundedModel]
    supported_modules: tuple[type[torch.nn.Module], ...]
    default_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


_IBP_SUPPORTED_MODULES = (
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.Tanh,
    torch.nn.Flatten,
    torch.nn.Dropout,
)

_CROWN_SUPPORTED_MODULES = (torch.nn.Linear, torch.nn.ReLU, torch.nn.Tanh)

_METHODS: dict[str, VerificationMethod] = {
    "IBP": VerificationMethod(
        name="IBP",
        bounded_model_cls=IntervalBoundedModel,
        supported_modules=_IBP_SUPPORTED_MODULES,
    ),
    "CROWN": VerificationMethod(
        name="CROWN",
        bounded_model_cls=CROWNBoundedModel,
        supported_modules=_CROWN_SUPPORTED_MODULES,
        default_kwargs={"relu_relaxation": "zero", "tanh_relaxation": "fixed"},
    ),
    "alpha-CROWN": VerificationMethod(
        name="alpha-CROWN",
        bounded_model_cls=CROWNBoundedModel,
        supported_modules=_CROWN_SUPPORTED_MODULES,
        default_kwargs={"relu_relaxation": "optimizable", "tanh_relaxation": "optimizable"},
    ),
}


def get_method(name: str) -> VerificationMethod:
    """Look up a registered verification method by name."""
    try:
        return _METHODS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown verification method {name!r}. Available methods: {available_methods()}"
        ) from exc


def register_method(method: VerificationMethod, *, overwrite: bool = False) -> None:
    """
    Register a new verification method (e.g. MIP or zonotope-based methods that pull in
    optional dependencies, or a custom backend) under its name.

    Args:
        method (VerificationMethod): The method to register.
        overwrite (bool, optional): If False, raises if a method with the same name is already
            registered.
    """
    if not overwrite and method.name in _METHODS:
        raise ValueError(
            f"Verification method {method.name!r} is already registered. Pass overwrite=True to replace it."
        )
    _METHODS[method.name] = method


def available_methods() -> list[str]:
    """Return the names of all currently registered verification methods."""
    return sorted(_METHODS)
