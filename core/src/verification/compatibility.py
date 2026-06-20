"""Reusable layer-compatibility checking for verification backends."""

from __future__ import annotations

import torch


class UnsupportedLayerError(ValueError):
    """
    Raised when one or more layers in a model are unsupported by a verification method.

    Attributes:
        method_name: Name of the verification method that rejected the model.
        violations: List of (index, module) pairs for every unsupported layer found.
    """

    def __init__(self, method_name: str, violations: list[tuple[int, torch.nn.Module]]):
        self.method_name = method_name
        self.violations = violations
        lines = [
            f"Model is not compatible with verification method {method_name!r}. "
            f"Found {len(violations)} unsupported layer(s):"
        ]
        for idx, module in violations:
            lines.append(f"  [{idx}] {type(module).__name__} -> {module}")
        super().__init__("\n".join(lines))


def check_model_compatibility(
    model: torch.nn.Sequential,
    supported_modules: tuple[type[torch.nn.Module], ...],
    *,
    method_name: str = "",
) -> None:
    """
    Validate that every layer in `model` is an instance of one of `supported_modules`.

    Args:
        model (torch.nn.Sequential): The model to validate.
        supported_modules (tuple[type[torch.nn.Module], ...]): Module types supported by the
            verification method being checked against.
        method_name (str, optional): Name of the verification method, used in the error message.

    Raises:
        UnsupportedLayerError: If one or more layers are not instances of `supported_modules`. The
            error lists every unsupported layer (index and type), not just the first one found.
    """
    violations = [
        (idx, module) for idx, module in enumerate(model) if not isinstance(module, supported_modules)
    ]
    if violations:
        raise UnsupportedLayerError(method_name, violations)
