"""Verification utilities for interval and zonotope bounds."""

from src.verification.api import (
    AdmissibleSet,
    VerificationResult,
    build_bounded_model,
    verify_point,
    verify_dataset,
)
from src.verification.registry import get_method, register_method, available_methods
from src.verification.compatibility import check_model_compatibility, UnsupportedLayerError

__all__ = [
    "AdmissibleSet",
    "VerificationResult",
    "build_bounded_model",
    "verify_point",
    "verify_dataset",
    "get_method",
    "register_method",
    "available_methods",
    "check_model_compatibility",
    "UnsupportedLayerError",
]
