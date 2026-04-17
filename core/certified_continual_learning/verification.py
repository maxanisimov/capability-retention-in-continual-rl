"""Compatibility wrapper for verification modules."""

from src.verification import interval_tensor
from src.verification import verify
from src.verification import zonotope_tensor

__all__ = ["interval_tensor", "verify", "zonotope_tensor"]
