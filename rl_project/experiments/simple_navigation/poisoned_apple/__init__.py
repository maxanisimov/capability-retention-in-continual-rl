"""
Poisoned Apple Environment Module

A Gymnasium-based grid world environment for continual learning research
where an agent must collect safe apples while avoiding poisoned ones.
"""

from .poisoned_apple_env import (
    PoisonedAppleEnv,
    make_task1_env,
    make_task2_env
)

__all__ = [
    "PoisonedAppleEnv",
    "make_task1_env",
    "make_task2_env"
]
