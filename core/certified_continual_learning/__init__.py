"""Canonical public API package for capability retention continual RL."""

import abstract_gradient_training as agt

from src.trainer import AGEMTrainer
from src.trainer import BufferTrainer
from src.trainer import EWCTrainer
from src.trainer import FisherTrainer
from src.trainer import IntervalTrainer
from src.trainer import SITrainer
from src.trainer import SimpleTrainer

__version__ = "0.1.0"

__all__ = [
    "AGEMTrainer",
    "BufferTrainer",
    "EWCTrainer",
    "FisherTrainer",
    "IntervalTrainer",
    "SITrainer",
    "SimpleTrainer",
    "agt",
]
