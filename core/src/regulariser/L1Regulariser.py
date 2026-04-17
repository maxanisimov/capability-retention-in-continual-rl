from src.regulariser.BaseRegulariser import BaseRegulariser

import torch
import torch.nn as nn


class L1Regulariser(BaseRegulariser):
    def __init__(self, lmbd: float = 1.0, **kwargs: dict):
        super().__init__()
        self.lmbd = lmbd

    def __call__(self, model: nn.Module = None, **kwargs: dict) -> float:
        assert model is not None, "Model must be provided for the regulariser."

        return self.lmbd * sum(torch.norm(param, 1) for param in model.parameters())
