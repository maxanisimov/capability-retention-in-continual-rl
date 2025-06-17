from src.regulariser.BaseRegulariser import BaseRegulariser

import torch.nn as nn


class MultiRegulariser(BaseRegulariser):
    def __init__(self, regularisers: list[BaseRegulariser], **kwargs: dict):
        super().__init__()
        self.regularisers = regularisers

    def __call__(
        self,
        model: nn.Module = None,
        **kwargs: dict,
    ) -> float:
        assert model is not None, "Model must be provided for the regulariser."

        total_penalty = 0.0
        for regulariser in self.regularisers:
            total_penalty += regulariser(model=model, **kwargs)

        return total_penalty
