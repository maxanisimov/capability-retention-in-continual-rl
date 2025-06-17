from src.regulariser.BaseRegulariser import BaseRegulariser

import torch
import torch.nn as nn


class UnbiasRegulariser(BaseRegulariser):
    def __init__(self, lmbd: float = 1.0, **kwargs: dict):
        super().__init__()
        self.lmbd = lmbd

    def __call__(
        self,
        model: nn.Module = None,
        **kwargs: dict,
    ) -> float:
        inputs = kwargs.get("inputs", None)
        domain = kwargs.get("domain", None)
        transformations = kwargs.get("transformations", None)

        assert model is not None, "Model must be provided for the regulariser."
        assert inputs is not None, "Inputs must be provided for the regulariser."

        if domain is not None:
            random_inputs = (
                torch.rand_like(inputs) * (domain[1] - domain[0]) + domain[0]
            )
        elif transformations is not None:
            random_inputs = torch.concatenate([f(inputs) for f in transformations], 0)
        else:
            random_inputs = torch.randn_like(inputs)
        logits = model(random_inputs)
        probs = logits.softmax(dim=-1)
        penalty = (probs - probs.mean(dim=1).unsqueeze(1)).norm(dim=1).sum()

        return penalty.item() * self.lmbd
