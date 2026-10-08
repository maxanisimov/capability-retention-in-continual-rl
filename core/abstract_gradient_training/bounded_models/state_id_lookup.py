"""Exact categorical-state lookup with ``nn.Linear``-compatible parameters.

``StateIdLookupLinear`` represents the operation performed by a linear layer on
a one-hot state vector without ever constructing that vector.  Its ``weight``
and ``bias`` names and shapes intentionally match :class:`torch.nn.Linear`, so
state dictionaries can be copied directly between a lookup certificate actor
and the ordinary dense actor used by Stable-Baselines3.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _state_ids(inputs: torch.Tensor, *, n_states: int) -> torch.Tensor:
    """Validate ``(batch,)``/``(batch, 1)`` integer state IDs and flatten them."""

    if inputs.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "StateIdLookupLinear expects int32/int64 state IDs; "
            f"got {inputs.dtype}."
        )
    if inputs.ndim == 2 and inputs.shape[1] == 1:
        inputs = inputs[:, 0]
    elif inputs.ndim != 1:
        raise ValueError(
            "StateIdLookupLinear expects state IDs with shape (batch,) or "
            f"(batch, 1); got {tuple(inputs.shape)}."
        )
    ids = inputs.to(dtype=torch.long)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(n_states)):
        raise IndexError(
            f"State ID outside [0, {int(n_states)}): "
            f"min={int(ids.min())}, max={int(ids.max())}."
        )
    return ids


class StateIdLookupLinear(nn.Module):
    """Evaluate ``Linear(one_hot(state_id))`` by gathering one weight column."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if int(in_features) <= 0 or int(out_features) <= 0:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear exactly so seeded dense and lookup actors initialise alike.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        ids = _state_ids(state_ids, n_states=self.in_features)
        outputs = self.weight.index_select(1, ids).transpose(0, 1)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        share_parameters: bool = False,
    ) -> "StateIdLookupLinear":
        """Create an equivalent lookup layer by copying or sharing parameters."""

        lookup = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if share_parameters:
            lookup.weight = linear.weight
            if linear.bias is not None:
                lookup.bias = linear.bias
        else:
            with torch.no_grad():
                lookup.weight.copy_(linear.weight)
                if linear.bias is not None:
                    assert lookup.bias is not None
                    lookup.bias.copy_(linear.bias)
        lookup.train(linear.training)
        return lookup

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def validate_point_state_id_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    n_states: int,
) -> torch.Tensor:
    """Return validated IDs for a degenerate categorical input interval."""

    if lower.shape != upper.shape or lower.dtype != upper.dtype:
        raise ValueError(
            "State-ID lower and upper inputs must have identical shape and dtype."
        )
    if not torch.equal(lower, upper):
        raise ValueError(
            "StateIdLookupLinear supports point state IDs only; categorical input "
            "intervals are not defined."
        )
    return _state_ids(lower, n_states=n_states)
