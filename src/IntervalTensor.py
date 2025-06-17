"""Wrapper for pytorch tensors to support interval arithmetic."""

from __future__ import annotations

import torch

import abstract_gradient_training as agt

# disable some pylint warnings so we can match pytorch's style
# pylint: disable=missing-docstring, invalid-name


class IntervalTensor:
    """A class representing an interval over a pytorch tensor."""

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor | None = None):
        """Initialise the interval. If not upper bound is given, treat the tensor as constant."""
        if not isinstance(lb, torch.Tensor) or not isinstance(ub, torch.Tensor | None):
            raise TypeError("lb and ub must be torch.Tensor or None.")
        if ub is not None and lb.shape != ub.shape:
            agt.interval_arithmetic.validate_interval(lb, ub)
        self.lb = lb
        self.ub = lb if ub is None else ub

    def __add__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Add two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(self.lb + other, self.ub + other)
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.lb + other.lb, self.ub + other.ub)
        return NotImplemented

    def __matmul__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Matrix product of two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_matmul(
                    self.lb, self.ub, other, other
                )
            )
        if isinstance(other, IntervalTensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_matmul(
                    self.lb, self.ub, other.lb, other.ub
                )
            )
        return NotImplemented

    def __rmatmul__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Matrix product of two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_matmul(
                    other, other, self.lb, self.ub
                )
            )
        if isinstance(other, IntervalTensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_matmul(
                    other.lb, other.ub, self.lb, self.ub
                )
            )
        return NotImplemented

    def __mul__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Elementwise multiply two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_elementwise(
                    self.lb, self.ub, other, other
                )
            )
        if isinstance(other, IntervalTensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_elementwise(
                    self.lb, self.ub, other.lb, other.ub
                )
            )
        return NotImplemented

    def __rmul__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Elementwise multiply two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_elementwise(
                    other, other, self.lb, self.ub
                )
            )
        if isinstance(other, IntervalTensor):
            return IntervalTensor(
                *agt.interval_arithmetic.propagate_elementwise(
                    other.lb, other.ub, self.lb, self.ub
                )
            )
        return NotImplemented

    def __sub__(self, other: torch.Tensor | IntervalTensor) -> IntervalTensor:
        """Subtract two intervals or an interval and a constant."""
        if isinstance(other, torch.Tensor):
            return IntervalTensor(self.lb - other, self.ub - other)
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.lb - other.ub, self.ub - other.lb)
        return NotImplemented

    def __neg__(self) -> IntervalTensor:
        """Negate the interval."""
        return IntervalTensor(-self.ub, -self.lb)

    def __truediv__(self, other: float | int) -> IntervalTensor:
        """Divide the interval by a scalar constant."""
        if isinstance(other, (float, int)):
            if other > 0:
                return IntervalTensor(self.lb / other, self.ub / other)
            if other < 0:
                return IntervalTensor(self.ub / other, self.lb / other)
            raise ZeroDivisionError("Cannot divide interval by zero.")
        return NotImplemented

    def __repr__(self) -> str:
        return f"IntervalTensor(\n{self.lb},\n{self.ub}\n)"

    def __str__(self) -> str:
        return f"IntervalTensor(\n{self.lb},\n{self.ub}\n)"

    def __getitem__(self, key) -> IntervalTensor:
        return IntervalTensor(self.lb[key], self.ub[key])

    def __iter__(self):
        """Allow tuple unpacking of the interval, e.g. lb, ub = interval."""
        yield self.lb
        yield self.ub

    def width(self, dim=None) -> torch.Tensor:
        """Return the width of the interval summed over the given dimension."""
        return (self.ub - self.lb).sum(dim=dim)

    def heaviside(self) -> IntervalTensor:
        return IntervalTensor(
            (self.lb > 0).type(self.dtype), (self.ub > 0).type(self.dtype)
        )

    def relu(self) -> IntervalTensor:
        return IntervalTensor(torch.relu(self.lb), torch.relu(self.ub))

    def abs(self) -> IntervalTensor:
        return IntervalTensor(
            torch.minimum(self.lb.abs(), self.ub.abs()),
            torch.maximum(self.ub.abs(), self.lb.abs()),
        )

    def sum(self, dim) -> IntervalTensor:
        return IntervalTensor(self.lb.sum(dim=dim), self.ub.sum(dim=dim))

    def unsqueeze(self, dim=-1) -> IntervalTensor:
        return IntervalTensor(self.lb.unsqueeze(dim), self.ub.unsqueeze(dim))

    def transpose(self, *args, **kwargs) -> IntervalTensor:
        return IntervalTensor(
            self.lb.transpose(*args, **kwargs), self.ub.transpose(*args, **kwargs)
        )

    def reshape(self, *args, **kwargs) -> IntervalTensor:
        return IntervalTensor(
            self.lb.reshape(*args, **kwargs), self.ub.reshape(*args, **kwargs)
        )

    def expand(self, *args, **kwargs) -> IntervalTensor:
        return IntervalTensor(
            self.lb.expand(*args, **kwargs), self.ub.expand(*args, **kwargs)
        )

    def flatten(self, *args, **kwargs) -> IntervalTensor:
        return IntervalTensor(
            self.lb.flatten(*args, **kwargs), self.ub.flatten(*args, **kwargs)
        )

    def size(self, *args, **kwargs) -> torch.Size:
        return self.lb.size(*args, **kwargs)

    def requires_grad_(self, requires_grad: bool) -> IntervalTensor:
        """Set the requires_grad flag for the interval."""
        return IntervalTensor(
            self.lb.requires_grad_(requires_grad), self.ub.requires_grad_(requires_grad)
        )

    @property
    def ndim(self) -> int:
        return self.lb.ndim

    @property
    def shape(self) -> torch.Size:
        return self.lb.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.lb.dtype

    @property
    def T(self) -> IntervalTensor:
        return IntervalTensor(self.lb.T, self.ub.T)

    @property
    def mT(self) -> IntervalTensor:
        return IntervalTensor(self.lb.mT, self.ub.mT)

    @property
    def device(self) -> torch.device:
        return self.lb.device

    @staticmethod
    def zeros_like(A) -> IntervalTensor:
        """Create a zero interval with the same shape as A."""
        if isinstance(A, IntervalTensor):
            z = torch.zeros_like(A.lb)
        else:
            z = torch.zeros_like(A)
        return IntervalTensor(z)

    @staticmethod
    def zeros(*args, **kwargs) -> IntervalTensor:
        """Create a zero interval with the given shape."""
        z = torch.zeros(*args, **kwargs)
        return IntervalTensor(z, z)

    def to(self, *args, **kwargs) -> IntervalTensor:
        """Move the interval to the given device."""
        return IntervalTensor(self.lb.to(*args, **kwargs), self.ub.to(*args, **kwargs))

    def concretize(self) -> IntervalTensor:
        """Return the concrete interval - to match the zonotope interface."""
        return self
