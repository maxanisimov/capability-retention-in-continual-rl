"""
A zonotope abstract domain defined by a center interval and a set of generators:

    X = {X_c + sum_{i in I} X_i * g_i | g_i in [l_i, u_i]}

where

    - X_c is the center interval
    - X_i are the generator intervals
    - g_i are the coefficients of the generators, which have bounds [l_i, u_i]

The crucial point is that the coefficients g_i are shared across all zonotopes, allowing for cancellations
and tighter verification.
"""

from __future__ import annotations

import torch

from src.verification.interval_tensor import IntervalTensor

# disable some pylint warnings so we can match pytorch's style
# pylint: disable=missing-docstring, invalid-name


class ZonotopeTensor:
    """A class representing a mixed interval-zonotope object over a pytorch tensor."""

    def __init__(
        self,
        center: torch.Tensor | IntervalTensor,
        generators: torch.Tensor | IntervalTensor,
        coefficients: IntervalTensor,
    ):
        """Initialise the zonotope with a center and generators."""
        # validate input types
        if not isinstance(center, torch.Tensor | IntervalTensor):
            raise TypeError("Center must be a torch.Tensor or IntervalTensor.")
        if not isinstance(generators, torch.Tensor | IntervalTensor):
            raise TypeError("Generators must be a torch.Tensor or IntervalTensor.")
        if not isinstance(coefficients, IntervalTensor):
            raise TypeError("Coefficients must be an IntervalTensor.")
        # validate input shapes
        if generators.shape != (coefficients.shape[0], *center.shape):
            raise ValueError("Inconsistent generator, center and coefficient shapes.")
        # convert constant center and generators to intervals
        if isinstance(center, torch.Tensor):
            center = IntervalTensor(center)
        if isinstance(generators, torch.Tensor):
            generators = IntervalTensor(generators)
        self.center = center
        self.generators = generators
        self.coeff = coefficients.reshape(-1, *[1] * center.ndim)

    def __add__(
        self, other: torch.Tensor | IntervalTensor | ZonotopeTensor
    ) -> ZonotopeTensor:
        """Add two zonotopes or a zonotope plus a constant using A + B."""
        if isinstance(other, torch.Tensor | IntervalTensor):
            return ZonotopeTensor(self.center + other, self.generators, self.coeff)
        if isinstance(other, ZonotopeTensor):
            # automatic broadcasting doesn't work due to the extra dimension in the generators
            generators_s = self.generators
            generators_o = other.generators
            while generators_s.ndim < generators_o.ndim:
                generators_s = generators_s.unsqueeze(1)
            while generators_o.ndim < generators_s.ndim:
                generators_o = generators_o.unsqueeze(1)
            return ZonotopeTensor(
                self.center + other.center, generators_s + generators_o, self.coeff
            )
        return NotImplemented

    def __mul__(self, other: torch.Tensor) -> ZonotopeTensor:
        """Multiply a zonotope by a constant using A * B."""
        if isinstance(other, torch.Tensor):
            center = self.center * other
            generators = self.generators * other
            return ZonotopeTensor(center, generators, self.coeff)
        return NotImplemented

    def __matmul__(
        self, other: torch.Tensor | IntervalTensor | ZonotopeTensor
    ) -> ZonotopeTensor:
        """Matrix product of two zonotopes A @ B."""
        if isinstance(other, torch.Tensor | IntervalTensor):
            center = self.center @ other
            generators = {k: v @ other for k, v in self.generators.items()}
            return ZonotopeTensor(center, generators, self.coeff)
        if isinstance(other, ZonotopeTensor):
            # handle self.center * other.center
            center = self.center @ other.center
            # handle first order terms
            generators = self.center @ other.generators + self.generators @ other.center
            # handle second order terms - we compute the pairwise product of all the generators
            second_order_generators = self.generators.unsqueeze(
                0
            ) @ other.generators.unsqueeze(1)
            second_order_coeff = self.coeff.unsqueeze(0) * other.coeff.unsqueeze(1)
            # accumulate the generator terms with the same coefficients
            second_order_generators = (
                second_order_generators + second_order_generators.transpose(0, 1)
            ) / 2
            # add the contribution of the second order terms to the new center
            center = center + (second_order_coeff * second_order_generators).sum((0, 1))
            return ZonotopeTensor(center, generators, self.coeff)
        return NotImplemented

    def __rmatmul__(
        self, other: torch.Tensor | IntervalTensor | ZonotopeTensor
    ) -> ZonotopeTensor:
        """Matrix product of two zonotopes A @ B."""
        if isinstance(other, torch.Tensor | IntervalTensor):
            center = other @ self.center
            generators = other @ self.generators
            return ZonotopeTensor(center, generators, self.coeff)
        if isinstance(other, ZonotopeTensor):
            center = other.center @ self.center
            generators = other.center @ self.generators + self.generators @ other.center
            second_order_term = (self.coeff * other.coeff) * (
                self.generators @ other.generators
            )
            center = center + second_order_term.sum(dim=0)
            return ZonotopeTensor(center, generators, self.coeff)
        return NotImplemented

    def relu(self) -> ZonotopeTensor:
        """Pass the zonotope through a ReLU activation function using the parallel relaxation."""
        # compute the coefficients of the relaxation
        conc = self.concretize()
        lower, upper = conc.lb, conc.ub
        # compute the slope of the relaxation
        slope = torch.zeros_like(lower)
        slope[lower > 0] = 1
        slope[(lower < 0) & (upper > 0)] = (upper / (upper - lower))[
            (lower < 0) & (upper > 0)
        ]
        # compute the bias of the relaxation
        bias = torch.zeros_like(lower)
        bias[(lower < 0) & (upper > 0)] = -(upper * lower / (upper - lower))[
            (lower < 0) & (upper > 0)
        ]
        # compute the new zonotope
        center = slope * self.center + IntervalTensor(torch.zeros_like(bias), bias)
        generators = slope * self.generators
        return ZonotopeTensor(center, generators, self.coeff)

    @property
    def T(self) -> ZonotopeTensor:
        """Transpose the zonotope."""
        center = self.center.T
        generators = self.generators.mT
        return ZonotopeTensor(center, generators, self.coeff)

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the zonotope."""
        return self.center.shape

    def concretize(self) -> IntervalTensor:
        """Concretize the zonotope to an interval."""
        return self.center + (self.coeff * self.generators).sum(dim=0)

    def to(self, *args, **kwargs) -> ZonotopeTensor:
        """Move the zonotope to the given device."""
        center = self.center.to(*args, **kwargs)
        generators = self.generators.to(*args, **kwargs)
        coeff = self.coeff.to(*args, **kwargs)
        return ZonotopeTensor(center, generators, coeff)
