"""
User-facing pointwise and dataset-level verification API.

This module unifies input-region and parameter-region (Rashomon set) uncertainty: both
flow into the same `BoundedModel.bound_forward(x_l, x_u)` call, since parameter bounds
are a property of the `BoundedModel` instance itself (`param_l`/`param_u`), separate
from the input bounds passed to `bound_forward`. There is no separate "Rashomon mode"
vs "input-region mode" - they are the same code path, optionally combined.

No new bound-propagation or certification math is introduced here. Everything bottoms
out in `abstract_gradient_training.bounded_models.BoundedModel.bound_forward` and the
existing functions in `src.verification.verify`.
"""

from __future__ import annotations

import dataclasses

import torch

from abstract_gradient_training.bounded_models import BoundedModel
from src.verification import verify
from src.verification.interval_tensor import IntervalTensor
from src.verification.registry import get_method
from src.verification.compatibility import check_model_compatibility


@dataclasses.dataclass
class AdmissibleSet:
    """
    Specifies the admissible (valid) output classes for a verification query.

    Exactly one of `valid_indices` or `multi_hot` should be provided. `as_multi_hot`
    normalises either representation into the (batch, n_classes) multi-hot tensor shape
    that `verify.bound_multi_label_accuracy` expects as its `targets` argument.
    """

    n_classes: int
    valid_indices: list[int] | None = None
    multi_hot: torch.Tensor | None = None  # shape (n_classes,) or (batch, n_classes)

    def as_multi_hot(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a (batch_size, n_classes) boolean mask of admissible classes."""
        if self.multi_hot is not None:
            mask = self.multi_hot.to(device=device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            return mask
        if self.valid_indices is not None:
            mask = torch.zeros(self.n_classes, dtype=torch.bool, device=device)
            mask[torch.tensor(self.valid_indices, device=device)] = True
            return mask.unsqueeze(0).expand(batch_size, -1)
        raise ValueError("AdmissibleSet requires either valid_indices or multi_hot to be set.")


@dataclasses.dataclass
class VerificationResult:
    """Result of a (pointwise or dataset) admissible-action-set certification."""

    certified: torch.Tensor  # bool, shape (batch,) - per-sample pass/fail
    margin: torch.Tensor | None  # soft margin per sample, shape (batch,), if soft=True
    logits_l: torch.Tensor
    logits_u: torch.Tensor
    method: str

    @property
    def all_certified(self) -> bool:
        """True iff every sample in the batch is certified."""
        return bool(self.certified.all().item())

    @property
    def certified_fraction(self) -> float:
        """Fraction of samples in the batch that are certified."""
        return float(self.certified.float().mean().item())


def build_bounded_model(
    model: torch.nn.Sequential,
    method: str,
    *,
    param_l: list[torch.Tensor] | None = None,
    param_u: list[torch.Tensor] | None = None,
    **method_kwargs,
) -> BoundedModel:
    """
    Construct a BoundedModel for `model` using the named verification `method`,
    validating layer compatibility up-front with a full violation report.

    Args:
        model (torch.nn.Sequential): The network to verify.
        method (str): Name of a registered verification method (see `registry.py`,
            e.g. "IBP", "CROWN", "alpha-CROWN").
        param_l (list[torch.Tensor], optional): Lower bound on each parameter of `model`.
            If given (together with `param_u`), the resulting BoundedModel encodes a
            parameter-uncertainty interval (e.g. a Rashomon-set member) in addition to
            whatever input bounds are later passed to `bound_forward`.
        param_u (list[torch.Tensor], optional): Upper bound on each parameter of `model`.
        **method_kwargs: Extra keyword arguments forwarded to the backend constructor,
            overriding the method's default kwargs.

    Returns:
        BoundedModel: A bounded model ready for `verify_point`/`verify_dataset`.

    Raises:
        UnsupportedLayerError: If `model` contains layers unsupported by `method`. The error
            lists every unsupported layer, not just the first.
        ValueError: If `method` is not a registered verification method.
    """
    spec = get_method(method)
    check_model_compatibility(model, spec.supported_modules, method_name=method)

    kwargs = {**spec.default_kwargs, **method_kwargs}
    trainable = param_l is not None or param_u is not None
    bounded_model = spec.bounded_model_cls(model, trainable=trainable, **kwargs)

    if trainable:
        if param_l is None or param_u is None:
            raise ValueError("Both param_l and param_u must be provided together.")
        for p_l_dst, p_l_src in zip(bounded_model.param_l, param_l):
            p_l_dst.data.copy_(p_l_src)
        for p_u_dst, p_u_src in zip(bounded_model.param_u, param_u):
            p_u_dst.data.copy_(p_u_src)
    return bounded_model


def verify_point(
    bounded_model: BoundedModel,
    x: torch.Tensor,
    admissible: AdmissibleSet,
    *,
    x_l: torch.Tensor | None = None,
    x_u: torch.Tensor | None = None,
    lower: bool = True,
    soft: bool = False,
    soft_temperature: float = 10.0,
    context_mask: torch.Tensor | None = None,
) -> VerificationResult:
    """
    Certify that argmax(model(x')) is in the admissible set for every x' in [x_l, x_u]
    (defaults to the point x if x_l/x_u are omitted) AND for every parameter setting in
    the bounded_model's [param_l, param_u] box (defaults to the nominal parameters if
    bounded_model was built with trainable=False). A single `bound_forward` call
    propagates both input- and parameter-interval uncertainty together.

    `x`'s leading dimension may be 1 (a single point/region) or >1 (a batch of
    independent points/regions sharing the same parameter interval) - see
    `verify_dataset` for a dataset-level convenience wrapper with optional chunking.

    Args:
        bounded_model (BoundedModel): A model built via `build_bounded_model` (or
            directly, e.g. a Rashomon-set member with parameter bounds already set).
        x (torch.Tensor): The nominal input(s), shape (batch, ...).
        admissible (AdmissibleSet): The admissible action set(s) for this query.
        x_l (torch.Tensor, optional): Lower bound on the input region. Defaults to `x`.
        x_u (torch.Tensor, optional): Upper bound on the input region. Defaults to `x`.
        lower (bool, optional): Whether to certify the worst-case (lower) bound (True,
            sound certification) or compute the best-case (upper) bound (False).
        soft (bool, optional): If True, also compute a differentiable soft margin.
        soft_temperature (float, optional): Softmax temperature used for the soft margin.
        context_mask (torch.Tensor, optional): Optional mask applied to the logits
            interval before certification.

    Returns:
        VerificationResult: Per-sample certification result.
    """
    x_l = x if x_l is None else x_l
    x_u = x if x_u is None else x_u

    logits = IntervalTensor(*bounded_model.bound_forward(x_l, x_u))
    if context_mask is not None:
        logits = logits * context_mask

    mask = admissible.as_multi_hot(x.shape[0], logits.device).float()

    certified = verify.bound_multi_label_accuracy(
        logits, mask, lower=lower, aggregation="none",
    ).bool()

    margin = None
    if soft:
        margin = verify.bound_multi_label_accuracy_margin(
            logits, mask, T=soft_temperature, lower=lower, aggregation="none",
        )

    logits_l, logits_u = logits
    return VerificationResult(
        certified=certified,
        margin=margin,
        logits_l=logits_l,
        logits_u=logits_u,
        method=type(bounded_model).__name__,
    )


def verify_dataset(
    bounded_model: BoundedModel,
    X: torch.Tensor,
    admissible: AdmissibleSet | torch.Tensor,
    *,
    X_l: torch.Tensor | None = None,
    X_u: torch.Tensor | None = None,
    lower: bool = True,
    soft: bool = False,
    soft_temperature: float = 10.0,
    context_mask: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> VerificationResult:
    """
    Dataset-level convenience wrapper over `verify_point`. `X` (and optional `X_l`, `X_u`)
    hold the whole certificate set with a leading batch dimension. `admissible` may be a
    single AdmissibleSet shared by every row (its multi-hot mask broadcasts) or a per-row
    multi-hot tensor of shape (N, n_classes).

    Batching reuses the same `bound_forward` call used by `verify_point` - there is no
    per-sample Python loop; `bound_forward` is already vectorized across the batch
    dimension by every BoundedModel backend. `batch_size` only chunks very large datasets
    to bound peak memory; results across chunks are concatenated and are identical to the
    unchunked result.

    Args:
        bounded_model (BoundedModel): A model built via `build_bounded_model` (or
            directly, e.g. a Rashomon-set member with parameter bounds already set).
        X (torch.Tensor): Nominal inputs, shape (N, ...).
        admissible (AdmissibleSet | torch.Tensor): Admissible action set(s); a raw tensor
            is interpreted as a per-row (or broadcastable) multi-hot mask.
        X_l (torch.Tensor, optional): Lower bound on each input region. Defaults to `X`.
        X_u (torch.Tensor, optional): Upper bound on each input region. Defaults to `X`.
        lower (bool, optional): See `verify_point`.
        soft (bool, optional): See `verify_point`.
        soft_temperature (float, optional): See `verify_point`.
        context_mask (torch.Tensor, optional): See `verify_point`.
        batch_size (int, optional): If given, verify in chunks of this size instead of
            all at once, to bound peak memory on large datasets.

    Returns:
        VerificationResult: Per-sample certification result over the whole dataset.
    """
    if isinstance(admissible, torch.Tensor):
        admissible = AdmissibleSet(n_classes=admissible.shape[-1], multi_hot=admissible)

    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        return verify_point(
            bounded_model, X, admissible,
            x_l=X_l, x_u=X_u, lower=lower, soft=soft,
            soft_temperature=soft_temperature, context_mask=context_mask,
        )

    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_admissible = admissible
        if admissible.multi_hot is not None and admissible.multi_hot.ndim == 2:
            chunk_admissible = AdmissibleSet(
                n_classes=admissible.n_classes, multi_hot=admissible.multi_hot[start:end],
            )
        chunks.append(
            verify_point(
                bounded_model, X[start:end], chunk_admissible,
                x_l=None if X_l is None else X_l[start:end],
                x_u=None if X_u is None else X_u[start:end],
                lower=lower, soft=soft, soft_temperature=soft_temperature,
                context_mask=context_mask,
            )
        )
    return VerificationResult(
        certified=torch.cat([c.certified for c in chunks]),
        margin=None if chunks[0].margin is None else torch.cat([c.margin for c in chunks]),
        logits_l=torch.cat([c.logits_l for c in chunks]),
        logits_u=torch.cat([c.logits_u for c in chunks]),
        method=chunks[0].method,
    )
