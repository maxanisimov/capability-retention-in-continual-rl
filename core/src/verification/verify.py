"""Verification using zonotope and interval arithmetic."""

from typing import Literal

import torch

from src.verification.zonotope_tensor import ZonotopeTensor
from src.IntervalTensor import IntervalTensor
from src.utils.general import split_generators, InContextHead
import sklearn

# pylint: disable=not-callable

SurrogateForm = Literal["auto", "logsumexp"]
ResolvedSurrogateForm = Literal["probability", "logsumexp"]


def resolve_surrogate_form(mode: str, surrogate: str) -> ResolvedSurrogateForm:
    """Resolve the public surrogate option to the concrete per-sample formula."""
    _validate_multi_label_mode(mode)
    if surrogate not in {"auto", "logsumexp"}:
        raise ValueError(
            f"Unsupported surrogate form: {surrogate!r}. Expected 'auto' or 'logsumexp'."
        )
    if surrogate == "logsumexp" or mode == "all":
        return "logsumexp"
    return "probability"


def bound_forward_pass(
    model: torch.nn.Sequential,
    generators: torch.Tensor,
    coefficients: IntervalTensor,
    inputs: torch.Tensor,
    use_zonotopes: bool = True,
) -> IntervalTensor:
    """
    Compute bounds on the output of a neural network using interval and zonotope arithmetic.

    Args:
        model (torch.nn.Sequential): The neural network model (used as the center of the zonotopes).
        generators (torch.Tensor): The generators of the zonotope over parameter space.
        coefficients (IntervalTensor): The coefficients of the zonotope.
        inputs (torch.Tensor): The input tensor to the model.
        use_zonotopes (bool): Whether to use zonotope verification. If False, uses interval tensors.

    Returns:
        IntervalTensor: The output bounds of the model.
    """
    # Construct zonotopes for each parameter of the network and convert them to interval tensors
    centers = list(model.parameters())
    generators = split_generators(generators, model)
    parameters = [
        ZonotopeTensor(p_c, p_g, coefficients) for p_c, p_g in zip(centers, generators)
    ]

    # If not using zonotopes, convert the parameters to interval tensors
    if not use_zonotopes:
        parameters = [p.concretize() for p in parameters]

    # Pass the input through the zonotope representation of the network
    x = IntervalTensor(inputs)
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            w, b = parameters.pop(0), parameters.pop(0)
            x = x @ w.T + b
        elif isinstance(layer, torch.nn.ReLU):
            x = x.relu()
        elif isinstance(layer, torch.nn.Tanh):
            x = x.tanh()
        elif isinstance(layer, torch.nn.Flatten):
            x = x.flatten(start_dim=1)
        elif isinstance(layer, InContextHead):
            x = x * layer.mask
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    return x.concretize()


def bound_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, lower: bool = True
) -> torch.Tensor:
    """
    Compute a lower bound on the accuracy of a model given its output logit interval and the true targets.
    """
    targets = targets.squeeze(dim=-1)
    targets_one_hot = torch.nn.functional.one_hot(
        targets, num_classes=logits.shape[1]
    ).float()  # type: ignore
    logits_l, logits_u = logits
    if lower:
        worst_case_logits = (
            targets_one_hot * logits_l + (1 - targets_one_hot) * logits_u
        )
        worst_case_preds = worst_case_logits.argmax(dim=1)
        acc_bound = (worst_case_preds == targets).float().mean()
    else:
        best_case_logits = targets_one_hot * logits_u + (1 - targets_one_hot) * logits_l
        best_case_preds = best_case_logits.argmax(dim=1)
        acc_bound = (best_case_preds == targets).float().mean()
    return acc_bound


def bound_balanced_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, lower: bool = True
) -> torch.Tensor:
    """
    Compute a lower bound on the balanced accuracy (average of recall for each class)  of a
    model given its output logit interval and the true targets.
    """
    targets = targets.squeeze(dim=-1)
    targets_one_hot = torch.nn.functional.one_hot(
        targets, num_classes=logits.shape[1]
    ).float()  # type: ignore
    logits_l, logits_u = logits
    if lower:
        worst_case_logits = (
            targets_one_hot * logits_l + (1 - targets_one_hot) * logits_u
        )
        worst_case_preds = worst_case_logits.argmax(dim=1)
        # acc_bound = (worst_case_preds == targets).float().mean()
        acc_bound = sklearn.metrics.balanced_accuracy_score(
            targets.cpu().numpy(), worst_case_preds.cpu().numpy()
        )
    else:
        best_case_logits = targets_one_hot * logits_u + (1 - targets_one_hot) * logits_l
        best_case_preds = best_case_logits.argmax(dim=1)
        # acc_bound = (best_case_preds == targets).float().mean()
        acc_bound = sklearn.metrics.balanced_accuracy_score(
            targets.cpu().numpy(), best_case_preds.cpu().numpy()
        )
    return acc_bound


def bound_soft_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, tau: float = 0.1, lower: bool = True
) -> torch.Tensor:
    """
    Compute a lower bound on the soft accuracy of a model given its output logit interval and
    the true targets. `tau` is a standard softmax temperature: `softmax(logits / tau)`. As
    `tau -> 0`, the softmax sharpens toward argmax; as `tau -> infinity`, it flattens toward
    uniform.
    """
    targets = targets.squeeze(dim=-1)
    targets_one_hot = torch.nn.functional.one_hot(
        targets, num_classes=logits.shape[1]
    ).float()  # type: ignore
    logits_l, logits_u = logits
    if lower:
        worst_case_logits = (
            targets_one_hot * logits_l + (1 - targets_one_hot) * logits_u
        )
        worst_case_preds = torch.nn.functional.softmax(worst_case_logits / tau, dim=1)
        correct_probs = worst_case_preds[
            torch.arange(worst_case_preds.size(0)), targets
        ]
    else:
        best_case_logits = targets_one_hot * logits_u + (1 - targets_one_hot) * logits_l
        best_case_preds = torch.nn.functional.softmax(best_case_logits / tau, dim=1)
        correct_probs = best_case_preds[torch.arange(best_case_preds.size(0)), targets]
    return correct_probs.mean()


def _aggregate_per_sample(values: torch.Tensor, aggregation: str) -> torch.Tensor:
    if aggregation == 'min':
        return values.min()
    elif aggregation == 'mean':
        return values.mean()
    elif aggregation == 'none':
        return values
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")


def _validate_multi_label_mode(mode: str) -> None:
    if mode not in {'any', 'all'}:
        raise ValueError(f"Unsupported multi-label mode: {mode!r}. Expected 'any' or 'all'.")


def bound_multi_label_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, lower: bool = True,
    aggregation: str = 'min', mode: str = 'any'
) -> torch.Tensor:
    """
    Compute a bound on the accuracy of a model for multi-label problems.

    With ``mode='any'`` (the historical behavior), a lower-bound sample is
    certified correct when the best valid action's lower-bound logit exceeds
    every invalid action's upper-bound logit, i.e.:

        max_{k ∈ valid} logits_l[k]  >  max_{j ∉ valid} logits_u[j]

    With ``mode='all'``, every valid action must beat every invalid action:

        min_{k ∈ valid} logits_l[k]  >  max_{j ∉ valid} logits_u[j]

    This mirrors the single-label ``bound_accuracy`` logic and is sound: any
    model whose parameters sit inside the interval is guaranteed to predict a
    valid action for that sample.

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Multi-hot tensor of shape (batch_size, n_classes) where 1 indicates
                a valid action and 0 indicates an invalid action.
        lower: Whether to compute lower bound (True) or upper bound (False)
        aggregation: Method to aggregate per-sample correctness into a single bound.
        mode: ``'any'`` certifies at least one valid argmax winner; ``'all'`` certifies
            all valid logits above all invalid logits.

    Returns:
        Aggregated accuracy bound tensor
    """
    _validate_multi_label_mode(mode)
    logits_l, logits_u = logits

    valid_mask = targets.bool()
    invalid_mask = ~valid_mask
    has_valid = valid_mask.any(dim=1)
    no_invalid = (~invalid_mask.any(dim=1))

    if lower:
        # Worst-case: valid logits at their minimum, invalid logits at their maximum.
        # Valid logit lower bound vs worst invalid logit upper bound.
        NEG_INF = torch.tensor(float('-inf'), device=logits_l.device)
        POS_INF = torch.tensor(float('inf'), device=logits_l.device)

        if mode == 'any':
            valid_lower = logits_l.masked_fill(~valid_mask, NEG_INF).max(dim=1).values
        else:
            valid_lower = logits_l.masked_fill(~valid_mask, POS_INF).min(dim=1).values
        worst_invalid_upper = logits_u.masked_fill(~invalid_mask, NEG_INF).max(dim=1).values

        # Where there are no invalid actions the sample is trivially correct
        correct = ((valid_lower > worst_invalid_upper) & has_valid) | no_invalid
    else:
        # Best-case: valid logits at their maximum, invalid logits at their minimum.
        NEG_INF = torch.tensor(float('-inf'), device=logits_u.device)
        POS_INF = torch.tensor(float('inf'), device=logits_u.device)

        if mode == 'any':
            valid_upper = logits_u.masked_fill(~valid_mask, NEG_INF).max(dim=1).values
        else:
            valid_upper = logits_u.masked_fill(~valid_mask, POS_INF).min(dim=1).values
        worst_invalid_lower = logits_l.masked_fill(~invalid_mask, NEG_INF).max(dim=1).values

        correct = ((valid_upper > worst_invalid_lower) & has_valid) | no_invalid

    return _aggregate_per_sample(correct.float(), aggregation)


def bound_multi_label_soft_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, tau: float = 0.1, lower: bool = True, aggregation: str = 'min'
) -> torch.Tensor:
    """
    Compute a bound on the soft accuracy of a model for multi-label problems.

    Soft accuracy is defined as the total softmax mass on valid actions.
    For a sound lower bound we construct worst-case logits: lower bounds for
    valid actions and upper bounds for invalid actions (minimising the mass on
    valid classes).

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Multi-hot tensor of shape (batch_size, n_classes) where 1 indicates
                a valid action and 0 indicates an invalid action.
        tau: Standard softmax temperature: `softmax(logits / tau)`. As `tau -> 0`, soft
            accuracy approaches hard accuracy (sharpens toward argmax).
        lower: Whether to compute lower bound (True) or upper bound (False)
        aggregation: Method to aggregate per-sample correctness into a single bound.
    Returns:
        Soft accuracy bound tensor
    """
    logits_l, logits_u = logits

    valid_mask = targets.bool()
    valid_mask_float = valid_mask.float()

    if lower:
        # Worst-case: minimise probability on valid actions
        # valid logits at lower bound, invalid logits at upper bound
        worst_case_logits = valid_mask_float * logits_l + (1 - valid_mask_float) * logits_u
        probabilities = torch.nn.functional.softmax(worst_case_logits / tau, dim=1)
    else:
        # Best-case: maximise probability on valid actions
        best_case_logits = valid_mask_float * logits_u + (1 - valid_mask_float) * logits_l
        probabilities = torch.nn.functional.softmax(best_case_logits / tau, dim=1)

    # Sum probability mass on valid actions
    correct_probs = (probabilities * valid_mask_float).sum(dim=1)

    return _aggregate_per_sample(correct_probs, aggregation)


def bound_multi_label_logsumexp_accuracy_margin(
    logits: IntervalTensor,
    targets: torch.Tensor,
    *,
    tau: float = 0.1,
    lower: bool = True,
    aggregation: str = "min",
    mode: str = "any",
) -> torch.Tensor:
    """Compute the temperature-scaled LogSumExp admissible-action margin.

    ``mode="any"`` uses a lower smooth approximation to the best valid logit
    and an upper smooth approximation to the best invalid logit::

        tau * LSE(valid / tau) - tau * log(n_valid)
        - tau * LSE(invalid / tau)

    The cardinality correction is required for soundness: a positive margin
    guarantees that one individual valid action beats every invalid action.

    ``mode="all"`` uses the lower smooth approximation to the worst valid
    logit already used historically by the Rashomon engine::

        -tau * LSE(-valid / tau) - tau * LSE(invalid / tau)

    Rows with no invalid actions are unconstrained and receive a finite positive
    margin. Rows with no valid actions fail closed with a finite negative margin.
    """
    _validate_multi_label_mode(mode)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    logits_l, logits_u = logits
    valid_mask = targets.bool()
    invalid_mask = ~valid_mask
    has_valid = valid_mask.any(dim=1)
    has_invalid = invalid_mask.any(dim=1)

    source_valid = logits_l if lower else logits_u
    source_invalid = logits_u if lower else logits_l
    neg_inf = torch.tensor(float("-inf"), device=source_valid.device, dtype=source_valid.dtype)

    invalid_terms = (source_invalid / float(tau)).masked_fill(~invalid_mask, neg_inf)
    # Avoid differentiating logsumexp over an all--inf row before replacing the
    # unconstrained result below; that backward pass otherwise produces NaNs.
    invalid_terms = torch.where(
        has_invalid.unsqueeze(1), invalid_terms, torch.zeros_like(invalid_terms)
    )
    invalid_lse = float(tau) * torch.logsumexp(invalid_terms, dim=1)

    if mode == "any":
        valid_terms = (source_valid / float(tau)).masked_fill(~valid_mask, neg_inf)
        valid_terms = torch.where(
            has_valid.unsqueeze(1), valid_terms, torch.zeros_like(valid_terms)
        )
        valid_lse = float(tau) * torch.logsumexp(valid_terms, dim=1)
        valid_count = valid_mask.sum(dim=1).to(dtype=source_valid.dtype).clamp_min(1.0)
        margins = valid_lse - float(tau) * valid_count.log() - invalid_lse
    else:
        valid_terms = (-source_valid / float(tau)).masked_fill(~valid_mask, neg_inf)
        valid_terms = torch.where(
            has_valid.unsqueeze(1), valid_terms, torch.zeros_like(valid_terms)
        )
        valid_softmin = -float(tau) * torch.logsumexp(valid_terms, dim=1)
        margins = valid_softmin - invalid_lse

    margins = torch.where(has_invalid, margins, torch.ones_like(margins))
    margins = torch.where(has_valid, margins, -torch.ones_like(margins))
    return _aggregate_per_sample(margins, aggregation)


def bound_multi_label_accuracy_margin(
    logits: IntervalTensor, targets: torch.Tensor, *, tau: float = 0.1, lower: bool = True,
    aggregation: str = 'min', mode: str = 'any', surrogate: SurrogateForm = "auto",
) -> torch.Tensor:
    """
    Compute a soft margin for certifying multi-label accuracy.

    With ``mode='any'``, this computes the softmax probability mass assigned to
    valid actions and subtracts the threshold ``k / (k + 1)``, where ``k`` is
    the number of valid actions for that sample. A positive margin is a
    sufficient condition that at least one valid action has higher softmax
    probability than every invalid action.

    With ``mode='all'``, this uses a smooth lower bound on the strict logit
    ordering ``min(valid logits) > max(invalid logits)``:

        softmin_tau(valid_logits) - softmax_tau(invalid_logits)

    When ``lower`` is True, the probability mass is computed from worst-case
    interval logits: valid actions use their lower bounds and invalid actions
    use their upper bounds. When ``lower`` is False, the best-case interval
    logits are used instead.

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Multi-hot tensor of shape (batch_size, n_classes) where 1 indicates
                a valid action and 0 indicates an invalid action.
        tau: Standard softmax temperature: `softmax(logits / tau)`. As `tau -> 0`, the
            softmax margin sharpens toward an argmax-style margin; as `tau -> infinity`,
            it flattens toward uniform.
        lower: Whether to compute the worst-case lower-bound margin (True) or
            best-case upper-bound margin (False).
        aggregation: Method to aggregate per-sample margins ('mean' or 'min').
        mode: ``'any'`` preserves the historical admissible-argmax semantics;
            ``'all'`` requires every valid logit to beat every invalid logit.
        surrogate: ``'auto'`` preserves the historical formulas (probability
            mass for ``'any'``, LogSumExp for ``'all'``). ``'logsumexp'`` uses
            the temperature-scaled LogSumExp margin for either mode.
    Returns:
        Aggregated margin. Values >= 0 indicate the chosen aggregate clears the
        soundness threshold; negative values indicate a threshold violation.
    """
    resolved_surrogate = resolve_surrogate_form(mode, surrogate)
    if resolved_surrogate == "logsumexp":
        return bound_multi_label_logsumexp_accuracy_margin(
            logits,
            targets,
            tau=tau,
            lower=lower,
            aggregation=aggregation,
            mode=mode,
        )
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")
    logits_l, logits_u = logits

    valid_mask = targets.bool()
    valid_mask_float = valid_mask.float()

    if lower:
        # Worst-case: minimise probability on valid actions
        # valid logits at lower bound, invalid logits at upper bound
        worst_case_logits = valid_mask_float * logits_l + (1 - valid_mask_float) * logits_u
        probabilities = torch.nn.functional.softmax(worst_case_logits / tau, dim=1)
    else:
        # Best-case: maximise probability on valid actions
        best_case_logits = valid_mask_float * logits_u + (1 - valid_mask_float) * logits_l
        probabilities = torch.nn.functional.softmax(best_case_logits / tau, dim=1)

    # Sum probability mass on valid actions
    correct_probs = (probabilities * valid_mask_float).sum(dim=1)

    adm_set_cardinalities = valid_mask.sum(dim=1)
    sound_thresholds = adm_set_cardinalities / (1 + adm_set_cardinalities)

    margins = correct_probs - sound_thresholds

    return _aggregate_per_sample(margins, aggregation)
