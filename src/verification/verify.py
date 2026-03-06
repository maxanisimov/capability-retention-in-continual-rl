"""Verification using zonotope and interval arithmetic."""

import torch

from src.verification.zonotope_tensor import ZonotopeTensor
from src.verification.interval_tensor import IntervalTensor
from src.utils.general import split_generators, InContextHead
import sklearn

# pylint: disable=not-callable


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
    logits: IntervalTensor, targets: torch.Tensor, *, T=10, lower: bool = True
) -> torch.Tensor:
    """
    Compute a lower bound on the soft accuracy of a model given its output logit interval and the true targets.
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
        worst_case_preds = torch.nn.functional.softmax(worst_case_logits * T, dim=1)
        correct_probs = worst_case_preds[
            torch.arange(worst_case_preds.size(0)), targets
        ]
    else:
        best_case_logits = targets_one_hot * logits_u + (1 - targets_one_hot) * logits_l
        best_case_preds = torch.nn.functional.softmax(best_case_logits * T, dim=1)
        correct_probs = best_case_preds[torch.arange(best_case_preds.size(0)), targets]
    return correct_probs.mean()


def bound_multi_label_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, lower: bool = True
) -> torch.Tensor:
    """
    Compute a bound on the accuracy of a model for multi-label problems.

    For the lower bound (worst-case), a sample is certified correct when the
    best valid action's lower-bound logit exceeds every invalid action's
    upper-bound logit, i.e.::

        max_{k ∈ valid} logits_l[k]  >  max_{j ∉ valid} logits_u[j]

    This mirrors the single-label ``bound_accuracy`` logic and is sound: any
    model whose parameters sit inside the interval is guaranteed to predict a
    valid action for that sample.

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Tensor where each row contains valid class indices for that sample.
                Should be padded with -1 for variable length. Shape: (batch_size, max_labels)
        lower: Whether to compute lower bound (True) or upper bound (False)

    Returns:
        Accuracy bound tensor
    """
    logits_l, logits_u = logits
    batch_size, n_classes = logits_l.shape

    # Build a boolean mask of valid actions: (batch_size, n_classes)
    valid_mask = torch.zeros(batch_size, n_classes, dtype=torch.bool, device=targets.device)
    for col_idx in range(targets.shape[1]):
        col = targets[:, col_idx]
        col_valid = col != -1
        # Scatter True into valid_mask for valid action indices
        indices = col.clamp(min=0)
        valid_mask[torch.arange(batch_size, device=targets.device)[col_valid], indices[col_valid]] = True

    invalid_mask = ~valid_mask

    if lower:
        # Worst-case: valid logits at their minimum, invalid logits at their maximum.
        # Best valid logit lower bound vs worst invalid logit upper bound.
        NEG_INF = torch.tensor(float('-inf'), device=logits_l.device)
        POS_INF = torch.tensor(float('inf'), device=logits_l.device)

        best_valid_lower = logits_l.masked_fill(~valid_mask, NEG_INF).max(dim=1).values
        worst_invalid_upper = logits_u.masked_fill(~invalid_mask, NEG_INF).max(dim=1).values

        # Where there are no invalid actions the sample is trivially correct
        no_invalid = (~invalid_mask.any(dim=1))
        correct = (best_valid_lower > worst_invalid_upper) | no_invalid
    else:
        # Best-case: valid logits at their maximum, invalid logits at their minimum.
        NEG_INF = torch.tensor(float('-inf'), device=logits_u.device)

        best_valid_upper = logits_u.masked_fill(~valid_mask, NEG_INF).max(dim=1).values
        worst_invalid_lower = logits_l.masked_fill(~invalid_mask, NEG_INF).max(dim=1).values

        no_invalid = (~invalid_mask.any(dim=1))
        correct = (best_valid_upper > worst_invalid_lower) | no_invalid

    return correct.float().mean()


def bound_multi_label_soft_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, T=10, lower: bool = True
) -> torch.Tensor:
    """
    Compute a bound on the soft accuracy of a model for multi-label problems.

    Soft accuracy is defined as the total probability mass on valid actions.
    For a sound lower bound we construct worst-case logits: lower bounds for
    valid actions and upper bounds for invalid actions (minimising the mass on
    valid classes).

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Tensor where each row contains valid class indices for that sample.
                Should be padded with -1 for variable length. Shape: (batch_size, max_labels)
        T: Temperature parameter for softmax
        lower: Whether to compute lower bound (True) or upper bound (False)

    Returns:
        Soft accuracy bound tensor
    """
    logits_l, logits_u = logits
    batch_size, n_classes = logits_l.shape

    # Build a boolean mask of valid actions: (batch_size, n_classes)
    valid_mask = torch.zeros(batch_size, n_classes, dtype=torch.bool, device=targets.device)
    for col_idx in range(targets.shape[1]):
        col = targets[:, col_idx]
        col_valid = col != -1
        indices = col.clamp(min=0)
        valid_mask[torch.arange(batch_size, device=targets.device)[col_valid], indices[col_valid]] = True

    valid_mask_float = valid_mask.float()

    if lower:
        # Worst-case: minimise probability on valid actions
        # valid logits at lower bound, invalid logits at upper bound
        worst_case_logits = valid_mask_float * logits_l + (1 - valid_mask_float) * logits_u
        probabilities = torch.nn.functional.softmax(worst_case_logits * T, dim=1)
    else:
        # Best-case: maximise probability on valid actions
        best_case_logits = valid_mask_float * logits_u + (1 - valid_mask_float) * logits_l
        probabilities = torch.nn.functional.softmax(best_case_logits * T, dim=1)

    # Sum probability mass on valid actions
    correct_probs = (probabilities * valid_mask_float).sum(dim=1)

    return correct_probs.mean()

def bound_multi_label_lse_margin(
    logits: IntervalTensor, targets: torch.Tensor, *, T: float = 10, lower: bool = True
) -> torch.Tensor:
    """
    Compute a bound on the LSE margin surrogate for multi-label safety.

    The LSE margin surrogate is defined as:
        phi_tau = tau * log(sum_{safe} exp(z_a / tau)) - tau * log(sum_{unsafe} exp(z_a / tau))

    This approximates the true margin:
        m = max_{safe} z_a - max_{unsafe} z_a

    with a controllable gap that vanishes as tau -> 0. Safety (phi^safe = 1)
    is guaranteed when phi_tau > tau * log(N - K).

    For a sound lower bound on the surrogate, we construct worst-case logits:
    lower bounds for safe actions and upper bounds for unsafe actions.

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Tensor where each row contains valid class indices for that sample.
                 Should be padded with -1 for variable length. Shape: (batch_size, max_labels)
        T: Temperature parameter. Higher values give tighter approximation
             of the true margin but sharper gradients.
        lower: Whether to compute lower bound (True) or upper bound (False)

    Returns:
        LSE margin surrogate bound tensor (per-sample mean)
    """
    logits_l, logits_u = logits
    batch_size, n_classes = logits_l.shape
    tau = 1 / T

    # Build a boolean mask of valid (safe) actions: (batch_size, n_classes)
    valid_mask = torch.zeros(batch_size, n_classes, dtype=torch.bool, device=targets.device)
    for col_idx in range(targets.shape[1]):
        col = targets[:, col_idx]
        col_valid = col != -1
        indices = col.clamp(min=0)
        valid_mask[torch.arange(batch_size, device=targets.device)[col_valid], indices[col_valid]] = True

    valid_mask_float = valid_mask.float()
    invalid_mask_float = 1.0 - valid_mask_float

    if lower:
        # Worst-case: minimise the margin (safe logits low, unsafe logits high)
        safe_logits = valid_mask_float * logits_l + invalid_mask_float * (-1e9)
        unsafe_logits = invalid_mask_float * logits_u + valid_mask_float * (-1e9)
    else:
        # Best-case: maximise the margin (safe logits high, unsafe logits low)
        safe_logits = valid_mask_float * logits_u + invalid_mask_float * (-1e9)
        unsafe_logits = invalid_mask_float * logits_l + valid_mask_float * (-1e9)

    # Compute LSE margin: tau * logsumexp(safe / tau) - tau * logsumexp(unsafe / tau)
    lse_safe = tau * torch.logsumexp(safe_logits / tau, dim=1)
    lse_unsafe = tau * torch.logsumexp(unsafe_logits / tau, dim=1)

    margin_surrogate = lse_safe - lse_unsafe

    return margin_surrogate.mean()
