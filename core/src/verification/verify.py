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
    logits: IntervalTensor, targets: torch.Tensor, *, lower: bool = True,
    aggregation: str = 'min'
) -> torch.Tensor:
    """
    Compute a bound on the accuracy of a model for multi-label problems.

    For the lower bound (worst-case), a sample is certified correct when the
    best valid action's lower-bound logit exceeds every invalid action's
    upper-bound logit, i.e.:

        max_{k ∈ valid} logits_l[k]  >  max_{j ∉ valid} logits_u[j]

    This mirrors the single-label ``bound_accuracy`` logic and is sound: any
    model whose parameters sit inside the interval is guaranteed to predict a
    valid action for that sample.

    Args:
        logits: IntervalTensor containing logit bounds
        targets: Multi-hot tensor of shape (batch_size, n_classes) where 1 indicates
                a valid action and 0 indicates an invalid action.
        lower: Whether to compute lower bound (True) or upper bound (False)
        aggregation: Method to aggregate per-sample correctness into a single bound.

    Returns:
        Aggregated accuracy bound tensor
    """
    logits_l, logits_u = logits

    valid_mask = targets.bool()
    invalid_mask = ~valid_mask

    if lower:
        # Worst-case: valid logits at their minimum, invalid logits at their maximum.
        # Best valid logit lower bound vs worst invalid logit upper bound.
        NEG_INF = torch.tensor(float('-inf'), device=logits_l.device)

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

    if aggregation == 'min':
        return correct.float().min()
    elif aggregation == 'mean':
        return correct.float().mean()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")


def bound_multi_label_soft_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, T=10, lower: bool = True, aggregation: str = 'min'
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
        T: Temperature parameter for softmax (as T approaches infinity, soft accuracy approaches hard accuracy)
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
        probabilities = torch.nn.functional.softmax(worst_case_logits * T, dim=1)
    else:
        # Best-case: maximise probability on valid actions
        best_case_logits = valid_mask_float * logits_u + (1 - valid_mask_float) * logits_l
        probabilities = torch.nn.functional.softmax(best_case_logits * T, dim=1)

    # Sum probability mass on valid actions
    correct_probs = (probabilities * valid_mask_float).sum(dim=1)

    if aggregation == 'min':
        return correct_probs.min()
    elif aggregation == 'mean':
        return correct_probs.mean()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
