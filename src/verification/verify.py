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
    Compute a lower bound on the accuracy of a model for multi-label problems.
    
    Args:
        logits: IntervalTensor containing logit bounds
        targets: Tensor where each row contains valid class indices for that sample.
                Should be padded with -1 for variable length. Shape: (batch_size, max_labels)
        lower: Whether to compute lower bound (True) or upper bound (False)
    
    Returns:
        Accuracy bound tensor
    """
    logits_l, logits_u = logits
    batch_size = targets.shape[0]
    
    if lower:
        # For lower bound, we want the worst-case scenario
        # Use lower bounds for all classes to get conservative predictions
        worst_case_logits = logits_l
        predictions = worst_case_logits.argmax(dim=1)
    else:
        # For upper bound, we want the best-case scenario  
        # Use upper bounds for all classes to get optimistic predictions
        best_case_logits = logits_u
        predictions = best_case_logits.argmax(dim=1)
    
    # Check if predictions are in the set of valid targets for each sample
    correct_predictions = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
    
    for i in range(batch_size):
        # Get valid targets for this sample (exclude -1 padding)
        valid_targets = targets[i][targets[i] != -1]
        if len(valid_targets) > 0:
            # Check if prediction is in the set of valid targets
            correct_predictions[i] = predictions[i].item() in valid_targets
    
    return correct_predictions.float().mean()


def bound_multi_label_soft_accuracy(
    logits: IntervalTensor, targets: torch.Tensor, *, T=10, lower: bool = True
) -> torch.Tensor:
    """
    Compute a lower bound on the soft accuracy of a model for multi-label problems.
    
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
    batch_size = targets.shape[0]
    
    if lower:
        # For lower bound, use worst-case logits
        worst_case_logits = logits_l
        probabilities = torch.nn.functional.softmax(worst_case_logits * T, dim=1)
    else:
        # For upper bound, use best-case logits
        best_case_logits = logits_u
        probabilities = torch.nn.functional.softmax(best_case_logits * T, dim=1)
    
    # Compute probability mass assigned to valid targets for each sample
    correct_probs = torch.zeros(batch_size, device=targets.device)
    
    for i in range(batch_size):
        # Get valid targets for this sample (exclude -1 padding)
        valid_targets = targets[i][targets[i] != -1]
        if len(valid_targets) > 0:
            # Sum probabilities for all valid target classes
            correct_probs[i] = probabilities[i, valid_targets].sum()
    
    return correct_probs.mean()
