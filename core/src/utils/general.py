import torch
import random
import numpy as np
from abstract_gradient_training.bounded_models import BoundedModel


def split_generators(
    generators: torch.Tensor, model: torch.nn.Sequential
) -> list[torch.Tensor]:
    """
    Split the flattened generator tensor into a list of tensor each corresponding to a parameter in the model.

    Args:
        generators (torch.Tensor): The flattened generator tensor of shape (num_generators, num_params).
        model (torch.nn.Sequential): The neural network model with parameters of shape p_i.

    Returns:
        list[torch.Tensor]: A list of tensors of shape (num_generators, *p_i) for each parameter in the model.
    """
    param_sizes = [param.numel() for param in model.parameters()]
    n_params = sum(param_sizes)
    generators = generators.reshape(-1, n_params)
    if generators.ndim != 2:
        raise ValueError(
            f"Expected generator tensor of shape (num_generators, num_params), got {generators.shape}"
        )
    if generators.shape[1] != n_params:
        raise ValueError(
            f"Expected generator tensor with {n_params} columns, got {generators.shape[1]}"
        )

    split_generator = torch.split_with_sizes(generators, param_sizes, dim=1)
    split_generator = [
        e.reshape(-1, *param.shape)
        for e, param in zip(split_generator, model.parameters())
    ]
    return split_generator


def sort_parameter_bounds_by_width(
    param_bounds_l: list[torch.Tensor],
    param_bounds_u: list[torch.Tensor],
    descending: bool = True,
) -> list[dict[str, int | float]]:
    """
    Sort parameter intervals by element-wise width.

    Each output item corresponds to one scalar parameter and contains:
    - ``tensor_index``: index of the parameter tensor in the bounds list.
    - ``flat_index``: index of the scalar in the flattened parameter tensor.
    - ``width``: interval width (upper - lower) for that scalar parameter.
    """
    if len(param_bounds_l) != len(param_bounds_u):
        raise ValueError(
            "Expected the same number of lower and upper bound tensors, "
            f"got {len(param_bounds_l)} and {len(param_bounds_u)}."
        )

    width_entries: list[dict[str, int | float]] = []
    for tensor_index, (p_l, p_u) in enumerate(zip(param_bounds_l, param_bounds_u)):
        if p_l.shape != p_u.shape:
            raise ValueError(
                "Lower and upper bounds must have matching shapes for each tensor, "
                f"but tensor {tensor_index} has {p_l.shape} and {p_u.shape}."
            )

        widths = (p_u.reshape(-1) - p_l.reshape(-1)).detach().cpu()
        if (widths < 0).any():
            min_width = widths.min().item()
            raise ValueError(
                "Upper bounds must be >= lower bounds for all elements, "
                f"but tensor {tensor_index} has minimum width {min_width:.6e}."
            )

        for flat_index, width in enumerate(widths.tolist()):
            width_entries.append(
                {
                    "tensor_index": tensor_index,
                    "flat_index": flat_index,
                    "width": float(width),
                }
            )

    width_entries.sort(key=lambda item: item["width"], reverse=descending)
    return width_entries


class InContextHead(torch.nn.Module):
    """A model head that masks out output logits not corresponding to the current context."""

    def __init__(self, context_list: list, context_dim: int, device: str = "cuda"):
        super().__init__()
        self.mask = torch.ones(context_dim).to(device)
        self.current_context = None  # None means the global context
        self.context_list = context_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_context is None:
            return x
        return x * self.mask

    def set_context(self, context: int) -> None:
        if context == -1:
            self.current_context = None
            self.mask = torch.ones_like(self.mask)
            return
        if context < 0 or context >= len(self.context_list):
            raise ValueError(f"Context {context} is out of range.")
        self.current_context = context
        self.mask = torch.zeros_like(self.mask)
        for i in self.context_list[context]:
            self.mask[i] = 1

def set_seed(seed: int = 42, device: str = "cuda") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_inclusion(
    model: torch.nn.Module | BoundedModel,
    bounded_model: BoundedModel,
) -> int:
    """Return 0 if the model parameters are within the bounds of the bounded model, or the number of parameters that are not."""
    num_violations = 0
    if isinstance(model, torch.nn.Module):
        for p_l, p, p_u in zip(
            bounded_model.param_l, model.parameters(), bounded_model.param_u
        ):
            num_violations += (p_l > p).sum().item() + (p > p_u).sum().item()
    elif isinstance(model, BoundedModel):
        for p_l, p_u, pp_l, pp_u in zip(
            bounded_model.param_l, bounded_model.param_u, model.param_l, model.param_u
        ):
            num_violations += ((pp_l < p_l) | (pp_u > p_u)).sum().item()
    return num_violations


def print_bold(text):
    """
    Prints the given text in bold using ANSI escape codes.
    """
    # ANSI escape code for bold is \033[1m
    # ANSI escape code to reset all attributes is \033[0m
    bold_start = "\033[1m"
    bold_end = "\033[0m"

    print(bold_start + text + bold_end)


def print_colored(text: str, color: str):
    """
    Prints text to the console in a specified color.

    Args:
        text (str): The text to print.
        color (str): The desired color. Accepts 'green', 'red', or 'amber'.
    """
    # ANSI color codes
    colors = {
        "green": "\033[92m",  # 🟩
        "amber": "\033[93m",  # 🟨
        "red": "\033[91m",  # 🟥
    }
    reset_code = "\033[0m"

    color_code = colors.get(color.lower())

    if color_code:
        print(f"{color_code}{text}{reset_code}")
    else:
        # If the color is not found, print the text without color
        print(text)


def seed_worker(worker_id):
    """
    Sets the random seed for a DataLoader worker.
    Ensures that data loading is reproducible across runs.
    """
    # Get the initial seed set in the main process and add the worker ID
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def acc(X: torch.Tensor, y: torch.Tensor, model: torch.nn.Module) -> float:
    in_train_mode = model.training
    device = next(model.parameters()).device

    model.eval()

    X, y = X.to(device), y.to(device)
    outputs = model(X)

    if outputs.dim() > 1 and outputs.size(1) > 1:
        # multi-class classification
        preds = outputs.argmax(dim=1)
    else:
        # binary classification
        preds = (outputs > 0.5).long().squeeze()

    correct = (preds == y).sum().item()
    total = y.shape[0]

    if in_train_mode:
        # set model back to model.training == True if it was True beforehand, to make sure outer loop can function with minimal disruption
        model.train()

    return correct / total
