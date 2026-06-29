"""Compute a Rashomon set from a precomputed tabular safety shield.

The saved shield provides a binary safe-action mask indexed by discrete state id.
This script converts that table into the same one-hot state representation used by
the local PPO-Lagrangian baseline, fits a base policy on the resulting safe-action
demonstration dataset, and computes an IBP Rashomon set around that base policy.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from projects.safe_policy_optimisation.utils.minipacman_safe_rl import write_json  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "shield_rashomon"
)


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_shield_mask(shield_path: Path, *, risk_threshold: float | None = None) -> np.ndarray:
    """Load a binary shield mask from ``shield_q.pt``."""

    payload = _torch_load(shield_path)
    if "shield" in payload:
        mask = _as_numpy(payload["shield"]) != 0
    elif "action_risk" in payload:
        threshold = payload.get("risk_threshold", 0.0) if risk_threshold is None else risk_threshold
        mask = _as_numpy(payload["action_risk"]) <= float(threshold)
    else:
        raise KeyError(
            f"Shield artifact must contain 'shield' or 'action_risk'; keys={sorted(payload.keys())}.",
        )
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D shield mask, got shape {mask.shape}.")
    return mask.astype(np.float32)


def make_safe_behaviour_payload(mask: np.ndarray) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create PPO-compatible one-hot state features and multi-hot safe actions."""

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected shield mask shape (n_states, n_actions), got {mask.shape}.")
    n_states, n_actions = mask.shape
    safe_counts = mask.sum(axis=1)
    safe_state_ids = np.flatnonzero(safe_counts > 0)
    if safe_state_ids.size == 0:
        raise ValueError("Shield contains no states with at least one safe action.")

    states = torch.nn.functional.one_hot(
        torch.as_tensor(safe_state_ids, dtype=torch.long),
        num_classes=int(n_states),
    ).to(torch.float32)
    actions = torch.as_tensor(mask[safe_state_ids], dtype=torch.float32)
    metadata = {
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "dataset_size": int(safe_state_ids.size),
        "excluded_no_safe_action_states": int(n_states - safe_state_ids.size),
        "state_representation": "one_hot_discrete_observation",
        "safe_state_ids": [int(x) for x in safe_state_ids.tolist()],
    }
    return {"state": states, "actions": actions}, metadata


def build_base_policy(
    input_dim: int,
    n_actions: int,
    *,
    hidden_dim: int,
    n_hidden: int,
) -> nn.Sequential:
    """Build a Sequential policy compatible with IntervalTrainer."""

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for _ in range(int(n_hidden)):
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(n_actions)))
    return nn.Sequential(*layers)


def initialise_linear_policy_from_masks(model: nn.Sequential, dataset: dict[str, torch.Tensor], *, margin: float) -> bool:
    """Closed-form BC initializer for one-hot features and a single Linear layer."""

    if len(model) != 1 or not isinstance(model[0], nn.Linear):
        return False
    states = dataset["state"]
    actions = dataset["actions"]
    state_ids = states.argmax(dim=1)
    with torch.no_grad():
        layer = model[0]
        layer.weight.zero_()
        layer.bias.zero_()
        actions = actions.to(layer.weight.device)
        for row_idx, state_id in enumerate(state_ids.tolist()):
            safe = actions[row_idx].bool()
            layer.weight[:, int(state_id)] = torch.where(
                safe,
                torch.full_like(layer.weight[:, int(state_id)], float(margin)),
                torch.full_like(layer.weight[:, int(state_id)], -float(margin)),
            )
    return True


@torch.no_grad()
def allowed_action_accuracy(model: nn.Module, dataset: dict[str, torch.Tensor], *, device: str | torch.device) -> float:
    """Fraction of rows where the greedy action is inside the multi-hot safe set."""

    device_t = torch.device(device)
    model.eval()
    states = dataset["state"].to(device_t)
    actions = dataset["actions"].to(device_t)
    logits = model(states)
    preds = logits.argmax(dim=1)
    correct = actions[torch.arange(actions.shape[0], device=device_t), preds] > 0
    return float(correct.float().mean().item())


def safe_action_bc_loss(logits: torch.Tensor, safe_actions: torch.Tensor) -> torch.Tensor:
    """Negative log probability assigned to the set of safe actions."""

    allowed = safe_actions.bool()
    masked_logits = logits.masked_fill(~allowed, -1e9)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(masked_logits, dim=1)).mean()


def fit_base_policy(
    model: nn.Sequential,
    dataset: dict[str, torch.Tensor],
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    direct_linear_init: bool = True,
    linear_init_margin: float = 10.0,
) -> dict[str, Any]:
    """Fit the base policy until greedy allowed-action accuracy reaches 100%."""

    torch.manual_seed(int(seed))
    device_t = torch.device(device)
    model.to(device_t)
    used_direct_init = False
    if direct_linear_init:
        used_direct_init = initialise_linear_policy_from_masks(
            model,
            dataset,
            margin=linear_init_margin,
        )

    initial_accuracy = allowed_action_accuracy(model, dataset, device=device_t)
    if initial_accuracy >= 1.0:
        return {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": initial_accuracy,
            "epochs_run": 0,
            "reached_target": True,
            "used_direct_linear_init": used_direct_init,
        }

    tensor_dataset = TensorDataset(dataset["state"], dataset["actions"])
    loader = DataLoader(
        tensor_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    final_accuracy = initial_accuracy
    epochs_run = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for states, safe_actions in loader:
            states = states.to(device_t)
            safe_actions = safe_actions.to(device_t)
            loss = safe_action_bc_loss(model(states), safe_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        final_accuracy = allowed_action_accuracy(model, dataset, device=device_t)
        if final_accuracy >= 1.0:
            break

    return {
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "epochs_run": int(epochs_run),
        "reached_target": bool(final_accuracy >= 1.0),
        "used_direct_linear_init": used_direct_init,
    }


def calibrate_inverse_temperature(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    *,
    inverse_temp_start: int,
    inverse_temp_max: int,
    device: str | torch.device,
) -> tuple[int, float, float]:
    """Find the first inverse temperature whose valid-action mass clears the threshold."""

    if inverse_temp_start > inverse_temp_max:
        raise ValueError("--inverse-temp-start must be <= --inverse-temp-max.")
    device_t = torch.device(device)
    states = dataset["state"].to(device_t)
    masks = dataset["actions"].to(device_t)
    max_valid = float(masks.sum(dim=1).max().item())
    if max_valid <= 0:
        raise ValueError("Dataset contains no valid actions.")
    threshold = max_valid / (1.0 + max_valid)
    model.eval()
    with torch.no_grad():
        logits = model(states)
        min_valid_mass = float("-inf")
        for inverse_temp in range(int(inverse_temp_start), int(inverse_temp_max) + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_mass = (probs * masks).sum(dim=1)
            min_valid_mass = float(valid_mass.min().item())
            if min_valid_mass >= threshold:
                return int(inverse_temp), float(min_valid_mass), float(threshold)
    raise ValueError(
        "Could not calibrate inverse temperature for Rashomon surrogate: "
        f"min_valid_mass={min_valid_mass:.6f}, threshold={threshold:.6f}.",
    )


def compute_rashomon_bounds(
    model: nn.Sequential,
    dataset: dict[str, torch.Tensor],
    *,
    seed: int,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    inverse_temp: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    """Run IntervalTrainer and select a 100%-certified Rashomon box."""

    from src.trainer.IntervalTrainer import IntervalTrainer

    tensor_dataset = TensorDataset(dataset["state"], dataset["actions"])
    interval_trainer = IntervalTrainer(
        model=model,
        accuracy=1.0,
        min_acc_increment=0,
        seed=int(seed),
        n_certificate_samples=int(certificate_samples),
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        batch_size=int(batch_size),
    )
    interval_trainer.compute_rashomon_set(
        dataset=tensor_dataset,
        temperatures={None: 1.0 / float(inverse_temp)},
    )
    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in interval_trainer.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        raise ValueError(f"No Rashomon certificate reached 1.0; certificates={cert_values}.")

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [param.detach().cpu() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().cpu() for param in bounded_model.param_u]
    metadata = {
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(value) for value in cert_values],
        "temperatures": {str(key): float(value) for key, value in interval_trainer.temperatures.items()},
    }
    return param_bounds_l, param_bounds_u, bounded_model, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a base policy from a saved shield and compute a Rashomon set.",
    )
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hidden", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-max-epochs", type=int, default=1000)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--linear-init-margin", type=float, default=10.0)
    parser.add_argument("--no-direct-linear-init", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=2000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument("--rashomon-batch-size", type=int, default=500)
    parser.add_argument("--certificate-samples", type=int, default=1000)
    parser.add_argument("--inverse-temp-start", type=int, default=1)
    parser.add_argument("--inverse-temp-max", type=int, default=1000)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mask = load_shield_mask(args.shield_path, risk_threshold=args.risk_threshold)
    dataset, dataset_metadata = make_safe_behaviour_payload(mask)
    n_states = int(dataset["state"].shape[1])
    n_actions = int(dataset["actions"].shape[1])

    model = build_base_policy(
        n_states,
        n_actions,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    )
    bc_metrics = fit_base_policy(
        model,
        dataset,
        lr=args.bc_lr,
        max_epochs=args.bc_max_epochs,
        batch_size=args.bc_batch_size,
        seed=args.seed,
        device=args.device,
        direct_linear_init=not args.no_direct_linear_init,
        linear_init_margin=args.linear_init_margin,
    )
    if not bc_metrics["reached_target"]:
        raise RuntimeError(
            "Base policy did not reach 100% allowed-action accuracy: "
            f"final_accuracy={bc_metrics['final_accuracy']:.6f}.",
        )

    inverse_temp, min_valid_mass, surrogate_threshold = calibrate_inverse_temperature(
        model,
        dataset,
        inverse_temp_start=args.inverse_temp_start,
        inverse_temp_max=args.inverse_temp_max,
        device=args.device,
    )
    param_bounds_l, param_bounds_u, bounded_model, rashomon_metadata = compute_rashomon_bounds(
        model,
        dataset,
        seed=args.seed,
        n_iters=args.rashomon_n_iters,
        checkpoint=args.rashomon_checkpoint,
        batch_size=args.rashomon_batch_size,
        certificate_samples=args.certificate_samples,
        inverse_temp=inverse_temp,
    )

    safe_dataset_path = run_dir / "safe_behaviour_dataset.pt"
    rashomon_dataset_path = run_dir / "rashomon_dataset.pt"
    base_policy_path = run_dir / "base_policy.pt"
    bounded_model_path = run_dir / "rashomon_bounded_model.pt"
    bounds_path = run_dir / "rashomon_param_bounds.pt"

    torch.save(dataset, safe_dataset_path)
    torch.save(dataset, rashomon_dataset_path)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": {
                "input_dim": n_states,
                "n_actions": n_actions,
                "hidden_dim": int(args.hidden_dim),
                "n_hidden": int(args.n_hidden),
                "activation": "Tanh",
            },
            "bc_metrics": bc_metrics,
        },
        base_policy_path,
    )
    torch.save(bounded_model, bounded_model_path)
    torch.save({"param_bounds_l": param_bounds_l, "param_bounds_u": param_bounds_u}, bounds_path)

    summary = {
        "shield_path": str(args.shield_path),
        "run_dir": str(run_dir),
        "safe_behaviour_dataset_path": str(safe_dataset_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "base_policy_path": str(base_policy_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "dataset": dataset_metadata,
        "base_policy": bc_metrics,
        "rashomon": {
            "inverse_temperature": int(inverse_temp),
            "min_valid_mass": float(min_valid_mass),
            "surrogate_threshold": float(surrogate_threshold),
            "n_iters": int(args.rashomon_n_iters),
            "checkpoint": int(args.rashomon_checkpoint),
            "batch_size": int(args.rashomon_batch_size),
            "certificate_samples": int(args.certificate_samples),
            **rashomon_metadata,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
