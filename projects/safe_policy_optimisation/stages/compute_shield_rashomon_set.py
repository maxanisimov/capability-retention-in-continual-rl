"""Compute a Rashomon set from a precomputed tabular safety shield.

The saved shield provides a binary safe-action mask indexed by discrete state id.
This script converts that table into the same one-hot state representation used by
the local PPO-Lagrangian baseline, fits a base policy on the resulting safe-action
demonstration dataset, and computes an IBP Rashomon set around that base policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.shield import (  # noqa: E402
    load_shield_mask as _load_shield_mask,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "shield_rashomon"
)


def load_shield_mask(shield_path: Path, *, risk_threshold: float | None = None) -> np.ndarray:
    """Load a float32 ``(state, action)`` shield mask, auto-detecting the source.

    Rashomon-set construction needs the mask as float32 one-hot features, so this
    delegates to :func:`...utils.shield.load_shield_mask` with ``source="auto"``.
    """

    return _load_shield_mask(
        shield_path, source="auto", risk_threshold=risk_threshold, dtype=np.float32
    )


def make_safe_behaviour_payload(
    mask: np.ndarray,
    state_to_features: Any = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """State features and multi-hot safe actions for the BC dataset.

    ``state_to_features`` (the env's ``state_to_features``) selects the state
    representation the base policy -- and hence the PSPO actor -- is fitted on:

    * ``None``: one-hot over discrete state ids (the historical view).
    * a callable: the env's normalised decoded features, so the base policy has
      the same feature input the deployed PSPO actor receives.
    """

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected shield mask shape (n_states, n_actions), got {mask.shape}.")
    n_states, n_actions = mask.shape
    safe_counts = mask.sum(axis=1)
    safe_state_ids = np.flatnonzero(safe_counts > 0)
    if safe_state_ids.size == 0:
        raise ValueError("Shield contains no states with at least one safe action.")

    if state_to_features is None:
        states = torch.nn.functional.one_hot(
            torch.as_tensor(safe_state_ids, dtype=torch.long),
            num_classes=int(n_states),
        ).to(torch.float32)
        representation = "one_hot_discrete_observation"
    else:
        feature_rows = np.stack([np.asarray(state_to_features(int(s))) for s in safe_state_ids])
        states = torch.as_tensor(feature_rows, dtype=torch.float32)
        representation = "decoded_features"
    actions = torch.as_tensor(mask[safe_state_ids], dtype=torch.float32)
    metadata = {
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "feature_dim": int(states.shape[1]),
        "dataset_size": int(safe_state_ids.size),
        "excluded_no_safe_action_states": int(n_states - safe_state_ids.size),
        "state_representation": representation,
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


def safe_action_margins(
    logits: torch.Tensor, safe_actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state ``best safe logit - best unsafe logit``.

    Positive means the greedy action is shield-safe; the magnitude is how much
    the parameters can move before it stops being so. Returns the margins plus a
    mask of the rows that actually have an unsafe action to compete with (rows
    where every action is safe are unconstrained and are excluded).
    """

    allowed = safe_actions.bool()
    neg = torch.finfo(logits.dtype).min
    best_safe = logits.masked_fill(~allowed, neg).max(dim=1).values
    best_unsafe = logits.masked_fill(allowed, neg).max(dim=1).values
    contested = (~allowed).any(dim=1)
    return best_safe - best_unsafe, contested


def safe_action_margin_loss(
    logits: torch.Tensor, safe_actions: torch.Tensor, *, target_margin: float
) -> torch.Tensor:
    """Hinge pushing every contested state to at least ``target_margin``."""

    margins, contested = safe_action_margins(logits, safe_actions)
    if not bool(contested.any()):
        return logits.new_zeros(())
    return torch.relu(float(target_margin) - margins[contested]).mean()


@torch.no_grad()
def minimum_safe_action_margin(
    model: nn.Module, dataset: dict[str, torch.Tensor], *, device: str | torch.device
) -> float:
    """Worst-case per-state safety margin over the whole dataset."""

    device_t = torch.device(device)
    model.eval()
    logits = model(dataset["state"].to(device_t))
    margins, contested = safe_action_margins(logits, dataset["actions"].to(device_t))
    if not bool(contested.any()):
        return float("inf")
    return float(margins[contested].min().item())


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
    target_margin: float = 10.0,
    margin_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Fit the base policy until every state has a safety margin of ``target_margin``.

    Stopping at bare 100% allowed-action accuracy is not enough. It yields a
    knife-edge policy whose greedy action is safe by an arbitrarily small logit
    gap, so the first gradient step of downstream training flips actions and
    every candidate update fails verification. The closed-form linear
    initialiser has never had this problem because it writes +/-``margin``
    directly; this makes the gradient path (the only option once the base policy
    has hidden layers) aim for the same separation.
    """

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
    initial_margin = minimum_safe_action_margin(model, dataset, device=device_t)
    if initial_accuracy >= 1.0 and initial_margin >= float(target_margin):
        return {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": initial_accuracy,
            "epochs_run": 0,
            "reached_target": True,
            "used_direct_linear_init": used_direct_init,
            "initial_min_margin": initial_margin,
            "final_min_margin": initial_margin,
            "target_margin": float(target_margin),
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
            logits = model(states)
            loss = safe_action_bc_loss(logits, safe_actions)
            if float(margin_loss_weight) > 0.0:
                loss = loss + float(margin_loss_weight) * safe_action_margin_loss(
                    logits, safe_actions, target_margin=target_margin
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        final_accuracy = allowed_action_accuracy(model, dataset, device=device_t)
        final_margin = minimum_safe_action_margin(model, dataset, device=device_t)
        if final_accuracy >= 1.0 and final_margin >= float(target_margin):
            break

    return {
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "epochs_run": int(epochs_run),
        # Accuracy alone is not the bar any more: a policy that is safe by a
        # vanishing margin is unusable downstream.
        "reached_target": bool(final_accuracy >= 1.0 and final_margin >= float(target_margin)),
        "used_direct_linear_init": used_direct_init,
        "initial_min_margin": initial_margin,
        "final_min_margin": final_margin,
        "target_margin": float(target_margin),
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
    growth_method: str = "IBP",
    growth_method_kwargs: dict[str, Any] | None = None,
    certification_method: str = "IBP",
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    """Run IntervalTrainer and select a 100%-certified Rashomon box.

    ``growth_method`` selects the verification backend that actually drives box
    growth (both the differentiable soft surrogate and the hard per-iteration
    accuracy check the Lagrangian penalises against). Defaults to ``"IBP"``,
    matching the historical behaviour: cheap, but conservative fast with
    depth, so the box can never grow past what IBP's own bound is willing to
    certify along the way. A tighter method (e.g. ``"CROWN"``) lets the
    optimizer keep growing into regions IBP would have prematurely penalised,
    at the cost of a more expensive bound per iteration.

    ``certification_method`` selects the verification backend used to compute
    the *reported* checkpoint certificates (see ``src.verification.registry``
    for the registered options, e.g. ``"IBP"``, ``"CROWN"``, ``"alpha-CROWN"``).
    This is independent of ``growth_method`` and only changes which checkpoints
    get confirmed as fully certified (``min_hard_acc >= 1.0``) when their
    ``(param_l, param_u)`` are re-verified after the fact -- it cannot recover
    a bigger box than whatever ``growth_method``'s trajectory actually explored.
    """

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
        growth_method=growth_method,
        growth_method_kwargs=growth_method_kwargs,
        certification_method=certification_method,
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
    # When given, the BC base policy (and hence the PSPO actor) is fitted on the
    # env's decoded feature representation instead of a one-hot state id, so it
    # matches the observation the deployed PSPO actor receives.
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", default=None)
    parser.add_argument(
        "--state-representation",
        choices=("features", "one_hot"),
        default="features",
        help="BC input representation. 'features' requires --env-id.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=2,
        help=(
            "Hidden layers in the BC base policy, and hence in the PSPO actor and "
            "critic (SB3 applies a flat net_arch to both). Default 2 matches the "
            "[64, 64] MLP every baseline uses, so PSPO is compared like-for-like. "
            "0 gives the older linear/tabular actor."
        ),
    )
    # Feature-based BC needs more optimisation than one-hot to reach the safety
    # margin: nearby feature points can require different safe actions, so the
    # decision boundary is finer. These defaults suit both representations
    # (one-hot converges well within them and stops early).
    parser.add_argument("--bc-lr", type=float, default=3e-3)
    parser.add_argument("--bc-max-epochs", type=int, default=8000)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--linear-init-margin", type=float, default=10.0)
    parser.add_argument(
        "--bc-target-margin",
        type=float,
        default=10.0,
        help=(
            "Minimum required gap between the best safe and best unsafe logit in "
            "every state. Matches --linear-init-margin so the gradient path (used "
            "whenever the base policy has hidden layers) reaches the same "
            "separation the closed-form linear initialiser writes directly. "
            "Stopping at bare 100%% accuracy leaves a knife-edge policy whose "
            "greedy actions flip on the first gradient step downstream."
        ),
    )
    parser.add_argument(
        "--bc-margin-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the margin hinge added to the BC loss. 0 disables it.",
    )
    parser.add_argument("--no-direct-linear-init", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=2000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument("--rashomon-batch-size", type=int, default=500)
    parser.add_argument("--certificate-samples", type=int, default=1000)
    parser.add_argument("--inverse-temp-start", type=int, default=1)
    parser.add_argument("--inverse-temp-max", type=int, default=1000)
    parser.add_argument(
        "--growth-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verification backend that actually drives box growth (the "
            "differentiable soft surrogate and the hard per-iteration "
            "accuracy check the Lagrangian penalises against). Defaults to "
            "IBP: cheap, but conservative fast with depth, so the box can "
            "never grow past what IBP's own bound is willing to certify "
            "along the way. A tighter method (e.g. CROWN) lets the "
            "optimizer keep growing into regions IBP would have "
            "prematurely penalised, at the cost of a slower bound per "
            "iteration. Independent of --certification-method."
        ),
    )
    parser.add_argument(
        "--certification-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verification backend used for the reported Rashomon checkpoint "
            "certificates (see src.verification.registry), independent of "
            "--growth-method. Only affects which checkpoints get confirmed "
            "as fully certified; it cannot recover a bigger box than "
            "--growth-method's trajectory actually explored."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mask = load_shield_mask(args.shield_path, risk_threshold=args.risk_threshold)
    state_to_features = None
    if args.state_representation == "features":
        if not args.env_id:
            raise ValueError("--state-representation features requires --env-id.")
        from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs
        from projects.safe_policy_optimisation.utils.safe_crl_bridge import make_custom_masa_env

        feature_env = make_custom_masa_env(
            args.env_id,
            max_episode_steps=None,
            env_kwargs=parse_env_kwargs(args.env_kwargs),
        ).unwrapped
        state_to_features = feature_env.state_to_features
    dataset, dataset_metadata = make_safe_behaviour_payload(mask, state_to_features)
    input_dim = int(dataset["state"].shape[1])
    n_actions = int(dataset["actions"].shape[1])

    model = build_base_policy(
        input_dim,
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
        target_margin=args.bc_target_margin,
        margin_loss_weight=args.bc_margin_loss_weight,
    )
    if not bc_metrics["reached_target"]:
        raise RuntimeError(
            "Base policy did not reach 100% allowed-action accuracy at the required "
            f"safety margin: final_accuracy={bc_metrics['final_accuracy']:.6f}, "
            f"final_min_margin={bc_metrics['final_min_margin']:.4f} "
            f"(target {bc_metrics['target_margin']:.4f}). Raise --bc-max-epochs or "
            "--bc-lr, or lower --bc-target-margin.",
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
        growth_method=args.growth_method,
        certification_method=args.certification_method,
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
                "input_dim": input_dim,
                "n_actions": n_actions,
                "hidden_dim": int(args.hidden_dim),
                "n_hidden": int(args.n_hidden),
                "activation": "Tanh",
                "state_representation": dataset_metadata["state_representation"],
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
            "growth_method": args.growth_method,
            "certification_method": args.certification_method,
            **rashomon_metadata,
        },
    }
    write_json(run_dir / "summary.json", summary)
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
