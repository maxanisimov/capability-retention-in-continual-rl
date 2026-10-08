"""Train an interval-certified PSPO base policy from complete input boxes.

Despite the module name this stage is environment-agnostic whenever
``--certificate-dataset`` supplies the boxes: ``--obs-dim`` and ``--n-actions``
select the actor's shape (2/3 for MountainCar, 8/4 for LunarLander).  Only the
built-in fallback box below is MountainCar-specific.


The MountainCar shield restricts the safety-critical states with position in
``[-1.2, -1.0]`` and negative velocity. This stage conservatively represents
that open velocity condition by the closed input box
``[-1.2, -1.0] x [-0.07, 0.0]`` and differentiates through sound IBP output
bounds. An optional shield-only behavioural-cloning phase can precede the
complete-box certificate refinement.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation.adaptive_safe_ppo import (  # noqa: E402
    validate_interval_certificate_dataset,
)
from provably_safe_policy_optimisation.safe_init import (  # noqa: E402
    certify_with_verifier,
    refine_intervals_until_certified,
)

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402
from projects.safe_policy_optimisation.utils.seeding import (  # noqa: E402
    set_global_seeds,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "mountaincar_pspo_initialisation"
)
N_ACTIONS = 3
PUSH_RIGHT_ACTION = 2


def build_policy(
    *, hidden_dim: int, n_hidden: int, obs_dim: int = 2, n_actions: int = N_ACTIONS
) -> nn.Sequential:
    """Build the Sequential actor format consumed by the PSPO stage."""

    layers: list[nn.Module] = []
    input_dim = int(obs_dim)
    for _ in range(n_hidden):
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, int(n_actions)))
    return nn.Sequential(*layers)


def critical_interval_tensors(
    *,
    critical_min_position: float,
    critical_max_position: float,
    min_velocity: float,
    max_velocity: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the complete critical box and its push-right action mask."""

    if critical_min_position > critical_max_position:
        raise ValueError("critical_min_position must be <= critical_max_position.")
    if min_velocity > max_velocity:
        raise ValueError("min_velocity must be <= max_velocity.")
    x_l = torch.tensor(
        [[critical_min_position, min_velocity]], dtype=torch.float32, device=device
    )
    x_u = torch.tensor(
        [[critical_max_position, max_velocity]], dtype=torch.float32, device=device
    )
    safe_mask = torch.zeros((1, N_ACTIONS), dtype=torch.bool, device=device)
    safe_mask[0, PUSH_RIGHT_ACTION] = True
    return x_l, x_u, safe_mask


def behavioural_clone_on_shield(
    model: nn.Sequential,
    *,
    critical_min_position: float,
    critical_max_position: float,
    min_velocity: float,
    max_velocity: float,
    n_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Fit the set-valued shield on uniformly sampled MountainCar states."""

    if n_samples <= 0 or epochs <= 0:
        raise ValueError("BC sampling and epoch counts must both be positive.")
    if learning_rate <= 0:
        raise ValueError("BC learning rate must be positive.")
    rng = np.random.default_rng(seed)
    observations = torch.as_tensor(
        rng.uniform(
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            size=(n_samples, 2),
        ),
        dtype=torch.float32,
        device=device,
    )
    critical = (
        (observations[:, 0] >= float(critical_min_position))
        & (observations[:, 0] <= float(critical_max_position))
        & (observations[:, 1] >= float(min_velocity))
        & (observations[:, 1] < float(max_velocity))
    )
    safe_mask = torch.ones((n_samples, N_ACTIONS), dtype=torch.bool, device=device)
    safe_mask[critical] = False
    safe_mask[critical, PUSH_RIGHT_ACTION] = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        logits = model(observations)
        safe_logits = logits.masked_fill(~safe_mask, -1e9)
        loss = -(
            torch.logsumexp(safe_logits, dim=1) - torch.logsumexp(logits, dim=1)
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        greedy = model(observations).argmax(dim=1)
        sampled_safe = safe_mask[torch.arange(n_samples, device=device), greedy]
    return {
        "samples": int(n_samples),
        "critical_samples": int(critical.sum().item()),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "final_safe_mass_loss": final_loss,
        "sampled_greedy_safe_rate": float(sampled_safe.float().mean().item()),
    }


def behavioural_clone_on_interval_boxes(
    model: nn.Sequential,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    safe_mask: torch.Tensor,
    *,
    n_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Behaviourally clone a set-valued boxed shield before verification."""

    if n_samples < len(x_l):
        raise ValueError("--bc-samples must be at least the number of boxes.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    box_ids = torch.arange(n_samples, dtype=torch.long) % len(x_l)
    box_ids = box_ids[torch.randperm(n_samples, generator=generator)].to(device)
    random = torch.rand((n_samples, x_l.shape[1]), generator=generator).to(device)
    observations = x_l[box_ids] + random * (x_u[box_ids] - x_l[box_ids])
    sample_masks = safe_mask[box_ids].bool()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        logits = model(observations)
        safe_logits = logits.masked_fill(~sample_masks, -1e9)
        loss = -(
            torch.logsumexp(safe_logits, dim=1) - torch.logsumexp(logits, dim=1)
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        greedy = model(observations).argmax(dim=1)
        sampled_safe = sample_masks[
            torch.arange(n_samples, device=device), greedy
        ]
    return {
        "samples": int(n_samples),
        "boxes": int(len(x_l)),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "final_safe_mass_loss": final_loss,
        "sampled_greedy_safe_rate": float(sampled_safe.float().mean().item()),
    }


def _load_warm_start(
    model: nn.Sequential,
    path: Path,
    *,
    hidden_dim: int,
    n_hidden: int,
    obs_dim: int = 2,
    n_actions: int = N_ACTIONS,
) -> None:
    """Load a compatible PSPO ``base_policy.pt`` as an optional warm start."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Warm start must be a base_policy.pt payload with state_dict.")
    expected = {
        "input_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "activation": "Tanh",
    }
    architecture = dict(payload.get("architecture", {}))
    mismatch = {
        key: {"expected": value, "actual": architecture.get(key)}
        for key, value in expected.items()
        if architecture.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Warm-start architecture is incompatible: {mismatch}.")
    model.load_state_dict(dict(payload["state_dict"]), strict=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PSPO policy initialization by differentiating through the "
            "complete MountainCar safety-critical input interval."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=1_000)
    parser.add_argument("--target-margin", type=float, default=0.1)
    parser.add_argument(
        "--certification-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=2,
        help=(
            "Observation dimension of the certified boxes (default: 2 for "
            "MountainCar; 8 for LunarLander)."
        ),
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=N_ACTIONS,
        help="Discrete action count (default: 3 for MountainCar; 4 for LunarLander).",
    )
    parser.add_argument("--critical-min-position", type=float, default=-1.2)
    parser.add_argument("--critical-max-position", type=float, default=-1.0)
    parser.add_argument("--min-velocity", type=float, default=-0.07)
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.0,
        help=(
            "Upper closure of the safety-critical velocity set v < 0. The closed "
            "verification box includes v=0 conservatively."
        ),
    )
    parser.add_argument(
        "--bc-samples",
        type=int,
        default=0,
        help="Uniform shield-BC samples; zero disables the optional BC phase.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="Fixed shield-only BC epochs; enabled together with --bc-samples.",
    )
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=None,
        help="BC learning rate (defaults to --learning-rate).",
    )
    parser.add_argument("--warm-start-base-policy", type=Path, default=None)
    parser.add_argument(
        "--certificate-dataset",
        type=Path,
        default=None,
        help=(
            "Optional TensorDataset(X_l, X_u, safe_mask) containing multiple "
            "boxed shield regions. Defaults to the legacy single critical box."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive.")
    if args.n_hidden < 0:
        raise ValueError("--n-hidden must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.max_epochs < 0:
        raise ValueError("--max-epochs must be non-negative.")
    if args.target_margin <= 0:
        raise ValueError("--target-margin must be positive.")
    if args.bc_samples < 0 or args.bc_epochs < 0:
        raise ValueError("--bc-samples and --bc-epochs must be non-negative.")
    if bool(args.bc_samples) != bool(args.bc_epochs):
        raise ValueError("--bc-samples and --bc-epochs must be enabled together.")
    if args.bc_learning_rate is not None and args.bc_learning_rate <= 0:
        raise ValueError("--bc-learning-rate must be positive.")
    obs_dim = int(args.obs_dim)
    n_actions = int(args.n_actions)
    if obs_dim <= 0:
        raise ValueError("--obs-dim must be positive.")
    if n_actions <= 1:
        raise ValueError("--n-actions must be at least 2.")
    if args.certificate_dataset is None and (obs_dim, n_actions) != (2, N_ACTIONS):
        raise ValueError(
            "The built-in MountainCar critical box is 2-dimensional with "
            f"{N_ACTIONS} actions; supply --certificate-dataset for "
            f"obs_dim={obs_dim}, n_actions={n_actions}."
        )

    set_global_seeds(int(args.seed))
    device = torch.device(args.device)
    model = build_policy(
        hidden_dim=int(args.hidden_dim),
        n_hidden=int(args.n_hidden),
        obs_dim=obs_dim,
        n_actions=n_actions,
    ).to(device)
    if args.warm_start_base_policy is not None:
        _load_warm_start(
            model,
            Path(args.warm_start_base_policy),
            hidden_dim=int(args.hidden_dim),
            n_hidden=int(args.n_hidden),
            obs_dim=obs_dim,
            n_actions=n_actions,
        )

    if args.certificate_dataset is not None:
        loaded = torch.load(
            Path(args.certificate_dataset), map_location="cpu", weights_only=False
        )
        if not isinstance(loaded, TensorDataset):
            raise ValueError("--certificate-dataset must contain a TensorDataset.")
        loaded = validate_interval_certificate_dataset(
            loaded, observation_shape=(obs_dim,), n_actions=n_actions
        )
        x_l, x_u, safe_mask = (
            tensor.to(device) for tensor in loaded.tensors
        )
    else:
        x_l, x_u, safe_mask = critical_interval_tensors(
            critical_min_position=float(args.critical_min_position),
            critical_max_position=float(args.critical_max_position),
            min_velocity=float(args.min_velocity),
            max_velocity=float(args.max_velocity),
            device=device,
        )

    bc_report = None
    if args.bc_samples:
        if args.certificate_dataset is not None:
            bc_report = behavioural_clone_on_interval_boxes(
                model,
                x_l,
                x_u,
                safe_mask,
                n_samples=int(args.bc_samples),
                epochs=int(args.bc_epochs),
                learning_rate=float(args.bc_learning_rate or args.learning_rate),
                seed=int(args.seed),
                device=device,
            )
        else:
            bc_report = behavioural_clone_on_shield(
                model,
                critical_min_position=float(args.critical_min_position),
                critical_max_position=float(args.critical_max_position),
                min_velocity=float(args.min_velocity),
                max_velocity=float(args.max_velocity),
                n_samples=int(args.bc_samples),
                epochs=int(args.bc_epochs),
                learning_rate=float(args.bc_learning_rate or args.learning_rate),
                seed=int(args.seed),
                device=device,
            )
    log_info(
        "Training the certified initialization over "
        f"{len(x_l)} complete critical interval box(es)."
    )
    report = refine_intervals_until_certified(
        model,
        x_l,
        x_u,
        safe_mask,
        list(model.parameters()),
        lr=float(args.learning_rate),
        max_epochs=int(args.max_epochs),
        target_margin=float(args.target_margin),
        certification_method=str(args.certification_method),
    )

    # Fail closed: independently repeat the final authoritative check before
    # creating the reusable base-policy artifact.
    certified_fraction, all_certified = certify_with_verifier(
        model,
        x_l,
        x_u,
        safe_mask,
        method=str(args.certification_method),
    )
    if not all_certified:
        raise RuntimeError(
            "Final independent verification failed; base_policy.pt will not be saved."
        )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    architecture = {
        "input_dim": obs_dim,
        "n_actions": n_actions,
        "hidden_dim": int(args.hidden_dim),
        "n_hidden": int(args.n_hidden),
        "activation": "Tanh",
        "state_representation": "continuous_features",
    }
    interval_dataset = TensorDataset(
        x_l.detach().cpu(), x_u.detach().cpu(), safe_mask.detach().cpu().float()
    )
    interval_dataset_path = run_dir / "critical_interval_dataset.pt"
    base_policy_path = run_dir / "base_policy.pt"
    torch.save(interval_dataset, interval_dataset_path)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "architecture": architecture,
            "interval_initialisation": dataclasses.asdict(report),
            "behavioural_cloning": bc_report,
        },
        base_policy_path,
    )

    config = {
        "algorithm": "mountaincar_pspo_safe_initialisation",
        "seed": int(args.seed),
        "device": str(args.device),
        "architecture": architecture,
        "optimizer": "Adam",
        "learning_rate": float(args.learning_rate),
        "max_epochs": int(args.max_epochs),
        "target_margin": float(args.target_margin),
        "certification_method": str(args.certification_method),
        "uses_sampled_critical_states": bool(args.bc_samples),
        "behavioural_cloning": bc_report,
        "certificate_dataset_source": (
            str(Path(args.certificate_dataset).resolve())
            if args.certificate_dataset is not None
            else None
        ),
        "critical_interval_count": int(len(x_l)),
        "warm_start_base_policy": (
            str(Path(args.warm_start_base_policy).resolve())
            if args.warm_start_base_policy is not None
            else None
        ),
    }
    summary = {
        "run_dir": str(run_dir.resolve()),
        "base_policy_path": str(base_policy_path.resolve()),
        "critical_interval_dataset_path": str(interval_dataset_path.resolve()),
        "architecture": architecture,
        "behavioural_cloning": bc_report,
        "initialisation": dataclasses.asdict(report),
        "final_verification": {
            "certified_fraction": float(certified_fraction),
            "all_certified": bool(all_certified),
        },
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)
    log_info(
        "Certified base policy written to "
        f"{base_policy_path} after {report.epochs} optimizer epochs."
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
