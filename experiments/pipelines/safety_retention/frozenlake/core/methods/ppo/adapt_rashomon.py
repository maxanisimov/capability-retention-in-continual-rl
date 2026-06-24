"""Rashomon-certificate-constrained downstream PPO adaptation."""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import torch

from experiments.pipelines.safety_retention.frozenlake.core.config import get_pipeline_config
from experiments.pipelines.safety_retention.frozenlake.core.paths import mode_run_dir
from experiments.pipelines.safety_retention.frozenlake.core.safety import to_tensor_dataset, validate_rashomon_payload
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    add_adaptation_args,
    downstream_ppo_config,
    finalize_downstream_run,
    load_source_for_adaptation,
    make_downstream_env,
    set_seeds,
    validate_deterministic,
    validate_rl,
)
from experiments.utils.ppo_utils import ppo_train


MODE = "downstream_rashomon"


def _compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_payload: dict[str, torch.Tensor],
    seed: int,
    n_iters: int,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    from src.trainer.IntervalTrainer import IntervalTrainer

    validate_rashomon_payload(rashomon_payload)
    rashomon_dataset = to_tensor_dataset(rashomon_payload)
    valid_counts = rashomon_payload["actions"].sum(dim=1)
    max_valid = float(valid_counts.max().item())
    if max_valid <= 0:
        raise RuntimeError("Rashomon dataset contains no valid actions.")
    surrogate_threshold = max_valid / (1.0 + max_valid)

    actor.eval()
    with torch.no_grad():
        states = rashomon_payload["state"]
        masks = rashomon_payload["actions"]
        logits = actor(states)
        selected_inverse_temp: int | None = None
        min_valid_mass = float("-inf")
        for inverse_temp in range(inverse_temp_start, inverse_temp_max + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_mass = (probs * masks).sum(dim=1)
            min_valid_mass = float(valid_mass.min().item())
            if min_valid_mass >= surrogate_threshold:
                selected_inverse_temp = inverse_temp
                break
    if selected_inverse_temp is None:
        raise ValueError(
            "Could not find inverse temperature satisfying safety Rashomon threshold: "
            f"min_valid_mass={min_valid_mass:.6f}, threshold={surrogate_threshold:.6f}.",
        )

    interval_trainer = IntervalTrainer(
        model=actor,
        accuracy=1.0,
        min_acc_increment=0,
        seed=seed,
        n_iters=n_iters,
        checkpoint=checkpoint,
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        temperatures={None: 1.0 / selected_inverse_temp},
    )
    cert_values = [
        min((c.min_hard_acc for c in certs), default=float("-inf"))
        for certs in interval_trainer.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        raise ValueError(f"No Rashomon certificate reached 1.0; certificates={cert_values}")
    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [param.detach().cpu() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().cpu() for param in bounded_model.param_u]
    metadata = {
        "surrogate_threshold": float(surrogate_threshold),
        "surrogate_aggregation": "min",
        "rashomon_min_hard_spec": 1.0,
        "inverse_temperature": int(selected_inverse_temp),
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(v) for v in cert_values],
    }
    return param_bounds_l, param_bounds_u, bounded_model, metadata


def adapt_rashomon(args: argparse.Namespace) -> Path:
    validate_rl(args.rl)
    validate_deterministic(args.deterministic)
    cfg = get_pipeline_config(args.layout)
    set_seeds(args.seed)
    source_actor, source_critic, source_dir = load_source_for_adaptation(cfg, args)
    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else cfg.downstream_total_timesteps
    )

    rashomon_dataset_path = source_dir / "rashomon_dataset.pt"
    if not rashomon_dataset_path.exists():
        raise FileNotFoundError(f"Source Rashomon dataset not found: {rashomon_dataset_path}")
    rashomon_payload = torch.load(rashomon_dataset_path, map_location="cpu", weights_only=False)
    actor_bounds_l, actor_bounds_u, bounded_model, rashomon_metadata = _compute_rashomon_bounds(
        actor=copy.deepcopy(source_actor),
        rashomon_payload=rashomon_payload,
        seed=args.seed,
        n_iters=int(args.rashomon_n_iters or cfg.rashomon_n_iters),
        inverse_temp_start=int(args.inverse_temp_start or cfg.inverse_temp_start),
        inverse_temp_max=int(args.inverse_temp_max or cfg.inverse_temp_max),
        checkpoint=int(args.rashomon_checkpoint or cfg.rashomon_checkpoint),
    )

    train_env = make_downstream_env(cfg, shaped=True)
    early_stop_env = make_downstream_env(cfg, shaped=False)
    ppo_cfg = downstream_ppo_config(cfg, seed=args.seed, device=args.device, total_timesteps=total_timesteps)
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=source_critic,
        actor_param_bounds_l=actor_bounds_l,
        actor_param_bounds_u=actor_bounds_u,
        early_stop_eval_env=early_stop_env,
        return_training_data=True,
    )
    train_env.close()
    early_stop_env.close()

    run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, MODE)
    run_dir.mkdir(parents=True, exist_ok=True)
    rashomon_dataset_out_path = run_dir / "rashomon_dataset.pt"
    bounded_model_path = run_dir / "rashomon_bounded_model.pt"
    bounds_path = run_dir / "rashomon_param_bounds.pt"
    torch.save(rashomon_payload, rashomon_dataset_out_path)
    torch.save(bounded_model, bounded_model_path)
    torch.save({"param_bounds_l": actor_bounds_l, "param_bounds_u": actor_bounds_u}, bounds_path)

    return finalize_downstream_run(
        cfg,
        args,
        mode=MODE,
        actor=actor,
        critic=critic,
        training_data=training_data,
        source_dir=source_dir,
        ppo_cfg=ppo_cfg,
        total_timesteps=total_timesteps,
        extra_artifacts={
            "rashomon_dataset_path": str(rashomon_dataset_out_path),
            "rashomon_bounded_model_path": str(bounded_model_path),
            "rashomon_param_bounds_path": str(bounds_path),
        },
        extra_settings={
            "rashomon_settings_source": cfg.rashomon_settings_source,
            "rashomon_n_iters": int(args.rashomon_n_iters or cfg.rashomon_n_iters),
            "inverse_temp_start": int(args.inverse_temp_start or cfg.inverse_temp_start),
            "inverse_temp_max": int(args.inverse_temp_max or cfg.inverse_temp_max),
            "rashomon_checkpoint": int(args.rashomon_checkpoint or cfg.rashomon_checkpoint),
            "surrogate_aggregation": cfg.rashomon_surrogate_aggregation,
            "rashomon_min_hard_spec": float(cfg.rashomon_min_hard_spec),
            "rashomon_dataset_size": int(rashomon_payload["state"].shape[0]),
            **rashomon_metadata,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rashomon-constrained downstream PPO adaptation.")
    add_adaptation_args(parser)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        validate_rl(args.rl)
        validate_deterministic(args.deterministic)
        cfg = get_pipeline_config(args.layout)
        run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, MODE)
        print(f"Dry run: mode={MODE} layout={cfg.layout} seed={args.seed} run_dir={run_dir}")
        return 0
    adapt_rashomon(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
