"""Train and certify DQN on MountainCar edge-safety regions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation import (  # noqa: E402
    ProvablySafeDQN,
    RegionShield,
)
from provably_safe_policy_optimisation.safe_init import (  # noqa: E402
    certify_with_verifier,
)

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.seeding import (  # noqa: E402
    EPISODE_SEED_OFFSET,
    TRAIN_SEED_OFFSET,
    set_global_seeds,
)

ALGORITHM_NAME = "mountaincar_edge_safe_dqn"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "mountaincar_edge_safety"
)

MOUNTAINCAR_LOW = np.array([-1.2, -0.07], dtype=np.float32)
MOUNTAINCAR_HIGH = np.array([0.6, 0.07], dtype=np.float32)
LEFT_ACTION = 0
RIGHT_ACTION = 2


def make_edge_shield(
    *,
    left_threshold: float = -1.05,
    right_threshold: float = 0.45,
    seed: int | None = None,
) -> RegionShield:
    """Return a box shield for MountainCar edge-critical regions."""

    return RegionShield.from_boxes(
        [
            (MOUNTAINCAR_LOW, [left_threshold, MOUNTAINCAR_HIGH[1]], [RIGHT_ACTION]),
            ([right_threshold, MOUNTAINCAR_LOW[1]], MOUNTAINCAR_HIGH, [LEFT_ACTION]),
        ],
        n_actions=3,
        seed=seed,
    )


def certify_edge_regions(model: ProvablySafeDQN, shield: RegionShield) -> dict[str, Any]:
    """Certify greedy Q-network safety over the shield's box regions."""

    x_l_t, x_u_t, safe_mask = edge_region_certificate_tensors(model, shield)
    certified_fraction, all_certified = certify_with_verifier(
        model.policy.q_net.q_net,
        x_l_t,
        x_u_t,
        safe_mask,
    )
    return {
        "certified_fraction": float(certified_fraction),
        "all_certified": bool(all_certified),
    }


def edge_region_certificate_tensors(
    model: ProvablySafeDQN,
    shield: RegionShield,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return edge-region input intervals and admissible action masks."""

    if not shield.boxes:
        raise ValueError("Edge certification requires a RegionShield built from boxes.")
    low = np.asarray(model.observation_space.low, dtype=np.float32)
    high = np.asarray(model.observation_space.high, dtype=np.float32)
    x_l = np.stack([np.clip(lo, low, high) for lo, _ in shield.boxes])
    x_u = np.stack([np.clip(hi, low, high) for _, hi in shield.boxes])
    x_l_t = torch.as_tensor(x_l, dtype=torch.float32, device=model.device)
    x_u_t = torch.as_tensor(x_u, dtype=torch.float32, device=model.device)
    safe_mask = torch.as_tensor(shield.mask[: len(shield.boxes)], dtype=torch.bool, device=model.device)
    return x_l_t, x_u_t, safe_mask


def make_safe_parameter_dataset(
    model: ProvablySafeDQN,
    shield: RegionShield,
) -> tuple[TensorDataset, dict[str, Any]]:
    """Build the interval certificate dataset for safe parameter-space search."""

    x_l, x_u, safe_mask = edge_region_certificate_tensors(model, shield)
    dataset = TensorDataset(x_l.detach().cpu(), x_u.detach().cpu(), safe_mask.detach().cpu().float())
    metadata = {
        "dataset_size": int(x_l.shape[0]),
        "input_dim": int(x_l.shape[1]),
        "n_actions": int(safe_mask.shape[1]),
        "has_input_intervals": True,
        "edge_region_lows": x_l.detach().cpu().tolist(),
        "edge_region_highs": x_u.detach().cpu().tolist(),
        "safe_action_masks": safe_mask.detach().cpu().int().tolist(),
    }
    return dataset, metadata


def _parameter_box_width(param_bounds_l: list[torch.Tensor], param_bounds_u: list[torch.Tensor]) -> float:
    width = torch.tensor(0.0)
    for lower, upper in zip(param_bounds_l, param_bounds_u):
        width = width + (upper - lower).sum().detach().cpu()
    return float(width.item())


def compute_safe_parameter_space(
    model: ProvablySafeDQN,
    shield: RegionShield,
    *,
    seed: int,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    growth_method: str,
    certification_method: str,
) -> tuple[TensorDataset, list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    """Compute a certified parameter box around the current safe Q-network."""

    from src.trainer.IntervalTrainer import IntervalTrainer

    dataset, dataset_metadata = make_safe_parameter_dataset(model, shield)
    interval_trainer = IntervalTrainer(
        model=model.policy.q_net.q_net,
        accuracy=1.0,
        min_acc_increment=0,
        seed=int(seed),
        n_certificate_samples=int(certificate_samples),
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        batch_size=int(batch_size),
    )
    interval_trainer.compute_rashomon_set(
        dataset=dataset,
        has_input_intervals=True,
        use_outer_bbox=False,
        growth_method=growth_method,
        certification_method=certification_method,
    )
    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in interval_trainer.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        raise ValueError(f"No safe parameter-space certificate reached 1.0; certificates={cert_values}.")

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [param.detach().cpu() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().cpu() for param in bounded_model.param_u]
    metadata = {
        "dataset": dataset_metadata,
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(value) for value in cert_values],
        "temperatures": {str(key): float(value) for key, value in interval_trainer.temperatures.items()},
        "parameter_box_width": _parameter_box_width(param_bounds_l, param_bounds_u),
        "growth_method": growth_method,
        "certification_method": certification_method,
        "n_iters": int(n_iters),
        "checkpoint": int(checkpoint),
        "batch_size": int(batch_size),
        "certificate_samples": int(certificate_samples),
    }
    return dataset, param_bounds_l, param_bounds_u, bounded_model, metadata


def evaluate_edge_safety(
    model: ProvablySafeDQN,
    shield: RegionShield,
    *,
    episodes: int,
    seed: int,
    shielded_execution: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate greedy policy safety on pre-step edge-critical observations."""

    env = gym.make("MountainCar-v0")
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    total_length = 0
    total_edge_visits = 0
    total_unsafe_greedy = 0
    total_unsafe_executed = 0
    try:
        for episode in range(episodes):
            obs, _info = env.reset(seed=seed + EPISODE_SEED_OFFSET + episode)
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_edge_visits = 0
            ep_unsafe_greedy = 0
            ep_unsafe_executed = 0
            while not done:
                state = int(shield.obs_to_state(np.asarray(obs, dtype=np.float32))[0])
                critical = state < len(shield.boxes or [])
                greedy_action = int(np.asarray(model.predict(obs, deterministic=True)[0]).item())
                executed_action = greedy_action
                unsafe_greedy = critical and not bool(shield.mask[state, greedy_action])
                if shielded_execution and unsafe_greedy:
                    executed_action = int(shield.safe_actions(state)[0])
                unsafe_executed = critical and not bool(shield.mask[state, executed_action])

                obs, reward, terminated, truncated, _info = env.step(executed_action)
                done = bool(terminated or truncated)
                ep_reward += float(reward)
                ep_len += 1
                ep_edge_visits += int(critical)
                ep_unsafe_greedy += int(unsafe_greedy)
                ep_unsafe_executed += int(unsafe_executed)

            total_reward += ep_reward
            total_length += ep_len
            total_edge_visits += ep_edge_visits
            total_unsafe_greedy += ep_unsafe_greedy
            total_unsafe_executed += ep_unsafe_executed
            rows.append(
                {
                    "algorithm": ALGORITHM_NAME,
                    "episode": episode,
                    "reward": ep_reward,
                    "length": ep_len,
                    "edge_region_visits": ep_edge_visits,
                    "unsafe_greedy_action_count": ep_unsafe_greedy,
                    "unsafe_executed_action_count": ep_unsafe_executed,
                    "shielded_execution": bool(shielded_execution),
                }
            )
    finally:
        env.close()

    summary = {
        "episodes": int(episodes),
        "mean_reward": total_reward / episodes if episodes else 0.0,
        "mean_length": total_length / episodes if episodes else 0.0,
        "edge_region_visits": int(total_edge_visits),
        "unsafe_greedy_action_count": int(total_unsafe_greedy),
        "unsafe_executed_action_count": int(total_unsafe_executed),
        "edge_greedy_safety_rate": (
            1.0 - total_unsafe_greedy / total_edge_visits if total_edge_visits else 1.0
        ),
        "edge_executed_safety_rate": (
            1.0 - total_unsafe_executed / total_edge_visits if total_edge_visits else 1.0
        ),
        "shielded_execution": bool(shielded_execution),
    }
    return summary, rows


def write_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "episode",
        "reward",
        "length",
        "edge_region_visits",
        "unsafe_greedy_action_count",
        "unsafe_executed_action_count",
        "shielded_execution",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DQN on MountainCar-v0 and certify edge-region action safety.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--left-threshold", type=float, default=-1.05)
    parser.add_argument("--right-threshold", type=float, default=0.45)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--net-arch", type=int, nargs="*", default=[16])
    parser.add_argument("--safe-init-samples", type=int, default=4096)
    parser.add_argument("--safe-init-lr", type=float, default=5e-2)
    parser.add_argument("--safe-init-bc-epochs", type=int, default=300)
    parser.add_argument("--safe-init-refine-epochs", type=int, default=500)
    parser.add_argument("--safe-init-target-margin", type=float, default=0.1)
    parser.add_argument(
        "--no-safe-parameter-space",
        action="store_true",
        help="Skip safe parameter-space computation and projected DQN training.",
    )
    parser.add_argument("--rashomon-n-iters", type=int, default=2_000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument("--rashomon-batch-size", type=int, default=2)
    parser.add_argument("--certificate-samples", type=int, default=2)
    parser.add_argument(
        "--growth-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help="Verification backend used to grow the safe parameter box.",
    )
    parser.add_argument(
        "--certification-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help="Verification backend used to certify saved parameter-box checkpoints.",
    )
    parser.add_argument(
        "--shielded-eval",
        action="store_true",
        help="Execute shield-overridden actions during final evaluation.",
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.total_timesteps < 0:
        raise ValueError("--total-timesteps must be non-negative.")
    if args.eval_episodes < 0:
        raise ValueError("--eval-episodes must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative.")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1].")
    if args.train_freq <= 0:
        raise ValueError("--train-freq must be positive.")
    if args.gradient_steps == 0 or args.gradient_steps < -1:
        raise ValueError("--gradient-steps must be positive, or -1 for SB3's train-freq match mode.")
    if args.target_update_interval <= 0:
        raise ValueError("--target-update-interval must be positive.")
    if not 0 <= args.exploration_fraction <= 1:
        raise ValueError("--exploration-fraction must be in [0, 1].")
    if not 0 <= args.exploration_initial_eps <= 1:
        raise ValueError("--exploration-initial-eps must be in [0, 1].")
    if not 0 <= args.exploration_final_eps <= 1:
        raise ValueError("--exploration-final-eps must be in [0, 1].")

    set_global_seeds(args.seed)
    run_id = args.run_id or f"seed_{args.seed}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    shield = make_edge_shield(
        left_threshold=float(args.left_threshold),
        right_threshold=float(args.right_threshold),
        seed=int(args.seed),
    )
    safe_parameter_space: dict[str, Any] | None = None
    env = gym.make("MountainCar-v0")
    try:
        env.reset(seed=int(args.seed) + TRAIN_SEED_OFFSET)
        model = ProvablySafeDQN(
            "MlpPolicy",
            env,
            shield=shield,
            seed=int(args.seed),
            shield_seed=int(args.seed),
            device=args.device,
            learning_rate=float(args.learning_rate),
            learning_starts=int(args.learning_starts),
            buffer_size=int(args.buffer_size),
            batch_size=int(args.batch_size),
            gamma=float(args.gamma),
            train_freq=int(args.train_freq),
            gradient_steps=int(args.gradient_steps),
            target_update_interval=int(args.target_update_interval),
            exploration_fraction=float(args.exploration_fraction),
            exploration_initial_eps=float(args.exploration_initial_eps),
            exploration_final_eps=float(args.exploration_final_eps),
            policy_kwargs={"net_arch": list(args.net_arch)},
            verbose=0,
        )

        pretrain_report = model.pretrain_on_shield(
            n_samples=int(args.safe_init_samples),
            lr=float(args.safe_init_lr),
            bc_max_epochs=int(args.safe_init_bc_epochs),
            refine_max_epochs=int(args.safe_init_refine_epochs),
            require_certified=True,
            target_margin=float(args.safe_init_target_margin),
            seed=int(args.seed),
        )

        pretrain_certificate = {
            "bc_epochs": int(pretrain_report.bc_epochs),
            "sampled_greedy_safe_rate": float(pretrain_report.sampled_greedy_safe_rate),
            "refine_epochs": int(pretrain_report.refine_epochs),
            "final_ibp_margin": (
                None if pretrain_report.final_ibp_margin is None else float(pretrain_report.final_ibp_margin)
            ),
            "certified_fraction": pretrain_report.certified_fraction,
            "all_certified": pretrain_report.all_certified,
        }

        if not args.no_safe_parameter_space:
            (
                safe_parameter_dataset,
                param_bounds_l,
                param_bounds_u,
                bounded_model,
                safe_parameter_metadata,
            ) = compute_safe_parameter_space(
                model,
                shield,
                seed=int(args.seed),
                n_iters=int(args.rashomon_n_iters),
                checkpoint=int(args.rashomon_checkpoint),
                batch_size=int(args.rashomon_batch_size),
                certificate_samples=int(args.certificate_samples),
                growth_method=args.growth_method,
                certification_method=args.certification_method,
            )
            safe_parameter_dataset_path = output_dir / "safe_parameter_dataset.pt"
            safe_parameter_bounds_path = output_dir / "safe_param_bounds.pt"
            safe_parameter_bounded_model_path = output_dir / "safe_parameter_bounded_model.pt"
            torch.save(safe_parameter_dataset, safe_parameter_dataset_path)
            torch.save(
                {"param_bounds_l": param_bounds_l, "param_bounds_u": param_bounds_u},
                safe_parameter_bounds_path,
            )
            torch.save(bounded_model, safe_parameter_bounded_model_path)
            model.set_projection_bounds(param_bounds_l, param_bounds_u, project_on_set=True)
            safe_parameter_space = {
                **safe_parameter_metadata,
                "attached_to_optimizer": True,
                "is_within_bounds_before_training": bool(model.is_within_bounds(atol=1e-6)),
                "max_violation_before_training": float(model.max_violation()),
                "artifacts": {
                    "dataset": str(safe_parameter_dataset_path),
                    "bounds": str(safe_parameter_bounds_path),
                    "bounded_model": str(safe_parameter_bounded_model_path),
                },
            }

        if args.total_timesteps > 0:
            model.learn(total_timesteps=int(args.total_timesteps), progress_bar=False)

        final_certificate = certify_edge_regions(model, shield)
        evaluation, episode_rows = evaluate_edge_safety(
            model,
            shield,
            episodes=int(args.eval_episodes),
            seed=int(args.seed),
            shielded_execution=bool(args.shielded_eval),
        )
        model.save(output_dir / "model.zip")
    finally:
        env.close()

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": "MountainCar-v0",
        "seed": int(args.seed),
        "total_timesteps": int(args.total_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "edge_regions": {
            "left": {
                "position": [-1.2, float(args.left_threshold)],
                "safe_actions": [RIGHT_ACTION],
            },
            "right": {
                "position": [float(args.right_threshold), 0.6],
                "safe_actions": [LEFT_ACTION],
            },
            "velocity": [-0.07, 0.07],
        },
        "dqn": {
            "learning_rate": float(args.learning_rate),
            "learning_starts": int(args.learning_starts),
            "buffer_size": int(args.buffer_size),
            "batch_size": int(args.batch_size),
            "gamma": float(args.gamma),
            "train_freq": int(args.train_freq),
            "gradient_steps": int(args.gradient_steps),
            "target_update_interval": int(args.target_update_interval),
            "exploration_fraction": float(args.exploration_fraction),
            "exploration_initial_eps": float(args.exploration_initial_eps),
            "exploration_final_eps": float(args.exploration_final_eps),
            "net_arch": list(args.net_arch),
        },
        "safe_init": {
            "n_samples": int(args.safe_init_samples),
            "lr": float(args.safe_init_lr),
            "bc_max_epochs": int(args.safe_init_bc_epochs),
            "refine_max_epochs": int(args.safe_init_refine_epochs),
            "target_margin": float(args.safe_init_target_margin),
        },
        "safe_parameter_space": {
            "enabled": not bool(args.no_safe_parameter_space),
            "n_iters": int(args.rashomon_n_iters),
            "checkpoint": int(args.rashomon_checkpoint),
            "batch_size": int(args.rashomon_batch_size),
            "certificate_samples": int(args.certificate_samples),
            "growth_method": args.growth_method,
            "certification_method": args.certification_method,
        },
        "shielded_eval": bool(args.shielded_eval),
    }
    summary = {
        "algorithm": ALGORITHM_NAME,
        "run_id": run_id,
        "config": config,
        "pretrain_certificate": pretrain_certificate,
        "safe_parameter_space": safe_parameter_space,
        "final_certificate": final_certificate,
        "evaluation": evaluation,
        "shield_diagnostics": model.shield_diagnostics(),
        "projection": {
            "active": bool(model.policy.optimizer.has_bounds),
            "is_within_bounds": bool(model.is_within_bounds(atol=1e-6)),
            "max_violation": float(model.max_violation()),
            "diagnostics": model.projection_diagnostics(),
        },
        "artifacts": {
            "config": str(output_dir / "config.json"),
            "summary": str(output_dir / "summary.json"),
            "episodes": str(output_dir / "episodes.csv"),
            "model": str(output_dir / "model.zip"),
        },
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)
    write_episode_rows(output_dir / "episodes.csv", episode_rows)
    return summary


def main() -> None:
    summary = run_experiment(build_parser().parse_args())
    print(f"Wrote MountainCar edge-safety run to {summary['artifacts']['summary']}")


if __name__ == "__main__":
    main()
