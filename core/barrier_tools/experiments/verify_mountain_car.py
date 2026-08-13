"""Run barrier-certificate verification for ``MountainCarContinuous-v0``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml

from barrier_tools.barriers.polynomial import PolynomialBarrier
from barrier_tools.dynamics.mountain_car import MountainCarContinuousDynamics
from barrier_tools.policies.pytorch_policy import FeedForwardPolicy, load_policy
from barrier_tools.sets.boxes import Box
from barrier_tools.synthesis.cegis import CegisConfig, run_cegis
from barrier_tools.synthesis.learner import LearnerConfig
from barrier_tools.verification.branch_and_bound import (
    BarrierSpecification,
    BranchAndBoundVerifier,
    VerifierConfig,
)
from projects.safe_policy_optimisation.utils.io import write_json
from projects.safe_policy_optimisation.utils.seeding import set_global_seeds

DEFAULT_OUTPUT_DIR = Path("artifacts/barrier_policy_verification")


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("MountainCar barrier config must be a YAML mapping.")
    return payload


def _box_from_config(payload: list[list[float]]) -> Box:
    return Box.from_lists(payload)


def build_spec(config: dict[str, Any]) -> BarrierSpecification:
    safety = config.get("safety", {})
    verifier = config.get("verifier", {})
    p_safe = float(safety.get("p_safe", -1.0))
    initial_boxes = [_box_from_config(box) for box in safety.get("initial_boxes", [[[-0.6, -0.4], [0.0, 0.0]]])]
    unsafe_boxes = [
        _box_from_config(box)
        for box in safety.get("unsafe_boxes", [[[-1.2, p_safe - 1e-4], [-0.07, 0.07]]])
    ]
    invariant_boxes = [
        _box_from_config(box)
        for box in safety.get("invariant_boxes", [[[p_safe, 0.6], [-0.07, 0.07]]])
    ]
    return BarrierSpecification(
        initial_boxes=initial_boxes,
        unsafe_boxes=unsafe_boxes,
        invariant_boxes=invariant_boxes,
        safety_threshold=p_safe,
        alpha=float(verifier.get("alpha", 0.1)),
        eps_init=float(verifier.get("eps_init", 1e-5)),
        eps_unsafe=float(verifier.get("eps_unsafe", 1e-5)),
        eps_inv=float(verifier.get("eps_inv", 1e-6)),
    )


def build_verifier(config: dict[str, Any]) -> BranchAndBoundVerifier:
    verifier_cfg = config.get("verifier", {})
    return BranchAndBoundVerifier(
        MountainCarContinuousDynamics(),
        config=VerifierConfig(
            max_depth=int(verifier_cfg.get("max_depth", 18)),
            max_boxes=int(verifier_cfg.get("max_boxes", 50_000)),
            min_width=float(verifier_cfg.get("min_width", 1e-4)),
            timeout_seconds=verifier_cfg.get("timeout_seconds"),
            sample_counterexamples=bool(verifier_cfg.get("sample_counterexamples", True)),
        ),
    )


def build_learner_config(config: dict[str, Any]) -> LearnerConfig:
    learner = config.get("learner", {})
    return LearnerConfig(
        degree=int(learner.get("degree", 4)),
        use_energy_base=bool(learner.get("use_energy_base", True)),
        epochs=int(learner.get("epochs", 1_000)),
        batch_size=int(learner.get("batch_size", 512)),
        learning_rate=float(learner.get("learning_rate", 1e-2)),
        samples_initial=int(learner.get("samples_initial", 512)),
        samples_unsafe=int(learner.get("samples_unsafe", 512)),
        samples_domain=int(learner.get("samples_domain", 2_048)),
        boundary_noise=float(learner.get("boundary_noise", 0.02)),
        seed=int(config.get("seed", 0)),
        weight_initial=float(learner.get("weight_initial", 1.0)),
        weight_unsafe=float(learner.get("weight_unsafe", 1.0)),
        weight_invariance=float(learner.get("weight_invariance", 1.0)),
        weight_nonempty=float(learner.get("weight_nonempty", 0.1)),
    )


def empirical_rollouts(policy: object, *, episodes: int, seed: int, p_safe: float) -> dict[str, Any]:
    """Run empirical rollouts; this is not a proof."""

    env = gym.make("MountainCarContinuous-v0")
    violations = 0
    lengths: list[int] = []
    returns: list[float] = []
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed) + 10_000 + episode)
            done = False
            length = 0
            total_reward = 0.0
            while not done:
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = policy.forward(state).squeeze(0).detach().cpu().numpy()  # type: ignore[attr-defined]
                obs, reward, terminated, truncated, _info = env.step(action)
                violations += int(float(obs[0]) < float(p_safe))
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            lengths.append(length)
            returns.append(total_reward)
    finally:
        env.close()
    return {
        "kind": "empirical_rollout_not_proof",
        "episodes": int(episodes),
        "unsafe_state_visits": int(violations),
        "mean_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "mean_return": float(sum(returns) / len(returns)) if returns else 0.0,
    }


def save_plots(barrier: PolynomialBarrier, output_dir: Path, spec: BarrierSpecification) -> None:
    """Save a simple barrier level-set plot."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = torch.linspace(-1.2, 0.6, 180)
    v = torch.linspace(-0.07, 0.07, 120)
    pp, vv = torch.meshgrid(p, v, indexing="ij")
    states = torch.stack([pp.reshape(-1), vv.reshape(-1)], dim=-1)
    with torch.no_grad():
        h = barrier.value(states).reshape(pp.shape).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    contour = ax.contourf(pp.numpy(), vv.numpy(), h, levels=30)
    ax.contour(pp.numpy(), vv.numpy(), h, levels=[0.0], colors="white", linewidths=1.5)
    ax.axvline(spec.safety_threshold, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title("Barrier candidate h(p, v)")
    fig.colorbar(contour, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "barrier_level_set.png", dpi=160)
    plt.close(fig)


def run(config_path: Path, *, policy_override: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 0))
    set_global_seeds(seed)
    run_id = str(config.get("run_id", f"seed_{seed}"))
    output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR)) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_config = dict(config.get("policy", {"type": "generated", "seed": seed}))
    if policy_override is not None:
        policy_config["checkpoint"] = str(policy_override)
        policy_config.setdefault("type", "pytorch")
    policy = load_policy(policy_config, device=str(config.get("device", "cpu")))
    if isinstance(policy, FeedForwardPolicy):
        torch.save(policy.state_dict(), output_dir / "policy_state_dict.pt")

    dynamics = MountainCarContinuousDynamics()
    spec = build_spec(config)
    verifier = build_verifier(config)
    learner_config = build_learner_config(config)
    cegis_config = CegisConfig(iterations=int(config.get("cegis", {}).get("iterations", 3)))

    barrier, report = run_cegis(policy, dynamics, spec, verifier, learner_config, cegis_config)
    barrier_path = output_dir / "barrier.pt"
    barrier.save(barrier_path)  # type: ignore[attr-defined]
    report.save(output_dir / "report.json")
    empirical = empirical_rollouts(
        policy,
        episodes=int(config.get("rollouts", {}).get("episodes", 10)),
        seed=seed,
        p_safe=spec.safety_threshold,
    )
    write_json(output_dir / "empirical_rollouts.json", empirical)
    save_plots(barrier, output_dir, spec)  # type: ignore[arg-type]
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    summary = {
        "report": report.to_dict(),
        "empirical_rollouts": empirical,
        "artifacts": {
            "barrier": str(barrier_path),
            "report": str(output_dir / "report.json"),
            "plot": str(output_dir / "barrier_level_set.png"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mountain_car.yaml"))
    parser.add_argument("--policy", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args.config, policy_override=args.policy)
    print(summary["report"]["status"])


if __name__ == "__main__":
    main()
