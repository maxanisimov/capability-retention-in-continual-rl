from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ibp-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ibp-cache")

import numpy as np
import torch.nn as nn
from stable_baselines3 import DQN, PPO


ACTION_NAMES = {
    0: "push_left",
    1: "no_push",
    2: "push_right",
}


def affine_interval(
    weight: np.ndarray,
    bias: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight_pos = np.maximum(weight, 0.0)
    weight_neg = np.minimum(weight, 0.0)
    next_lower = weight_pos @ lower + weight_neg @ upper + bias
    next_upper = weight_pos @ upper + weight_neg @ lower + bias
    return next_lower, next_upper


def relu_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(lower, 0.0), np.maximum(upper, 0.0)


def tanh_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.tanh(lower), np.tanh(upper)


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def propagate_modules(
    modules: list[nn.Module],
    input_lower: np.ndarray,
    input_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    lower = input_lower.astype(np.float64)
    upper = input_upper.astype(np.float64)
    trace: list[dict[str, Any]] = [
        {
            "layer": "input",
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        }
    ]

    for index, module in enumerate(modules):
        if isinstance(module, nn.Linear):
            lower, upper = affine_interval(
                tensor_to_numpy(module.weight),
                tensor_to_numpy(module.bias),
                lower,
                upper,
            )
            layer_type = "linear"
        elif isinstance(module, nn.ReLU):
            lower, upper = relu_interval(lower, upper)
            layer_type = "relu"
        elif isinstance(module, nn.Tanh):
            lower, upper = tanh_interval(lower, upper)
            layer_type = "tanh"
        elif isinstance(module, nn.Flatten):
            layer_type = "flatten"
        else:
            raise TypeError(f"Unsupported module for IBP: {module!r}")

        trace.append(
            {
                "layer": f"{index}:{layer_type}",
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            }
        )

    return lower, upper, trace


def dqn_modules(model: DQN) -> list[nn.Module]:
    return list(model.policy.q_net.q_net)


def ppo_actor_modules(model: PPO) -> list[nn.Module]:
    return list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]


def certify_right_action(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_action: int,
) -> dict[str, Any]:
    competing_actions = [action for action in range(lower.size) if action != target_action]
    best_competing_upper = max(float(upper[action]) for action in competing_actions)
    target_lower = float(lower[target_action])
    margin = target_lower - best_competing_upper
    return {
        "target_action": target_action,
        "target_action_name": ACTION_NAMES[target_action],
        "certified": bool(margin > 0.0),
        "margin": margin,
        "target_lower": target_lower,
        "best_competing_upper": best_competing_upper,
        "output_lower": lower.tolist(),
        "output_upper": upper.tolist(),
    }


def sample_actions(
    model: Any,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = rng.uniform(input_lower, input_upper, size=(n, input_lower.size))
    actions = []
    for sample in samples:
        action, _ = model.predict(sample.astype(np.float32), deterministic=True)
        actions.append(int(np.asarray(action).item()))
    counts = {
        ACTION_NAMES[action]: int(np.count_nonzero(np.asarray(actions) == action))
        for action in ACTION_NAMES
    }
    return {
        "n": n,
        "seed": seed,
        "counts": counts,
        "all_right": bool(all(action == 2 for action in actions)),
    }


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    input_lower = np.asarray(args.input_lower, dtype=np.float64)
    input_upper = np.asarray(args.input_upper, dtype=np.float64)

    if np.any(input_lower > input_upper):
        raise ValueError("Every input lower bound must be <= its upper bound.")

    dqn = DQN.load(args.dqn_model)
    ppo = PPO.load(args.ppo_model)

    dqn_lower, dqn_upper, dqn_trace = propagate_modules(
        dqn_modules(dqn),
        input_lower,
        input_upper,
    )
    ppo_lower, ppo_upper, ppo_trace = propagate_modules(
        ppo_actor_modules(ppo),
        input_lower,
        input_upper,
    )

    result = {
        "input_lower": input_lower.tolist(),
        "input_upper": input_upper.tolist(),
        "target_property": "deterministic action is push_right for every state in the box",
        "action_mapping": ACTION_NAMES,
        "dqn": {
            "model_path": str(args.dqn_model),
            "ibp": certify_right_action(dqn_lower, dqn_upper, target_action=2),
            "sample_check": sample_actions(
                dqn,
                input_lower,
                input_upper,
                n=args.samples,
                seed=args.seed,
            ),
            "trace": dqn_trace,
        },
        "ppo": {
            "model_path": str(args.ppo_model),
            "ibp": certify_right_action(ppo_lower, ppo_upper, target_action=2),
            "sample_check": sample_actions(
                ppo,
                input_lower,
                input_upper,
                n=args.samples,
                seed=args.seed,
            ),
            "trace": ppo_trace,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use IBP to verify MountainCar policies over an input box."
    )
    parser.add_argument(
        "--input-lower",
        type=float,
        nargs=2,
        default=[-1.2, 0.0],
        metavar=("POSITION", "VELOCITY"),
    )
    parser.add_argument(
        "--input-upper",
        type=float,
        nargs=2,
        default=[-1.0, 0.07],
        metavar=("POSITION", "VELOCITY"),
    )
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=Path("artifacts/mountaincar/dqn/model.zip"),
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path("artifacts/mountaincar/ppo/model.zip"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/mountaincar/ibp_danger_zone_verification.json"),
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_verification(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    for name in ("dqn", "ppo"):
        ibp = result[name]["ibp"]
        sample_check = result[name]["sample_check"]
        status = "CERTIFIED" if ibp["certified"] else "NOT CERTIFIED"
        print(
            f"{name.upper()}: {status} | "
            f"margin={ibp['margin']:.6f} | "
            f"output_lower={np.asarray(ibp['output_lower'])} | "
            f"output_upper={np.asarray(ibp['output_upper'])} | "
            f"samples_all_right={sample_check['all_right']}"
        )

    print(f"Wrote {args.out}")
    all_certified = all(result[name]["ibp"]["certified"] for name in ("dqn", "ppo"))
    return 0 if all_certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
