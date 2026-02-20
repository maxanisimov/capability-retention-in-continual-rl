from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from poisoned_apple_env import PoisonedAppleEnv, evaluate_policy
from rl_project.utils.ppo_utils import PPOConfig, ppo_train


@dataclass
class EnvSetup:
    name: str
    grid_size: int
    max_steps: int
    agent_start_pos: tuple[int, int]
    task1_safe_apples: list[tuple[int, int]]
    task1_poisoned_apples: list[tuple[int, int]]
    task2_safe_apples: list[tuple[int, int]]
    task2_poisoned_apples: list[tuple[int, int]]


DEFAULT_SETUPS = [
    EnvSetup(
        name="swap_4x4_1s1p",
        grid_size=4,
        max_steps=8,
        agent_start_pos=(0, 0),
        task1_safe_apples=[(1, 1)],
        task1_poisoned_apples=[(2, 2)],
        task2_safe_apples=[(2, 2)],
        task2_poisoned_apples=[(1, 1)],
    ),
    EnvSetup(
        name="swap_5x5_1s1p",
        grid_size=5,
        max_steps=10,
        agent_start_pos=(0, 0),
        task1_safe_apples=[(1, 1)],
        task1_poisoned_apples=[(3, 3)],
        task2_safe_apples=[(3, 3)],
        task2_poisoned_apples=[(1, 1)],
    ),
    EnvSetup(
        name="swap_6x6_1s1p",
        grid_size=6,
        max_steps=12,
        agent_start_pos=(0, 0),
        task1_safe_apples=[(1, 1)],
        task1_poisoned_apples=[(4, 4)],
        task2_safe_apples=[(4, 4)],
        task2_poisoned_apples=[(1, 1)],
    ),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an unsafe adapted actor and produce paper-ready forgetting visualizations."
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="swap_4x4_1s1p",
        choices=[s.name for s in DEFAULT_SETUPS],
        help="Environment setup to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task1-steps", type=int, default=1_000)
    parser.add_argument("--task2-steps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display figures in addition to saving.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "paper_figures"),
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_setup_by_name(name: str) -> EnvSetup:
    for setup in DEFAULT_SETUPS:
        if setup.name == name:
            return setup
    raise ValueError(f"Unknown setup: {name}")


def make_env(setup: EnvSetup, task: int, seed: int) -> PoisonedAppleEnv:
    safe = setup.task1_safe_apples if task == 1 else setup.task2_safe_apples
    poison = setup.task1_poisoned_apples if task == 1 else setup.task2_poisoned_apples
    return PoisonedAppleEnv(
        grid_size=setup.grid_size,
        agent_start_pos=setup.agent_start_pos,
        safe_apple_positions=safe,
        poisoned_apple_positions=poison,
        observation_type="flat",
        max_steps=setup.max_steps,
        seed=seed,
    )


def actor_action(actor: torch.nn.Module, obs: np.ndarray) -> int:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(actor(obs_tensor), dim=1).item())


def collect_trajectory(env: PoisonedAppleEnv, actor: torch.nn.Module) -> dict:
    obs, _ = env.reset()
    path = [tuple(int(x) for x in env.agent_pos)]
    actions = []
    rewards = []
    poisoned_hit_steps = []
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        action = actor_action(actor, obs)
        poisoned_before = len(env.poisoned_apples)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        total_reward += float(reward)

        path.append(tuple(int(x) for x in env.agent_pos))
        actions.append(action)
        rewards.append(float(reward))
        hit_poisoned = len(env.poisoned_apples) < poisoned_before
        if hit_poisoned:
            poisoned_hit_steps.append(step)
            # Stop the trace immediately when the unsafe state is reached.
            break

    return {
        "path": path,
        "actions": actions,
        "rewards": rewards,
        "poisoned_hit_steps": poisoned_hit_steps,
        "total_reward": total_reward,
    }


def get_task_apple_sets(setup: EnvSetup, task: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if task == 1:
        safe = set(setup.task1_safe_apples)
        poison = set(setup.task1_poisoned_apples)
    else:
        safe = set(setup.task2_safe_apples)
        poison = set(setup.task2_poisoned_apples)
    return safe, poison


def draw_background(ax: plt.Axes, setup: EnvSetup, task: int) -> None:
    safe, poison = get_task_apple_sets(setup, task)
    ax.set_xlim(-0.5, setup.grid_size - 0.5)
    ax.set_ylim(-0.5, setup.grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(setup.grid_size))
    ax.set_yticks(range(setup.grid_size))
    ax.grid(True, linewidth=0.8, alpha=0.3)
    ax.invert_yaxis()
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for r, c in safe:
        ax.scatter(c, r, s=230, c="tab:green", marker="o", zorder=3)
        ax.text(c, r, "A", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    for r, c in poison:
        ax.scatter(c, r, s=230, c="tab:red", marker="o", zorder=3)
        ax.text(c, r, "P", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    sr, sc = setup.agent_start_pos
    ax.scatter(sc, sr, s=170, facecolors="none", edgecolors="tab:blue", linewidths=2.2, zorder=4)

def draw_trajectory(
    ax: plt.Axes,
    setup: EnvSetup,
    task: int,
    actor: torch.nn.Module,
    title: str,
    seed: int,
) -> dict:
    env = make_env(setup, task=task, seed=seed)
    trace = collect_trajectory(env, actor)
    draw_background(ax, setup, task)

    path = trace["path"]
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    ax.plot(xs, ys, color="tab:blue", linewidth=2.3, alpha=0.9, zorder=6)
    ax.scatter(xs[0], ys[0], c="tab:blue", s=70, marker="s", zorder=7)
    ax.scatter(xs[-1], ys[-1], c="black", s=70, marker="*", zorder=8)

    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 0.05, y - 0.18, f"{idx}", fontsize=7, color="black", zorder=8)

    for poisoned_step in trace["poisoned_hit_steps"]:
        xh, yh = xs[poisoned_step], ys[poisoned_step]
        ax.scatter(xh, yh, s=160, marker="x", c="crimson", linewidths=2.5, zorder=9)

    ax.set_title(title, fontsize=10)
    return trace


def save_metrics_csv(path: str, metrics: dict[str, dict[str, float]]) -> None:
    fieldnames = ["model_task", "avg_reward", "avg_performance_success", "avg_safety_success", "avg_overall_success"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, vals in metrics.items():
            row = {"model_task": key}
            row.update(vals)
            writer.writerow(row)


def make_forgetting_figure(
    setup: EnvSetup,
    seed: int,
    standard_actor: torch.nn.Module,
    amnesic_actor: torch.nn.Module,
    metrics: dict[str, dict[str, float]],
    output_dir: str,
    dpi: int,
    show: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))

    t_noadapt_t1 = draw_trajectory(
        axes[0, 0], setup, task=1, actor=standard_actor, title="(A) NoAdapt on Task 1", seed=seed
    )
    draw_trajectory(
        axes[0, 1], setup, task=2, actor=standard_actor, title="(B) NoAdapt on Task 2", seed=seed
    )
    t_amnesic_t1 = draw_trajectory(
        axes[1, 0], setup, task=1, actor=amnesic_actor, title="(C) UnsafeAdapt on Task 1", seed=seed
    )
    draw_trajectory(
        axes[1, 1], setup, task=2, actor=amnesic_actor, title="(D) UnsafeAdapt on Task 2", seed=seed
    )

    safety_noadapt = metrics["NoAdapt / Task 1"]["avg_safety_success"]
    safety_amnesic = metrics["UnsafeAdapt / Task 1"]["avg_safety_success"]
    forget_score = safety_noadapt - safety_amnesic
    fig.suptitle(
        (
            f"Unsafe Adaptation Causes Task-1 Safety Forgetting ({setup.name}, seed={seed})\n"
            f"Task-1 safety: NoAdapt={safety_noadapt:.2f}, UnsafeAdapt={safety_amnesic:.2f}, "
            f"forgetting={forget_score:.2f}; "
            f"T1 poisoned hits (NoAdapt/UnsafeAdapt)="
            f"{len(t_noadapt_t1['poisoned_hit_steps'])}/{len(t_amnesic_t1['poisoned_hit_steps'])}"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    png_path = os.path.join(output_dir, f"{setup.name}_seed{seed}_forgetting_panels.png")
    pdf_path = os.path.join(output_dir, f"{setup.name}_seed{seed}_forgetting_panels.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved panel figure: {png_path}")
    print(f"Saved panel figure: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    setup = get_setup_by_name(args.setup)
    set_all_seeds(args.seed)

    env1 = make_env(setup, task=1, seed=args.seed)
    env2 = make_env(setup, task=2, seed=args.seed)

    noadapt_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.task1_steps,
        eval_episodes=0,
        device=args.device,
    )
    standard_actor, standard_critic = ppo_train(env=env1, cfg=noadapt_cfg)

    noadapt_task1 = evaluate_policy(env1, standard_actor, num_episodes=args.eval_episodes)
    if noadapt_task1["avg_safety_success"] < 1.0:
        raise ValueError(
            f"NoAdapt is not fully safe on Task 1 for this run: {noadapt_task1['avg_safety_success']:.3f}. "
            "Choose a setup/seed where Task 1 safety starts at 1.0."
        )

    set_all_seeds(args.seed)
    unsafe_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.task2_steps,
        eval_episodes=0,
        device=args.device,
    )
    amnesic_actor, _ = ppo_train(
        env=env2,
        cfg=unsafe_cfg,
        actor_warm_start=standard_actor,
        critic_warm_start=standard_critic,
    )

    metrics = {
        "NoAdapt / Task 1": noadapt_task1,
        "NoAdapt / Task 2": evaluate_policy(env2, standard_actor, num_episodes=args.eval_episodes),
        "UnsafeAdapt / Task 1": evaluate_policy(env1, amnesic_actor, num_episodes=args.eval_episodes),
        "UnsafeAdapt / Task 2": evaluate_policy(env2, amnesic_actor, num_episodes=args.eval_episodes),
    }

    print(
        f"[{setup.name}] seed={args.seed} t1={args.task1_steps} t2={args.task2_steps} | "
        f"NoAdapt T1 safety={metrics['NoAdapt / Task 1']['avg_safety_success']:.3f} | "
        f"UnsafeAdapt T1 safety={metrics['UnsafeAdapt / Task 1']['avg_safety_success']:.3f} | "
        f"UnsafeAdapt T2 safety={metrics['UnsafeAdapt / Task 2']['avg_safety_success']:.3f}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_dir, f"{setup.name}_seed{args.seed}_metrics.csv")
    save_metrics_csv(metrics_csv_path, metrics)
    print(f"Saved metrics CSV: {metrics_csv_path}")

    make_forgetting_figure(
        setup=setup,
        seed=args.seed,
        standard_actor=standard_actor,
        amnesic_actor=amnesic_actor,
        metrics=metrics,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
