from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ibp-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ibp-cache")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from mountaincar_utils import (
    ENV_ID,
    eval_stats_to_dict,
    evaluate_policy,
    make_env,
    save_policy_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO policy for MountainCar-v0 and save final frames."
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-timesteps", type=int, default=400_000)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/mountaincar/ppo"))
    parser.add_argument("--success-threshold", type=float, default=-110.0)
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--quick-eval-episodes", type=int, default=20)
    parser.add_argument("--confirm-eval-episodes", type=int, default=100)
    parser.add_argument("--render-seed", type=int, default=10_007)
    parser.add_argument("--render-seed-search", type=int, default=50)
    parser.add_argument("--frame-limit", type=int, default=200)
    parser.add_argument("--gif-duration-ms", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_random_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(make_env(seed=args.seed, shaped=True))
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_range=0.2,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=args.verbose,
    )

    total_timesteps = 0
    best_mean_return = float("-inf")
    best_model_path = args.out_dir / "best_model.zip"
    solved = False
    quick_eval = None
    confirm_eval = None
    history: list[dict[str, float | int | bool]] = []

    try:
        while total_timesteps < args.max_timesteps:
            chunk = min(args.chunk_timesteps, args.max_timesteps - total_timesteps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=(total_timesteps == 0),
                progress_bar=args.progress_bar,
            )
            total_timesteps += chunk

            quick_eval = evaluate_policy(
                model,
                episodes=args.quick_eval_episodes,
                seed=args.seed + total_timesteps,
            )
            is_best = quick_eval.mean_return > best_mean_return
            if is_best:
                best_mean_return = quick_eval.mean_return
                model.save(best_model_path)

            history.append(
                {
                    "timesteps": total_timesteps,
                    "mean_return": quick_eval.mean_return,
                    "success_rate": quick_eval.success_rate,
                    "is_best": is_best,
                }
            )

            print(
                f"[PPO] timesteps={total_timesteps} "
                f"quick_mean={quick_eval.mean_return:.2f} "
                f"success_rate={quick_eval.success_rate:.2f}",
                flush=True,
            )

            if quick_eval.mean_return >= args.success_threshold:
                confirm_eval = evaluate_policy(
                    model,
                    episodes=args.confirm_eval_episodes,
                    seed=args.seed + total_timesteps + 100_000,
                )
                print(
                    f"[PPO] confirm_mean={confirm_eval.mean_return:.2f} "
                    f"success_rate={confirm_eval.success_rate:.2f}",
                    flush=True,
                )
                if confirm_eval.mean_return >= args.success_threshold:
                    solved = True
                    model.save(best_model_path)
                    break
    finally:
        train_env.close()

    final_model = PPO.load(best_model_path)
    final_eval = evaluate_policy(
        final_model,
        episodes=args.confirm_eval_episodes,
        seed=args.seed + 200_000,
    )
    frames = save_policy_frames(
        final_model,
        frames_dir=args.out_dir / "frames",
        gif_path=args.out_dir / "final_policy.gif",
        seed=args.render_seed,
        render_seed_search=args.render_seed_search,
        frame_limit=args.frame_limit,
        gif_duration_ms=args.gif_duration_ms,
    )

    final_model_path = args.out_dir / "model.zip"
    final_model.save(final_model_path)
    metrics = {
        "algorithm": "PPO",
        "env_id": ENV_ID,
        "seed": args.seed,
        "max_timesteps": args.max_timesteps,
        "total_timesteps": total_timesteps,
        "success_threshold": args.success_threshold,
        "solved": (solved or final_eval.mean_return >= args.success_threshold)
        and final_eval.mean_return >= args.success_threshold,
        "quick_eval": eval_stats_to_dict(quick_eval) if quick_eval else None,
        "confirm_eval": eval_stats_to_dict(confirm_eval) if confirm_eval else None,
        "final_eval": eval_stats_to_dict(final_eval),
        "history": history,
        "model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "frames": frames,
    }
    write_json(args.out_dir / "eval_metrics.json", metrics)

    if metrics["solved"]:
        print(f"[PPO] solved; artifacts written to {args.out_dir}", flush=True)
        return 0

    print(
        f"[PPO] did not reach mean return {args.success_threshold}; "
        f"best/final artifacts written to {args.out_dir}",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
