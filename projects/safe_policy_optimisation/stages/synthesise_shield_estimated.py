"""Synthesise a safety shield from an ESTIMATED critic (data-driven, no dynamics).

Alternative to ``synthesise_shield.py``'s exact value iteration. Instead of
reading the environment's transition matrix, this:

1. collects a dataset of diverse interactions via **uniform-random** rollouts
   (only ``reset()``/``step()`` and the per-step observed cost are used -- the
   transition matrix is never read);
2. estimates the empirical transition model ``P_hat(s'|s,a)`` from counts;
3. runs the **same** eventual-unsafe-risk value iteration on ``P_hat`` via the
   shared solver ``safe_crl.utils.shield_utils.synthesise_shield_from_successor_dict``;
4. writes a ``shield_q.pt`` payload with the **same keys** as the exact
   synthesizer (plus estimation provenance), so every downstream consumer
   (``load_shield_mask``, PPO-Shield, ``compute_shield_rashomon_set``, PSPO)
   works unchanged.

Conservative by construction: any state-action never observed is routed to a
virtual absorbing *unsafe* sink (risk 1.0), so it is masked out. Poorer data
coverage therefore yields a strictly more conservative shield, converging to the
exact-VI shield as the number of episodes grows.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_crl.utils.masa_tabular_envs import make_custom_masa_env  # noqa: E402
from projects.safe_crl.utils.shield_utils import (  # noqa: E402
    synthesise_shield_from_successor_dict,
)
from projects.safe_crl.pipelines.safety_retention.synthesise_shield import (  # noqa: E402
    FROZEN_LAKE_ENV,
)
from projects.safe_crl.pipelines.safety_retention.task_library import (  # noqa: E402
    environment_subdir,
    load_masa_task,
    masa_env_kwargs,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

PROJECT_ROOT = REPO_ROOT / "projects" / "safe_policy_optimisation"


class _EmpiricalSuccessorEnv(gym.Env):
    """Minimal env exposing an EMPIRICAL successor-states dict to the shared VI
    solver in place of the true dynamics. Only the attributes the solver reads
    (``action_space``/``observation_space``/``get_successor_states_dict``) are
    provided."""

    def __init__(self, successors, transition_probs, n_states, n_actions):
        self._successors = successors
        self._transition_probs = transition_probs
        self._n_states = int(n_states)
        self.action_space = spaces.Discrete(int(n_actions))
        self.observation_space = spaces.Discrete(int(n_states))
        self.has_successor_states_dict = True

    @property
    def unwrapped(self):  # the solver calls env.unwrapped
        return self

    def get_successor_states_dict(self):
        return self._successors, self._transition_probs


def collect_uniform_data(env, *, n_episodes, seed, unsafe_cost_threshold):
    """Uniform-random rollouts -> (transition counts, observed states, observed
    unsafe states). Uses only reset()/step() plus the observed per-state cost
    ``cost_fn(label_fn(state))``; the transition matrix is never touched."""
    unwrapped = env.unwrapped
    n_actions = int(unwrapped.action_space.n)
    rng = np.random.default_rng(seed)
    counts: dict = defaultdict(lambda: defaultdict(int))
    observed_states: set[int] = set()
    observed_unsafe: set[int] = set()

    def _record(state_id):
        state_id = int(state_id)
        observed_states.add(state_id)
        if float(unwrapped.cost_fn(unwrapped.label_fn(state_id))) > unsafe_cost_threshold:
            observed_unsafe.add(state_id)

    for episode in range(int(n_episodes)):
        env.reset(seed=int(seed) + episode)
        s = int(unwrapped._state)
        _record(s)
        done = False
        while not done:
            a = int(rng.integers(n_actions))
            _obs, _r, terminated, truncated, _info = env.step(a)
            s_next = int(unwrapped._state)
            _record(s_next)
            counts[(s, a)][s_next] += 1
            done = bool(terminated or truncated)
            s = s_next
    return counts, observed_states, observed_unsafe


def build_empirical_successor_dict(counts, *, n_states, n_actions):
    """Empirical ``(successors, transition_probs)`` + a virtual absorbing unsafe
    sink (state id ``n_states``) that every UNOBSERVED (s,a) transitions to, so
    untried actions get risk 1.0 rather than a false zero. Returns
    ``(successors, transition_probs, n_states_aug, sink_id)``."""
    sink = int(n_states)
    n_aug = int(n_states) + 1
    obs_succ: dict = defaultdict(set)
    for (s, a), nexts in counts.items():
        obs_succ[int(s)].update(int(x) for x in nexts)

    successors: dict[int, list[int]] = {}
    transition_probs: dict[tuple[int, int], list[float]] = {}
    for s in range(n_states):
        support = sorted(obs_succ.get(s, set())) + [sink]  # sink always last
        successors[s] = support
        index = {sid: i for i, sid in enumerate(support)}
        for a in range(n_actions):
            nexts = counts.get((s, a))
            probs = [0.0] * len(support)
            if nexts:
                total = float(sum(nexts.values()))
                for sid, c in nexts.items():
                    probs[index[int(sid)]] = c / total
            else:
                probs[index[sink]] = 1.0  # unobserved (s,a) -> unsafe sink
            transition_probs[(s, a)] = probs
    successors[sink] = [sink]  # absorbing
    for a in range(n_actions):
        transition_probs[(sink, a)] = [1.0]
    return successors, transition_probs, n_aug, sink


def _estimate_shield(env, *, n_episodes, seed, unsafe_cost_threshold, theta, max_vi_steps):
    unwrapped = env.unwrapped
    n_actions = int(unwrapped.action_space.n)
    n_states = int(getattr(unwrapped, "_n_states", 0) or unwrapped.observation_space.n)

    counts, observed_states, observed_unsafe = collect_uniform_data(
        env, n_episodes=n_episodes, seed=seed, unsafe_cost_threshold=unsafe_cost_threshold,
    )
    successors, transition_probs, n_aug, _sink = build_empirical_successor_dict(
        counts, n_states=n_states, n_actions=n_actions,
    )

    observed_safe = observed_states - observed_unsafe

    def data_label_fn(state_id):  # passthrough; cost_fn consumes the id
        return int(state_id)

    def data_cost_fn(labels):  # unseen / sink / observed-unsafe -> 1.0
        return 0.0 if int(labels) in observed_safe else 1.0

    shim = _EmpiricalSuccessorEnv(successors, transition_probs, n_aug, n_actions)
    shield_aug, info = synthesise_shield_from_successor_dict(
        shim,
        label_fn=data_label_fn,
        cost_fn=data_cost_fn,
        shield_type="probabilistic",
        theta=theta,
        max_vi_steps=max_vi_steps,
        unsafe_cost_threshold=unsafe_cost_threshold,
        return_info=True,
    )
    # Drop the virtual sink row -> real (n_states, n_actions) tables.
    shield = np.asarray(shield_aug)[:n_states, :]
    action_risk = np.asarray(info.action_risk, dtype=np.float64)[:n_states, :]
    state_risk = np.asarray(info.state_risk, dtype=np.float64)[:n_states]
    coverage = {
        "state_action_coverage": len(counts) / float(n_states * n_actions),
        "n_state_action_observed": int(len(counts)),
        "n_states_visited": int(len(observed_states)),
        "n_states_total": int(n_states),
        "n_unsafe_states_observed": int(len(observed_unsafe)),
    }
    return shield, action_risk, state_risk, info, n_states, n_actions, coverage


def run(args: argparse.Namespace) -> Path:
    if args.env is None or args.task is None:
        raise ValueError("--env and --task are required for estimated shield synthesis.")
    if args.env == FROZEN_LAKE_ENV:
        raise ValueError("Estimated-critic synthesis targets the MASA tabular envs, not FrozenLake.")
    if args.n_rollout_episodes is None or int(args.n_rollout_episodes) <= 0:
        raise ValueError("--n-rollout-episodes must be a positive integer for estimated synthesis.")

    cli_override = json.loads(args.env_kwargs) if args.env_kwargs else None
    task_block = load_masa_task(args.env, args.task)
    env_kwargs = masa_env_kwargs(task_block, cli_override) or None
    max_episode_steps = getattr(args, "max_episode_steps", None)
    if max_episode_steps is None and task_block.get("max_episode_steps") is not None:
        max_episode_steps = int(task_block["max_episode_steps"])

    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "shields" / environment_subdir(args.env) / args.task)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shield_q.pt"

    theta = float(getattr(args, "theta", 1e-10))
    max_vi_steps = int(getattr(args, "max_vi_steps", 1000))
    unsafe_cost_threshold = float(getattr(args, "unsafe_cost_threshold", 0.5))
    n_episodes = int(args.n_rollout_episodes)
    seed = int(getattr(args, "seed", 0))

    env = make_custom_masa_env(args.env, max_episode_steps=max_episode_steps, env_kwargs=env_kwargs)
    try:
        unwrapped = env.unwrapped
        start_state = int(getattr(unwrapped, "_start_state", 0))
        log_info(
            f"Estimating safety critic from data: {n_episodes} uniform-random episodes "
            f"on {args.env} (seed={seed})"
        )
        shield, action_risk, state_risk, info, n_states, n_actions, coverage = _estimate_shield(
            env,
            n_episodes=n_episodes,
            seed=seed,
            unsafe_cost_threshold=unsafe_cost_threshold,
            theta=theta,
            max_vi_steps=max_vi_steps,
        )
    finally:
        env.close()

    q_safety = 1.0 - action_risk
    state_safety = 1.0 - state_risk

    payload = {
        "env": args.env,
        "env_kwargs": env_kwargs,
        "max_episode_steps": None if max_episode_steps is None else int(max_episode_steps),
        "task": args.task,
        "semantics": "avoid_unsafe_forever",
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "theta": theta,
        "max_vi_steps": max_vi_steps,
        "init_safety_bound": float(getattr(args, "init_safety_bound", 0.5)),
        "granularity": int(getattr(args, "granularity", 20)),
        "unsafe_cost_threshold": unsafe_cost_threshold,
        "q_safety": torch.from_numpy(np.ascontiguousarray(q_safety, dtype=np.float64)),
        "action_risk": torch.from_numpy(np.ascontiguousarray(action_risk, dtype=np.float64)),
        "state_safety": torch.from_numpy(np.ascontiguousarray(state_safety, dtype=np.float64)),
        "state_risk": torch.from_numpy(np.ascontiguousarray(state_risk, dtype=np.float64)),
        "shield": torch.from_numpy(np.ascontiguousarray(np.asarray(shield), dtype=np.int64)),
        "vi_steps": None if info.vi_steps is None else int(info.vi_steps),
        "vi_residual": None if info.vi_residual is None else float(info.vi_residual),
        "start_state": start_state,
        # estimation provenance / diagnostics
        "shield_method": "estimated_critic",
        "n_rollout_episodes": n_episodes,
        "behaviour_policy": getattr(args, "behaviour_policy", "uniform"),
        "estimation_seed": seed,
        "estimation_coverage": coverage,
    }
    torch.save(payload, output_path)
    log_info(f"Saved estimated shield to {output_path}")
    log_info(
        f"Data coverage: {coverage['state_action_coverage']:.3f} of (state,action) pairs observed; "
        f"{coverage['n_states_visited']}/{coverage['n_states_total']} states visited"
    )
    log_info(f"Start-state eventual-safety value: {float(state_safety[start_state]):.3f}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesise a safety shield from an ESTIMATED critic: collect diverse "
            "uniform-random interaction data, estimate the transition model, and run "
            "the shared eventual-unsafe-risk VI on it (no access to the true dynamics)."
        ),
    )
    parser.add_argument("--env", type=str, default=None, help="MASA Custom...-v0 environment id.")
    parser.add_argument("--task", type=str, default=None, help="Task-library key defining the env instance.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for data collection + env.reset.")
    parser.add_argument(
        "--n-rollout-episodes", type=int, default=5000,
        help="Number of uniform-random episodes to collect (the data-coverage knob).",
    )
    parser.add_argument(
        "--behaviour-policy", type=str, default="uniform", choices=("uniform",),
        help="Behaviour policy used to collect the interaction data.",
    )
    parser.add_argument("--theta", type=float, default=1e-10, help="VI convergence tolerance.")
    parser.add_argument("--max-vi-steps", type=int, default=1000, help="Maximum VI steps.")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Env max episode length.")
    parser.add_argument("--init-safety-bound", type=float, default=0.5, help="Recorded in metadata only.")
    parser.add_argument("--granularity", type=int, default=20, help="Recorded in metadata only.")
    parser.add_argument("--unsafe-cost-threshold", type=float, default=0.5, help="Cost threshold for unsafe states.")
    parser.add_argument("--env-kwargs", type=str, default=None, help="Optional JSON dict overriding task env_kwargs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dir to write shield_q.pt into.")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
