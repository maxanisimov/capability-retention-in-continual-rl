"""Synthesise a probabilistic safety shield for a tabular environment and save its Q-function.

Shared across the safety_retention environment pipelines. Two modes, selected by ``--env``:

1. ``--env FrozenLake-v1`` (default): the gym FrozenLake driven by a task in
   FrozenLake/settings/tasks/tasks.yaml (one map plus its own slip dynamics). ``--task`` is
   the tasks.yaml key, e.g. ``diagonal_4x4_stochastic_source``. The shield is synthesised by
   value iteration over ``env.unwrapped.P``; each cell+action gets its probability of
   EVENTUAL SAFETY (max probability of never falling into a hole).

2. ``--env Custom...-v0``: a MASA-style tabular environment from
   projects.safe_crl.utils.masa_tabular_envs (CustomMiniPacman-v0, CustomBridgeCrossing-v0, ...).
   The shield is synthesised with projects.safe_crl.utils.shield_utils (exact tabular value
   iteration over the env's transition model, using its label_fn/cost_fn to define the
   unsafe set). Small envs use the dense transition matrix; larger ones (e.g. Pacman) use
   the sparse successor-states dict.

Only synthesises and SAVES the shield Q-function. Output goes to the environment directory:
``<safety_retention>/<env>/artifacts/shields/<task>/shield_q.pt`` (``<env>`` is the family
umbrella dir: ``FrozenLake`` for FrozenLake-v1, else the env id with its version suffix
stripped, e.g. ``CustomColourBombGridWorldV2-v0`` -> ``CustomColourBombGridWorld``). Use
plot_shield.py to render it.

    python synthesise_shield.py --task diagonal_4x4_stochastic_source
    python synthesise_shield.py --env CustomMiniPacman-v0 --task minipacman_default
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Some MASA env constructors touch SDL; keep it headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch

_SAFETY_RETENTION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from projects.safe_crl.pipelines.safety_retention.task_library import (
    environment_subdir,
    load_masa_task,
    masa_env_kwargs,
)

FROZEN_LAKE_ENV = "FrozenLake-v1"
ACTION_MEANING: dict[int, str] = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def default_output_dir(env_id: str, task: str) -> Path:
    return _SAFETY_RETENTION_ROOT / environment_subdir(env_id) / "artifacts" / "shields" / task


# --------------------------------------------------------------------------------------
# FrozenLake-v1 (tasks.yaml) path
# --------------------------------------------------------------------------------------
def build_frozenlake_env(task: dict) -> tuple[gym.Env, list[str]]:
    """Construct the FrozenLake env for a task and return (env, desc). No rendering needed
    for synthesis; the transition model ``env.unwrapped.P`` is built at construction."""
    desc = [str(row) for row in task["map"]]
    deterministic = bool(task.get("deterministic", True))
    slip_probability = float(task.get("slip_probability", 0.0))
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=not deterministic,
        success_rate=1.0 - slip_probability,
        max_episode_steps=int(task["max_episode_steps"]),
    )
    return env, desc


def classify_states(desc: list[str]) -> tuple[int, int, set[int], set[int]]:
    """Return (nrow, ncol, hole_states, goal_states) for a FrozenLake map."""
    nrow, ncol = len(desc), len(desc[0])
    holes: set[int] = set()
    goals: set[int] = set()
    for r in range(nrow):
        for c in range(ncol):
            tile = desc[r][c]
            state = r * ncol + c
            if tile == "H":
                holes.add(state)
            elif tile == "G":
                goals.add(state)
    return nrow, ncol, holes, goals


def min_hole_reachability(
    transitions: dict,
    n_states: int,
    holes: set[int],
    goals: set[int],
    *,
    tol: float = 1e-12,
    max_iters: int = 100_000,
) -> np.ndarray:
    """Minimal probability of eventually reaching a hole, per state (value iteration).

    Holes are absorbing-unsafe (r = 1), the goal is absorbing-safe (r = 0); for every
    other (transient) state r(s) = min_a sum_s' P(s'|s,a) r(s'). Iterating from r = 0 on
    transient states converges upward to the least fixed point, i.e. the minimal
    reachability probability of the hole set.
    """
    reach = np.zeros(n_states, dtype=np.float64)
    for hole in holes:
        reach[hole] = 1.0
    terminal = holes | goals

    for _ in range(max_iters):
        delta = 0.0
        new_reach = reach.copy()
        for state in range(n_states):
            if state in terminal:
                continue
            best = None
            for action in range(len(transitions[state])):
                acc = 0.0
                for prob, next_state, _reward, _terminated in transitions[state][action]:
                    acc += prob * reach[next_state]
                if best is None or acc < best:
                    best = acc
            if best is not None:
                new_reach[state] = best
                delta = max(delta, abs(best - reach[state]))
        reach = new_reach
        if delta < tol:
            break
    return reach


def action_safety_probabilities(
    transitions: dict,
    n_states: int,
    hole_reach: np.ndarray,
) -> np.ndarray:
    """Per (state, action) eventual-safety probability = 1 - sum_s' P(s'|s,a) r(s')."""
    n_actions = 4
    q_safety = np.zeros((n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            acc = 0.0
            for prob, next_state, _reward, _terminated in transitions[state][action]:
                acc += prob * hole_reach[next_state]
            q_safety[state, action] = 1.0 - acc
    return q_safety


def run_frozenlake_task(args: argparse.Namespace, output_path: Path) -> float:
    from projects.safe_crl.pipelines.safety_retention.FrozenLake.core.reference_settings import (
        load_task_definition,
    )

    task = load_task_definition(args.task)
    env, desc = build_frozenlake_env(task)
    try:
        transitions = env.unwrapped.P
        nrow, ncol, holes, goals = classify_states(desc)
        n_states = nrow * ncol
        hole_reach = min_hole_reachability(transitions, n_states, holes, goals)
        q_safety = action_safety_probabilities(transitions, n_states, hole_reach)
        state_safety = 1.0 - hole_reach
    finally:
        env.close()

    payload = {
        "env": FROZEN_LAKE_ENV,
        "task": args.task,
        "semantics": "avoid_holes_forever",  # value = P(eventually never enter a hole)
        "map": list(desc),
        "n_states": int(n_states),
        "nrow": int(nrow),
        "ncol": int(ncol),
        "n_actions": 4,
        "action_meaning": dict(ACTION_MEANING),
        "deterministic": bool(task.get("deterministic", True)),
        "slip_probability": float(task.get("slip_probability", 0.0)),
        "holes": sorted(holes),
        "goals": sorted(goals),
        "q_safety": torch.from_numpy(np.ascontiguousarray(q_safety, dtype=np.float64)),
        "state_safety": torch.from_numpy(np.ascontiguousarray(state_safety, dtype=np.float64)),
        "hole_reachability": torch.from_numpy(np.ascontiguousarray(hole_reach, dtype=np.float64)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return float(state_safety[0])


# --------------------------------------------------------------------------------------
# MASA tabular envs path
# --------------------------------------------------------------------------------------
def run_masa_env(args: argparse.Namespace, output_path: Path) -> float:
    from projects.safe_crl.utils.masa_tabular_envs import make_custom_masa_env
    from projects.safe_crl.utils.shield_utils import (
        synthesise_probabilistic_shield,
        synthesise_shield_from_successor_dict,
    )

    # The env's dynamics come from the family tasks.yaml: its `stochasticity` block (and any
    # `env_kwargs`) are applied to the constructor, so editing the yaml changes the env the
    # shield is synthesised for. --env-kwargs remains an optional one-off override on top.
    cli_override = json.loads(args.env_kwargs) if args.env_kwargs else None
    task_block = load_masa_task(args.env, args.task)
    env_kwargs = masa_env_kwargs(task_block, cli_override) or None

    env = make_custom_masa_env(args.env, env_kwargs=env_kwargs)
    env.reset(seed=args.seed)
    try:
        unwrapped = env.unwrapped
        transition_matrix = unwrapped.get_transition_matrix()
        if transition_matrix is not None:
            # Small envs expose a dense transition matrix.
            shield, info = synthesise_probabilistic_shield(
                env,
                transition_matrix_fn=lambda e: e.get_transition_matrix(),
                label_fn=unwrapped.label_fn,
                cost_fn=unwrapped.cost_fn,
                risk_threshold=args.risk_threshold,
                use_masa_helper=False,
                return_info=True,
            )
        elif unwrapped.has_successor_states_dict:
            # Larger envs (e.g. Pacman) only expose a sparse successor-states dict, since a
            # dense O(|S|^2) matrix is infeasible; exact VI runs on the sparse support.
            shield, info = synthesise_shield_from_successor_dict(
                env,
                label_fn=unwrapped.label_fn,
                cost_fn=unwrapped.cost_fn,
                shield_type="probabilistic",
                risk_threshold=args.risk_threshold,
                return_info=True,
            )
        else:
            raise ValueError(
                f"{args.env} exposes neither a dense transition matrix nor a successor-states "
                "dict; exact tabular shield synthesis requires one of them.",
            )

        action_risk = np.asarray(info.action_risk, dtype=np.float64)
        state_risk = np.asarray(info.state_risk, dtype=np.float64)
        q_safety = 1.0 - action_risk            # per-(state, action) eventual-safety probability
        state_safety = 1.0 - state_risk
        n_states, n_actions = q_safety.shape
        start_state = int(getattr(unwrapped, "_start_state", 0))

        payload = {
            "env": args.env,
            "env_kwargs": env_kwargs,
            "task": args.task,
            "semantics": "avoid_unsafe_forever",  # value = P(eventually never enter an unsafe state)
            "n_states": int(n_states),
            "n_actions": int(n_actions),
            "risk_threshold": float(args.risk_threshold),
            "q_safety": torch.from_numpy(np.ascontiguousarray(q_safety, dtype=np.float64)),
            "action_risk": torch.from_numpy(np.ascontiguousarray(action_risk, dtype=np.float64)),
            "state_safety": torch.from_numpy(np.ascontiguousarray(state_safety, dtype=np.float64)),
            "state_risk": torch.from_numpy(np.ascontiguousarray(state_risk, dtype=np.float64)),
            "shield": torch.from_numpy(np.ascontiguousarray(np.asarray(shield), dtype=np.int64)),
            "vi_steps": None if info.vi_steps is None else int(info.vi_steps),
            "vi_residual": None if info.vi_residual is None else float(info.vi_residual),
            "start_state": start_state,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        return float(state_safety[start_state])
    finally:
        env.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesise a probabilistic safety shield for a tabular env and save its Q-function.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=FROZEN_LAKE_ENV,
        help=(
            "Environment id. 'FrozenLake-v1' (default) uses FrozenLake/settings/tasks/tasks.yaml "
            "via --task; a 'Custom...-v0' id uses a MASA tabular env from "
            "projects.safe_crl.utils.masa_tabular_envs."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "A task-library key from the env family's settings/tasks/tasks.yaml. For "
            "FrozenLake-v1 that is FrozenLake/settings/tasks/tasks.yaml (e.g. "
            "diagonal_4x4_stochastic_source); for a MASA env it is the matching key whose "
            "`stochasticity`/`env_kwargs` block defines the env instance."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to env.reset() during synthesis (MASA envs).",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.0,
        help="MASA shield: allow actions whose eventual unsafe risk is at most this. Default 0.0.",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default=None,
        help=(
            "MASA shield: optional JSON dict of constructor kwargs that OVERRIDE the task's "
            "env_kwargs/stochasticity for a one-off run (the task yaml is the usual source)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory. Default: <safety_retention>/<env>/artifacts/shields/<task>/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir or default_output_dir(args.env, args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shield_q.pt"

    if args.env == FROZEN_LAKE_ENV:
        start_safety = run_frozenlake_task(args, output_path)
        plot_hint = f"python plot_shield.py --task {args.task}"
    else:
        start_safety = run_masa_env(args, output_path)
        plot_hint = f"python plot_shield.py --env {args.env} --task {args.task}"

    print(f"Saved shield Q-function to {output_path}")
    print(f"Start-state eventual-safety value: {start_safety:.3f}")
    print(f"Plot it with: {plot_hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
