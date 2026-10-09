#!/usr/bin/env python3
"""LaTeX table describing the evaluation environments.

For the six MASA environments (exact env_id/env_kwargs of the canonical PSPO
runs) and the stochastic 128x128 FrozenLake, report the state and action space
sizes, unsafe and terminal states, states reachable from the start, safety-
critical states, the stochastic components and the policy's parameter count.

* Reachable: states that can be visited from the start state(s) under some
  action sequence, including unsafe and terminal states that can be entered.
* Safety-critical: states, not themselves unsafe, in which the shield used by
  the experiments forbids at least one action.
* Policy parameters: the actor of the trained policy, counted from its saved
  initial-policy state dict.

Colour Bomb v2 uses the shield of the results reported in the paper, i.e. the
version without random actions (see inputs/colour_bomb_v2/_pre_slipfix_20261007).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import projects.safe_crl.utils.masa_tabular_envs  # noqa: E402,F401  registers Custom* ids
from plot_pspo_variant_reward_time_aamas import CONTROLS, DEFAULT_RUNS, ENVIRONMENTS  # noqa: E402

INPUTS = DEFAULT_RUNS.parent / "inputs"
SHIELD_PATHS = {environment: INPUTS / environment / "shield_q.pt" for environment, _ in ENVIRONMENTS}
SHIELD_PATHS["colour_bomb_v2"] = INPUTS / "colour_bomb_v2/_pre_slipfix_20261007/shield_q.pt"
FROZENLAKE_RUN = DEFAULT_RUNS / "pspo_stochastic_frozenlake128_safety_only_state_id_lookup"
DEFAULT_OUTPUT = REPO / "projects/safe_policy_optimisation/results/environments"

STOCHASTICITY = {
    "media_streaming": "Packet arrival w.p.\\ 0.1 (slow) or 0.9 (fast); playback w.p.\\ 0.7",
    "colour_bomb": "Random action w.p.\\ 0.1",
    "colour_bomb_v2": "Random start (20 states) and restart after each goal; no random actions",
    "bridge_crossing": "Random action w.p.\\ 0.04",
    "bridge_crossing_v2": "Random action w.p.\\ 0.04",
    "mini_pacman": "Ghost moves randomly w.p.\\ 0.6, otherwise towards the agent",
    "frozenlake128": "Slip: intended move w.p.\\ 0.8, each perpendicular move w.p.\\ 0.1",
}
LABELS = {environment: title.replace("\n", " ") for environment, title in ENVIRONMENTS}
LABELS["frozenlake128"] = "FrozenLake ($128\\times128$)"


def parameter_count(base_policy: Path) -> int:
    state_dict = torch.load(base_policy, map_location="cpu", weights_only=False)["state_dict"]
    return int(sum(tensor.numel() for tensor in state_dict.values()))


def reachable(starts, successors, stop) -> int:
    seen, queue = set(starts), deque(starts)
    while queue:
        state = queue.popleft()
        if state in stop:
            continue
        for successor in successors(state):
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return len(seen)


def safety_critical(shield_path: Path, n_actions: int) -> int:
    payload = torch.load(shield_path, map_location="cpu", weights_only=False)
    allowed = np.asarray(payload["shield"]).astype(bool).sum(axis=1)
    return int(((allowed > 0) & (allowed < n_actions)).sum())


def masa_row(environment: str) -> dict:
    config = json.loads(
        (DEFAULT_RUNS / CONTROLS[environment] / "two_hidden" / environment / "seed0" / "config.json").read_text()
    )
    kwargs = {key: value for key, value in config["env_kwargs"].items() if key != "observation_mode"}
    env = gym.make(config["env_id"], observation_mode="index", **kwargs).unwrapped
    n_states, n_actions = int(env.observation_space.n), int(env.action_space.n)
    unsafe = {state for state in range(n_states) if env.cost_fn(env.label_fn(state)) > 0}
    if environment == "mini_pacman":
        # Episodes end when the agent reaches the terminal cell, whatever the ghost does.
        terminal = {index for key, index in env._state_map.items()  # noqa: SLF001
                    if (key[1], key[0]) == (env._agent_term_x, env._agent_term_y)}  # noqa: SLF001
        successors = lambda state: {int(next_state) for next_state in env._successor_states[state]}  # noqa: E731,SLF001
    else:
        # Media Streaming has no terminal states: its episodes end after a fixed number of steps.
        terminal = set(getattr(env, "_terminal_states", []) or [])
        matrix = env._transition_matrix  # noqa: SLF001
        successors = lambda state: set(np.flatnonzero(matrix[:, state, :].sum(axis=1)).tolist())  # noqa: E731
    starts = list(getattr(env, "_start_states", None) or [env._start_state])  # noqa: SLF001
    return {
        "states": n_states, "actions": n_actions, "unsafe": len(unsafe), "terminal": len(terminal),
        "reachable": reachable(starts, successors, unsafe | terminal),
        "safety_critical": safety_critical(SHIELD_PATHS[environment], n_actions),
        "policy_parameters": parameter_count(Path(config["base_policy_path"])),
    }


def frozenlake_row() -> dict:
    layout = (FROZENLAKE_RUN / "_inputs/layout.txt").read_text().split()
    size, n_actions = len(layout), 4
    n_states = size * size
    cell = lambda state: layout[state // size][state % size]  # noqa: E731
    holes = {state for state in range(n_states) if cell(state) == "H"}
    goals = {state for state in range(n_states) if cell(state) == "G"}
    start = next(state for state in range(n_states) if cell(state) == "S")
    moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # Gymnasium: left, down, right, up

    def move(state: int, direction: int) -> int:
        row, column = divmod(state, size)
        d_row, d_column = moves[direction]
        return min(max(row + d_row, 0), size - 1) * size + min(max(column + d_column, 0), size - 1)

    def successors(state: int) -> set[int]:
        # The intended move and both perpendicular slips.
        return {move(state, direction) for action in range(n_actions)
                for direction in (action, (action - 1) % 4, (action + 1) % 4)}

    return {
        "states": n_states, "actions": n_actions, "unsafe": len(holes), "terminal": len(holes | goals),
        "reachable": reachable([start], successors, holes | goals),
        "safety_critical": safety_critical(FROZENLAKE_RUN / "_inputs/shield_q.pt", n_actions),
        "policy_parameters": parameter_count(FROZENLAKE_RUN / "_inputs/base_policy.pt"),
    }


def latex_table(rows: dict[str, dict]) -> str:
    def header(*lines: str) -> str:
        return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"

    columns = [
        header("$|S|$"), header("$|A|$"), header("Unsafe", "states"), header("Terminal", "states"),
        header("Reachable", "from start"), header("Safety-critical", "states"),
        header("Stochastic", "components"), header("Policy", "parameters"),
    ]
    lines = [
        "% Generated by scripts/generate_environment_description_table.py. Paste-ready for the",
        "% AAMAS (acmart-based) class, which already loads booktabs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Evaluation environments. $|S|$ and $|A|$ are the numbers of states and actions. "
        "Terminal states end an episode (Media Streaming episodes end after 40 steps). Reachable "
        "counts the states that can be visited from the start state(s). Safety-critical states are "
        "states, not themselves unsafe, in which the shield forbids at least one action: the MASA "
        "shields allow only the safest actions under sound safety value iteration, and the FrozenLake "
        "shield is the almost-sure (winning-set) shield. A random action replaces the chosen action "
        "by a uniformly random different one. Policy parameters count the actor network (two hidden "
        "layers of 64 units, one-hot state input).}",
        "\\label{tab:environments}",
        "\\begin{tabular}{lrrrrrr>{\\raggedright\\arraybackslash}p{3.8cm}r}",
        "\\toprule",
        "Environment & " + " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for environment, row in rows.items():
        numbers = [f"{row[key]:,}" for key in ("states", "actions", "unsafe", "terminal", "reachable", "safety_critical")]
        lines.append(f"{LABELS[environment]} & " + " & ".join(numbers)
                     + f" & {STOCHASTICITY[environment]} & {row['policy_parameters']:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = {environment: masa_row(environment) for environment, _ in ENVIRONMENTS}
    rows["frozenlake128"] = frozenlake_row()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = latex_table(rows)
    (args.output_dir / "environment_description_table.tex").write_text(table)
    (args.output_dir / "environment_description.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(table)
    print(f"Saved environment_description_table.tex and environment_description.json in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
