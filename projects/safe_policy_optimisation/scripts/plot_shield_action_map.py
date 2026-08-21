"""Plot the shield's allowed actions as arrows over a tabular grid-world layout.

Renders one figure per environment: lava, goal and start cells are shaded, and
every action the shield permits in a state is drawn as an arrow from that cell's
centre (a dot marks a permitted "stay").

Usage::

    python projects/safe_policy_optimisation/scripts/plot_shield_action_map.py \
        --envs bridge_crossing bridge_crossing_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import projects.safe_crl.utils.masa_tabular_envs  # noqa: F401  (registers the envs)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import load_shield_mask

REPO = Path(__file__).resolve().parents[3]
INPUTS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/inputs"
FIGURES = REPO / "projects/safe_policy_optimisation/figures"

ENVIRONMENTS = {
    "bridge_crossing": ("CustomBridgeCrossing-v0", "Bridge Crossing v1"),
    "bridge_crossing_v2": ("CustomBridgeCrossingV2-v0", "Bridge Crossing v2"),
}
ENV_KWARGS = {"observation_mode": "index", "slip_prob": 0.04}

LAVA_COLOUR = "#b03030"
GOAL_COLOUR = "#8fbf8f"
START_COLOUR = "#7fa8d0"
ARROW_COLOUR = "#1a3f6b"


def action_directions(
    transitions: np.ndarray, grid_size: int, terminal: set[int]
) -> dict[int, tuple[int, int]]:
    """Infer each action's (drow, dcol) as the modal displacement.

    ``transitions`` is indexed ``[next_state, state, action]``; the intended move
    is the most likely successor, so the mode over states recovers the direction
    even where walls distort individual cells. Terminal states are absorbing --
    every action there stays put -- so they are excluded or they would dominate.
    """

    n_states, n_actions = transitions.shape[1], transitions.shape[2]
    directions: dict[int, tuple[int, int]] = {}
    for action in range(n_actions):
        counts: dict[tuple[int, int], int] = {}
        for state in range(n_states):
            if state in terminal:
                continue
            successor = int(np.argmax(transitions[:, state, action]))
            delta = (
                successor // grid_size - state // grid_size,
                successor % grid_size - state % grid_size,
            )
            counts[delta] = counts.get(delta, 0) + 1
        directions[action] = max(counts.items(), key=lambda item: item[1])[0]
    return directions


def shield_reachable_states(
    transitions: np.ndarray, mask: np.ndarray, start: int, terminal: set[int]
) -> set[int]:
    """States reachable from ``start`` using only shield-permitted actions."""

    seen = {start}
    frontier = [start]
    while frontier:
        following = []
        for state in frontier:
            if state in terminal:
                continue
            for action in range(mask.shape[1]):
                if not mask[state][action]:
                    continue
                for successor in np.nonzero(transitions[:, state, action])[0]:
                    successor = int(successor)
                    if successor not in seen:
                        seen.add(successor)
                        following.append(successor)
        frontier = following
    return seen


def plot_environment(name: str, env_id: str, title: str, out_dir: Path) -> Path:
    env = gym.make(env_id, **ENV_KWARGS).unwrapped
    grid_size = env._grid_size
    transitions = np.asarray(env._transition_matrix)
    mask = np.asarray(load_shield_mask(INPUTS / name / "shield_q.pt"), dtype=int)
    lava = set(env._lava_states)
    goal = set(env._goal_states)
    start = int(env._start_state)

    directions = action_directions(transitions, grid_size, lava | goal)
    reachable = shield_reachable_states(transitions, mask, start, lava | goal)

    fig, ax = plt.subplots(figsize=(11, 11))
    for row in range(grid_size):
        for col in range(grid_size):
            state = row * grid_size + col
            if state in lava:
                colour = LAVA_COLOUR
            elif state in goal:
                colour = GOAL_COLOUR
            elif state == start:
                colour = START_COLOUR
            else:
                colour = "white"
            ax.add_patch(
                mpatches.Rectangle(
                    (col, row), 1, 1, facecolor=colour, edgecolor="#cccccc", linewidth=0.5
                )
            )

    length = 0.34
    for row in range(grid_size):
        for col in range(grid_size):
            state = row * grid_size + col
            if state in lava or state in goal:
                continue
            centre_x, centre_y = col + 0.5, row + 0.5
            for action in range(mask.shape[1]):
                if not mask[state][action]:
                    continue
                drow, dcol = directions[action]
                if (drow, dcol) == (0, 0):
                    ax.plot(centre_x, centre_y, "o", color=ARROW_COLOUR, markersize=3)
                    continue
                ax.arrow(
                    centre_x,
                    centre_y,
                    dcol * length,
                    drow * length,
                    head_width=0.16,
                    head_length=0.14,
                    fc=ARROW_COLOUR,
                    ec=ARROW_COLOUR,
                    length_includes_head=True,
                    linewidth=0.9,
                )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)  # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(grid_size) + 0.5)
    ax.set_xticklabels(range(grid_size), fontsize=7)
    ax.set_yticks(np.arange(grid_size) + 0.5)
    ax.set_yticklabels(range(grid_size), fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    permitted = int(mask.sum())
    contested = [s for s in range(grid_size**2) if s not in lava and int(mask[s].sum()) < mask.shape[1]]
    ax.set_title(
        f"{title} — actions permitted by the shield\n"
        f"{permitted} permitted state-action pairs; "
        f"{len(contested)} states restricted; "
        f"goal reachable under the shield: {bool(reachable & goal)}",
        fontsize=12,
        pad=14,
    )
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=LAVA_COLOUR, edgecolor="#cccccc", label="lava"),
            mpatches.Patch(facecolor=GOAL_COLOUR, edgecolor="#cccccc", label="goal"),
            mpatches.Patch(facecolor=START_COLOUR, edgecolor="#cccccc", label="start"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"shield_action_map_{name}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"shield_action_map_{name}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"{title}: action directions (drow, dcol) = {directions}")
    print(
        f"  permitted state-action pairs: {permitted} / {grid_size**2 * mask.shape[1]}"
        f" | restricted states: {len(contested)}"
        f" | states reachable under the shield: {len(reachable)}"
        f" | goal reachable: {bool(reachable & goal)}"
    )
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS), choices=list(ENVIRONMENTS))
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    for name in args.envs:
        env_id, title = ENVIRONMENTS[name]
        print("wrote", plot_environment(name, env_id, title, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
