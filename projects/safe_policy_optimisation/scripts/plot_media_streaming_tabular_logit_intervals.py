#!/usr/bin/env python
"""Plot Media Streaming logit intervals from a precomputed PSPO region."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.stages.train_ppo_shield import load_shield_mask  # noqa: E402


DEFAULT_RASHOMON_DIRS = {
    "tabular": (
        REPO
        / "outputs/_pspo_hparam/media_streaming_tabular_precomputed_crown_iters40_margin10/set"
    ),
    "one_hidden": (
        REPO
        / "outputs/_pspo_hparam/media_streaming_1hidden_precomputed_iters10k20k30k_margins0p5_2_5"
        / "precomputed/iters_20000__margin_0p5/set"
    ),
}
DEFAULT_SHIELD_PATH = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/inputs/media_streaming/shield_q.pt"
)
DEFAULT_OUTPUT_ROOT = (
    REPO / "projects/safe_policy_optimisation/results/pspo/precomputed/logit_intervals"
)

ARCHITECTURES = {
    "tabular": {
        "n_hidden": 0,
        "title": "tabular",
        "output_name": "media_streaming_tabular_logit_intervals",
    },
    "one_hidden": {
        "n_hidden": 1,
        "title": "one-hidden-layer",
        "output_name": "media_streaming_one_hidden_logit_intervals",
    },
}

BUFFER_SIZE = 21
REPRESENTATIVE_STATES = [
    (0, 0),
    (10, 10),
    (20, 19),
    (0, 20),
    (10, 20),
    (20, 21),
]
ACTION_LABELS = ["Slow (a=0)", "Fast (a=1)"]
SAFE_COLOR = "#248A57"
UNSAFE_COLOR = "#C64E43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="tabular")
    parser.add_argument("--rashomon-dir", type=Path, default=None)
    parser.add_argument("--shield-path", type=Path, default=DEFAULT_SHIELD_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_region(
    rashomon_dir: Path,
) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
    bounds: dict[str, Any] = torch.load(
        rashomon_dir / "rashomon_param_bounds.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy: dict[str, Any] = torch.load(
        rashomon_dir / "base_policy.pt",
        map_location="cpu",
        weights_only=False,
    )
    architecture = policy.get("architecture") or {}
    if int(architecture.get("n_hidden", -1)) not in (0, 1):
        raise ValueError("Expected a tabular or one-hidden-layer actor.")
    if int(architecture.get("input_dim", -1)) != 462:
        raise ValueError(f"Expected 462 input states, got {architecture.get('input_dim')!r}.")
    if int(architecture.get("n_actions", -1)) != 2:
        raise ValueError(f"Expected two actions, got {architecture.get('n_actions')!r}.")

    lower = list(bounds["param_bounds_l"])
    upper = list(bounds["param_bounds_u"])
    state_dict = policy["state_dict"]
    expected_bounds = 2 if int(architecture["n_hidden"]) == 0 else 4
    if len(lower) != expected_bounds or len(upper) != expected_bounds:
        raise ValueError(
            f"Expected {expected_bounds} parameter bounds, got lower={len(lower)}, upper={len(upper)}."
        )
    return (
        architecture,
        [tensor.detach() for tensor in lower],
        [tensor.detach() for tensor in upper],
        {name: tensor.detach() for name, tensor in state_dict.items()},
    )


def action_logit_intervals(
    architecture: dict[str, Any],
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    state_dict: dict[str, torch.Tensor],
    state: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lower, initial-policy, and upper action logits for one one-hot state."""

    if int(architecture["n_hidden"]) == 0:
        weight_l, bias_l = lower
        weight_u, bias_u = upper
        weight = state_dict["0.weight"]
        bias = state_dict["0.bias"]
        return (
            weight_l[:, state] + bias_l,
            weight[:, state] + bias,
            weight_u[:, state] + bias_u,
        )

    first_weight_l, first_bias_l, output_weight_l, output_bias_l = lower
    first_weight_u, first_bias_u, output_weight_u, output_bias_u = upper
    first_weight = state_dict["0.weight"]
    first_bias = state_dict["0.bias"]
    output_weight = state_dict["2.weight"]
    output_bias = state_dict["2.bias"]

    hidden_pre_l = first_weight_l[:, state] + first_bias_l
    hidden_pre_u = first_weight_u[:, state] + first_bias_u
    hidden_l = torch.tanh(hidden_pre_l)
    hidden_u = torch.tanh(hidden_pre_u)

    products = torch.stack(
        (
            output_weight_l * hidden_l.unsqueeze(0),
            output_weight_l * hidden_u.unsqueeze(0),
            output_weight_u * hidden_l.unsqueeze(0),
            output_weight_u * hidden_u.unsqueeze(0),
        ),
        dim=0,
    )
    logits_l = products.min(dim=0).values.sum(dim=1) + output_bias_l
    logits_u = products.max(dim=0).values.sum(dim=1) + output_bias_u

    hidden = torch.tanh(first_weight[:, state] + first_bias)
    nominal = output_weight @ hidden + output_bias
    return logits_l, nominal, logits_u


def state_id(buffer_level: int, fast_count: int) -> int:
    return int(fast_count) * BUFFER_SIZE + int(buffer_level)


def interval_rows(
    rashomon_dir: Path, shield_path: Path, *, expected_n_hidden: int
) -> list[dict[str, int | float | bool | str]]:
    architecture, lower, upper, state_dict = load_region(rashomon_dir)
    if int(architecture["n_hidden"]) != expected_n_hidden:
        raise ValueError(
            f"Requested n_hidden={expected_n_hidden}, but artifact has "
            f"n_hidden={architecture['n_hidden']}."
        )
    shield = np.asarray(load_shield_mask(shield_path), dtype=bool)
    if shield.shape != (462, 2):
        raise ValueError(f"Expected shield shape (462, 2), got {shield.shape}.")

    rows: list[dict[str, int | float | bool | str]] = []
    for buffer_level, fast_count in REPRESENTATIVE_STATES:
        state = state_id(buffer_level, fast_count)
        logits_l, nominal, logits_u = action_logit_intervals(
            architecture, lower, upper, state_dict, state
        )
        for action, action_label in enumerate(ACTION_LABELS):
            rows.append(
                {
                    "state_id": state,
                    "buffer_level": buffer_level,
                    "fast_count": fast_count,
                    "action": action,
                    "action_label": action_label,
                    "safe": bool(shield[state, action]),
                    "lower_logit": float(logits_l[action]),
                    "nominal_logit": float(nominal[action]),
                    "upper_logit": float(logits_u[action]),
                }
            )
    return rows


def write_csv(
    rows: list[dict[str, int | float | bool | str]], output_dir: Path, output_name: str
) -> None:
    with (output_dir / f"{output_name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, int | float | bool | str]],
    output_dir: Path,
    *,
    architecture_title: str,
    output_name: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    all_lower = [float(row["lower_logit"]) for row in rows]
    all_upper = [float(row["upper_logit"]) for row in rows]
    span = max(all_upper) - min(all_lower)
    y_min = min(all_lower) - 0.08 * span
    y_max = max(all_upper) + 0.16 * span

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), sharey=True)
    for ax, (buffer_level, fast_count) in zip(axes.flat, REPRESENTATIVE_STATES):
        state = state_id(buffer_level, fast_count)
        state_rows = [row for row in rows if int(row["state_id"]) == state]
        for x, row in enumerate(state_rows):
            nominal = float(row["nominal_logit"])
            lower = float(row["lower_logit"])
            upper = float(row["upper_logit"])
            safe = bool(row["safe"])
            color = SAFE_COLOR if safe else UNSAFE_COLOR
            ax.bar(
                x,
                upper - lower,
                bottom=lower,
                width=0.58,
                color=color,
                edgecolor=color,
                linewidth=1.4,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                x,
                nominal,
                marker="D",
                s=34,
                color="#202020",
                zorder=4,
            )
            label_y = upper + 0.035 * span
            ax.text(
                x,
                label_y,
                f"[{lower:.3f}, {upper:.3f}]",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#303030",
            )

        ax.axhline(0.0, color="#4A4A4A", linewidth=0.8, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, 1], ACTION_LABELS)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_title(
            f"State {state}: buffer={buffer_level}, fast count={fast_count}",
            pad=8,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Action logit")

    legend = [
        Patch(facecolor=SAFE_COLOR, edgecolor=SAFE_COLOR, label="Shield-safe action"),
        Patch(facecolor=UNSAFE_COLOR, edgecolor=UNSAFE_COLOR, label="Shield-unsafe action"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#202020",
            markerfacecolor="#202020",
            linestyle="None",
            label="Initial neural-policy logit",
        ),
    ]
    fig.suptitle(
        f"Media Streaming: logit intervals from the {architecture_title} PSPO safe parameter region",
        fontsize=15,
        y=0.99,
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    stem = output_dir / output_name
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    architecture_config = ARCHITECTURES[args.architecture]
    rashomon_dir = (args.rashomon_dir or DEFAULT_RASHOMON_DIRS[args.architecture]).resolve()
    shield_path = args.shield_path.resolve()
    output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_ROOT / f"media_streaming_{args.architecture}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = str(architecture_config["output_name"])
    rows = interval_rows(
        rashomon_dir,
        shield_path,
        expected_n_hidden=int(architecture_config["n_hidden"]),
    )
    write_csv(rows, output_dir, output_name)
    plot(
        rows,
        output_dir,
        architecture_title=str(architecture_config["title"]),
        output_name=output_name,
    )
    print(f"Wrote {output_dir / f'{output_name}.png'}")
    print(f"Wrote {output_dir / f'{output_name}.pdf'}")
    print(f"Wrote {output_dir / f'{output_name}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
