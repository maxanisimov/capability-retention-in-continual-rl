#!/usr/bin/env python
"""Illustrate what each term of PSPO's safe-initialisation loss forces on the logits.

Three hypothetical actions -- ``a_1`` safe, ``a_2`` unsafe, ``a_3`` safe -- under
the three objectives that are added in turn:

* **(a) multi-label CE.** ``logsumexp(safe) - logsumexp(all)`` in
  ``behavioural_clone`` (``core/provably_safe_policy_optimisation/safe_init.py``)
  maximises the *total* probability on the safe set. That is already satisfied
  when a single safe action takes essentially all of the mass, so the objective
  is indifferent to the remaining safe action, which may end up *below* the
  unsafe one. The greedy action is safe, but the policy is degenerate.
* **(b) minimum margin.** ``--bc-target-margin`` (default 2.0, see
  ``compute_shield_rashomon_set.py``) requires
  ``min_{a in safe} z_a - max_{b not in safe} z_b >= margin``, so *every* safe
  logit clears *every* unsafe logit -- not just the largest one.
* **(c) safe-action entropy.** ``--bc-safe-action-entropy-weight`` with
  ``--bc-min-safe-action-entropy`` (0.95) drives the conditional distribution
  over the safe actions towards uniform, which equalises the safe logits while
  the margin keeps them above the unsafe one.

The logit values below are **illustrative constants**. They do not come from any
environment, shield, or trained policy; the figure explains the objectives, it
does not report a result.

Sizing follows ``_paper_style``: 5.5 in is the ICLR single-column text width, so
include the PDF at native scale (no ``width=``) or the fonts stop matching the
sizes set here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    INK_PRIMARY,
    INK_SECONDARY,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)

DEFAULT_OUTPUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
OUTPUT_STEM = "safe_policy_initialisation_logits"

# Status colours, re-stepped from the palette's good/critical pair so the two
# separate under colour-vision deficiency: the palette steps (#0ca30c/#d03b3b)
# measure only dE 4.1 under deuteranopia, below the floor of 6. These measure
# 15.4 and pass every check. Shape and the axis labels carry the same
# distinction, so nothing here depends on hue alone.
SAFE_COLOR = "#0a6b18"
UNSAFE_COLOR = "#f0736f"
SAFE_MARKER = "o"
UNSAFE_MARKER = "X"

ACTION_LABELS = ("$a_1$\n(safe)", "$a_2$\n(unsafe)", "$a_3$\n(safe)")
IS_SAFE = (True, False, True)

# Illustrative logits only -- see the module docstring.
# The unsafe logit is held at 0.5 across (b) and (c) so that the eye tracks the
# safe actions, which are what those two objectives move.
PANELS = (
    {
        "title": "(a) multi-label CE only",
        # a_3 sits clearly *below* the unsafe action: the objective is met by
        # a_1 alone, so it constrains the second safe action not at all.
        "logits": (4.0, -0.6, -1.8),
        "note": "collapses onto\none safe action",
    },
    {
        "title": "(b) + minimum margin",
        "logits": (4.0, 0.5, 2.5),
        "note": None,
    },
    {
        "title": "(c) + safe-action entropy",
        "logits": (3.25, 0.5, 3.25),
        "note": None,
    },
)

Y_LIMITS = (-2.6, 4.9)
MARKER_SIZE = 7.5
STEM_WIDTH = 0.9
BASELINE = Y_LIMITS[0]


def _draw_panel(ax, panel: dict) -> None:
    """Draw one lollipop panel: a stem to the baseline plus the logit marker."""
    positions = range(len(ACTION_LABELS))
    for x, logit, safe in zip(positions, panel["logits"], IS_SAFE):
        colour = SAFE_COLOR if safe else UNSAFE_COLOR
        ax.plot(
            [x, x],
            [BASELINE, logit],
            color=colour,
            linewidth=STEM_WIDTH,
            alpha=0.55,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [x],
            [logit],
            marker=SAFE_MARKER if safe else UNSAFE_MARKER,
            markersize=MARKER_SIZE,
            color=colour,
            markeredgecolor=INK_PRIMARY,
            markeredgewidth=0.6,
            linestyle="none",
            zorder=3,
        )

    ax.set_title(panel["title"], pad=3.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(ACTION_LABELS)
    ax.set_xlim(-0.6, len(ACTION_LABELS) - 0.4)
    ax.set_ylim(*Y_LIMITS)
    # x is categorical, so only the y grid carries information.
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def _annotate_unsafe_level(ax, logits: tuple[float, ...]) -> None:
    """Faint reference line at the unsafe logit -- the level the margin is measured from."""
    ax.axhline(
        logits[1],
        color=UNSAFE_COLOR,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        alpha=0.85,
        zorder=1,
    )


def _annotate_margin(ax, logits: tuple[float, ...]) -> None:
    """Arrow spanning the gap the margin term enforces.

    Placed in the empty column between the unsafe action and the lowest safe
    one, and bracketed by two reference levels so it reads as the distance
    between *all* safe logits and *all* unsafe ones, not as one pair.
    """
    lowest_safe = min(logits[0], logits[2])
    highest_unsafe = logits[1]
    ax.axhline(
        lowest_safe,
        color=SAFE_COLOR,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        alpha=0.8,
        zorder=1,
    )
    x = 1.5
    ax.annotate(
        "",
        xy=(x, lowest_safe),
        xytext=(x, highest_unsafe),
        arrowprops={
            "arrowstyle": "<->",
            "color": INK_SECONDARY,
            "linewidth": 0.7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(
        x + 0.1,
        (lowest_safe + highest_unsafe) / 2,
        "margin",
        ha="left",
        va="center",
        fontsize=6,
        color=INK_SECONDARY,
    )


def _annotate_equal(ax, logits: tuple[float, ...]) -> None:
    """Dashed connector showing the two safe logits are equal."""
    level = logits[0]
    ax.plot(
        [0, 2],
        [level, level],
        color=SAFE_COLOR,
        linewidth=0.8,
        linestyle=(0, (2, 1.6)),
        alpha=0.9,
        zorder=1,
    )
    ax.text(
        1.0,
        level + 0.28,
        "equal",
        ha="center",
        va="bottom",
        fontsize=6,
        color=SAFE_COLOR,
    )


def build_figure():
    """Return the three-panel figure at exactly the ICLR text width."""
    apply_paper_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TEXT_WIDTH_IN, 1.9),
        sharey=True,
        layout="constrained",
    )

    for ax, panel in zip(axes, PANELS):
        _draw_panel(ax, panel)

    axes[0].set_ylabel("logit $z_a$")
    axes[0].text(
        1.62,
        4.35,
        PANELS[0]["note"],
        ha="center",
        va="top",
        fontsize=6,
        color=INK_SECONDARY,
        linespacing=1.15,
    )

    _annotate_unsafe_level(axes[1], PANELS[1]["logits"])
    _annotate_margin(axes[1], PANELS[1]["logits"])
    _annotate_unsafe_level(axes[2], PANELS[2]["logits"])
    _annotate_equal(axes[2], PANELS[2]["logits"])

    handles = [
        Line2D(
            [],
            [],
            marker=SAFE_MARKER,
            color=SAFE_COLOR,
            markeredgecolor=INK_PRIMARY,
            markeredgewidth=0.6,
            markersize=5.5,
            linestyle="none",
            label="shield-safe",
        ),
        Line2D(
            [],
            [],
            marker=UNSAFE_MARKER,
            color=UNSAFE_COLOR,
            markeredgecolor=INK_PRIMARY,
            markeredgewidth=0.6,
            markersize=5.5,
            linestyle="none",
            label="shield-unsafe",
        ),
    ]
    # Figure-level row above the panels: every in-axes position collides with
    # either a mark or a stem, since the stems run the full height of each panel.
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.4,
        borderpad=0.0,
        fontsize=6.5,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    # No bbox_inches="tight" -- it re-crops the canvas so the PDF stops matching
    # figsize. See the _paper_style module docstring.
    save_paper_figure(fig, args.output_dir / OUTPUT_STEM)
    plt.close(fig)
    print(f"wrote {args.output_dir / OUTPUT_STEM}.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
