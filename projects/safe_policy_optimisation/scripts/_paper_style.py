"""Shared figure geometry, palette and typography for the paper figures.

Currently targeting **ICLR**, which unlike ICML is a *single-column* layout:

    \\textwidth   5.5 in      (ICLR conference style)
    \\textheight  9.0 in

So there is no ``figure*`` -- every figure is a plain ``figure`` at 5.5 in
wide. That is 1.25 in narrower than the ICML two-column full width the figures
were previously built for, which matters when six environment panels have to
share the row.

Every figure is rendered at its *final* printed size so it can be included with
``\\includegraphics`` at native scale -- no ``width=`` rescaling, which would
shrink the fonts away from the values set here.

Two consequences worth remembering when editing the plot scripts:

* Never pass ``bbox_inches="tight"`` to ``savefig``. It re-crops the canvas, so
  the saved PDF stops matching ``figsize`` (the old 12.0 in figures landed at
  11.9 in). Use ``layout="constrained"`` instead and let matplotlib pack the
  content inside a fixed canvas.
* Font sizes here are absolute points on the printed page. They are small
  because the figures are dense; do not scale them up without also giving the
  figure more room.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ICLR single-column geometry.
TEXT_WIDTH_IN = 5.5
TEXT_HEIGHT_IN = 9.0

# Method -> colour. One mapping for curves and bars alike, so a method reads
# the same in every figure. Green is reserved for PSPO, which is additionally
# the only method drawn with a solid line.
METHOD_COLORS = {
    "ppo_policy": "black",
    "ppo_lagrangian": "red",
    "ppo_pid_lagrangian": "orange",
    "cpo": "blue",
    "pspo": "green",
    "ppo_shield": "purple",
    # Same trained weights as ppo_shield with the runtime shield removed, so it
    # gets the light partner of that colour rather than an unrelated hue.
    "ppo_shield_nominal": "plum",
}

# Dash patterns for the curve figures. PSPO is solid; everything else is
# broken, so the highlighted method stays readable in greyscale too.
METHOD_LINESTYLES = {
    "ppo_policy": (0, (4, 1.5)),
    "ppo_lagrangian": (0, (1, 1.2)),
    "ppo_pid_lagrangian": (0, (5, 1.5, 1, 1.5)),
    "cpo": (0, (3, 1, 1, 1)),
    "pspo": "solid",
    "ppo_shield": (0, (6, 2)),
    # Shorter dashes than ppo_shield: the pair overlays in the shield-removal
    # curves, where colour alone (purple vs plum) is thin at 1.0 pt.
    "ppo_shield_nominal": (0, (2, 1.2)),
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_paper_style() -> None:
    """Set the rcParams used by every paper figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6.5,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "lines.linewidth": 1.0,
            # Embed real fonts rather than outlines, for camera-ready checks.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_paper_figure(fig, stem, *, dpi: int = 400) -> None:
    """Write PDF + PNG at exactly the figure's declared size.

    ``bbox_inches`` is deliberately not passed -- see the module docstring.
    """
    fig.savefig(stem.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor=fig.get_facecolor())
