"""Functions for setting a consistent plotting style for all figures in the paper."""

import matplotlib.pyplot as plt
from matplotlib.image import imread
from tempfile import NamedTemporaryFile
import scienceplots

plt.style.use(["science", "no-latex", "scatter"])

# set plotting context
tex_fonts = {
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 8,
}

plt.rcParams.update(tex_fonts)


def set_figure_size(
    fig, fraction=1.0, subplots=(1, 1), portrait=False, shrink_height=1.0, dpi=300
):
    """
    Useful function for setting figure dimensions to avoid scaling in LaTeX from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/. The function allows you to
    save a figure to a PDF with the correct dimensions, so that you can include
    it in the latex document without scaling it, keeping the font size consistent between
    figures.

    Parameters
    ----------
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = (
        505.89  # document width in pt, as revealed by \showthe\textwidth in latex
    )

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if portrait is False:
        fig_height_in = (
            shrink_height * fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        )
    else:
        fig_height_in = (
            shrink_height * fig_width_in / golden_ratio * (subplots[0] / subplots[1])
        )

    apply_figure_size(fig, (fig_width_in, fig_height_in), dpi=dpi)


def apply_figure_size(fig, size, dpi=300, eps=1e-2, give_up=2, min_size_px=10):
    """Code from https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
    which fixes the super annoying matplotlib figure size handling."""
    target_width, target_height = size
    set_width, set_height = target_width, target_height  # reasonable starting point
    deltas = []  # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        with NamedTemporaryFile(suffix=".png") as f:
            fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
            actual_height, actual_width, _ = imread(f.name).shape
            actual_height, actual_width = actual_height / dpi, actual_width / dpi
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(
            abs(actual_width - target_width) + abs(actual_height - target_height)
        )
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False
