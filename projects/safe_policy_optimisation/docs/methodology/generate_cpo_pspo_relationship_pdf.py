#!/usr/bin/env python3
"""Generate a concise two-page note comparing CPO and PSPO."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "cpo_pspo_relationship.pdf"

NAVY = "#17365D"
BLUE = "#2F6B9A"
PALE_BLUE = "#EAF2F8"
GREEN = "#26734D"
PALE_GREEN = "#EAF6EF"
GOLD = "#B7791F"
PALE_GOLD = "#FFF7E6"
INK = "#1E293B"
MUTED = "#526274"
LINE = "#C8D2DC"
WHITE = "#FFFFFF"


def _base_page(page_number: int):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.974), 1, 0.026, color=NAVY, linewidth=0))
    fig.text(0.07, 0.035, "CPO and PSPO: optimisation relationship", fontsize=7.5, color=MUTED)
    fig.text(0.93, 0.035, str(page_number), fontsize=7.5, color=MUTED, ha="right")
    return fig, ax


def _box(ax, x, y, w, h, *, face, edge=LINE, radius=0.012, linewidth=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def _wrapped(fig, x, y, text, *, width, size=9.2, color=INK, weight="normal",
             linespacing=1.25, va="top", ha="left"):
    fig.text(
        x, y, textwrap.fill(text, width=width), fontsize=size, color=color,
        fontweight=weight, va=va, ha=ha, linespacing=linespacing,
    )


def _title(fig, subtitle: str):
    fig.text(0.07, 0.935, "How CPO and PSPO are related", fontsize=23, color=NAVY,
             fontweight="bold", va="top")
    fig.text(0.07, 0.895, subtitle, fontsize=10.5, color=MUTED, va="top")


def _page_one(pdf: PdfPages) -> None:
    fig, ax = _base_page(1)
    _title(fig, "Same constrained-RL template; different safety set, evidence, and update rule")

    _box(ax, 0.07, 0.779, 0.86, 0.082, face=PALE_GOLD, edge="#E6C36A")
    fig.text(0.095, 0.839, "Bottom line", fontsize=11.5, color=GOLD, fontweight="bold", va="top")
    _wrapped(
        fig, 0.095, 0.814,
        "PSPO is not literally a special case of the CPO algorithm. It is better viewed as a sibling "
        "constrained policy-optimisation method: CPO enforces a local expected-cost constraint, while "
        "PSPO projects PPO proposals into a formally verified, shield-defined parameter region.",
        width=111, size=9.4,
    )

    fig.text(0.07, 0.746, "A common abstraction", fontsize=14, color=NAVY, fontweight="bold")
    fig.text(
        0.5, 0.704,
        r"$\underset{\theta}{\mathrm{maximise}}\;J_R(\theta)"
        r"\qquad\mathrm{subject\ to}\qquad\theta\in\mathcal{C}_{\mathrm{safe}}$",
        fontsize=16, color=INK, ha="center",
    )
    _wrapped(
        fig, 0.10, 0.666,
        "The important difference is what C_safe means and how an update is kept inside it.",
        width=95, size=9.5, color=MUTED, ha="left",
    )

    # Two method cards.
    _box(ax, 0.07, 0.348, 0.405, 0.280, face=PALE_BLUE, edge="#9EBED5")
    _box(ax, 0.525, 0.348, 0.405, 0.280, face=PALE_GREEN, edge="#9BC9AE")
    fig.text(0.095, 0.598, "CPO", fontsize=17, color=BLUE, fontweight="bold")
    fig.text(0.095, 0.572, "Expected-cost constrained trust-region step", fontsize=9.2,
             color=MUTED)
    fig.text(0.095, 0.527, r"$\max_{\Delta\theta}\;g^\top\Delta\theta$", fontsize=14, color=INK)
    fig.text(0.115, 0.492, r"$c+b^\top\Delta\theta\leq 0$", fontsize=12.5, color=INK)
    fig.text(0.115, 0.459, r"$\frac{1}{2}\Delta\theta^\top F\Delta\theta\leq\delta$",
             fontsize=12.5, color=INK)
    _wrapped(
        fig, 0.095, 0.420,
        "Here c is current expected cost minus the budget; b is the cost-surrogate gradient; "
        "and F is the KL/Fisher curvature. A dual solution, conjugate gradients, and line search "
        "produce the next policy (or a cost-recovery step).",
        width=51, size=8.6,
    )

    fig.text(0.55, 0.598, "PSPO", fontsize=17, color=GREEN, fontweight="bold")
    fig.text(0.55, 0.572, "Verified-set constrained projected PPO", fontsize=9.2, color=MUTED)
    fig.text(
        0.55, 0.525,
        r"$\mathcal{C}_{H}=\{\theta:\ \forall s,\ "
        r"\arg\max_a z_{\theta,a}(s)\in A_H(s)\}$",
        fontsize=11.4, color=INK,
    )
    fig.text(0.55, 0.481, r"$\Theta_k^{\mathrm{LID}}\subseteq\mathcal{C}_H$",
             fontsize=13, color=INK)
    fig.text(
        0.55, 0.444,
        r"$\theta_{k+1}=\operatorname{Proj}_{\Theta_k^{\mathrm{LID}}}(\bar\theta_{k+1})$",
        fontsize=12.2, color=INK,
    )
    _wrapped(
        fig, 0.55, 0.410,
        "The shield supplies admissible actions A_H(s). IBP certifies a Local Invariant Domain "
        "(LID) in actor-parameter space. PPO supplies the reward-improving proposal; projection "
        "or fail-closed reversion supplies safety.",
        width=49, size=8.6,
    )

    fig.text(0.07, 0.303, "The relationship in one line", fontsize=14, color=NAVY,
             fontweight="bold")
    _box(ax, 0.105, 0.198, 0.79, 0.075, face="#F7F9FB", edge=LINE)
    fig.text(0.5, 0.246, "unconstrained reward update", fontsize=9, color=MUTED, ha="center")
    fig.text(0.20, 0.215, "candidate policy", fontsize=10, color=INK, ha="center")
    fig.text(0.50, 0.215, r"$\longrightarrow\quad$ safety-feasible update $\quad\longrightarrow$",
             fontsize=12, color=NAVY, ha="center")
    fig.text(0.80, 0.215, "next policy", fontsize=10, color=INK, ha="center")
    fig.text(0.50, 0.163,
             "CPO changes the step direction using cost and KL; PSPO changes the feasible parameter set using the shield and LID.",
             fontsize=8.7, color=MUTED, ha="center")

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _page_two(pdf: PdfPages) -> None:
    fig, ax = _base_page(2)
    fig.text(0.07, 0.935, "Why the distinction matters", fontsize=23, color=NAVY,
             fontweight="bold", va="top")
    fig.text(0.07, 0.895, "Optimisation mechanics and guarantee scope", fontsize=10.5,
             color=MUTED, va="top")

    fig.text(0.07, 0.848, "One update", fontsize=14, color=NAVY, fontweight="bold")

    # CPO flow.
    fig.text(0.08, 0.806, "CPO", fontsize=11, color=BLUE, fontweight="bold")
    cpo_flow = ["rollout", "estimate reward\n& cost advantages", "solve local\nQCQP", "line search /\nrecovery", "next policy"]
    x_positions = [0.10, 0.285, 0.49, 0.68, 0.87]
    for i, (x, label) in enumerate(zip(x_positions, cpo_flow, strict=True)):
        _box(ax, x - 0.07, 0.735, 0.14, 0.048, face=PALE_BLUE, edge="#9EBED5", radius=0.008)
        fig.text(x, 0.759, label, fontsize=7.8, color=INK, ha="center", va="center")
        if i < len(x_positions) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.078, 0.759), xytext=(x + 0.078, 0.759),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    # PSPO flow.
    fig.text(0.08, 0.691, "PSPO", fontsize=11, color=GREEN, fontweight="bold")
    pspo_flow = ["shielded\nrollout", "PPO reward\nproposal", "grow & verify\ndirectional LID", "project or\nrevert", "certified actor"]
    for i, (x, label) in enumerate(zip(x_positions, pspo_flow, strict=True)):
        _box(ax, x - 0.07, 0.620, 0.14, 0.048, face=PALE_GREEN, edge="#9BC9AE", radius=0.008)
        fig.text(x, 0.644, label, fontsize=7.8, color=INK, ha="center", va="center")
        if i < len(x_positions) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.078, 0.644), xytext=(x + 0.078, 0.644),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    fig.text(0.07, 0.575, "Side-by-side", fontsize=14, color=NAVY, fontweight="bold")
    left, mid, right = 0.07, 0.33, 0.63
    top = 0.535
    row_h = 0.052
    rows = [
        ("Safety quantity", "Expected cumulative cost", "Greedy shield admissibility at every certified state"),
        ("Evidence", "Rollout estimates and learned cost critic", "Exhaustive shield data and sound interval bounds"),
        ("Local geometry", "KL/Fisher ellipsoid + linearised cost half-space", "Certified parameter orthotope (LID)"),
        ("Reward optimiser", "TRPO-family natural-gradient step", "PPO proposal followed by Euclidean projection"),
        ("Failure handling", "Backtracking or cost-recovery step", "Last certified LID or fail-closed reversion"),
        ("Guarantee target", "Approximate satisfaction of an expected-cost budget", "All-state action certificate at enforcement boundaries"),
        ("Certified deployment", "Usually the stochastic policy", "Greedy actor; stochastic sampling needs an added certificate"),
    ]
    ax.add_patch(plt.Rectangle((left, top), right - left + 0.30, 0.038, color=NAVY, linewidth=0))
    fig.text(left + 0.01, top + 0.019, "Dimension", fontsize=8.5, color=WHITE, fontweight="bold", va="center")
    fig.text(mid + 0.01, top + 0.019, "CPO", fontsize=8.5, color=WHITE, fontweight="bold", va="center")
    fig.text(right + 0.01, top + 0.019, "PSPO", fontsize=8.5, color=WHITE, fontweight="bold", va="center")
    for index, (dimension, cpo, pspo) in enumerate(rows):
        y = top - (index + 1) * row_h
        face = "#F8FAFC" if index % 2 == 0 else WHITE
        ax.add_patch(plt.Rectangle((left, y), 0.86, row_h, color=face, ec=LINE, lw=0.5))
        _wrapped(fig, left + 0.01, y + row_h - 0.012, dimension, width=24, size=7.6, weight="bold")
        _wrapped(fig, mid + 0.01, y + row_h - 0.012, cpo, width=36, size=7.4)
        _wrapped(fig, right + 0.01, y + row_h - 0.012, pspo, width=38, size=7.4)

    verdict_y = 0.066
    _box(ax, 0.07, verdict_y, 0.86, 0.090, face=PALE_GOLD, edge="#E6C36A")
    fig.text(0.095, verdict_y + 0.072, "Best shorthand", fontsize=10.5, color=GOLD,
             fontweight="bold", va="top")
    _wrapped(
        fig, 0.095, verdict_y + 0.049,
        "PSPO is projected PPO over a shield-derived, verified inner approximation of the safe-policy set. "
        "It resembles CPO at the level of constrained optimisation, but it is not obtained by choosing a "
        "special CPO cost function or trust-region radius.",
        width=110, size=8.8,
    )

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _page_three(pdf: PdfPages) -> None:
    fig, ax = _base_page(3)
    fig.text(0.07, 0.935, "When the ‘special case’ intuition is useful", fontsize=21,
             color=NAVY, fontweight="bold", va="top")
    fig.text(0.07, 0.895, "A useful conceptual bridge—and its limits", fontsize=10.5,
             color=MUTED, va="top")

    _box(ax, 0.07, 0.700, 0.86, 0.145, face=PALE_BLUE, edge="#9EBED5")
    fig.text(0.095, 0.817, "Encode shield violations as cost", fontsize=12.5, color=BLUE,
             fontweight="bold")
    fig.text(0.5, 0.766,
             r"$c_H(s,a)=\mathbf{1}\{a\notin A_H(s)\},\qquad J_C(\theta)=\mathbb{E}_{\pi_\theta}\!\left[\sum_t c_H(s_t,a_t)\right]$",
             fontsize=13, color=INK, ha="center")
    _wrapped(
        fig, 0.095, 0.730,
        "With non-negative costs and budget d=0, exact satisfaction forces zero shield violations on "
        "state-action pairs that receive positive occupancy. This makes the high-level feasible-policy "
        "problem look similar to PSPO.",
        width=110, size=8.9,
    )

    fig.text(0.07, 0.653, "Why that still does not make PSPO CPO", fontsize=14, color=NAVY,
             fontweight="bold")
    reasons = [
        ("Occupancy versus all states", "CPO constrains an expectation under the rollout distribution. PSPO certifies every enumerated state in its declared domain, including states absent from a finite rollout."),
        ("Approximation versus verification", "CPO linearises cost and KL and estimates them from data. PSPO accepts an LID only after a hard, sound interval certificate."),
        ("Different step", "CPO jointly chooses a reward/cost-aware trust-region direction. PSPO first computes a PPO reward proposal, then constructs a safe region and projects the proposal."),
        ("Different safety semantics", "PSPO certifies that the greedy action is shield-admissible. Trajectory safety additionally inherits the shield's soundness; stochastic deployment requires a probability-level extension."),
    ]
    y = 0.605
    for number, (heading, body) in enumerate(reasons, start=1):
        ax.add_patch(plt.Circle((0.095, y + 0.003), 0.014, color=NAVY))
        fig.text(0.095, y + 0.003, str(number), color=WHITE, fontsize=8, ha="center", va="center",
                 fontweight="bold")
        fig.text(0.122, y + 0.014, heading, fontsize=9.5, color=INK, fontweight="bold", va="top")
        _wrapped(fig, 0.122, y - 0.009, body, width=101, size=8.4, color=MUTED)
        y -= 0.105

    _box(ax, 0.07, 0.107, 0.86, 0.082, face=PALE_GREEN, edge="#9BC9AE")
    fig.text(0.095, 0.166, "A unified research direction", fontsize=11, color=GREEN,
             fontweight="bold", va="top")
    _wrapped(
        fig, 0.095, 0.141,
        "Use CPO-style reward/cost geometry to propose efficient directions, but accept each update only "
        "inside a PSPO-style verified LID. This would combine expected-cost guidance with a hard statewise "
        "certificate; it is a hybrid, not a specialisation.",
        width=110, size=8.8,
    )

    _wrapped(
        fig, 0.07, 0.091,
        "Sources: Achiam et al., Constrained Policy Optimization, ICML 2017 (arXiv:1705.10528); "
        "repository PSPO methodology (pspo_main.tex) and CPO implementation (core/safe_rl_baselines/cpo.py).",
        width=150, size=6.6, color=MUTED,
    )

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })
    metadata = {
        "Title": "How CPO and PSPO are related",
        "Subject": "Comparison of constrained policy optimisation and verified-set projected policy optimisation",
        "Author": "CertifiedContinualLearning project",
    }
    with PdfPages(args.output, metadata=metadata) as pdf:
        _page_one(pdf)
        _page_two(pdf)
        _page_three(pdf)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
