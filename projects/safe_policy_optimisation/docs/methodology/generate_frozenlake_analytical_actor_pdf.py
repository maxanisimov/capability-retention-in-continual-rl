#!/usr/bin/env python3
"""Generate a three-page code-based note; do not change experiment sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-actor-pdf-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]
NAVY = "#17324F"
BLUE = "#286891"
GREEN = "#287454"
INK = "#243443"
MUTED = "#596C7C"
LINE = "#CAD6DF"


def paragraph(fig, x, y, value, *, width=105, size=10, color=INK):
    item = fig.text(
        x,
        y,
        textwrap.fill(value, width=width),
        va="top",
        fontsize=size,
        color=color,
        linespacing=1.35,
    )
    right = 0.465 if x == 0.09 and width <= 50 else 0.93
    renderer = fig.canvas.get_renderer()
    while (
        item.get_window_extent(renderer).transformed(fig.transFigure.inverted()).x1
        > right
    ):
        width -= 1
        if width < 10:
            raise AssertionError("Cannot fit paragraph")
        item.set_text(textwrap.fill(value, width=width))
    return item


def page(number, title, subtitle):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.973), 1, 0.027, color=NAVY, lw=0))
    fig.text(0.07, 0.939, title, fontsize=22, color=NAVY, weight="bold", va="top")
    fig.text(0.07, 0.900, subtitle, fontsize=10, color=MUTED, va="top")
    fig.text(
        0.07,
        0.035,
        "FrozenLake PSPO | Implementation note | 8 October 2026",
        fontsize=8,
        color=MUTED,
    )
    fig.text(0.93, 0.035, str(number), ha="right", fontsize=8, color=MUTED)
    return fig, ax


def box(ax, x, y, w, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor=color,
            edgecolor=LINE,
            linewidth=0.8,
        )
    )


def heading(fig, y, value):
    fig.text(0.07, y, value, fontsize=13, color=NAVY, weight="bold", va="top")


def save(pdf, fig, number, preview):
    # Validate visible text extents, catching accidental clipping at page edges.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        if (
            bounds.x0 < 0.015
            or bounds.x1 > 0.985
            or bounds.y0 < 0.015
            or bounds.y1 > 0.97
        ):
            raise AssertionError(
                f"Text outside page {number}: {item.get_text()!r}: {bounds}"
            )
    pdf.savefig(fig)
    if preview is not None:
        fig.savefig(preview / f"page_{number}.png", dpi=130)
    plt.close(fig)


def first_page(pdf, preview):
    fig, ax = page(
        1,
        "Fitted vs analytical initialisation",
        "What changes in FrozenLake PSPO - and what does not",
    )
    box(ax, 0.07, 0.771, 0.86, 0.093, "#FFF6E6")
    fig.text(
        0.09,
        0.847,
        "Historical clarification",
        color="#96651B",
        fontsize=11,
        weight="bold",
        va="top",
    )
    paragraph(
        fig,
        0.09,
        0.821,
        "The generic PSPO pipeline fits multilayer safe actors. But the older near-perfect "
        "FrozenLake runs already used an analytical actor, with a goal-directed witness "
        "preference. The current actor removes that preference. Dense one-hot evaluation "
        "is a separate issue, not an alternative construction algorithm.",
        width=109,
        size=9.1,
    )

    box(ax, 0.07, 0.519, 0.41, 0.216, "#EEF4F9")
    box(ax, 0.52, 0.519, 0.41, 0.216, "#EFF7F1")
    fig.text(
        0.09,
        0.715,
        "Fitted / iterative method",
        fontsize=13,
        color=BLUE,
        weight="bold",
        va="top",
    )
    paragraph(
        fig,
        0.09,
        0.682,
        "Safety mask + state inputs -> minibatch Adam -> exhaustive safety check. "
        "Learn weights that assign high probability to the safe-action set; optional "
        "margin and entropy terms depend on the experiment. A finite fit may fail "
        "the requested criterion.",
        width=47,
        size=10,
    )
    fig.text(
        0.54,
        0.715,
        "Current closed form",
        fontsize=13,
        color=GREEN,
        weight="bold",
        va="top",
    )
    paragraph(
        fig,
        0.54,
        0.682,
        "Safety mask -> prescribed logits -> inverse-tanh weights -> exact safety check. "
        "Every safe action receives +2; every unsafe action receives -2. No fitting "
        "epochs, reward targets or goal-directed preferences are needed.",
        width=46,
        size=10,
    )

    heading(fig, 0.479, "The fitted objective is a safety objective, not a task reward")
    fig.text(
        0.50,
        0.426,
        r"$L_{\rm safe}=\mathbb{E}_{s\in W}\left[\log\sum_a e^{z_a(s)}"
        r"-\log\sum_{a\in A_{\rm safe}(s)}e^{z_a(s)}\right]$",
        fontsize=14,
        color=INK,
        ha="center",
    )
    paragraph(
        fig,
        0.07,
        0.395,
        "The generic fitter adds optional margin/entropy losses, or uses a safe-mass "
        "objective. Both fitted and analytical initialisation can be reward-independent: "
        "the distinction is solving for weights by optimization versus setting them directly.",
        width=108,
        size=10,
    )

    heading(fig, 0.303, "Practical comparison")
    comparisons = [
        ("Inputs", "Safety-labelled states", "Safety action mask only"),
        (
            "Weights",
            "Found iteratively; criterion checked",
            "Closed-form target logits; checked",
        ),
        (
            "Safe-action ranking",
            "Depends on fit and entropy objective",
            "All safe actions initially tied",
        ),
        (
            "Applicability",
            "General architectures / state features",
            "Specific tabular tanh architecture",
        ),
        (
            "Reward learning",
            "Still performed later by PPO",
            "Still performed later by PPO",
        ),
    ]
    for i, (label, left, right) in enumerate(comparisons):
        y = 0.266 - i * 0.034
        ax.add_patch(
            plt.Rectangle(
                (0.07, y - 0.025),
                0.86,
                0.034,
                facecolor="#F4F7FA" if i % 2 == 0 else "white",
                lw=0,
            )
        )
        fig.text(0.08, y, label, fontsize=8.8, color=INK, va="top", weight="bold")
        fig.text(0.27, y, left, fontsize=8.6, color=INK, va="top")
        fig.text(0.61, y, right, fontsize=8.6, color=INK, va="top")
    paragraph(
        fig,
        0.07,
        0.075,
        "Sources: compute_shield_rashomon_set.py:96,459,841; "
        "run_stochastic_frozenlake_pspo.py:97; stochastic_frozenlake128.md historical note.",
        width=118,
        size=7.7,
        color=MUTED,
    )
    save(pdf, fig, 1, preview)


def second_page(pdf, preview, checks):
    fig, ax = page(
        2,
        "How the analytical weights work",
        "One-hot input selects a weight column; inverse tanh fixes the logits",
    )
    heading(fig, 0.849, "1. Prescribe the output, without using rewards")
    fig.text(
        0.50,
        0.799,
        r"$z^*_{s,a}=+2\ \mathrm{if}\ a\in A_{\rm safe}(s),"
        r"\qquad z^*_{s,a}=-2\ \mathrm{otherwise}$",
        fontsize=14,
        ha="center",
        color=INK,
    )
    paragraph(
        fig,
        0.07,
        0.768,
        "Actor architecture: N -> 64 -> 64 -> 4, with tanh after each hidden layer. "
        "All biases are zero; W2 is the 64-dimensional identity; W3 = [4 I4, 0]. "
        "The first four hidden units encode the four action logits.",
        width=107,
    )
    heading(fig, 0.681, "2. Invert the two nonlinearities")
    fig.text(
        0.50,
        0.632,
        r"$(W_1)_{a,s}=\operatorname{atanh}\left("
        r"\operatorname{atanh}(z^*_{s,a}/4)\right)$",
        fontsize=18,
        ha="center",
        color=GREEN,
    )
    fig.text(
        0.50,
        0.580,
        r"$z_a(e_s)=4\tanh\left(\tanh((W_1)_{a,s})\right)=z^*_{s,a}$",
        fontsize=16,
        ha="center",
        color=INK,
    )
    paragraph(
        fig,
        0.07,
        0.543,
        "For target +2 the encoded weight is about +0.617387; for -2 it is the negative. "
        "The inverse is real for |z*| < 4 tanh(1), about 3.046. The remaining 60 first-layer "
        "rows have small deterministic random weights and zero initial output weights; "
        "they do not affect initial logits but provide capacity for later PPO updates.",
        width=108,
    )

    heading(fig, 0.432, "3. Audit without building every one-hot input")
    box(ax, 0.07, 0.291, 0.86, 0.106, "#F4F7FA")
    fig.text(
        0.09,
        0.380,
        "# Illustrative dense reference: same already-built actor\n"
        "dense = network(torch.eye(N))\n\n"
        "# Current column-based evaluation: identical first-layer responses\n"
        "h0 = network[0].weight.T + network[0].bias\n"
        "columns = network[4](network[3](network[2](network[1](h0))))",
        fontsize=8.5,
        fontfamily="monospace",
        color=INK,
        va="top",
        linespacing=1.2,
    )
    paragraph(
        fig,
        0.07,
        0.266,
        "This changes the audit's computation, not the actor. Initial parameter storage "
        "and the column audit scale with N x 64, rather than requiring N x N input storage. "
        "Batched dense evaluation can also avoid the full identity, but still uses dense batches.",
        width=108,
        size=9.6,
    )

    box(ax, 0.07, 0.123, 0.86, 0.066, "#EFF7F1")
    paragraph(
        fig,
        0.09,
        0.175,
        f"Numerical check on the real 16x16 actor: dense versus column audit maximum "
        f"absolute difference {checks['dense_vs_columns_max_error']:.2g}; target-logit error "
        f"{checks['target_logit_max_error']:.2g}; all {checks['winning_states']} winning "
        "states have safe greedy actions. No large-layout identity was allocated.",
        width=111,
        size=9.1,
    )
    paragraph(
        fig,
        0.07,
        0.098,
        "Guarantee scope: greedy admissibility, not zero unsafe probability under softmax "
        "sampling. Finite -2 unsafe logits retain nonzero probability; the training shield "
        "handles sampled actions. Initialization alone does not promise goal success.",
        width=117,
        size=8.2,
        color=MUTED,
    )
    save(pdf, fig, 2, preview)


def third_page(pdf, preview):
    fig, ax = page(
        3,
        "Memory savings - and remaining costs",
        "Exact tensor sizes; not measured process peaks or runtime speedups",
    )
    heading(
        fig,
        0.848,
        "The actor audit and the training certificate are different allocations",
    )
    paragraph(
        fig,
        0.07,
        0.812,
        "N is the state count, H=64 the hidden width, and |W| the winning-state count. "
        "The current map masks give |W| = 220, 880, 3,520, 14,080 and 56,320. "
        "Sizes below use float32: 4 bytes/value, 1 GiB = 2^30 bytes.",
        width=108,
    )

    labels = [
        "Layout",
        "States N",
        "Full identity\nGiB",
        "First-layer weights\nMiB",
        "Certificate states\nGiB",
    ]
    sides = (16, 32, 64, 128, 256)
    winning = (220, 880, 3520, 14080, 56320)
    rows = []
    for side, count in zip(sides, winning, strict=True):
        n = side**2
        rows.append(
            [
                f"{side} x {side}",
                f"{n:,}",
                f"{4 * n * n / 2**30:.6g}",
                f"{4 * n * 64 / 2**20:g}",
                f"{4 * count * n / 2**30:.6g}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=labels,
        cellLoc="center",
        colWidths=[0.13, 0.13, 0.19, 0.22, 0.23],
        bbox=(0.07, 0.555, 0.86, 0.19),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold", fontsize=8.2)
        else:
            cell.set_facecolor("#F4F7FA" if row % 2 else "white")
    fig.text(
        0.50,
        0.522,
        r"$\mathrm{identity}:4N^2\quad\mathrm{first\ layer}:4NH"
        r"\quad\mathrm{certificate}:4|W|N\quad\mathrm{bytes}$",
        fontsize=12,
        ha="center",
        color=INK,
    )
    paragraph(
        fig,
        0.07,
        0.492,
        "The first-layer column is only one parameter tensor, not total model memory. "
        "The analytic audit avoids the illustrative 16 GiB identity at 256x256. It does "
        "not eliminate the 13.75 GiB one-hot certificate used by the current RL implementation.",
        width=108,
        size=9.6,
    )

    heading(fig, 0.399, "Why initialization can still be memory-heavy")
    paragraph(
        fig,
        0.07,
        0.364,
        "The certificate builder first creates int64 one-hot states (27.5 GiB at 256x256) "
        "and casts them to float32 (13.75 GiB). Those buffers can overlap during conversion; "
        "verification adds further temporary allocations. Multiple workers multiply the "
        "cost. Fewer training timesteps reduce runtime, not this state-count-dependent peak.",
        width=108,
        size=9.7,
    )

    heading(
        fig, 0.259, "Separate optimizations - do not attribute them to the closed form"
    )
    paragraph(
        fig,
        0.07,
        0.224,
        "Lookup mode stores 56,320 int64 certificate state IDs in about 0.430 MiB instead "
        "of a dense one-hot matrix, while retaining exact first-layer semantics. The current "
        "scalability sweeps use one-hot mode. Independently, the sparse environment avoids "
        "a 128 GiB float64 transition tensor at 256x256. Neither is the inverse-tanh construction.",
        width=108,
        size=9.5,
    )
    paragraph(
        fig,
        0.07,
        0.117,
        "Evidence: current initialization and certificate builders; archived FrozenLake source "
        "snapshots; prior exact reconstruction of all 50 verify-first initial actors. No matched "
        "fitted-versus-analytical speed benchmark was run. Companion Markdown contains full "
        "repository references and paper-ready wording.",
        width=115,
        size=8.3,
        color=MUTED,
    )
    save(pdf, fig, 3, preview)


def numerical_check():
    import numpy as np
    import torch

    from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_pspo import (
        build_safe_actor,
    )
    from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
        SparseFrozenLake,
        synthesise_shield,
    )

    torch.set_num_threads(1)
    env = SparseFrozenLake(size=16)
    try:
        mask, winning, _, _ = synthesise_shield(env)
    finally:
        env.close()
    payload, _ = build_safe_actor(mask)
    network = torch.nn.Sequential(
        torch.nn.Linear(len(mask), 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 4),
    )
    network.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        dense = network(torch.eye(len(mask)))
        h0 = network[0].weight.T + network[0].bias
        columns = network[4](network[3](network[2](network[1](h0))))
        targets = torch.tensor(np.where(mask, 2.0, -2.0), dtype=torch.float32)
        torch.testing.assert_close(dense, columns, rtol=0, atol=1e-6)
        torch.testing.assert_close(columns, targets, rtol=0, atol=1e-5)
        ids = np.flatnonzero(winning)
        assert np.all(mask[ids, columns.argmax(dim=1).numpy()[ids]])
    return {
        "layout": "16x16",
        "winning_states": int(winning.sum()),
        "dense_vs_columns_max_error": float((dense - columns).abs().max()),
        "target_logit_max_error": float((columns - targets).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "frozenlake_analytical_actor_construction.pdf",
    )
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    checks = numerical_check()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    with PdfPages(
        args.output,
        metadata={
            "Title": "FrozenLake PSPO: fitted versus analytical actor initialisation",
            "Subject": "Code-based construction, dense-audit equivalence, and memory scaling",
            "Author": "CertifiedContinualLearning",
        },
    ) as pdf:
        first_page(pdf, args.preview_dir)
        second_page(pdf, args.preview_dir, checks)
        third_page(pdf, args.preview_dir)
    print(
        json.dumps(
            {
                "pdf": str(args.output.resolve()),
                "pages": 3,
                "bytes": args.output.stat().st_size,
                "numerical_check": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
