"""Render the adjacent implementation guide using only Matplotlib.

The Markdown is the editable source. Its explicit page breaks and one diagram
marker give a small, reproducible PDF without requiring a TeX installation.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 9

PAGE_WIDTH = 595.276
PAGE_HEIGHT = 841.890
MARGIN = 48.0
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN
INK = "#182735"
ACCENT = "#176873"


@lru_cache(maxsize=8192)
def text_width(text: str, size: float, family: str, weight: str) -> float:
    font = FontProperties(family=family, size=size, weight=weight)
    return TextToPath().get_text_width_height_descent(text, font, ismath=False)[0]


def wrap(text: str, size: float, family: str, weight: str) -> list[str]:
    """Wrap by rendered font width, including unusually long source paths."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if text_width(candidate, size, family, weight) <= TEXT_WIDTH:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while text_width(word, size, family, weight) > TEXT_WIDTH:
            cut = len(word) - 1
            while text_width(word[:cut], size, family, weight) > TEXT_WIDTH:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


class Page:
    def __init__(self, number: int, total: int):
        self.fig = plt.figure(figsize=(PAGE_WIDTH / 72, PAGE_HEIGHT / 72))
        self.y = PAGE_HEIGHT - MARGIN
        self.number = number
        self.fig.text(
            MARGIN / PAGE_WIDTH,
            27 / PAGE_HEIGHT,
            "PSPO implementation guide • 7 October 2026",
            fontsize=8,
            color="#617482",
        )
        self.fig.text(
            (PAGE_WIDTH - MARGIN) / PAGE_WIDTH,
            27 / PAGE_HEIGHT,
            f"{number} / {total}",
            ha="right",
            fontsize=8,
            color="#617482",
        )

    def paragraph(
        self,
        text: str,
        *,
        size: float = 10.3,
        weight: str = "normal",
        color: str = INK,
        after: float = 8,
    ) -> None:
        for line in wrap(text, size, "DejaVu Sans", weight):
            self.fig.text(
                MARGIN / PAGE_WIDTH,
                self.y / PAGE_HEIGHT,
                line,
                va="top",
                fontsize=size,
                fontfamily="DejaVu Sans",
                fontweight=weight,
                color=color,
            )
            self.y -= size * 1.40
        self.y -= after

    def heading(self, text: str, *, title: bool = False) -> None:
        if title:
            self.paragraph(
                text, size=20 if self.number == 1 else 18, weight="bold", after=12
            )
        else:
            self.y -= 3
            self.paragraph(text, size=12.0, weight="bold", color=ACCENT, after=6)

    def equation(self, source: str) -> None:
        text = "$" + source.replace(r"\hbox", r"\mathrm") + "$"
        size = 12.0
        font = FontProperties(size=size)
        width = TextToPath().get_text_width_height_descent(text, font, ismath=True)[0]
        if width > TEXT_WIDTH:
            size *= TEXT_WIDTH / width
        if size < 9.5:
            raise ValueError(f"Equation is too wide: {source}")
        self.fig.text(
            0.5,
            self.y / PAGE_HEIGHT,
            text,
            va="top",
            ha="center",
            fontsize=size,
            color=INK,
        )
        self.y -= size * 2.5 + 5

    def code(self, source: list[str]) -> None:
        size = 8.4
        line_height = 11.5
        self.y -= 2
        for line in source:
            if text_width(line, size, "DejaVu Sans Mono", "normal") > TEXT_WIDTH:
                raise ValueError(f"Code line is too wide: {line}")
            self.fig.text(
                MARGIN / PAGE_WIDTH,
                self.y / PAGE_HEIGHT,
                line,
                va="top",
                fontsize=size,
                fontfamily="DejaVu Sans Mono",
                color=INK,
            )
            self.y -= line_height
        self.y -= 10

    def diagram(self) -> None:
        height = 87
        ax = self.fig.add_axes(
            [
                MARGIN / PAGE_WIDTH,
                (self.y - height) / PAGE_HEIGHT,
                TEXT_WIDTH / PAGE_WIDTH,
                height / PAGE_HEIGHT,
            ]
        )
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.5, 0.8)
        ax.axis("off")
        alpha = 0.62
        ax.plot([0, alpha], [0.2, 0.2], color=ACCENT, lw=5, solid_capstyle="round")
        ax.plot([alpha, 1], [0.2, 0.2], color="#87929A", lw=2, ls="--")
        ax.scatter([0, alpha, 1], [0.2] * 3, c=[INK, ACCENT, INK], s=35, zorder=5)
        ax.text(0, 0.54, r"$\theta_k$", ha="center", fontsize=12)
        ax.text(alpha, 0.54, r"$\theta_k+\alpha\Delta$", ha="center", fontsize=12)
        ax.text(1, 0.54, r"$\theta_p$", ha="center", fontsize=12)
        ax.text(
            alpha / 2, -0.10, "certified prefix", ha="center", fontsize=9, color=ACCENT
        )
        ax.text(
            (alpha + 1) / 2,
            -0.10,
            "outside this prefix",
            ha="center",
            fontsize=9,
            color="#617482",
        )
        ax.text(
            0.5,
            -0.41,
            "Schematic in parameter space; not a measured step length.",
            ha="center",
            fontsize=8,
            color="#617482",
        )
        self.y -= height + 6

    def validate(self) -> None:
        if self.y < 48:
            raise ValueError(f"Page {self.number} overflows: bottom={self.y:.1f} pt")


def render(source: Path, output: Path, previews: Path | None) -> None:
    pages = source.read_text(encoding="utf-8").split("<!-- pagebreak -->")
    if previews is not None:
        previews.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        output,
        metadata={
            "Title": "PSPO line-segment implementation",
            "Subject": "Geometry, certification, bisection and safe-update enforcement",
            "Author": "CertifiedContinualLearning implementation guide",
        },
    ) as pdf:
        for number, content in enumerate(pages, start=1):
            page = Page(number, len(pages))
            lines = content.strip().splitlines()
            index = 0
            while index < len(lines):
                line = lines[index].strip()
                index += 1
                if not line:
                    continue
                if line == "<!-- segment_diagram -->":
                    page.diagram()
                elif line.startswith("# "):
                    page.heading(line[2:], title=True)
                elif line.startswith("## "):
                    page.heading(line[3:])
                elif line == "$$":
                    equation = []
                    while lines[index].strip() != "$$":
                        equation.append(lines[index].strip())
                        index += 1
                    index += 1
                    page.equation(" ".join(equation))
                elif line.startswith("```"):
                    code = []
                    while not lines[index].startswith("```"):
                        code.append(lines[index])
                        index += 1
                    index += 1
                    page.code(code)
                else:
                    paragraph = [line]
                    while index < len(lines) and lines[index].strip():
                        paragraph.append(lines[index].strip())
                        index += 1
                    page.paragraph(" ".join(paragraph))
            page.validate()
            pdf.savefig(page.fig)
            if previews is not None:
                page.fig.savefig(previews / f"page_{number}.png", dpi=120)
            print(f"Page {number}: {page.y:.1f} pt bottom clearance")
            plt.close(page.fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previews", type=Path, help="Optional directory for PNG page previews"
    )
    args = parser.parse_args()
    source = Path(__file__).with_name("pspo_line_segment_implementation.md")
    render(source, source.with_suffix(".pdf"), args.previews)


if __name__ == "__main__":
    main()
