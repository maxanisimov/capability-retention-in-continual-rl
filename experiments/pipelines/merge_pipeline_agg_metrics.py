"""Merge per-setting aggregate metrics LaTeX tables into one paper table.

The expected inputs are the per-pipeline ``aggregate_layout_metrics.tex`` files
emitted by the FrozenLake and LunarLander aggregation scripts. Each input table
should have a first ``Policy`` column followed by one or more metric columns.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_TABLE_NAME = "aggregate_layout_metrics.tex"
DEFAULT_OUTPUT_NAME = "merge_pipeline_agg_metrics.tex"

METHOD_ORDER = [
    "noadapt",
    "source",
    "unconstrained",
    "uncadapt",
    "ewc",
    "rashomon",
    "downstream_rashomon",
    "rashomon+",
    "rashomon_plus",
    "downstream_rashomon_plus",
    "downstream_rashomon_nonconvex",
]

DEFAULT_SETTING_ORDER = [
    "twitchy",
    "sluggish",
    "underpowered",
    "diagonal_10x10",
    "diagonal_20x20",
    "diagonal_30x30",
    "diagonal_60x60",
]

METHOD_RENAME = {
    "noadapt": r"\textit{NoAdapt}",
    "source": r"\textit{NoAdapt}",
    "unconstrained": "UncAdapt",
    "uncadapt": "UncAdapt",
    "ewc": "EWC",
    "rashomon": r"\textsc{Rashomon} (ours)",
    "downstream_rashomon": r"\textsc{Rashomon} (ours)",
    "rashomon+": r"\textsc{Rashomon+} (ours)",
    "rashomon_plus": r"\textsc{Rashomon+} (ours)",
    "downstream_rashomon_plus": r"\textsc{Rashomon+} (ours)",
    "downstream_rashomon_nonconvex": r"\textsc{Rashomon+} (ours)",
}

METRIC_HEADER_RENAME = {
    "source total reward": r"$J_{\mathrm{source}} \uparrow$",
    "source": r"$J_{\mathrm{source}} \uparrow$",
    "source success rate": r"$\mathrm{SR}_{\mathrm{source}} \uparrow$",
    "downstream total reward": r"$J_{\mathrm{down}} \uparrow$",
    "downstream": r"$J_{\mathrm{down}} \uparrow$",
    "downstream success rate": r"$\mathrm{SR}_{\mathrm{down}} \uparrow$",
}


@dataclass(frozen=True)
class MetricCell:
    raw: str
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class MethodRow:
    method: str
    cells: tuple[MetricCell, ...]
    input_index: int


@dataclass(frozen=True)
class ParsedTable:
    path: Path
    setting: str
    metric_headers: tuple[str, ...]
    rows: tuple[MethodRow, ...]
    caption: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _latex_unescape_text(text: str) -> str:
    return (
        text.strip()
        .replace(r"\_", "_")
        .replace(r"\&", "&")
        .replace(r"\%", "%")
        .replace(r"\#", "#")
    )


def _strip_latex_text_wrappers(text: str) -> str:
    out = text.strip()
    previous = None
    wrapper_re = re.compile(r"\\(?:textit|textsc|texttt|mathbf|underline)\{([^{}]*)\}")
    while previous != out:
        previous = out
        out = wrapper_re.sub(r"\1", out)
    out = out.replace("(ours)", "")
    out = out.replace("$", "")
    return _latex_unescape_text(out).strip()


def _normalize_key(text: str) -> str:
    normalized = _strip_latex_text_wrappers(text).lower()
    normalized = normalized.replace(r"\mathrm", "")
    normalized = normalized.replace("+", "_plus")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _split_latex_row(line: str) -> list[str]:
    cleaned = line.strip()
    if cleaned.endswith(r"\\"):
        cleaned = cleaned[:-2]
    return [cell.strip() for cell in cleaned.split("&")]


def _caption_from_text(text: str) -> str | None:
    match = re.search(r"\\caption\{(?P<caption>.*?)\}", text, flags=re.DOTALL)
    if match is None:
        return None
    return " ".join(match.group("caption").split())


def _setting_from_caption(caption: str | None) -> str | None:
    if caption is None:
        return None
    match = re.search(r"\((?P<setting>[^()]*)\)", caption)
    if match is None:
        return None
    return _latex_unescape_text(match.group("setting"))


def _pretty_setting_name(raw: str) -> str:
    setting = _latex_unescape_text(raw)
    for prefix in (
        "deterministic__default_to_",
        "deterministic_default_to_",
        "default_to_",
        "cond_",
    ):
        if setting.startswith(prefix):
            setting = setting.removeprefix(prefix)
            break
    for suffix in ("_vehicle", "_layout", "_pipeline"):
        if setting.endswith(suffix):
            setting = setting.removesuffix(suffix)
            break
    return setting


def _parse_metric_cell(cell: str) -> MetricCell:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cell.replace(",", ""))
    if len(numbers) >= 2:
        return MetricCell(raw=cell, mean=float(numbers[0]), std=float(numbers[1]))
    return MetricCell(raw=cell)


def _tabular_lines(tex: str, path: Path) -> list[str]:
    begin_match = re.search(r"\\begin\{tabular\}\{[^{}]*\}", tex)
    if begin_match is None:
        raise ValueError(f"No tabular environment found in {path}")
    end_idx = tex.find(r"\end{tabular}", begin_match.end())
    if end_idx < 0:
        raise ValueError(f"No closing tabular environment found in {path}")
    return tex[begin_match.end() : end_idx].splitlines()


def parse_aggregate_table(path: Path, *, pretty_setting: bool = True) -> ParsedTable:
    tex = path.read_text(encoding="utf-8")
    caption = _caption_from_text(tex)
    setting = _setting_from_caption(caption) or path.parent.name
    if pretty_setting:
        setting = _pretty_setting_name(setting)

    lines = [line.strip() for line in _tabular_lines(tex, path) if line.strip()]
    header_line: str | None = None
    body_lines: list[str] = []
    in_body = False

    for line in lines:
        if line in {r"\toprule", r"\cmidrule", r"\bottomrule"}:
            continue
        if line.startswith(r"\cmidrule"):
            continue
        if line == r"\midrule":
            in_body = True
            continue
        if line == r"\bottomrule":
            break
        if not in_body:
            if "&" in line and line.endswith(r"\\"):
                header_line = line
            continue
        if line.startswith(r"\bottomrule"):
            break
        if "&" in line and line.endswith(r"\\"):
            body_lines.append(line)

    if header_line is None:
        raise ValueError(f"No header row found in {path}")

    header_cells = _split_latex_row(header_line)
    if not header_cells:
        raise ValueError(f"Empty header row in {path}")
    if _normalize_key(header_cells[0]) != "policy":
        raise ValueError(
            f"Expected first column to be Policy in {path}, got {header_cells[0]!r}",
        )

    metric_headers = tuple(_strip_latex_text_wrappers(cell) for cell in header_cells[1:])
    if not metric_headers:
        raise ValueError(f"No metric columns found in {path}")

    rows: list[MethodRow] = []
    for input_index, line in enumerate(body_lines):
        cells = _split_latex_row(line)
        if len(cells) != len(header_cells):
            raise ValueError(
                f"Expected {len(header_cells)} cells in {path}, got {len(cells)}: {line}",
            )
        rows.append(
            MethodRow(
                method=_strip_latex_text_wrappers(cells[0]),
                cells=tuple(_parse_metric_cell(cell) for cell in cells[1:]),
                input_index=input_index,
            ),
        )

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    return ParsedTable(
        path=path,
        setting=setting,
        metric_headers=metric_headers,
        rows=tuple(rows),
        caption=caption,
    )


def _format_metric_header(header: str) -> str:
    normalized = _strip_latex_text_wrappers(header).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return METRIC_HEADER_RENAME.get(normalized, _escape_latex(header))


def _format_float(value: float) -> str:
    rounded = round(value, 2)
    if abs(rounded) < 0.005:
        rounded = 0.0
    return f"{rounded:.2f}"


def _format_metric_cell(cell: MetricCell, *, bold: bool = False, underline: bool = False) -> str:
    if cell.mean is None or cell.std is None:
        return cell.raw

    mean = _format_float(cell.mean)
    std = _format_float(cell.std)
    if bold:
        return rf"$\mathbf{{{mean}}} \pm {std}$"
    if underline:
        return rf"$\underline{{{mean}}} \pm {std}$"
    return rf"${mean} \pm {std}$"


def _metric_rule(header: str) -> str:
    normalized = _strip_latex_text_wrappers(header).lower()
    if "failure" in normalized or "cost" in normalized:
        return "min"
    return "max"


def _is_downstream_metric(header: str) -> bool:
    normalized = _strip_latex_text_wrappers(header).lower()
    return "downstream" in normalized or normalized == "down"


def _best_and_second(values: list[float], *, rule: str) -> tuple[float | None, float | None]:
    distinct: list[float] = []
    for value in values:
        if not any(math.isclose(value, old, abs_tol=1e-12) for old in distinct):
            distinct.append(value)
    if not distinct:
        return None, None
    distinct.sort(reverse=(rule == "max"))
    best = distinct[0]
    second = distinct[1] if len(distinct) > 1 else None
    return best, second


def _format_method(method: str) -> str:
    key = _normalize_key(method)
    if key in METHOD_RENAME:
        return METHOD_RENAME[key]
    return _escape_latex(method)


def _method_sort_key(row: MethodRow) -> tuple[int, int, str]:
    key = _normalize_key(row.method)
    try:
        order_index = METHOD_ORDER.index(key)
    except ValueError:
        order_index = len(METHOD_ORDER)
    return order_index, row.input_index, row.method


def _format_setting(setting: str) -> str:
    return rf"\texttt{{{_escape_latex(setting)}}}"


def _infer_domain(tables: list[ParsedTable]) -> str | None:
    joined = " ".join(
        [str(table.path) for table in tables]
        + [table.caption or "" for table in tables]
    ).lower()
    if "lunarlander" in joined or "lunar_lander" in joined:
        return "lunarlander"
    if "frozenlake" in joined or "frozen_lake" in joined:
        return "frozenlake"
    return None


def _default_caption(tables: list[ParsedTable]) -> str:
    domain = _infer_domain(tables)
    if domain == "lunarlander":
        name = "Lunar Lander"
    elif domain == "frozenlake":
        name = "Frozen Lake"
    else:
        name = "Pipeline"
    return (
        f"{name} results across source--downstream settings. "
        r"Values are mean $\pm$ standard deviation across seeds."
    )


def _default_label(tables: list[ParsedTable]) -> str:
    domain = _infer_domain(tables)
    if domain in {"lunarlander", "frozenlake"}:
        return f"tab:{domain}_merged_pipeline_agg_metrics"
    return "tab:merged_pipeline_agg_metrics"


def _validate_metric_headers(tables: list[ParsedTable]) -> tuple[str, ...]:
    first = tables[0].metric_headers
    for table in tables[1:]:
        if table.metric_headers != first:
            raise ValueError(
                "All input tables must have matching metric columns. "
                f"{tables[0].path} has {first}, but {table.path} has {table.metric_headers}.",
            )
    return first


def build_merged_latex_table(
    tables: list[ParsedTable],
    *,
    caption: str | None = None,
    label: str | None = None,
    table_env: str = "table*",
    position: str = "t",
    font_size: str = r"\small",
    tabcolsep: str = "4.5pt",
    underline_second_best: str = "downstream",
) -> str:
    if not tables:
        raise ValueError("No tables supplied.")

    metric_headers = _validate_metric_headers(tables)
    col_spec = "ll" + ("c" * len(metric_headers))
    header_cells = ["Setting", "Method"] + [
        _format_metric_header(header) for header in metric_headers
    ]

    lines: list[str] = [
        rf"\begin{{{table_env}}}[{position}]",
        r"\centering",
        font_size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    for table_index, table in enumerate(tables):
        rows = sorted(table.rows, key=_method_sort_key)
        best_by_metric: dict[int, float | None] = {}
        second_by_metric: dict[int, float | None] = {}
        for metric_index, header in enumerate(metric_headers):
            values = [
                row.cells[metric_index].mean
                for row in rows
                if row.cells[metric_index].mean is not None
            ]
            present_values = [value for value in values if value is not None]
            best, second = _best_and_second(
                present_values,
                rule=_metric_rule(header),
            )
            best_by_metric[metric_index] = best
            second_by_metric[metric_index] = second

        lines.append(rf"\multirow{{{len(rows)}}}{{*}}{{{_format_setting(table.setting)}}}")
        for row in rows:
            row_cells = ["&", _format_method(row.method)]
            for metric_index, cell in enumerate(row.cells):
                best = best_by_metric[metric_index]
                second = second_by_metric[metric_index]
                bold = (
                    cell.mean is not None
                    and best is not None
                    and math.isclose(cell.mean, best, abs_tol=1e-12)
                )
                underline_allowed = underline_second_best == "all" or (
                    underline_second_best == "downstream"
                    and _is_downstream_metric(metric_headers[metric_index])
                )
                underline = (
                    not bold
                    and underline_allowed
                    and cell.mean is not None
                    and second is not None
                    and math.isclose(cell.mean, second, abs_tol=1e-12)
                )
                row_cells.append(_format_metric_cell(cell, bold=bold, underline=underline))
            lines.append(" ".join(row_cells[:1]) + " " + " & ".join(row_cells[1:]) + r" \\")

        if table_index != len(tables) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption or _default_caption(tables)}}}",
            rf"\label{{{label or _default_label(tables)}}}",
            rf"\end{{{table_env}}}",
        ],
    )
    return "\n".join(lines)


def _discover_tables(input_root: Path, table_name: str) -> list[Path]:
    return sorted(path for path in input_root.glob(f"*/{table_name}") if path.is_file())


def _sort_tables(
    tables: list[ParsedTable],
    *,
    setting_order: list[str] | None,
) -> list[ParsedTable]:
    selected_order = setting_order or DEFAULT_SETTING_ORDER
    order = {_pretty_setting_name(setting): index for index, setting in enumerate(selected_order)}
    return sorted(tables, key=lambda table: (order.get(table.setting, len(order)), table.setting))


def _default_input_root_for_pipeline(pipeline: str) -> Path:
    return _repo_root() / "experiments" / "pipelines" / pipeline / "artifacts" / "runs"


def _resolve_input_paths(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    if args.tables:
        paths = [Path(path) for path in args.tables]
        return paths, None

    input_root = Path(args.input_root) if args.input_root is not None else None
    if input_root is None and args.pipeline is not None:
        input_root = _default_input_root_for_pipeline(args.pipeline)
    if input_root is None:
        input_root = Path.cwd()

    paths = _discover_tables(input_root, args.table_name)
    return paths, input_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-setting aggregate_layout_metrics.tex files into one "
            "LaTeX table with a multirow Setting column."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help=(
            "Pipeline name under experiments/pipelines, e.g. lunarlander or frozenlake. "
            "Ignored when --input-root or --tables is supplied."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Directory containing <setting>/aggregate_layout_metrics.tex files. "
            "Defaults to experiments/pipelines/<pipeline>/artifacts/runs when "
            "--pipeline is supplied, otherwise the current working directory."
        ),
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit aggregate_layout_metrics.tex files to merge.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=DEFAULT_INPUT_TABLE_NAME,
        help="Input table filename to find under each setting directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output .tex path. Defaults to <input-root>/{DEFAULT_OUTPUT_NAME}, "
            f"or ./{DEFAULT_OUTPUT_NAME} when --tables is used."
        ),
    )
    parser.add_argument(
        "--setting-order",
        nargs="+",
        default=None,
        help="Optional explicit setting order after pretty-name normalization.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Optional replacement caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional replacement LaTeX label.",
    )
    parser.add_argument(
        "--table-env",
        type=str,
        default="table*",
        help="LaTeX table environment to emit.",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="t",
        help="LaTeX float position option.",
    )
    parser.add_argument(
        "--font-size",
        type=str,
        default=r"\small",
        help="LaTeX font size command placed before the tabular.",
    )
    parser.add_argument(
        "--tabcolsep",
        type=str,
        default="4.5pt",
        help="Value for \\tabcolsep.",
    )
    parser.add_argument(
        "--underline-second-best",
        choices=["none", "downstream", "all"],
        default="downstream",
        help=(
            "Underline the second-best mean in no columns, downstream columns "
            "(default), or all metric columns."
        ),
    )
    parser.add_argument(
        "--no-pretty-setting",
        action="store_true",
        help="Use raw setting directory/caption names instead of shortened labels.",
    )
    args = parser.parse_args()

    input_paths, input_root = _resolve_input_paths(args)
    if not input_paths:
        raise FileNotFoundError(
            "No input tables found. Provide --pipeline, --input-root, or --tables. "
            f"Looked for */{args.table_name}"
            + (f" under {input_root}" if input_root is not None else "."),
        )

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input table(s): {missing}")

    parsed_tables = [
        parse_aggregate_table(path, pretty_setting=not args.no_pretty_setting)
        for path in input_paths
    ]
    parsed_tables = _sort_tables(parsed_tables, setting_order=args.setting_order)

    latex = build_merged_latex_table(
        parsed_tables,
        caption=args.caption,
        label=args.label,
        table_env=args.table_env,
        position=args.position,
        font_size=args.font_size,
        tabcolsep=args.tabcolsep,
        underline_second_best=args.underline_second_best,
    )

    output = args.output
    if output is None:
        output = (input_root or Path.cwd()) / DEFAULT_OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\nWrote merged LaTeX table: {output}")


if __name__ == "__main__":
    main()
