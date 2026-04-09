#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUTS_DIR = Path(
    "/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning/rl_project/experiments/frozen_lake/outputs"
)

POLICY_ORDER = ("Source", "UnsafeAdapt", "EWC", "SafeAdapt")
POLICY_LATEX_LABEL = {
    "Source": "Source",
    "UnsafeAdapt": "UnsafeAdapt",
    "EWC": "EWC",
    "SafeAdapt": r"\textsc{SafeAdapt} (ours)",
}


def _normalize_policy_name(raw: str) -> str | None:
    key = raw.strip().lower()
    if "unsafeadapt" in key:
        return "UnsafeAdapt"
    if "safeadapt" in key:
        return "SafeAdapt"
    if key == "source":
        return "Source"
    if key == "ewc":
        return "EWC"
    return None


def _escape_env_name(env_name: str) -> str:
    return env_name.replace("_", r"\_")


def _parse_task2_table(table_path: Path) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in table_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        if stripped.startswith("Policy &"):
            continue
        parts = [p.strip() for p in stripped[:-2].split("&")]
        if len(parts) != 3:
            continue
        policy_name = _normalize_policy_name(parts[0])
        if policy_name is None:
            continue
        rows[policy_name] = (parts[1], parts[2])
    return rows


def _build_combined_table(
    env_to_rows: dict[str, dict[str, tuple[str, str]]],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{FrozenLake Task 2 results across environments (mean $\pm$ std over seeds).}",
        r"\label{tab:frozenlake_task2_combined}",
        r"\begin{tabularx}{\textwidth}{",
        r"l l ",
        r">{\centering\arraybackslash}X ",
        r">{\centering\arraybackslash}X",
        r"}",
        r"\toprule",
        r"Environment & Policy & Avg Total Reward & Success Rate \\",
        r"\midrule",
        "",
    ]

    env_names = sorted(env_to_rows.keys())
    for env_idx, env_name in enumerate(env_names):
        env_rows = env_to_rows[env_name]
        escaped_env = _escape_env_name(env_name)

        for policy_idx, policy in enumerate(POLICY_ORDER):
            reward, success = env_rows.get(policy, ("--", "--"))
            env_cell = (
                rf"\multirow{{4}}{{*}}{{\texttt{{{escaped_env}}}}}"
                if policy_idx == 0
                else ""
            )
            lines.append(
                f"{env_cell} & {POLICY_LATEX_LABEL[policy]} & "
                f"{reward} & {success} \\\\"
            )

        if env_idx != len(env_names) - 1:
            lines.append(r"\midrule")
            lines.append("")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _collect_environment_tables(outputs_dir: Path) -> dict[str, dict[str, tuple[str, str]]]:
    env_tables: dict[str, dict[str, tuple[str, str]]] = {}
    for env_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        task2_table = env_dir / "aggregated" / "task2_table.tex"
        if not task2_table.exists():
            continue
        parsed_rows = _parse_task2_table(task2_table)
        if not parsed_rows:
            continue
        env_tables[env_dir.name] = parsed_rows
    return env_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-environment aggregated Task 2 latex tables into one table."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Path to the FrozenLake outputs directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Destination .tex file. Defaults to <outputs-dir>/combined_task2_table.tex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir
    output_file = args.output_file or (outputs_dir / "combined_task2_table.tex")

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    env_tables = _collect_environment_tables(outputs_dir)
    if not env_tables:
        raise RuntimeError(
            "No valid task2 tables found. Expected files at <env>/aggregated/task2_table.tex"
        )

    combined_table = _build_combined_table(env_tables)
    output_file.write_text(combined_table, encoding="utf-8")
    print(f"Combined Task 2 table written to: {output_file}")
    print(f"Environments included: {len(env_tables)}")


if __name__ == "__main__":
    main()

