#!/usr/bin/env python3
"""Parse combined_task1_table.tex and produce one summary plot.

The input LaTeX table is expected to contain rows with:
  Environment, Policy, Provably Safe?, Phi_sc, Phi_traj, Avg Total Reward
where each metric cell is formatted as: <mean> $\\pm$ <std>.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "outputs" / "combined_task1_table.tex"
    default_output = script_dir / "outputs" / "combined_task1_summary_plot.png"

    parser = argparse.ArgumentParser(description="Plot summary from combined Task-1 LaTeX table.")
    parser.add_argument(
        "--input-tex",
        type=Path,
        default=default_input,
        help="Path to combined_task1_table.tex",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save output plot (.png/.pdf/.svg)",
    )
    return parser.parse_args()


def _clean_env(raw: str) -> str:
    m = re.search(r"\\texttt\{([^}]*)\}", raw)
    if m:
        return m.group(1).replace(r"\_", "_")
    return raw.strip().replace(r"\_", "_")


def _clean_policy(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"\\textsc\{([^}]*)\}", r"\1", s)
    s = s.replace("(ours)", "")
    s = s.replace("{", "").replace("}", "")
    s = s.strip()
    # Normalize spacing/casing for robust ordering.
    if s.lower() == "safeadapt":
        return "SafeAdapt"
    if s.lower() == "unsafeadapt":
        return "UnsafeAdapt"
    if s.lower() == "source":
        return "Source"
    if s.lower() == "ewc":
        return "EWC"
    return s


def _parse_mean_std(cell: str) -> tuple[float, float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", cell)
    if len(nums) < 2:
        raise ValueError(f"Could not parse mean/std from cell: {cell}")
    return float(nums[0]), float(nums[1])


def parse_latex_table(tex_path: Path) -> pd.DataFrame:
    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    rows: list[dict] = []
    current_env: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        if stripped.startswith("Environment") or stripped.startswith("Policy"):
            continue
        if stripped.startswith("\\") and not stripped.startswith(r"\multirow"):
            continue

        parts = [p.strip() for p in stripped[:-2].split("&")]
        if len(parts) != 6:
            continue

        env_field, policy_field = parts[0], parts[1]

        if env_field.startswith(r"\multirow"):
            current_env = _clean_env(env_field)
        elif env_field:
            current_env = _clean_env(env_field)

        if current_env is None:
            continue

        policy = _clean_policy(policy_field)

        phi_sc_mean, phi_sc_std = _parse_mean_std(parts[3])
        phi_tr_mean, phi_tr_std = _parse_mean_std(parts[4])
        rew_mean, rew_std = _parse_mean_std(parts[5])

        rows.append(
            {
                "Environment": current_env,
                "Policy": policy,
                "phi_sc_mean": phi_sc_mean,
                "phi_sc_std": phi_sc_std,
                "phi_traj_mean": phi_tr_mean,
                "phi_traj_std": phi_tr_std,
                "reward_mean": rew_mean,
                "reward_std": rew_std,
            }
        )

    if not rows:
        raise RuntimeError(f"No data rows parsed from: {tex_path}")

    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    env_order = list(dict.fromkeys(df["Environment"].tolist()))
    policy_order = ["Source", "UnsafeAdapt", "EWC", "SafeAdapt"]
    available_policies = [p for p in policy_order if p in set(df["Policy"])]

    metric_specs = [
        ("phi_sc", r"$\\Phi_{\\mathrm{sc}}(\\pi)$"),
        ("phi_traj", r"$\\Phi_{\\mathrm{traj.}}(\\pi)$"),
        ("reward", "Avg Total Reward"),
    ]

    colors = {
        "Source": "#4C78A8",
        "UnsafeAdapt": "#F58518",
        "EWC": "#54A24B",
        "SafeAdapt": "#E45756",
    }

    x = np.arange(len(env_order))
    width = 0.18 if len(available_policies) >= 4 else 0.22

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (metric_key, metric_title) in zip(axes, metric_specs):
        for i, policy in enumerate(available_policies):
            sub = df[df["Policy"] == policy].copy()
            sub = sub.set_index("Environment").reindex(env_order)

            means = sub[f"{metric_key}_mean"].to_numpy(dtype=float)
            stds = sub[f"{metric_key}_std"].to_numpy(dtype=float)

            offset = (i - (len(available_policies) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=policy,
                color=colors.get(policy, None),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
            )

        ax.set_title(metric_title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(env_order, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    # Sensible shared y-limits (all metrics are in [0,1] for this setup).
    for ax in axes:
        ax.set_ylim(0.0, 1.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(available_policies), frameon=False)
    fig.suptitle("FrozenLake Task-1 Summary Across Environments", y=1.04, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_tex = args.input_tex.resolve()
    output = args.output.resolve()

    if not input_tex.exists():
        raise FileNotFoundError(f"Input LaTeX table does not exist: {input_tex}")

    df = parse_latex_table(input_tex)
    make_plot(df, output)

    print("=" * 80)
    print("COMBINED TASK-1 PLOT")
    print("=" * 80)
    print(f"Input table : {input_tex}")
    print(f"Rows parsed : {len(df)}")
    print(f"Output plot : {output}")


if __name__ == "__main__":
    main()
