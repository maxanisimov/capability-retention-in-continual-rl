# %% [markdown]
# # Rashomon Sweep Notebook: `deterministic_vehicle_sluggish`
# 
# This notebook runs and analyzes **Rashomon adaptation sweeps** for LunarLander task setting `deterministic_vehicle_sluggish`.
# 
# It is designed to help answer:
# - Which Rashomon constraints are too restrictive?
# - Which settings improve downstream adaptation while preserving source performance?
# 
# The notebook uses:
# - `experiments/pipelines/behaviour_retention/lunarlander/cli/launch_multi_seed.py` (mode = `downstream_rashomon`)
# - per-seed `run_summary.yaml` files for metric aggregation.
# 

# %%
from __future__ import annotations

import itertools
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
try:
    from IPython.display import display
except ImportError:  # pragma: no cover - fallback for plain Python execution
    def display(x):  # type: ignore[no-redef]
        print(x)


# %%
def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "experiments").exists():
            return candidate
    raise RuntimeError("Could not find repository root from current working directory.")

REPO_ROOT = find_repo_root()
PIPELINE_ROOT = REPO_ROOT / "experiments/pipelines/behaviour_retention/lunarlander"
LAUNCHER = PIPELINE_ROOT / "cli/launch_multi_seed.py"
OUTPUTS_ROOT = PIPELINE_ROOT / "outputs"
TASK_SETTING = "deterministic_vehicle_sluggish"

# IMPORTANT: set these before launching.
SEEDS = list(range(10))
DEFAULT_CORES = sorted(os.sched_getaffinity(0))
CORES = DEFAULT_CORES[: min(8, len(DEFAULT_CORES))]

# Safety toggles
RUN_SWEEP = False          # flip to True when you are ready to launch jobs
DRY_RUN_ONLY = False       # if True, appends --dry-run to launcher command
FORCE_DISABLE_TASK_NEUTRALIZATION = True
PRINT_COMMANDS = True

print(f"REPO_ROOT={REPO_ROOT}")
print(f"TASK_SETTING={TASK_SETTING}")
print(f"SEEDS={SEEDS}")
print(f"CORES={CORES}")


# %%
def read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def seed_dir(seed: int) -> Path:
    return OUTPUTS_ROOT / TASK_SETTING / f"seed_{seed}"


def noadapt_summary_path(seed: int) -> Path:
    preferred = seed_dir(seed) / "noadapt" / "run_summary.yaml"
    legacy = seed_dir(seed) / "source" / "run_summary.yaml"
    return preferred if preferred.exists() else legacy


def rashomon_summary_path(seed: int, run_subdir: str) -> Path:
    return seed_dir(seed) / run_subdir / "run_summary.yaml"


def _float_tag(value: float) -> str:
    text = f"{value:.6g}".replace("-", "m").replace(".", "p")
    return text


def make_run_subdir(cfg: dict[str, Any]) -> str:
    return (
        "downstream_rashomon_sweep"
        f"_hs{_float_tag(float(cfg['rashomon_min_hard_spec']))}"
        f"_agg{cfg['rashomon_surrogate_aggregation']}"
        f"_rr{int(cfg['rashomon_rollouts'])}"
        f"_ni{int(cfg['rashomon_n_iters'])}"
        f"_ts{int(cfg['total_timesteps'])}"
        f"_lr{_float_tag(float(cfg['lr']))}"
    )


def build_rashomon_launch_cmd(
    cfg: dict[str, Any],
    *,
    seeds: list[int],
    cores: list[int],
    dry_run: bool = False,
) -> list[str]:
    run_subdir = make_run_subdir(cfg)

    cmd = [
        sys.executable,
        str(LAUNCHER),
        "--mode", "downstream_rashomon",
        "--pipeline", TASK_SETTING,
        "--seeds", *[str(s) for s in seeds],
        "--cores", *[str(c) for c in cores],
        "--source-run-root", str(OUTPUTS_ROOT),
        "--outputs-root", str(OUTPUTS_ROOT),
        "--run-subdir", run_subdir,
    ]

    if FORCE_DISABLE_TASK_NEUTRALIZATION:
        cmd.append("--disable-task-neutralization")

    if dry_run:
        cmd.append("--dry-run")

    # Forwarded args for adapt_rashomon.py
    cmd.extend(
        [
            "--",
            "--total-timesteps", str(int(cfg["total_timesteps"])),
            "--lr", str(float(cfg["lr"])),
            "--rashomon-rollouts", str(int(cfg["rashomon_rollouts"])),
            "--rashomon-n-iters", str(int(cfg["rashomon_n_iters"])),
            "--rashomon-min-hard-spec", str(float(cfg["rashomon_min_hard_spec"])),
            "--rashomon-surrogate-aggregation", str(cfg["rashomon_surrogate_aggregation"]),
        ],
    )

    return cmd


def run_cmd(cmd: list[str]) -> tuple[int, float]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start
    return int(proc.returncode), float(elapsed)


# %% [markdown]
# ## Define sweep space
# 
# Start with a moderate grid first. Expand only after seeing trends.
# 
# Recommended first-pass focus:
# - relax hard-spec (`1.0 -> 0.995/0.99/0.98`)
# - compare aggregation (`min` vs `mean`)
# - increase source rollouts (`100 -> 200`)
# - optionally lower PPO LR (`3e-4 -> 1e-4`)
# 

# %%
sweep_grid = {
    "rashomon_min_hard_spec": [1.0], # [1.0, 0.995, 0.99, 0.98],
    "rashomon_surrogate_aggregation": ['min'], # ["min", "mean"],
    "rashomon_rollouts": [1], # [100, 200],
    "rashomon_n_iters": [50_000, 100_000],
    "total_timesteps": [400_000],
    "lr": [3e-4, 1e-4],
}

keys = list(sweep_grid.keys())
all_cfgs = [dict(zip(keys, values)) for values in itertools.product(*[sweep_grid[k] for k in keys])]

# Optional: cap how many configs to run in one pass.
MAX_CONFIGS = 24
configs = all_cfgs[:MAX_CONFIGS] if MAX_CONFIGS is not None else all_cfgs

preview = []
for i, cfg in enumerate(configs):
    run_subdir = make_run_subdir(cfg)
    cmd = build_rashomon_launch_cmd(cfg, seeds=SEEDS, cores=CORES, dry_run=DRY_RUN_ONLY)
    preview.append(
        {
            "idx": i,
            "run_subdir": run_subdir,
            "min_hard_spec": cfg["rashomon_min_hard_spec"],
            "aggregation": cfg["rashomon_surrogate_aggregation"],
            "rollouts": cfg["rashomon_rollouts"],
            "n_iters": cfg["rashomon_n_iters"],
            "timesteps": cfg["total_timesteps"],
            "lr": cfg["lr"],
            "cmd": " ".join(shlex.quote(x) for x in cmd),
        },
    )

preview_df = pd.DataFrame(preview)
print(f"Prepared {len(preview_df)} configs (from full grid of {len(all_cfgs)}).")
display(preview_df[["idx", "run_subdir", "min_hard_spec", "aggregation", "rollouts", "n_iters", "timesteps", "lr"]])


# %%
# Inspect exact commands before running
if PRINT_COMMANDS:
    for _, row in preview_df.iterrows():
        print(f"\n[{row['idx']}] {row['run_subdir']}")
        print(row["cmd"])


# %% [markdown]
# ## Launch sweep
# 
# This cell launches one multi-seed job per config.
# 
# - Keep `RUN_SWEEP=False` for safety.
# - Set `RUN_SWEEP=True` when ready.
# - Launcher writes per-seed logs under `outputs/<task_setting>/multi_seed_logs/downstream_rashomon`.
# 

# %%
RUN_SWEEP = True
launch_records: list[dict[str, Any]] = []
if not RUN_SWEEP:
    print("RUN_SWEEP=False -> skipping execution.")
else:
    for i, cfg in enumerate(configs):
        run_subdir = make_run_subdir(cfg)
        cmd = build_rashomon_launch_cmd(cfg, seeds=SEEDS, cores=CORES, dry_run=DRY_RUN_ONLY)

        print(f"\n=== [{i+1}/{len(configs)}] {run_subdir} ===")
        rc, elapsed = run_cmd(cmd)
        print(f"return_code={rc} | elapsed={elapsed:.1f}s")

        launch_records.append(
            {
                "idx": i,
                "run_subdir": run_subdir,
                "return_code": rc,
                "elapsed_sec": elapsed,
                **cfg,
            },
        )

launch_df = pd.DataFrame(launch_records)
if not launch_df.empty:
    display(launch_df)


# %% [markdown]
# ## Aggregate results from `run_summary.yaml`
# 
# This computes per-config means/std across completed seeds and compares against:
# - `NoAdapt` (NoAdapt policy evaluated on source/downstream)
# - `downstream_unconstrained`
# - `downstream_ewc`
# 

# %%
def collect_noadapt_baseline(seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        data = read_yaml(noadapt_summary_path(seed))
        if not data:
            continue
        rr = data.get("run_results", {})
        rows.append(
            {
                "seed": seed,
                "source_mean_reward": rr.get("source_mean_reward"),
                "source_failure_rate": rr.get("source_failure_rate"),
                "downstream_mean_reward": rr.get("downstream_mean_reward"),
                "downstream_failure_rate": rr.get("downstream_failure_rate"),
            },
        )
    return pd.DataFrame(rows)


def collect_rashomon_config(run_subdir: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        data = read_yaml(rashomon_summary_path(seed, run_subdir))
        if not data:
            continue
        rr = data.get("run_results", {})
        rs = data.get("run_settings", {})
        rows.append(
            {
                "seed": seed,
                "run_subdir": run_subdir,
                "source_mean_reward": rr.get("source_mean_reward"),
                "source_failure_rate": rr.get("source_failure_rate"),
                "downstream_mean_reward": rr.get("downstream_mean_reward"),
                "downstream_failure_rate": rr.get("downstream_failure_rate"),
                "selected_certificate": rr.get("selected_certificate"),
                "rashomon_dataset_size": rr.get("rashomon_dataset_size"),
                "inverse_temperature": rr.get("inverse_temperature"),
                "cfg_min_hard_spec": rs.get("rashomon_min_hard_spec", rs.get("min_hard_spec")),
                "cfg_aggregation": rs.get("surrogate_aggregation"),
                "cfg_rollouts": rs.get("rashomon_rollouts"),
                "cfg_n_iters": rs.get("rashomon_n_iters"),
            },
        )
    return pd.DataFrame(rows)


def load_policy_reference(policy: str) -> dict[str, float] | None:
    agg_path = OUTPUTS_ROOT / TASK_SETTING / "aggregate_layout_metrics.csv"
    if not agg_path.exists():
        return None
    df = pd.read_csv(agg_path)
    row = df[df["policy"] == policy]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "source_total_reward_mean": float(row["source_total_reward_mean"]),
        "downstream_total_reward_mean": float(row["downstream_total_reward_mean"]),
    }


source_df = collect_noadapt_baseline(SEEDS)
if source_df.empty:
    raise RuntimeError("No NoAdapt run summaries found. Train noadapt first.")

source_ref_src = float(source_df["source_mean_reward"].mean())
source_ref_dst = float(source_df["downstream_mean_reward"].mean())

unconstrained_ref = load_policy_reference("downstream_unconstrained")
ewc_ref = load_policy_reference("downstream_ewc")

finite_downstream_refs = [
    v
    for v in [
        unconstrained_ref["downstream_total_reward_mean"] if unconstrained_ref else float("-inf"),
        ewc_ref["downstream_total_reward_mean"] if ewc_ref else float("-inf"),
    ]
    if np.isfinite(v)
]
best_non_rash_dst = max(finite_downstream_refs) if finite_downstream_refs else float("nan")

if not np.isfinite(best_non_rash_dst):
    best_non_rash_dst = float("nan")

rows = []
for cfg in configs:
    run_subdir = make_run_subdir(cfg)
    rdf = collect_rashomon_config(run_subdir, SEEDS)
    if rdf.empty:
        rows.append(
            {
                "run_subdir": run_subdir,
                "n_seeds": 0,
                "source_mean_reward": np.nan,
                "source_std_reward": np.nan,
                "source_failure_rate": np.nan,
                "downstream_mean_reward": np.nan,
                "downstream_std_reward": np.nan,
                "downstream_failure_rate": np.nan,
                "selected_certificate_mean": np.nan,
                "dataset_size_mean": np.nan,
                "inverse_temperature_mean": np.nan,
                "delta_source_vs_noadapt": np.nan,
                "delta_downstream_vs_noadapt": np.nan,
                "abs_source_gap_vs_noadapt": np.nan,
                "downstream_ratio_to_best_unconstrained_ewc": np.nan,
                "meets_source_preservation_abs5": False,
                "meets_downstream_90pct_best": False,
                "score": np.nan,
                **cfg,
            },
        )
        continue

    src_mean = float(rdf["source_mean_reward"].mean())
    dst_mean = float(rdf["downstream_mean_reward"].mean())

    row = {
        "run_subdir": run_subdir,
        "n_seeds": int(len(rdf)),
        "source_mean_reward": src_mean,
        "source_std_reward": float(rdf["source_mean_reward"].std(ddof=0)),
        "source_failure_rate": float(rdf["source_failure_rate"].mean()),
        "downstream_mean_reward": dst_mean,
        "downstream_std_reward": float(rdf["downstream_mean_reward"].std(ddof=0)),
        "downstream_failure_rate": float(rdf["downstream_failure_rate"].mean()),
        "selected_certificate_mean": float(rdf["selected_certificate"].mean()),
        "dataset_size_mean": float(rdf["rashomon_dataset_size"].mean()),
        "inverse_temperature_mean": float(rdf["inverse_temperature"].mean()),
        "delta_source_vs_noadapt": src_mean - source_ref_src,
        "delta_downstream_vs_noadapt": dst_mean - source_ref_dst,
        "abs_source_gap_vs_noadapt": abs(src_mean - source_ref_src),
        "downstream_ratio_to_best_unconstrained_ewc": (
            dst_mean / best_non_rash_dst if np.isfinite(best_non_rash_dst) and best_non_rash_dst != 0 else np.nan
        ),
        "meets_source_preservation_abs5": abs(src_mean - source_ref_src) <= 5.0,
        "meets_downstream_90pct_best": (
            dst_mean >= 0.9 * best_non_rash_dst if np.isfinite(best_non_rash_dst) else False
        ),
        **cfg,
    }
    # simple composite score: prioritize downstream, penalize source drift
    row["score"] = row["downstream_mean_reward"] - 2.0 * row["abs_source_gap_vs_noadapt"]
    rows.append(row)

results_df = pd.DataFrame(rows)
if not results_df.empty:
    results_df = results_df.sort_values(by=["score", "downstream_mean_reward"], ascending=False)

print("NoAdapt reference (NoAdapt policy):")
print(f"  source_mean_reward={source_ref_src:.3f}")
print(f"  downstream_mean_reward={source_ref_dst:.3f}")
if unconstrained_ref:
    print(f"Unconstrained downstream_mean_reward={unconstrained_ref['downstream_total_reward_mean']:.3f}")
if ewc_ref:
    print(f"EWC downstream_mean_reward={ewc_ref['downstream_total_reward_mean']:.3f}")
print(f"Best(non-Rashomon) downstream_mean_reward={best_non_rash_dst:.3f}")

cols = [
    "run_subdir",
    "n_seeds",
    "source_mean_reward",
    "downstream_mean_reward",
    "delta_source_vs_noadapt",
    "delta_downstream_vs_noadapt",
    "downstream_ratio_to_best_unconstrained_ewc",
    "meets_source_preservation_abs5",
    "meets_downstream_90pct_best",
    "score",
    "rashomon_min_hard_spec",
    "rashomon_surrogate_aggregation",
    "rashomon_rollouts",
    "rashomon_n_iters",
    "total_timesteps",
    "lr",
]
display(results_df[cols])


# %%
# Optional quick scatter: source preservation vs downstream reward
if not results_df.empty:
    try:
        ax = results_df.plot(
            kind="scatter",
            x="abs_source_gap_vs_noadapt",
            y="downstream_mean_reward",
            figsize=(7, 5),
            title="Rashomon sweep: source drift vs downstream reward",
        )
        ax.set_xlabel("Absolute source gap vs NoAdapt")
        ax.set_ylabel("Downstream mean reward")

        top = results_df.head(5)
        for _, r in top.iterrows():
            ax.annotate(r["run_subdir"], (r["abs_source_gap_vs_noadapt"], r["downstream_mean_reward"]), fontsize=8)
    except ImportError as exc:
        print(f"Skipping plot because plotting backend is unavailable: {exc}")


# %%
# Save table for later reporting
out_csv = OUTPUTS_ROOT / TASK_SETTING / "rashomon_sweep_results_deterministic_vehicle_sluggish.csv"
results_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
