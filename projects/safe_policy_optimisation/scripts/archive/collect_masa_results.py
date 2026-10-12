#!/usr/bin/env python3
"""Collect every MASA-environment result into one canonical directory.

The six MASA environments (arXiv 2503.07671) are currently spread across three
unrelated places: the four default-budget environments under
``outputs/_sweeps_2hidden_*_baselines_only/``, Bridge Crossing v2's 1.6M
extended baselines under ``runs/rl_baselines_extended_budgets_1600k/``, and
MiniPacman's 2M extended baselines under ``runs/rl_baselines_extended_budgets/``
-- whose ``bridge_crossing_v2/`` is a *different*, 1.0M run and deliberately not
used here. PSPO adds four more source roots on top.

This script copies the result-bearing files (and only those) into

    artifacts/paper_2503_07671/runs/masa_all_envs/<env_key>/seed<N>/

laid out so that the consolidated per-environment directory works as *both* the
``baseline_root`` and the ``adaptive_root`` of an ``EnvironmentSpec`` in
``plot_extended_budget_learning_curves.py``: baseline methods keep their
``ppo_policy/``, ``ppo_lagrangian/``, ``cpo/``, ``ppo_shield/`` subdirectories
and PSPO's files sit at the seed root, exactly as the plot scripts expect.

Model checkpoints, tensorboard event files and the large per-step action logs
are skipped -- the raw trees total ~47 GB, the result files a few hundred MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_OUTPUT = RUNS / "masa_all_envs"

EXPECTED_SEEDS = tuple(range(10))

# Baseline method subdirectories copied verbatim from the sweep seed dir.
BASELINE_METHOD_DIRS = ("ppo_policy", "ppo_lagrangian", "cpo", "ppo_shield")

# Files worth keeping, by exact name. Everything else is skipped unless it is a
# .csv under a learning_curves/ directory (handled separately).
KEEP_NAMES = frozenset(
    {"metrics.json", "summary.json", "config.json", "training_episodes.csv"}
)

# Explicitly dropped: big and reproducible from the checkpoint if ever needed.
SKIP_NAMES = frozenset(
    {"model.zip", "episodes.csv", "executed_unsafe_actions.csv",
     "early_stop_evaluations.csv", "episodes_shielded.csv", "episodes_nominal.csv"}
)

# The baseline sweep writes an aggregate summary.json at the seed root, and so
# does PSPO. They would collide once merged, so the baseline one is renamed.
BASELINE_ROOT_SUMMARY = "baselines_summary.json"


@dataclass(frozen=True)
class EnvSource:
    key: str
    label: str
    budget: int
    baseline_root: Path
    pspo_root: Path
    note: str = ""


ENVIRONMENTS: tuple[EnvSource, ...] = (
    EnvSource(
        "media_streaming", "Media Streaming", 25_000,
        REPO / "outputs/_sweeps_2hidden_media_streaming_baselines_only/media_streaming",
        RUNS / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
             / "two_hidden/media_streaming",
    ),
    EnvSource(
        "colour_bomb", "Colour Bomb v1", 25_000,
        REPO / "outputs/_sweeps_2hidden_colour_bomb_baselines_only/colour_bomb",
        RUNS / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
             / "two_hidden/colour_bomb",
    ),
    EnvSource(
        "colour_bomb_v2", "Colour Bomb v2", 100_000,
        REPO / "outputs/_sweeps_2hidden_colour_bomb_v2_baselines_only/colour_bomb_v2",
        RUNS / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
             / "two_hidden/colour_bomb_v2",
    ),
    EnvSource(
        "bridge_crossing", "Bridge Crossing v1", 200_000,
        REPO / "outputs/_sweeps_2hidden_bridge_crossing_baselines_only/bridge_crossing",
        RUNS / "pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1"
             / "two_hidden/bridge_crossing",
    ),
    EnvSource(
        "bridge_crossing_v2", "Bridge Crossing v2", 1_600_000,
        RUNS / "rl_baselines_extended_budgets_1600k/bridge_crossing_v2",
        RUNS / "pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base"
             / "two_hidden/bridge_crossing_v2",
        note="extended budget; baselines matched at 1.6M",
    ),
    EnvSource(
        "mini_pacman", "MiniPacman", 2_000_000,
        RUNS / "rl_baselines_extended_budgets/mini_pacman",
        RUNS / "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base"
             / "two_hidden/mini_pacman",
        note="extended budget; baselines matched at 2M",
    ),
)


@dataclass
class EnvReport:
    key: str
    files: int = 0
    bytes_copied: int = 0
    missing: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(path: Path) -> bool:
    """True if this file is a result artefact worth consolidating."""
    if path.name in SKIP_NAMES:
        return False
    if path.name in KEEP_NAMES:
        return True
    return path.suffix == ".csv" and "learning_curves" in path.parts


def copy_tree(
    source: Path, destination: Path, key_root: Path, report: EnvReport, dry_run: bool
) -> None:
    """Copy the wanted files under *source* into *destination*.

    Checksums are keyed by the copied file's path relative to *key_root* (the
    consolidated output root), so the manifest reads as a map of the tree it
    describes rather than of absolute paths. Only the JSON files are hashed:
    they carry every number that reaches the paper, whereas hashing the CSVs
    would mean re-reading several GB over NFS for little provenance gain.
    """
    if not source.is_dir():
        report.missing.append(str(source))
        return
    for item in sorted(source.rglob("*")):
        if not item.is_file() or not wanted(item):
            continue
        target = destination / item.relative_to(source)
        report.files += 1
        report.bytes_copied += item.stat().st_size
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        if target.suffix == ".json":
            report.checksums[str(target.relative_to(key_root))] = sha256(target)


def collect_environment(
    environment: EnvSource, output_root: Path, dry_run: bool
) -> EnvReport:
    report = EnvReport(key=environment.key)
    env_out = output_root / environment.key
    for seed in EXPECTED_SEEDS:
        seed_out = env_out / f"seed{seed}"

        # Baselines: keep the per-method subdirectory layout, but only copy the
        # four method directories plus the renamed root summary.
        baseline_seed = environment.baseline_root / f"seed{seed}"
        if not baseline_seed.is_dir():
            report.missing.append(str(baseline_seed))
        else:
            for method in BASELINE_METHOD_DIRS:
                copy_tree(
                    baseline_seed / method, seed_out / method, output_root,
                    report, dry_run,
                )
            root_summary = baseline_seed / "summary.json"
            if root_summary.is_file():
                report.files += 1
                report.bytes_copied += root_summary.stat().st_size
                if not dry_run:
                    seed_out.mkdir(parents=True, exist_ok=True)
                    target = seed_out / BASELINE_ROOT_SUMMARY
                    shutil.copy2(root_summary, target)
                    report.checksums[str(target.relative_to(output_root))] = sha256(target)

        # PSPO: files live at the seed root, which is where the plot scripts
        # look for the adaptive method.
        copy_tree(
            environment.pspo_root / f"seed{seed}", seed_out, output_root,
            report, dry_run,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be copied without writing")
    parser.add_argument("--environment", action="append", dest="environments",
                        choices=[e.key for e in ENVIRONMENTS],
                        help="restrict to these environments (repeatable)")
    args = parser.parse_args(argv)

    selected = [
        e for e in ENVIRONMENTS
        if not args.environments or e.key in args.environments
    ]

    output_root = args.output_dir
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    reports: list[EnvReport] = []
    for environment in selected:
        report = collect_environment(environment, output_root, args.dry_run)
        reports.append(report)
        size_mb = report.bytes_copied / (1 << 20)
        status = "OK" if not report.missing else f"{len(report.missing)} MISSING"
        print(f"{environment.key:<20} {report.files:>5} files  {size_mb:>8.1f} MB  {status}")
        for path in report.missing:
            print(f"    missing: {path}", file=sys.stderr)

    total_files = sum(r.files for r in reports)
    total_mb = sum(r.bytes_copied for r in reports) / (1 << 20)
    print(f"\n{'TOTAL':<20} {total_files:>5} files  {total_mb:>8.1f} MB")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "Consolidated MASA results. Each <env>/ directory serves as both the "
            "baseline_root and adaptive_root of an EnvironmentSpec: baseline "
            "methods under their method subdirectories, PSPO at the seed root."
        ),
        "expected_seeds": list(EXPECTED_SEEDS),
        "baseline_methods": list(BASELINE_METHOD_DIRS),
        "baseline_root_summary_renamed_to": BASELINE_ROOT_SUMMARY,
        "environments": [
            {
                "key": e.key,
                "label": e.label,
                "budget_timesteps": e.budget,
                "source_baseline_root": str(e.baseline_root.relative_to(REPO)),
                "source_pspo_root": str(e.pspo_root.relative_to(REPO)),
                "note": e.note,
                "files_copied": r.files,
                "bytes_copied": r.bytes_copied,
                "missing_sources": r.missing,
                "sha256": r.checksums,
            }
            for e, r in zip(selected, reports)
        ],
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 1 if any(r.missing for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
