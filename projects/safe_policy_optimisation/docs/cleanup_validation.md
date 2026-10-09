# Cleanup validation — 2026-10-09

The reorganisation was checked against the local paper data before committing.

- Active project suite: **371 tests passed** (`unittest discover -s projects/safe_policy_optimisation/tests -p 'test_*unittest.py'`).
- Core safe-RL baseline suite: **38 tests passed** (`unittest discover -s core/safe_rl_baselines -p 'test_*unittest.py'`).
- Earlier RL-SGF integration check: **40 focused tests passed**, covering native RL-SGF, shared baseline wiring, checkpoints and projected baselines.
- Parsed 207 active/archived Python files and checked that active modules do not import moved modules or the archive.
- Confirmed the tracked launcher settings reproduce all **20** historical architecture/environment settings exactly. The shell launcher passes `bash -n` and a segment-PSPO dry run for Colour Bomb with one seed.
- Checked the pipeline registry and CLI help for PSPO, RL-SGF, projected baselines, segment ablations, FrozenLake shaping and inference timing.

Regenerated the following into temporary output directories:

```bash
python projects/safe_policy_optimisation/scripts/plot_masa_learning_curve_comparisons.py --output-dir /tmp/pspo-cleanup-validation/masa
python projects/safe_policy_optimisation/scripts/generate_masa_all_methods_table.py --output-dir /tmp/pspo-cleanup-validation/table
python projects/safe_policy_optimisation/scripts/generate_safe_projection_comparison_assets.py --output-dir /tmp/pspo-cleanup-validation/projection
python projects/safe_policy_optimisation/scripts/plot_frozenlake_scalability.py --output-dir /tmp/pspo-cleanup-validation/frozenlake
```

All **14 CSV outputs** matched the existing files in shape, column order and
every numeric value exactly. This includes 1,020 table seed observations,
102 table summaries, 42 projected-baseline summaries, 940 FrozenLake seed
points, 94 FrozenLake aggregate points, and all three MASA curve comparisons.
Source-path strings can change when RL-SGF data move from the worktree;
measurements and statistics did not change. Generated PDFs/PNGs completed;
no pixel-by-pixel comparison was performed.

All 204 recorded archive destinations exist and their original paths are
vacant. The output moves cover **11,771 files / 358,559,812,581 bytes**;
renames preserved device/inode identity. The two imported RL-SGF cohorts
were checked file-by-file using SHA-256. A host-process check found no running
experiment using any selected archival destination's original path. Four
FrozenLake output roots used by live jobs were retained in place.

No experiments were retrained and no running workers were stopped. This
validation checks relocation, imports, launcher wiring and reproducibility of
the retained analysis; it does not revalidate the algorithms' formal guarantees.
Archived research tests are excluded from the active suite. Bulk artifacts
remain local and ignored by Git; the manifests document their locations.
