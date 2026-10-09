# Historical safe policy optimisation work

This directory preserves experiments outside the retained PSPO paper workflow.
The [paper inventory](../docs/paper_inventory.md) defines the active scope.

| Path | Contents |
| --- | --- |
| `scripts/`, `stages/`, `utils/`, `tests/` | Superseded adaptive/precomputed wrappers, architecture sweeps, continuous-control prototypes and their tests |
| `settings/` | Earlier ablation configurations |
| `docs/` | Historical reports, talk material, goal-aware FrozenLake results and the previous project/launcher documentation |
| `artifacts/`, `outputs/`, `figures/`, `results/` | Local historical runs and generated assets; ignored by Git |
| `backups/` | Local `.orig` patch backups; ignored by Git |

[`manifest.json`](manifest.json) records original and archived locations, reasons,
file counts and sizes for this cleanup. No experiment output was deleted.
Source files and documents are tracked, while bulk artifacts remain local.
Existing historical folders already in the archive predate this manifest.

For restoration, locate an entry's `archived` and `original` paths (both relative
to the project), check that the original location is free, then move it back.
Keep whole run directories together. Saved absolute paths in historical
configurations retain their original meaning; restore those dependencies or
supply explicit path overrides before rerunning old code. The archived programs
are research history and are excluded from the active test command. Some still
expect the old layout or data no longer present in a source-only checkout.

Goal-aware FrozenLake results, including the earlier unified final-policy
rollouts, are archived separately from the active safety-only actor protocol.
The previous `.trash/frozenlake_goal_guided_20261007T214759Z` is preserved under
`artifacts/.trash/` here. The RL-SGF source worktree was left untouched; the paper's
two required RL-SGF cohorts were copied into the active tree with matching hashes.
