# Experiment and plotting scripts

Active scripts are grouped by their main use:

| Directory | Contents |
| --- | --- |
| [`experiments/`](experiments/) | 28 training, evaluation and orchestration scripts, including the `lab_cluster/` launchers |
| [`plots/`](plots/) | 25 figure, table, rendering and result-analysis scripts, including shared plotting helpers |
| [`archive/`](archive/README.md) | 29 retired scripts preserved for reference |

The [experiment script map](../docs/experiment_script_map.md) lists the entry
points and dependencies for each experiment and figure. The
[launch command tables](../docs/launch_commands.md) give commands for the
experiments, plots, analyses and cluster tools. The
[running guide](../docs/running_experiments.md) describes experiment setup.

Run commands from the repository root using the new paths, for example:

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/experiments/launch_pspo_multi_env.py --help
.venv/bin/python projects/safe_policy_optimisation/scripts/plots/plot_masa_learning_curve_comparisons.py --help
```

`compare_pspo_initial_final_rewards.py` is in `experiments/` because it runs
policy evaluations. `run_pspo_lid_geometry_comparison.py` is also there because
it trains the policies it compares. Analyses of saved results, such as
`analyse_pspo_four_ablations.py`, belong to `plots/`.

Python imports use the full package path under
`projects.safe_policy_optimisation.scripts.experiments` or
`projects.safe_policy_optimisation.scripts.plots`. Experiment runners may still
import shared method/environment definitions from plot modules. Experiment
data, checkpoints and figure output directories retain their existing paths.

Active PSPO training defaults to line-segment geometry with region-first
enforcement. Use `--safe-region-shape orthotope` to select orthotope training
and `--verify-first` (or `--verify-first true` on the MASA launcher) to select
proposal verification first. The shell launcher accepts `SAFE_REGION_SHAPE`
and `VERIFY_FIRST`. The inference benchmark uses `--pspo-variant orthotope`.
The fixed-LID/directional-growth ablations explicitly require orthotope mode;
historical plotting cohorts and archived scripts retain their provenance.

`run_seed_experiments.py` defaults to the standard baseline groups, including
plain PPO. Its historical precomputed `rashomon` group is opt-in; use
`launch_pspo_multi_env.py` for active PSPO training with the new geometry default.
