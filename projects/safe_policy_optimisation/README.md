# Safe policy optimisation (PSPO)

Experiment pipelines and analysis for the PSPO paper. The active tree contains
PPO, CPO, PPO-Lagrangian, PPO-PID-Lagrangian, RL-SGF, PPO-Shield, PSPO and final
projection baselines, together with the paper's ablations, timing, temperature
and safety-only FrozenLake studies.

Start with the [paper inventory](docs/paper_inventory.md) for the exact scripts,
cohorts and generated assets. [Running experiments](docs/running_experiments.md)
covers the launchers. Superseded experiments and outputs live under
[`archive/`](archive/README.md).
The [cleanup validation](docs/cleanup_validation.md) records tests and numerical
reproduction checks.

## Setup and checks

Run from the repository root. Python 3.10 or newer is required.

```bash
python -m pip install -e '.[rl,viz]'
python -m unittest discover -s projects/safe_policy_optimisation/tests -p 'test_*unittest.py'
python -m unittest discover -s core/safe_rl_baselines -p 'test_*unittest.py'
```

For the existing local environment, use `.venv/bin/python`. An uninstalled
checkout can use `PYTHONPATH=core:.`; use `OMP_NUM_THREADS=1` for small tests.

## Layout

| Path | Purpose |
| --- | --- |
| `run_experiment.py` | Declarative pipeline entry point; `--list-pipelines` lists settings |
| `settings/` | Pipeline/task YAML and tracked PSPO launcher hyperparameters |
| `stages/` | Training, shield synthesis, certification and evaluation stages |
| `scripts/` | Paper study launchers, aggregation, tables and figures |
| `utils/` | Configuration, environments, logging, checkpoints and evaluation |
| `tests/` | Active experiment and analysis regression tests |
| `docs/` | Reproduction inventory, protocols and methodology |
| `notebooks/` | PSPO update-mechanism illustrations |
| `artifacts/paper_2503_07671/` | Local shields, datasets, checkpoints and run records |
| `figures/`, `results/` | Generated paper assets and analysis |
| `archive/` | Historical source, tests, documents and local output files |

The implementation of PSPO is in `core/provably_safe_policy_optimisation/`;
the native cost-constrained baselines, including RL-SGF, are in
`core/safe_rl_baselines/`. Environment/shield adapters reuse `projects/safe_crl/`
through `utils/safe_crl_bridge.py`.

`train_pspo.py` is the active PSPO stage. `train_pspo_precomputed.py` also remains
because its certification and actor-mapping helpers are shared by active
training and projected baselines. Historical `pspo_adaptive` names in retained
run directories are provenance, not separate active algorithms.

## Reproduce the main analysis

These commands read existing local experiment data; a source-only clone needs
the cohorts listed in the inventory, or new training runs with matching settings.
RL-SGF defaults resolve within this project and do not require another worktree.

```bash
python projects/safe_policy_optimisation/scripts/plot_masa_learning_curve_comparisons.py
python projects/safe_policy_optimisation/scripts/generate_masa_all_methods_table.py
python projects/safe_policy_optimisation/scripts/generate_safe_projection_comparison_assets.py
python projects/safe_policy_optimisation/scripts/plot_frozenlake_scalability.py
```

Use `--help` for output overrides. Main MASA learning-curve comparisons use
region-first PSPO-LS; the four-variant comparison explicitly distinguishes
orthotope/line-segment geometry and region-first/verify-first enforcement.
Older final-policy bar generators retain their original orthotope controls.
The inventory records this distinction so figures are not silently relabelled.

Generated checkpoints, logs, figures and tables are kept locally and ignored by
Git. Archived source and documentation are versioned; moving generated outputs
into the archive does not publish them. See the archive manifest for original
locations and restoration instructions.
