# Running PSPO paper experiments

Run commands from the repository root after the [setup](../README.md#setup-and-checks).
The [launch command tables](launch_commands.md) cover the revamped
`scripts/experiments/` and `scripts/plots/` entry points.
The [paper inventory](paper_inventory.md) identifies the source cohorts for the
reported comparisons. Use new output names when launching additional studies.

## Pipelines and baseline stages

```bash
python projects/safe_policy_optimisation/run_experiment.py --list-pipelines
python projects/safe_policy_optimisation/run_experiment.py --pipeline paper_2503_07671_colour_bomb
python projects/safe_policy_optimisation/scripts/run_seed_sweep.py --pipeline deterministic_minipacman --n-seeds 3 --dry-run
```

The pipeline runner reads `settings/<group>/{pipelines,tasks}.yaml`. The seed
sweep shares initial inputs, schedules seeds and aggregates metrics. Stage CLIs
accept `--help`; `train_ppo.py`, `train_cpo.py`, `train_ppo_lagrangian.py`,
`train_rl_sgf.py` and `train_ppo_shield.py` expose the individual baselines.
RL-SGF is opt-in in `experiments/run_baselines_pspo_init_ablation.py` and
`experiments/run_safe_initialised_baseline_projection.py` via `--methods rl_sgf`.
New training uses PPO proposals followed by an SGF constraint by default; select
`--rl-sgf-update-mode reinforce` for the original updates. See
[RL-SGF update modes](rl_sgf_update_modes.md) for sampling budgets and legacy compatibility.

RL-SGF's safe-initialised and final-projected cohorts use the same static LID
inputs as the other projected methods. Its cost budget and gradient-flow
settings must be specified for the intended protocol; see `train_rl_sgf.py
--help`. Adding it to the registry does not change the existing FrozenLake
shaping study's supported cost-based methods.

## PSPO launcher

`experiments/run_pspo_one_env.sh` resolves hyperparameters from the tracked
`settings/pspo_launcher_hyperparameters.json` plus `utils/pspo_defaults.py`.
This replaces the former dependency on ignored hyperparameter-search reports.
`CPU_IDS` must name available CPUs, one per requested seed.

```bash
ENV_NAME=colour_bomb SEEDS=0 CPU_IDS=8 DRY_RUN=1 \
  RUN_NAME=paper_preview SAFE_REGION_SHAPE=segment \
  bash projects/safe_policy_optimisation/scripts/experiments/run_pspo_one_env.sh
```

Choose a CPU allowed by your host before running. `DRY_RUN=1` prints commands;
remove it to train. Line segments are the default. `SAFE_REGION_SHAPE=orthotope` selects the orthotope variant;
`VERIFY_FIRST=true` enables verify-first enforcement. Other useful overrides
are `TOTAL_TIMESTEPS`, `ADAPTIVE_FREQ`, `BASE_POLICY_PATH`, `OUTPUT_BASE` and
`SKIP_EXISTING`. The launcher uses the repository's `.venv/bin/python`.

For coordinated runs see `experiments/launch_pspo_multi_env.py`, `run_pspo_seed_experiments.py`,
`experiments/run_pspo_lid_geometry_comparison.py` and the dedicated
[verify-first segment cohort protocol](verify_first_segment_cohort.md).

## Paper studies

| Study | Launcher / protocol |
| --- | --- |
| Initialisation/update ablations | `experiments/run_pspo_four_ablations.py` (default segment; `--safe-region-shape orthotope` for fixed-LID/directional-growth controls); [protocol](pspo_four_ablations.md) |
| Line-segment initialisation ablations | `experiments/run_pspo_segment_init_ablations.py` (`--cpu-ids`, `--dry-run`) |
| Safe initialisation + final projection | `experiments/run_safe_initialised_baseline_projection.py` |
| Policy initialisation experiments | `experiments/run_policy_initialisation_ablations.sh`; [launch notes](pspo_policy_initialisation_launch.md) |
| Temperature | `experiments/run_temperature_sweep.py`; `plots/plot_temperature_sweep.py` |
| Training time | `experiments/run_timed_main_comparison.py`; [protocol](timed_main_comparison.md) |
| Inference time | `experiments/run_inference_time_comparison.py`; [protocol](inference_time_comparison.md) |
| FrozenLake safety-only initialisation | `experiments/run_stochastic_frozenlake_pspo.py`; [actor construction](methodology/frozenlake_analytical_actor_construction.md) |
| FrozenLake shaped PPO/PSPO and scalability | `experiments/run_frozenlake_shaping_sweep.py`, `experiments/run_frozenlake_segment_frequency_sweep.py`; [protocol](frozenlake_reward_shaping.md) |

The current FrozenLake protocol constructs the actor from the safety mask;
reward shaping applies during training and reported evaluation uses raw reward.
Historical goal-aware actor experiments are documented in
[`archive/docs/stochastic_frozenlake128.md`](../archive/docs/stochastic_frozenlake128.md).

## Outputs and provenance

Retain each run's configuration, seed, initial policy, shield, final checkpoint,
metrics and learning curves together. Timing reports distinguish wall time,
training intervals and inference costs; they must not be interchanged.
Generated data are ignored by Git and are not supplied by a fresh clone.
Some main baseline and ablation inputs remain at repository-root `outputs/`
and `artifacts/`; the inventory lists those dependencies.
