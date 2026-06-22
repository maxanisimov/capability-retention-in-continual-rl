# FrozenLake Pipeline

This pipeline is organized around reusable implementation modules and thin command entrypoints, matching the LunarLander pipeline layout.

## `core/` vs `cli/`

- `core/`: implementation modules for environment construction, training/adaptation, evaluation, analysis, and orchestration.
- `cli/`: thin command entrypoints for direct script execution.
- Top-level legacy scripts, such as `train_source_policy.py`, remain compatibility wrappers.

## Canonical Settings Locations

- Task pipelines: `settings/tasks/task_pipelines.yaml`
- Task definitions: `settings/tasks/task_definitions.yaml`
- Source environment maps: `settings/tasks/source_envs.yaml`
- Downstream environment maps: `settings/tasks/downstream_envs.yaml`
- Source training settings: `settings/source/train_source_policy_settings.yaml`
- Adaptation settings (PPO): `settings/adaptation/ppo.yaml`
- Adaptation settings (EWC): `settings/adaptation/ewc.yaml`
- Adaptation settings (Rashomon): `settings/adaptation/rashomon.yaml`

Legacy settings filenames in `settings/` are preserved as symlink shims.

## Preferred Run Commands

Run from repository root.

Single source run:

```bash
python experiments/pipelines/trajectory_retention/frozenlake/run_experiment.py \
  --mode source \
  --pipeline diagonal_30x30 \
  --seed 0
```

Single downstream run:

```bash
python experiments/pipelines/trajectory_retention/frozenlake/run_experiment.py \
  --mode downstream_ewc \
  --pipeline diagonal_30x30 \
  --seed 0
```

Generic multi-seed launcher:

```bash
python experiments/pipelines/trajectory_retention/frozenlake/cli/launch_multi_seed.py \
  --mode downstream_ewc \
  --pipeline diagonal_30x30 \
  --seeds 0 1 2 3
```

Full source-plus-downstream launcher:

```bash
python experiments/pipelines/trajectory_retention/frozenlake/cli/launch_full_pipeline_multi_seed.py \
  --pipeline diagonal_30x30 \
  --seeds 0 1 2 3
```

`--pipeline` and `--layout` are aliases. New runs default to `artifacts/runs`; readers and launchers still resolve existing `outputs/` runs and legacy `source` policy directories.

