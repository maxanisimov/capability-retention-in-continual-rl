# LunarLander Pipeline

This pipeline is organized around a separation between reusable implementation code and executable command entrypoints.

## `core/` vs `cli/`

- `core/`: implementation modules (training/adaptation logic, env construction, evaluation, orchestration helpers).
- `cli/`: thin command entrypoints that call into `core/` (recommended for direct script execution).
- Top-level legacy scripts (for example `train_source_policy.py`) are compatibility wrappers and still work.

## Canonical Settings Locations

- Task definitions: `settings/tasks/task_settings.yaml`
- Source training settings: `settings/source/train_source_policy_settings.yaml`
- Adaptation settings (PPO): `settings/adaptation/ppo.yaml`
- Adaptation settings (EWC): `settings/adaptation/ewc.yaml`
- Adaptation settings (Rashomon): `settings/adaptation/rashomon.yaml`

Note: legacy settings filenames in `settings/` are preserved as symlink shims for backward compatibility.

## Preferred Run Commands

Run from repository root.

Single run by mode:

```bash
python experiments/pipelines/lunarlander/run_experiment.py \
  --mode source \
  --task-setting default \
  --seed 0
```

Downstream adaptation (example):

```bash
python experiments/pipelines/lunarlander/run_experiment.py \
  --mode downstream_unconstrained \
  --task-setting default \
  --seed 0
```

Multi-seed launcher (generic):

```bash
python experiments/pipelines/lunarlander/cli/launch_multi_seed.py \
  --mode downstream_ewc \
  --task-setting default \
  --seeds 0 1 2 3
```

Optional dry run to inspect resolved command:

```bash
python experiments/pipelines/lunarlander/run_experiment.py \
  --mode source \
  --dry-run
```
