# LunarLander Directory Reorganization Report (2026-04-20)

## Scope
This report documents the implementation of the six requested migration steps for:

- `experiments/pipelines/lunarlander`

## Completed Work
1. Centralized output/path logic:
- Added `core/orchestration/run_paths.py`.
- Centralized:
  - canonical and legacy outputs roots
  - settings file defaults
  - seed run directory resolution
  - source checkpoint resolution with legacy fallback
  - policy directory resolution with legacy fallback
- Updated core entrypoints to use these helpers instead of hard-coded paths.

2. Moved shared env/task-loading logic out of `train_source_policy.py`:
- Added `core/env/wrappers.py` with:
  - `AppendTaskIDObservationWrapper`
  - `ActionRepeatWrapper`
  - `ActionDelayWrapper`
  - `ActionNoiseWrapper`
  - `LunarLanderCrashSafetyWrapper`
- Added `core/env/task_loading.py` with:
  - `load_task_settings`
  - `resolve_lunarlander_dynamics`
  - backward-compatible alias names
- Added `core/env/env_factory.py` with:
  - `make_lunarlander_env`
  - fallback behavior for unsupported gym kwargs
  - backward-compatible alias name

3. Converted script layout into thin wrappers:
- Added core implementation modules:
  - `core/methods/source_train.py`
  - `core/methods/adapt_unconstrained.py`
  - `core/methods/adapt_ewc.py`
  - `core/methods/adapt_rashomon.py`
  - `core/eval/evaluate_policy.py`
  - `core/eval/rollout_video.py`
  - `core/analysis/aggregate_layout_metrics.py`
  - `core/orchestration/run_experiment.py`
- Added new `cli/` wrappers:
  - `train_source.py`
  - `adapt_unconstrained.py`
  - `adapt_ewc.py`
  - `adapt_rashomon.py`
  - `evaluate_policy.py`
  - `rollout_video.py`
  - `aggregate_layout_metrics.py`
  - `run_experiment.py`
  - `launch_multi_seed.py`
- Replaced legacy top-level scripts with compatibility wrappers that delegate to core modules.

4. Replaced duplicate multi-seed launchers:
- Added generic launcher:
  - `core/orchestration/launch_multi_seed.py`
- Legacy multi-seed script names are now compatibility wrappers that call the generic launcher with fixed mode:
  - `train_source_policy_multi_seed.py`
  - `downstream_adaptation_unconstrained_multi_seed.py`
  - `downstream_adaptation_ewc_multi_seed.py`
  - `downstream_adaptation_rashomon_multi_seed.py`

5. Moved config files into grouped settings folders with compatibility shims:
- New canonical paths:
  - `settings/tasks/task_settings.yaml`
  - `settings/source/train_source_policy_settings.yaml`
  - `settings/adaptation/ppo.yaml`
  - `settings/adaptation/ewc.yaml`
  - `settings/adaptation/rashomon.yaml`
- Legacy paths retained as symlinks:
  - `settings/task_settings.yaml`
  - `settings/train_source_policy_settings.yaml`
  - `settings/downstream_adaptation_settings_ppo.yaml`
  - `settings/downstream_adaptation_settings_ewc.yaml`
  - `settings/downstream_adaptation_settings_rashomon.yaml`

6. Moved implementation note document:
- Moved:
  - `LUNARLANDER_V4_IMPLEMENTATION.md`
- To:
  - `docs/implementation_notes.md`

## Compatibility Notes
- Legacy script filenames are preserved as wrappers.
- Legacy settings file paths are preserved via symlinks.
- Output resolution supports legacy layouts under `outputs/`.
- `tunable_lunarlander.py` remains import-compatible as a wrapper around:
  - `core/env/tunable_lunarlander.py`

## Validation Performed
- Compiled entire LunarLander subtree:
  - `python -m compileall -q experiments/pipelines/lunarlander`
- Verified launcher help for:
  - `run_experiment.py`
  - `train_source_policy_multi_seed.py`
  - `cli/launch_multi_seed.py`
- Verified `run_experiment.py --dry-run` resolves new CLI/core paths.

## Follow-up Recommendation
- Add a concise `README.md` under `experiments/pipelines/lunarlander` describing:
  - new `core/` vs `cli/` responsibilities
  - canonical settings locations
  - preferred run commands
