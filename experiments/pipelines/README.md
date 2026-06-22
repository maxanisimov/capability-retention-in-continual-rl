# `experiments/pipelines/` conventions

Each subdirectory is one Gymnasium environment's continual-learning pipeline (source training +
downstream adaptation). `frozenlake/` and `lunarlander/` are the reference implementations; new or
updated pipelines should follow their structure.

## Layout

- `core/env/` — environment factory and task-loading logic.
- `core/methods/` — `source_train.py`, `adapt_<method>.py` (the real training/adaptation code).
- `core/orchestration/` — `run_paths.py` (default path resolution), `launch_multi_seed.py`
  (CPU-pinned multi-seed runner), `launch_full_pipeline_multi_seed.py` (job-graph orchestrator).
- `cli/` — thin CLI entrypoints invoked as subprocesses by the orchestration layer.
- Top-level `train_source_policy[_multi_seed].py`, `downstream_adaptation_<method>[_multi_seed].py`
  — backward-compatible wrapper scripts that just import and call into `core/`.
- `_shared/` — code genuinely identical across pipelines (e.g. the multi-seed subprocess scheduler,
  `neutralize_task_feature`). Only extract here when behavior is verified identical across all
  call sites; prefer leaving pipeline-specific logic (env factories, CLI arg shapes) local.

## Naming conventions

- Entrypoints: `train_source_policy.py` / `train_source_policy_multi_seed.py`,
  `downstream_adaptation_<method>.py` / `downstream_adaptation_<method>_multi_seed.py`.
- Aggregation/analysis scripts: `aggregate_<metric>_<scope>.py`.
- Superseded results or old pipeline versions go under `experiments/archive/`, mirroring their
  original relative path — not ad hoc `_PREV`/`v1`/`v2` suffixes next to the live path.
- A pipeline with a real implementation gets the full `core/` + `cli/` split; an env-only stub
  (no training pipeline yet) keeps just `core/env/`.
