# `projects/safe_crl/pipelines/` conventions

Pipelines are organized by **what they preserve under continual adaptation**, then by
**environment**:

- `safety_retention/` — preserves a hard safety property (e.g. action masks that avoid
  FrozenLake holes) while adapting to a new task.
- `trajectory_retention/` — preserves the previously-learned greedy trajectory/behavior on the
  source task (EWC/Rashomon-style continual learning), no safety constraint.
- `envs/` — environment-only stubs (just `core/env/`, no `core/methods/` yet) that haven't
  committed to either retention category because no adaptation method has been implemented for
  them yet.

Within each category, each subdirectory is one Gymnasium environment's pipeline (source
training + downstream adaptation). `trajectory_retention/frozenlake/` and
`trajectory_retention/lunarlander/` are the reference implementations; new or updated pipelines
should follow their internal structure.

A bare environment name (e.g. `frozenlake`) is not guaranteed unique across categories — scripts
that take a `--pipeline` name should accept (and tools that print errors should suggest) the
qualified `<category>/<name>` form (e.g. `safety_retention/frozenlake`) to disambiguate.

## Layout (within a pipeline directory)

- `core/env/` — environment factory and task-loading logic.
- `core/methods/` — `source_train.py`, `adapt_<method>.py` (the real training/adaptation code).
- `core/orchestration/` — `run_paths.py` (default path resolution), `launch_multi_seed.py`
  (CPU-pinned multi-seed runner), `launch_full_pipeline_multi_seed.py` (job-graph orchestrator).
- `cli/` — thin CLI entrypoints invoked as subprocesses by the orchestration layer.
- Top-level `train_source_policy[_multi_seed].py`, `downstream_adaptation_<method>[_multi_seed].py`
  — backward-compatible wrapper scripts that just import and call into `core/`.
- `_shared/` (sibling of `safety_retention/`, `trajectory_retention/`, `envs/`) — code genuinely
  identical across pipelines regardless of category (e.g. the multi-seed subprocess scheduler,
  `neutralize_task_feature`). Only extract here when behavior is verified identical across all
  call sites; prefer leaving pipeline-specific logic (env factories, CLI arg shapes) local.

## Settings convention

Every `settings/tasks/*.yaml` entry should declare:

- `deterministic: true|false`
- `dynamics: {...}` — transition-dynamics params (FrozenLake grid/layout, LunarLander
  gravity/engine power/mass scales).
- `stochasticity: {...}` — only when `deterministic: false` (e.g. LunarLander wind/turbulence/
  action-noise params).

These are additive to whatever flat keys a pipeline's loader already reads.

## Naming conventions

- Entrypoints: `train_source_policy.py` / `train_source_policy_multi_seed.py`,
  `downstream_adaptation_<method>.py` / `downstream_adaptation_<method>_multi_seed.py`.
- Aggregation/analysis scripts: `aggregate_<metric>_<scope>.py`.
- Superseded results or old pipeline versions go under `projects/safe_crl/archive/`, mirroring their
  original relative path — not ad hoc `_PREV`/`v1`/`v2` suffixes next to the live path.
- A pipeline with a real implementation gets the full `core/` + `cli/` split; an env-only stub
  (no training pipeline yet) keeps just `core/env/` and lives under `envs/`.
