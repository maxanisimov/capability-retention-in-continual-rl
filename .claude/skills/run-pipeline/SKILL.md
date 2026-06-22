---
name: run-pipeline
description: Launch an experiment pipeline under experiments/pipelines/<behaviour_retention|safety>/<env>/. Use when the user asks to run, train, or launch a FrozenLake/LunarLander/LavaCrossing/etc. experiment pipeline, or a multi-seed sweep.
disable-model-invocation: true
---

Run a pipeline under `experiments/pipelines/`. Pipeline name/env/family/flags come from `$ARGUMENTS`.

1. Resolve the pipeline directory: `experiments/pipelines/<family>/<env>/`, where `<family>` is `behaviour_retention` (preserving source-task behaviour) or `safety` (shield-generated safety datasets). If `$ARGUMENTS` doesn't name a family, check which of `experiments/pipelines/behaviour_retention/<env>/` and `experiments/pipelines/safety/<env>/` exist; if both exist, ask the user which one they mean.

2. **Read that pipeline's `README.md` before running anything.** Exact CLI flags (`--mode`, `--pipeline`, `--seed`, `--shield-type`, etc.) differ per pipeline and per family — do not assume flags from one pipeline's README apply to another. The root `README.md`'s "Running Main Experiments" section is stale and must not be used as a reference.

3. For a single run, use the pipeline's `run_experiment.py` (or pipeline-specific script like `run_safe_line_search_ppo.py` / `run_lagrangian_ppo.py`) with flags exactly as documented in that pipeline's README.

4. For multi-seed runs, check the pipeline's `cli/` directory for launchers such as `launch_full_pipeline_multi_seed.py` or `launch_adaptation_multi_seed.py`, and follow the usage example in the README (these usually accept `--seeds` and `--cores`, pinning one seed per CPU core).

5. Ensure the project venv is active (`source .venv/bin/activate` from the repo root) and run all commands from the repository root.

6. After the run, report where outputs landed — check the pipeline's README for its `artifacts/`/`outputs/` location, and surface any `results_table.csv` or summary metrics if produced.
