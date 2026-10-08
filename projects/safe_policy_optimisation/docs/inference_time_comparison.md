# Inference timing comparison

`scripts/run_inference_time_comparison.py` benchmarks the existing main-paper
PSPO and PPO-Shield checkpoints for six environments, across seeds 0–9.
It never retrains or overwrites the source runs. The timing-only retraining
sweep did not save final models, so its policies cannot be used here.

Each job executes exactly 1,000,000 measured environment transitions after
1,000 unmeasured warm-up transitions. Inference is deterministic and online,
with batch size one. Episodes reset on either termination or truncation;
the measured reset seed is `10000 + training_seed + episode_number`.
PPO-Shield also resets its fallback RNG after warm-up.

PSPO uses the saved actor without a runtime shield or LID computation.
PPO-Shield uses the saved actor plus the same `Shield` action override as the
canonical shield-on evaluation. Both checkpoint types are loaded using
`stable_baselines3.PPO.load` to recover their unchanged actor/critic policies
without constructing the PSPO training machinery. `predict` is inherited
unchanged by the custom classes. This benchmark does not use lookup mode.

Each process is pinned to a separate CPU, with Torch and numerical-library
thread counts set to one. Up to 120 jobs run concurrently, leaving eight CPU
IDs outside the pool. Measurements are host- and concurrent-load-dependent,
not isolated hardware latency measurements.

Recorded quantities:

- `inference_s`: cumulative wall time for `predict`, action conversion, and
  runtime shield checking/overrides, if applicable. This excludes environment
  stepping and resetting, model loading, and warm-up.
- `prediction_s`: the `predict` portion of `inference_s`.
- `shield_s`: the shield override portion; zero for PSPO.
- `rollout_s`: the whole measured rollout, including inference, environment
  steps/resets and loop overhead. Progress-file writes are excluded. Timing
  bookkeeping introduces small overhead; no overhead subtraction is applied.
- `mean_inference_us_per_step`: `inference_s / measured_steps * 1e6`.

No rewards, safety rates, trajectories, or new checkpoints are persisted.
Progress files update every 10,000 transitions. Worker exceptions are captured
in per-job JSON or worker logs. There are no automatic retries.

The supervisor creates `aggregate.json`, `aggregate.csv`, and `report.md`,
using completed seeds only and recording the sample size. Final cells require
all ten seeds. Uncertainty is twice the sample standard deviation divided by
the square root of the number of seeds, not uncertainty across individual
transitions.

Example launch (the output directory must not already exist):

```bash
MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib .venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_inference_time_comparison.py \
  --output-dir projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/inference_main_TIMESTAMP \
  --launch
```

Detached launches must execute on the actual host, not inside an ephemeral
sandbox. Read `status.json` and per-job `progress.json` to monitor the run.
