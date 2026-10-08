# Fresh timing comparison and PSPO lookup sweep

The dedicated runner reproduces the six-environment main comparison's nominal
budgets and saved per-seed hyperparameters. It runs PPO, PPO-Lagrangian,
PPO-PID-Lagrangian, CPO, PPO-Shield, PSPO (one-hot), and PSPO (state-ID lookup)
independently across seeds 0–9: 420 training jobs. Twelve additional jobs freshly
fit a reward-free PSPO base policy once per environment and representation, with
initialisation seed 0. Each base is shared across its ten RL seeds.

| Environment | Nominal training steps |
|---|---:|
| Media Streaming | 25,000 |
| Colour Bomb v1 | 25,000 |
| Colour Bomb v2 | 100,000 |
| Bridge Crossing v1 | 200,000 |
| Bridge Crossing v2 | 1,600,000 |
| MiniPacman | 2,000,000 |

Two 64-unit hidden layers, CPU execution, and the canonical orthotope,
region-first PSPO configuration are retained. Each worker is pinned to one core
and Torch/BLAS use one thread. Eight cores are reserved. Pending jobs are queued
when the requested concurrency exceeds capacity; no CPU oversubscription is used.

## Output policy

Every training job writes `training_time.json`. Only lookup PSPO additionally
writes `metrics.json`, containing final deterministic evaluation mean total
reward, safety rate, and episode count (100 for full runs).

Normal trainer JSON/CSV/TensorBoard outputs, episode histories, model checkpoints,
and console output are suppressed rather than written and subsequently deleted.
Periodic and final evaluations still execute for matched workloads. Their results
are not persisted for the six standard methods. Accordingly `training_loop_s`
includes periodic evaluation, but excludes final evaluation outside `learn()`.

Initialisation retains only the base parameters required by training, a minimal
timing/architecture summary, and its timing record. Behaviour cloning uses only
the supplied safe-action mask; no environment reward samples or rollouts are used.
No initialisation outcome metrics are stored in the base-policy payload.

## Timing boundaries

- `training_loop_s`: exact `learn()` duration; contains nested safety/LID work.
- `rl_stage_s`: setup, training, evaluations, and in-memory finalisation.
- `process_wall_time_s`: subprocess launch through exit, captured by a dedicated
  wait thread, not by the supervisor's polling interval. Scheduling wait is excluded.
- `policy_initialisation_s`: complete reward-free initialisation inside its stage.
- `behaviour_cloning_fit_s`: fitting alone inside initialisation.
- `initialisation_process_s`: full shared initialisation process, referenced by RL runs.
- `cold_single_seed_process_s`: shared initialisation process plus one RL process
  for PSPO; just the method process for other methods. Do not sum this across ten
  PSPO seeds: the shared initialisation cost is incurred once, not ten times.
- `lid_s`: cumulative complete LID computations, including any final adaptive
  update. Engine and calibration subtimers and computation count are also recorded.
  LID times are already inside training/stage/process time; do not add them again.
- Verification, projection, safety enforcement, and adaptive finalisation are
  separately recorded when available; these can overlap other timing boundaries.

Lookup changes state representation and its associated first-layer implementation,
not the network width/depth or PSPO settings. Native and lookup bases are freshly
fitted to the same safety/entropy objectives; no speedup is assumed in advance.
Concurrent workloads still affect observed wall times, even on the same host.

## Launch and monitor

Use a new output directory; existing runs are never overwritten. Launch outside
an ephemeral sandbox so the detached supervisor survives the launching shell:

```bash
MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib .venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_timed_main_comparison.py \
  --output-dir projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/timing_main_NEW_RUN \
  --max-parallel 120 --launch
```

`manifest.json` records all jobs, dependencies, original configuration sources,
CPU allocation, and timing definitions. `status.json` records running, pending,
completed, failed, and dependency-blocked jobs. `supervisor.json` records its PID;
`supervisor.log` contains only job completion status and process durations.
Failures are recorded in the corresponding timing JSON. A failed initialisation
blocks only its dependent PSPO jobs. This launcher deliberately does not silently
retry or resume, so failed attempts cannot be confused with completed timings.

For a quick protocol test, select `--environments media_streaming --seeds 0
--smoke --max-parallel 4`. Smoke uses eight training steps and two final episodes,
and is explicitly marked in its manifest; its timings are not paper results.
