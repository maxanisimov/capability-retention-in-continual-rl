# Plain PPO on FrozenLake 128×128 and 256×256

`scripts/run_frozenlake_ppo_sweep.py` prepares a new cohort containing both
sizes and seeds 0–9. One detached screen supervisor launches all 20 workers
simultaneously on distinct idle physical cores, with numerical-library thread
counts set to one. Cores 0–7 are reserved; CPU availability is sampled for five
seconds, requiring every SMT sibling to average at least 95% idle.

The runs reuse the validated layouts, slip probabilities, reward definition,
training budget and PPO hyperparameters from the completed safety-only PSPO
lookup cohorts. They use **plain SB3 PPO**, not PPO-Shield: standard random
actor/critic initialisation, native discrete observations with ordinary SB3
one-hot preprocessing, and no action masking during training or evaluation.
The supplied shield mask only audits unsafe action proposals. No safe actor,
goal-directed witness or reward shaping is used to initialise PPO.

Each seed has a nominal budget of 2,000,000 environment steps, two 64-wide tanh
hidden layers, rollout length 2,048, minibatch size 64, ten epochs, learning
rate 0.0003 and discount 0.999. The episode limits are 5,000 steps for 128×128
and 10,000 for 256×256. Final evaluation uses 100 unshielded greedy episodes;
learning curves use ten episodes every 100,000 steps. Early stopping is off.

Goal attainment is recovered from return plus 0.001 × episode length, since
reaching the goal can have a negative return. Original stage metrics are kept
in `metrics_reward_threshold.json`; `metrics.json` reports actual goal success.
Models, episode CSVs, learning curves, reward and safety metrics are retained.

Prepare on the host, choosing a new run name:

```bash
env PYTHONPATH=core:. MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_sweep.py \
  --run-name ppo_frozenlake_128_256_TIMESTAMP
```

Launch that cohort in screen:

```bash
screen -dmS ppo-frozenlake-128-256-TIMESTAMP \
  -L -Logfile projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/ppo_frozenlake_128_256_TIMESTAMP/_orchestrator/screen.log \
  env PYTHONPATH=core:. MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_sweep.py \
  --supervise projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/ppo_frozenlake_128_256_TIMESTAMP/_orchestrator/launch_manifest.json
```

The cohort's `status.json` tracks completion; `_orchestrator/` holds the screen
and per-worker logs, supervisor PID, and manifest with commands, CPUs and input
and source hashes. `source_snapshot.zip` preserves the launcher, worker, PPO
stage and environment implementation used for this cohort. Each
`frozenlakeSIZE/ppo/seedN/` holds normal stage outputs, `status.json`,
`process.json` and `runtime.json`. The latter measures full worker-process
elapsed time, including startup, training, evaluations and saving; it is not
a training-only timer. Each size has `aggregate.json` and `per_seed.csv`,
updated after worker completions, with mean ± two standard errors.

`--smoke` prepares only one 64-step job per size, with 128-step episode limits
and two evaluation episodes, under separate `_smoke/` directories. Smoke
outputs must never be used as paper results. The launcher refuses existing
cohort directories and does not automatically retry or resume failed runs.
Run detached screen sessions on the host, not inside an ephemeral sandbox.
