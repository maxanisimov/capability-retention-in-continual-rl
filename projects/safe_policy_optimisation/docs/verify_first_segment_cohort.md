# Verify-first PSPO with certified line segments

`scripts/run_pspo_verify_first_segment.py` prepares and supervises 60 parallel
jobs: the six main paper environments and seeds 0–9. Each worker is pinned to
its own CPU, with numerical-library threads set to one. CPU selection samples
five seconds of host idle time and requires at least 90% idle, reserving CPU
IDs 0–7 outside the pool.

The launcher reads each existing `segment_lid/two_hidden/ENV/seedN/config.json`,
reuses the exact initial actor named there, and changes `verify_first` to true.
Segment geometry, tolerance/splits, PPO hyperparameters, timestep budget,
enforcement cadence, shield, and evaluation protocol are retained. No lookup
mode is enabled and no new initial policies are fitted. Verify-first uses
`AdaptiveSafePPO`: verify the proposed greedy actor, accept it if safe, and
otherwise search/project onto a certified segment from the previous safe actor.

The full nominal budgets are 25k steps for Media Streaming and Colour Bomb v1,
100k for Colour Bomb v2, 200k for Bridge Crossing v1, 1.6M for Bridge Crossing
v2, and 2M for MiniPacman. Final evaluation is greedy and unshielded, using 100
episodes per seed. Enforcement is every rollout except MiniPacman, where it
is every 100 rollouts (with final enforcement of the remainder).

Prepare on the actual host, using a new run name:

```bash
MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_pspo_verify_first_segment.py \
  --run-name pspo_verifyfirst_segment_masa_TIMESTAMP
```

Then run the supervisor in a detached screen session:

```bash
screen -dmS pspo-verifyfirst-segment-TIMESTAMP \
  -L -Logfile projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/pspo_verifyfirst_segment_masa_TIMESTAMP/_orchestrator/screen.log \
  env MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_pspo_verify_first_segment.py \
  --supervise projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/pspo_verifyfirst_segment_masa_TIMESTAMP/_orchestrator/launch_manifest.json
```

The cohort directory contains `status.json` and `_orchestrator/launch_manifest.json`.
The manifest records all commands, CPU assignments, source configurations,
initial-policy hashes, shield hashes, and relevant source hashes. Worker logs
are under `_orchestrator/ENV_seedN.log`; environment completion logs retain
the same format as the existing variant plot inputs. Every seed directory
under `two_hidden/ENV/seedN` stores its model, metrics, configuration, training
curves, `process.json` (PID/CPU), and `runtime.json` (RL-process elapsed time).
Runtime excludes initialisation, which is reused, but includes process setup,
training, evaluations, and artifact saving.

The launcher refuses an existing output directory. It does not automatically
retry or resume failures. `--smoke` creates six separate eight-step jobs and
must never be used for paper results. Detached screen launches must execute
on the host, not in an ephemeral sandbox. Historical experiments are read-only.
