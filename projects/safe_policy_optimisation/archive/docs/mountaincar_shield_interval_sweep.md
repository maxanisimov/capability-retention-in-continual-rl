# MountainCar PSPO negative-velocity shield sweep

Six nested position intervals, each with seeds 0–9: `[-1.2, u]`, where
`u = -1.15, -1.10, -1.05, -1.00, -0.95, -0.90`. The existing MountainCar
shield permits only push right (action 2) inside the interval when `v < 0`.
All actions are admissible otherwise. The complete certificate uses the
conservative closed velocity interval `[-0.07, 0]`; it is not a sampled audit.

Every interval/seed receives a separately fitted safe initial actor: two tanh
hidden layers of width 64, 4,096 interval-box BC samples and 200 BC epochs,
followed by at most 2,000 IBP refinement epochs (Adam lr 0.01, margin 2).
There is no reward training or warm start in the initialisation. Comparisons
therefore measure the full shield-specific pipeline, not just a change of
shield around identical initial weights.

Production runs match the recent MountainCar configuration: 400,000 requested
steps (400,384 with full rollouts), lr 0.001, rollouts 1,024, minibatches 128,
ten epochs, gamma 0.99, GAE 0.95, clip 0.2, entropy 0, shaped training reward.
PSPO is region-first, directional orthotope, replace mode, every train phase,
200 growth iterations, weighted-width objective, logsumexp surrogate, IBP
growth and certification. Early stopping is disabled.

Final evaluation is the nominal greedy actor **without runtime shielding**,
100 episodes per pair using reset seeds `training_seed + 10000 + episode`.
Native unshaped episode reward and wall-avoidance safety are reported.
Safety is the fraction of episodes that never touch the physical left wall;
action compliance is reported separately. Certification of an action rule
does not prove wall avoidance. The trainer completes final safety enforcement
and repeats full-box certification before saving and evaluating the actor.
Do not substitute intermediate learning-curve entries for the final results.

## Launch and monitoring

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_mountaincar_shield_interval_sweep.py --launch
```

The command prints a unique timestamped output root and detached screen name.
The screen controller samples CPU idleness over five seconds, admits a physical
core only if every SMT sibling is at least 95% idle, pins each concurrent worker
to one distinct CPU, and sets numerical threads to one. A worker keeps that core
for initialisation, training, and evaluation. Excess pairs queue and all six
intervals are retained. No unrelated processes are changed. Idleness is an
admission-time observation, not an exclusive OS reservation against other users.

`experiment.json` records settings, input/source hashes, dirty git status, and
the source ZIP. `launch_manifest.json` records CPU samples and job assignments;
each pair has `status.json`, `commands.json`, `initialisation/`, `training/`,
and a validated `result.json`. `_logs/` contains per-pair logs and screen output.
The controller updates `per_seed.csv`, `aggregate.json`, and `report.md` after
each completion, reporting mean ±2 standard errors across completed seeds.
Incomplete and failed results are explicitly labelled. Completed runs must
pass certificate, fixed-budget, base-policy hash, evaluation, and independent
wall-audit checks before entering the report.

```bash
screen -ls
screen -r SCREEN_NAME
.venv/bin/python projects/safe_policy_optimisation/scripts/run_mountaincar_shield_interval_sweep.py \
    --report-only --output-root OUTPUT_ROOT
```

Use `--launch --smoke` for two endpoint intervals on seed 0, one 1,024-step
rollout, one PPO epoch, two growth iterations, and two final evaluation episodes;
initialisation and full-box verification remain unchanged. Use
`--max-concurrent N` to cap concurrency without discarding pairs.

A controller lock prevents duplicate dispatch. Existing roots are refused
unless `--launch --resume --output-root OUTPUT_ROOT` is used explicitly; resume
only dispatches untouched pairs and refuses any running/interrupted statuses.
Failed/existing pair directories are never overwritten. Source/input changes
fail closed; use a fresh root for a changed experiment.
