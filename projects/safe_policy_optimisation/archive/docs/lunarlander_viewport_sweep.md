# LunarLander PSPO viewport-shield sweep

Six shields with `m = 0.95, 0.90, 0.85, 0.80, 0.75, 0.70`, seeds 0–9 each:
only action 1 for `x ∈ [m,1]`, only action 3 for `x ∈ [-1,-m]`, all actions
otherwise. This orders the user's reversed left-band endpoints. Actions are
chosen for actual inward side-engine impulse when upright, as verified by
stepping the installed Box2D environment. The rule does not
depend on velocity or attitude and cannot guarantee containment when the craft
is tilted or carries outward momentum. No descent/attitude rules are added.

Each certificate contains two complete eight-dimensional boxes (left then
right), spanning all declared observation bounds in the other seven
coordinates. The float32 x thresholds are conservatively rounded outwards.
The validator requires the exact two complete boxes and their inward-only
action masks; no sampled certificate or partial-band coverage is permitted.

Each pair receives a fresh separately certified 8→64→64→4 tanh actor,
4,096 interval-BC samples, 200 BC epochs, then at most 2,000 IBP refinement
epochs at lr 0.01 and target margin 2. No warm start or reward pretraining.
Initialisation must pass its independent final verifier before PPO starts.

PSPO matches the existing LunarLander configuration: 500,000 requested steps
(501,760 after complete rollouts), 1,000-step episode cap, lr 0.0003,
2,048-step rollouts, minibatches 256, ten epochs, gamma 0.99, GAE 0.95,
clip 0.2, entropy 0.01, native unshaped reward, discrete actions, wind disabled.
Safety enforcement is region-first directional orthotopes, every train phase,
replace mode, 200 iterations, weighted-width objective, logsumexp surrogate,
IBP growth and certification. Early stopping is disabled. Certificates state
the greedy action constraint, not safety of stochastic nominal exploration;
training additionally uses the runtime action shield.

Final evaluation is nominal greedy without runtime shielding, 100 episodes per
pair with reset seeds `training_seed + 10000 + episode_index`. Safety measures
episodes never reaching `abs(x) >= 1`, using Gymnasium's raw normalisation
before float32 conversion and including terminal observations. An independent
audit repeats the same evaluation and checks reward, length, and safety against
the saved records. `trajectory_audit_episodes.csv` preserves termination and
truncation flags, escape, crash, and landing outcomes. No truncation is reported
separately from successful landing: escape and crashes also terminate. Landing
means the body sleeps without a crash, viewport exit, or truncation. The
existing native-reward >200 success metric is preserved and also aggregated.

## Launch

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_lunarlander_viewport_sweep.py --launch
```

Run outside the temporary process sandbox so screen persists. Outputs use a
new timestamped directory under `outputs/continuous_state_shields/pspo/`.
The detached screen controller reuses the MountainCar sweep scheduler: one
worker per dedicated idle physical core, all SMT siblings at least 95% idle
over five seconds, `taskset` single-CPU affinity inherited by both training
stages, and numerical-library thread limits of one. Excess jobs queue; no
shield or seed is dropped. Admission is not an exclusive OS reservation
against other users. Concurrency can be capped using `--max-concurrent N`.

Source/input hashes, source ZIP, configuration, dirty git status, commands,
CPU samples, assignments, logs, statuses, and validated final results are saved.
The controller automatically updates `report.md`, `aggregate.json`, and
`per_seed.csv` after completions, with mean ±2 standard errors across completed
seeds and explicit failure/completion counts. Existing roots are never
overwritten. Explicit `--launch --resume --output-root ROOT` only dispatches
untouched pairs; running/interrupted pairs require inspection before resuming.

```bash
screen -r SCREEN_NAME
.venv/bin/python projects/safe_policy_optimisation/scripts/run_lunarlander_viewport_sweep.py \
    --report-only --output-root OUTPUT_ROOT
```

`--launch --smoke --max-concurrent 2` runs the two endpoint shields on seed 0,
one rollout, one PPO epoch, two LID iterations, and two evaluation episodes.
Safe initialisation and full-box verification are unchanged in the smoke test.
Use a fresh root for source/configuration changes; failed pair artifacts are
retained and never silently replaced.

## Production launch (18 September 2026)

Output root:
`outputs/continuous_state_shields/pspo/20260918T063053132065Z_lunarlander_viewport_sweep/`

Screen session:
`pspo-ll-viewport-20260918T063053132065Z_lunarlander_viewport_sweep`

This is the 60-pair sweep with the physically verified inward action mapping.
Earlier timestamped `_smoke` directories are preflight-only, not production
results. The first preflight used reversed impulses and is excluded; the
corrected endpoint preflight and regression tests passed before this launch.
