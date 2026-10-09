# FrozenLake training-only potential shaping

Implementation: `utils/frozen_lake_reward_shaping.py`.
PPO test runner: `scripts/run_frozenlake_ppo_shaping.py`.

The opt-in Gymnasium wrapper returns
`r_train = r_raw + scale * (gamma * Phi(next_state) - Phi(state))`.
`Phi` is `1 - d / D` for reachable nonterminal cells, where `d` is the reverse
BFS distance to the goal through non-hole grid cells and `D` is the maximum
finite distance. Goals, holes, and unreachable cells have zero potential.
This is a geometry heuristic, not an optimal policy or expected stochastic
hitting-time computation. It uses no shield mask and costs remain unchanged.

Only training uses the wrapper. Evaluation uses the original environment and
reports goal attainment from the environment's success flag, never a shaped
reward threshold. Raw reward remains `goal_indicator - 0.001 * episode_length`
by default. The step penalty is not removed in this shaping-only experiment.
The PPO actor is randomly initialized with no warm start, action mask, witness
policy, or goal-dependent parameter fitting. The existing PSPO safety-only
initializer is not modified and never consumes this potential. When using this
wrapper with PSPO, construct and audit its initial actor first, then wrap only
the RL training environment; do not pass the potential to initialization.

## Episode-boundary conventions

- Goals and holes: zero next potential; retain the terminal shaping correction.
- `--timeout-mode bootstrap` (default): retain the actual next-state potential
  at a time-limit truncation and retain the truncated flag. SB3 PPO adds the
  discounted terminal value, learned in shaped coordinates, so the residual
  potential cancels. A truncated sample is not treated as a true terminal state.
- `--timeout-mode terminal`: zero next potential at a timeout and turn it into
  a true terminated transition, disabling SB3's time-limit value bootstrap.
  This is a different finite-horizon task convention. Match it in the control.

Use the learner's exact gamma. The training CSV checks the telescoping identity
`G_shaped - G_raw = scale * (gamma**T * Phi_end - Phi_start)` episode by episode.
Potential shaping preserves the underlying discounted objective with these
boundary conventions; it does not fix objectives that reward early failure
more highly than sufficiently slow successful trajectories.

## Reproduce a one-seed 16x16 comparison

Run these commands with a new output directory; existing run IDs are protected
against overwriting. They may run concurrently in separate processes.

```bash
PYTHONPATH=core:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_shaping.py \
  --size 16 --seed 0 --total-timesteps 200000 --shaping-scale 1 \
  --output-dir projects/safe_policy_optimisation/artifacts/frozenlake_shaping_example \
  --run-id shaped_seed0

PYTHONPATH=core:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_shaping.py \
  --size 16 --seed 0 --total-timesteps 200000 --shaping-scale 0 \
  --output-dir projects/safe_policy_optimisation/artifacts/frozenlake_shaping_example \
  --run-id unshaped_seed0
```

Both runs use stochastic transitions (intended direction 0.8, perpendicular
directions 0.1 each), 625-step episode caps, gamma 0.999, two 64-unit tanh layers,
rollout length 2048, 10 optimization epochs, no entropy bonus, no early stopping,
and 100 greedy, unshielded final evaluation episodes. SB3 rounds the requested
200,000 steps up to a complete rollout (200,704 actual steps). Both initial
parameter hashes are recorded so paired initialization can be checked.

Artifacts include `config.json`, source hashes/snapshot, `potential.npz`,
`training_episodes.csv` (raw/shaped and discounted returns, goals, safety, and
telescoping errors), raw evaluation curves, `episodes.csv`, `initial_metrics.json`,
`metrics.json`, `summary.json`, and `model.zip`. Timing distinguishes learning
including periodic evaluation, learning excluding periodic evaluation, and final
evaluation. This dedicated runner leaves source-locked live sweeps unchanged.

References: [Ng et al., ICML 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf);
[Grzes, AAMAS 2017](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf).

## Completed 16x16 pilot (2026-10-07)

Seed 0, 200,704 actual training steps, 100 raw greedy evaluation episodes:

| PPO variant | Raw total reward | Safety | Goal |
| --- | ---: | ---: | ---: |
| Potential shaping, scale 1 | 0.94173 | 98% | 98% |
| Matched unshaped control, scale 0 | 0.86749 | 91% | 91% |

The first successful training trajectory occurred at step 6,768 with shaping
versus 29,266 without it. Initialization parameter hashes were identical.
Training time excluding periodic evaluation was 316.51 s with shaping versus
186.04 s without it in this concurrent pilot: earlier learning in environment
steps did not translate into a shorter full run here. One seed cannot establish
across-seed uncertainty or scalability to larger maps.

Full audited results and artifacts:
[pilot report](../artifacts/paper_2503_07671/runs/ppo_shaping_frozenlake16_20261007T220300Z/README.md).

## PSPO with shaping and safety-only initialization

PSPO pilot runner: `scripts/run_frozenlake_pspo_shaping.py`. It builds the actor
only from the safety action mask, with equal logits for every safe action. The
raw PSPO model is constructed and audited before the shaping wrapper is created.
An exact actor hash check ensures that attaching shaping does not alter the
initial parameters. Goal distances, reward targets, and witness policies are
not inputs to actor initialization.

```bash
PYTHONPATH=core:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_pspo_shaping.py \
  --size 64 --seed 0 --total-timesteps 200000 --shaping-scale 1 --gamma 0.999 \
  --max-episode-steps 2500 --eval-episodes 100 \
  --output-dir projects/safe_policy_optimisation/artifacts/frozenlake_pspo_shaping_example \
  --run-id shaped_seed0
```

Defaults match the existing FrozenLake PSPO orthotope configuration: directional
region-first growth, safety enforcement every 100 rollouts, 200 LID iterations,
minibatch 256, and exhaustive certification. At 200,000 requested steps there are
only 98 rollouts, so enforcement occurs at the mandatory final flush. Intermediate
unshielded curves are **uncertified proposed policies**, not guaranteed-safe
deployment checkpoints. Increase the budget or explicitly reduce
`--safety-frequency` to study more frequent certified updates; those are different
experiment settings.

Training uses the existing PSPO runtime shield. Final evaluation uses original
rewards and greedy actions without a runtime shield. The runner certifies the
final policy on every safety-winning state before saving it, and records its
final learning-curve point only after the final pending update has been enforced.
The guarantee is for greedy actions, not arbitrary stochastic action sampling.

PSPO artifacts additionally contain the safety-only actor/mask and initialization
audit in `_inputs/`, pre-finalization raw metrics, an exhaustive greedy
`certificate.json`, and LID/safety-enforcement timing in `summary.json`.

## Ten-seed three-method screen sweep

`scripts/run_frozenlake_shaping_sweep.py` prepares 30 CPU-pinned jobs: PSPO,
PPO-Shield and PPO, each with seeds 0--9. It selects distinct idle physical cores
after checking both hyperthreads, limits each worker to one compute thread, and
refuses existing output roots. The supervisor validates source hashes and holds
a lock to prevent duplicate launches. It starts every job concurrently and
records per-job PIDs, logs, completion/failure status, and wall times.

Use `--methods ppo pspo` to select only those two methods (20 jobs), keeping
the three-method sweep as the default. For a 128x128, 400k-step comparison,
add `--size 128 --total-timesteps 400000`. Complete 2048-step rollouts give
401,408 actual steps per run, with PSPO enforcement at step 204,800 and a
mandatory final flush. The size-scaled episode cap is 5000; shaping remains
training-only and final greedy evaluation is unshielded for both methods.

For all four established safe-RL baselines, select
`--methods ppo_lagrangian ppo_pid_lagrangian cpo ppo_shield` (40 jobs across
seeds 0--9). The dedicated `run_frozenlake_safe_baseline_shaping.py` worker
uses the existing baseline factory without changing its optimization rules:
cost limit 0, cost discount 0.99, cost GAE 0.95, Lagrangian multiplier initially
0, Lagrangian dual learning rate 0.1, and the established PID gains. CPO retains
its trust-region defaults (target KL 0.02, ten conjugate-gradient iterations,
twenty critic updates), rather than replacing its update with PPO epochs.
All actors/critics start randomly, before attaching the shaping wrapper;
no safe-policy warm start or action shield is used for Lagrangian, PID or CPO.
Costs are unchanged, and the existing baseline collectors bootstrap both
critics at time-limit truncations. The primary PPO-Shield evaluation keeps its
runtime shield on, with an additional unshielded nominal evaluation reported
separately. Every final metric is measured with the original unshaped reward
and paired transition seeds `seed+10000` through `seed+10099`.

The non-SB3 cost baselines save native `model.pt` checkpoints and stream the
same raw/shaped exploration audit as PPO/PSPO. Their periodic raw reward curves
are evaluated at the first complete rollout at or beyond each 20k-step
checkpoint; this is within 2047 steps of the requested interval. Final scoring
always follows the last optimizer update at 401,408 steps. Training times
exclude periodic evaluation and final evaluation, with initialization recorded
separately. Existing completed PPO/PSPO runs can be reused as references; do
not combine smoke tests or different training budgets into the production
aggregate.

Run from the repository root with a new output path:

```bash
task_run=projects/safe_policy_optimisation/artifacts/frozenlake32_shaping_example
PYTHONPATH=core:. .venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_frozenlake_shaping_sweep.py \
  --size 32 --total-timesteps 200000 --eval-episodes 100 \
  --screen-session frozenlake32-shaping --output-dir "$task_run"
screen -L -Logfile "$task_run/_logs/supervisor.log" -dmS frozenlake32-shaping \
  env PYTHONPATH=core:. .venv/bin/python -u \
  projects/safe_policy_optimisation/scripts/run_frozenlake_shaping_sweep.py \
  --supervise "$task_run/_orchestrator/launch_manifest.json"
```

All methods use gamma 0.999, scale-one potential shaping only during training,
the original slippery dynamics, a 1250-step cap at 32x32, and two 64-unit tanh
layers with one-hot observations. Evaluation always uses original rewards.
PSPO and PPO use unshielded greedy evaluation; the primary PPO-Shield evaluation
retains its runtime shield. Separate nominal (shield-off) PPO-Shield metrics and
curves are also saved. PPO and PPO-Shield have identical random initial actor
and critic parameters for each paired seed; PSPO retains the safety-only actor.
PSPO uses the 100-rollout cadence described above, so this 98-rollout budget has
one final certified projection.

Results live under `METHOD/seedN/`. `status.json` tracks completion;
`_process/METHOD_seedN.json` records each worker; `_logs/` contains worker and
supervisor logs. `aggregate.json` and `seed_results.csv` update as runs finish,
with mean +/- two sample standard errors across completed training seeds and
explicit seed counts (a single completed seed has no standard-error estimate).
PSPO summaries include actor/model initialization, shield synthesis, and LID
timing. The nominal PPO-Shield aggregate is diagnostic, not a fourth training run.
