# Stochastic FrozenLake 128×128 PSPO experiment

> **Historical-result note.** The completed results below used a goal-aware
> witness preference in the initial actor. The current FrozenLake launchers use
> `safety_only_shield_mask_v1`, with no reward, goal, witness, preferred-action,
> or goal-distance input. Tabular and lookup actors assign equal logits to all
> shield-permitted actions; the feature actor is fitted solely to the shield
> mask's safety margin. New runs are written beneath `_safety_only`-suffixed
> output directories; the historical numbers below have not been relabelled as
> results of the new setup.

The fixed layout has four 24×24 hole islands, a broad central cross and a safe
perimeter. Start is the upper-left corner; goal is the lower-right corner. Its
16,384 discrete state ids and four actions give 65,536 nominal state–action
pairs. There are 14,080 safety-winning states and 55,168 permitted state–action
pairs; 13,696 states permit multiple safe actions.

Dynamics are slippery: intended movement has probability 0.8, and each
perpendicular movement has probability 0.1. At grid boundaries, movement is
clamped as in Gymnasium. The lightweight adapter reuses Gymnasium's sparse
transition lists and does not allocate an S×A×S dense transition array.

## Safety and goal validation

The shield is the greatest fixed point of non-hole states possessing an action
whose every positive-probability successor remains in the set. The start lies
in that set. This is a zero-risk safety guarantee for shield-compliant policies,
relative to the exact model, not a sampled risk estimate.

All safety-winning states can reach the goal in the union graph of permitted
transitions. Four distinct deterministic witness policies were constructed.
Every selected action outside the goal has a positive-probability successor
with strictly smaller integer goal distance, and all its successors remain
safety-winning. No non-goal closed recurrent class is therefore possible;
each witness reaches the goal almost surely without visiting a hole. This
does not guarantee success before a finite timeout. Each witness additionally
passed 100 seeded simulation episodes without a safety violation or timeout.

## Training configuration

- Seeds: 0–9; fixed layout shared by every seed.
- Budget: 2,000,000 environment steps per seed; 5,000 steps per episode.
- Reward: goal indicator minus 0.001 per step, including the terminal step.
- Actor: two tanh hidden layers of width 64, with one-hot discrete observations.
- Initial actor: analytic safe-action logits, not reward-trained; all safe logits
  are 2, unsafe logits −2, and the action of witness policy 0 has logit 2.5.
  The same initial actor is used for all seeds. It is exactly safe on every
  safety-winning state and its greedy policy equals witness 0.
- PPO: learning rate 0.0003, rollout 2,048, minibatch 64, ten epochs, gamma 0.999,
  GAE lambda 0.95, clip 0.2, entropy coefficient zero.
- PSPO: adaptive directional orthotopes, replace mode, region-first enforcement
  every 100 rollouts, 200 region-growth iterations per computation, minibatch
  256, inverse temperature 1, all-safe-vs-unsafe logsumexp certification.
- Certification covers **every safety-winning state**, without certificate
  subsampling. Training exploration uses the runtime shield, as in the existing
  PSPO stage. Periodic enforcement is not per-gradient-step nominal safety.
- Final evaluation: nominal greedy actor, no runtime shield, 100 episodes per
  seed, reset seeds starting at training seed + 10,000; exhaustive final actor
  safety audit. Learning curves use ten episodes every 100,000 steps.

With step penalties, positive return is not synonymous with reaching the goal.
The experiment preserves the shared stage's original threshold-based metrics
as `metrics_reward_threshold.json` and reports actual goal attainment in
`metrics.json`, recovering the goal indicator from return plus 0.001×length.

## Running and monitoring

From the repository root:

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_stochastic_frozenlake_pspo.py --prepare
taskset -c <idle-cpu> .venv/bin/python projects/safe_policy_optimisation/scripts/run_stochastic_frozenlake_pspo.py --worker 0 --smoke
.venv/bin/python projects/safe_policy_optimisation/scripts/run_stochastic_frozenlake_pspo.py --launch
```

The launch uses detached screen session `pspo-frozenlake128`. The controller
samples CPU idle time for five seconds and admits distinct physical cores only
if every SMT sibling averages at least 95% idle. Each worker is pinned to one
CPU, and numerical-library thread counts are one. If fewer than ten cores are
available, remaining seeds are queued. A file lock prevents duplicate controllers.

Artifacts live under
`artifacts/paper_2503_07671/runs/pspo_stochastic_frozenlake128/`:

- `_inputs/`: layout text/image, shield, initial actor, witness policies and proof.
- `experiment.json`, `source_snapshot.zip`: configuration and source provenance.
- `launch_manifest.json`: per-seed CPU, PID, command and log path.
- `_logs/seedN.log`, `_logs/screen.log`: training and controller logs.
- `seedN/status.json`, `summary.json`, `metrics.json`, `model.zip`: run outputs.
- `aggregate.json`, `per_seed.csv`: summaries regenerated as workers finish.

The experiment budget is not an estimate of runtime. Large one-hot inputs and
exhaustive parameter-region certification may make these runs expensive.

## Results (10 seeds, completed 2026-09-17)

Launched 2026-09-16 20:55 UTC, all ten workers exited 0 by 2026-09-17 00:54 UTC;
about 3.9 hours of wall time per seed on one pinned core.

| quantity | value |
| --- | --- |
| mean total reward | 0.6656 ± 0.0002 (2se) |
| safety rate | 1.000 (0 of 10 seeds with any violation) |
| success rate (goal reached) | 1.000 |
| mean evaluation episode length | 334.4 steps |
| final exhaustive actor alignment | 1.0 on every seed |

Evaluation is the nominal greedy actor with no runtime shield, 100 episodes per
seed. Seed spread is negligible: per-seed reward runs 0.6650 to 0.6658. Return
and length are consistent with reaching the goal every episode under the 0.001
step penalty (1 − 0.001 × 334.4 = 0.666). The reward-threshold metric preserved
in `metrics_reward_threshold.json` agrees at 1.0, so the two success definitions
do not diverge here.

`final_exact_all_state_alignment` is 1.0 on all ten seeds: the final greedy actor
selects a shield-permitted action in **every** one of the 14,080 safety-winning
states, checked exhaustively rather than on visited states. Combined with the
shield's fixed-point construction this is a zero-risk guarantee for the returned
policy, relative to the exact model.

Training itself logged zero violations and zero unsafe executed actions across
all 2,000,896 exploration steps per seed, with the runtime shield overriding
0.66% of proposed actions on average (11,375 of 2.0M on seed 0).

### Enforcement behaviour

Ten enforcement events per seed: nine on the 100-rollout cadence (every 204,800
steps) plus a final flush at 2,000,896. **Every event is `projected`; none is
`accepted_unchanged`**, and the projection displacement is flat across training
(L2 21–29 on seed 0, first event to last). LID dominates the cost at roughly
190 s per event, about 31 minutes of the 3.9-hour run; projection itself is
0.02 s.

This is the same decision mix that accompanies reward collapse on LunarLander
descent — 100% projected, displacement flat — yet here reward and safety both
finish at ceiling. An all-projected mix is therefore not sufficient on its own to
explain the continuous-state collapse; see
`memory/pspo-continuous-projection-collapse`.

### Safety holds at certified checkpoints, not between them

The unshielded learning curve is evaluated just *before* each projection, and it
shows the uncertified policy is genuinely unsafe mid-training. Averaged over
seeds, the last curve point carries safety 0.57 and return 0.217 (range 0.1–1.0
safety); after the final flush every seed is at 1.000 and 0.666. Seed 0 is the
extreme case: 0.10 safety and −0.010 return immediately before the flush.

So the projection is doing real work — it restores exact safety and raises
reward — but the guarantee applies to certified checkpoints only. Periodic
enforcement is not per-gradient-step nominal safety, and the curve files should
not be read as if it were.
