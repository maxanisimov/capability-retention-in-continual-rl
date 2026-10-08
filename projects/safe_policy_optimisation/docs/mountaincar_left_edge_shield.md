# MountainCar left-edge shield — PSPO experiment

A deliberately minimal continuous-state shield for `MountainCar-v0`, built to
test whether PSPO can certify a specification that leaves PPO real room to move.
It is the counterpoint to the LunarLander descent shield, where every update is
projected and reward collapses.

## The specification

Push-left (action 0) is inadmissible inside a 0.05-wide strip against the left
wall, `x ∈ [-1.2001, -1.15]`, spanning the full velocity range `[-0.07, 0.07]`.
Both other actions stay admissible inside the strip, and all three are admissible
outside it. One certificate region; the certificate covers it exactly.

Built by `scripts/build_mountaincar_left_edge_shield.py`, which emits the
`mountaincar-boxes` runtime `.npz` **and** the matching
`TensorDataset(X_l, X_u, safe_mask)` certificate from the same float32 arrays —
`train_pspo_continuous.py` requires the two to agree to `atol=1e-7`. No new
shield class was needed: `MountainCarIntervalBoxShield` already accepts arbitrary
boxes with per-box action masks.

## A band this thin cannot be an invariance shield

This is an *action constraint*, not wall avoidance, and that is forced rather
than chosen. At `x = -1.15` the reachable leftward speed is about `-0.062`
(forward reachability from the init distribution), while the maximum rightward
acceleration there is `force + gravity·|cos 3x| ≈ 0.0034` per step. Braking from
that speed needs roughly `0.57` of position; the strip offers `0.05`. Simulated
under permanent push-right from `x = -1.15`, the wall is still reached at
`v = -0.062`, `-0.04` and `-0.02`; only `|v| ≲ 0.01` is recoverable.

So no 0.05-wide band is inductively invariant, and the action-level constraint is
the only soundly enforceable specification on it. That is also all PSPO ever
certifies — state → admissible-action sets, never trajectories.

## Training configuration

- Seeds 0–9, 400,000 environment steps each, 200-step episode cap.
- Actor: two tanh hidden layers of width 64 over the raw 2-D continuous state.
- Base policies from `outputs/continuous_state_shields/pspo_initialisation/20260917_mountaincar_left_edge/`,
  one per seed, referenced by SHA-256 in each `config.json`.
- Shaped MountainCar reward (`mountaincar_shaped_reward: true`).
- PPO: lr 0.001, rollout 1,024, minibatch 128, ten epochs, gamma 0.99,
  GAE lambda 0.95, clip 0.2, entropy coefficient zero.
- PSPO: adaptive directional orthotopes, `rashomon_project`, `train_phase`
  granularity every train phase, 200 growth iterations, IBP for both growth and
  certification, `weighted_width` objective, logsumexp surrogate.
- Evaluation: nominal greedy actor, **no runtime shield**, 100 episodes per seed.

Roughly 15 minutes per seed, all ten exit 0. Artifacts in
`outputs/continuous_state_shields/pspo/20260917_mountaincar_left_edge/`
(see `memory/artifacts-overwrite-hazard` before relaunching).

## Results (10 seeds, 2026-09-17)

| quantity | value |
| --- | --- |
| mean total reward | −180.56 ± 14.38 (2se) |
| empirical safety rate | 0.759 ± 0.197 |
| success rate (reward > −110) | 0.000 on every seed |
| mean evaluation episode length | 180.6 steps |
| proposed-action compliance | **1.0000 on every seed** |

**Compliance is perfect.** Zero unsafe proposed actions in 180,564 greedy
proposed-action checks at deployment, and zero across 4,003,840 shielded
exploration checks during training (shield intervention rate 2.5e−6). PSPO
delivers exactly what it certifies.

**Empirical safety is not, and that gap was predicted.** 241 of 1,000 evaluation
episodes still touch the wall: the car obeys the strip and then coasts in on
momentum it built outside it. Nothing about the certificate claims otherwise —
see the reachability argument above.

### Compliance and safety are anti-correlated here

| seed | 2 | 3 | 4 | 6 | 7 | 5 | 8 | 0 | 9 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| safety | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 0.76 | 0.39 | 0.34 | 0.25 |
| reward | −200.0 | −194.0 | −200.0 | −200.0 | −200.0 | −141.1 | −145.9 | −168.5 | −173.3 | −182.9 |

The five seeds at safety 1.00 never reach `x = -1.2` and score −194 to −200 —
they never learn the swing-up at all. The five that touch the wall score −141 to
−183. Wall contact is a *byproduct of learning the task*, so any shield that
could actually prevent it would have to forbid the swing-up. This is the old
MountainCar degeneracy seen from the other side, and it is why the safety rate
should not be read as a quality metric on this task.

### Enforcement behaviour

391 enforcement events per seed, 3,910 in total:

| decision | count | share |
| --- | --- | --- |
| `reverted_lid_failure` | 1,687 | 43.1% |
| `projected` | 1,210 | 30.9% |
| `accepted_contained` | 1,013 | 25.9% |

**This certificate genuinely leaves PPO room**: a quarter of updates are accepted
unchanged, against 100% `projected` for the LunarLander descent shield
(`memory/pspo-continuous-projection-collapse`). That is a win on certification,
not on reward.

It also corrects an inference from the 2026-09-16 segment post-mortem:
`reverted_lid_failure` is **not** segment-specific. It is 43.1% of updates here,
on orthotope regions, where the LID step fails to certify any region and the
trainer copies back the previously certified policy.

LID dominates the cost — 334 s of Rashomon engine time per seed, about 37% of the
run — while projection itself totals 0.12 s. 20,284 growth iterations per seed
were saved by containment checks out of 57,916 spent.

## Reproducing

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/build_mountaincar_left_edge_shield.py \
    --output-dir outputs/continuous_state_shields/synthesised/mountaincar_left_edge \
    --run-id left_edge
projects/safe_policy_optimisation/scripts/run_pspo_mountaincar_left_edge.sh
```

The launcher pins each seed to its own CPU and records the assignment in
`launch_manifest.tsv`; per-seed logs land in `logs/seed_N.log`.
