# PSPO reward-improvement experiments

Completed 2026-08-23. All rewards are means across training seeds. Uncertainty is
reported as two standard errors (2SE), not as a confidence interval.

## Selection protocol

Each candidate was first trained with seeds 0--4 and evaluated for 100 fresh,
unshielded, deterministic episodes per seed, beginning at reset seed
`40000 + training_seed`. A candidate qualified for confirmation only if:

1. its greedy final policy selected a shield-allowed action on every shield state;
2. its held-out mean reward improved by at least 0.10 on a bridge environment or
   0.03 on MiniPacman; and
3. at least four of five matched seeds were no worse than the control.

Qualifying candidates were extended with seeds 5--9. MiniPacman candidates that
failed the pilot gate were not extended.

## Held-out selection results

| Environment | Candidate | Seeds | Candidate reward (2SE) | Control reward (2SE) | Gain | Empirical safety | Exact alignment | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Bridge Crossing v1 | margin 1, 200 iterations, 200k steps | 10 | 0.551 +/- 0.311 | 0.244 +/- 0.265 | +0.307 | 0.995 | 1.000 | confirmed |
| Bridge Crossing v1 | margin 2, 100 iterations, 200k steps | 10 | 0.297 +/- 0.302 | 0.244 +/- 0.265 | +0.053 | 0.997 | 1.000 | rejected |
| Bridge Crossing v2 | margin 1, 200 iterations, 200k steps | 5 | 0.000 +/- 0.000 | 0.200 +/- 0.400 | -0.200 | 1.000 | 1.000 | rejected |
| Bridge Crossing v2 | margin 2, 200 iterations, 400k steps | 10 | 0.314 +/- 0.301 | 0.184 +/- 0.246 | +0.130 | 1.000 | 1.000 | confirmed |
| MiniPacman | margin 1, 200 iterations, frequency 100, 500k steps | 5 | 0.480 +/- 0.017 | 0.480 +/- 0.014 | +0.000 | 1.000 | 1.000 | rejected (3/5 no worse) |
| MiniPacman | margin 2, 200 iterations, frequency 50, 500k steps | 5 | 0.476 +/- 0.010 | 0.480 +/- 0.014 | -0.004 | 1.000 | 1.000 | rejected (3/5 no worse) |

The margin-2 candidates reused the exact original margin-2 base policy. This
avoids treating a regenerated but numerically different initialization as a
training effect.

## Retained PSPO policies and standard final tests

| Environment | Retained configuration | Seeds | Reward (2SE) | Safety rate |
|---|---|---:|---:|---:|
| Bridge Crossing v1 | margin 1, 200 iterations, 200k steps | 10 | 0.567 +/- 0.315 | 1.000 |
| Bridge Crossing v2 | margin 2, 200 iterations, 400k steps | 10 | 0.309 +/- 0.302 | 1.000 |
| MiniPacman | existing margin 2, 200 iterations, frequency 100 | 10 | 0.554 +/- 0.014 | 1.000 |

The bridge rows use the confirmed candidate runs. Neither MiniPacman pilot beat
the gate, so the existing ten-seed result remains retained.

## Comparison with RL baselines

These are the standard final-test aggregates (10 seeds, 100 episodes per seed).
Reward uncertainty is 2SE. Safety is the mean fraction of unshielded evaluation
trajectories within the cost limit, except `PPO-Shield`, whose reported policy is
evaluated with the runtime shield as indicated by its method name.

| Environment | Method | Reward (2SE) | Safety |
|---|---|---:|---:|
| Bridge Crossing v1 | retained PSPO | 0.567 +/- 0.315 | 1.000 |
|  | PPO | 0.296 +/- 0.302 | 0.873 |
|  | PPO-Lagrangian | 0.100 +/- 0.200 | 1.000 |
|  | PPO-PID-Lagrangian | 0.000 +/- 0.000 | 0.999 |
|  | CPO | 0.000 +/- 0.000 | 1.000 |
|  | PPO-Shield | 1.000 +/- 0.000 | 1.000 |
| Bridge Crossing v2 | retained PSPO | 0.309 +/- 0.302 | 1.000 |
|  | PPO | 0.880 +/- 0.196 | 0.980 |
|  | PPO-Lagrangian | 0.394 +/- 0.322 | 0.994 |
|  | PPO-PID-Lagrangian | 0.476 +/- 0.321 | 1.000 |
|  | CPO | 0.698 +/- 0.305 | 0.998 |
|  | PPO-Shield | 0.900 +/- 0.200 | 1.000 |
| MiniPacman | retained PSPO | 0.554 +/- 0.014 | 1.000 |
|  | PPO | 0.924 +/- 0.032 | 0.217 |
|  | PPO-Lagrangian | 0.592 +/- 0.074 | 0.972 |
|  | PPO-PID-Lagrangian | 0.584 +/- 0.090 | 0.975 |
|  | CPO | 0.005 +/- 0.007 | 0.998 |
|  | PPO-Shield | 0.947 +/- 0.051 | 1.000 |

The v1 change substantially improves PSPO over unshielded PPO and the penalty
methods, but PPO-Shield still has the best reward. The v2 extension improves the
old PSPO control but remains below all reward-competitive baselines. MiniPacman
does not benefit from either tested change; PPO-Shield has much higher reward at
the same measured safety, while unshielded PPO trades safety for reward.

## Safety interpretation

`Empirical safety` is measured from the fresh unshielded trajectories. `Exact
alignment` enumerates every shield state with at least one safe action and checks
that the deterministic greedy policy selects a permitted action. The bridge v1
held-out trajectory rate of 0.995 reflects environment stochasticity; it is not
an unsafe proposed action by the policy. Across the audited candidates, the
number of unsafe proposed actions and unsafe enumerated shield states was zero.

## Result artifacts

- Bridge v1 winner: `artifacts/paper_2503_07671/runs/pspo_reward_pilot_bcv1_m1_i200_f1_t200k`
- Bridge v2 winner: `artifacts/paper_2503_07671/runs/pspo_reward_pilot_bcv2_m2_reuse_i200_f1_t400k`
- MiniPacman margin-1 pilot: `artifacts/paper_2503_07671/runs/pspo_reward_pilot_mini_m1_i200_f100_t500k`
- MiniPacman margin-2 pilot: `artifacts/paper_2503_07671/runs/pspo_reward_pilot_mini_m2_reuse_i200_f50_t500k`
- Baselines: `docs/two_hidden_safe_rl_baselines/safe_rl_baseline_summary.csv`
