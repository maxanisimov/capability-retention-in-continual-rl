# PSPO vs RL baselines under extended training budgets

Generated 2026-08-27 by
`scripts/compare_extended_budget_pspo_vs_baselines.py`.

All rewards and safety rates are means over training seeds of a fresh,
unshielded, deterministic evaluation (100 episodes per seed). Uncertainty is
two standard errors of the seed mean, not a confidence interval. `PPO-Shield`
is the one method evaluated with its runtime shield attached.

## Bridge Crossing v2

| Setting | Budget (steps) | Method | Reward (2SE) | Safety (2SE) | Seeds |
|---|---:|---|---:|---:|---:|
| standard | 400,000 | PSPO | 0.309 +/- 0.302 | 1.000 +/- 0.000 | 10 |
| standard | 200,000 | PPO | 0.880 +/- 0.196 | 0.980 +/- 0.005 | 10 |
| standard | 200,000 | PPO-Lagrangian | 0.394 +/- 0.322 | 0.994 +/- 0.006 | 10 |
| standard | 200,000 | PPO-PID-Lagrangian | 0.476 +/- 0.321 | 1.000 +/- 0.000 | 10 |
| standard | 200,000 | CPO | 0.698 +/- 0.305 | 0.998 +/- 0.004 | 10 |
| standard | 200,000 | PPO-Shield | 0.900 +/- 0.200 | 1.000 +/- 0.000 | 10 |
| standard | 200,000 | PPO-Shield-Nominal | 0.433 +/- 0.233 | 0.995 +/- 0.010 | 10 |
| extended | 1,600,000 | PSPO | 0.900 +/- 0.200 | 1.000 +/- 0.000 | 10 |
| extended | 1,000,000 | PPO | 0.977 +/- 0.004 | 0.977 +/- 0.004 | 10 |
| extended | 1,000,000 | PPO-Lagrangian | 0.829 +/- 0.218 | 0.935 +/- 0.106 | 10 |
| extended | 1,000,000 | PPO-PID-Lagrangian | 0.798 +/- 0.266 | 0.998 +/- 0.003 | 10 |
| extended | 1,000,000 | CPO | 0.698 +/- 0.305 | 0.998 +/- 0.004 | 10 |
| extended | 1,000,000 | PPO-Shield | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 10 |
| extended | 1,000,000 | PPO-Shield-Nominal | 0.399 +/- 0.207 | 1.000 +/- 0.000 | 10 |

## MiniPacman

| Setting | Budget (steps) | Method | Reward (2SE) | Safety (2SE) | Seeds |
|---|---:|---|---:|---:|---:|
| standard | 500,000 | PSPO | 0.839 +/- 0.069 | 1.000 +/- 0.000 | 10 |
| standard | 500,000 | PPO | 0.924 +/- 0.032 | 0.217 +/- 0.069 | 10 |
| standard | 500,000 | PPO-Lagrangian | 0.592 +/- 0.074 | 0.972 +/- 0.017 | 10 |
| standard | 500,000 | PPO-PID-Lagrangian | 0.584 +/- 0.090 | 0.975 +/- 0.022 | 10 |
| standard | 500,000 | CPO | 0.005 +/- 0.007 | 0.998 +/- 0.004 | 10 |
| standard | 500,000 | PPO-Shield | 0.947 +/- 0.051 | 1.000 +/- 0.000 | 10 |
| standard | 500,000 | PPO-Shield-Nominal | 0.735 +/- 0.072 | 0.449 +/- 0.077 | 10 |
| extended | 2,000,000 | PSPO | 0.956 +/- 0.019 | 1.000 +/- 0.000 | 10 |
| extended | 2,000,000 | PPO | 0.985 +/- 0.011 | 0.240 +/- 0.051 | 10 |
| extended | 2,000,000 | PPO-Lagrangian | 0.707 +/- 0.083 | 0.961 +/- 0.024 | 10 |
| extended | 2,000,000 | PPO-PID-Lagrangian | 0.803 +/- 0.077 | 0.919 +/- 0.029 | 10 |
| extended | 2,000,000 | CPO | 0.134 +/- 0.109 | 0.992 +/- 0.007 | 10 |
| extended | 2,000,000 | PPO-Shield | 0.953 +/- 0.036 | 1.000 +/- 0.000 | 10 |
| extended | 2,000,000 | PPO-Shield-Nominal | 0.740 +/- 0.087 | 0.461 +/- 0.085 | 10 |
