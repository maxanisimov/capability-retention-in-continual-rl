# Two-hidden-layer baseline results

This table is generated from completed `metrics.json` files in the
`outputs/_sweeps_2hidden_*_baselines_only` raw sweep directories.

Included methods: PPO, PPO-Lagrangian, PPO-PID-Lagrangian, CPO, PPO-Shield,
and PPO-Shield-Nominal where result files exist. PSPO/Rashomon runs are not
included in this baseline summary.

| Environment | Method | Role | n seeds | Total reward mean ± s.e. | Safety mean ± s.e. |
|---|---|---|---:|---:|---:|
| Bridge Crossing v1 | PPO | reference_baseline | 10 | 0.296 ± 0.151 | 0.873 ± 0.089 |
| Bridge Crossing v1 | PPO-Lagrangian | safe_rl_baseline | 10 | 0.100 ± 0.100 | 1.000 ± 0.000 |
| Bridge Crossing v1 | PPO-PID-Lagrangian | safe_rl_baseline | 10 | 0.000 ± 0.000 | 0.999 ± 0.001 |
| Bridge Crossing v1 | CPO | safe_rl_baseline | 10 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| Bridge Crossing v1 | PPO-Shield | safe_rl_baseline | 0 | — | — |
| Bridge Crossing v1 | PPO-Shield-Nominal | reference_baseline | 0 | — | — |
| Bridge Crossing v2 | PPO | reference_baseline | 10 | 0.880 ± 0.098 | 0.980 ± 0.003 |
| Bridge Crossing v2 | PPO-Lagrangian | safe_rl_baseline | 10 | 0.394 ± 0.161 | 0.994 ± 0.003 |
| Bridge Crossing v2 | PPO-PID-Lagrangian | safe_rl_baseline | 10 | 0.476 ± 0.160 | 1.000 ± 0.000 |
| Bridge Crossing v2 | CPO | safe_rl_baseline | 10 | 0.698 ± 0.152 | 0.998 ± 0.002 |
| Bridge Crossing v2 | PPO-Shield | safe_rl_baseline | 0 | — | — |
| Bridge Crossing v2 | PPO-Shield-Nominal | reference_baseline | 0 | — | — |
| Colour Bomb v1 | PPO | reference_baseline | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Colour Bomb v1 | PPO-Lagrangian | safe_rl_baseline | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Colour Bomb v1 | PPO-PID-Lagrangian | safe_rl_baseline | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Colour Bomb v1 | CPO | safe_rl_baseline | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Colour Bomb v1 | PPO-Shield | safe_rl_baseline | 0 | — | — |
| Colour Bomb v1 | PPO-Shield-Nominal | reference_baseline | 0 | — | — |
| Colour Bomb v2 | PPO | reference_baseline | 10 | 36.387 ± 0.267 | 0.000 ± 0.000 |
| Colour Bomb v2 | PPO-Lagrangian | safe_rl_baseline | 10 | 0.613 ± 0.285 | 1.000 ± 0.000 |
| Colour Bomb v2 | PPO-PID-Lagrangian | safe_rl_baseline | 10 | 1.687 ± 1.129 | 1.000 ± 0.000 |
| Colour Bomb v2 | CPO | safe_rl_baseline | 10 | 0.102 ± 0.041 | 1.000 ± 0.000 |
| Colour Bomb v2 | PPO-Shield | safe_rl_baseline | 10 | 23.360 ± 1.053 | 1.000 ± 0.000 |
| Colour Bomb v2 | PPO-Shield-Nominal | reference_baseline | 10 | 23.360 ± 1.053 | 1.000 ± 0.000 |
| Media Streaming | PPO | reference_baseline | 10 | -0.040 ± 0.040 | 0.059 ± 0.045 |
| Media Streaming | PPO-Lagrangian | safe_rl_baseline | 10 | -3.624 ± 0.396 | 1.000 ± 0.000 |
| Media Streaming | PPO-PID-Lagrangian | safe_rl_baseline | 10 | -4.207 ± 0.357 | 1.000 ± 0.000 |
| Media Streaming | CPO | safe_rl_baseline | 10 | -24.170 ± 0.013 | 1.000 ± 0.000 |
| Media Streaming | PPO-Shield | safe_rl_baseline | 0 | — | — |
| Media Streaming | PPO-Shield-Nominal | reference_baseline | 0 | — | — |
| MiniPacman | PPO | reference_baseline | 10 | 0.924 ± 0.016 | 0.217 ± 0.034 |
| MiniPacman | PPO-Lagrangian | safe_rl_baseline | 10 | 0.592 ± 0.037 | 0.972 ± 0.008 |
| MiniPacman | PPO-PID-Lagrangian | safe_rl_baseline | 10 | 0.584 ± 0.045 | 0.975 ± 0.011 |
| MiniPacman | CPO | safe_rl_baseline | 10 | 0.005 ± 0.003 | 0.998 ± 0.002 |
| MiniPacman | PPO-Shield | safe_rl_baseline | 10 | 0.947 ± 0.026 | 1.000 ± 0.000 |
| MiniPacman | PPO-Shield-Nominal | reference_baseline | 10 | 0.735 ± 0.036 | 0.449 ± 0.039 |
