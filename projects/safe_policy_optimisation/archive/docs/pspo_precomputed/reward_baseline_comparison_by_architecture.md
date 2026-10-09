# PSPO Precomputed Reward vs Baselines by Architecture

Values are mean +/- standard error over 10 seeds where available. The PSPO
column uses the best completed PSPO-precomputed result by mean total reward for
each environment and actor-critic architecture in the current output tree.

## Tabular Actor-Critic, Index Encoding

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Media Streaming | -1.444 +/- 0.020 | 0.000 +/- 0.000 | -3.624 +/- 0.396 | -4.207 +/- 0.357 | -24.170 +/- 0.013 | -1.493 +/- 0.008 | 0.000 +/- 0.000 |
| Colour Bomb | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.455 +/- 0.107 |
| Bridge Crossing v1 | 1.000 +/- 0.000 | 0.192 +/- 0.128 | 0.100 +/- 0.100 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| Bridge Crossing v2 | 1.000 +/- 0.000 | 0.894 +/- 0.099 | 0.394 +/- 0.161 | 0.476 +/- 0.160 | 0.698 +/- 0.152 | 1.000 +/- 0.000 | 0.036 +/- 0.002 |
| Colour Bomb v2 | 4.970 +/- 1.107 | 4.331 +/- 0.825 | 0.613 +/- 0.285 | 1.687 +/- 1.129 | 0.102 +/- 0.041 | 3.655 +/- 0.803 | 1.140 +/- 0.184 |
| MiniPacman | 0.733 +/- 0.032 | 0.977 +/- 0.009 | 0.592 +/- 0.037 | 0.584 +/- 0.045 | 0.005 +/- 0.003 | 0.942 +/- 0.008 | 0.868 +/- 0.031 |

## One-Hidden-Layer Actor-Critic, Index Encoding

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Media Streaming | -21.763 +/- 0.536 | 0.000 +/- 0.000 | -4.558 +/- 0.556 | -5.540 +/- 0.397 | -24.170 +/- 0.013 | -1.455 +/- 0.025 | 0.000 +/- 0.000 |
| Colour Bomb | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.606 +/- 0.126 |
| Bridge Crossing v1 | 1.000 +/- 0.000 | 0.196 +/- 0.131 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.001 +/- 0.001 |
| Bridge Crossing v2 | 0.000 +/- 0.000 | 0.495 +/- 0.162 | 0.500 +/- 0.167 | 0.598 +/- 0.163 | 0.385 +/- 0.158 | 0.600 +/- 0.163 | 0.456 +/- 0.148 |
| Colour Bomb v2 | 4.269 +/- 0.616 | 36.410 +/- 0.373 | 0.437 +/- 0.077 | 0.795 +/- 0.150 | 0.032 +/- 0.017 | 24.590 +/- 0.084 | 24.590 +/- 0.084 |
| MiniPacman | 0.671 +/- 0.035 | 0.987 +/- 0.004 | 0.358 +/- 0.053 | 0.445 +/- 0.037 | 0.000 +/- 0.000 | 0.953 +/- 0.009 | 0.774 +/- 0.027 |

## Two-Hidden-Layer Actor-Critic, Index Encoding

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Media Streaming | -24.170 +/- 0.013 | -0.040 +/- 0.040 | -3.624 +/- 0.396 | -4.207 +/- 0.357 | -24.170 +/- 0.013 | -1.443 +/- 0.017 | -0.003 +/- 0.003 |
| Colour Bomb | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.455 +/- 0.107 |
| Bridge Crossing v1 | 0.745 +/- 0.124 | 0.296 +/- 0.151 | 0.100 +/- 0.100 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.009 +/- 0.006 |
| Bridge Crossing v2 | 0.900 +/- 0.100 | 0.880 +/- 0.098 | 0.394 +/- 0.161 | 0.476 +/- 0.160 | 0.698 +/- 0.152 | 0.900 +/- 0.100 | 0.433 +/- 0.117 |
| Colour Bomb v2 | 8.771 +/- 1.267 | 36.387 +/- 0.267 | 0.613 +/- 0.285 | 1.687 +/- 1.129 | 0.102 +/- 0.041 | 23.360 +/- 1.053 | 23.360 +/- 1.053 |

Two-hidden MiniPacman is omitted because the current architecture-results
configuration has no completed two-hidden MiniPacman baseline run mapping.
