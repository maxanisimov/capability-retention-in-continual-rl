# Tabular PSPO Precomputed Baseline Comparison

Values are mean +/- standard error over 10 seeds. The PSPO precomputed column
uses the best completed tabular PSPO-precomputed result available for each
environment.

## Total Reward

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge Crossing v1 | 1.000 +/- 0.000 | 0.192 +/- 0.128 | 0.100 +/- 0.100 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| Bridge Crossing v2 | 1.000 +/- 0.000 | 0.894 +/- 0.099 | 0.394 +/- 0.161 | 0.476 +/- 0.160 | 0.698 +/- 0.152 | 1.000 +/- 0.000 | 0.036 +/- 0.002 |
| Colour Bomb v1 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.455 +/- 0.107 |
| Colour Bomb v2 | 4.970 +/- 1.107 | 4.331 +/- 0.825 | 0.613 +/- 0.285 | 1.687 +/- 1.129 | 0.102 +/- 0.041 | 3.655 +/- 0.803 | 1.140 +/- 0.184 |
| Media Streaming | -1.493 +/- 0.008 | 0.000 +/- 0.000 | -3.624 +/- 0.396 | -4.207 +/- 0.357 | -24.170 +/- 0.013 | -1.493 +/- 0.008 | 0.000 +/- 0.000 |
| MiniPacman | 0.733 +/- 0.032 | 0.977 +/- 0.009 | 0.592 +/- 0.037 | 0.584 +/- 0.045 | 0.005 +/- 0.003 | 0.942 +/- 0.008 | 0.868 +/- 0.031 |

## Safety Rate

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge Crossing v1 | 1.000 +/- 0.000 | 0.992 +/- 0.005 | 1.000 +/- 0.000 | 0.999 +/- 0.001 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.724 +/- 0.017 |
| Bridge Crossing v2 | 1.000 +/- 0.000 | 0.994 +/- 0.003 | 0.994 +/- 0.003 | 1.000 +/- 0.000 | 0.998 +/- 0.002 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Colour Bomb v1 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.885 +/- 0.058 |
| Colour Bomb v2 | 1.000 +/- 0.000 | 0.258 +/- 0.031 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.439 +/- 0.055 |
| Media Streaming | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| MiniPacman | 1.000 +/- 0.000 | 0.092 +/- 0.019 | 0.972 +/- 0.008 | 0.975 +/- 0.011 | 0.998 +/- 0.002 | 1.000 +/- 0.000 | 0.269 +/- 0.037 |

## Success Rate

| Environment | PSPO precomputed | PPO | PPO-Lag | PPO-PID-Lag | CPO | PPO-Shield | PPO-Shield-Nominal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge Crossing v1 | 1.000 +/- 0.000 | 0.192 +/- 0.128 | 0.100 +/- 0.100 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| Bridge Crossing v2 | 1.000 +/- 0.000 | 0.894 +/- 0.099 | 0.394 +/- 0.161 | 0.476 +/- 0.160 | 0.698 +/- 0.152 | 1.000 +/- 0.000 | 0.036 +/- 0.002 |
| Colour Bomb v1 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.455 +/- 0.107 |
| Colour Bomb v2 | 0.807 +/- 0.027 | 0.750 +/- 0.033 | 0.287 +/- 0.069 | 0.345 +/- 0.087 | 0.085 +/- 0.032 | 0.793 +/- 0.040 | 0.518 +/- 0.044 |
| Media Streaming | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| MiniPacman | 0.733 +/- 0.032 | 0.977 +/- 0.009 | 0.592 +/- 0.037 | 0.584 +/- 0.045 | 0.005 +/- 0.003 | 0.942 +/- 0.008 | 0.868 +/- 0.031 |

## PSPO Sources

| Environment | Best PSPO source |
|---|---|
| Bridge Crossing v1 | `outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json` |
| Bridge Crossing v2 | `outputs/_sweeps_tabular_hiter/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json` |
| Colour Bomb v1 | `outputs/_sweeps_tabular_colour_bomb_pspo/paper_2503_07671_colour_bomb/aggregate/aggregated_metrics.json` |
| Colour Bomb v2 | `outputs/_sweeps_tabular_colour_bomb_v2_pspo_precomputed_10k_margin0p5/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json` |
| Media Streaming | `outputs/_sweeps_tabular/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json` |
| MiniPacman | `outputs/_sweeps_tabular/paper_2503_07671_minipacman/aggregate/aggregated_metrics.json` |
