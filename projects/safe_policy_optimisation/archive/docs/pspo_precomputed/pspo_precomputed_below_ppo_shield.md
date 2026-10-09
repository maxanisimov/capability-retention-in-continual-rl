# PSPO Precomputed Results Below PPO-Shield

Environment/architecture pairs where the best completed PSPO-precomputed total reward is below the best completed PPO-Shield total reward for the same inferred actor-critic architecture.

| Environment | Architecture | PSPO reward | PSPO SEM | PPO-Shield reward | PPO-Shield SEM | Gap | rashomon_n_iters | bc_target_margin | PSPO source | PPO-Shield source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bridge Crossing v1 | two_hidden | 0.27 | 0.137 | 1 | 0 | -0.73 | 6000 | 2 | outputs/_sweeps/20260724_124311/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json | outputs/_sweeps/20260724_124311/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json |
| Bridge Crossing v2 | two_hidden | 0.372 | 0.152 | 0.9 | 0.1 | -0.528 | 2000 | 2 | outputs/_sweeps/20260723_231451/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json | outputs/_sweeps/20260724_152054/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json |
| Colour Bomb v2 | one_hidden | 0 | 0 | 24.59 | 0.084 | -24.59 | 2000 | 10 | outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json | outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json |
| Colour Bomb v2 | two_hidden | 0 | 0 | 23.36 | 1.053 | -23.36 | 2000 | 2 | outputs/_sweeps/20260723_233403/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json | outputs/_sweeps/20260723_233403/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json |
| Media Streaming | two_hidden | -24.17 | 0.013 | -1.443 | 0.017 | -22.727 | 2000 | 2 | outputs/_sweeps/20260723_204829/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json | outputs/_sweeps/20260723_204829/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json |
| MiniPacman | tabular | 0.733 | 0.032 | 0.942 | 0.008 | -0.209 | 2000 | 10 | outputs/_sweeps_tabular/paper_2503_07671_minipacman/aggregate/aggregated_metrics.json | outputs/_sweeps_tabular/paper_2503_07671_minipacman/aggregate/aggregated_metrics.json |
