# PSPO Safe-Update Ablation

Comparison: PSPO vs PPO-Shield-Nominal. PPO-Shield-Nominal is treated as the ablation where PSPO's safe policy updates are removed and the resulting policy is evaluated without the shield.

## Tabular Actor-Critic, Index Encoding

- PSPO improves total reward in 4/6 environments: Colour Bomb, Bridge Crossing v1, Bridge Crossing v2, Colour Bomb v2.
- PSPO improves safety rate in 5/6 environments: Media Streaming, Colour Bomb, Bridge Crossing v1, Colour Bomb v2, MiniPacman.
- PSPO matches nominal safety in 1/6 environments: Bridge Crossing v2.
- Nominal remains competitive or better on both reward and safety in 0/6 environments: none.

| Environment | PSPO Reward | Nominal Reward | Delta Reward | PSPO Safety | Nominal Safety | Delta Safety |
|---|---:|---:|---:|---:|---:|---:|
| Media Streaming | -1.444 | 0 | -1.444 | 1 | 0 | 1 |
| Colour Bomb | 1 | 0.455 | 0.545 | 1 | 0.885 | 0.115 |
| Bridge Crossing v1 | 1 | 0 | 1 | 1 | 0.724 | 0.276 |
| Bridge Crossing v2 | 1 | 0.036 | 0.964 | 1 | 1 | 0 |
| Colour Bomb v2 | 4.97 | 1.14 | 3.83 | 1 | 0.439 | 0.561 |
| MiniPacman | 0.733 | 0.868 | -0.135 | 1 | 0.269 | 0.731 |

## One-Hidden-Layer Actor-Critic, Index Encoding

- PSPO improves total reward in 2/6 environments: Colour Bomb, Bridge Crossing v1.
- PSPO improves safety rate in 5/6 environments: Media Streaming, Colour Bomb, Bridge Crossing v1, Bridge Crossing v2, MiniPacman.
- PSPO matches nominal safety in 1/6 environments: Colour Bomb v2.
- Nominal remains competitive or better on both reward and safety in 1/6 environments: Colour Bomb v2.

| Environment | PSPO Reward | Nominal Reward | Delta Reward | PSPO Safety | Nominal Safety | Delta Safety |
|---|---:|---:|---:|---:|---:|---:|
| Media Streaming | -21.92 | 0 | -21.92 | 1 | 0 | 1 |
| Colour Bomb | 1 | 0.606 | 0.394 | 1 | 0.882 | 0.118 |
| Bridge Crossing v1 | 1 | 0.001 | 0.999 | 1 | 0.442 | 0.558 |
| Bridge Crossing v2 | 0 | 0.456 | -0.456 | 1 | 0.992 | 0.008 |
| Colour Bomb v2 | 4.269 | 24.59 | -20.32 | 1 | 1 | 0 |
| MiniPacman | 0.575 | 0.774 | -0.199 | 1 | 0.395 | 0.605 |

## Two-Hidden-Layer Actor-Critic, Index Encoding

- PSPO improves total reward in 2/5 environments: Colour Bomb, Bridge Crossing v1.
- PSPO improves safety rate in 4/5 environments: Media Streaming, Colour Bomb, Bridge Crossing v1, Bridge Crossing v2.
- PSPO matches nominal safety in 1/5 environments: Colour Bomb v2.
- Nominal remains competitive or better on both reward and safety in 1/5 environments: Colour Bomb v2.

| Environment | PSPO Reward | Nominal Reward | Delta Reward | PSPO Safety | Nominal Safety | Delta Safety |
|---|---:|---:|---:|---:|---:|---:|
| Media Streaming | -24.17 | -0.003 | -24.17 | 1 | 0.006 | 0.994 |
| Colour Bomb | 1 | 0.455 | 0.545 | 1 | 0.885 | 0.115 |
| Bridge Crossing v1 | 0.745 | 0.009 | 0.736 | 1 | 0.398 | 0.602 |
| Bridge Crossing v2 | 0.372 | 0.433 | -0.061 | 1 | 0.995 | 0.005 |
| Colour Bomb v2 | 8.771 | 23.36 | -14.59 | 1 | 1 | 0 |
