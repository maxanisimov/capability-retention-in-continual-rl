# Two-hidden-layer baseline results

This directory is the canonical paper-facing index for available
two-hidden-layer baseline results. Safe-RL baselines are grouped together
with PPO and PPO-Shield-Nominal reference baselines so the reward/safety
tradeoff is visible in one place.

Generated files:

- `safe_rl_baseline_summary.csv`: cross-seed mean/std/standard-error for
  total reward, safety rate, and success rate.
- `safe_rl_baseline_summary.md`: compact human-readable table.
- `safe_rl_baseline_per_seed.csv`: one row per environment/method/seed.
- `manifest.json`: source sweep roots and generation metadata.

Raw source sweep roots:

- `Bridge Crossing v1`: `outputs/_sweeps_2hidden_bridge_crossing_baselines_only/bridge_crossing`
- `Bridge Crossing v2`: `outputs/_sweeps_2hidden_bridge_crossing_v2_baselines_only/bridge_crossing_v2`
- `Colour Bomb v1`: `outputs/_sweeps_2hidden_colour_bomb_baselines_only/colour_bomb`
- `Colour Bomb v2`: `outputs/_sweeps_2hidden_colour_bomb_v2_baselines_only/colour_bomb_v2`
- `Media Streaming`: `outputs/_sweeps_2hidden_media_streaming_baselines_only/media_streaming`
- `MiniPacman`: `outputs/_sweeps_2hidden_minipacman_baselines_only/paper_2503_07671_minipacman`

Result availability is determined from final `metrics.json` files. A method
with `seed_count = 0` has no completed result in the current
two-hidden sweep tree for that environment.

Regenerate with:

```bash
python3 projects/safe_policy_optimisation/scripts/collect_two_hidden_safe_rl_baselines.py
```
