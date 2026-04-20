# FrozenLake Scaled (10x10 .. 100x100)

This folder contains large diagonal **source** environments and PPO settings that were empirically verified to train a successful policy for each size from `10x10` to `100x100`.

## Files
- `source_envs.yaml`: generated diagonal source maps (`diagonal_10x10` ... `diagonal_100x100`), with `max_episode_steps = 4 * grid_size`.
- `downstream_envs.yaml`: downstream maps created from source by swapping `F/H`, with holes disallowed next to `S` and `G`.
- `successful_ppo_settings.yaml`: successful PPO hyperparameters per environment size and evaluation outcomes.
- `generate_source_envs.py`: regenerates `source_envs.yaml`.
- `generate_downstream_envs.py`: regenerates `downstream_envs.yaml` from `source_envs.yaml`.
- `sweep_scaled_ppo.py`: reruns the training sweep and rewrites `successful_ppo_settings.yaml`.
- `train_policy_and_plot.py`: trains one selected layout from the stored config and saves the policy trajectory plot.
- `evaluate_post_hoc.py`: evaluates saved source/downstream policies for a selected layout and seed, writes `post_hoc_eval.yaml`, and saves one trajectory figure per evaluated `(policy, environment)` pair.

## Notes
- All successful settings were validated with deterministic raw-environment evaluation (`20` episodes, mean reward `1.0`).
- Training uses:
  - coordinate observations (`row`, `col`, `task`), and
  - dense reward shaping during training only.
- Final success is measured on the original sparse-reward environment (no reward shaping).

## Reproduce
```bash
python experiments/pipelines/frozenlake_scaled/generate_source_envs.py
python experiments/pipelines/frozenlake_scaled/generate_downstream_envs.py
python experiments/pipelines/frozenlake_scaled/sweep_scaled_ppo.py --seed 0
python experiments/pipelines/frozenlake_scaled/train_policy_and_plot.py --layout diagonal_30x30 --seed 0
python experiments/pipelines/frozenlake_scaled/evaluate_post_hoc.py --layout diagonal_30x30 --seed 0 --policies both --eval-mode matching --episodes 1
# Optional ablation: --activation relu (default is --activation tanh)
# Optional speed-up: add --skip-plots to skip trajectory figure generation.
```
