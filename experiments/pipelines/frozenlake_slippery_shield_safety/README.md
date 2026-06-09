# FrozenLake Slippery Shield Safety Pipeline

Stochastic `FrozenLake-v1` continual-learning pipeline with shield-generated safety demonstrations and `is_slippery=True` dynamics:

- NoAdapt source training plus supervised safety fine-tuning against synthesized shield masks.
- Unconstrained downstream PPO adaptation.
- EWC downstream adaptation.
- Rashomon downstream adaptation from shield-derived multi-hot action masks.
- SafeLineSearch PPO adaptation using a verified source-safety margin gate.
- Lagrangian PPO adaptation using the same verified source-safety margin as a penalty.
- PPO training uses dense reward shaping, while early stopping and final
  evaluation use sparse slippery dynamics over 100 episodes by default.
- Early stopping uses success/failure-rate thresholds: success rate `>= 0.8`
  and failure rate `<= 0.2` after at least five PPO updates.
- Shield synthesis defaults to a probabilistic eventual-risk shield. Source
  safety targets use the minimum-risk actions from value iteration, so they do
  not depend on `shield_risk_threshold`; that threshold remains available for
  explicit thresholded shield visualisation and ablations.
- Slippery dynamics default to `success_rate=1/3`, matching Gymnasium
  FrozenLake. The chosen action is executed with probability `success_rate`;
  the remaining probability is split across the two side-slip directions.
- Source runs save `shield_safety_probabilities.png`, which overlays allowed
  shield actions on the source environment frame. Arrow colour is
  `1 - action_risk`, i.e. the probability of staying safe after that action.
- Supported layouts are `diagonal_4x4`, `diagonal_6x6`,
  `route_switch_6x6`, and `old_route_blocked_6x6`.

Run from the repository root with the project environment active:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_experiment.py \
  --mode source \
  --pipeline diagonal_6x6 \
  --seed 0 \
  --shield-risk-threshold 0.1 \
  --success-rate 0.8
```

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_experiment.py \
  --mode source \
  --pipeline diagonal_4x4 \
  --seed 0
```

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/cli/launch_full_pipeline_multi_seed.py \
  --pipeline diagonal_4x4 \
  --seeds 0 1 2 \
  --cores 0 1 2 \
  --success-rate 0.8
```

The full launcher pins each active seed pipeline to one CPU core for source
plus all downstream methods. If there are more seeds than selected cores, it
runs the remaining seeds in later waves.

Run one adaptation method across seeds in parallel, with one active seed pinned
to each selected CPU core:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/cli/launch_adaptation_multi_seed.py \
  --mode downstream_rashomon \
  --pipeline diagonal_4x4 \
  --seeds 0 1 2 \
  --cores 0 1 2
```

Run the verified-margin SafeLineSearch baseline for one seed:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_safe_line_search_ppo.py \
  --pipeline diagonal_4x4 \
  --seed 0
```

Run the verified-margin Lagrangian baseline for one seed:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_lagrangian_ppo.py \
  --pipeline diagonal_4x4 \
  --seed 0
```

Aggregate the default safety metrics after multi-seed runs:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/aggregate_metrics_frozenlake_slippery_shield_safety.py \
  --pipeline diagonal_4x4
```

Generate source/downstream initial-frame figures for one layout:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/plot_initial_frames.py \
  --pipeline diagonal_6x6
```

Synthesize a shield without training and plot it on the source frame:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/plot_synthesised_shield.py \
  --pipeline diagonal_4x4 \
  --success-rate 0.8 \
  --shield-risk-threshold 0.05
```

Artifacts default to `experiments/pipelines/frozenlake_slippery_shield_safety/artifacts/runs`.
Initial-frame figures default to
`experiments/pipelines/frozenlake_slippery_shield_safety/artifacts/figures/initial_frames`.
Synthesized shield figures default to
`experiments/pipelines/frozenlake_slippery_shield_safety/artifacts/figures/shields`.
