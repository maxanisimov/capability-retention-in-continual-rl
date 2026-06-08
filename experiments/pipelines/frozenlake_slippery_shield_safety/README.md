# FrozenLake Slippery Shield Safety Pipeline

Small deterministic `diagonal_4x4` continual-learning pipeline with shield-generated safety demonstrations:

- NoAdapt source training plus supervised safety fine-tuning against synthesized shield masks.
- Unconstrained downstream PPO adaptation.
- EWC downstream adaptation.
- Rashomon downstream adaptation from shield-derived multi-hot action masks.
- SafeLineSearch PPO adaptation using a verified source-safety margin gate.
- Lagrangian PPO adaptation using the same verified source-safety margin as a penalty.
- PPO, EWC, and Rashomon training settings mirror the existing FrozenLake
  `diagonal_10x10` settings from `experiments/pipelines/frozenlake/settings`.
- PPO training uses dense reward shaping, while early stopping and final
  evaluation use sparse deterministic rewards. Success rate is reported but is
  not an early-stop criterion.
- Shield synthesis defaults to the deterministic almost-sure shield. Use
  `--shield-type probabilistic` with `--shield-risk-threshold` to generate
  risk-thresholded masks.

Run from the repository root with the project environment active:

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_experiment.py \
  --mode source \
  --pipeline diagonal_4x4 \
  --seed 0
```

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/run_experiment.py \
  --mode source \
  --pipeline diagonal_4x4 \
  --seed 0 \
  --shield-type probabilistic \
  --shield-risk-threshold 0.05
```

```bash
python experiments/pipelines/frozenlake_slippery_shield_safety/cli/launch_full_pipeline_multi_seed.py \
  --pipeline diagonal_4x4 \
  --seeds 0 1 2 \
  --cores 0 1 2
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
  --pipeline diagonal_10x10
```

Artifacts default to `experiments/pipelines/frozenlake_slippery_shield_safety/artifacts/runs`.
Initial-frame figures default to
`experiments/pipelines/frozenlake_slippery_shield_safety/artifacts/figures/initial_frames`.
