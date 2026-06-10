# FrozenLake Safety Pipeline

Small deterministic `diagonal_4x4` continual-learning pipeline with:

- NoAdapt source training plus supervised safety fine-tuning.
- Unconstrained downstream PPO adaptation.
- EWC downstream adaptation.
- PPO, EWC, and Rashomon training settings mirror the existing FrozenLake
  `diagonal_10x10` settings from `experiments/pipelines/behaviour_retention/frozenlake/settings`.
- PPO training uses dense reward shaping, while early stopping and final
  evaluation use sparse deterministic rewards. Success rate is reported but is
  not an early-stop criterion.

Run from the repository root with the project environment active:

```bash
python experiments/pipelines/behaviour_retention/frozenlake_safety_constrained/run_experiment.py \
  --mode source \
  --pipeline diagonal_4x4 \
  --seed 0
```

```bash
python experiments/pipelines/behaviour_retention/frozenlake_safety_constrained/cli/launch_full_pipeline_multi_seed.py \
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
python experiments/pipelines/behaviour_retention/frozenlake_safety_constrained/cli/launch_adaptation_multi_seed.py \
  --mode downstream_rashomon \
  --pipeline diagonal_4x4 \
  --seeds 0 1 2 \
  --cores 0 1 2
```

Aggregate the default safety metrics after multi-seed runs:

```bash
python experiments/pipelines/behaviour_retention/frozenlake_safety_constrained/aggregate_metrics_frozenlake_safety.py \
  --pipeline diagonal_4x4
```

Artifacts default to `experiments/pipelines/behaviour_retention/frozenlake_safety_constrained/artifacts/runs`.
