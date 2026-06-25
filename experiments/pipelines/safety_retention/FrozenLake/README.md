# FrozenLake Safety Pipeline

Small deterministic `diagonal_4x4` continual-learning pipeline with:

- NoAdapt source training plus supervised safety fine-tuning.
- Unconstrained downstream PPO adaptation.
- EWC downstream adaptation.
- Rashomon-certificate-constrained downstream adaptation.
- PPO, EWC, and Rashomon training settings mirror the existing FrozenLake
  `diagonal_10x10` settings from `experiments/pipelines/trajectory_retention/frozenlake/settings`.
- PPO training uses dense reward shaping, while early stopping and final
  evaluation use sparse deterministic rewards. Success rate is reported but is
  not an early-stop criterion.

## One command: run + aggregate

```bash
python experiments/pipelines/safety_retention/frozenlake/run_pipeline.py \
  --pipeline diagonal_4x4 --rl ppo --deterministic \
  --methods unconstrained ewc rashomon --seeds 0 1 2 3 4
```

This launches source training plus the requested adaptation methods across
all seeds (CPU-pinned, one subprocess per seed), then writes an aggregated
CSV + LaTeX table of safety and performance metrics for both the source and
downstream tasks.

`--rl` only supports `ppo` today (see `core/methods/ppo/`; add a sibling
`core/methods/<rl>/` package to support another algorithm). `--deterministic`
only supports `True` today — `core/safety.py`'s Rashomon certificates and
safety-critical-state checks assume deterministic FrozenLake transitions, so
`--no-deterministic` is rejected until that machinery is reworked.

## Artifact layout

Runs are tagged by `--rl`/`--deterministic` so different settings never
collide:

```
artifacts/runs/<layout>/<rl>_<deterministic|stochastic>/seed_<n>/
  ├── noadapt/
  ├── downstream_unconstrained/
  ├── downstream_ewc/
  └── downstream_rashomon/
```

e.g. `artifacts/runs/diagonal_4x4/ppo_deterministic/seed_0/downstream_rashomon/`.

## One script per training run

Each method is its own runnable script and can be run standalone for a
single seed:

```bash
python experiments/pipelines/safety_retention/frozenlake/cli/train_source.py --pipeline diagonal_4x4 --seed 0
python experiments/pipelines/safety_retention/frozenlake/cli/adapt_unconstrained.py --pipeline diagonal_4x4 --seed 0
python experiments/pipelines/safety_retention/frozenlake/cli/adapt_ewc.py --pipeline diagonal_4x4 --seed 0
python experiments/pipelines/safety_retention/frozenlake/cli/adapt_rashomon.py --pipeline diagonal_4x4 --seed 0
```

## Running one seed's full pipeline

```bash
python experiments/pipelines/safety_retention/frozenlake/cli/run_seed_pipeline.py \
  --pipeline diagonal_4x4 --seed 0 --methods unconstrained ewc rashomon
```

Trains the source policy (skipped if `--resume-policy skip_completed` and
already complete), then runs each requested adaptation method in sequence.

## Multi-seed launcher

```bash
python experiments/pipelines/safety_retention/frozenlake/cli/launch_multi_seed.py \
  --pipeline diagonal_4x4 --methods unconstrained ewc rashomon \
  --seeds 0 1 2 3 4 --cores 0 1 2 3 4
```

Fans `run_seed_pipeline.py` out across seeds via the shared
`experiments/pipelines/_shared/multi_seed_launcher.py` scheduler (CPU-pinned,
one subprocess per seed, one log file per seed).

## Aggregating results

```bash
python experiments/pipelines/safety_retention/frozenlake/cli/aggregate_metrics.py --pipeline diagonal_4x4
```

Reads `run_summary.yaml` from every seed/method under the tagged run
directory and writes `aggregate_metrics_frozenlake_safety.{csv,tex}` with
mean ± std for safety and performance metrics on both the source and
downstream tasks.
