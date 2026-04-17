# capability-retention-in-continual-rl

Codebase for capability-retention experiments in continual reinforcement learning, including certified/Rashomon-set tooling and experiment pipelines.

This repository contains:
- colleague/core reusable work under `core/`,
- the canonical Python package API (`core/certified_continual_learning/`),
- the core certification/training code (`core/src/`, `core/abstract_gradient_training/`),
- reusable RL utilities (`experiments/utils/`),
- paper-facing experiment pipelines in `experiments/pipelines/`.

## What This Repo Is For
The main public entry points are the two safe continual RL environment pipelines:
- `FrozenLake`
- `PoisonedApple`

Each pipeline supports:
- single-seed runs,
- multi-seed runs,
- postprocessing/aggregation scripts.

## Repository Structure
```text
.
├── core/
│   ├── abstract_gradient_training/    # Bounded/certified training components
│   ├── certified_continual_learning/  # Canonical package API layer
│   ├── src/                           # Legacy compatibility import path
│   ├── configs/                       # Trainer/config presets
│   ├── scripts/                       # Colleague/tutorial scripts
│   └── notebooks/                     # Colleague/tutorial notebooks
├── experiments/
│   ├── utils/                         # PPO/EWC/env plotting helper utilities
│   ├── pipelines/
│   │   ├── frozen_lake/              # FrozenLake experiment pipeline
│   │   ├── poisoned_apple/           # PoisonedApple experiment pipeline
│   │   ├── frozenlake_scaled/        # Scaled FrozenLake experiments
│   │   ├── lunarlander/              # LunarLander experiments
│   │   ├── highway/                  # Highway parking experiments
│   │   ├── visualisations/           # Figure/frame generation helpers
│   │   └── README.md                 # Experiment quick-start guide
│   ├── notebooks/                     # Research notebooks
│   └── scripts/                       # Older exploratory scripts
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Installation / Setup
From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Notes:
- The requirements file is comprehensive (research/dev environment).
- Run commands from repository root for consistent relative paths.

## Import Migration Notes
New canonical imports should use:

```python
from certified_continual_learning.trainer import IntervalTrainer
```

Legacy imports remain supported during migration:

```python
from src.trainer import IntervalTrainer
```

## Running Main Experiments
See also: `experiments/pipelines/README.md`.

### FrozenLake
Single seed:
```bash
python experiments/pipelines/frozen_lake/run_train_and_adapt.py \
  --cfg standard_4x4 --seed 0
```

Multi-seed (`0..9` default):
```bash
python experiments/pipelines/frozen_lake/run_train_and_adapt_multi_seed.py \
  --cfg standard_4x4
```

Postprocessing examples:
```bash
python experiments/pipelines/frozen_lake/postprocessing/aggregate_downstream_results.py \
  --base-dir experiments/pipelines/frozen_lake/outputs/standard_4x4
```

### PoisonedApple
Single seed:
```bash
python experiments/pipelines/poisoned_apple/run_train_and_adapt.py \
  --cfg simple_6x6 --seed 0
```

Multi-seed (`0..9` default):
```bash
python experiments/pipelines/poisoned_apple/run_train_and_adapt_multi_seed.py \
  --cfg simple_6x6
```

Postprocessing example:
```bash
python experiments/pipelines/poisoned_apple/postprocessing/aggregate_metrics_across_seeds.py \
  --cfg simple_6x6
```

## Outputs / Results Locations
Per-run artifacts are stored under:

- FrozenLake: `experiments/pipelines/frozen_lake/outputs/<cfg>/<seed>/`
- PoisonedApple: `experiments/pipelines/poisoned_apple/outputs/<cfg>/<seed>/`

Typical contents:
- `source/` (source policy artifacts)
- `downstream/` (adaptation artifacts)
- `plots/` (generated figures)
- `results_table.csv` (run-level summary table)

Logs are stored under:
- `experiments/pipelines/<env>/logs/...`

## Notes
- `archive/` directories contain older exploratory material and are not part of the main release pipeline.
- `experiments/scripts/` includes older prototypes; prefer the experiment entry points above for reproducibility.
