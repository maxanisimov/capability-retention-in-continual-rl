# capability-retention-in-continual-rl

Codebase for capability-retention experiments in continual reinforcement learning, including certified/Rashomon-set tooling and experiment pipelines.

This repository is organised into three top-level areas:
- `core/` for reusable certified continual-learning functionality and utilities,
- `tutorials/` for tutorial notebooks,
- `projects/` for project-specific experiment pipelines, utilities, and artifacts.

## Projects
- `projects/safe_crl/` contains the current safe continual RL experiment tree.
- `projects/safe_policy_optimisation/` is scaffolded for future safe policy optimisation experiments.

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
├── tutorials/                         # Tutorial notebooks
├── projects/
│   ├── safe_crl/
│   │   ├── utils/                     # Project-specific RL utilities
│   │   ├── pipelines/                 # Experiment pipelines
│   │   ├── artifacts/                 # Pipeline artifacts
│   │   ├── notebooks/                 # Research notebooks
│   │   └── scripts/                   # Older exploratory scripts
│   └── safe_policy_optimisation/
│       ├── utils/
│       ├── pipelines/
│       └── artifacts/
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
from projects.safe_crl.utils.ppo_utils import PPOConfig
```

Legacy imports remain supported during migration:

```python
from src.trainer import IntervalTrainer
```

## Running Experiments
See `projects/safe_crl/pipelines/README.md` and pipeline-local READMEs for entry points. Run commands from the repository root so imports resolve through the canonical `projects.safe_crl` package path.

## Notes
- `archive/` directories contain older exploratory material and are not part of the main release pipeline.
- `projects/safe_crl/scripts/` includes older prototypes; prefer the experiment entry points above for reproducibility.
