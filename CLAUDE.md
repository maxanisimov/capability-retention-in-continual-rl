# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`capability-retention-in-continual-rl` studies capability retention in continual reinforcement learning, including certified/Rashomon-set tooling and experiment pipelines.

- `core/certified_continual_learning/` — canonical package API (e.g. `from certified_continual_learning.trainer import IntervalTrainer`). Use this for new code.
- `core/src/` — legacy import path (`from src.trainer import IntervalTrainer`), kept for backward compatibility during migration. Don't add new code here.
- `core/abstract_gradient_training/` — bounded/certified training components (interval/zonotope arithmetic).
- `experiments/utils/` — reusable RL utilities (PPO/EWC/shielding, MASA tabular envs, plotting).
- `experiments/pipelines/` — paper-facing experiment pipelines, organized into canonical families (see `experiments/pipelines/README.md`):
  - `behaviour_retention/<env>/` — preserve source-task behaviour from source-policy trajectory data (e.g. `frozenlake/`, `lunarlander/`, `frozenlake_safety_constrained/`).
  - `safety/<env>/` — shield-generated safety datasets (e.g. `frozenlake/`, `frozenlake_slippery/`, `lavacrossing/`).
  - Each pipeline dir has its own `run_experiment.py`, a `cli/` with multi-seed launchers, and a `README.md` documenting exact invocation — read that README before running, command-line args differ per pipeline.
  - Top-level `experiments/pipelines/frozenlake/`, `lunarlander/`, `frozenlake_safety/`, etc. (without a family prefix) are compatibility delegates; use the canonical `behaviour_retention/`/`safety/` path for new work.
- `experiments/scripts/` and `archive/` directories — older exploratory prototypes, not part of the maintained pipeline. Don't extend these; prefer the pipeline entry points.

Run all commands from the repository root.

**Note:** the root `README.md`'s "Running Main Experiments" section (referencing `experiments/pipelines/frozen_lake/run_train_and_adapt.py`) is stale and doesn't match the current pipeline layout above — trust `experiments/pipelines/README.md` and the per-pipeline `README.md` instead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Testing

Tests use `unittest`, not `pytest`, and follow a `*_unittest.py` naming convention (not `test_*.py` in most of `experiments/`). Run the suite for the module you touched:

```bash
PYTHONPATH=. python -m unittest discover -s <module_dir> -p "*unittest*" -v
```

e.g. `PYTHONPATH=. python -m unittest discover -s experiments/utils -p "*unittest*" -v`

Always run the relevant unittest suite after changing code in `core/` or `experiments/` before considering the change done.

## Lint/Format

Ruff (line length 88, double quotes, import sorting) is configured in `ruff.toml` and enforced via pre-commit (`.pre-commit-config.yaml`):

```bash
ruff check --fix
ruff format
```

## Code Style

- Type hints throughout, including PEP 585 generics (`tuple[float, float]`) — this requires Python >=3.10.
- Google-style docstrings with `Args:`/`Returns:` sections.
- Prefer the `certified_continual_learning.*` import path over the legacy `src.*` path in new code.

## Commit Style

Use Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.) for new commits going forward.
