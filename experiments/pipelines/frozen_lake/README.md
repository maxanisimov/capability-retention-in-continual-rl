# FrozenLake Experiments

## Entry Points
- `run_train_and_adapt.py`: run one configuration/seed end-to-end.
- `run_train_and_adapt_multi_seed.py`: run one configuration over multiple seeds (default `0..9`).
- `postprocessing/`: aggregation, plotting, and LaTeX table scripts.

## Quick Start
Single seed:
```bash
python experiments/pipelines/frozen_lake/run_train_and_adapt.py --cfg standard_4x4 --seed 0
```

Multi-seed:
```bash
python experiments/pipelines/frozen_lake/run_train_and_adapt_multi_seed.py --cfg standard_4x4
```

Dry-run multi-seed commands:
```bash
python experiments/pipelines/frozen_lake/run_train_and_adapt_multi_seed.py --cfg standard_4x4 --dry-run
```

## Output Layout
```text
outputs/<cfg>/<seed>/
├── source/
├── downstream/
├── plots/
└── results_table.csv
```

Logs are written to `logs/<cfg>/<seed>/...`.
