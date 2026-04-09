# PoisonedApple Experiments

## Entry Points
- `run_train_and_adapt.py`: run one configuration/seed end-to-end.
- `run_train_and_adapt_multi_seed.py`: run one configuration over multiple seeds (default `0..9`).
- `postprocessing/aggregate_metrics_across_seeds.py`: aggregate across seeds and export CSV/LaTeX.

## Quick Start
Single seed:
```bash
python rl_project/experiments/poisoned_apple/run_train_and_adapt.py --cfg simple_6x6 --seed 0
```

Multi-seed:
```bash
python rl_project/experiments/poisoned_apple/run_train_and_adapt_multi_seed.py --cfg simple_6x6
```

Dry-run multi-seed commands:
```bash
python rl_project/experiments/poisoned_apple/run_train_and_adapt_multi_seed.py --cfg simple_6x6 --dry-run
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
