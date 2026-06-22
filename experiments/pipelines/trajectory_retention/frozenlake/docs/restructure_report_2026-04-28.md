# FrozenLake Directory Reorganization Report (2026-04-28)

## Completed Work

- Added `core/` implementation packages for env helpers, methods, evaluation, analysis, and orchestration.
- Added `cli/` wrappers and preserved top-level legacy script filenames as delegates.
- Moved settings into grouped folders and kept legacy settings filenames as symlinks.
- Added `artifacts/runs` defaults with `outputs/` and legacy `source` fallback resolution.
- Added a deprecated `experiments.pipelines.frozenlake_scaled` compatibility package.
- Added unit coverage for task loading, path fallback, full-pipeline scheduling, and aggregate metrics.

## Compatibility Notes

- Existing output data is not moved.
- New source-policy outputs use `noadapt`; readers fall back to legacy `source`.
- `--pipeline` and `--layout` are equivalent in new command surfaces.

