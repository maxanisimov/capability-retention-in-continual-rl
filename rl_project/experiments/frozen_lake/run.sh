#!/usr/bin/env bash
set -euo pipefail

# Adjust these
SEEDS=(0 1 2 3 4 5 6 7 8 9)
CORES=(0 1 2 3 4 5 6 7 8 9)   # physical/logical cores to use
CFG="diagonal_16x16"

mkdir -p "logs/${CFG}"

# Prevent PyTorch/BLAS oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

trap 'jobs -pr | xargs -r kill; wait; exit 130' INT TERM

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  core="${CORES[$((i % ${#CORES[@]}))]}"
  
  mkdir -p "logs/${CFG}/${seed}"

  taskset -c "$core" \
    python3 run_train_and_adapt.py --cfg "$CFG" --seed "$seed" \
    > "logs/${CFG}/${seed}/run.log" 2>&1 &
done

wait
