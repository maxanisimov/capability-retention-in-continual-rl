#!/usr/bin/env bash
set -euo pipefail

# Adjust these
SEEDS=(1 2 3 4 5 6 7 8 9)
CFG="diagonal_16x16"

mkdir -p "logs/${CFG}"

# Prevent PyTorch/BLAS oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1

# Detect one logical CPU per physical core.
if command -v lscpu >/dev/null 2>&1; then
  mapfile -t CORES < <(lscpu -p=CPU,CORE | awk -F, '!/^#/ && !seen[$2]++ {print $1}')
else
  echo "WARNING: lscpu not found; falling back to all logical CPUs."
  mapfile -t CORES < <(seq 0 "$(( $(nproc) - 1 ))")
fi

if [ "${#CORES[@]}" -eq 0 ]; then
  echo "ERROR: Could not detect CPU cores."
  exit 1
fi

# Cap concurrency so each job has a dedicated core.
MAX_JOBS="${MAX_JOBS:-${#CORES[@]}}"
if [ "$MAX_JOBS" -le 0 ]; then
  echo "ERROR: MAX_JOBS must be >= 1"
  exit 1
fi
if [ "$MAX_JOBS" -gt "${#CORES[@]}" ]; then
  MAX_JOBS="${#CORES[@]}"
fi

trap 'jobs -pr | xargs -r kill; wait; exit 130' INT TERM

echo "Config: ${CFG}"
echo "Seeds: ${SEEDS[*]}"
echo "Detected physical cores: ${#CORES[@]}"
echo "Max concurrent jobs: ${MAX_JOBS}"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  core="${CORES[$((i % MAX_JOBS))]}"

  # Throttle launches to avoid CPU overload.
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 1
  done

  mkdir -p "logs/${CFG}/${seed}"

  taskset -c "$core" \
    python3 run_train_and_adapt.py --cfg "$CFG" --seed "$seed" \
    > "logs/${CFG}/${seed}/run.log" 2>&1 &

  echo "Launched seed=${seed} on cpu=${core}"
done

wait
echo "All runs completed."
