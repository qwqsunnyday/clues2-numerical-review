#!/bin/bash
#SBATCH --job-name=clues_bounds_v02
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm/gate_%j.out
#SBATCH --error=logs/slurm/gate_%j.err
set -euo pipefail
umask 077
cd /REVIEW_WORKSPACE/numerical_fix_v01/revision_v02
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_BOUNDSCHECK=1
export NUMBA_CACHE_DIR="$PWD/gate_results/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"
/REVIEW_ENV/venv/bin/python -B -m unittest discover -s tests -v
printf 'PASS\n' > gate_results/NUMERICAL.PASS
