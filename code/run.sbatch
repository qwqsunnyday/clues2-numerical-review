#!/bin/bash
#SBATCH --job-name=clues_fixed_v02
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/slurm/fixed_%A_%a.out
#SBATCH --error=logs/slurm/fixed_%A_%a.err
set -euo pipefail
umask 077
trap 'rc=$?; echo "FINISH run=clues_numerical_fix_v02 job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID exit=$rc time=$(date -Is)"; exit "$rc"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /REVIEW_WORKSPACE/numerical_fix_v01/revision_v02
echo "START job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) time=$(date -Is)"
py=/REVIEW_ENV/venv/bin/python
"$py" -B -m unittest discover -s tests -p test_profile_search.py -v
"$py" -B -u code/run_fixed.py --config config.json --index "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK" --output-root "$PWD/results_r03"
