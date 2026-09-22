#!/bin/bash
#SBATCH --job-name=clues_traj_v02
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/trajectory_%A_%a.out
#SBATCH --error=logs/slurm/trajectory_%A_%a.err
set -euo pipefail
trap 'rc=$?; echo "end_status=$rc job=$SLURM_JOB_ID"; exit "$rc"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /REVIEW_WORKSPACE/numerical_fix_v01/revision_v02/trajectory_v02
export NUMBA_BOUNDSCHECK=1
/REVIEW_ENV/venv/bin/python -B -m unittest discover -s tests -v
echo "run=conditional_trajectory_v02 job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID attempt=${SLURM_RESTART_COUNT:-0}"
/REVIEW_ENV/venv/bin/python code/run_trajectory.py --task "$SLURM_ARRAY_TASK_ID"
