#!/bin/bash
#SBATCH --job-name=clues_traj_audit
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0,2%2
#SBATCH --output=logs/slurm/ablation_%A_%a.out
#SBATCH --error=logs/slurm/ablation_%A_%a.err
set -euo pipefail
trap 'rc=$?; echo "end_status=$rc job=$SLURM_JOB_ID"; exit "$rc"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /REVIEW_WORKSPACE/numerical_fix_v01/trajectory_v01
echo "run=trajectory_controlled_ablation_v01 job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID attempt=${SLURM_RESTART_COUNT:-0}"
/REVIEW_ENV/venv/bin/python code/audit_ablation.py --task "$SLURM_ARRAY_TASK_ID"
