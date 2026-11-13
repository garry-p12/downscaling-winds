#!/bin/bash
#SBATCH -J wind-diff
#SBATCH -o logs/wind_diff_%j.out
#SBATCH -e logs/wind_diff_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
# Residual diffusion on the station-fusion runs (WIND_WORK_LOG 7d, path B): train, sample 25 members of 2023 on both
# fold stores, score. Needs the obs training (slurm/vista_wind_obs.sh) to have finished. Resumes from its checkpoint.
set -uo pipefail
cd "${WORK}/downscaling"; mkdir -p logs
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python" DOWNSCALE_MAX_WORKERS=8 PYTHONUNBUFFERED=1 DOWNSCALE_DEVICE=cuda
$PYTHON -u scripts/wind_diffusion.py --epochs "${EPOCHS:-60}" --tag "${TAG:-obs}" || echo "!!! diffusion failed"
echo "=== done $(date)"
