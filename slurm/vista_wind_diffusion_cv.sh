#!/bin/bash
#SBATCH -J wind-diffcv
#SBATCH -o logs/wind_diffcv_%j.out
#SBATCH -e logs/wind_diffcv_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A ATM23014
# Spread calibration for the 2022-residual diffusion (WIND_WORK_LOG 7d): two cross-fit models on alternating months
# of 2022, each sampling the months it did not see, then fit the inflation on those out-of-fold members and score the
# inflated 2023 ensemble (scripts/wind_diffusion_calib.py). Needs the obsval run (job 1143913) to exist.
set -uo pipefail
cd "${WORK}/downscaling"; mkdir -p logs
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python" DOWNSCALE_MAX_WORKERS=8 PYTHONUNBUFFERED=1 DOWNSCALE_DEVICE=cuda
CV="${CV:-obsvalcv}" EPOCHS="${EPOCHS:-600}"
ODD="1 3 5 7 9 11"; EVEN="2 4 6 8 10 12"
$PYTHON -u scripts/wind_diffusion.py --train-split val --months $ODD --sample-split val --sample-months $EVEN \
    --epochs "$EPOCHS" --tag "${CV}a" || echo "!!! cv half a failed"
$PYTHON -u scripts/wind_diffusion.py --train-split val --months $EVEN --sample-split val --sample-months $ODD \
    --epochs "$EPOCHS" --tag "${CV}b" || echo "!!! cv half b failed"
$PYTHON -u scripts/wind_diffusion_calib.py --cv "$CV" --tag obsval || echo "!!! calibration failed"
echo "=== done $(date)"
