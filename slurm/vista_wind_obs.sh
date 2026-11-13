#!/bin/bash
#SBATCH -J wind-obs
#SBATCH -o logs/wind_obs_%j.out
#SBATCH -e logs/wind_obs_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
# Station-fusion experiment (WIND_WORK_LOG 7c/7d): train plan group "obs" on the fold-0 store, then score every
# fusion run on both fold stores against POWER, AORC, kernel interpolation and the no-station models, then draw maps.
# Prerequisite (login node): python scripts/wind_obs_store.py   (builds data_store_sc/wind_obs_f{0,1})
# Resumable: training skips runs with a checkpoint; the eval is cheap and simply reruns.
set -uo pipefail
cd "${WORK}/downscaling"; mkdir -p logs
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python" DOWNSCALE_MAX_WORKERS=8 PYTHONUNBUFFERED=1
WGROUPS=obs bash slurm/vista_wind_train.sh || echo "!!! some obs training failed"
export DOWNSCALE_DEVICE=cuda
$PYTHON -u scripts/wind_obs_eval.py || echo "!!! obs eval failed"
$PYTHON -u scripts/wind_inference_maps.py --runs w10o0_LO_s1337 w10o0_A8O_s1337 w10o0_unet_LO_s1337 \
    --only inputs inference testyear || echo "!!! maps failed"
echo "=== done $(date)"
