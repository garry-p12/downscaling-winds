#!/bin/bash
#SBATCH -J wind-obs4
#SBATCH -o logs/wind_obs4_%j.out
#SBATCH -e logs/wind_obs4_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
# Two-stage control (WIND_WORK_LOG 7d): train plan group "obs4" (DeepSD LO fusion, 3 seeds) on 2019-22 instead of
# 2019-21, then rescore all fusion runs (obs_eval, with seed means) and compare with the 2022 corrector
# (scripts/wind_two_stage.py). Requires the opt-in DOWNSCALE_TRAIN_SPLITS in training/train.py and training/dataset.py.
set -uo pipefail
cd "${WORK}/downscaling"; mkdir -p logs
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python" DOWNSCALE_MAX_WORKERS=8 PYTHONUNBUFFERED=1
DOWNSCALE_TRAIN_SPLITS=train,val WGROUPS=obs4 bash slurm/vista_wind_train.sh || echo "!!! some obs4 training failed"
grep -h "training splits" logs/wind_obs4_${SLURM_JOB_ID}.out | head -1 || echo "!!! split override not seen"
export DOWNSCALE_DEVICE=cuda
$PYTHON -u scripts/wind_obs_eval.py || echo "!!! obs eval failed"
$PYTHON -u scripts/wind_two_stage.py || echo "!!! two-stage comparison failed"
$PYTHON -u scripts/wind_inference_maps.py --runs w10o0_LO4y_s1337 --only inference testyear || echo "!!! maps failed"
echo "=== done $(date)"
