#!/bin/bash
#SBATCH -J wind-maps
#SBATCH -o logs/wind_maps_%j.out
#SBATCH -e logs/wind_maps_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM23014
# Map sheets that need more memory than a login node allows (the 1 km store): scripts/wind_inference_maps.py with
# the arguments in $MAPARGS, e.g.  sbatch --export=ALL,MAPARGS="--runs w1_L_s1337 --only ladder testyear" slurm/vista_wind_maps.sh
set -uo pipefail
cd "${WORK}/downscaling"; mkdir -p logs
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python" DOWNSCALE_MAX_WORKERS=8 PYTHONUNBUFFERED=1
: "${MAPARGS:?set MAPARGS}"
$PYTHON -u scripts/wind_inference_maps.py $MAPARGS || echo "!!! maps failed"
echo "=== done $(date)"
