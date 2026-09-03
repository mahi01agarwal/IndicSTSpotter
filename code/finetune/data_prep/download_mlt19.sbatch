#!/usr/bin/env bash
#SBATCH --job-name=mlt19_dl
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# Detached MLT-2019 download (wget the open file host). Pinned to gnode063 so $SCR
# resolves to the same node-local scratch as the Bharat data.
# Submit from the ada LOGIN node:  sbatch ~/download_mlt19.sbatch
# Both this file AND download_mlt19.sh must sit in ~ (login node can't read $SCR).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
DEST="$SCR/mlt19" bash "$HOME/download_mlt19.sh"
