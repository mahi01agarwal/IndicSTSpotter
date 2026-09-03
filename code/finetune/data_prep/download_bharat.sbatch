#!/usr/bin/env bash
#SBATCH --job-name=bharat_dl
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --mail-type=END

#SBATCH --output=%x_%j.log
# Fully-detached Bharat download (follows the standard ada SLURM template; no --qos).
# Pinned to gnode063 so $SCR (node-local scratch) resolves to the same place every time.
# Submit from the ada LOGIN node:  sbatch ~/download_bharat.sbatch
# Both this file AND download_bharat.sh must sit in ~ (login node can't read $SCR).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"
DEST="$SCR/bharat" bash "$HOME/download_bharat.sh"
