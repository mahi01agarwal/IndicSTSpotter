#!/usr/bin/env bash
#SBATCH --job-name=mlt19_aria
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# Re-fetch MLT19 Part1 + GT with aria2c (Part2 already downloaded). Then unzip all.
# Submit from ada LOGIN node:  sbatch ~/download_mlt19_aria.sbatch
# Needs download_aria.sh in ~ too.  scancel the old wget job first.
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

DEST="$SCR/mlt19"
rm -f "$DEST/ImagesPart1.zip"        # drop the slow wget partial; aria2c fetches clean

DEST="$DEST" bash "$HOME/download_aria.sh" \
  "https://datasets.cvc.uab.es/rrc/ImagesPart1.zip" \
  "https://datasets.cvc.uab.es/rrc/train_gt_t13.zip"

echo "[unzip] ..."
cd "$DEST"
unzip -n -q ImagesPart1.zip  -d ImagesPart1
unzip -n -q ImagesPart2.zip  -d ImagesPart2
unzip -n -q train_gt_t13.zip -d gt
echo "[done] P1=$(find ImagesPart1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)" \
     "P2=$(find ImagesPart2 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)" \
     "gt=$(find gt -name 'gt_*.txt' | wc -l)"
