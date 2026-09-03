#!/usr/bin/env bash
#SBATCH --job-name=mlt17_aria
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# MLT17 (RRC ch=8) Task-1 TRAINING set: 8 image parts + localisation/script/transcription GT.
# All links are direct on the open host datasets.cvc.uab.es (aria2c, cert check off).
# Validation-set IMAGES are login-gated (cookie) so they're skipped here.
# Submit from ada LOGIN node:  sbatch ~/download_mlt17_aria.sbatch   (needs ~/download_aria.sh)
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

DEST="$SCR/mlt17"

DEST="$DEST" bash "$HOME/download_aria.sh" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_1.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_2.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_3.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_4.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_5.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_6.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_7.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_images_8.zip" \
  "https://datasets.cvc.uab.es/rrc/ch8_training_localization_transcription_gt_v2.zip"

echo "[unzip] images ..."
cd "$DEST"
for z in ch8_training_images_*.zip; do unzip -n -q "$z" -d images; done
echo "[unzip] gt ..."
unzip -n -q ch8_training_localization_transcription_gt_v2.zip -d gt

echo "[done] imgs=$(find images -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | wc -l)" \
     "gt=$(find gt -type f -name '*.txt' | wc -l)"
echo "[peek] sample image + gt names:"
find images -type f \( -iname '*.jpg' -o -iname '*.png' \) | head -3
find gt -type f -name '*.txt' | head -3
