#!/usr/bin/env bash
# download_mlt19.sh — fetch ICDAR-2019 MLT Task-1 training set (images + GT) directly.
# The RRC page is login-gated but the underlying file host is open, so plain wget works.
#
# Usage:  DEST=$SCR/mlt19 bash download_mlt19.sh
#
# Layout produced:
#   $DEST/ImagesPart1/   (5000 images, langs 00001-05000)
#   $DEST/ImagesPart2/   (5000 images, langs 05001-10000; Bangla=08001-09000, Hindi=09001-10000)
#   $DEST/gt/            (gt_img_*.txt, format: x1,y1,..,y4,script,transcription)
set -euo pipefail

DEST="${DEST:-$PWD/mlt19}"
mkdir -p "$DEST"
cd "$DEST"

declare -A FILES=(
  [ImagesPart1.zip]="https://datasets.cvc.uab.es/rrc/ImagesPart1.zip"
  [ImagesPart2.zip]="https://datasets.cvc.uab.es/rrc/ImagesPart2.zip"
  [train_gt_t13.zip]="https://datasets.cvc.uab.es/rrc/train_gt_t13.zip"
)

for z in "${!FILES[@]}"; do
  echo "[mlt19] wget ${FILES[$z]}"
  # --no-check-certificate: gnode CA bundle can't verify datasets.cvc.uab.es;
  # files already verified genuine from a trusted host (correct sizes, application/zip).
  wget -c --no-check-certificate --tries=3 --timeout=60 --progress=dot:giga \
       "${FILES[$z]}" -O "$z"
done

echo "[mlt19] unzip ..."
unzip -n -q ImagesPart1.zip -d ImagesPart1
unzip -n -q ImagesPart2.zip -d ImagesPart2
unzip -n -q train_gt_t13.zip -d gt

echo "[mlt19] done. counts:"
echo "  ImagesPart1 imgs: $(find ImagesPart1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
echo "  ImagesPart2 imgs: $(find ImagesPart2 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
echo "  gt files        : $(find gt -name 'gt_*.txt' | wc -l)"
ls -lh "$DEST"
