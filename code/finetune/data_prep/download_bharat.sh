#!/usr/bin/env bash
# download_bharat.sh — fetch the Bharat Scene Text Dataset (BSTD) from Google Drive.
#
# Run on a box with internet + disk space (recommended: an ada gnode session, in tmux).
# gvlab can't reach external mirrors reliably -> download on ada/nltm, then relay.
#
# Usage:
#   DEST=/ssd_scratch/cvit/mahima/bharat bash download_bharat.sh            # both zips
#   DEST=... WHAT=recognition bash download_bharat.sh                       # recognition only
#   DEST=... WHAT=detection   bash download_bharat.sh                       # detection only
#
# What you get:
#   recognition.zip -> ready cropped-word split (train/test_recognition_data.json) -> RECOGNIZER
#   detection.zip   -> full images + BSTD_v17.57.json polygon GT (~17 GB)          -> DETECTOR
set -euo pipefail

DEST="${DEST:-$PWD/bharat}"
WHAT="${WHAT:-both}"                 # both | recognition | detection

# Google Drive file IDs (from the BSTD GitHub README)
REC_ID="1d8yOLWrStRTmB8nIJG3mi-w5P9IzB_z8"
DET_ID="1S7KUYfB-lQvbu6GtvZxxDPOD5ZZ080M0"

mkdir -p "$DEST"
cd "$DEST"

# --- ensure gdown (handles Drive's large-file confirm token) ---
if ! python -c "import gdown" 2>/dev/null; then
  echo "[bharat] installing gdown ..."
  python -m pip install --user -U gdown
fi

fetch () {  # $1=file_id  $2=output_name
  local id="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "[bharat] $out already present ($(du -h "$out" | cut -f1)) — skipping download"
  else
    echo "[bharat] downloading $out ..."
    python -m gdown "https://drive.google.com/uc?id=${id}" -O "$out"
  fi
  echo "[bharat] unzipping $out ..."
  unzip -n -q "$out" -d "${out%.zip}" && echo "[bharat] -> ${DEST}/${out%.zip}/"
}

case "$WHAT" in
  recognition) fetch "$REC_ID" recognition.zip ;;
  detection)   fetch "$DET_ID" detection.zip ;;
  both)        fetch "$REC_ID" recognition.zip
               fetch "$DET_ID" detection.zip ;;
  *) echo "unknown WHAT=$WHAT (use both|recognition|detection)"; exit 1 ;;
esac

echo
echo "[bharat] done. Contents of $DEST:"
ls -lh "$DEST"
echo
echo "[bharat] sanity — recognition JSONs:"
find "$DEST/recognition" -maxdepth 2 -name '*recognition_data.json' 2>/dev/null || true
echo "[bharat] sanity — detection annotation:"
find "$DEST/detection" -maxdepth 2 -name 'BSTD_*.json' 2>/dev/null || true
