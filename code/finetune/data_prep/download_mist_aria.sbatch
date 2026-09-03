#!/usr/bin/env bash
#SBATCH --job-name=mist_aria
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH -c 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# MIST (CVIT, WACV'26) — multilingual incidental scene-text DETECTION dataset.
# Single direct zip on the open CVIT host (no login/cookie): ~42.3 GB, 12,373 images
# (Train 10,379 / Test 1,985 benchmark), LabelMe-JSON polygon GT, 11 scripts + English, 1920x1080.
# Layout inside the zip:
#     Images/{Train,Test,Test_stale_*}/<stem>.png
#     gt/{Train,Test,Test_stale_*}/<stem>.json
# We keep only Train + Test; the *_stale_* dirs are superseded re-annotations (deleted below).
# NO GPU requested (CPU-only aria2+unzip) -> runs concurrently with the 4-GPU rec job on gnode063,
# won't queue behind it. Submit from ada LOGIN node:  sbatch ~/download_mist_aria.sbatch  (needs ~/download_aria.sh)
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

DEST="$SCR/mist"

DEST="$DEST" bash "$HOME/download_aria.sh" \
  "https://cvit.iiit.ac.in/images/datasets/mist/mist.zip"

echo "[unzip] mist.zip -> $DEST (this is ~42 GB; needs plenty of free space on \$SCR) ..."
cd "$DEST"
unzip -n -q mist.zip

echo "[prune] removing superseded *_stale_* re-annotation dirs ..."
rm -rf Images/*_stale_* gt/*_stale_* || true

echo "[done] counts:"
echo "  Images/Train: $(find Images/Train -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l)"
echo "  Images/Test : $(find Images/Test  -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l)"
echo "  gt/Train    : $(find gt/Train -type f -name '*.json' 2>/dev/null | wc -l)"
echo "  gt/Test     : $(find gt/Test  -type f -name '*.json' 2>/dev/null | wc -l)"

echo "[peek] one Train GT (verify schema: shapes[].points / care / script) BEFORE converting:"
PEEK=$(find gt/Train -type f -name '*.json' -print -quit)   # -quit: first match, no SIGPIPE (pipefail-safe)
echo "  file: $PEEK"
python - "$PEEK" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1], encoding='utf-8'))
print("  top-level keys:", list(d.keys()))
sh = d.get('shapes', [])
print("  #shapes:", len(sh))
if sh:
    print("  shape[0] keys:", list(sh[0].keys()))
    print("  shape[0] care/script/#pts:", sh[0].get('care'), sh[0].get('script'), len(sh[0].get('points', [])))
    print("  care values:", dict(collections.Counter(s.get('care') for s in sh)))
    print("  scripts:", dict(collections.Counter(str(s.get('script')).lower() for s in sh)))
PY
echo "[note] if the schema matches (points/care/script), run mist_to_db.py for Train and Test."
