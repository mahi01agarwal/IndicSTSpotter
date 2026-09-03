#!/usr/bin/env bash
#SBATCH --job-name=v6det_real
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# v6 DETECTOR fine-tune on ada, REAL-ONLY (Bharat+MLT17+MLT19), 4 GPUs (paddle.distributed).
# Warm-starts from PP-OCRv6_medium_det. Real det labels are already in PaddleOCR format with
# ABSOLUTE image paths, so data_dir=/ and make_det_labels.py is NOT used (that's synth-only).
#
# Prereqs: merged labels exist (run merge_det_db.py first, see below), $SCR/PaddleOCR repo,
#          $SCR/ocr env (Paddle 3.1), make_det_config.py at $SCR/FineTune/make_det_config.py,
#          pretrained weight at $SCR/PP-OCRv6_medium_det.pdparams.
# Submit from ada LOGIN node:  sbatch ~/train_v6_det_ada.sbatch
# Resume after stop: re-submit (auto-detects latest checkpoint).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

# ---- knobs ----
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-8}                 # per-card; effective = 4*BATCH. Drop to 4 if OOM.
EPOCHS=${EPOCHS:-100}             # best_accuracy is picked on the real val set; can scancel once it plateaus
MERGED=${MERGED:-$SCR/det_data_real/merged_real}
OUTDIR=${OUTDIR:-./output/v6_det_real}
CONFIG=${CONFIG:-my_v6_det_real.yml}

FT=$SCR/FineTune
REPO=$SCR/PaddleOCR
BASE=configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml
PRETRAINED=$SCR/PP-OCRv6_medium_det.pdparams

cd "$REPO"
echo "[node] $(hostname)"; nvidia-smi -L || true

# ---- sanity ----
[ -f "$BASE" ] || { echo "[fatal] base config missing: $REPO/$BASE"; exit 1; }
[ -f "$FT/make_det_config.py" ] || { echo "[fatal] missing $FT/make_det_config.py — sync it to \$SCR/FineTune"; exit 1; }
[ -f "$MERGED/det_train.txt" ] || { echo "[fatal] no merged labels at $MERGED — run merge_det_db.py first"; exit 1; }
if [ ! -f "$PRETRAINED" ]; then
    echo "[fetch] pretrained weight missing — pulling from bcebos (ada gnode has internet)"
    wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams \
        -O "$PRETRAINED" || { echo "[fatal] weight fetch failed"; exit 1; }
fi

# ---- write config (data_dir=/ because label paths are absolute) ----
echo "[prep] writing $CONFIG from $BASE"
python "$FT/make_det_config.py" --base "$BASE" --det-data "$MERGED" --data-dir / \
    --pretrained "$PRETRAINED" --epochs "$EPOCHS" --batch-size "$BATCH" --out "$CONFIG"

# ---- train ----
OPTS="Global.save_model_dir=$OUTDIR Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=1"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"
    echo "[resume] from $OUTDIR/latest"
else
    echo "[fresh] warm-starting from PP-OCRv6_medium_det pretrained"
fi

python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] real-only detector training finished (or stopped — re-submit to resume)."
