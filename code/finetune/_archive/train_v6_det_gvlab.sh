#!/usr/bin/env bash
# v6 DETECTOR fine-tune on gvlab (no SLURM). Warm-started from PP-OCRv6_medium_det.
# Self-preps det labels + config, then trains on both RTX 4090s via paddle.distributed.
#
# Prereqs on gvlab (done once):
#   - conda env with Paddle 3.1 GPU at $GV/env
#   - PaddleOCR cloned at $GV/PaddleOCR  (must contain configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml)
#   - dataset at   $GV/out_dataset_25k   (images/ + annotations/)
#   - FineTune scripts at $GV/FineTune
#   - pretrained weight already staged at $PRETRAINED (gvlab can't reach bcebos — fetch on ada, see notes)
#
# Run inside tmux:   tmux new -s det ;  bash $GV/FineTune/train_v6_det_gvlab.sh
# Resume after a stop/reboot: just run it again (auto-detects the latest checkpoint).
set -euo pipefail

# ---- knobs (env-overridable) ----
GV=${GV:-/scratch/ocr}
GPUS=${GPUS:-0,1}                 # both 4090s; set to "0" for a single GPU
BATCH=${BATCH:-8}                 # per-card. 24GB fits more; raise to 12-16 to use VRAM. Drop to 4 if OOM.
EPOCHS=${EPOCHS:-100}

FT=$GV/FineTune
DATASET=${DATASET:-$GV/out_dataset_25k}        # override for the realistic set
DET=${DET:-$GV/det_data_25k}                    # det labels derived from DATASET
BASE=configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml
CONFIG=${CONFIG:-my_v6_det.yml}
OUTDIR=${OUTDIR:-./output/v6_det}
PRETRAINED=./pretrain/PP-OCRv6_medium_det.pdparams

# ---- conda ----
CONDA_BASE=""
[ -n "${CONDA_EXE:-}" ] && CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
             "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }
    done
fi
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "[fatal] conda.sh not found under '$CONDA_BASE' — activate your env before running"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$GV/env"

cd "$GV/PaddleOCR"
echo "[node] $(hostname)"; nvidia-smi -L || true
mkdir -p pretrain "$OUTDIR"

# ---- sanity: base config + pretrained must exist ----
[ -f "$BASE" ] || { echo "[fatal] base config missing: $BASE  (need a recent PaddleOCR clone with PP-OCRv6)"; exit 1; }
if [ ! -f "$PRETRAINED" ]; then
    echo "[fatal] pretrained weight missing: $PRETRAINED"
    echo "        gvlab cannot reach bcebos. Fetch it on ada and relay via nltm:"
    echo "          # on ada (gnode w/ internet):"
    echo "          wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams -O \$SCR/PP-OCRv6_medium_det.pdparams"
    echo "          rsync -av \$SCR/PP-OCRv6_medium_det.pdparams mahima@10.4.16.81:~/"
    echo "          # on gvlab:"
    echo "          rsync -av mahima@10.4.16.81:~/PP-OCRv6_medium_det.pdparams $GV/PaddleOCR/pretrain/"
    exit 1
fi

# ---- prep det labels (guarded) ----
if [ ! -f "$DET/det_train.txt" ]; then
    echo "[prep] building det labels from $DATASET"
    python "$FT/make_det_labels.py" --dataset-dir "$DATASET" --out-dir "$DET"
fi

# ---- write config ----
echo "[prep] writing $CONFIG"
python "$FT/make_det_config.py" --base "$BASE" --det-data "$DET" --data-dir "$DATASET" \
    --pretrained "$PRETRAINED" --epochs "$EPOCHS" --batch-size "$BATCH" --out "$CONFIG"

# ---- train (single -o, all overrides together) ----
# det EVAL must be batch 1 (eval images keep native/variable sizes -> can't be stacked)
OPTS="Global.save_model_dir=$OUTDIR Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=1"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"
    echo "[resume] from $OUTDIR/latest"
else
    echo "[fresh] warm-starting from PP-OCRv6_medium_det pretrained"
fi

python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] v6 detector training finished (or stopped — re-run to resume)."
