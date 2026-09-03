#!/usr/bin/env bash
#SBATCH --job-name=v6_det
#SBATCH -A mahima.agarwal
#SBATCH --partition=u22
#SBATCH --nodelist=gnode063
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --output=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v6_det_%j.log
#SBATCH --error=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v6_det_%j.log
#SBATCH --signal=B:USR1@120
#
# v6 DETECTOR fine-tune, warm-started from PP-OCRv6_medium_det pretrained. Self-preps
# (det labels + config) then trains. Pinned to gnode063 (scratch is node-local).
# Override node: sbatch --nodelist=gnodeXXX ~/train_v6_det.sh   (must be set up there).
# Resume after TIMEOUT: just sbatch again (auto-detects latest checkpoint).
set -e

SCR=${SCR:-/ssd_scratch/cvit/mahima}
FT=$SCR/FineTune
DATASET=${DATASET:-$SCR/out_dataset_25k}      # override for the realistic set
DET=${DET:-$SCR/det_data_25k}                 # det labels derived from DATASET
BASE=configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml
CONFIG=${CONFIG:-my_v6_det.yml}
OUTDIR=${OUTDIR:-./output/v6_det}
PRETRAINED=./pretrain/PP-OCRv6_medium_det.pdparams
PRETRAINED_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams
BATCH=${BATCH:-8}                             # det is memory-heavy; drop to 4 if OOM on a 24GB card

# --- conda (auto-detect base) ---
CONDA_BASE=""
for c in "$HOME/miniconda3" "$HOME/anaconda3" "/home2/$USER/miniconda3" \
         "/home2/$USER/anaconda3" "/home/$USER/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }
done
[ -z "$CONDA_BASE" ] && { echo "[fatal] conda base not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

cd "$SCR/PaddleOCR"
echo "[node] $(hostname)"; nvidia-smi -L || true
mkdir -p pretrain

# --- prep (guarded) ---
if [ ! -f "$DET/det_train.txt" ]; then
    echo "[prep] building det labels from $DATASET"
    python "$FT/make_det_labels.py" --dataset-dir "$DATASET" --out-dir "$DET"
fi
[ -f "$PRETRAINED" ] || { echo "[prep] downloading v6 det pretrained"; wget -q "$PRETRAINED_URL" -O "$PRETRAINED"; }
echo "[prep] writing $CONFIG"
python "$FT/make_det_config.py" --base "$BASE" --det-data "$DET" --data-dir "$DATASET" \
    --pretrained "$PRETRAINED" --epochs 100 --batch-size $BATCH --out "$CONFIG"

# --- train ---
# det EVAL must be batch 1 (eval images keep native/variable sizes -> can't be stacked)
OPTS="Global.save_model_dir=$OUTDIR Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=1"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] from $OUTDIR/latest"
else
    echo "[fresh] warm-starting from PP-OCRv6_medium_det pretrained"
fi
python tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] v6 detector training finished (or timed out — re-sbatch to resume)"
