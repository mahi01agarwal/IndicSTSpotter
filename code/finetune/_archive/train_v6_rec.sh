#!/usr/bin/env bash
#SBATCH --job-name=v6_rec
#SBATCH -A mahima.agarwal
#SBATCH --partition=u22
#SBATCH --nodelist=gnode063
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --output=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v6_rec_%j.log
#SBATCH --error=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v6_rec_%j.log
#
# v6 UNIFIED RECOGNIZER fine-tune (all scripts, one Unicode-block dict), warm-started
# from PP-OCRv6_medium_rec pretrained (backbone transfers; head reinitialises for our
# unified dict). Self-preps then trains. Pinned to gnode063 (scratch is node-local).
# Override node: sbatch --nodelist=gnodeXXX ~/train_v6_rec.sh   (must be set up there).
# Resume after TIMEOUT: just sbatch again (auto-detects latest checkpoint).
set -e

SCR=/ssd_scratch/cvit/mahima
FT=$SCR/FineTune
DATASET=$SCR/out_dataset_25k                 # the new 25k scenes
REC=$SCR/rec_data_25k                         # rec crops derived from them (separate from the 5k rec_data)
BASE=configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml
CONFIG=my_v6_unified.yml
OUTDIR=./output/v6_unified_rec              # new dir (Unicode dict) — separate from any old corpus-dict run
PRETRAINED=./pretrain/PP-OCRv6_medium_rec.pdparams
PRETRAINED_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
MAXTL=50                                     # generous; bump if build_unified reports a longer label
BATCH=32

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

# --- prep (guarded / idempotent) ---
if [ ! -f "$REC/devanagari_train.txt" ]; then
    echo "[prep] cropping words (all scripts) from $DATASET"
    python "$FT/crop_words.py" --dataset-dir "$DATASET" --out-dir "$REC" --skip-curved
fi
echo "[prep] pooling unified train/val lists"
python "$FT/build_unified.py" --rec-data "$REC"          # writes unified_{train,val}.txt (+ corpus dict, overwritten next)
echo "[prep] rebuilding dicts to full Unicode block (incl. unified_dict.txt)"
python "$FT/build_unicode_dict.py" --rec-data "$REC"     # Unicode per-script dicts + unified_dict union
[ -f "$PRETRAINED" ] || { echo "[prep] downloading v6 rec pretrained"; wget -q "$PRETRAINED_URL" -O "$PRETRAINED"; }
echo "[prep] writing $CONFIG"
python "$FT/make_config.py" --base "$BASE" --script unified --rec-data "$REC" \
    --pretrained "$PRETRAINED" --epochs 20 --max-text-length $MAXTL --out "$CONFIG"

# --- train ---
OPTS="Global.save_model_dir=$OUTDIR Train.sampler.first_bs=$BATCH Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=64"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] from $OUTDIR/latest"
else
    echo "[fresh] warm-starting from PP-OCRv6_medium_rec pretrained (head reinitialises for the unified dict)"
fi
python tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] v6 unified recognizer training finished (or timed out — re-sbatch to resume)"
