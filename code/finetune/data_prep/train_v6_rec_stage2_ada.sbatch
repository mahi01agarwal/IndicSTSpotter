#!/usr/bin/env bash
#SBATCH --job-name=v6rec_s2
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# STAGE-2 recogniser: REAL-ONLY fine-tune ON TOP OF the balanced synth+real rec (plan A').
# Warm-start = the balanced rec's weights (Global.pretrained_model, weights-only: fresh
# optimiser + LR schedule, NOT a resume). The dict is FROZEN to the balanced rec's own
# unified_dict.txt (1638 chars -> head 1640/1644), so the CTC + GTC heads LOAD instead of
# reinitialising -> a true continuation. Real-only because the per-script Bharat eval showed
# the balanced rec already beats synth-only on every script (A' confirmed); the new signal
# is the added real (IIIT-IndicSTR + IndicSTR12) plus dropping synth. Low LR + few epochs so
# we refine without eroding the sparse scripts.
# Submit from ada LOGIN node:  sbatch ~/train_v6_rec_stage2_ada.sbatch
# Resume after a stop: re-submit (auto-detects the stage-2 latest checkpoint).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

# ---- knobs ----
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-8}                  # per-card; 8*4 = eff 32 (matches the stage-1 recipe)
EPOCHS=${EPOCHS:-6}                # short refinement pass (A': don't over-train on real)
BASE_LR=${BASE_LR:-0.00005}        # ~10x below stage-1 peak (~5e-4): refine, don't relearn
WARMUP=${WARMUP:-0}                # little/no warmup for a continuation
LATIN_CAP=${LATIN_CAP:-40000}      # real Latin (MLT) is huge; keep ~= Indic
REAL_OVERSAMPLE=${REAL_OVERSAMPLE:-1}   # raise to weight real harder if sparse scripts need it
MAXTL=${MAXTL:-76}

SYNTH_REC=${SYNTH_REC:-$SCR/rec_data_25k}          # holds the FROZEN unified_dict.txt
COMBINED=${COMBINED:-$SCR/rec_data_stage2}          # real-only merged train/val
OUTDIR=${OUTDIR:-./output/v6_rec_stage2}
CONFIG=${CONFIG:-my_v6_rec_stage2.yml}
BASE_CKPT=${BASE_CKPT:-$SCR/PaddleOCR/output/v6_rec_synthreal/best_accuracy}   # A' warm-start
FROZEN_DICT=${FROZEN_DICT:-$SYNTH_REC/unified_dict.txt}                        # DO NOT rebuild

FT=$SCR/FineTune
DP=$FT/data_prep
REPO=$SCR/PaddleOCR
BASE=configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml

# REAL-ONLY training pool. IndicSTR12 = TRAIN split only (run the patched
# indicstr12_to_paddle.py first; its test split is held out as a benchmark).
REAL_TRAIN=( "$SCR/rec_data_real/bharat/bharat_rec_train.txt"
             "$SCR/rec_data_real/mlt17/mlt17_rec.txt"
             "$SCR/rec_data_real/mlt19/mlt19_rec.txt"
             "$SCR/rec_data_real/iiit_indicstr/iiit_indicstr_rec.txt"
             "$SCR/rec_data_real/indicstr12/indicstr12_rec.txt" )

cd "$REPO"; echo "[node] $(hostname)"; nvidia-smi -L || true

# ---- pre-flight: the frozen dict + warm-start checkpoint MUST exist ----
[ -f "${BASE_CKPT}.pdparams" ] || { echo "[ERR] warm-start ckpt ${BASE_CKPT}.pdparams missing"; exit 1; }
[ -f "$FROZEN_DICT" ] || { echo "[ERR] frozen dict $FROZEN_DICT missing"; exit 1; }
echo "[frozen dict] $FROZEN_DICT  ($(wc -l < "$FROZEN_DICT") chars)"

# ---- 1. real-only merge (NO synth inputs; dict is NOT rebuilt) ----
echo "[prep] merging REAL-ONLY (latin-cap=$LATIN_CAP, real-oversample=x$REAL_OVERSAMPLE)"
python "$DP/merge_rec_db.py" \
  --real-inputs "${REAL_TRAIN[@]}" \
  --synth-cap-per-script 0 --real-oversample "$REAL_OVERSAMPLE" \
  --latin-cap "$LATIN_CAP" --out-dir "$COMBINED"

# ---- 2. config: FROZEN dict + warm-start from the balanced rec ----
echo "[prep] writing $CONFIG (dict frozen to stage-1; pretrained = balanced rec)"
python "$FT/make_config.py" --base "$BASE" --script unified --rec-data "$COMBINED" \
  --dict "$FROZEN_DICT" --pretrained "${BASE_CKPT}.pdparams" \
  --epochs "$EPOCHS" --max-text-length "$MAXTL" --out "$CONFIG"

# ---- 3. train (distributed) ----
# All real crop paths are ABSOLUTE, so no data_dir override is needed (unlike stage-1's
# synth+real mix). Warm-start is weights-only via Global.pretrained_model (make_config set it)
# -> fresh optimiser + a new low-LR cosine, NOT a resume of stage-1's schedule.
OPTS="Global.save_model_dir=$OUTDIR"
OPTS="$OPTS Train.sampler.first_bs=$BATCH Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=64"
OPTS="$OPTS Optimizer.lr.learning_rate=$BASE_LR Optimizer.lr.warmup_epoch=$WARMUP"
if [ -f "$OUTDIR/latest.pdparams" ]; then
  OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] stage-2 from $OUTDIR/latest"
else
  echo "[fresh] warm-starting (weights-only) from $BASE_CKPT — dict frozen, heads load"
fi
echo "[opts] $OPTS"
python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] stage-2 real-only recogniser finished (or stopped — re-submit to resume)."
