#!/usr/bin/env bash
#SBATCH --job-name=v6rec_s1syn
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# STAGE-1 (synth-only) recogniser for the CLEAN synth->real ablation arm (plan item 3).
# Identical to the ORIGINAL synth-only recipe (train_v6_rec.sh) EXCEPT the dict:
#   the old v6_unified_rec rebuilt a synth-only dict (1440 chars) -> its head can't take the
#   frozen 1638 dict without a reinit, which is *the* reason A' had to warm-start from the
#   BALANCED rec. Here we FREEZE the current 1638-char unified_dict.txt at stage-1, so the
#   real-only stage-2 warm-starts cleanly (CTC/GTC heads LOAD, no reinit) -> a true
#   "pure synth pretrain -> real fine-tune" path to compare per-language vs A'.
# Warm-start = stock PP-OCRv6_medium_rec (backbone transfers; the CTC/GTC heads REINITIALISE
# for the 1638 dict -> a HIGH initial loss/low acc is EXPECTED here; it's the cold head, not a
# bug. Contrast stage-2, which warm-starts from THIS and should start at a low loss.)
# Data = the existing synth pool ($SCR/rec_data_25k/unified_{train,val}.txt). We do NOT re-crop
# and do NOT rebuild the dict (rebuilding would clobber the 1638 dict back to a 1440 synth one).
# Submit from ada LOGIN node:  sbatch ~/train_v6_rec_stage1_synth_ada.sbatch
# Resume after a stop: re-submit (auto-detects the stage-1 latest checkpoint).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

# ---- knobs ----
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-8}                  # per-card; 8*4 = eff 32 (== original single-GPU recipe -> base LR stays valid)
EPOCHS=${EPOCHS:-12}               # deadline-aware synth pretrain. Raise to 20 (= original/ balanced stage-1) if
                                   # time allows; this is the heaviest run (full ~2.5M synth crops). See ETA note.
MAXTL=${MAXTL:-76}                 # MUST equal the stage-2 recipe's max_text_length, or the GTC/NRTR head's
                                   # positional embeddings differ and stage-2 would reinit that head on warm-start.

SYNTH_REC=${SYNTH_REC:-$SCR/rec_data_25k}           # holds unified_{train,val}.txt AND the FROZEN 1638 unified_dict.txt
OUTDIR=${OUTDIR:-./output/v6_rec_synthonly}          # NEW dir (1638 dict) — separate from old v6_unified_rec (1440 dict)
CONFIG=${CONFIG:-my_v6_rec_synthonly.yml}
FROZEN_DICT=${FROZEN_DICT:-$SYNTH_REC/unified_dict.txt}   # DO NOT rebuild — this is the 1638 dict

FT=$SCR/FineTune
REPO=$SCR/PaddleOCR
BASE=configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml
PRETRAINED=$REPO/pretrain/PP-OCRv6_medium_rec.pdparams
PRETRAINED_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams

cd "$REPO"; echo "[node] $(hostname)"; nvidia-smi -L || true
mkdir -p pretrain

# ---- pre-flight: synth pool + FROZEN 1638 dict MUST exist; NOTHING is rebuilt ----
for f in "$SYNTH_REC/unified_train.txt" "$SYNTH_REC/unified_val.txt" "$FROZEN_DICT"; do
  [ -f "$f" ] || { echo "[ERR] missing $f (synth crops purged from scratch? re-run crop_words.py first)"; exit 1; }
done
NDICT=$(wc -l < "$FROZEN_DICT")
echo "[frozen dict] $FROZEN_DICT  ($NDICT chars)"
[ "$NDICT" -ge 1600 ] || { echo "[ERR] dict has $NDICT chars — looks like the OLD synth-only (1440) dict, not the frozen 1638. Aborting so we don't repeat the head-reinit bug."; exit 1; }

# ---- pre-flight: drop pathological long labels (CTC-safety) ----
# The synth pool contains miscropped "words" up to ~154 chars. With MAXTL=76 they slip past
# filtering and make the CTC target longer than the crop's feature width -> inf/NaN loss that
# poisons the weights (this killed the first run at epoch 2). Cap at 50 chars = the OLD synth
# run's proven-safe limit. Idempotent; keeps a one-time .bak of the originals.
for f in unified_train unified_val; do
  src="$SYNTH_REC/$f.txt"
  if awk -F'\t' 'length($2)>50{c++} END{exit !(c>0)}' "$src" 2>/dev/null; then
    [ -f "$src.bak" ] || cp "$src" "$src.bak"
    n0=$(wc -l < "$src.bak"); awk -F'\t' 'length($2)<=50' "$src.bak" > "$src"; n1=$(wc -l < "$src")
    echo "[filter] $f: $n0 -> $n1 crops (dropped $((n0-n1)) labels >50 chars)"
  fi
done

# ---- pretrained ----
[ -f "$PRETRAINED" ] || { echo "[prep] downloading v6 rec pretrained"; wget -q "$PRETRAINED_URL" -O "$PRETRAINED"; }

# ---- config: FROZEN 1638 dict + synth-only pool + warm-start from stock pretrained ----
echo "[prep] writing $CONFIG (dict FROZEN to 1638; synth-only pool; warm-start = stock v6 rec)"
python "$FT/make_config.py" --base "$BASE" --script unified --rec-data "$SYNTH_REC" \
  --dict "$FROZEN_DICT" --pretrained "$PRETRAINED" \
  --epochs "$EPOCHS" --max-text-length "$MAXTL" --out "$CONFIG"

# ---- train (distributed) ----
# Synth crop paths in unified_train.txt are RELATIVE (crops/<script>/...) -> data_dir=$SYNTH_REC.
# make_config already set data_dir=$SYNTH_REC (== --rec-data); re-asserted below to be explicit.
# NO LR override: this is a full pretrain -> base 5e-4 / cosine / warmup 5 (the original recipe),
# NOT the low-LR refine that stage-2 uses.
OPTS="Global.save_model_dir=$OUTDIR"
OPTS="$OPTS Train.sampler.first_bs=$BATCH Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=64"
OPTS="$OPTS Train.dataset.data_dir=$SYNTH_REC Eval.dataset.data_dir=$SYNTH_REC"
OPTS="$OPTS Optimizer.clip_norm=5.0"   # gradient clipping — safety net against any residual loss/grad spike (the run had none)
if [ -f "$OUTDIR/latest.pdparams" ]; then
  OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] stage-1 from $OUTDIR/latest"
else
  echo "[fresh] warm-starting from stock PP-OCRv6_medium_rec — CTC/GTC heads REINITIALISE for the 1638 dict (high initial loss is expected)"
fi
echo "[opts] $OPTS"
python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] stage-1 synth-only recogniser finished (or stopped — re-submit to resume). Export it, then run stage-2 real-only on top."
