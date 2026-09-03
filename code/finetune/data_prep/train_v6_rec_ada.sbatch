#!/usr/bin/env bash
#SBATCH --job-name=v6rec_sr
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode063
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# v6 UNIFIED RECOGNIZER, SYNTH+REAL, 4 GPUs (paddle.distributed).
# Warm-start PP-OCRv6_medium_rec (backbone transfers; CTC head reinitialises for the
# unified multi-script dict). Self-preps: synthetic crops -> unified dict (covering real
# chars) -> synth+real merge (Latin-capped) -> config -> train.
# Submit from ada LOGIN node:  sbatch ~/train_v6_rec_ada.sbatch
# Resume after stop: re-submit (auto-detects latest checkpoint).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

# ---- knobs ----
GPUS=${GPUS:-0,1,2,3}
BATCH=${BATCH:-8}                 # per-card; 8*4 = effective 32 == the single-GPU recipe -> base LR stays valid.
EPOCHS=${EPOCHS:-20}
# --- data balance: synthetic is redundant (earlier synth-only model already learned it);
# the ONLY new signal is real, so CAP SYNTH PER SCRIPT and keep all real -> real ~30-40% of mix. ---
SYNTH_CAP=${SYNTH_CAP:-5000}       # thin synthetic floor per script (0 = keep all ~2.5M -> real drowned).
                                   # With IIIT+IndicSTR12 real added, every script has 23k+ real, so synth
                                   # is now a small diversity supplement, not the bulk. -> real ~88% of mix.
REAL_OVERSAMPLE=${REAL_OVERSAMPLE:-1}  # duplicate real crops in train (1 = none; raise to weight real harder)
LATIN_CAP=${LATIN_CAP:-40000}     # final cap on combined Latin (real Latin from MLT is huge; keep ~= Indic)
DATASET=${DATASET:-$SCR/out_dataset_25k}
SYNTH_REC=${SYNTH_REC:-$SCR/rec_data_25k}
COMBINED=${COMBINED:-$SCR/rec_data_combined}
OUTDIR=${OUTDIR:-./output/v6_rec_synthreal}
CONFIG=${CONFIG:-my_v6_rec_synthreal.yml}
MAXTL=${MAXTL:-76}

FT=$SCR/FineTune
DP=$FT/data_prep
REPO=$SCR/PaddleOCR
BASE=configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml
PRETRAINED=$REPO/pretrain/PP-OCRv6_medium_rec.pdparams
PRETRAINED_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams

# real crop label files (training pool; Bharat TEST + IIIT-IndicSTR TEST held out as benchmarks)
REAL_TRAIN=( "$SCR/rec_data_real/bharat/bharat_rec_train.txt"
             "$SCR/rec_data_real/mlt17/mlt17_rec.txt"
             "$SCR/rec_data_real/mlt19/mlt19_rec.txt"
             "$SCR/rec_data_real/iiit_indicstr/iiit_indicstr_rec.txt"
             "$SCR/rec_data_real/indicstr12/indicstr12_rec.txt" )

cd "$REPO"; echo "[node] $(hostname)"; nvidia-smi -L || true
mkdir -p pretrain

# ---- 1. synthetic crops (guarded) ----
if [ ! -f "$SYNTH_REC/unified_train.txt" ]; then
  if ls "$SYNTH_REC"/*_train.txt >/dev/null 2>&1; then
    echo "[prep] per-script synth crops present; pooling"
  else
    echo "[prep] cropping synthetic words from $DATASET -> $SYNTH_REC"
    python "$FT/crop_words.py" --dataset-dir "$DATASET" --out-dir "$SYNTH_REC" --skip-curved
  fi
  python "$FT/build_unified.py" --rec-data "$SYNTH_REC"
fi

# ---- 2. unified dict (Unicode blocks UNION real observed chars) ----
echo "[prep] building unified dict (real chars unioned in)"
python "$FT/build_unicode_dict.py" --rec-data "$SYNTH_REC" --extra-labels "${REAL_TRAIN[@]}"

# ---- 3. synth+real merge (synth per-script capped, real kept in full) ----
echo "[prep] merging synth+real (synth-cap=$SYNTH_CAP/script, real-oversample=x$REAL_OVERSAMPLE, latin-cap=$LATIN_CAP)"
python "$DP/merge_rec_db.py" \
  --synth-inputs "$SYNTH_REC/unified_train.txt" "$SYNTH_REC/unified_val.txt" \
  --real-inputs "${REAL_TRAIN[@]}" \
  --synth-cap-per-script "$SYNTH_CAP" --real-oversample "$REAL_OVERSAMPLE" \
  --latin-cap "$LATIN_CAP" --out-dir "$COMBINED"

# ---- 4. pretrained + config ----
[ -f "$PRETRAINED" ] || { echo "[prep] downloading v6 rec pretrained"; wget -q "$PRETRAINED_URL" -O "$PRETRAINED"; }
echo "[prep] writing $CONFIG"
python "$FT/make_config.py" --base "$BASE" --script unified --rec-data "$COMBINED" \
  --dict "$SYNTH_REC/unified_dict.txt" --pretrained "$PRETRAINED" \
  --epochs "$EPOCHS" --max-text-length "$MAXTL" --out "$CONFIG"

# ---- 5. train (distributed) ----
# data_dir override: the merged label file mixes ABSOLUTE real-crop paths with RELATIVE
# synthetic paths (crops/<script>/..., relative to $SYNTH_REC=rec_data_25k). PaddleOCR does
# os.path.join(data_dir, path): absolute real paths ignore data_dir, relative synth paths need
# data_dir=$SYNTH_REC to resolve. make_config wrote data_dir=$COMBINED, so override it here.
OPTS="Global.save_model_dir=$OUTDIR Train.sampler.first_bs=$BATCH Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=64 Train.dataset.data_dir=$SYNTH_REC Eval.dataset.data_dir=$SYNTH_REC"
if [ -f "$OUTDIR/latest.pdparams" ]; then
  OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] from $OUTDIR/latest"
else
  echo "[fresh] warm-starting from PP-OCRv6_medium_rec (head reinitialises for unified dict)"
fi
python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] synth+real recognizer training finished (or stopped — re-submit to resume)."
