#!/usr/bin/env bash
#SBATCH --job-name=v6det_mist
#SBATCH --partition=u22
#SBATCH -A mahima.agarwal
#SBATCH --nodelist=gnode065
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.log
# v6 DETECTOR train on ada — REAL + MIST (Bharat+MLT17+MLT19 + MIST-Train), for the
# dense/tiny-text weakness. Labels are absolute-path PaddleOCR DB, so data_dir=/.
#
# WARMSTART knob picks the init:
#   WARMSTART=stock  (DEFAULT) — fresh from stock PP-OCRv6_medium_det. This is the CLEAN ablation:
#       identical init + recipe to v6_det_real, only +MIST data differs -> the delta is attributable
#       to MIST ("MIST improves dense-text detection"). The union already contains all real data, so
#       a fresh run loses nothing vs warm-starting. Use this for the paper's real-vs-real+MIST number.
#   WARMSTART=real   — continue from the trained real detector (v6_det_real/best_accuracy). Cheaper/
#       faster, but confounds the comparison (real data seen ~2x, extra compute). Only a time-saver
#       fallback if fresh convergence is too slow before the deadline; report it as "continued fine-tune".
#
# Runs on 2 GPUs so it can share gnode063 with the recogniser (rec on GPUS=0,1, this on 2,3).
# For a 4-GPU solo run: --gres=gpu:4, GPUS=0,1,2,3, BATCH stays per-card.
#
# PREREQS (run first, in order):
#   1) sbatch ~/download_mist_aria.sbatch                       # -> $SCR/mist/{Images,gt}/{Train,Test}
#   2) python $SCR/FineTune/data_prep/mist_to_db.py --root $SCR/mist --split Train \
#          --out $SCR/det_data_real/mist/mist_det_train.txt
#      python $SCR/FineTune/data_prep/mist_to_db.py --root $SCR/mist --split Test \
#          --out $SCR/det_data_real/mist/mist_det_test.txt      # held-out MIST benchmark
#   3) python $SCR/FineTune/data_prep/merge_det_db.py \
#          --inputs $SCR/det_data_real/bharat/bharat_det_train.txt \
#                   $SCR/det_data_real/mlt17/mlt17_det.txt \
#                   $SCR/det_data_real/mlt19/mlt19_det.txt \
#                   $SCR/det_data_real/mist/mist_det_train.txt \
#          --out-dir $SCR/det_data_real_mist/merged        # do NOT add mist_det_test.txt
# Submit from ada LOGIN node:  sbatch ~/train_v6_det_mist_ada.sbatch
# Resume after stop: re-submit (auto-detects latest checkpoint).
set -euo pipefail
export SCR=/ssd_scratch/cvit/mahima
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

# ---- knobs ----
WARMSTART=${WARMSTART:-stock}    # stock (fresh, clean ablation — DEFAULT) | real (continue from v6_det_real)
# GPUs: auto-derive from SLURM's allocation (CUDA_VISIBLE_DEVICES is 0..N-1 inside the job)
# so this works whether the job gets 2 or 4 cards. Override with GPUS=0,1,2,3 if needed.
if [ -z "${GPUS:-}" ]; then
    N=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$N" -ge 1 ] || N=1
    GPUS=$(python3 -c "print(','.join(str(i) for i in range($N)))")
fi
BATCH=${BATCH:-4}                # per-card. Drop to 4 if OOM at 1920 side-len.
EPOCHS=${EPOCHS:-100}            # best_accuracy picked on the merged real val set; scancel on plateau
MERGED=${MERGED:-$SCR/det_data_real_mist/merged}
OUTDIR=${OUTDIR:-./output/v6_det_mist}
CONFIG=${CONFIG:-my_v6_det_mist.yml}

FT=$SCR/FineTune
REPO=$SCR/PaddleOCR
BASE=configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml
STOCK=$SCR/PP-OCRv6_medium_det.pdparams
REAL_CKPT=$REPO/output/v6_det_real/best_accuracy      # prefix (Paddle appends .pdparams)

cd "$REPO"
echo "[node] $(hostname)"; nvidia-smi -L || true
# Pin CUDA to the cards that actually enumerate. SLURM's --gres can hand the job a dead GPU
# index (gnode063 GPU3 fell off the bus 2026-08-10), and the launcher's CUDA init then fails
# node-wide with "no GPU detected". nvidia-smi -L above already dropped it from $GPUS, so this
# self-adjusts: 3 live cards now, back to 4 automatically once GPU3 is repaired.
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[gpus] GPUS=$GPUS  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  BATCH=$BATCH  eff_batch=$(( $(echo $GPUS | tr ',' '\n' | wc -l) * BATCH ))"

# ---- sanity ----
[ -f "$BASE" ] || { echo "[fatal] base config missing: $REPO/$BASE"; exit 1; }
[ -f "$FT/make_det_config.py" ] || { echo "[fatal] missing $FT/make_det_config.py — sync to \$SCR/FineTune"; exit 1; }
[ -f "$MERGED/det_train.txt" ] || { echo "[fatal] no merged labels at $MERGED — run merge_det_db.py (see header) first"; exit 1; }

# ---- pick init weight from WARMSTART knob ----
if [ "$WARMSTART" = "real" ]; then
    [ -f "${REAL_CKPT}.pdparams" ] || { echo "[fatal] WARMSTART=real but ${REAL_CKPT}.pdparams missing"; exit 1; }
    PRETRAINED="$REAL_CKPT"
    echo "[init] WARMSTART=real -> continue from trained real detector: $PRETRAINED (confounds ablation)"
else
    PRETRAINED="$STOCK"
    if [ ! -f "$STOCK" ]; then
        echo "[fetch] stock det weight missing — pulling from bcebos (gnode has internet)"
        wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams \
            -O "$STOCK" || { echo "[fatal] weight fetch failed"; exit 1; }
    fi
    echo "[init] WARMSTART=stock -> fresh from stock PP-OCRv6_medium_det (clean real-vs-real+MIST ablation)"
fi

# ---- write config (data_dir=/ because label paths are absolute) ----
# NOTE: training-time eval keeps the default 960 side-len (11GB-card OOM safety, see make_det_config.py).
# The dense 1920 recall benchmark is a SEPARATE full-res eval (eval_det_real.py / --det-limit-side-len 1920).
echo "[prep] writing $CONFIG from $BASE"
python "$FT/make_det_config.py" --base "$BASE" --det-data "$MERGED" --data-dir / \
    --pretrained "$PRETRAINED" --epochs "$EPOCHS" --batch-size "$BATCH" --out "$CONFIG"

# ---- train ----
OPTS="Global.save_model_dir=$OUTDIR Train.loader.batch_size_per_card=$BATCH Eval.loader.batch_size_per_card=1"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"
    echo "[resume] from $OUTDIR/latest"
fi

python -m paddle.distributed.launch --gpus "$GPUS" tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] real+MIST detector training finished (or stopped — re-submit to resume)."
echo "[next] eval: Bharat dense recall (--det-limit-side-len 1920), MIST-Test benchmark, real Hi/Mr;"
echo "       compare vs v6_det_real. Also check Bharat/MIST image overlap before reporting both."
