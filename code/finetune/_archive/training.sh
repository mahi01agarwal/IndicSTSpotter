#!/usr/bin/env bash
#SBATCH --job-name=dev_warm
#SBATCH -A mahima.agarwal
#SBATCH --partition=u22
#SBATCH --nodelist=gnode063
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --output=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_dev_warm_%j.log
#SBATCH --error=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_dev_warm_%j.log
#
# Long TRAINING run for the Devanagari warm-start fine-tune (step 11).
# Assumes prep is already done on THIS node's scratch (crop_words -> build_unicode_dict
# -> wget specialist weights -> make_config wrote my_devanagari_warm.yml).
#
# Pinned to gnode063 (scratch is node-local -> data only lives on the node you staged).
# Use a different node:   sbatch --nodelist=gnodeXXX ~/training.sh   (must be set up there first)
# Resume after TIMEOUT:   just `sbatch` it again (auto-detects the latest checkpoint).
#
# Submit from your ada HOME:   sbatch ~/training.sh
# Status:                      squeue -u $USER
# Live log:                    tail -f /ssd_scratch/cvit/mahima/PaddleOCR/slurm_dev_warm_<jobid>.log
set -e

SCR=/ssd_scratch/cvit/mahima
CONFIG=my_devanagari_warm.yml
OUTDIR=./output/devanagari_warm_rec

# --- conda (batch shells don't source .bashrc; auto-detect the base) ---
CONDA_BASE=""
for c in "$HOME/miniconda3" "$HOME/anaconda3" "/home2/$USER/miniconda3" \
         "/home2/$USER/anaconda3" "/home/$USER/miniconda3" "/opt/conda"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then CONDA_BASE="$c"; break; fi
done
[ -z "$CONDA_BASE" ] && { echo "[fatal] conda base not found — set CONDA_BASE manually"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SCR/ocr"

cd "$SCR/PaddleOCR"
echo "[node] $(hostname)"; nvidia-smi -L || true

if [ ! -f "$CONFIG" ]; then
    echo "[fatal] $SCR/PaddleOCR/$CONFIG missing — run the prep steps (crop_words/build_unicode_dict/make_config) first"
    exit 1
fi

# ALL overrides under one -o. Add resume only if a checkpoint already exists.
OPTS="Global.save_model_dir=$OUTDIR Train.sampler.first_bs=32 Train.loader.batch_size_per_card=32 Eval.loader.batch_size_per_card=64"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"
    echo "[resume] continuing from $OUTDIR/latest"
else
    echo "[fresh] warm-starting from the specialist weights named in $CONFIG"
fi

python tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] training finished (or hit the time limit — re-sbatch to resume)"
