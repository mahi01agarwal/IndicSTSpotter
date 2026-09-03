#!/usr/bin/env bash
#SBATCH --job-name=v5_specdict
#SBATCH -A mahima.agarwal
#SBATCH --partition=u22
#SBATCH --nodelist=gnode063
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=48:00:00
#SBATCH --output=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v5_specdict_%j.log
#SBATCH --error=/ssd_scratch/cvit/mahima/PaddleOCR/slurm_v5_specdict_%j.log
#
# v5 Devanagari rec fine-tune with the SPECIALIST dict (ppocrv5_devanagari_dict.txt,
# 568 chars) so the CTC/NRTR head transfers instead of reinitialising. Warm-start from
# devanagari_PP-OCRv5_mobile_rec. Runs as a clean sbatch job (its own GPU) — do NOT run
# this interactively while another job holds the GPU, or the backbone forward crashes.
# PREREQ: my_devanagari_specdict.yml already written by make_config.py.
# Resume after TIMEOUT: just sbatch again.
set -e

SCR=/ssd_scratch/cvit/mahima
CONFIG=my_devanagari_specdict.yml
OUTDIR=./output/devanagari_specdict_rec

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
[ -f "$CONFIG" ] || { echo "[fatal] $CONFIG missing — run make_config.py first"; exit 1; }

OPTS="Global.save_model_dir=$OUTDIR Train.sampler.first_bs=32 Train.loader.batch_size_per_card=32 Eval.loader.batch_size_per_card=64"
if [ -f "$OUTDIR/latest.pdparams" ]; then
    OPTS="$OPTS Global.checkpoints=$OUTDIR/latest"; echo "[resume] from $OUTDIR/latest"
else
    echo "[fresh] warm-start from specialist weights + specialist dict (head should load)"
fi
python tools/train.py -c "$CONFIG" -o $OPTS
echo "[done] specdict training finished (or timed out — re-sbatch to resume)"
