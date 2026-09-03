# PaddleOCR PP-OCRv5 recognition fine-tuning (on ada)

Fine-tune the **PP-OCRv5 text-recognition** model on our synthetic Indic word crops.

- **Why v5, not v6:** PP-OCRv6 has no Indic support at all; PP-OCRv5's recognition architecture +
  pretrained weights are the right base. We fine-tune **recognition** (the detector is script-agnostic —
  reuse the stock one).
- **Uniform approach for all 11 scripts:** start from the *generic* PP-OCRv5 rec pretrained and swap in a
  **per-script character dictionary**. The backbone transfers; the recognition head retrains for that
  script's characters. This works for the 5 scripts v5 already supports **and** the 6 it doesn't
  (Kannada, Malayalam, Bengali, Gujarati, Gurmukhi, Odia).

---

## 1. Environment on ada (ssd_scratch)

ada's `/home` is **full** and `/ssd_scratch` is large but **node-local** (per GPU node). So we build the
env *and* stage data on scratch inside a GPU job, and copy trained models **off** scratch before the job
ends. `$SCR` = your scratch dir.

```bash
# 1a. get an interactive GPU node (SLURM; flags vary on ada)
ssh mahima.agarwal@ada.iiit.ac.in
sinteractive -c 9 -g 1                 # or:  srun --gres=gpu:1 -c 9 --mem 40G --time 12:00:00 --pty bash
nvidia-smi                             # confirm a GPU
nvcc --version                         # NOTE the CUDA version -> picks the paddle wheel below

# 1b. work on scratch
export SCR=/ssd_scratch/cvit/$USER
mkdir -p $SCR && cd $SCR

# 1c. miniconda on scratch (home is over quota)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p $SCR/miniconda3
source $SCR/miniconda3/bin/activate
conda create -p $SCR/envs/paddle python=3.10 -y
conda activate $SCR/envs/paddle

# 1d. PaddlePaddle-GPU — match the CUDA from `nvcc` (cu118 / cu126 / cu129 index)
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -c "import paddle; paddle.utils.run_check()"     # must report GPU works

# 1e. PaddleOCR source (training tools) + deps
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && pip install -r requirements.txt && pip install -e . && cd ..

# 1f. stage the dataset + this folder onto scratch
rsync -av mahima@10.4.16.81:~/indic_spotter_synth/out_dataset $SCR/
rsync -av mahima@10.4.16.81:~/indic_spotter_synth/FineTune  $SCR/    # or copy from your Mac
pip install pillow                                                    # for crop_words.py
```

> **Caveats:**
> - If `wget`/`pip`/`git` fail on the compute node (no internet), set the IIIT proxy (`export http_proxy=...`)
>   or pre-download on the login node.
> - `/ssd_scratch` is **ephemeral** — recreate the env each session (this README is the script), and
>   `rsync` the small trained model back to `~/` or off-cluster **before releasing the node**.

## 2. Crop words -> PaddleOCR rec format

```bash
cd $SCR/FineTune
python crop_words.py --dataset-dir $SCR/out_dataset --out-dir $SCR/rec_data --skip-curved
```
Produces per script: `rec_data/crops/<script>/…`, `<script>_train.txt`, `<script>_val.txt`,
`<script>_dict.txt`. Note each script's printed `chars` (dict size) and `max_text_len`.

## 3. Pretrained model + config (start with ONE script, e.g. `devanagari`)

```bash
cd $SCR/PaddleOCR
mkdir -p pretrain
# generic PP-OCRv5 rec pretrained (works for any script via a custom dict):
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams \
     -O pretrain/PP-OCRv5_server_rec.pdparams

# build the per-script config (no hand-editing); set --max-text-length >= what crop_words.py printed
pip install pyyaml
python $SCR/FineTune/make_config.py \
    --base configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
    --script devanagari \
    --rec-data $SCR/rec_data \
    --pretrained ./pretrain/PP-OCRv5_server_rec.pdparams \
    --max-text-length 40 --out my_devanagari.yml
```
> `make_config.py` injects the per-script dict, train/val lists, data dir, pretrained path and
> `use_space_char` into the base config. The dict differs from the pretrained's, so the recognition
> **head** re-initialises and retrains while the backbone transfers — expected (you may see a
> "shape mismatch, skipping" note for the head). If the base config name differs, check
> `ls configs/rec/PP-OCRv5/`.

## 4. Train / eval / export

```bash
python tools/train.py -c my_devanagari.yml
python tools/eval.py  -c my_devanagari.yml -o Global.checkpoints=./output/devanagari_rec/best_accuracy
python tools/export_model.py -c my_devanagari.yml \
    -o Global.checkpoints=./output/devanagari_rec/best_accuracy \
       Global.save_inference_dir=./output/devanagari_infer

# copy the trained model OFF ephemeral scratch before the job ends:
rsync -av ./output/devanagari_infer mahima@10.4.16.81:~/paddle_finetunes/
```

## 5. Next scripts / notes
- Repeat §3–4 per script, pointing `character_dict_path` + `label_file_list` at that script's files.
  Supported scripts (Devanagari, Tamil, Telugu, Urdu) will converge fastest; the six unsupported ones
  start from a random head so give them more epochs / data.
- To use the fine-tuned recogniser end-to-end: pair the exported inference model with the **stock**
  PP-OCRv5 detector (`PaddleOCR(text_recognition_model_dir="…/devanagari_infer", lang=…)` or via a config).
- This recognises **pre-cropped words**; it is not a full-page reader on its own (that's the detector's job).
  For a single model that reads a whole page across all 11 scripts, that's the LightOnOCR fine-tune track.
