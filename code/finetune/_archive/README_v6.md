# PP-OCRv6 recognition fine-tuning

**Same pipeline as v5** — it reuses `crop_words.py` (the word crops + per-script dicts) and
`make_config.py`. Only **two** things differ from `README.md`: the **base config** and the
**pretrained weights**. The cropped data is version-independent, so **don't redo `crop_words.py`.**

## Prereqs (already in place from the v5 run)
- Env + cloned PaddleOCR on scratch, `out_dataset` staged, and `rec_data/` built by `crop_words.py`.

## v6-specific steps (from inside the cloned PaddleOCR repo)
```bash
export SCR=/ssd_scratch/cvit/mahima
cd $SCR/PaddleOCR

# 1. find the exact v6 rec config name in your clone:
ls configs/rec/PP-OCRv6/

# 2. download the v6 rec pretrained .pdparams
#    URL: check that config's `Global.pretrained_model` line, or the PP-OCRv6 model list.
mkdir -p pretrain
wget <PP-OCRv6 rec pretrained URL> -O pretrain/PP-OCRv6_medium_rec.pdparams

# 3. build the config (make_config.py works for ANY base config):
python $SCR/FineTune/make_config.py \
    --base configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml \
    --script devanagari \
    --rec-data $SCR/rec_data \
    --pretrained ./pretrain/PP-OCRv6_medium_rec.pdparams \
    --max-text-length 40 --out my_devanagari_v6.yml

# 4. train in tmux — same batch overrides as v5, single -o:
python tools/train.py -c my_devanagari_v6.yml \
    -o Train.sampler.first_bs=32 Train.loader.batch_size_per_card=32 Eval.loader.batch_size_per_card=64 \
    2>&1 | tee -a train_devanagari_v6.log

# 5. eval / export — identical to v5, just pointing at ./output/devanagari_rec (v6 run)
python tools/eval.py -c my_devanagari_v6.yml -o Global.checkpoints=./output/devanagari_rec/best_accuracy Eval.loader.batch_size_per_card=64
python tools/export_model.py -c my_devanagari_v6.yml \
    -o Global.checkpoints=./output/devanagari_rec/best_accuracy Global.save_inference_dir=./output/devanagari_v6_infer
```

Or just run **`run_v6.sh <script>`** after setting the two variables at its top (steps 2–3), then train.

## Notes
- v6 has **no Indic warm-start** → the recognition head trains from scratch (same as our uniform v5 approach),
  so it may need a few more epochs to match. Its **backbone is newer**, which can help.
- `make_config.py`'s NRTR-head tweak is wrapped in `try/except`, so it's safe even though v6's head structure differs.
- Everything downstream (eval, export, the multilingual Colab's fine-tuned slot) is the same — just point paths
  at the v6 output/inference dir.
- To make it **unified** later: run `crop_words.py` for all scripts, merge the per-script dicts + label files
  into one, and train a single v6 model over that combined dictionary.
