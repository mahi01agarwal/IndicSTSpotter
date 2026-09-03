# Model weights

Weights are **not** stored in git (they're binary; hosted off-repo). All models are small —
each inference export is ~60 MB; the full set incl. ablations is ~0.6 GB.

> **Download links: _TODO_ — fill in after uploading to Hugging Face / Zenodo.**

## Released (inference exports)

| Model | Params | Role | Reproduces | Download |
|---|---|---|---|---|
| **rec_stage2** (stage2A) | 15.8M | headline recognizer (synth-pretrain → real fine-tune) | T3, T5, T6, T7, T8 | _TODO_ |
| **det_mist** | 15.5M | headline detector (real + MIST, dense scenes) | T4, T6, T7, T8 | _TODO_ |
| `dicts/unified_dict.txt` | — | frozen 1,638-char dictionary (in git) | all rec | in repo |

### Ablation / supporting models (optional, still small)

| Model | Role |
|---|---|
| rec_stage1 | synth-only pretrain (ablation ladder) |
| rec_balanced | synth+real balanced (warm-start source for stage2A) |
| rec_stage2_B | stage-2 variant (A′/B tie-breaker) |
| det_real | real-only detector (no MIST; ablation) |
| ablations/* | det_synthonly, rec_unified_1440dict, v5_devanagari_specdict, … |

## Use

Each download is a PaddleOCR inference dir (`inference.yml` + `*.pdmodel`/`*.pdiparams`).
Point an eval at it with `--model <label> <dir>` (rec) or `--det-dir/--rec-dir <dir>` (det/e2e).
The `model_name` is `PP-OCRv6_medium_rec` / `PP-OCRv6_medium_det` (the eval defaults).
See `REPRODUCE.md` for the exact per-table commands.

## Recreate an inference export from a training checkpoint

Training checkpoints (with optimizer state) are **not** released — inference exports are enough.
To re-export from a checkpoint, always pass the frozen dict so the CTC head classes align:

```bash
python code/PaddleOCR/tools/export_model.py -c <config.yml> \
  -o Global.pretrained_model=checkpoints/<model>/best_model/model \
     Global.save_inference_dir=inference/<model> \
     Global.character_dict_path=/home/mahima/IndicSTSpotter/dicts/unified_dict.txt
```
