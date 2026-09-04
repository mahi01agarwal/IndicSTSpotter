# Model weights

Weights are hosted on Hugging Face (not in git). All models are small — each inference
export is ~60 MB.

**Repo:** <https://huggingface.co/MahimaSinghal/IndicSTSpotter>  *(private during review; made public at release)*

## Released (inference exports)

| Model | Params | Role | Reproduces | Location |
|---|---|---|---|---|
| **rec_stage2** | 15.8M | headline recognizer (synth-pretrain → real fine-tune) | T3, T5, T6, T7, T8 | HF repo, `rec_stage2/` |
| **det_mist** | 15.5M | headline detector (real + MIST, dense scenes) | T4, T6, T7, T8 | HF repo, `det_mist/` |
| `unified_dict.txt` | — | frozen 1,638-char dictionary | all rec | in this repo (`dicts/`) + HF repo |

Download:

```python
from huggingface_hub import snapshot_download
d = snapshot_download("MahimaSinghal/IndicSTSpotter", repo_type="model")
# -> d/rec_stage2, d/det_mist, d/unified_dict.txt
```

## Not hosted (ablation / supporting models)

Available on request; not required to reproduce the paper's headline tables:
`rec_stage1` (synth-only), `rec_balanced` (warm-start source for stage2), `rec_stage2_B`
(A′/B variant), `det_real` (no-MIST detector), and the other ablation exports.

## Use

Each model dir is a PaddleOCR inference export (`inference.yml` + `*.pdmodel`/`*.pdiparams`).
Point an eval at it via `--model <label> <dir>` (rec) or `--det-dir` / `--rec-dir` (det/e2e);
`model_name` is `PP-OCRv6_medium_rec` / `PP-OCRv6_medium_det`. See `REPRODUCE.md` for the exact
per-table commands.

## Recreate an inference export from a training checkpoint

Training checkpoints (with optimizer state) are not released — inference exports are enough.
To re-export, always pass the frozen dict so the CTC head classes align:

```bash
python code/PaddleOCR/tools/export_model.py -c <config.yml> \
  -o Global.pretrained_model=checkpoints/<model>/best_model/model \
     Global.save_inference_dir=inference/<model> \
     Global.character_dict_path=/home/mahima/IndicSTSpotter/dicts/unified_dict.txt
```
