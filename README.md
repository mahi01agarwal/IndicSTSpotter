# IndicSTSpotter

A unified multilingual scene-text spotter for **11 Indic scripts + Latin**: a single
detector and a single recognizer over one **frozen 1,638-character dictionary** — no
script identification, no per-language weights. The 15.8M-parameter recognizer matches or
surpasses much larger routed / per-script systems at a fraction of the parameters.

> Research code for the IndicSTSpotter paper. **Repo is private while the paper is under
> review.** Datasets and model weights are **not** stored here — see below.

## What's in this repo

| Path | Contents |
|---|---|
| `code/finetune/` | data-prep converters, dict builder, training-config generators, eval scripts |
| `code/configs/` | custom PaddleOCR training configs |
| `code/synth/` | Indic-aware synthetic-scene generation |
| `dicts/unified_dict.txt` | the **frozen 1,638-char shared dictionary** (all rec models use this exact file) |
| `REPRODUCE.md` | table-by-table reproduction commands + model zoo |
| `DATA.md` | how to obtain every dataset (official links + converters) |
| `MODELS.md` | model weights + download links (hosted off-repo) |

**Not in this repo** (obtained separately):
- **Datasets** (~85 GB, third-party licenses) → `DATA.md` for official links + our converters.
- **Model weights** (inference exports, ~0.6 GB) → `MODELS.md` (hosted on Hugging Face / Zenodo).
- The upstream **PaddleOCR** fork → pinned commit noted in `REPRODUCE.md`; not vendored here.

## Quick start

```bash
conda activate paddle          # Paddle 3.1 + PaddleOCR; jiwer, opencv, pillow
```

1. Get the datasets you need — see `DATA.md`.
2. Download model weights — see `MODELS.md`.
3. Reproduce any paper table — see `REPRODUCE.md` (each table → exact command).

## Method (one line)

Detector trained on **real data only** (script-agnostic; DB/PP-OCRv6 backbone); recognizer
**pretrained on synthetic crops then fine-tuned on real data** (script-dependent; SVTR encoder
+ CTC head over the frozen 1,638-char dictionary). See the paper for details.

## License

Code released for research purposes. Third-party datasets retain their original licenses
(see `DATA.md`).
