# Datasets

IndicSTSpotter is trained and evaluated on public third-party datasets plus our own
synthetic data. **We do not redistribute the third-party datasets** — obtain each from its
official source (each retains its own license), then run our converters to reproduce the
unified PaddleOCR format and our leak-free splits.

Download helpers and converters live in `code/finetune/data_prep/`.

## Real datasets

| Dataset | Paper ref | Use | Get it | Convert with |
|---|---|---|---|---|
| **Bharat Scene Text (BSTD)** | [6] | det + rec train, rec/e2e test | `data_prep/download_bharat.sh` (Google Drive via gdown) | `bharat_det_to_paddle.py`, `bharat_rec_to_paddle.py` |
| **MLT-2017** | [34] | det + rec train | `data_prep/download_mlt17_aria.sbatch` (open mirror) | `mlt_to_db.py`, `mlt_crop_rec.py` |
| **MLT-2019** | [33] | det + rec train | `data_prep/download_mlt19_aria.sbatch` | `mlt_to_db.py`, `mlt_crop_rec.py` |
| **MIST** | [31] | det train (dense) + test | `data_prep/download_mist_aria.sbatch` (CVIT) | `mist_to_db.py` |
| **IIIT-IndicSTR-Word** | [30] | rec train + test (Table 5) | IIIT-Hyderabad (see paper ref) | `iiit_indicstr_to_paddle.py` |
| **IndicSTR12** | [26] | rec train + test (Table 3) | `https://cvit.iiit.ac.in/images/datasets/IndicSTR12/real.zip` | `indicstr12_to_paddle.py` |
| **BhashiniSTR** | — | held-out test only (T4, T6) | (held-out benchmark; see paper) | scenes + per-image GT JSON |

Each `download_*` script has the exact URLs/commands. Run the matching converter to emit
PaddleOCR label files under `data/rec_labels/` and `data/det_data_real/`.

## Splits (leak-free, reproducible)

- **IndicSTR12** ships no split → we make a deterministic per-**scene** split
  (md5 of `<lang>/<scene>`, `--split-seed 1234 --test-frac 0.2`) so no signboard straddles
  train/test. Re-running the converter reproduces the identical split on any machine.
- **IIIT-IndicSTR-Word**, **BSTD**, **MIST** ship their own train/test splits — test held out.
- **BhashiniSTR** is entirely held out (never used for training).

## Synthetic data (ours)

The synthetic corpus (25K scenes / ~2.26M word crops) is generated, not shipped. Release the
**generator** rather than the crops:
- `code/synth/` + the Indic-aware renderer (HarfBuzz/RAQM shaping, script-aware fonts,
  real-background compositing).
- Config: `dataset_config_25k.json` (scene count, per-script sampling, augmentations).

Regenerate with the renderer; a snapshot may optionally be mirrored on Zenodo/HF for convenience.

## Dictionary

`dicts/unified_dict.txt` (1,638 chars) is the frozen shared vocabulary built from the union
of the relevant Unicode blocks + characters observed in the combined synth+real corpus. It is
**checked into git** (small, and reproduction depends on it exactly). Do not substitute another
dict — see `REPRODUCE.md` §Dictionary.
