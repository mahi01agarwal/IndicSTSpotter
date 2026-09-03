# IndicSTSpotter — Reproduction Guide

How to reproduce every table in the paper from the checkpoints and code in this repo.
All paths are relative to `~/IndicSTSpotter` on the dev server unless noted.

> **Metric note:** the eval scripts print WRR/CRR as **fractions** (e.g. `0.7130`).
> The paper reports **percent** (`71.3`). Multiply by 100.
> Every per-script / per-language table also prints `__ALL__` (micro, word-pooled)
> and `__MACRO__` (unweighted mean over groups). **The paper's "Avg" column is `__MACRO__`.**

---

## 0. Environment

```bash
conda activate paddle          # Paddle 3.1 + PaddleOCR; jiwer, opencv, pillow
```

- Recognizer/detector base: `PP-OCRv6_medium_rec` / `PP-OCRv6_medium_det`
  (this is the `--rec-name` / `--det-name` every eval expects; it's the default).
- **Frozen character dictionary:** `dicts/unified_dict.txt` — **1638 chars**.
  All rec models share this exact dict; the CTC head is `[·, 1640]` (1638 + space + blank).
  ⚠ Do **not** substitute `indic_full_dict.txt` (1698) or `unified_charlist.txt` (891) —
  a wrong dict shifts the CTC class indices and produces garbage. See §Dictionary.

---

## 1. Model zoo

| Name | Inference dir | Checkpoint | What it is |
|---|---|---|---|
| **stage2A** ("IndicSTSpotter" rec) | `inference/rec_stage2` | `checkpoints/rec_stage2/best_model` | real-only fine-tune, warm-started from **balanced**, frozen 1638 dict. **The paper's headline recognizer (T3, T5, T6).** |
| stage2B | `inference/rec_stage2_B` | `checkpoints/rec_stage2_B/best_model` | stage-2 variant (A′/B tie-breaker). Internal ablation. |
| balanced | `inference/ablations/rec_balanced` | `checkpoints/ablations/rec_balanced` | synth+real balanced; warm-start source for stage2A. |
| stage1 | `inference/rec_stage1` | `checkpoints/rec_stage1/best_model` | synth-only pretrain (1638 dict). Ablation ladder. |
| **det_mist** (detector) | `inference/det_mist` | `checkpoints/det_mist/best_model` | DB detector, real + MIST. **The paper's headline detector** (dense scenes, eval @1920). |
| det_real | `inference/det_real` | `checkpoints/det_real/best_model` | DB detector, real-only (Bharat+MLT17+MLT19). Ablation (no MIST). |

Other ablation exports: `inference/ablations/{det_synthonly, det_synthonly_ema, rec_unified_1440dict, v5_devanagari_specdict}`.

---

## 2. Reproduce each table

### Table 3 — IndicSTR12 word recognition (WRR/CRR, 11 languages)

IndicSTR12 crops are flat in one `images/` dir with the language in the filename prefix,
so build a per-language *view* first (symlinks; the crop's true language = filename prefix):

```bash
cd ~/IndicSTSpotter/eval
rm -rf indicstr12_bylang indicstr12_rec_test_bylang.txt; mkdir -p indicstr12_bylang
BY=$(pwd)/indicstr12_bylang
awk -F'\t' -v by="$BY" '{n=split($1,a,"/"); f=a[n]; split(f,b,"_"); lang=b[1]; dir=by"/"lang; if(!(lang in seen)){system("mkdir -p \""dir"\""); seen[lang]=1} system("ln -sf \""$1"\" \""dir"/"f"\""); print dir"/"f"\t"$2}' \
  ~/IndicSTSpotter/data/rec_labels/indicstr12/indicstr12_rec_test.txt > indicstr12_rec_test_bylang.txt
```

```bash
cd ~/IndicSTSpotter && python code/finetune/eval_rec_gt.py \
  --labels eval/indicstr12_rec_test_bylang.txt --group-by language \
  --model stage2A inference/rec_stage2 \
  --out eval/T3_indicstr12
```

`__MACRO__` should read **0.7130 WRR / 0.9070 CRR** (paper: 71.3 / 90.7).
Add `--model stage1 inference/rec_stage1 --model balanced inference/ablations/rec_balanced --model stage2B inference/rec_stage2_B` for the full ablation ladder.

### Table 5 — IIIT-IndicSTR-Word recognition (WRR/CRR, 10 languages)

IIIT test labels are regenerated from the raw data (test split is held out, not shipped):

```bash
python ~/IndicSTSpotter/code/finetune/data_prep/iiit_indicstr_to_paddle.py \
  --root ~/IndicSTSpotter/data/iiit_indicstr --splits test \
  --out ~/IndicSTSpotter/data/rec_labels/iiit_indicstr_rec_test.txt
```

IIIT crops already sit under `<lang>/test/images/`, so 10 directory-level symlinks give the per-language view:

```bash
cd ~/IndicSTSpotter/eval
rm -rf iiit_bylang iiit_indicstr_rec_test_bylang.txt; mkdir -p iiit_bylang
for L in $(ls ~/IndicSTSpotter/data/iiit_indicstr); do [ -d ~/IndicSTSpotter/data/iiit_indicstr/$L/test/images ] && ln -sfn ~/IndicSTSpotter/data/iiit_indicstr/$L/test/images iiit_bylang/$L; done
BY=$(pwd)/iiit_bylang
awk -F'\t' -v by="$BY" '{n=split($1,a,"/"); lang=a[n-3]; f=a[n]; print by"/"lang"/"f"\t"$2}' \
  ~/IndicSTSpotter/data/rec_labels/iiit_indicstr_rec_test.txt > iiit_indicstr_rec_test_bylang.txt
```

```bash
cd ~/IndicSTSpotter && python code/finetune/eval_rec_gt.py \
  --labels eval/iiit_indicstr_rec_test_bylang.txt --group-by language \
  --model stage2A inference/rec_stage2 \
  --out eval/T5_iiit
```

Paper "Ours": WRR avg **74.4**, CRR avg **91.0**.

### Table 4 — Word detection P/R/F1 (BhashiniSTR + BSTD)

Detector-only, ICDAR-2015 IoU≥0.5 matching. Dense scenes need the high-res inference cap (1920).

```bash
cd ~/IndicSTSpotter && python code/finetune/_archive/eval_reference/eval_det_real.py \
  --images-dir <BHASHINISTR_SCENES_DIR> \
  --ft-det-dir inference/det_mist --det-limit-side-len 1920 \
  --out eval/T4_bhashini_det
```

⚠ **VERIFY (a):** confirm BhashiniSTR is stored as scene images + per-image GT JSON
(`{"words":[{x,y,w,h,text}]}`) that `eval_det_real.py --images-dir` expects.
⚠ **VERIFY (b):** how BSTD detection P/R/F1 was produced — `eval_det_real.py` takes an
images-dir, but BSTD det GT is a label file (`bharat_det_test.txt`). Either BSTD test was
also materialized as images+JSON, or a label-file path was used. Resolve before quoting T4.

### Table 6 — BhashiniSTR recognition (oracle det) + end-to-end (WRR, 11 lang)

Recognition on GT boxes (oracle detection):
```bash
cd ~/IndicSTSpotter && python code/finetune/eval_rec_gt.py \
  --images-dir <BHASHINISTR_SCENES_DIR> --group-by language \
  --model stage2A inference/rec_stage2 --out eval/T6_bhashini_recGT
```
Full end-to-end (detector + recognizer):
```bash
cd ~/IndicSTSpotter && python code/finetune/_archive/eval_reference/eval_e2e_real.py \
  --images-dir <BHASHINISTR_SCENES_DIR> \
  --det-dir inference/det_mist --rec-dir inference/rec_stage2 \
  --det-limit-side-len 1920 --out eval/T6_bhashini_e2e
```
Paper: GT+Rec (oracle det) macro **70.6** WRR; full e2e **45.3** WRR.

### Table 7 — BSTD page-level end-to-end (P/R/F1 per script + GT+Rec ceiling)

```bash
cd ~/IndicSTSpotter && python code/finetune/_archive/eval_reference/eval_e2e_bharat_perscript.py \
  --labels data/det_data_real/bharat/bharat_det_test.txt \
  --det-dir inference/det_mist \
  --model stage2A inference/rec_stage2 \
  --det-limit-side-len 1920 --out eval/T7_bstd_e2e
```
The **GT+Rec (Ceiling)** row = recognition on GT boxes (see Table 8 oracle-TD command).
Paper full-system BSTD: **57.0 F1 / 45.3 WRR / 79.1 CRR**.

### Table 8 — BSTD WRR/CRR per script (full e2e, +oracle TD)

Full end-to-end per-script WRR/CRR: same command as Table 7 (it prints e2e WRR/CRR per script).
Paper: **33.8 WRR / 47.6 CRR**.

**+oracle TD** (feed GT boxes → recognizer only; per **script**, merged Beng\*/Deva\* as in the caption):
```bash
cd ~/IndicSTSpotter && python code/finetune/_archive/eval_reference/eval_rec_perscript.py \
  --labels data/rec_labels/bharat_rec_test.txt --group-by script \
  --model stage2A inference/rec_stage2 --out eval/T8_bstd_oracleTD
```
Paper +oracle TD: **57.8 WRR / 83.9 CRR**. (The **+SI** row is the *routed baseline*
IndicPhotoOCR+oracle TD+SI — a comparison system, **not** an IndicSTSpotter component.)

---

## 3. Data preparation (regenerate benchmarks from raw datasets)

Converters live in `code/finetune/data_prep/`. Downloads: `download_*.sbatch/.sh`.

| Dataset | Converter | Output |
|---|---|---|
| BSTD (Bharat) rec | `bharat_rec_to_paddle.py` | `bharat_rec_{train,test}.txt` |
| BSTD (Bharat) det | `bharat_det_to_paddle.py` | `bharat_det_{train,test}.txt` |
| MLT17/19 | `mlt_crop_rec.py` / `mlt_to_db.py` | rec crops / det labels |
| MIST | `mist_to_db.py` | `mist_det_{train,test}.txt` |
| IIIT-IndicSTR-Word | `iiit_indicstr_to_paddle.py` | `iiit_indicstr_rec[_test].txt` |
| IndicSTR12 | `indicstr12_to_paddle.py` | `indicstr12_rec[_test].txt` (scene-level split, seed 1234, test-frac 0.2) |
| merge (rec/det) | `merge_rec_db.py` / `merge_det_db.py` | pooled train sets |

**Leak-free splits:** IndicSTR12 uses a deterministic per-**scene** md5 split (seed 1234) so no
signboard straddles train/test. IIIT/BSTD ship their own test splits (held out).

**License note:** the release ships the **converters + split lists**, not raw third-party
images (BSTD, MLT, IIIT-IndicSTR, IndicSTR12, MIST each have their own licenses). The
**synthetic data + generator** are ours to release.

---

## 4. Training (how each checkpoint was produced)

Launchers in `code/finetune/data_prep/*.sbatch` (4×GPU, `paddle.distributed`). Each warm-starts
from a stock PP-OCRv6 weight and writes its config via a generator, then `tools/train.py`.

| Model | Launcher | Data | Init | Dict |
|---|---|---|---|---|
| det_real | `train_v6_det_ada.sbatch` | Bharat+MLT17+MLT19 | stock det | — |
| det_mist | `train_v6_det_mist_ada.sbatch` | + MIST-Train | stock det (WARMSTART=stock) | — |
| rec stage1 | `train_v6_rec_stage1_synth_ada.sbatch` | synth pool (~2.5M crops / 25k scenes) | stock rec | frozen 1638 |
| rec balanced | `train_v6_rec_ada.sbatch` | synth + real | stock rec | builds 1638 |
| rec stage2A | `train_v6_rec_stage2_ada.sbatch` | real-only | **balanced** best_accuracy | frozen 1638 |

⚠ **VERIFY (c):** the sbatch files call `make_config.py` / `make_det_config.py` to generate the
training `.yml`. Confirm these generators are in the repo (`code/finetune/`), and that the exact
generated configs (`my_v6_det_real.yml`, `my_v6_det_mist.yml`, `my_v6_rec_synthonly.yml`,
`my_v6_rec_stage2.yml`) are saved. Only `my_v6_unified.yml` + `my_devanagari_specdict.yml` were
found in `code/configs/` — the rest are generated at train time and must be preserved for release.

Export a checkpoint to an inference dir (as done for rec_stage2):
```bash
python code/PaddleOCR/tools/export_model.py -c <config.yml> \
  -o Global.pretrained_model=checkpoints/<model>/best_model/model \
     Global.save_inference_dir=inference/<model> \
     Global.character_dict_path=/home/mahima/IndicSTSpotter/dicts/unified_dict.txt
```

---

## 5. Dictionary (critical)

`dicts/unified_dict.txt` (1638 lines) is the frozen shared vocabulary. It was **recovered
verbatim** from the `character_dict` embedded in exported models' `inference.yml` (byte-identical
across `rec_stage1` and `rec_balanced` exports). It is **not** rebuildable from data (the exact
set depends on unrecorded converter args). **Any exported PaddleX model is a dict backup.**

The dict is intentionally "dirty" (contains a few Hangul/CJK/BOM/U+FFFD chars leaked from MLT
multilingual crops) — the model has real classes for them; removing any char requires retraining.

---

## Open punch-list (before release)

- [ ] **VERIFY (a)** BhashiniSTR on-disk format (scenes + GT JSON?) — blocks T4, T6.
- [ ] **VERIFY (b)** BSTD detection P/R/F1 mechanism for T4.
- [ ] **VERIFY (c)** `make_config.py` / `make_det_config.py` present + all training configs saved.
- [ ] Confirm `bharat_det_test.txt` and `mist_det_test.txt` exist (T7, T8, MIST benchmark).
- [ ] README (overview, install, model download links), `requirements.txt` / conda export.
- [ ] Model hosting for large checkpoints (HF / Drive).
- [ ] Deduplicate `code/synth/FineTune/` (mirrors `code/finetune/`).
- [ ] Quarantine/label `indic_full_dict.txt` (1698) and `unified_charlist.txt` (891) so no one grabs the wrong dict.
- [ ] Finish MLT19 download (training data; low priority).
