<div align="center">

# IndicSTSpotter

### Multilingual Scene Text Spotting: How Far Can We Go with One OCR?

A single **detector** + single **recognizer** over one **frozen 1,638-character dictionary** —
covering **11 Indic scripts + Latin** with *no script identification* and *no per-language weights*.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](requirements.txt)
[![PaddlePaddle 3.1](https://img.shields.io/badge/PaddlePaddle-3.1-orange.svg)](THIRD_PARTY.md)
[![Models on 🤗](https://img.shields.io/badge/🤗%20Models-Hugging%20Face-yellow.svg)](https://huggingface.co/MahimaSinghal/IndicSTSpotter)

</div>

<!-- TEASER: add a qualitative results figure here for maximum impact, e.g.
<p align="center"><img src="assets/teaser.png" width="90%"></p>
Drop a detect+recognize grid on real Indic signage into assets/ and uncomment. -->

---

## Overview

Scene text recognition is largely solved for high-resource languages, but multilingual Indic
text remains hard: 12 writing systems, dense conjuncts, visually similar scripts, and scarce
real annotations. Most systems answer this with **per-script models** or a **script-identification
router** — both of which grow linearly in parameters and can misroute.

**IndicSTSpotter** asks how far a *single* unified model can go instead. The result: one
15.8M-parameter recognizer covering all 11 Indic scripts + Latin that **matches or surpasses
specialized systems 16–200× its size** on character recognition — with the residual end-to-end
error dominated by *detection* in dense scenes, not recognition.

## Highlights

- **One recognizer, every script.** A shared SVTR + CTC recognizer over a frozen 1,638-char
  dictionary — no per-language heads, no script router.
- **Tiny.** 15.8M-param recognizer / 31.3M total spotter vs. 330M–3.3B for routed & expert-merged baselines.
- **Task-specific training.** Detector on **real data only** (script-agnostic); recognizer
  **synthetic-pretrained then real-fine-tuned** (script-dependent).
- **Reproducible.** Unified benchmark suite with leak-free splits, pinned deps, released weights,
  and a table-by-table [reproduction guide](REPRODUCE.md).

## Results

Word recognition on held-out benchmarks (macro-average, %; higher is better):

| Benchmark | Languages | WRR | CRR |
|---|:--:|:--:|:--:|
| IndicSTR12 | 11 | **71.3** | **90.7** |
| IIIT-IndicSTR-Word | 10 | **74.4** | **91.0** |

Parameter efficiency — recognizer size for comparable or broader script coverage:

| System | Coverage | Recognizer params |
|---|:--:|:--:|
| CRNN | per-script (11) | 91.3M |
| PARSeq | per-script (11) | 261.8M |
| IndicPhotoOCR (routed) | 11 + L | 356M |
| **IndicSTSpotter (ours)** | **unified (11 + L)** | **15.8M** |

See the [paper](#citation) for the full per-script/per-language tables and end-to-end results.

## Installation

```bash
conda create -n paddle python=3.10 && conda activate paddle
pip install -r requirements.txt
```

Set up the pinned PaddleOCR fork (required for training; not for inference) — see
[THIRD_PARTY.md](THIRD_PARTY.md).

## Quick start

```python
from huggingface_hub import snapshot_download
from paddleocr import TextDetection, TextRecognition

d   = snapshot_download("MahimaSinghal/IndicSTSpotter", repo_type="model")
det = TextDetection(model_name="PP-OCRv6_medium_det", model_dir=f"{d}/det_mist")
rec = TextRecognition(model_name="PP-OCRv6_medium_rec", model_dir=f"{d}/rec_stage2")

# detect word boxes -> crop -> recognize. See eval scripts in code/finetune/ for the full pipeline.
```

## Reproduce the paper

Every table maps to an exact command in **[REPRODUCE.md](REPRODUCE.md)**. Example — IndicSTR12
word recognition (Table 3):

```bash
python code/finetune/eval_rec_gt.py \
  --labels eval/indicstr12_rec_test_bylang.txt --group-by language \
  --model stage2A inference/rec_stage2 --out eval/T3_indicstr12
```

## Models & data

- **Models** → [MODELS.md](MODELS.md) · weights on [🤗 Hugging Face](https://huggingface.co/MahimaSinghal/IndicSTSpotter)
- **Datasets** → [DATA.md](DATA.md) — official download links + our converters (third-party datasets are not redistributed)

## Repository structure

```
IndicSTSpotter/
├── code/
│   ├── finetune/        # dataset converters, dict builder, config generators, eval scripts
│   ├── configs/         # custom PaddleOCR training configs
│   └── synth/           # Indic-aware synthetic scene generator
├── dicts/
│   └── unified_dict.txt # the frozen 1,638-character shared dictionary
├── patches/             # required PaddleOCR patch (see THIRD_PARTY.md)
├── REPRODUCE.md         # table-by-table reproduction
├── DATA.md · MODELS.md · THIRD_PARTY.md
└── requirements.txt · LICENSE
```

Datasets, checkpoints, and the PaddleOCR fork are obtained separately (see the docs above) and
are intentionally not committed.

## Citation

> Under anonymous review. Citation and BibTeX will be added upon publication.

## License

Released under the [MIT License](LICENSE) for research purposes. Built on
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0); third-party datasets retain
their original licenses (see [DATA.md](DATA.md)).
