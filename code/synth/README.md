# Indic-aware synthetic scene-text generation

Generates the synthetic corpus used to **pretrain the IndicSTSpotter recognizer**:
25,000 multi-script scenes at 885×497, yielding ~2.26M word crops across
**11 Indic scripts + Latin**.

Text is shaped with **HarfBuzz/RAQM** (so conjuncts, vowel signs and RTL joining render
correctly), drawn with script-validated fonts, composited onto **real background photographs**,
and augmented with geometric, perspective, curved-text, photometric and typographic effects.
Placement is overlap-aware and sampling is script-balanced. Every scene carries polygon-level
word annotations.

---

## Requirements

```bash
pip install -r requirements.txt
```

Plus two external assets, which are **not redistributed here**:

- **Fonts** — per-script Unicode fonts (Devanagari, Bengali, Gujarati, Gurmukhi, Oriya, Tamil,
  Telugu, Kannada, Malayalam, Arabic/Urdu, Latin). Fonts carry their own licenses; obtain them
  from their upstream foundries/repositories. `fontlists/` records the exact set we used.
- **Backgrounds** — a folder of real photographs used as scene backdrops (`bg_dir` in the config).

## Pipeline

### 1. Install and validate fonts

Fonts are discovered through **fontconfig** (not a directory path), so they must be installed
and cached first:

```bash
python copyfonts.py                 # stage the per-script fonts into ~/.fonts
fc-cache -f ~/.fonts                # rebuild the fontconfig cache
python check_fonts.py               # HarfBuzz shaping check per script
```

`check_fonts.py` renders probe strings for each script and rejects any font that produces
missing glyphs (`.notdef`) or fails to shape conjuncts. Rejected fonts are recorded in
`bad_fonts.txt` / `bad_fonts.delete.txt` and skipped at generation time.

### 2. Build the vocabulary

```bash
python build_vocab.py               # -> vocab/<script>.txt
```

Per script, a deduplicated word list is built from multilingual corpora: languages sharing a
script are merged, native numerals retained, and punctuation-only tokens removed.

### 3. Generate scenes

**Recognizer corpus** (the paper's 25K set — dense 7×6 grid, ~121 words/scene):

```bash
bash generate_25k.sh
```

**Detector corpus** (sparser, realistic layout — used for the synth-only detector ablation):

```bash
GEN=generate_realistic.py CONFIG=dataset_config_det.json OUT=out_dataset_det bash generate_25k.sh
```

`generate_25k.sh` shards generation across `WORKERS` parallel processes (default 6) writing
disjoint global scene indices into one output dir, then merges the per-shard manifests.
Generation is CPU-bound, so throughput scales with cores:

```bash
WORKERS=12 bash generate_25k.sh            # more workers
BG_DIR=~/data/places365 bash generate_25k.sh   # override background source
```

Per-scene seeding makes every image reproducible regardless of shard count.

**Output** (`out_dataset_25k/`):

```
images/         scene_<idx>.jpg          rendered scenes
annotations/    scene_<idx>.json         polygon-level word annotations
manifest.jsonl                           one record per scene
```

### 4. Extract recognizer crops

```bash
python ../finetune/crop_words.py --root out_dataset_25k --out-dir <rec_data_dir>
```

Crops each annotated word into a training image and emits PaddleOCR-format rec labels
(`unified_train.txt` / `unified_val.txt`) — the ~2.26M crops used for recognizer pretraining.

> Labels longer than 50 characters must be filtered before training: over-long synthetic
> labels make the CTC target longer than the crop's feature width and produce NaN loss.
> The training scripts do this automatically.

## Configuration

| File | Used for |
|---|---|
| `dataset_config_25k.json` | recognizer corpus — 25,000 scenes, 885×497, 7×6 grid, script-balanced |
| `dataset_config_det.json` | detector corpus — realistic sparse layout |

Key knobs: `script_weights` (per-script sampling; Latin is up-weighted to reflect real signage),
`rotate_prob` / `perspective_prob` / `elastic_prob` / `curve_prob` (geometry), and the
photometric and typographic probabilities. CLI flags override config values.

## Files

| File | Role |
|---|---|
| `generate.py` | base renderer: font discovery, vocab sampling, placement, compositing, augmentation |
| `generate_transforms.py` | geometric/perspective/curve transform layer |
| `generate_grid_transforms.py` | dense grid generator — **the recognizer corpus** |
| `generate_realistic.py` | sparse realistic generator — the detector corpus |
| `generate_25k.sh` | sharded parallel driver + manifest merge |
| `build_vocab.py` | per-script vocabulary construction |
| `copyfonts.py`, `check_fonts.py`, `fontlists/` | font staging, HarfBuzz validation, font manifest |
| `../finetune/crop_words.py` | scenes → recognizer word crops + labels |
| `render_paper_gallery.py` | renders the sample-scene figure from the paper |
