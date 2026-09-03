# Indic Scene-Text Synth (grid layout, for text spotting)

Generate a synthetic training dataset for **Indic scene-text spotting** — both
**detection** (where text is) and **recognition** (what it says). The goal is a
spotter that finds and reads text *anywhere* in a natural scene, at any size,
color, font, script, position, or orientation.

So text is placed **all over** the image (no depth/segmentation/surface fitting —
that's intentional), using a **random grid layout** that varies text density
across the image and across the dataset. This prevents the spotter from learning
shortcuts like "text is always top-right" or "text is always large."

Scripts: **Telugu, Devanagari (Hindi), Tamil, Kannada, Malayalam, Bengali,
Latin (English)** — mixed within each image.

## How the layout works

The image is divided into a grid of cells (`--cell-size`). For each cell we
randomly decide:

- whether it's empty (`--empty-cell-prob`) — creates sparse regions,
- how many instances to place if not (`1..--max-per-cell`) — creates dense
  regions,
- and where within the cell each instance goes.

So some regions are text-dense, others empty, and the pattern differs every
image. Within placement, a new instance is rejected (re-tried, then skipped) if
its box overlaps an existing one with **IoU > `--iou-thresh`** (default 0.4) —
partial overlap is fine and realistic; full occlusion would make an instance
invisible and useless for training.

## Diversity / augmentations

The dataset is deliberately pushed to be as visually varied as possible so the
spotter generalizes instead of memorizing. Everything is probability-gated and
tunable from the CLI.

**Per instance**
- **Content:** single words, multi-word phrases (`--multiword-prob`), Latin
  case variation (lower/UPPER/Title), and numeric/alphanumeric tokens
  (`--number-prob`, e.g. `42-310`, `9:05`, `128B`).
- **Font & weight:** random font for the script, plus faux-**bold**
  (`--bold-prob`). Real bold/italic font files are picked up by `fc-list` too.
- **Size:** `--min-size`..`--max-size` (default 20–80 px) for scale invariance.
- **Geometry (one homography → exact quad):** rotation (`--rotate-prob`, from
  {90,-90,45,-45,30,-30}), faux-**italic** shear (`--italic-prob`), and mild
  **perspective** jitter (`--perspective-prob`, `--perspective-frac`). All three
  are composed into a single warp, and the recorded `quad` is the warp
  destination — so the label is pixel-exact regardless of the distortion.
- **Color:** random foreground color, contrast-checked against the local
  background (light text on dark, dark on light).
- **Style:** outline/**stroke** (`--stroke-prob`), drop **shadow**
  (`--shadow-prob`), partial **transparency** (`--faded-prob`), and per-instance
  **defocus blur** (`--inst-blur-prob`).

**Whole image** (applied after compositing — labels unaffected)
- brightness / contrast / saturation jitter (`--aug-prob`), **hue** shift
  (`--hue-prob`), occasional **grayscale** (`--gray-prob`), **gaussian noise**
  (`--noise-prob`), light **blur** (`--img-blur-prob`), and **randomized JPEG
  quality** (60–95) for realistic compression artifacts.

Run `python generate.py -h` for the full list of knobs and defaults.

## Setup

Pillow must be built **with Raqm** for correct Indic shaping — the prebuilt
conda/pip Pillow is **not** linked against libraqm even if libraqm is installed.
This exact sequence is the working fix:

```bash
conda create -n indic_synth python=3.10 -y
conda activate indic_synth
conda install -c conda-forge libraqm -y
pip install --no-binary :all: --no-cache-dir pillow
# verify -> must print True:
python -c "from PIL import features; print(features.check('raqm'))"

pip install numpy opencv-python-headless tqdm
```

**Fonts:** install your Indic fonts in `~/.fonts/` and register them so
fontconfig sees them:

```bash
fc-cache -f ~/.fonts
# sanity check (should list font files):
fc-list :lang=te file        # Telugu
fc-list :lang=ta file        # Tamil
```

`generate.py` discovers fonts at startup via `fc-list :lang=<code>` per script —
**nothing is hardcoded**. If a script shows 0 fonts at startup, fontconfig can't
see a font for it; fix `~/.fonts` + `fc-cache`.

## Run

```bash
python generate.py \
    --bg-dir path/to/backgrounds \
    --vocab-dir ./vocab \
    --out-dir ./out \
    --num-scenes 50 \
    --cell-size 200 --max-per-cell 8 --empty-cell-prob 0.3 \
    --min-size 20 --max-size 80 \
    --rotate-prob 0.25 --bold-prob 0.2 \
    --iou-thresh 0.4
```

## Output

```
out/
├── images/        scene_00000.jpg ...           the scene images
├── annotations/   scene_00000.json ...          one JSON per image (same name)
└── debug/         scene_00000.jpg ...           quads + text drawn, for inspection
```

Each annotation JSON:

```json
{
  "image_file": "scene_00000.jpg",
  "width": 1024,
  "height": 768,
  "instances": [
    {
      "text": "నగరం",
      "script": "telugu",
      "font": "Noto Sans Telugu",
      "font_size": 48,
      "rotation_deg": 0,
      "bbox": [x, y, w, h],
      "quad": [x1, y1, x2, y2, x3, y3, x4, y4]
    }
  ]
}
```

`bbox` is the axis-aligned box; `quad` is the tight 4-corner polygon (for rotated
text the pre-rotation box corners are transformed by the same rotation), which is
much tighter than an axis-aligned box around rotated text — better for training a
spotter.

The **debug images** draw the quads and the rendered text so you can eyeball
layout quality without any extra tooling.

## Vocab

Starter word lists live in `vocab/*.txt` (UTF-8, one word per line, `#`
comments). They are **small bootstrap lists (~60 words each)** — see the comment
at the top of each file. For a real dataset, replace them with a larger corpus
(e.g. words mined from real signage or scene-text benchmark transcriptions); no
code change needed, just drop in bigger files with the same names.

## Backgrounds

Bring your own — a folder of natural scene photos (`--bg-dir`, searched
recursively). **Only a handful of test images won't generalize:** a real
training run needs **hundreds to thousands of diverse real scene images**, or the
spotter overfits the backgrounds. The generator warns if it finds fewer than 50.

## What to upgrade next

- **More & more diverse backgrounds** — now the single biggest remaining lever
  for generalization (the per-instance/whole-image augmentations are already in).
- **Larger vocab** per script — swap the `vocab/*.txt` files (same names).
- **Even richer effects if your domain needs them** — curved / multi-line text,
  emboss/gradient fills, motion blur, raindrop/glare overlays, and a continuous
  (vs discrete) rotation range. Hooks are in `make_scene` / `augment_scene`.
- **Recognition crops** — if you also want word-crop training data, a small
  script can cut each `quad` out of the images using the JSON.
```
