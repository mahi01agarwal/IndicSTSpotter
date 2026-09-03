#!/usr/bin/env python3
"""Realistic-layout variant of the Indic scene-text generator.

Where generate.py fills the frame with a dense grid of text (max diversity),
this script aims for how text actually appears in real photos:

  - SPARSE: a few text groups per image (not dozens).
  - GROUPED like signs: each group is 1-3 words on one line sharing a single
    font / size / color (a signboard), instead of every word being random.
  - SMALL, distance-biased sizes (skewed toward small text).
  - MINIMAL overlap: groups essentially don't overlap (low IoU threshold).
  - MOSTLY HORIZONTAL: only isolated single words are occasionally rotated /
    given perspective.

Same output format and ground truth as generate.py (per-word instances, exact
quads). It reuses generate.py's low-level helpers, so generate.py is unchanged.

Run:
    python generate_realistic.py --bg-dir BG --out-dir out --num-scenes 50
"""

import os
import json
import argparse
import random

import numpy as np
from PIL import Image, ImageFilter

# Reuse the proven helpers (importing does not modify generate.py).
import generate as G
from generate_grid_transforms import reading_order


# ---------------------------------------------------------------------------
# Size sampling (biased toward small / distant text)
# ---------------------------------------------------------------------------

def sample_size(rng, lo, hi, bias):
    """Random size in [lo, hi], skewed toward `lo` (bias > 1 = smaller)."""
    return int(round(lo + (hi - lo) * (rng.random() ** bias)))


# ---------------------------------------------------------------------------
# Group style (shared across the words of one sign) + compositing
# ---------------------------------------------------------------------------

def make_style(color, rng, args):
    """Decide stroke/shadow/opacity/blur ONCE per group, for a consistent sign."""
    stroke = rng.random() < args.stroke_prob
    return {
        "color": color,
        "shadow": rng.random() < args.shadow_prob,
        "shoff": (rng.randint(1, 4), rng.randint(1, 4)),
        "stroke": stroke,
        "sw": rng.randint(1, 2),
        "scolor": (15, 15, 15) if sum(color) > 384 else (240, 240, 240),
        "opacity": (rng.uniform(0.6, 0.85)
                    if rng.random() < args.faded_prob else 1.0),
        "blur": rng.random() < args.inst_blur_prob,
    }


def stamp(canvas, mask_L, x, y, style, rng):
    """Composite one glyph mask onto the canvas using a group's style."""
    if style["shadow"]:
        sh = mask_L.filter(ImageFilter.GaussianBlur(2)).point(
            lambda v: int(v * 0.45))
        G.paste_mask(canvas, (0, 0, 0), sh,
                     x + style["shoff"][0], y + style["shoff"][1])
    if style["stroke"]:
        dil = mask_L.filter(ImageFilter.MaxFilter(2 * style["sw"] + 1))
        G.paste_mask(canvas, style["scolor"], dil, x, y)
    m = mask_L
    if style["blur"]:
        m = mask_L.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0)))
    G.paste_mask(canvas, style["color"], m, x, y, style["opacity"])


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def make_scene(bg, fonts_idx, vocab, rng, args):
    W, H = bg.size
    canvas = bg.convert("RGB")
    bg_arr = np.asarray(canvas)
    avail = [s for s in G.SCRIPTS if fonts_idx.get(s) and vocab.get(s)]

    placed, instances, debug = [], [], []
    n_groups = rng.randint(args.min_groups, args.max_groups)

    for _ in range(n_groups):
        script = rng.choice(avail)
        font = rng.choice(fonts_idx[script])
        size = sample_size(rng, args.min_size, args.max_size, args.size_bias)
        bold = rng.random() < args.bold_prob

        # how many words in this sign (weighted toward 1-2)
        ks = list(range(1, args.max_words + 1))
        n_words = rng.choices(ks, weights=[1.0 / k for k in ks])[0]
        words = [rng.choice(vocab[script]) for _ in range(n_words)]

        pairs = [(w, G.render_mask(w, font, size, bold)) for w in words]
        pairs = [(w, m) for (w, m) in pairs if m is not None]
        if not pairs:
            continue

        single = len(pairs) == 1
        gap = max(4, int(size * 0.3))

        # footprint: rotated/perspective box for a lone word, else horizontal row
        if single:
            angle = (rng.choice([90, -90, 45, -45, 30, -30, 15, -15])
                     if rng.random() < args.rotate_prob else 0)
            mask = pairs[0][1]
            dst, bw, bh = G.build_geometry(mask.width, mask.height,
                                           angle, rng, args)
        else:
            angle = 0
            bw = sum(m.width for _, m in pairs) + gap * (len(pairs) - 1)
            bh = max(m.height for _, m in pairs)

        if bw >= W or bh >= H or bw < 3 or bh < 3:
            continue

        # find a non-overlapping spot
        spot = None
        for _try in range(args.place_tries):
            x = rng.randint(0, W - bw)
            y = rng.randint(0, H - bh)
            box = (x, y, x + bw, y + bh)
            if all(G.iou(box, b) <= args.iou_thresh for b in placed):
                spot = (x, y)
                break
        if spot is None:
            continue
        x, y = spot
        placed.append((x, y, x + bw, y + bh))

        color = G.pick_color(bg_arr, (x, y, x + bw, y + bh), rng)
        style = make_style(color, rng, args)

        if single:
            mask = pairs[0][1]
            alpha = G.warp_alpha(mask, dst, bw, bh)
            stamp(canvas, alpha, x, y, style, rng)
            quad = [(x + px, y + py) for px, py in dst]
            _record(instances, debug, pairs[0][0], script, font, size,
                    angle, quad, font)
        else:
            cx = x
            for w, m in pairs:
                py = y + (bh - m.height) // 2
                stamp(canvas, m, cx, py, style, rng)
                quad = [(cx, py), (cx + m.width, py),
                        (cx + m.width, py + m.height), (cx, py + m.height)]
                _record(instances, debug, w, script, font, size, 0, quad, font)
                cx += m.width + gap

    return canvas, instances, debug


def _record(instances, debug, text, script, font_path, size, angle, quad, fp):
    xs = [q[0] for q in quad]
    ys = [q[1] for q in quad]
    bbox = [int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs) - min(xs))), int(round(max(ys) - min(ys)))]
    instances.append({
        "text": text, "script": script, "font": G.family_of(font_path),
        "font_size": size, "rotation_deg": angle, "bbox": bbox,
        "quad": [int(round(c)) for q in quad for c in q],
    })
    debug.append(([(int(q[0]), int(q[1])) for q in quad], text, fp))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Realistic (sparse, sign-like) Indic scene-text synth.")
    ap.add_argument("--bg-dir", default=None,
                    help="Background image dir. Required, but may come from --config.")
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--out-dir", default="./out_realistic")
    ap.add_argument("--num-scenes", type=int, default=20)

    # density / grouping. Measured on 50 real Marathi + 50 real Hindi scene photos:
    # ~11 words/image. Groups average ~1.6 words (n_words is weighted 1/k), so
    # 2-6 groups lands at ~3-10 words/image, i.e. the real range. The old 12-30
    # defaults gave ~20-48 words/image; the grid generator that actually produced
    # the 25k gave ~121. Check any new dataset with FineTune/det_dataset_stats.py
    # before training -- over-dense data is what makes the detector over-fire.
    ap.add_argument("--min-groups", type=int, default=2,
                    help="Min text groups (signs/labels) per image.")
    ap.add_argument("--max-groups", type=int, default=6,
                    help="Max text groups per image.")
    ap.add_argument("--max-words", type=int, default=4,
                    help="Max words per group (weighted toward 1-2).")
    ap.add_argument("--iou-thresh", type=float, default=0.05,
                    help="Essentially no overlap between groups.")
    ap.add_argument("--place-tries", type=int, default=80)

    # size: mostly small (distant), with a few larger -> wide spread, no giants
    ap.add_argument("--min-size", type=int, default=12)
    ap.add_argument("--max-size", type=int, default=50)
    ap.add_argument("--size-bias", type=float, default=2.0,
                    help=">1 skews sizes smaller (distant text).")

    # geometry (only isolated single words)
    ap.add_argument("--rotate-prob", type=float, default=0.12)
    ap.add_argument("--italic-prob", type=float, default=0.1)
    ap.add_argument("--perspective-prob", type=float, default=0.15)
    ap.add_argument("--perspective-frac", type=float, default=0.10)

    # appearance (subtle)
    ap.add_argument("--bold-prob", type=float, default=0.15)
    ap.add_argument("--stroke-prob", type=float, default=0.25)
    ap.add_argument("--shadow-prob", type=float, default=0.2)
    ap.add_argument("--faded-prob", type=float, default=0.1)
    ap.add_argument("--inst-blur-prob", type=float, default=0.1)

    # whole-image augmentation (light)
    ap.add_argument("--aug-prob", type=float, default=0.4)
    ap.add_argument("--hue-prob", type=float, default=0.2)
    ap.add_argument("--gray-prob", type=float, default=0.08)
    ap.add_argument("--img-blur-prob", type=float, default=0.15)
    ap.add_argument("--noise-prob", type=float, default=0.2)

    ap.add_argument("--resize", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    # ---- dataset-run plumbing (mirrors generate_grid_transforms.py) ----
    ap.add_argument("--start-index", type=int, default=0,
                    help="Global index of the first scene this process writes, so "
                         "generate_25k.sh can shard this script across workers.")
    ap.add_argument("--config", default=None,
                    help="JSON file whose keys set any option above "
                         "(underscore keys; CLI flags still override).")
    ap.add_argument("--script-weights", type=json.loads, default=None,
                    help='JSON like {"latin":1.5,"devanagari":1.0,...}.')
    ap.add_argument("--emit-transcription", action="store_true",
                    help="Also write a reading-order transcription + manifest.jsonl.")
    ap.add_argument("--only-script", default=None,
                    help="Restrict every word to this one script.")

    # ---- hard negatives (detector only) ----
    ap.add_argument("--negative-frac", type=float, default=0.0,
                    help="Fraction of scenes rendered with NO text at all. These are "
                         "the hard negatives: the detector sees real background clutter "
                         "(awnings, grilles, trusses) with every pixel supervised as "
                         "background. 0.15-0.25 is a sane range for det training; keep "
                         "at 0 for recogniser data, which cannot use textless images.")
    return ap


def main():
    pre = argparse.ArgumentParser(add_help=False)      # peek at --config first
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    ap = build_argparser()
    if known.config:
        with open(known.config, encoding="utf-8") as f:
            cfg = json.load(f)
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    args = ap.parse_args()
    if not args.bg_dir:
        raise SystemExit("--bg-dir is required (pass it on the CLI or in --config).")

    fonts_idx = G.discover_fonts()
    vocab = G.load_vocab(args.vocab_dir)
    if not any(fonts_idx[s] and vocab[s] for s in G.SCRIPTS):
        raise SystemExit("No script has both fonts and vocab. Check "
                         "`fc-cache -f ~/.fonts` and the vocab/ folder.")

    bgs = G.discover_backgrounds(args.bg_dir)
    if not bgs:
        raise SystemExit(f"No background images in {args.bg_dir}")
    if len(bgs) < 50:
        print(f"\n[bg] WARNING: only {len(bgs)} background image(s). Use "
              f"hundreds-thousands of diverse real scenes for a real dataset.\n")

    img_dir = os.path.join(args.out_dir, "images")
    ann_dir = os.path.join(args.out_dir, "annotations")
    dbg_dir = os.path.join(args.out_dir, "debug")
    for d in (img_dir, ann_dir, dbg_dir):
        os.makedirs(d, exist_ok=True)

    # save the exact settings used (shard 0 only, so parallel workers don't clobber it)
    if args.start_index == 0:
        with open(os.path.join(args.out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in vars(args).items() if k != "config"},
                      f, ensure_ascii=False, indent=2, default=str)
    man_name = ("manifest.jsonl" if args.start_index == 0
                else f"manifest.part_{args.start_index:08d}.jsonl")
    manifest = (open(os.path.join(args.out_dir, man_name), "w", encoding="utf-8")
                if args.emit_transcription else None)

    n_neg = 0
    end = args.start_index + args.num_scenes
    for i in range(args.start_index, end):
        # per-scene seed: process-independent, so any shard reproduces any index
        seed_i = (args.seed * 2654435761 + i) & 0xFFFFFFFF
        rng = random.Random(seed_i)
        np.random.seed(seed_i)

        bg = Image.open(rng.choice(bgs)).convert("RGB")
        if args.resize and max(bg.size) > args.resize:
            sc = args.resize / max(bg.size)
            bg = bg.resize((max(1, int(bg.width * sc)),
                            max(1, int(bg.height * sc))), Image.LANCZOS)

        # A hard negative is the background rendered with no text at all. It still
        # gets the same photometric augmentation, so negatives are not separable
        # from positives by colour/blur statistics -- only by containing no text.
        negative = rng.random() < args.negative_frac
        if negative:
            scene, instances, debug = bg.convert("RGB"), [], []
            n_neg += 1
        else:
            scene, instances, debug = make_scene(bg, fonts_idx, vocab, rng, args)
        scene = G.augment_scene(scene, rng, args)

        name = f"scene_{i:06d}"
        scene.save(os.path.join(img_dir, name + ".jpg"),
                   quality=rng.randint(70, 95))
        ann = {"image_file": name + ".jpg", "width": scene.width,
               "height": scene.height, "instances": instances,
               "negative": negative}
        if args.emit_transcription:
            text = reading_order(instances)
            ann["transcription"] = text
            if not negative:          # textless images are useless to the recogniser
                m = {"image": f"images/{name}.jpg", "text": text}
                if args.only_script:
                    m["script"] = args.only_script
                manifest.write(json.dumps(m, ensure_ascii=False) + "\n")
                manifest.flush()
        with open(os.path.join(ann_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        G.draw_debug(scene, debug).save(os.path.join(dbg_dir, name + ".jpg"),
                                        quality=92)
        print(f"[gen] {i - args.start_index + 1}/{args.num_scenes}  {name}: "
              f"{'NEGATIVE (no text)' if negative else f'{len(instances)} word instance(s)'}")

    if manifest:
        manifest.close()

    print(f"\n[done] {args.num_scenes} scenes ({n_neg} negatives) -> {args.out_dir}")
    print(f"  images:      {img_dir}")
    print(f"  annotations: {ann_dir}")
    print(f"  debug:       {dbg_dir}")


if __name__ == "__main__":
    main()
