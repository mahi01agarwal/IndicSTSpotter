#!/usr/bin/env python3
"""Per-cell packed Indic scene-text generator.

Layout model (what this script does differently):
  - Divide the image into a ROWS x COLS grid (e.g. 3x3 = 9 cells).
  - Each cell gets its OWN random density: some cells dense (e.g. ~15 texts),
    some sparse, some empty.
  - Every text in a cell is rendered at a random size (mix of very small and
    big), random color, and is kept ENTIRELY INSIDE that cell.
  - Within a cell, texts do NOT overlap (a small gap is enforced). Because each
    text is contained in its cell, cross-cell overlap can't happen either.

Same output format / exact-quad ground truth as the other scripts; reuses
generate.py's helpers, so generate.py is unchanged.

Run:
    python generate_grid_cells.py --bg-dir BG --out-dir out --rows 3 --cols 3
"""

import os
import json
import argparse
import random

import numpy as np
from PIL import ImageFilter, Image

import generate as G


def _overlap(a, b, margin):
    """True if boxes a,b are closer than `margin` (overlap or too tight)."""
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def try_place(cell, cell_boxes, fonts_idx, vocab, avail, bg_arr, rng, args,
              size_cap):
    """Attempt to render+position one text inside `cell` without overlap.

    Returns (box, alpha_L, quad, color, word, script, font, size, angle) or None.
    Retries smaller sizes if a candidate doesn't fit the cell.
    """
    cx0, cy0, cx1, cy1 = cell
    cw, ch = cx1 - cx0, cy1 - cy0
    for _s in range(args.size_tries):
        script = rng.choice(avail)
        font = rng.choice(fonts_idx[script])
        word = rng.choice(vocab[script])
        # bias sizes small so many texts fit inside one cell
        size = int(round(args.min_size +
                         (size_cap - args.min_size) * (rng.random() ** args.size_bias)))
        bold = rng.random() < args.bold_prob

        mask = G.render_mask(word, font, size, bold)
        if mask is None:
            continue
        angle = (rng.choice([90, -90, 30, -30, 15, -15])
                 if rng.random() < args.rotate_prob else 0)
        dst, pw, ph = G.build_geometry(mask.width, mask.height, angle, rng, args)
        if pw > cw or ph > ch:
            continue  # too big for the cell -> try a smaller size

        for _p in range(args.pos_tries):
            x = rng.randint(cx0, cx1 - pw)
            y = rng.randint(cy0, cy1 - ph)
            box = (x, y, x + pw, y + ph)
            if all(not _overlap(box, b, args.margin) for b in cell_boxes):
                alpha = G.warp_alpha(mask, dst, pw, ph)
                quad = [(x + px, y + py) for px, py in dst]
                color = G.pick_color(bg_arr, box, rng)
                return (box, alpha, quad, color, word, script, font, size, angle)
    return None


def composite(canvas, alpha, x, y, color, rng, args):
    """Composite one glyph mask with per-instance shadow/stroke/opacity/blur."""
    if rng.random() < args.shadow_prob:
        sh = alpha.filter(ImageFilter.GaussianBlur(2)).point(
            lambda v: int(v * 0.45))
        G.paste_mask(canvas, (0, 0, 0), sh,
                     x + rng.randint(1, 3), y + rng.randint(1, 3))
    if rng.random() < args.stroke_prob:
        dil = alpha.filter(ImageFilter.MaxFilter(3))
        sc = (15, 15, 15) if sum(color) > 384 else (240, 240, 240)
        G.paste_mask(canvas, sc, dil, x, y)
    a2 = alpha
    if rng.random() < args.inst_blur_prob:
        a2 = alpha.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0)))
    opacity = (rng.uniform(0.6, 0.85) if rng.random() < args.faded_prob else 1.0)
    G.paste_mask(canvas, color, a2, x, y, opacity)


def make_scene(bg, fonts_idx, vocab, rng, args):
    W, H = bg.size
    canvas = bg.convert("RGB")
    bg_arr = np.asarray(canvas)
    avail = [s for s in G.SCRIPTS if fonts_idx.get(s) and vocab.get(s)]

    instances, debug = [], []

    for r in range(args.rows):
        if rng.random() < args.row_empty_prob:
            continue  # leave an entire row empty (band of no text)
        for c in range(args.cols):
            cx0, cx1 = int(c * W / args.cols), int((c + 1) * W / args.cols)
            cy0, cy1 = int(r * H / args.rows), int((r + 1) * H / args.rows)
            cw, ch = cx1 - cx0, cy1 - cy0
            if cw < 10 or ch < 10:
                continue
            if rng.random() < args.empty_cell_prob:
                continue  # leave this cell empty

            count = rng.randint(1, args.max_per_cell)
            # cap size so even rotated text can fit the cell; min 1 above floor.
            size_cap = max(args.min_size + 1,
                           min(args.max_size, int(min(cw, ch) * 0.8)))
            cell_boxes = []

            for _ in range(count):
                res = try_place((cx0, cy0, cx1, cy1), cell_boxes, fonts_idx,
                                vocab, avail, bg_arr, rng, args, size_cap)
                if res is None:
                    continue  # cell too crowded for another -> stop trying this one
                box, alpha, quad, color, word, script, font, size, angle = res
                cell_boxes.append(box)
                composite(canvas, alpha, box[0], box[1], color, rng, args)

                xs = [q[0] for q in quad]
                ys = [q[1] for q in quad]
                bbox = [int(round(min(xs))), int(round(min(ys))),
                        int(round(max(xs) - min(xs))),
                        int(round(max(ys) - min(ys)))]
                instances.append({
                    "text": word, "script": script, "font": G.family_of(font),
                    "font_size": size, "rotation_deg": angle, "bbox": bbox,
                    "quad": [int(round(v)) for q in quad for v in q],
                })
                debug.append(([(int(q[0]), int(q[1])) for q in quad], word, font))

    return canvas, instances, debug


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Per-cell packed Indic scene-text synth (varied density).")
    ap.add_argument("--bg-dir", required=True)
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--out-dir", default="./out_cells")
    ap.add_argument("--num-scenes", type=int, default=20)

    # grid + density (a matrix of cells with varied per-cell/per-row density)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--max-per-cell", type=int, default=20,
                    help="Max texts in a dense cell (count is random per cell).")
    ap.add_argument("--empty-cell-prob", type=float, default=0.22,
                    help="Chance an individual cell is empty.")
    ap.add_argument("--row-empty-prob", type=float, default=0.15,
                    help="Chance a whole row is left empty (text-free band).")
    ap.add_argument("--margin", type=int, default=2,
                    help="Min gap (px) enforced between texts in a cell.")

    # size: small, biased smaller, so many fit inside one cell (no giants)
    ap.add_argument("--min-size", type=int, default=9)
    ap.add_argument("--max-size", type=int, default=30)
    ap.add_argument("--size-bias", type=float, default=1.8,
                    help=">1 skews sizes smaller so more texts fit per cell.")
    ap.add_argument("--size-tries", type=int, default=6)
    ap.add_argument("--pos-tries", type=int, default=40)

    # geometry (subtle)
    ap.add_argument("--rotate-prob", type=float, default=0.12)
    ap.add_argument("--italic-prob", type=float, default=0.1)
    ap.add_argument("--perspective-prob", type=float, default=0.12)
    ap.add_argument("--perspective-frac", type=float, default=0.08)

    # appearance
    ap.add_argument("--bold-prob", type=float, default=0.2)
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
    return ap


def main():
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

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

    for i in range(args.num_scenes):
        bg = Image.open(rng.choice(bgs)).convert("RGB")
        if args.resize and max(bg.size) > args.resize:
            sc = args.resize / max(bg.size)
            bg = bg.resize((max(1, int(bg.width * sc)),
                            max(1, int(bg.height * sc))), Image.LANCZOS)

        scene, instances, debug = make_scene(bg, fonts_idx, vocab, rng, args)
        scene = G.augment_scene(scene, rng, args)

        name = f"scene_{i:05d}"
        scene.save(os.path.join(img_dir, name + ".jpg"),
                   quality=rng.randint(70, 95))
        with open(os.path.join(ann_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({"image_file": name + ".jpg",
                       "width": scene.width, "height": scene.height,
                       "instances": instances}, f, ensure_ascii=False, indent=2)
        G.draw_debug(scene, debug).save(os.path.join(dbg_dir, name + ".jpg"),
                                        quality=92)
        print(f"[gen] {i + 1}/{args.num_scenes}  {name}: "
              f"{len(instances)} text(s)")

    print(f"\n[done] {args.num_scenes} scenes -> {args.out_dir}")
    print(f"  images:      {img_dir}")
    print(f"  annotations: {ann_dir}")
    print(f"  debug:       {dbg_dir}")


if __name__ == "__main__":
    main()
