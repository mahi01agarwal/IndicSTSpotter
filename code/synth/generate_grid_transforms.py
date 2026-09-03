#!/usr/bin/env python3
"""Grid-cell layout + a fraction of transform-augmented text (combined).

Combines two earlier scripts so each image has BOTH:
  - dense, per-cell packed NORMAL text (the grid_cells look: a matrix of cells
    with varied density, small sizes, some empty cells/rows), and
  - a fraction of the same texts carrying transforms (perspective / elastic /
    motion blur / defocus blur).

Each text independently may or may not be transformed -- so most are crisp and
normal, some are distorted, all inside their cell and non-overlapping. Labels
stay exact (bbox = true pixel extent; quad = projective corners, or axis box for
elastic). Reuses generate.py + generate_transforms.py; leaves both unchanged.

Run:
    python generate_grid_transforms.py --bg-dir BG --out-dir out --num-scenes 50
"""

import os
import json
import argparse
import random

import numpy as np
from PIL import Image

import generate as G
import generate_transforms as T


def _overlap(a, b, margin):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def reading_order(instances, line_frac=0.6):
    """Join instance texts into a page transcription: rows top->bottom, words
    left->right within a row. Rows are formed by clustering on the y-centre
    (tolerance = line_frac x the word height). Urdu words are already correctly
    shaped (RTL) internally; inter-word order here is spatial left->right."""
    items = []
    for it in instances:
        x, y, w, h = it["bbox"]
        items.append((y + h / 2.0, x, h, it["text"]))
    items.sort(key=lambda t: t[0])                     # by vertical centre
    lines = []                                         # each: [running_cy, n, [(x, text)]]
    for cy, x, h, txt in items:
        for ln in lines:
            if abs(cy - ln[0]) <= line_frac * h:
                ln[2].append((x, txt))
                ln[0] = (ln[0] * ln[1] + cy) / (ln[1] + 1)
                ln[1] += 1
                break
        else:
            lines.append([cy, 1, [(x, txt)]])
    lines.sort(key=lambda ln: ln[0])
    return "\n".join(" ".join(t for _, t in sorted(ln[2])) for ln in lines)


def try_place(cell, cell_boxes, fonts_idx, vocab, avail, weights, rng, args, size_cap):
    """Render one (possibly transformed) text and fit it inside the cell."""
    cx0, cy0, cx1, cy1 = cell
    cw, ch = cx1 - cx0, cy1 - cy0
    for _s in range(args.size_tries):
        script = rng.choices(avail, weights=weights, k=1)[0]
        font = rng.choice(fonts_idx[script])
        word = G.make_text(script, vocab[script], rng, args)
        size = int(round(args.min_size +
                         (size_cap - args.min_size) * (rng.random() ** args.size_bias)))
        inst = T.build_instance(word, font, size, rng, args)
        if inst is None:
            continue
        ix0, iy0, iw, ih = inst["ink"]
        if iw > cw or ih > ch:
            continue
        for _p in range(args.pos_tries):
            px = rng.randint(cx0, cx1 - iw)
            py = rng.randint(cy0, cy1 - ih)
            box = (px, py, px + iw, py + ih)
            if all(not _overlap(box, b, args.margin) for b in cell_boxes):
                return box, inst, word, script, font, size, (px - ix0, py - iy0)
    return None


def make_scene(bg, fonts_idx, vocab, rng, args):
    W, H = bg.size
    canvas = bg.convert("RGB")
    bg_arr = np.asarray(canvas)
    avail = [s for s in G.SCRIPTS if fonts_idx.get(s) and vocab.get(s)]
    if getattr(args, "only_script", None):
        avail = [s for s in avail if s == args.only_script]
    weights = [max(0.0, (args.script_weights or {}).get(s, 1.0)) for s in avail]
    if not any(weights):                       # all-zero -> fall back to uniform
        weights = [1.0] * len(avail)

    instances, debug = [], []

    for r in range(args.rows):
        if rng.random() < args.row_empty_prob:
            continue
        for c in range(args.cols):
            cx0, cx1 = int(c * W / args.cols), int((c + 1) * W / args.cols)
            cy0, cy1 = int(r * H / args.rows), int((r + 1) * H / args.rows)
            cw, ch = cx1 - cx0, cy1 - cy0
            if cw < 10 or ch < 10:
                continue
            if rng.random() < args.empty_cell_prob:
                continue

            count = rng.randint(1, args.max_per_cell)
            size_cap = max(args.min_size + 1,
                           min(args.max_size, int(min(cw, ch) * 0.8)))
            cell_boxes = []

            for _ in range(count):
                res = try_place((cx0, cy0, cx1, cy1), cell_boxes, fonts_idx,
                                vocab, avail, weights, rng, args, size_cap)
                if res is None:
                    continue
                box, inst, word, script, font, size, origin = res
                cell_boxes.append(box)
                px, py = box[0], box[1]
                iw, ih = box[2] - box[0], box[3] - box[1]

                color = G.pick_color(bg_arr, box, rng)
                T.composite(canvas, inst["alpha"], origin[0], origin[1],
                            color, rng, args)

                if inst["elastic"] or inst["curved"]:
                    quad = [(px, py), (px + iw, py),
                            (px + iw, py + ih), (px, py + ih)]
                else:
                    quad = [(qx + origin[0], qy + origin[1])
                            for qx, qy in inst["geom_quad"]]

                rec = {
                    "text": word, "script": script, "font": G.family_of(font),
                    "font_size": size, "rotation_deg": 0,
                    "bbox": [px, py, iw, ih],
                    "quad": [int(round(v)) for q in quad for v in q],
                }
                dbg_pts = [(int(q[0]), int(q[1])) for q in quad]
                if inst["curved"] and inst["curve_poly"]:
                    poly = [(int(round(x + origin[0])), int(round(y + origin[1])))
                            for x, y in inst["curve_poly"]]
                    rec["polygon"] = [v for pt in poly for v in pt]
                    dbg_pts = poly
                instances.append(rec)
                debug.append((dbg_pts, word, font))

    return canvas, instances, debug


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Grid-cell normal text + fraction transformed (combined).")
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--out-dir", default="./out_grid_tf")
    ap.add_argument("--num-scenes", type=int, default=20)

    # grid + density
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--max-per-cell", type=int, default=20)
    ap.add_argument("--empty-cell-prob", type=float, default=0.22)
    ap.add_argument("--row-empty-prob", type=float, default=0.15)
    ap.add_argument("--margin", type=int, default=2)

    # size (small, biased so many fit)
    ap.add_argument("--min-size", type=int, default=9)
    ap.add_argument("--max-size", type=int, default=30)
    ap.add_argument("--size-bias", type=float, default=1.8)
    ap.add_argument("--size-tries", type=int, default=6)
    ap.add_argument("--pos-tries", type=int, default=40)

    # geometry
    ap.add_argument("--rotate-prob", type=float, default=0.12)
    ap.add_argument("--italic-prob", type=float, default=0.1)
    ap.add_argument("--perspective-prob", type=float, default=0.22)
    ap.add_argument("--perspective-frac", type=float, default=0.1)

    # pixel transforms (kept modest -- text here is small, keep it readable)
    ap.add_argument("--elastic-prob", type=float, default=0.15)
    ap.add_argument("--elastic-strength", type=float, nargs=2, default=(2.0, 4.0))
    ap.add_argument("--motion-prob", type=float, default=0.15)
    ap.add_argument("--motion-kmin", type=int, default=3)
    ap.add_argument("--motion-kmax", type=int, default=9)
    ap.add_argument("--defocus-prob", type=float, default=0.12)
    ap.add_argument("--defocus-max", type=int, default=2)
    # curved text (bend the shaped mask along an arch/wave)
    ap.add_argument("--curve-prob", type=float, default=0.18)
    ap.add_argument("--curve-amp", type=float, nargs=2, default=(0.12, 0.35),
                    help="Bend amplitude as a fraction of text height (min max).")
    ap.add_argument("--curve-wave-prob", type=float, default=0.3,
                    help="Of curved texts, fraction that wave vs. a single arch.")

    # appearance + content
    ap.add_argument("--bold-prob", type=float, default=0.2)
    ap.add_argument("--stroke-prob", type=float, default=0.25)
    ap.add_argument("--shadow-prob", type=float, default=0.2)
    ap.add_argument("--faded-prob", type=float, default=0.1)
    ap.add_argument("--multiword-prob", type=float, default=0.15)
    ap.add_argument("--number-prob", type=float, default=0.2)

    # whole-image augmentation (light)
    ap.add_argument("--aug-prob", type=float, default=0.4)
    ap.add_argument("--hue-prob", type=float, default=0.2)
    ap.add_argument("--gray-prob", type=float, default=0.08)
    ap.add_argument("--img-blur-prob", type=float, default=0.12)
    ap.add_argument("--noise-prob", type=float, default=0.2)

    ap.add_argument("--resize", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    # canvas: fixed dimension, two orientations (some landscape, some portrait).
    # When --canvas-w and --canvas-h are set they REPLACE --resize: each bg is
    # cover-fit to exactly (885x497) or its flip (497x885), chosen by --portrait-prob.
    ap.add_argument("--canvas-w", type=int, default=0)
    ap.add_argument("--canvas-h", type=int, default=0)
    ap.add_argument("--portrait-prob", type=float, default=0.5,
                    help="Fraction of images in portrait (flipped) orientation.")

    # sharded parallel generation: run several processes, each a disjoint slice.
    # Per-scene seeding (seed, global-index) makes every scene reproducible and
    # independent of which shard produced it.
    ap.add_argument("--start-index", type=int, default=0,
                    help="Global index of the first scene this process writes.")

    # dataset run: config file + per-script sampling weights + OCR manifest
    ap.add_argument("--config", default=None,
                    help="JSON file whose keys set any option below "
                         "(underscore keys; CLI flags still override).")
    ap.add_argument("--script-weights", type=json.loads, default=None,
                    help='JSON like {"latin":1.5,"telugu":1.0,...}; scripts not '
                         "listed default to weight 1.0. Omit for uniform.")
    ap.add_argument("--emit-transcription", action="store_true",
                    help="Also write a reading-order transcription per image and "
                         "a dataset-level manifest.jsonl (for OCR fine-tuning).")
    ap.add_argument("--only-script", default=None,
                    help="Restrict every word to this one script (for single-script "
                         "eval sets); tags each manifest line with it.")
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
    if args.only_script and not (fonts_idx.get(args.only_script) and vocab.get(args.only_script)):
        raise SystemExit(f"--only-script {args.only_script}: no fonts+vocab for that script.")

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

    # save the exact settings used, for reproducibility (shard 0 only, so parallel
    # workers don't clobber the same file mid-write)
    if args.start_index == 0:
        with open(os.path.join(args.out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in vars(args).items() if k != "config"},
                      f, ensure_ascii=False, indent=2, default=str)
    # each shard writes its own manifest part; merge with:
    #   cat OUT/manifest.part_*.jsonl > OUT/manifest.jsonl
    man_name = ("manifest.jsonl" if args.start_index == 0
                else f"manifest.part_{args.start_index:08d}.jsonl")
    manifest = (open(os.path.join(args.out_dir, man_name), "w", encoding="utf-8")
                if args.emit_transcription else None)

    end = args.start_index + args.num_scenes
    for i in range(args.start_index, end):
        # per-scene seed: process-independent, so any shard reproduces any index
        seed_i = (args.seed * 2654435761 + i) & 0xFFFFFFFF
        rng = random.Random(seed_i)
        np.random.seed(seed_i)

        bg = Image.open(rng.choice(bgs)).convert("RGB")
        if args.canvas_w and args.canvas_h:         # fixed dimension, landscape or portrait
            wc, hc = G.sample_canvas(rng, args.canvas_w, args.canvas_h, args.portrait_prob)
            bg = G.fit_cover(bg, wc, hc)
        elif args.resize and max(bg.size) > args.resize:
            sc = args.resize / max(bg.size)
            bg = bg.resize((max(1, int(bg.width * sc)),
                            max(1, int(bg.height * sc))), Image.LANCZOS)

        scene, instances, debug = make_scene(bg, fonts_idx, vocab, rng, args)
        scene = G.augment_scene(scene, rng, args)

        name = f"scene_{i:06d}"
        scene.save(os.path.join(img_dir, name + ".jpg"), quality=rng.randint(70, 95))
        ann = {"image_file": name + ".jpg", "width": scene.width,
               "height": scene.height, "instances": instances}
        if args.emit_transcription:
            text = reading_order(instances)
            ann["transcription"] = text
            m = {"image": f"images/{name}.jpg", "text": text}
            if args.only_script:
                m["script"] = args.only_script
            manifest.write(json.dumps(m, ensure_ascii=False) + "\n")
            manifest.flush()
        with open(os.path.join(ann_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        G.draw_debug(scene, debug).save(os.path.join(dbg_dir, name + ".jpg"),
                                        quality=92)
        print(f"[gen] {i - args.start_index + 1}/{args.num_scenes}  {name}: "
              f"{len(instances)} text(s)")

    if manifest:
        manifest.close()

    print(f"\n[done] {args.num_scenes} scenes -> {args.out_dir}")
    print(f"  images:      {img_dir}")
    print(f"  annotations: {ann_dir}")
    print(f"  debug:       {dbg_dir}")
    if args.emit_transcription:
        print(f"  manifest:    {os.path.join(args.out_dir, 'manifest.jsonl')}  "
              f"(image -> transcription)")


if __name__ == "__main__":
    main()
