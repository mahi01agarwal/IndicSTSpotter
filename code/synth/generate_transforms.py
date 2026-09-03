#!/usr/bin/env python3
"""Transform-augmented Indic scene-text generator (for dataset diversity).

Adds heavier per-instance distortions on top of the usual rendering, so the
spotter sees text under realistic capture/print artifacts:

  - PERSPECTIVE  (projective warp; exact rotated quad kept)
  - ELASTIC      (smooth non-rigid warp, like wrinkled/curved print)
  - MOTION BLUR  (directional blur, like camera/subject motion)
  - DEFOCUS BLUR (disk blur, like out-of-focus text)

These mirror albumentations' Perspective / ElasticTransform / MotionBlur /
Defocus, but are implemented directly in OpenCV and applied PER TEXT PATCH so
each instance's ground-truth label stays exact (albumentations is not required).

Placement is scattered with STRICT no-overlap (IoU threshold ~0). Labels:
  - bbox  = tight axis box of the actual rendered pixels (always exact, covers
            blur/elastic spread);
  - quad  = exact projective corners for perspective/rotation; for elastic
            instances (non-rigid) the axis box is used as the quad.

Leaves the other generate*.py scripts unchanged; reuses generate.py helpers.

Run:
    python generate_transforms.py --bg-dir BG --out-dir out --num-scenes 50
"""

import os
import json
import argparse
import random

import numpy as np
import cv2
from PIL import Image, ImageFilter

import generate as G


# ---------------------------------------------------------------------------
# Pixel transforms (operate on an 'L' alpha as a uint8 ndarray)
# ---------------------------------------------------------------------------

def t_elastic(a, rng, strength, sigma):
    h, w = a.shape
    rs = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
    dx = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * strength
    dy = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * strength
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(a, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def t_motion(a, rng, k):
    k = max(3, k | 1)  # odd kernel
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    s = kern.sum()
    return cv2.filter2D(a, -1, kern / (s if s > 0 else 1))


def t_defocus(a, r):
    r = max(1, r)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
    return cv2.filter2D(a, -1, disk / disk.sum())


def t_curve(a, rng, args):
    """Bend the (already-shaped) text along a smooth arch or wave.

    Shaping happened on flat text, so warping the finished mask keeps every
    script correct (Devanagari conjuncts, Urdu joining, ...). Column x is shifted
    vertically by dy(x): a half-sine ARCH (curving up or down) or a gentle WAVE.
    """
    h, w = a.shape
    amp = rng.uniform(*args.curve_amp) * h
    pad = int(np.ceil(amp)) + 2                      # room so the bend isn't clipped
    a = cv2.copyMakeBorder(a, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    h2, w2 = a.shape
    xs = np.arange(w2, dtype=np.float32)
    if rng.random() < args.curve_wave_prob:          # wave: 0.75-1.75 cycles
        dy = amp * np.sin(2 * np.pi * rng.uniform(0.75, 1.75) * xs / max(1, w2 - 1)
                          + rng.uniform(0, 2 * np.pi))
    else:                                            # arch: half sine, up or down
        dy = rng.choice([-1.0, 1.0]) * amp * np.sin(np.pi * xs / max(1, w2 - 1))
    gx, gy = np.meshgrid(xs, np.arange(h2, dtype=np.float32))
    return cv2.remap(a, gx, (gy - dy[None, :]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def curve_polygon(alpha, n=14, thresh=16):
    """Trace the inked text's top+bottom edges as a polygon (patch coords).

    Samples `n` columns across the ink; for each takes the highest and lowest lit
    pixel, so the outline follows the actual curved glyphs. Returns a list of
    (x, y) points going top-left..top-right then bottom-right..bottom-left, or None.
    """
    lit_cols = np.where((alpha > thresh).any(axis=0))[0]
    if lit_cols.size < 2:
        return None
    top, bot = [], []
    for x in np.linspace(lit_cols[0], lit_cols[-1], n).round().astype(int):
        rows = np.where(alpha[:, x] > thresh)[0]
        if rows.size == 0:
            continue
        top.append((int(x), int(rows[0])))
        bot.append((int(x), int(rows[-1])))
    if len(top) < 2:
        return None
    return top + bot[::-1]


# ---------------------------------------------------------------------------
# One transformed text patch (geometry + pixel effects), labels tracked
# ---------------------------------------------------------------------------

PAD = 22  # transparent border so blur/elastic spread isn't clipped


def build_instance(text, font, size, rng, args):
    """Render + warp + distort one text. Returns a dict or None."""
    mask = G.render_mask(text, font, size, bold=rng.random() < args.bold_prob)
    if mask is None:
        return None

    angle = (rng.choice([90, -90, 45, -45, 30, -30, 15, -15])
             if rng.random() < args.rotate_prob else 0)
    dst, pw, ph = G.build_geometry(mask.width, mask.height, angle, rng, args)
    dst = dst + PAD
    alpha = G.warp_alpha(mask, dst, pw + 2 * PAD, ph + 2 * PAD)

    a = np.asarray(alpha).copy()
    text_h = max(1, ph)                       # glyph height in px, to scale effects to size

    # curve is a shape bend (stays readable; rec crops skip curved anyway) — kept separate
    curved = rng.random() < args.curve_prob
    if curved:
        a = t_curve(a, rng, args)

    # AT MOST ONE degradation per instance (elastic | motion | defocus), each scaled to
    # text height so small text is never wiped. Stacking these was what produced
    # illegible "blob" words. Total P(any) = elastic+motion+defocus probs.
    elastic = False
    r = rng.random()
    p_el, p_mo = args.elastic_prob, args.motion_prob
    p_de = args.defocus_prob
    if r < p_el:
        elastic = True
        strength = rng.uniform(*args.elastic_strength) * min(1.0, text_h / 40.0)
        a = t_elastic(a, rng, strength=strength, sigma=rng.uniform(4, 7))
    elif r < p_el + p_mo:
        kmax = min(args.motion_kmax, max(args.motion_kmin, int(text_h * 0.22)))  # <=22% of height
        a = t_motion(a, rng, k=rng.randint(args.motion_kmin, kmax))
    elif r < p_el + p_mo + p_de:
        rmax = min(args.defocus_max, max(1, int(text_h * 0.06)))                  # <=6% of height
        a = t_defocus(a, rng.randint(1, rmax))
    alpha = Image.fromarray(a, "L")

    ink = alpha.getbbox()
    if ink is None:
        return None
    ix0, iy0, ix1, iy1 = ink
    iw, ih = ix1 - ix0, iy1 - iy0

    # legibility guard — reject renders that stopped looking like text, so try_place
    # resamples a clean one (it retries up to size_tries). Catches:
    #   low peak opacity      -> blur/defocus smears & washed-out fades
    #   tiny ink box          -> collapsed / over-bent slivers
    #   negligible coverage   -> near-empty boxes
    crop = a[iy0:iy1, ix0:ix1]
    if (iw < 3 or ih < 6
            or int(crop.max()) < 120
            or float((crop > 60).mean()) < 0.04):
        return None

    return {
        "alpha": alpha,
        "ink": (ix0, iy0, ix1 - ix0, iy1 - iy0),
        "geom_quad": [(float(x), float(y)) for x, y in dst],
        "elastic": elastic,
        "curved": curved,
        # polygon tracing the arc, in patch coords (None when not curved)
        "curve_poly": curve_polygon(np.asarray(alpha)) if curved else None,
    }


def composite(canvas, alpha, ox, oy, color, rng, args):
    if rng.random() < args.shadow_prob:
        sh = alpha.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.45))
        G.paste_mask(canvas, (0, 0, 0), sh, ox + rng.randint(1, 3), oy + rng.randint(1, 3))
    if rng.random() < args.stroke_prob:
        dil = alpha.filter(ImageFilter.MaxFilter(3))
        sc = (15, 15, 15) if sum(color) > 384 else (240, 240, 240)
        G.paste_mask(canvas, sc, dil, ox, oy)
    opacity = rng.uniform(0.78, 0.92) if rng.random() < args.faded_prob else 1.0
    G.paste_mask(canvas, color, alpha, ox, oy, opacity)


def sample_size(rng, lo, hi, bias):
    return int(round(lo + (hi - lo) * (rng.random() ** bias)))


def make_scene(bg, fonts_idx, vocab, rng, args):
    W, H = bg.size
    canvas = bg.convert("RGB")
    bg_arr = np.asarray(canvas)
    avail = [s for s in G.SCRIPTS if fonts_idx.get(s) and vocab.get(s)]

    placed, instances, debug = [], [], []
    target = rng.randint(args.min_instances, args.max_instances)
    attempts = 0
    while len(instances) < target and attempts < target * 5:
        attempts += 1
        script = rng.choice(avail)
        font = rng.choice(fonts_idx[script])
        text = G.make_text(script, vocab[script], rng, args)
        size = sample_size(rng, args.min_size, args.max_size, args.size_bias)

        inst = build_instance(text, font, size, rng, args)
        if inst is None:
            continue
        ix0, iy0, iw, ih = inst["ink"]
        if iw >= W or ih >= H or iw < 2 or ih < 2:
            continue

        box = None
        for _try in range(args.place_tries):
            px = rng.randint(0, W - iw)
            py = rng.randint(0, H - ih)
            cand = (px, py, px + iw, py + ih)
            if all(G.iou(cand, b) <= args.iou_thresh for b in placed):
                box = cand
                break
        if box is None:
            continue
        placed.append(box)
        px, py = box[0], box[1]
        ox, oy = px - ix0, py - iy0   # patch origin so ink lands at (px,py)

        color = G.pick_color(bg_arr, box, rng)
        composite(canvas, inst["alpha"], ox, oy, color, rng, args)

        if inst["elastic"] or inst["curved"]:
            quad = [(px, py), (px + iw, py), (px + iw, py + ih), (px, py + ih)]
        else:
            quad = [(qx + ox, qy + oy) for qx, qy in inst["geom_quad"]]

        rec = {
            "text": text, "script": script, "font": G.family_of(font),
            "font_size": size, "rotation_deg": 0, "bbox": [px, py, iw, ih],
            "quad": [int(round(c)) for q in quad for c in q],
        }
        dbg_pts = [(int(q[0]), int(q[1])) for q in quad]
        if inst["curved"] and inst["curve_poly"]:
            poly = [(int(round(x + ox)), int(round(y + oy))) for x, y in inst["curve_poly"]]
            rec["polygon"] = [v for pt in poly for v in pt]
            dbg_pts = poly
        instances.append(rec)
        debug.append((dbg_pts, text, font))

    return canvas, instances, debug


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Transform-augmented Indic scene-text synth (no overlap).")
    ap.add_argument("--bg-dir", required=True)
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--out-dir", default="./out_transforms")
    ap.add_argument("--num-scenes", type=int, default=20)

    # density (scattered, strict no overlap)
    ap.add_argument("--min-instances", type=int, default=25)
    ap.add_argument("--max-instances", type=int, default=60)
    ap.add_argument("--iou-thresh", type=float, default=0.02)
    ap.add_argument("--place-tries", type=int, default=60)

    # size
    ap.add_argument("--min-size", type=int, default=16)
    ap.add_argument("--max-size", type=int, default=56)
    ap.add_argument("--size-bias", type=float, default=1.6)

    # geometry (perspective is part of build_geometry)
    ap.add_argument("--rotate-prob", type=float, default=0.2)
    ap.add_argument("--italic-prob", type=float, default=0.15)
    ap.add_argument("--perspective-prob", type=float, default=0.35)
    ap.add_argument("--perspective-frac", type=float, default=0.14)

    # the requested pixel transforms
    ap.add_argument("--elastic-prob", type=float, default=0.25)
    ap.add_argument("--elastic-strength", type=float, nargs=2, default=(3.0, 7.0))
    ap.add_argument("--motion-prob", type=float, default=0.25)
    ap.add_argument("--motion-kmin", type=int, default=5)
    ap.add_argument("--motion-kmax", type=int, default=15)
    ap.add_argument("--defocus-prob", type=float, default=0.2)
    ap.add_argument("--defocus-max", type=int, default=4)
    # curved text (bend the shaped mask along an arch/wave)
    ap.add_argument("--curve-prob", type=float, default=0.2)
    ap.add_argument("--curve-amp", type=float, nargs=2, default=(0.15, 0.45),
                    help="Bend amplitude as a fraction of text height (min max).")
    ap.add_argument("--curve-wave-prob", type=float, default=0.35,
                    help="Of curved texts, fraction that wave vs. a single arch.")

    # appearance + content
    ap.add_argument("--bold-prob", type=float, default=0.2)
    ap.add_argument("--stroke-prob", type=float, default=0.25)
    ap.add_argument("--shadow-prob", type=float, default=0.2)
    ap.add_argument("--faded-prob", type=float, default=0.1)
    ap.add_argument("--multiword-prob", type=float, default=0.3)
    ap.add_argument("--number-prob", type=float, default=0.2)

    # whole-image augmentation (light)
    ap.add_argument("--aug-prob", type=float, default=0.4)
    ap.add_argument("--hue-prob", type=float, default=0.2)
    ap.add_argument("--gray-prob", type=float, default=0.08)
    ap.add_argument("--img-blur-prob", type=float, default=0.12)
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
        scene.save(os.path.join(img_dir, name + ".jpg"), quality=rng.randint(70, 95))
        with open(os.path.join(ann_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"image_file": name + ".jpg", "width": scene.width,
                       "height": scene.height, "instances": instances},
                      f, ensure_ascii=False, indent=2)
        G.draw_debug(scene, debug).save(os.path.join(dbg_dir, name + ".jpg"),
                                        quality=92)
        print(f"[gen] {i + 1}/{args.num_scenes}  {name}: {len(instances)} text(s)")

    print(f"\n[done] {args.num_scenes} scenes -> {args.out_dir}")
    print(f"  images:      {img_dir}")
    print(f"  annotations: {ann_dir}")
    print(f"  debug:       {dbg_dir}")


if __name__ == "__main__":
    main()
