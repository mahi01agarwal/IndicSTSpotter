#!/usr/bin/env python3
"""Synthetic Indic scene-text dataset generator for text spotting.

Goal: train a detector+recognizer that finds and reads text ANYWHERE in a
natural scene, at any size/color/font/script/position/orientation. Text is
placed all over the image (no surface/depth awareness -- intentional), with a
random grid layout that varies density across the frame and across the dataset,
so the spotter can't learn shortcuts ("text is always top-right / always large").

This version maximizes visual diversity:
  - geometry per instance: rotation + faux-italic shear + mild perspective,
    all folded into one homography so the recorded quad is exact;
  - appearance per instance: random font/size, contrast-aware random color,
    faux-bold, outline/stroke, drop shadow, partial transparency, defocus blur;
  - content: single words, multi-word phrases, case variation, numeric tokens;
  - whole-image: brightness/contrast/saturation/hue jitter, occasional
    grayscale, gaussian noise, light blur, randomized JPEG quality.

Shaping is Pillow + Raqm (HarfBuzz). Fonts are discovered via `fc-list`.

Run:
    python generate.py --bg-dir BG --out-dir out --num-scenes 50
"""

import os
import json
import glob
import math
import random
import argparse
import subprocess

import numpy as np
import cv2
from PIL import (Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance,
                 features)

if not features.check("raqm"):
    raise SystemExit(
        "Pillow has no Raqm support (features.check('raqm') is False) -- Indic "
        "shaping will be WRONG. Rebuild Pillow from source against libraqm:\n"
        "  conda install -c conda-forge libraqm -y\n"
        "  pip install --no-binary :all: --no-cache-dir pillow")

RAQM = ImageFont.Layout.RAQM

# script key -> fontconfig language code (for fc-list) + vocab filename.
# Several corpus languages share a script (hindi+marathi -> devanagari;
# bengali+assamese+manipuri -> bengali); build_vocab.py folds them together.
SCRIPTS = {
    "telugu":     {"lang": "te", "vocab": "telugu.txt"},
    "devanagari": {"lang": "hi", "vocab": "devanagari.txt"},
    "tamil":      {"lang": "ta", "vocab": "tamil.txt"},
    "kannada":    {"lang": "kn", "vocab": "kannada.txt"},
    "malayalam":  {"lang": "ml", "vocab": "malayalam.txt"},
    "bengali":    {"lang": "bn", "vocab": "bengali.txt"},
    "gujarati":   {"lang": "gu", "vocab": "gujarati.txt"},
    "gurmukhi":   {"lang": "pa", "vocab": "gurmukhi.txt"},
    "oriya":      {"lang": "or", "vocab": "oriya.txt"},
    "urdu":       {"lang": "ur", "vocab": "urdu.txt"},
    "latin":      {"lang": "en", "vocab": "latin.txt"},
}

ROTATIONS = [0, 90, -90, 45, -45, 30, -30]


# ---------------------------------------------------------------------------
# Font discovery (fontconfig) + vocab
# ---------------------------------------------------------------------------

def discover_fonts():
    """script -> list of font paths, via `fc-list :lang=<code>`. Nothing hardcoded."""
    idx = {}
    for s, meta in SCRIPTS.items():
        paths = []
        try:
            out = subprocess.run(["fc-list", f":lang={meta['lang']}", "file"],
                                 capture_output=True, text=True, check=True).stdout
        except Exception as e:
            print(f"[fonts] fc-list failed for {s} ({meta['lang']}): {e}")
            out = ""
        for line in out.splitlines():
            p = line.split(":")[0].strip()
            if p and os.path.isfile(p) and p.lower().endswith(
                    (".ttf", ".otf", ".ttc", ".otc")):
                paths.append(p)
        idx[s] = sorted(set(paths))
        print(f"[fonts] {s:11s} ({meta['lang']}): {len(idx[s])} font(s)")
    return idx


def load_vocab(vocab_dir):
    vocab = {}
    for s, meta in SCRIPTS.items():
        path = os.path.join(vocab_dir, meta["vocab"])
        words = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#"):
                        words.append(w)
        else:
            print(f"[vocab] MISSING {path}")
        vocab[s] = words
        print(f"[vocab] {s:11s}: {len(words)} word(s)")
    return vocab


_family_cache = {}


def family_of(font_path):
    if font_path not in _family_cache:
        try:
            _family_cache[font_path] = ImageFont.truetype(font_path, 12).getname()[0]
        except Exception:
            _family_cache[font_path] = os.path.basename(font_path)
    return _family_cache[font_path]


# ---------------------------------------------------------------------------
# Content: words, phrases, case variation, numbers
# ---------------------------------------------------------------------------

def random_number(rng):
    fmts = [
        lambda: str(rng.randint(0, 9999)),
        lambda: f"{rng.randint(1, 99)}-{rng.randint(1, 999)}",
        lambda: f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}",
        lambda: f"{rng.randint(1, 500)}{rng.choice(['A', 'B', 'C', ''])}",
        lambda: "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 6))),
    ]
    return rng.choice(fmts)()


def make_text(script, words, rng, args):
    """Build a text string: word / multi-word phrase / number, with case variety."""
    if script == "latin" and rng.random() < args.number_prob:
        return random_number(rng)

    r = rng.random()
    n = 3 if r < args.multiword_prob * 0.25 else (2 if r < args.multiword_prob else 1)
    text = " ".join(rng.choice(words) for _ in range(n))

    if script == "latin":
        text = rng.choice([str.lower, str.upper, str.title, lambda x: x])(text)
    return text


# ---------------------------------------------------------------------------
# Rendering + unified geometry (rotation + shear + perspective -> homography)
# ---------------------------------------------------------------------------

def render_mask(text, font_path, size, bold, ss=3):
    """Shape `text` -> tight 'L' glyph mask (or None). Supersampled for clean AA."""
    font = ImageFont.truetype(font_path, size * ss, layout_engine=RAQM)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    w, h, pad = max(1, r - l), max(1, b - t), 6 * ss
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=255)
    if bold:
        img = img.filter(ImageFilter.MaxFilter(3))
    bb = img.getbbox()
    if bb is None:
        return None
    img = img.crop(bb)
    if ss > 1:
        img = img.resize((max(1, img.width // ss), max(1, img.height // ss)),
                         Image.LANCZOS)
    return img


def build_geometry(w0, h0, angle, rng, args):
    """Return (dst_corners 4x2 in patch coords, patch_w, patch_h).

    Composes (in order) faux-italic shear -> rotation -> mild perspective jitter
    on the tight box corners [TL,TR,BR,BL]. Because the same corners feed the
    warp homography AND the recorded quad, the label is exact for all of them.
    """
    pts = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float64)
    p = pts - [w0 / 2.0, h0 / 2.0]

    if rng.random() < args.italic_prob:                      # horizontal shear
        p[:, 0] += rng.uniform(-0.35, 0.35) * p[:, 1]

    a = math.radians(angle)                                   # rotation (y-down CCW)
    cos, sin = math.cos(a), math.sin(a)
    p = p @ np.array([[cos, -sin], [sin, cos]])

    if rng.random() < args.perspective_prob:                  # projective jitter
        jit = args.perspective_frac * max(w0, h0)
        p = p + np.array([[rng.uniform(-jit, jit), rng.uniform(-jit, jit)]
                          for _ in range(4)])

    p = p - p.min(axis=0)
    patch_w = int(math.ceil(p[:, 0].max())) + 1
    patch_h = int(math.ceil(p[:, 1].max())) + 1
    return p, patch_w, patch_h


def warp_alpha(mask, dst_pts, patch_w, patch_h):
    src = np.array([[0, 0], [mask.width, 0],
                    [mask.width, mask.height], [0, mask.height]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst_pts.astype(np.float32))
    a = cv2.warpPerspective(np.asarray(mask), H, (patch_w, patch_h),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    return Image.fromarray(a, "L")


# ---------------------------------------------------------------------------
# Color + compositing helpers
# ---------------------------------------------------------------------------

def pick_color(bg_arr, box, rng):
    """Random foreground color contrasting the background under `box`."""
    H, W = bg_arr.shape[:2]
    x0 = max(0, min(W - 1, box[0]))
    y0 = max(0, min(H - 1, box[1]))
    x1 = max(x0 + 1, min(W, box[2]))
    y1 = max(y0 + 1, min(H, box[3]))
    region = bg_arr[y0:y1, x0:x1].astype(np.float32)
    lum = (0.299 * region[..., 0] + 0.587 * region[..., 1] +
           0.114 * region[..., 2]).mean()
    if lum < 128:
        return tuple(rng.randint(205, 255) for _ in range(3))
    return tuple(rng.randint(0, 55) for _ in range(3))


def paste_mask(canvas, color, mask_L, x, y, opacity=1.0):
    """Composite a solid `color` through `mask_L` onto an RGB canvas (clips)."""
    if opacity < 1.0:
        mask_L = mask_L.point(lambda v: int(v * opacity))
    layer = Image.new("RGB", mask_L.size, tuple(int(c) for c in color))
    canvas.paste(layer, (int(x), int(y)), mask_L)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def make_scene(bg, fonts_idx, vocab, rng, args):
    W, H = bg.size
    canvas = bg.convert("RGB")
    bg_arr = np.asarray(canvas)
    avail = [s for s in SCRIPTS if fonts_idx.get(s) and vocab.get(s)]

    placed_boxes, instances, debug = [], [], []
    cs = args.cell_size

    for cy in range(0, H, cs):
        for cx in range(0, W, cs):
            if rng.random() < args.empty_cell_prob:
                continue
            cell_x1, cell_y1 = min(cx + cs, W), min(cy + cs, H)

            for _ in range(rng.randint(1, args.max_per_cell)):
                script = rng.choice(avail)
                text = make_text(script, vocab[script], rng, args)
                font_path = rng.choice(fonts_idx[script])
                # bias sizes toward the small end (squared) so most text looks
                # "distant"/small with only occasional large signage.
                size = int(args.min_size +
                           (args.max_size - args.min_size) * (rng.random() ** 2))
                bold = rng.random() < args.bold_prob
                angle = (rng.choice(ROTATIONS[1:])
                         if rng.random() < args.rotate_prob else 0)

                mask = render_mask(text, font_path, size, bold)
                if mask is None:
                    continue
                dst_pts, pw, ph = build_geometry(mask.width, mask.height,
                                                 angle, rng, args)
                if pw >= W or ph >= H or pw < 3 or ph < 3:
                    continue

                lo_x, hi_x = cx, min(cell_x1 - 1, W - pw)
                lo_y, hi_y = cy, min(cell_y1 - 1, H - ph)
                if lo_x > hi_x or lo_y > hi_y:
                    continue

                box = None
                for _try in range(args.iou_tries):
                    px, py = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
                    cand = (px, py, px + pw, py + ph)
                    if all(iou(cand, b) <= args.iou_thresh for b in placed_boxes):
                        box = cand
                        break
                if box is None:
                    continue
                placed_boxes.append(box)
                px, py = box[0], box[1]

                alpha = warp_alpha(mask, dst_pts, pw, ph)
                color = pick_color(bg_arr, box, rng)

                # drop shadow (under everything else)
                if rng.random() < args.shadow_prob:
                    sh = alpha.filter(ImageFilter.GaussianBlur(2)).point(
                        lambda v: int(v * 0.45))
                    paste_mask(canvas, (0, 0, 0), sh,
                               px + rng.randint(2, 6), py + rng.randint(2, 6))

                # outline / stroke (opposite brightness)
                if rng.random() < args.stroke_prob:
                    sw = rng.randint(1, 3)
                    dil = alpha.filter(ImageFilter.MaxFilter(2 * sw + 1))
                    sc = (15, 15, 15) if sum(color) > 384 else (240, 240, 240)
                    paste_mask(canvas, sc, dil, px, py)

                # the text itself: optional defocus + partial transparency
                a2 = alpha
                if rng.random() < args.inst_blur_prob:
                    a2 = alpha.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
                opacity = (rng.uniform(0.55, 0.85)
                           if rng.random() < args.faded_prob else 1.0)
                paste_mask(canvas, color, a2, px, py, opacity)

                quad = [(px + x, py + y) for x, y in dst_pts]
                xs = [q[0] for q in quad]
                ys = [q[1] for q in quad]
                bbox = [int(round(min(xs))), int(round(min(ys))),
                        int(round(max(xs) - min(xs))),
                        int(round(max(ys) - min(ys)))]
                instances.append({
                    "text": text, "script": script,
                    "font": family_of(font_path), "font_size": size,
                    "rotation_deg": angle, "bbox": bbox,
                    "quad": [int(round(c)) for q in quad for c in q],
                })
                debug.append(([(int(q[0]), int(q[1])) for q in quad],
                              text, font_path))

    return canvas, instances, debug


# ---------------------------------------------------------------------------
# Whole-image augmentation (does not change labels)
# ---------------------------------------------------------------------------

def hue_shift(img, delta):
    hsv = np.asarray(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(delta * 255)) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def augment_scene(img, rng, args):
    if rng.random() < args.aug_prob:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < args.aug_prob:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.3))
    if rng.random() < args.aug_prob:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < args.hue_prob:
        img = hue_shift(img, rng.uniform(-0.04, 0.04))
    if rng.random() < args.gray_prob:
        img = img.convert("L").convert("RGB")
    if rng.random() < args.img_blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    if rng.random() < args.noise_prob:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.normal(0, rng.uniform(4, 16), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def draw_debug(scene, debug):
    # Only draw the quad polygons -- no text labels (they clutter the view).
    dbg = scene.copy()
    d = ImageDraw.Draw(dbg)
    for pts, text, font_path in debug:
        d.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
    return dbg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_backgrounds(bg_dir):
    files = []
    for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        files += glob.glob(os.path.join(os.path.expanduser(bg_dir), "**",
                                        f"*.{e}"), recursive=True)
    return sorted(set(files))


def sample_canvas(rng, cw, ch, portrait_prob):
    """Pick the canvas orientation: exactly (long, short) landscape or its flip
    (short, long) portrait. Same fixed dimension every time — just rotated — so
    the whole dataset is 885x497 or 497x885 (with portrait_prob for the flip)."""
    long_, short_ = max(cw, ch), min(cw, ch)
    if rng.random() < portrait_prob:
        return short_, long_          # portrait  (W < H), e.g. 497 x 885
    return long_, short_              # landscape (W > H), e.g. 885 x 497


def fit_cover(bg, w, h):
    """Resize + center-crop `bg` to exactly cover a w x h canvas.

    Unlike a plain resize this fills the whole canvas at any target aspect and
    UPSCALES small backgrounds (e.g. 256px Places tiles) to the working size."""
    sc = max(w / bg.width, h / bg.height)
    nw, nh = max(w, int(round(bg.width * sc))), max(h, int(round(bg.height * sc)))
    bg = bg.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return bg.crop((left, top, left + w, top + h))


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Synthetic Indic scene-text dataset (grid layout, diverse).")
    ap.add_argument("--bg-dir", required=True)
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--out-dir", default="./out")
    ap.add_argument("--num-scenes", type=int, default=20)

    # grid / density  (defaults tuned for sparse, realistic scenes)
    ap.add_argument("--cell-size", type=int, default=256)
    ap.add_argument("--max-per-cell", type=int, default=3)
    ap.add_argument("--empty-cell-prob", type=float, default=0.55)

    # size / collision
    ap.add_argument("--min-size", type=int, default=14)
    ap.add_argument("--max-size", type=int, default=46,
                    help="Sizes are biased toward the small end (distant look).")
    ap.add_argument("--iou-thresh", type=float, default=0.15,
                    help="Reject placement if IoU with any box > this (less "
                         "overlap = more realistic).")
    ap.add_argument("--iou-tries", type=int, default=25)

    # per-instance geometry
    ap.add_argument("--rotate-prob", type=float, default=0.25)
    ap.add_argument("--italic-prob", type=float, default=0.2)
    ap.add_argument("--perspective-prob", type=float, default=0.2)
    ap.add_argument("--perspective-frac", type=float, default=0.08,
                    help="Max corner jitter as a fraction of text size.")

    # per-instance appearance
    ap.add_argument("--bold-prob", type=float, default=0.2)
    ap.add_argument("--stroke-prob", type=float, default=0.3)
    ap.add_argument("--shadow-prob", type=float, default=0.25)
    ap.add_argument("--inst-blur-prob", type=float, default=0.15)
    ap.add_argument("--faded-prob", type=float, default=0.15)

    # content
    ap.add_argument("--multiword-prob", type=float, default=0.3)
    ap.add_argument("--number-prob", type=float, default=0.2,
                    help="Chance a latin instance is a numeric token.")

    # whole-image augmentation
    ap.add_argument("--aug-prob", type=float, default=0.4,
                    help="Per-effect prob for brightness/contrast/saturation.")
    ap.add_argument("--hue-prob", type=float, default=0.2)
    ap.add_argument("--gray-prob", type=float, default=0.1)
    ap.add_argument("--img-blur-prob", type=float, default=0.2)
    ap.add_argument("--noise-prob", type=float, default=0.25)

    # misc
    ap.add_argument("--resize", type=int, default=1024,
                    help="Resize bg so longest side <= this (0 = keep).")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    fonts_idx = discover_fonts()
    vocab = load_vocab(args.vocab_dir)
    if not any(fonts_idx[s] and vocab[s] for s in SCRIPTS):
        raise SystemExit("No script has both fonts and vocab. Check "
                         "`fc-cache -f ~/.fonts` and the vocab/ folder.")

    bgs = discover_backgrounds(args.bg_dir)
    if not bgs:
        raise SystemExit(f"No background images in {args.bg_dir}")
    if len(bgs) < 50:
        print(f"\n[bg] WARNING: only {len(bgs)} background image(s). A real "
              f"training dataset needs hundreds-thousands of diverse real scene "
              f"images or the spotter overfits the backgrounds. Reusing them.\n")

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
        scene = augment_scene(scene, rng, args)        # after text -> labels unchanged

        name = f"scene_{i:05d}"
        scene.save(os.path.join(img_dir, name + ".jpg"),
                   quality=rng.randint(60, 95))         # varied JPEG artifacts
        with open(os.path.join(ann_dir, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({"image_file": name + ".jpg",
                       "width": scene.width, "height": scene.height,
                       "instances": instances}, f, ensure_ascii=False, indent=2)
        draw_debug(scene, debug).save(os.path.join(dbg_dir, name + ".jpg"),
                                      quality=92)
        print(f"[gen] {i + 1}/{args.num_scenes}  {name}: {len(instances)} instances")

    print(f"\n[done] {args.num_scenes} scenes -> {args.out_dir}")
    print(f"  images:      {img_dir}")
    print(f"  annotations: {ann_dir}  (one JSON per image)")
    print(f"  debug:       {dbg_dir}  (quads + text drawn)")


if __name__ == "__main__":
    main()
