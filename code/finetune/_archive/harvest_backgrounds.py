#!/usr/bin/env python3
"""Harvest TEXT-FREE, clutter-rich background crops from IDD (or any street dataset).

Why this exists
---------------
Indian street photos are the best available source of the structural clutter our
detector false-positives on (corrugated awnings, shop shutters, lattice trusses,
window grilles, railings) -- and simultaneously the most text-saturated images
there are. Every shopfront carries a signboard.

That matters because an unlabelled real signboard in a background image is a
POSITIVE that the loss punishes the detector for finding. Paste synthetic words
on top, label only those, and every real sign in frame becomes supervised
background: you would be training text SUPPRESSION. That is a worse failure than
the over-firing we already have, and much harder to notice -- it looks like
"fewer false positives" right up until recall collapses.

So we do not use whole street frames. We cut CROPS that provably contain no text,
and keep only crops that still carry real structure. IDD frames are 1920x1080, so
several ~900px crops come out of each one -- the pool multiplies rather than
shrinks, and every crop is clean by construction.

Two independent text filters, best used together:
  * --seg-dir      IDD semantic masks. Rejects any crop touching a text-bearing
                   class (billboard / traffic sign). Catches small and blurry
                   text a detector misses. No model needed.
  * --use-detector Stock PP-OCRv6 det pass. Catches painted shop names, banners
                   and wall text that carry no billboard label.
Use STOCK weights for this, never our fine-tuned detector -- ours over-fires on
exactly the clutter we are trying to keep, so it would reject the good crops.

Output feeds generate_realistic.py --bg-dir directly. The same pool serves both
roles: positives get synthetic text pasted on, and --negative-frac leaves the
rest textless as hard negatives.

Usage
  # segmentation-only (no paddle needed; runs anywhere)
  python harvest_backgrounds.py --images-dir IDD/leftImg8bit/train \
      --seg-dir IDD/gtFine/train --text-class-ids 20,21 --out-dir bg_idd

  # detector-only (needs paddleocr; run on ada/gvlab)
  python harvest_backgrounds.py --images-dir IDD/leftImg8bit/train \
      --use-detector --out-dir bg_idd

  # both (recommended)
  python harvest_backgrounds.py --images-dir IDD/leftImg8bit/train \
      --seg-dir IDD/gtFine/train --text-class-ids 20,21 \
      --use-detector --out-dir bg_idd
"""
import os
import json
import glob
import random
import argparse

import numpy as np
from PIL import Image


def find_images(images_dir):
    """IDD nests frames under per-drive subdirs, so recurse."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    out = []
    for e in exts:
        out += glob.glob(os.path.join(images_dir, "**", e), recursive=True)
    return sorted(out)


def seg_path_for(img_path, images_dir, seg_dir):
    """IDD mirrors the drive/frame tree between leftImg8bit/ and gtFine/."""
    rel = os.path.relpath(img_path, images_dir)
    stem = os.path.splitext(rel)[0]
    # IDD frames are <id>_leftImg8bit.png, masks <id>_gtFine_labellevel3Ids.png
    stem = stem.replace("_leftImg8bit", "")
    for suffix in ("_gtFine_labellevel3Ids.png", "_gtFine_labelids.png",
                   "_gtFine_labelIds.png", ".png"):
        p = os.path.join(seg_dir, stem + suffix)
        if os.path.isfile(p):
            return p
    return None


def texture_energy(gray):
    """Mean absolute gradient. Near 0 = flat sky/asphalt (a trivial negative that
    teaches nothing); high = the periodic structure we actually want to train on."""
    a = gray.astype(np.float32)
    if a.shape[0] < 2 or a.shape[1] < 2:
        return 0.0
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return float((gx + gy) / 2.0)


def detect_text_boxes(model, img_path):
    """Axis-aligned bounds of every stock-detector quad. Tolerates result shapes."""
    boxes = []
    try:
        for res in model.predict(input=img_path):
            polys = None
            for key in ("dt_polys", "boxes", "polys"):
                if isinstance(res, dict) and key in res:
                    polys = res[key]
                    break
            if polys is None:
                polys = getattr(res, "dt_polys", None)
            for quad in (polys if polys is not None else []):
                pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
                boxes.append((pts[:, 0].min(), pts[:, 1].min(),
                              pts[:, 0].max(), pts[:, 1].max()))
    except Exception as e:
        print(f"  [warn] detect failed on {os.path.basename(img_path)}: {e}")
    return boxes


def overlaps(crop, boxes, margin):
    """True if the crop touches any text box, dilated by `margin` px."""
    cx0, cy0, cx1, cy1 = crop
    for bx0, by0, bx1, by1 in boxes:
        if (cx0 < bx1 + margin and cx1 > bx0 - margin and
                cy0 < by1 + margin and cy1 > by0 - margin):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(
        description="Cut text-free, clutter-rich background crops for synth generation.")
    ap.add_argument("--images-dir", required=True, help="IDD leftImg8bit/<split> (recursed).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seg-dir", default=None, help="IDD gtFine/<split> label masks.")
    ap.add_argument("--text-class-ids", default=None,
                    help="Comma-separated label ids that carry text (billboard, traffic "
                         "sign). Map names->ids with IDD's helpers/anue_labels.py -- do "
                         "NOT guess, the ids differ between labellevel3Ids and labelIds.")
    ap.add_argument("--use-detector", action="store_true",
                    help="Also run a stock PaddleOCR detector to catch unlabelled text.")
    ap.add_argument("--det-model", default="PP-OCRv6_medium_det",
                    help="STOCK det model name. Never our fine-tuned weights.")
    ap.add_argument("--crops-per-image", type=int, default=3)
    ap.add_argument("--crop-min", type=int, default=700,
                    help="Min crop side. Keep under generate_realistic.py --resize (1024) "
                         "so crops are never upsampled.")
    ap.add_argument("--crop-max", type=int, default=1000)
    ap.add_argument("--margin", type=int, default=24, help="Dilation around text boxes.")
    ap.add_argument("--min-texture", type=float, default=6.0,
                    help="Reject flat crops (empty sky / bare asphalt).")
    ap.add_argument("--tries", type=int, default=40, help="Crop placement attempts per image.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N source frames.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seg_dir and not args.text_class_ids:
        raise SystemExit("--seg-dir needs --text-class-ids (map names via IDD's "
                         "helpers/anue_labels.py; guessing ids silently lets text through)")
    text_ids = set(int(x) for x in args.text_class_ids.split(",")) if args.text_class_ids else set()

    imgs = find_images(args.images_dir)
    if not imgs:
        raise SystemExit(f"no images under {args.images_dir}")
    if args.limit:
        imgs = imgs[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    model = None
    if args.use_detector:
        from paddleocr import TextDetection
        model = TextDetection(model_name=args.det_model)

    rng = random.Random(args.seed)
    n_out = n_seg_rej = n_det_rej = n_flat = 0
    provenance = []

    for n_done, img_path in enumerate(imgs, 1):
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        if min(W, H) < args.crop_min:
            continue

        seg = None
        if args.seg_dir:
            sp = seg_path_for(img_path, args.images_dir, args.seg_dir)
            if sp:
                seg = np.asarray(Image.open(sp))
                if seg.ndim == 3:
                    seg = seg[:, :, 0]
                if seg.shape[:2] != (H, W):      # mask stored at a different scale
                    seg = np.asarray(Image.fromarray(seg).resize((W, H), Image.NEAREST))

        boxes = detect_text_boxes(model, img_path) if model else []
        gray_full = np.asarray(im.convert("L"))

        kept = 0
        for _ in range(args.tries):
            if kept >= args.crops_per_image:
                break
            cw = rng.randint(args.crop_min, min(args.crop_max, W))
            ch = rng.randint(args.crop_min, min(args.crop_max, H))
            if cw > W or ch > H:
                continue
            x0 = rng.randint(0, W - cw)
            y0 = rng.randint(0, H - ch)
            crop = (x0, y0, x0 + cw, y0 + ch)

            if boxes and overlaps(crop, boxes, args.margin):
                n_det_rej += 1
                continue
            if seg is not None and text_ids:
                if np.isin(seg[y0:y0 + ch, x0:x0 + cw], list(text_ids)).any():
                    n_seg_rej += 1
                    continue
            if texture_energy(gray_full[y0:y0 + ch, x0:x0 + cw]) < args.min_texture:
                n_flat += 1
                continue

            stem = os.path.splitext(os.path.basename(img_path))[0]
            name = f"{stem}_c{kept}.jpg"
            im.crop(crop).save(os.path.join(args.out_dir, name), quality=95)
            provenance.append({"crop": name, "source": img_path, "box": list(crop)})
            kept += 1
            n_out += 1

        if n_done % 200 == 0:
            print(f"[harvest] {n_done}/{len(imgs)} frames -> {n_out} crops")

    with open(os.path.join(args.out_dir, "provenance.jsonl"), "w", encoding="utf-8") as f:
        for p in provenance:
            f.write(json.dumps(p) + "\n")

    print(f"\n[done] {n_out} text-free crops from {len(imgs)} frames -> {args.out_dir}")
    print(f"  rejected: detector-text={n_det_rej}  seg-text={n_seg_rej}  too-flat={n_flat}")
    if not args.use_detector and not args.seg_dir:
        print("  WARNING: no text filter ran. Every crop may contain real text, which "
              "trains text SUPPRESSION. Pass --seg-dir and/or --use-detector.")
    print(f"  next: generate_realistic.py --bg-dir {args.out_dir} --config dataset_config_det.json")


if __name__ == "__main__":
    main()
