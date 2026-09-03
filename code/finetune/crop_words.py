#!/usr/bin/env python3
"""Crop per-word images from the synthetic dataset into PaddleOCR recognition format.

PaddleOCR rec models train on CROPPED word/line images + an `image<TAB>label` file,
not full pages. Our scene images have per-word boxes + text in the annotations, so we
cut each word out. Output, per script:

    rec_data/
      crops/<script>/<scene>_<i>.jpg
      <script>_train.txt      # lines:  crops/<script>/xxx.jpg<TAB><text>
      <script>_val.txt
      <script>_dict.txt       # one character per line (no space; use_space_char handles spaces)

Run:
    python crop_words.py --dataset-dir $SCR/out_dataset --out-dir $SCR/rec_data --skip-curved
"""
import os
import json
import glob
import random
import argparse
from collections import defaultdict
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description="Crop word images -> PaddleOCR rec format.")
    ap.add_argument("--dataset-dir", required=True, help="out_dataset/ (has images/ + annotations/).")
    ap.add_argument("--out-dir", default="./rec_data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=8, help="Skip crops smaller than this (px).")
    ap.add_argument("--pad", type=int, default=2, help="Padding around each bbox (px).")
    ap.add_argument("--skip-curved", action="store_true",
                    help="Skip curved instances (bent text is poor for a line recognizer).")
    ap.add_argument("--scripts", nargs="*", default=None, help="Subset of scripts (default: all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann_dir = os.path.join(args.dataset_dir, "annotations")
    img_dir = os.path.join(args.dataset_dir, "images")
    os.makedirs(args.out_dir, exist_ok=True)

    records = defaultdict(list)          # script -> [(rel_path, text)]
    n_crops = n_skip = 0
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        ann = json.load(open(ann_path, encoding="utf-8"))
        img_path = os.path.join(img_dir, ann["image_file"])
        if not os.path.isfile(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        stem = os.path.splitext(ann["image_file"])[0]
        for i, inst in enumerate(ann.get("instances", [])):
            script = inst["script"]
            if args.scripts and script not in args.scripts:
                continue
            if args.skip_curved and "polygon" in inst:
                n_skip += 1
                continue
            text = inst.get("text", "").strip()
            if not text:
                continue
            x, y, w, h = inst["bbox"]
            if w < args.min_size or h < args.min_size:
                n_skip += 1
                continue
            x0, y0 = max(0, x - args.pad), max(0, y - args.pad)
            x1, y1 = min(W, x + w + args.pad), min(H, y + h + args.pad)
            crop = img.crop((x0, y0, x1, y1))
            sdir = os.path.join(args.out_dir, "crops", script)
            os.makedirs(sdir, exist_ok=True)
            rel = f"crops/{script}/{stem}_{i}.jpg"      # forward slashes for the label file
            crop.save(os.path.join(args.out_dir, rel), quality=95)
            records[script].append((rel, text))
            n_crops += 1

    rng = random.Random(args.seed)
    print(f"\n[crops] {n_crops} written, {n_skip} skipped\n")
    for script, recs in sorted(records.items()):
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * args.val_frac))
        val, train = recs[:n_val], recs[n_val:]
        for split, rs in (("train", train), ("val", val)):
            with open(os.path.join(args.out_dir, f"{script}_{split}.txt"), "w", encoding="utf-8") as f:
                for rel, text in rs:
                    f.write(f"{rel}\t{text}\n")
        # char dict: unique non-whitespace chars (spaces handled by use_space_char in the config)
        chars = sorted({c for _, t in recs for c in t if not c.isspace()})
        with open(os.path.join(args.out_dir, f"{script}_dict.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(chars) + "\n")
        maxlen = max((len(t) for _, t in recs), default=0)
        print(f"[{script:11s}] train={len(train):6d} val={len(val):5d} "
              f"chars={len(chars):4d} max_text_len={maxlen}")

    print(f"\n[done] -> {args.out_dir}  (per-script *_train.txt / *_val.txt / *_dict.txt)")


if __name__ == "__main__":
    main()
