#!/usr/bin/env python3
"""Run inference on a few images with the STOCK (before) and FINE-TUNED (after) models.

No ground truth needed — just detects + recognises, draws the boxes, and dumps the text.
For each image it writes an annotated .jpg (red boxes + an index per box) and a .txt
listing every box's coordinates and recognised string, under out/before/ and out/after/.

Needs: pip install pillow numpy
Run:
    python infer_images.py \
        --images    /ssd_scratch/cvit/mahima/test_images \
        --finetuned /ssd_scratch/cvit/mahima/PaddleOCR/output/unified_v6_infer \
        --finetuned-model-name PP-OCRv6_medium_rec \
        --out       /ssd_scratch/cvit/mahima/infer_out
"""
import os
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR


def list_images(path):
    if os.path.isdir(path):
        files = []
        for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            files += glob.glob(os.path.join(path, f"*.{e}"))
        return sorted(set(files))
    return sorted(glob.glob(path))


def extract(result):
    """List of (text, poly-as-Nx2-array) from a PaddleOCR pipeline result."""
    out = []
    for res in result:
        try:
            d = res.json
            d = d.get("res", d) if isinstance(d, dict) else {}
            texts = d.get("rec_texts") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            for t, p in zip(texts, polys):
                out.append((t, np.array(p).reshape(-1, 2)))
        except Exception:
            pass
    return out


def annotate(img_path, dets, out_img):
    img = Image.open(img_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    for i, (t, poly) in enumerate(dets):
        pts = [(float(x), float(y)) for x, y in poly]
        drw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
        drw.text((pts[0][0], max(0, pts[0][1] - 10)), str(i), fill=(255, 0, 0))
    img.save(out_img)


def run_model(ocr, images, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        dets = extract(ocr.predict(img_path))
        annotate(img_path, dets, os.path.join(out_dir, f"{name}.jpg"))
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            for i, (t, poly) in enumerate(dets):
                f.write(f"{i}\t{poly.astype(int).tolist()}\t{t}\n")
        print(f"[{tag}] {name}: {len(dets)} boxes -> {out_dir}/{name}.jpg (+ .txt)")


def main():
    ap = argparse.ArgumentParser(description="Inference (boxes + text) with stock vs fine-tuned.")
    ap.add_argument("--images", required=True, help="Folder or glob of images.")
    ap.add_argument("--finetuned", required=True, help="Exported fine-tuned inference dir.")
    ap.add_argument("--finetuned-model-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--out", default="./infer_out")
    ap.add_argument("--skip-stock", action="store_true", help="Only run the fine-tuned model.")
    args = ap.parse_args()

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"no images found at {args.images}")
    print(f"{len(images)} image(s)\n")

    common = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)

    if not args.skip_stock:
        print("=== BEFORE (stock) ===")
        run_model(PaddleOCR(**common), images, os.path.join(args.out, "before"), "before")

    print("\n=== AFTER (fine-tuned) ===")
    run_model(PaddleOCR(text_recognition_model_dir=args.finetuned,
                        text_recognition_model_name=args.finetuned_model_name, **common),
              images, os.path.join(args.out, "after"), "after")

    print(f"\ndone -> {args.out}/before/  and  {args.out}/after/   (annotated .jpg + .txt per image)")


if __name__ == "__main__":
    main()
