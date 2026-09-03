#!/usr/bin/env python3
"""Visualise detector boxes on a few images: our fine-tuned det vs the stock det.
Saves annotated images and prints box counts -> diagnoses over-detection (domain gap).

Run (in the paddle env, on a machine with the images + exported det):
    python draw_det.py --images-dir $SCR/real_marathi \
        --our-det $SCR/PaddleOCR/output/v6_det_realistic_infer --n 3 --out $SCR/det_vis
"""
import os
import glob
import argparse

from paddleocr import TextDetection


def n_boxes(res):
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return len(d.get("dt_polys") or [])
    except Exception:
        return -1


def run(tag, model_dir, imgs, outdir):
    kw = dict(model_name="PP-OCRv6_medium_det")
    if model_dir:
        kw["model_dir"] = model_dir
    det = TextDetection(**kw)
    os.makedirs(outdir, exist_ok=True)
    for img in imgs:
        for res in det.predict(input=img):
            print(f"[{tag}] {os.path.basename(img)}: {n_boxes(res)} boxes")
            res.save_to_img(outdir)


def main():
    ap = argparse.ArgumentParser(description="Visualise our det vs stock det boxes on a few images.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--our-det", required=True, help="Our exported det inference dir (the RAW export).")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="./det_vis")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))[:args.n]
    if not imgs:
        raise SystemExit(f"no .jpg under {args.images_dir}")
    print("images:", [os.path.basename(i) for i in imgs])
    run("stock", None, imgs, os.path.join(args.out, "stock"))
    run("ours", args.our_det, imgs, os.path.join(args.out, "ours"))
    print(f"\nvis saved under {args.out}/stock and {args.out}/ours")


if __name__ == "__main__":
    main()
