#!/usr/bin/env python3
"""Per-image visualisation of the end-to-end pipeline on real GT-box images.

For each image it draws:
  RED   = ground-truth word boxes (from the JSON)
  GREEN = the detector's boxes + the recognised text

so you can see, per image, how the pipeline is working — detection coverage
(green over red = found, lone red = missed, lone green = false positive) and the
recogniser's output. Uses the same stock-det + fine-tuned-rec pipeline as the eval.

Run:
  python viz_pipeline.py --images-dir $SCR/real_hindi \
    --rec-dir $SCR/PaddleOCR/output/v6_unified_infer --rec-name PP-OCRv6_medium_rec \
    --out $SCR/results/hindi_viz
"""
import os
import json
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR


def load_font(size):
    """A Devanagari-capable font if we can find one, else default (boxes still fine)."""
    cands = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        os.path.expanduser("~/.fonts/NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        os.path.join(os.getcwd(), "simfang.ttf"),
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description="Per-image bbox + text visualisation of the pipeline.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--rec-dir", required=True, help="fine-tuned recogniser inference dir")
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--det-name", default="PP-OCRv6_medium_det", help="stock detector name")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--out", default="./pipeline_viz")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick look)")
    ap.add_argument("--no-text", action="store_true", help="draw boxes only (skip recognised text)")
    ap.add_argument("--no-gt", action="store_true", help="skip the red GT boxes")
    ap.add_argument("--box-width", type=int, default=0, help="box thickness in px (0 = auto by image size)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    common = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)
    kw = dict(text_recognition_model_dir=args.rec_dir, text_recognition_model_name=args.rec_name,
              text_detection_model_name=args.det_name, **common)
    if args.det_dir:
        kw["text_detection_model_dir"] = args.det_dir
    ocr = PaddleOCR(**kw)

    jsons = sorted(glob.glob(os.path.join(args.images_dir, "*.json")))
    if args.limit:
        jsons = jsons[:args.limit]
    n = 0
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        img_path = next((os.path.join(args.images_dir, f"{stem}.{e}")
                         for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                         if os.path.isfile(os.path.join(args.images_dir, f"{stem}.{e}"))), None)
        if not img_path:
            continue
        base = Image.open(img_path).convert("RGB")
        bw = args.box_width or max(4, round(max(base.size) / 500))   # bold, scaled to image size
        fsize = max(20, round(max(base.size) / 80))
        font = load_font(fsize)

        # image 1 — RED: ground-truth boxes only  ->  <stem>_gt.jpg
        if not args.no_gt:
            im_gt = base.copy()
            dg = ImageDraw.Draw(im_gt)
            data = json.load(open(jp, encoding="utf-8"))
            for w in data.get("words", []):
                try:
                    x, y, ww, hh = int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"])
                    dg.rectangle([x, y, x + ww, y + hh], outline=(230, 20, 20), width=bw)
                except Exception:
                    pass
            im_gt.save(os.path.join(args.out, f"{stem}_gt.jpg"), quality=90)

        # image 2 — GREEN: detected boxes (+ recognised text) only  ->  <stem>_det.jpg
        im_det = base.copy()
        dd = ImageDraw.Draw(im_det)
        for res in ocr.predict(img_path):
            d = res.json
            d = d.get("res", d) if isinstance(d, dict) else {}
            texts = d.get("rec_texts") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            for t, poly in zip(texts, polys):
                pts = [tuple(map(int, pt)) for pt in np.array(poly).reshape(-1, 2)]
                dd.line(pts + [pts[0]], fill=(0, 190, 60), width=bw)
                if not args.no_text and t:
                    x0 = min(q[0] for q in pts)
                    y0 = min(q[1] for q in pts)
                    dd.text((x0, max(0, y0 - fsize - 2)), t, fill=(0, 120, 20), font=font)
        im_det.save(os.path.join(args.out, f"{stem}_det.jpg"), quality=90)
        n += 1
        print(f"  {n}/{len(jsons)} {stem}      ", end="\r")

    print(f"\n[done] {n} images -> {args.out}   (<stem>_gt.jpg = red GT boxes, <stem>_det.jpg = green detector boxes)")


if __name__ == "__main__":
    main()
