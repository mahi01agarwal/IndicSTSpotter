#!/usr/bin/env python3
"""CER/WER on REAL images with GT-box JSONs — compares 4 recognisers.

Each image has a JSON:  {"words":[{"x","y","w","h","line","text"}, ...]}.
We crop every GT word (using its box) and run each recogniser on the crop, then report
CER/WER per model. GT boxes are used directly -> a fair recogniser-only comparison.

Models (fine-tuned ones added only if their --*-dir is given):
  stock_v5   default: devanagari_PP-OCRv5_mobile_rec
  ft_v5      your fine-tuned v5 (Devanagari)
  stock_v6   default: PP-OCRv6_medium_rec
  ft_v6      your fine-tuned v6 (unified)

Needs: pip install jiwer pillow
"""
import os
import json
import glob
import argparse

from PIL import Image
import jiwer
from paddleocr import TextRecognition


def crop_gt(images_dir, out_crops, skip_texts, min_size):
    os.makedirs(out_crops, exist_ok=True)
    pairs = []                                        # (crop_path, gt_text)
    for jp in sorted(glob.glob(os.path.join(images_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        img_path = next((os.path.join(images_dir, f"{stem}.{e}")
                         for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                         if os.path.isfile(os.path.join(images_dir, f"{stem}.{e}"))), None)
        if img_path is None:
            print(f"[skip] no image for {jp}")
            continue
        data = json.load(open(jp, encoding="utf-8"))
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for i, w in enumerate(data.get("words", [])):
            text = str(w.get("text", "")).strip()
            if not text or text in skip_texts:
                continue
            x, y, ww, hh = int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"])
            if ww < min_size or hh < min_size:
                continue
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + ww), min(H, y + hh)
            if x1 <= x0 or y1 <= y0:
                continue
            cp = os.path.join(out_crops, f"{stem}_{i}.jpg")
            img.crop((x0, y0, x1, y1)).save(cp)
            pairs.append((cp, text))
    return pairs


def rec_text(res):
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return d.get("rec_text") or ""
    except Exception:
        return ""


def run(kwargs, paths, bs):
    model = TextRecognition(**kwargs)
    preds = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        for res in model.predict(input=chunk, batch_size=len(chunk)):
            preds.append(rec_text(res))
        print(f"    {min(i + bs, len(paths))}/{len(paths)}", end="\r")
    print()
    if len(preds) != len(paths):
        raise RuntimeError(f"{len(preds)} preds for {len(paths)} crops — result format differs")
    return preds


def main():
    ap = argparse.ArgumentParser(description="CER/WER on real GT-box images, 4 recognisers.")
    ap.add_argument("--images-dir", required=True, help="Folder with images + matching GT .json files.")
    ap.add_argument("--out", default="./real_eval")
    ap.add_argument("--ft-v5-dir", default=None, help="Fine-tuned v5 (Devanagari) inference dir.")
    ap.add_argument("--ft-v6-dir", default=None, help="Fine-tuned v6 (unified) inference dir.")
    ap.add_argument("--stock-v5-name", default="devanagari_PP-OCRv5_mobile_rec")
    ap.add_argument("--stock-v6-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--ft-v5-name", default="PP-OCRv5_server_rec")
    ap.add_argument("--ft-v6-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--skip-text", nargs="*", default=["NA"],
                    help='GT text values to ignore (default: "NA" placeholder boxes).')
    ap.add_argument("--min-size", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    pairs = crop_gt(args.images_dir, os.path.join(args.out, "crops"),
                    set(args.skip_text), args.min_size)
    if not pairs:
        raise SystemExit("no GT word crops produced — check --images-dir and the JSON format")
    paths = [p for p, _ in pairs]
    refs = [t for _, t in pairs]
    print(f"[data] {len(pairs)} GT word crops from {args.images_dir}\n")

    models = [("stock_v5", dict(model_name=args.stock_v5_name)),
              ("stock_v6", dict(model_name=args.stock_v6_name))]
    if args.ft_v5_dir:
        models.append(("ft_v5", dict(model_name=args.ft_v5_name, model_dir=args.ft_v5_dir)))
    if args.ft_v6_dir:
        models.append(("ft_v6", dict(model_name=args.ft_v6_name, model_dir=args.ft_v6_dir)))

    rows = []
    for label, kw in models:
        print(f"[{label}] {kw}")
        try:
            preds = run(kw, paths, args.batch_size)
            rows.append((label, len(paths),
                         float(jiwer.cer(refs, preds)), float(jiwer.wer(refs, preds))))
        except Exception as e:
            print(f"  [failed: {e}]")
            rows.append((label, len(paths), None, None))

    def f(x):
        return "  n/a " if x is None else f"{x:.4f}"

    print("\n=========== REAL images: CER / WER per model (lower is better) ===========")
    print(f"{'model':12s}{'n':>7s}{'CER':>10s}{'WER':>10s}")
    for l, n, c, w in rows:
        print(f"{l:12s}{n:>7d}{f(c):>10s}{f(w):>10s}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
