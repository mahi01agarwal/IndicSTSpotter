#!/usr/bin/env python3
"""Per-IMAGE CER / WER / CRR / WRR on REAL GT-box images, for one or more recognisers.

Same data + cropping as eval_real_crops.py (GT boxes -> recogniser-only, a fair
recogniser comparison), but instead of one pooled score it reports metrics PER IMAGE
and the AVERAGE across images, and writes a per-image CSV per model.

Each image has a JSON:  {"words":[{"x","y","w","h","line","text"}, ...]}.
Crops are grouped back to their source image by the JSON stem.

Definitions:
  CER = char error rate (jiwer, pooled over the image's words)
  WER = word error rate (jiwer, pooled over the image's words)
  CRR = 1 - CER   (clamped to [0,1])   # character recognition rate
  WRR = 1 - WER   (clamped to [0,1])   # word recognition rate
Averages printed are MACRO (mean over images, each image equal weight).
A pooled MICRO figure (each word equal weight) is also printed for reference.

Needs: pip install jiwer pillow
"""
import os
import csv
import json
import glob
import argparse
from collections import OrderedDict, defaultdict

from PIL import Image
import jiwer
from paddleocr import TextRecognition


def crop_gt(images_dir, out_crops, skip_texts, min_size):
    os.makedirs(out_crops, exist_ok=True)
    triples = []                                      # (crop_path, gt_text, stem)
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
            triples.append((cp, text, stem))
    return triples


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


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def metrics(refs, preds):
    """CER/WER over a list of (ref, pred) word pairs; None if jiwer can't score."""
    try:
        c = float(jiwer.cer(refs, preds))
        w = float(jiwer.wer(refs, preds))
        return c, w
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Per-image CER/WER/CRR/WRR on real GT-box images.")
    ap.add_argument("--images-dir", required=True, help="Folder with images + matching GT .json files.")
    ap.add_argument("--out", default="./real_eval_perimage")
    ap.add_argument("--ft-v5-dir", default=None, help="Fine-tuned v5 inference dir.")
    ap.add_argument("--ft-v6-dir", default=None, help="Fine-tuned v6 (unified) inference dir.")
    ap.add_argument("--stock-v5-name", default="devanagari_PP-OCRv5_mobile_rec")
    ap.add_argument("--stock-v6-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--ft-v5-name", default="PP-OCRv5_mobile_rec")   # base config's model_name
    ap.add_argument("--ft-v6-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of {stock_v5,stock_v6,ft_v5,ft_v6} to run (default: all available).")
    ap.add_argument("--skip-text", nargs="*", default=["NA"],
                    help='GT text values to ignore (default: "NA" placeholder boxes).')
    ap.add_argument("--min-size", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    triples = crop_gt(args.images_dir, os.path.join(args.out, "crops"),
                      set(args.skip_text), args.min_size)
    if not triples:
        raise SystemExit("no GT word crops produced — check --images-dir and the JSON format")
    paths = [p for p, _, _ in triples]
    refs = [t for _, t, _ in triples]
    stems = [s for _, _, s in triples]

    image_order = list(OrderedDict.fromkeys(stems))       # stable, first appearance
    idx_by_image = defaultdict(list)
    for k, s in enumerate(stems):
        idx_by_image[s].append(k)
    print(f"[data] {len(triples)} GT word crops from {len(image_order)} images\n")

    available = [("stock_v5", dict(model_name=args.stock_v5_name)),
                 ("stock_v6", dict(model_name=args.stock_v6_name))]
    if args.ft_v5_dir:
        available.append(("ft_v5", dict(model_name=args.ft_v5_name, model_dir=args.ft_v5_dir)))
    if args.ft_v6_dir:
        available.append(("ft_v6", dict(model_name=args.ft_v6_name, model_dir=args.ft_v6_dir)))
    if args.models:
        want = set(args.models)
        available = [(l, kw) for l, kw in available if l in want]
    if not available:
        raise SystemExit("no models selected — check --models / --ft-*-dir")

    summary = []            # (label, macroCER, macroWER, macroCRR, macroWRR, microCER, microWER, n_img_used)
    for label, kw in available:
        print(f"[{label}] {kw}")
        try:
            preds = run(kw, paths, args.batch_size)
        except Exception as e:
            print(f"  [failed: {e}]")
            continue

        words_csv = os.path.join(args.out, f"{label}_words.csv")
        with open(words_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "crop", "ref", "pred"])
            for k, (cp, ref, s) in enumerate(triples):
                wr.writerow([s, os.path.basename(cp), ref, preds[k]])

        per_image, cers, wers, crrs, wrrs = [], [], [], [], []
        for s in image_order:
            idxs = idx_by_image[s]
            r = [refs[k] for k in idxs]
            p = [preds[k] for k in idxs]
            c, w = metrics(r, p)
            if c is None:
                per_image.append((s, len(idxs), None, None, None, None))
                continue
            crr, wrr = clamp01(1.0 - c), clamp01(1.0 - w)
            per_image.append((s, len(idxs), c, w, crr, wrr))
            cers.append(c); wers.append(w); crrs.append(crr); wrrs.append(wrr)

        n_used = len(cers)
        csv_path = os.path.join(args.out, f"{label}_per_image.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "n_words", "CER", "WER", "CRR", "WRR"])
            for s, n, c, w, crr, wrr in per_image:
                if c is None:
                    wr.writerow([s, n, "NA", "NA", "NA", "NA"])
                else:
                    wr.writerow([s, n, f"{c:.4f}", f"{w:.4f}", f"{crr:.4f}", f"{wrr:.4f}"])
            if n_used:
                wr.writerow(["__AVG_over_images__", n_used,
                             f"{sum(cers)/n_used:.4f}", f"{sum(wers)/n_used:.4f}",
                             f"{sum(crrs)/n_used:.4f}", f"{sum(wrrs)/n_used:.4f}"])
        print(f"  -> {csv_path}  ({n_used}/{len(image_order)} images scored)")

        micro_c, micro_w = metrics(refs, preds)            # pooled over all words
        if n_used:
            summary.append((label,
                            sum(cers)/n_used, sum(wers)/n_used,
                            sum(crrs)/n_used, sum(wrrs)/n_used,
                            micro_c, micro_w, n_used))

    print("\n===== Per-image AVERAGE (macro: mean over images) | pooled (micro) =====")
    print(f"{'model':10s}{'CER':>9s}{'WER':>9s}{'CRR':>9s}{'WRR':>9s} | {'uCER':>9s}{'uWER':>9s}{'imgs':>6s}")
    for label, c, w, crr, wrr, mc, mw, n in summary:
        mcs = f"{mc:.4f}" if mc is not None else "   n/a"
        mws = f"{mw:.4f}" if mw is not None else "   n/a"
        print(f"{label:10s}{c:>9.4f}{w:>9.4f}{crr:>9.4f}{wrr:>9.4f} | {mcs:>9s}{mws:>9s}{n:>6d}")
    print("========================================================================")
    print(f"per-image CSVs written under {args.out}/")


if __name__ == "__main__":
    main()
