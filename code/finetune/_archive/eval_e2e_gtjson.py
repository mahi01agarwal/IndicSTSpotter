#!/usr/bin/env python3
"""End-to-end (DETECTOR + recogniser) eval on REAL images with per-word GT JSONs.

Runs the FULL PaddleOCR pipeline — by default **stock v6 detector + your fine-tuned v6
recogniser** — on whole images, joins recognised text in reading order, and compares to
the GT reading-order text built from each image's {"words":[{x,y,w,h,line,text}]} JSON.

Unlike the recogniser-only eval (which is fed GT boxes), this INCLUDES detector errors,
and reports how many words the detector actually found.

Table it prints (one row per run):
  Model | Total GT words | Words detected | CER | WER | CRR | WRR
CER/WER are page-level (reading-order concatenation), pooled MICRO over all images.
CRR = 1-CER, WRR = 1-WER (clamped).

Run:
  python eval_e2e_gtjson.py --images-dir $SCR/real_hindi \
    --rec-dir $SCR/PaddleOCR/output/v6_unified_infer --rec-name PP-OCRv6_medium_rec \
    --out $SCR/e2e_real_hindi --label "stock_v6_det + ft_v6_rec (Hindi)"
"""
import os
import csv
import json
import glob
import argparse

import numpy as np
import jiwer
from paddleocr import PaddleOCR


def gt_reading_order(words, skip_texts):
    """GT string in reading order (line asc, then x asc) + GT word count."""
    items = []
    for w in words:
        t = str(w.get("text", "")).strip()
        if not t or t in skip_texts:
            continue
        items.append((int(w.get("line", 0)), int(w.get("x", 0)), t))
    items.sort(key=lambda z: (z[0], z[1]))
    return " ".join(t for _, _, t in items), len(items)


def pred_reading_order(result):
    """Join recognised boxes in reading order; return (text, n_detected_boxes)."""
    items, n_det = [], 0
    for res in result:
        try:
            d = res.json
            d = d.get("res", d) if isinstance(d, dict) else {}
            texts = d.get("rec_texts") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            n_det += len(polys)
            for t, p in zip(texts, polys):
                arr = np.array(p).reshape(-1, 2)
                x = float(arr[:, 0].min())
                y = float(arr[:, 1].min())
                h = float(arr[:, 1].max() - arr[:, 1].min())
                items.append((y + h / 2.0, x, h, t))
        except Exception:
            pass
    items.sort(key=lambda t: t[0])
    lines = []
    for cy, x, h, t in items:
        for ln in lines:
            if abs(cy - ln[0]) <= 0.6 * h:
                ln[2].append((x, t))
                ln[0] = (ln[0] * ln[1] + cy) / (ln[1] + 1)
                ln[1] += 1
                break
        else:
            lines.append([cy, 1, [(x, t)]])
    lines.sort(key=lambda ln: ln[0])
    return "\n".join(" ".join(t for _, t in sorted(ln[2])) for ln in lines), n_det


def norm(s):
    return " ".join(str(s).split())


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def main():
    ap = argparse.ArgumentParser(description="End-to-end det+rec eval on real GT-JSON images.")
    ap.add_argument("--images-dir", required=True, help="Folder with images + matching GT .json.")
    ap.add_argument("--rec-dir", required=True, help="Fine-tuned recogniser inference dir.")
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--det-name", default="PP-OCRv6_medium_det", help="Stock v6 detector name.")
    ap.add_argument("--det-dir", default=None, help="Custom detector dir (default: stock by name).")
    ap.add_argument("--label", default="stock_v6_det + ft_v6_rec")
    ap.add_argument("--skip-text", nargs="*", default=["NA"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="./e2e_eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    skip = set(args.skip_text)
    jsons = sorted(glob.glob(os.path.join(args.images_dir, "*.json")))
    if args.limit:
        jsons = jsons[:args.limit]

    rows = []                                        # (img_path, gt_text, n_gt_words, stem)
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = next((os.path.join(args.images_dir, f"{stem}.{e}")
                    for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                    if os.path.isfile(os.path.join(args.images_dir, f"{stem}.{e}"))), None)
        if img is None:
            continue
        data = json.load(open(jp, encoding="utf-8"))
        gt, n = gt_reading_order(data.get("words", []), skip)
        if n == 0:
            continue
        rows.append((img, gt, n, stem))
    if not rows:
        raise SystemExit("no GT words found — check --images-dir / JSON format")
    print(f"[data] {len(rows)} images from {args.images_dir}")

    common = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)
    kw = dict(text_recognition_model_dir=args.rec_dir,
              text_recognition_model_name=args.rec_name,
              text_detection_model_name=args.det_name, **common)
    if args.det_dir:
        kw["text_detection_model_dir"] = args.det_dir
    print(f"[pipeline] det={args.det_dir or args.det_name}  rec={args.rec_dir}")
    ocr = PaddleOCR(**kw)

    refs, preds = [], []
    total_gt, total_det = 0, 0
    per_rows = []                                    # image,gt_words,det_words,CER,WER,CRR,WRR
    text_rows = []                                   # image,gt_text,pred_text (debug)
    for i, (img, gt, n, stem) in enumerate(rows):
        pred, n_det = pred_reading_order(ocr.predict(img))
        rg, rp = norm(gt), norm(pred)
        refs.append(rg)
        preds.append(rp)
        total_gt += n
        total_det += n_det
        try:
            c = float(jiwer.cer(rg, rp)); w = float(jiwer.wer(rg, rp))
        except Exception:
            c, w = float("nan"), float("nan")
        per_rows.append((stem, n, n_det, round(c, 4), round(w, 4),
                         round(clamp01(1 - c), 4), round(clamp01(1 - w), 4)))
        text_rows.append((stem, rg, rp))
        print(f"    {i + 1}/{len(rows)}", end="\r")
    print()

    mCER = float(jiwer.cer(refs, preds))
    mWER = float(jiwer.wer(refs, preds))
    CRR, WRR = clamp01(1.0 - mCER), clamp01(1.0 - mWER)

    # clean per-image metrics CSV (same style as the recogniser-only eval)
    with open(os.path.join(args.out, "per_image.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "gt_words", "det_words", "CER", "WER", "CRR", "WRR"])
        wr.writerows(per_rows)
    # separate text dump for eyeballing detector/recogniser mistakes
    with open(os.path.join(args.out, "per_image_text.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "gt_text", "pred_text"])
        wr.writerows(text_rows)

    print("\n=================== END-TO-END (detector + recogniser) ===================")
    print(f"{'Model':36s}{'GT words':>9s}{'Detected':>9s}{'CER':>8s}{'WER':>8s}{'CRR':>8s}{'WRR':>8s}")
    print(f"{args.label:36s}{total_gt:>9d}{total_det:>9d}{mCER:>8.4f}{mWER:>8.4f}{CRR:>8.4f}{WRR:>8.4f}")
    print("==========================================================================")
    print(f"per-image detail -> {args.out}/per_image.csv")


if __name__ == "__main__":
    main()
