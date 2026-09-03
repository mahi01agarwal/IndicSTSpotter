#!/usr/bin/env python3
"""Whole-IMAGE CER/WER/CRR/WRR: stock pipeline (before) vs fine-tuned recogniser (after).

Runs the FULL PaddleOCR pipeline (detector + recogniser) on complete scene images,
joins recognised text in reading order, and compares to each image's ground-truth
transcription from manifest.jsonl.
  before = stock pipeline (default detector + default recogniser)
  after  = stock detector + your fine-tuned unified recogniser

Metrics (edit-distance based; CRR = 1-CER, WRR = 1-WER, clamped to [0,1]):
  per-image -> one CER/WER/CRR/WRR per image      -> <model>_per_image.csv
  micro     -> pooled over ALL chars/words         (long pages dominate; primary number)
  macro     -> mean of the per-image scores        (every page equal weight)
Aggregates (micro+macro per model + before->after gain) -> aggregate.csv.

Whitespace (incl. the reading-order newlines) is collapsed to single spaces on BOTH
ref and pred, so we score text not layout. End-to-end has no clean 1:1 word alignment,
so the ilocr-style exact-match (Correct/Total) is intentionally NOT computed here -- use
the recogniser-only eval for that.

Needs: pip install jiwer
Run:
    python eval_fullpage.py \
        --manifest    /ssd_scratch/cvit/mahima/real_hindi/manifest_real.jsonl \
        --images-root /ssd_scratch/cvit/mahima/real_hindi \
        --finetuned   /ssd_scratch/cvit/mahima/PaddleOCR/output/v6_unified_infer \
        --finetuned-model-name PP-OCRv6_medium_rec \
        --out ./fullpage_real_hindi --limit 0
"""
import os
import csv
import json
import argparse

import numpy as np
import jiwer
from paddleocr import PaddleOCR


def load_manifest(path, images_root, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("text", "").strip():
                rows.append((os.path.join(images_root, r["image"]), r["text"]))
    return rows[:limit] if limit else rows


def joined_reading_order(result):
    """Join recognised boxes into one string: rows top->bottom, words left->right
    (same ordering the dataset's ground-truth transcription uses)."""
    items = []
    for res in result:
        try:
            d = res.json
            d = d.get("res", d) if isinstance(d, dict) else {}
            texts = d.get("rec_texts") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
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
    return "\n".join(" ".join(t for _, t in sorted(ln[2])) for ln in lines)


def run(ocr, rows):
    preds = []
    for i, (img, _) in enumerate(rows):
        preds.append(joined_reading_order(ocr.predict(img)))
        print(f"    {i + 1}/{len(rows)}", end="\r")
    print()
    return preds


def norm(s):
    """Collapse all whitespace (incl. reading-order newlines) to single spaces."""
    return " ".join(str(s).split())


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_pair(ref, pred):
    """Per-image CER/WER/CRR/WRR for one (ref, pred)."""
    c = float(jiwer.cer(ref, pred))
    w = float(jiwer.wer(ref, pred))
    return c, w, clamp01(1.0 - c), clamp01(1.0 - w)


def evaluate(refs, preds):
    """Return (per_image, micro, macro).
    per_image[i] = (CER,WER,CRR,WRR); micro/macro = (CER,WER,CRR,WRR)."""
    per = [score_pair(r, p) for r, p in zip(refs, preds)]
    mCER = float(jiwer.cer(refs, preds))               # pooled over all chars
    mWER = float(jiwer.wer(refs, preds))               # pooled over all words
    micro = (mCER, mWER, clamp01(1.0 - mCER), clamp01(1.0 - mWER))
    if per:
        macro = tuple(sum(col) / len(per) for col in zip(*per))   # per-image CRR/WRR already clamped
    else:
        macro = (0.0, 0.0, 0.0, 0.0)
    return per, micro, macro


def main():
    ap = argparse.ArgumentParser(description="Whole-image CER/WER/CRR/WRR: stock vs fine-tuned.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-root", required=True, help="Dir the manifest 'image' paths are relative to.")
    ap.add_argument("--finetuned", required=True, help="Exported unified inference dir.")
    ap.add_argument("--finetuned-model-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--detector", default=None,
                    help="Fine-tuned detector inference dir (default: stock detector).")
    ap.add_argument("--detector-name", default="PP-OCRv6_medium_det", help="Model name for --detector.")
    ap.add_argument("--det-box-thresh", type=float, default=None,
                    help="DB box score threshold; raise (e.g. 0.7-0.9) to suppress background false positives.")
    ap.add_argument("--det-unclip-ratio", type=float, default=None,
                    help="DB unclip ratio; lower (e.g. 1.3) = tighter boxes.")
    ap.add_argument("--det-thresh", type=float, default=None, help="DB pixel binarization threshold.")
    ap.add_argument("--limit", type=int, default=0, help="Images to eval (0 = all).")
    ap.add_argument("--skip-stock", action="store_true", help="Skip the 'before' stock pipeline.")
    ap.add_argument("--out", default="./fullpage_eval", help="Dir for the per-image + aggregate CSVs.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_manifest(args.manifest, args.images_root, args.limit)
    if not rows:
        raise SystemExit("no images with GT text found — check --manifest / --images-root")
    imgs = [os.path.basename(i) for i, _ in rows]
    refs = [norm(t) for _, t in rows]
    print(f"[data] {len(rows)} images\n")

    common = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)

    models = []
    if not args.skip_stock:
        models.append(("before", PaddleOCR(**common)))
    after_kw = dict(text_recognition_model_dir=args.finetuned,
                    text_recognition_model_name=args.finetuned_model_name, **common)
    if args.detector:
        after_kw["text_detection_model_dir"] = args.detector
        after_kw["text_detection_model_name"] = args.detector_name
        print(f"[after] fine-tuned detector: {args.detector}")
    else:
        print("[after] stock detector + fine-tuned recogniser")
    if args.det_box_thresh is not None:
        after_kw["text_det_box_thresh"] = args.det_box_thresh
    if args.det_unclip_ratio is not None:
        after_kw["text_det_unclip_ratio"] = args.det_unclip_ratio
    if args.det_thresh is not None:
        after_kw["text_det_thresh"] = args.det_thresh
    if any(v is not None for v in (args.det_box_thresh, args.det_unclip_ratio, args.det_thresh)):
        print(f"[after] det thresholds: box={args.det_box_thresh} unclip={args.det_unclip_ratio} thresh={args.det_thresh}")
    models.append(("after", PaddleOCR(**after_kw)))

    aggr = {}            # label -> (micro, macro)
    preds_by_label = {}  # label -> normalized preds (for the examples print)
    agg_rows = []        # rows for aggregate.csv
    for label, ocr in models:
        print(f"[{label}] running pipeline")
        preds = [norm(p) for p in run(ocr, rows)]
        preds_by_label[label] = preds
        per, micro, macro = evaluate(refs, preds)
        aggr[label] = (micro, macro)

        pi_path = os.path.join(args.out, f"{label}_per_image.csv")
        with open(pi_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "ref_chars", "ref_words", "CER", "WER", "CRR", "WRR", "ref", "pred"])
            for name, ref, pred, (c, w, crr, wrr) in zip(imgs, refs, preds, per):
                wr.writerow([name, len(ref), len(ref.split()),
                             f"{c:.4f}", f"{w:.4f}", f"{crr:.4f}", f"{wrr:.4f}", ref, pred])
        print(f"  per-image -> {pi_path}")

        for level, (C, W, CRR, WRR) in (("micro", micro), ("macro", macro)):
            agg_rows.append([label, level, f"{C:.4f}", f"{W:.4f}", f"{CRR:.4f}", f"{WRR:.4f}", len(rows)])

    if "before" in aggr and "after" in aggr:
        for li, level in enumerate(("micro", "macro")):
            b, a = aggr["before"][li], aggr["after"][li]      # (CER,WER,CRR,WRR)
            agg_rows.append(["gain", level,
                             f"{b[0] - a[0]:+.4f}", f"{b[1] - a[1]:+.4f}",   # CER/WER: down = better
                             f"{a[2] - b[2]:+.4f}", f"{a[3] - b[3]:+.4f}",   # CRR/WRR: up = better
                             len(rows)])

    agg_path = os.path.join(args.out, "aggregate.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "level", "CER", "WER", "CRR", "WRR", "n_images"])
        wr.writerows(agg_rows)

    print("\n===== whole-image CER / WER / CRR / WRR (CER/WER lower better; CRR/WRR higher better) =====")
    print(f"{'model':10s}{'level':7s}{'CER':>9s}{'WER':>9s}{'CRR':>9s}{'WRR':>9s}")
    for r in agg_rows:
        print(f"{r[0]:10s}{r[1]:7s}{r[2]:>9s}{r[3]:>9s}{r[4]:>9s}{r[5]:>9s}")
    print(f"\naggregate -> {agg_path}")
    print("=======================================================================================")

    if "after" in preds_by_label:
        print("\nexamples (image | ref[:60] | after-pred[:60]):")
        for name, ref, pred in list(zip(imgs, refs, preds_by_label["after"]))[:8]:
            print(f"  {name}")
            print(f"    ref : {ref[:60]!r}")
            print(f"    pred: {pred[:60]!r}")


if __name__ == "__main__":
    main()
