#!/usr/bin/env python3
"""Whole-image CER/WER/CRR/WRR for IndicPhotoOCR (IIT-J) end-to-end.

Scored IDENTICALLY to eval_fullpage.py (same norm + edit-distance CER/WER, CRR=1-CER,
WRR=1-WER clamped, micro+macro, same CSV layout) so its numbers drop straight into the
same comparison table as our PP-OCR pipeline.

IndicPhotoOCR returns text already in reading order:
    [["राजीव","चौक","मेट्रो","स्टेशन"], ["Rajiv","Chowk","Metro","Station"]]
-> joined to one whole-image string per image, compared to the same manifest GT.

Needs: IndicPhotoOCR installed (its own conda env) + jiwer.
Run (in the indicphotoocr env):
    python eval_indicphotoocr.py \
        --manifest    /ssd_scratch/cvit/mahima/real_marathi/manifest_real.jsonl \
        --images-root /ssd_scratch/cvit/mahima/real_marathi \
        --out ./ipocr_real_marathi --device cuda:0 --identifier-lang auto --batch-size 32
"""
import os
import csv
import json
import argparse

import jiwer
from IndicPhotoOCR.ocr import OCR


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


def to_text(result):
    """IndicPhotoOCR output -> one whole-image string (lines top->bottom, words L->R).
    Defensive across versions: list-of-lists-of-words, list of dicts, or a flat list."""
    lines = []
    if isinstance(result, list):
        for ln in result:
            if isinstance(ln, (list, tuple)):
                words = [str(w) for w in ln if isinstance(w, str)]
                if not words:
                    words = [str(w.get("text", "")) for w in ln if isinstance(w, dict)]
                lines.append(" ".join(w for w in words if w))
            elif isinstance(ln, dict):
                lines.append(str(ln.get("text", "")))
            elif isinstance(ln, str):
                lines.append(ln)
    elif isinstance(result, dict):
        lines.append(str(result.get("text", "")))
    return "\n".join(l for l in lines if l)


def norm(s):
    return " ".join(str(s).split())


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_pair(ref, pred):
    c = float(jiwer.cer(ref, pred))
    w = float(jiwer.wer(ref, pred))
    return c, w, clamp01(1.0 - c), clamp01(1.0 - w)


def evaluate(refs, preds):
    per = [score_pair(r, p) for r, p in zip(refs, preds)]
    mCER = float(jiwer.cer(refs, preds))
    mWER = float(jiwer.wer(refs, preds))
    micro = (mCER, mWER, clamp01(1.0 - mCER), clamp01(1.0 - mWER))
    macro = tuple(sum(col) / len(per) for col in zip(*per)) if per else (0.0, 0.0, 0.0, 0.0)
    return per, micro, macro


def main():
    ap = argparse.ArgumentParser(description="IndicPhotoOCR end-to-end whole-image CER/WER/CRR/WRR.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", default="./ipocr_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--identifier-lang", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_manifest(args.manifest, args.images_root, args.limit)
    if not rows:
        raise SystemExit("no images with GT text found — check --manifest / --images-root")
    imgs = [os.path.basename(i) for i, _ in rows]
    refs = [norm(t) for _, t in rows]
    print(f"[data] {len(rows)} images")

    ocr = OCR(device=args.device, identifier_lang=args.identifier_lang, verbose=False)
    preds = []
    for k, (img, _) in enumerate(rows):
        try:
            res = ocr.ocr(img, batch_size=args.batch_size)
        except TypeError:
            res = ocr.ocr(img)                      # older API without batch_size
        except Exception as e:
            print(f"\n  [warn] {imgs[k]}: {e}")
            res = []
        preds.append(norm(to_text(res)))
        print(f"    {k + 1}/{len(rows)}", end="\r")
    print()

    per, micro, macro = evaluate(refs, preds)

    pi = os.path.join(args.out, "indicphotoocr_per_image.csv")
    with open(pi, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "ref_chars", "ref_words", "CER", "WER", "CRR", "WRR", "ref", "pred"])
        for name, ref, pred, (c, w, crr, wrr) in zip(imgs, refs, preds, per):
            wr.writerow([name, len(ref), len(ref.split()),
                         f"{c:.4f}", f"{w:.4f}", f"{crr:.4f}", f"{wrr:.4f}", ref, pred])

    agg = os.path.join(args.out, "aggregate.csv")
    with open(agg, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "level", "CER", "WER", "CRR", "WRR", "n_images"])
        for level, (C, W, CRR, WRR) in (("micro", micro), ("macro", macro)):
            wr.writerow(["indicphotoocr", level, f"{C:.4f}", f"{W:.4f}", f"{CRR:.4f}", f"{WRR:.4f}", len(rows)])

    print(f"per-image -> {pi}\naggregate -> {agg}")
    print(f"\nmicro  CER {micro[0]:.4f}  WER {micro[1]:.4f}  CRR {micro[2]:.4f}  WRR {micro[3]:.4f}")
    print(f"macro  CER {macro[0]:.4f}  WER {macro[1]:.4f}  CRR {macro[2]:.4f}  WRR {macro[3]:.4f}")


if __name__ == "__main__":
    main()
