#!/usr/bin/env python3
"""CER / WER on word crops BEFORE vs AFTER fine-tuning.

Runs the stock PP-OCRv5 recogniser and your fine-tuned exported model over the SAME
val crops and prints an aggregate CER/WER table, plus a few example predictions.

These are pre-cropped words, so we use the recognition module directly (no detector).

Needs: pip install jiwer

Run (in the paddle env, after tools/export_model.py):
    python compare_baseline.py \
        --rec-data  /ssd_scratch/cvit/mahima/rec_data \
        --labels    /ssd_scratch/cvit/mahima/rec_data/devanagari_val.txt \
        --finetuned /ssd_scratch/cvit/mahima/PaddleOCR/output/devanagari_infer
"""
import os
import argparse

import jiwer
from paddleocr import TextRecognition


def read_labels(path, data_dir, limit=0):
    """PaddleOCR rec label file: <relative image path>\\t<text>"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rel, _, text = line.partition("\t")
            if text.strip():
                rows.append((os.path.join(data_dir, rel), text))
    return rows[:limit] if limit else rows


def rec_text(res):
    """Pull the recognised string out of a TextRecognition result."""
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return (d.get("rec_text") or "")
    except Exception:
        return ""


def run(model, paths, batch_size):
    preds = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        for res in model.predict(input=chunk, batch_size=len(chunk)):
            preds.append(rec_text(res))
        print(f"  {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    # guard against a length mismatch so zip/jiwer don't silently misalign
    if len(preds) != len(paths):
        raise SystemExit(f"got {len(preds)} predictions for {len(paths)} images — "
                         f"the result format may differ; inspect res.json")
    return preds


def main():
    ap = argparse.ArgumentParser(description="CER/WER before vs after fine-tuning.")
    ap.add_argument("--rec-data", required=True, help="data_dir the label paths are relative to.")
    ap.add_argument("--labels", required=True, help="e.g. devanagari_val.txt")
    ap.add_argument("--finetuned", required=True, help="Exported inference dir (devanagari_infer).")
    ap.add_argument("--stock-model", default="devanagari_PP-OCRv5_mobile_rec",
                    help="Stock model name used as the BEFORE baseline.")
    ap.add_argument("--finetuned-model-name", default="PP-OCRv5_server_rec",
                    help="Base architecture of the fine-tuned model — must match the name in its "
                         "inference.yml (else PaddleOCR errors 'Model name mismatch'). "
                         "Use PP-OCRv6_medium_rec for a v6 fine-tune.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all crops.")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rows = read_labels(args.labels, args.rec_data, args.limit)
    paths = [p for p, _ in rows]
    refs = [t for _, t in rows]
    print(f"[data] {len(rows)} crops from {args.labels}\n")

    print(f"[before] stock      : {args.stock_model}")
    before = run(TextRecognition(model_name=args.stock_model), paths, args.batch_size)

    print(f"[after]  fine-tuned : {args.finetuned}  (as {args.finetuned_model_name})")
    after = run(TextRecognition(model_name=args.finetuned_model_name, model_dir=args.finetuned),
                paths, args.batch_size)

    b_cer, b_wer = float(jiwer.cer(refs, before)), float(jiwer.wer(refs, before))
    a_cer, a_wer = float(jiwer.cer(refs, after)), float(jiwer.wer(refs, after))

    print("\n=========== Devanagari: before vs after fine-tuning ===========")
    print(f"{'':14s}{'CER':>10s}{'WER':>10s}     (lower is better)")
    print(f"{'before':14s}{b_cer:>10.4f}{b_wer:>10.4f}")
    print(f"{'after':14s}{a_cer:>10.4f}{a_wer:>10.4f}")
    print(f"{'absolute gain':14s}{b_cer - a_cer:>10.4f}{b_wer - a_wer:>10.4f}")
    if b_cer > 0:
        print(f"{'relative gain':14s}{100 * (b_cer - a_cer) / b_cer:>9.1f}%"
              f"{100 * (b_wer - a_wer) / max(b_wer, 1e-9):>9.1f}%")
    print("===============================================================")

    print("\nexamples  (reference | before | after):")
    for r, b, a in list(zip(refs, before, after))[:15]:
        print(f"  {r!r:28s} | {b!r:28s} | {a!r}")


if __name__ == "__main__":
    main()
