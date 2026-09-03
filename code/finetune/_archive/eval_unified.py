#!/usr/bin/env python3
"""Per-script CER/WER for the UNIFIED model: stock (before) vs fine-tuned (after).

Runs your ONE fine-tuned unified recogniser over each script's val crops, and — where a
stock PaddleOCR model exists for that script — the stock recogniser on the same crops,
then prints a per-script before/after table (+ CSV). Scripts with no stock model show
'n/a' for before (that contrast is itself the point: stock can't read them at all).

Needs: pip install jiwer
Run:
    python eval_unified.py \
        --rec-data  /ssd_scratch/cvit/mahima/rec_data \
        --finetuned /ssd_scratch/cvit/mahima/PaddleOCR/output/unified_v6_infer \
        --finetuned-model-name PP-OCRv6_medium_rec
"""
import os
import csv
import argparse

import jiwer
from paddleocr import TextRecognition

# stock PP-OCRv5 recogniser per script (None = no stock model exists for that script)
STOCK = {
    "devanagari": "devanagari_PP-OCRv5_mobile_rec",
    "tamil":      "ta_PP-OCRv5_mobile_rec",
    "telugu":     "te_PP-OCRv5_mobile_rec",
    "urdu":       "arabic_PP-OCRv5_mobile_rec",
    "latin":      "en_PP-OCRv5_mobile_rec",      # name may differ; if it errors, before = n/a
    "kannada": None, "malayalam": None, "bengali": None,
    "gujarati": None, "gurmukhi": None, "oriya": None,
}


def read_labels(path, data_dir, limit=0):
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
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return d.get("rec_text") or ""
    except Exception:
        return ""


def run(model, paths, bs):
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
    ap = argparse.ArgumentParser(description="Per-script CER/WER: stock vs unified fine-tuned.")
    ap.add_argument("--rec-data", required=True)
    ap.add_argument("--finetuned", required=True, help="Exported unified inference dir.")
    ap.add_argument("--finetuned-model-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--scripts", nargs="*", default=None, help="Subset (default: all with a _val.txt).")
    ap.add_argument("--skip-stock", action="store_true", help="Only compute the fine-tuned (after) numbers.")
    ap.add_argument("--limit", type=int, default=0, help="Crops per script (0 = all; use e.g. 300 for a quick pass).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-csv", default="unified_eval.csv")
    args = ap.parse_args()

    rec = os.path.abspath(args.rec_data)
    scripts = args.scripts or [s for s in STOCK if os.path.isfile(os.path.join(rec, f"{s}_val.txt"))]

    print(f"[load] fine-tuned unified model: {args.finetuned}  (as {args.finetuned_model_name})")
    ft = TextRecognition(model_name=args.finetuned_model_name, model_dir=args.finetuned)

    table = []
    for s in scripts:
        lp = os.path.join(rec, f"{s}_val.txt")
        if not os.path.isfile(lp):
            print(f"[skip] no {lp}")
            continue
        rows = read_labels(lp, rec, args.limit)
        paths = [p for p, _ in rows]
        refs = [t for _, t in rows]
        print(f"\n[{s}] {len(rows)} crops — fine-tuned:")
        after = run(ft, paths, args.batch_size)
        a_cer, a_wer = float(jiwer.cer(refs, after)), float(jiwer.wer(refs, after))

        b_cer = b_wer = None
        if not args.skip_stock and STOCK.get(s):
            print(f"[{s}] stock ({STOCK[s]}):")
            try:
                stock = TextRecognition(model_name=STOCK[s])
                before = run(stock, paths, args.batch_size)
                b_cer, b_wer = float(jiwer.cer(refs, before)), float(jiwer.wer(refs, before))
            except Exception as e:
                print(f"  [stock unavailable for {s}: {e}]")
        table.append((s, len(rows), b_cer, b_wer, a_cer, a_wer))

    def f(x):
        return " n/a " if x is None else f"{x:.3f}"

    print("\n============== UNIFIED model: per-script CER / WER (lower is better) ==============")
    print(f"{'script':12s}{'n':>6s}{'CER_before':>12s}{'WER_before':>12s}{'CER_after':>11s}{'WER_after':>11s}")
    for s, n, bc, bw, ac, aw in table:
        print(f"{s:12s}{n:>6d}{f(bc):>12s}{f(bw):>12s}{f(ac):>11s}{f(aw):>11s}")
    print("==================================================================================")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["script", "n", "cer_before", "wer_before", "cer_after", "wer_after"])
        w.writerows(table)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
