#!/usr/bin/env python3
"""eval_rec_perscript.py — PER-SCRIPT / PER-LANGUAGE recogniser eval on cropped word benchmarks.

Consumes an already-cropped PaddleOCR REC LABEL FILE (``<crop_path>\t<text>`` per line —
e.g. bharat_rec_test.txt, indicstr12_rec_test.txt) and reports CER/WER/CRR/WRR broken down
by group, via jiwer. Pass 2+ models for a side-by-side table in one run.

--group-by script   (default) : group by majority Unicode block (11 Indic scripts + Latin).
--group-by language          : group by the LANGUAGE folder in each crop path
                               (Bharat paths are ``.../Recognition/<split>/<lang>/<f>.jpg``).
                               Use this to match the Bharat Scene Text (BSTD) paper's Table 7/8,
                               which report per-LANGUAGE WRR — Assamese, Bengali, English,
                               Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi,
                               Tamil, Telugu. (Hindi vs Marathi and Assamese vs Bengali share a
                               script, so only the language folder — not Unicode — can split them.)

Two summary rows are reported: __ALL__ = micro-average (word-pooled over all crops) and
__MACRO__ = unweighted mean over the listed groups (BSTD's "Average" column is this macro mean).

    python eval_rec_perscript.py \
        --labels $SCR/rec_data_real/bharat/bharat_rec_test.txt --group-by language \
        --model synth    $SCR/PaddleOCR/output/v6_unified_infer \
        --model balanced $SCR/PaddleOCR/output/v6_rec_synthreal_infer \
        --model stage2   $SCR/PaddleOCR/output/v6_rec_stage2_infer \
        --out $SCR/rec_perlang_bharat

Add --stock to also score the vendor PP-OCRv6_medium_rec.  Needs: jiwer, paddleocr.
"""
import os
import csv
import argparse
import collections

import jiwer
from paddleocr import TextRecognition

# script name -> inclusive Unicode block(s) (majority-vote script inference; matches merge_rec_db.py)
BLOCKS = [
    ('devanagari', [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ('bengali',    [(0x0980, 0x09FF)]),
    ('gurmukhi',   [(0x0A00, 0x0A7F)]),
    ('gujarati',   [(0x0A80, 0x0AFF)]),
    ('oriya',      [(0x0B00, 0x0B7F)]),
    ('tamil',      [(0x0B80, 0x0BFF)]),
    ('telugu',     [(0x0C00, 0x0C7F)]),
    ('kannada',    [(0x0C80, 0x0CFF)]),
    ('malayalam',  [(0x0D00, 0x0D7F)]),
]

SCRIPT_ORDER = ['devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya',
                'tamil', 'telugu', 'kannada', 'malayalam', 'latin', 'other']

# BSTD Table 7/8 language columns, in their order (12 = 11 Indic + English).
LANG_ORDER = ['assamese', 'bengali', 'english', 'gujarati', 'hindi', 'kannada',
              'malayalam', 'marathi', 'odia', 'punjabi', 'tamil', 'telugu']
# any extra Bharat languages not in the BSTD table (e.g. manipuri) are appended, then 'other'.


def script_of(text):
    counts = collections.Counter()
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x0030 <= cp <= 0x0039):    # Latin letters/ext + digits
            counts['latin'] += 1; continue
        for name, ranges in BLOCKS:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1; break
    if not counts:
        return 'other'
    return counts.most_common(1)[0][0]


def lang_of(path):
    """Language = the folder that directly contains the crop (Bharat: .../<split>/<lang>/<f>.jpg)."""
    lang = os.path.basename(os.path.dirname(path)).strip().lower()
    return lang or 'other'


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def load_pairs(label_files, data_dir, group_by):
    """-> [(abs_crop_path, text, group_key)], skipping lines whose crop is missing on disk."""
    out, missing, seen = [], 0, set()
    key = lang_of if group_by == 'language' else (lambda p: None)
    for lf in label_files:
        with open(lf, encoding='utf-8') as f:
            for ln in f:
                ln = ln.rstrip('\n')
                if not ln.strip() or '\t' not in ln:
                    continue
                path, text = ln.split('\t', 1)
                text = text.strip()
                if not text:
                    continue
                p = path if os.path.isabs(path) else os.path.join(data_dir, path)
                if p in seen:
                    continue
                seen.add(p)
                if not os.path.isfile(p):
                    missing += 1; continue
                g = lang_of(p) if group_by == 'language' else script_of(text)
                out.append((p, text, g))
    if missing:
        print(f"[warn] {missing} crops referenced in labels were missing on disk (skipped)")
    return out


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


def per_group_scores(refs, preds, groups, order):
    """-> {group/'__ALL__'/'__MACRO__': dict(n, cer, wer, crr, wrr)}.
    Per-group and __ALL__ are micro (word-pooled); __MACRO__ is the unweighted mean of the
    per-group rates over `order` groups present (BSTD's "Average"). 'other' excluded from macro."""
    by = collections.defaultdict(lambda: ([], []))
    for r, p, g in zip(refs, preds, groups):
        by[g][0].append(r); by[g][1].append(p)
    out = {}
    for g, (R, P) in by.items():
        cer, wer = float(jiwer.cer(R, P)), float(jiwer.wer(R, P))
        out[g] = dict(n=len(R), cer=cer, wer=wer, crr=clamp01(1 - cer), wrr=clamp01(1 - wer))
    cer, wer = float(jiwer.cer(refs, preds)), float(jiwer.wer(refs, preds))
    out['__ALL__'] = dict(n=len(refs), cer=cer, wer=wer, crr=clamp01(1 - cer), wrr=clamp01(1 - wer))
    macro_groups = [g for g in order if g != 'other' and g in out]
    if macro_groups:
        avg = lambda k: sum(out[g][k] for g in macro_groups) / len(macro_groups)
        out['__MACRO__'] = dict(n=len(macro_groups), cer=avg('cer'), wer=avg('wer'),
                                crr=avg('crr'), wrr=avg('wrr'))
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-script / per-language recogniser eval on rec label files.")
    ap.add_argument("--labels", nargs='+', required=True,
                    help="rec label file(s): <crop_path>\\t<text> (e.g. bharat_rec_test.txt).")
    ap.add_argument("--group-by", choices=['script', 'language'], default='script',
                    help="'script' (Unicode block, default) or 'language' (crop-path folder; "
                         "matches the BSTD paper's per-language tables).")
    ap.add_argument("--model", nargs=2, action='append', metavar=('LABEL', 'MODEL_DIR'),
                    default=[], help="a recogniser to score: LABEL and its inference dir. Repeatable.")
    ap.add_argument("--stock", action='store_true', help="also score stock PP-OCRv6_medium_rec.")
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec",
                    help="model_name for the fine-tuned inference dirs (default PP-OCRv6_medium_rec).")
    ap.add_argument("--data-dir", default="/", help="base dir for RELATIVE crop paths (absolute ignored).")
    ap.add_argument("--out", default="./rec_perscript")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    models = list(args.model)
    if args.stock:
        models.append(("stock", None))
    if not models:
        raise SystemExit("no models given — pass --model LABEL DIR (repeatable) and/or --stock")

    os.makedirs(args.out, exist_ok=True)
    pairs = load_pairs(args.labels, args.data_dir, args.group_by)
    if not pairs:
        raise SystemExit("no usable (crop, text) pairs — check --labels / --data-dir")
    paths = [p for p, _, _ in pairs]
    refs = [t for _, t, _ in pairs]
    groups = [g for _, _, g in pairs]
    dist = collections.Counter(groups)

    # display order: known groups first (in canonical order), then any extras seen, then 'other'.
    base = LANG_ORDER if args.group_by == 'language' else SCRIPT_ORDER
    extras = sorted(g for g in dist if g not in base and g != 'other')
    ORDER = [g for g in base if g != 'other'] + extras + (['other'] if dist.get('other') else [])
    print(f"[data] {len(pairs)} crops | group-by {args.group_by} | "
          + ", ".join(f"{g}={dist[g]}" for g in ORDER if dist.get(g)) + "\n")

    SUMMARY = ['__ALL__', '__MACRO__']
    results = {}                                    # model_label -> per_group_scores dict
    for label, mdir in models:
        kw = dict(model_name=args.rec_name, model_dir=mdir) if mdir else dict(model_name=args.rec_name)
        print(f"[{label}] {kw}")
        try:
            preds = run(kw, paths, args.batch_size)
        except Exception as e:
            print(f"  [failed: {e}]"); continue
        results[label] = per_group_scores(refs, preds, groups, ORDER)
        with open(os.path.join(args.out, f"{label}_preds.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh); wr.writerow([args.group_by, "ref", "pred"])
            for r, p, g in zip(refs, preds, groups):
                wr.writerow([g, r, p])
        sc = results[label]
        print(f"  {args.group_by:12s}{'n':>7s}{'CER':>9s}{'WER':>9s}{'CRR':>9s}{'WRR':>9s}")
        for g in ORDER + SUMMARY:
            if g in sc:
                m = sc[g]
                print(f"  {g:12s}{m['n']:>7d}{m['cer']:>9.4f}{m['wer']:>9.4f}{m['crr']:>9.4f}{m['wrr']:>9.4f}")

    # ---- side-by-side comparison tables + CSVs (WRR is the BSTD headline; CRR alongside) ----
    if results:
        labels = [l for l, _ in models if l in results]
        ref_sc = results[labels[0]]
        for metric in ('wrr', 'crr'):
            print(f"\n===== per-{args.group_by} {metric.upper()} "
                  f"({'word' if metric == 'wrr' else 'char'} recognition, higher is better) =====")
            print(f"{args.group_by:12s}{'n':>7s}" + "".join(f"{l:>12s}" for l in labels))
            for g in ORDER + SUMMARY:
                if g not in ref_sc:
                    continue
                row = f"{g:12s}{ref_sc[g]['n']:>7d}"
                for l in labels:
                    row += f"{results[l][g][metric]:>12.4f}" if g in results[l] else f"{'-':>12s}"
                print(row)
            print("=" * (19 + 12 * len(labels)))
            comp = os.path.join(args.out, f"per_{args.group_by}_{metric}.csv")
            with open(comp, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh); wr.writerow([args.group_by, "n"] + labels)
                for g in ORDER + SUMMARY:
                    if g in ref_sc:
                        wr.writerow([g, ref_sc[g]['n']]
                                    + [f"{results[l][g][metric]:.4f}" if g in results[l] else "" for l in labels])
            print(f"-> {comp}")


if __name__ == "__main__":
    main()
