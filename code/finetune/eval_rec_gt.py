#!/usr/bin/env python3
"""eval_rec_gt.py -- recogniser-only evaluation on GROUND-TRUTH word boxes (jiwer).

No detector involved: the recogniser is fed perfect GT crops, so this measures pure
recognition quality (the "rec ceiling"). Reports CER / WER / CRR / WRR via jiwer, both
micro (word-pooled) and macro (unweighted mean over groups), for one or more recognisers
side by side -- paper-ready per-language / per-script tables.

INPUTS  (either or both; repeat the flag for several sources)
  --labels FILE       a PaddleOCR rec label file, '<crop_path>\\t<text>' per line, crops
                      already on disk        (e.g. Bharat  bharat_rec_test.txt).
  --images-dir DIR    a folder of images + matching GT JSON  {"words":[{x,y,w,h,text}, ...]};
                      the GT boxes are cropped on the fly   (e.g. real_hindi / real_marathi).

GROUPING  (--group-by)
  language (default)  labels : the language folder in the crop path
                      (Bharat: .../<split>/<lang>/<file>.jpg) -> per-language rows that line up
                      with the BSTD paper's Table 7.  images-dir : the dataset folder name
                      (real_hindi -> 'real_hindi'), kept distinct from Bharat's own hindi row.
  script              majority Unicode block (11 Indic scripts + Latin) for every crop.
  none                one row per source (its folder / file name) -- use for a single-language set.

METRICS  (identical jiwer convention to the paper's per-script table)
  CER, WER via jiwer ;  CRR = clamp01(1 - CER) ;  WRR = clamp01(1 - WER)
  per group + __ALL__ (micro, word-pooled) + __MACRO__ (unweighted mean over groups = BSTD 'Average').
  Word substitution/deletion/insertion/hit counts (jiwer) go into the CSVs for error analysis.

OUTPUTS  (under --out)
  per_<groupby>_wrr.csv, per_<groupby>_crr.csv   models side by side (the paper tables)
  <model>_by_group.csv                           full metrics + S/D/I/H per group, per model
  <model>_preds.csv                              every (group, ref, pred) -- error analysis

Needs: jiwer, paddleocr, pillow.

EXAMPLES
  # Bharat GT crops, per language, stage-2 B vs A' vs stock
  python eval_rec_gt.py --group-by language \\
      --labels $SCR/rec_data_real/bharat/bharat_rec_test.txt \\
      --model stage2B $SCR/PaddleOCR/output/v6_rec_stage2_B_infer \\
      --model stage2A $SCR/PaddleOCR/output/v6_rec_stage2_infer \\
      --stock --out $SCR/rec_gt_bharat

  # Real Hindi + Real Marathi GT boxes (one pooled row each)
  python eval_rec_gt.py --group-by none \\
      --images-dir $SCR/real_hindi --images-dir $SCR/real_marathi \\
      --model stage2B $SCR/PaddleOCR/output/v6_rec_stage2_B_infer \\
      --model stage2A $SCR/PaddleOCR/output/v6_rec_stage2_infer \\
      --out $SCR/rec_gt_real
"""
import os
import csv
import glob
import json
import argparse
import collections

import jiwer
from paddleocr import TextRecognition

# script name -> inclusive Unicode block(s); majority vote (matches merge_rec_db.py / eval_rec_perscript.py)
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
SCRIPT_ORDER = ['devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya', 'tamil',
                'telugu', 'kannada', 'malayalam', 'latin', 'other']
# BSTD Table 7/8 language columns, in their order (12 = 11 Indic + English).
LANG_ORDER = ['assamese', 'bengali', 'english', 'gujarati', 'hindi', 'kannada',
              'malayalam', 'marathi', 'odia', 'punjabi', 'tamil', 'telugu']


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def script_of(text):
    counts = collections.Counter()
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x0030 <= cp <= 0x0039):   # Latin letters/ext + digits
            counts['latin'] += 1
            continue
        for name, ranges in BLOCKS:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else 'other'


def lang_of(path):
    """Language = the folder that directly contains the crop (Bharat: .../<split>/<lang>/<f>.jpg)."""
    return os.path.basename(os.path.dirname(path)).strip().lower() or 'other'


def crops_from_labels(label_file, data_dir, group_by, skip, seen):
    """Read '<crop_path>\\t<text>' lines -> [(abs_crop_path, text, group)]; skip missing/ignored."""
    src = os.path.splitext(os.path.basename(label_file))[0]
    out, missing = [], 0
    with open(label_file, encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if '\t' not in ln:
                continue
            path, text = ln.split('\t', 1)
            text = text.strip()
            if not text or text in skip:
                continue
            p = path if os.path.isabs(path) else os.path.join(data_dir, path)
            if p in seen:
                continue
            seen.add(p)
            if not os.path.isfile(p):
                missing += 1
                continue
            g = lang_of(p) if group_by == 'language' else (script_of(text) if group_by == 'script' else src)
            out.append((p, text, g))
    if missing:
        print(f"[warn] {src}: {missing} crops referenced in labels were missing on disk (skipped)")
    return out


def crops_from_images(images_dir, out_crops, group_by, skip, min_size):
    """Crop every GT word box from images + JSON -> [(crop_path, text, group)]. group = folder name
    (kept distinct from Bharat's per-language rows) unless --group-by script."""
    from PIL import Image
    os.makedirs(out_crops, exist_ok=True)
    tag = os.path.basename(os.path.normpath(images_dir))
    out = []
    for jp in sorted(glob.glob(os.path.join(images_dir, '*.json'))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        img_path = next((os.path.join(images_dir, f"{stem}.{e}")
                         for e in ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
                         if os.path.isfile(os.path.join(images_dir, f"{stem}.{e}"))), None)
        if img_path is None:
            print(f"[skip] no image for {jp}")
            continue
        try:
            data = json.load(open(jp, encoding='utf-8'))
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[skip] {stem}: {e}")
            continue
        W, H = img.size
        for i, w in enumerate(data.get('words', [])):
            text = str(w.get('text', '')).strip()
            if not text or text in skip:
                continue
            try:
                x, y, ww, hh = int(w['x']), int(w['y']), int(w['w']), int(w['h'])
            except (KeyError, ValueError, TypeError):
                continue
            if ww < min_size or hh < min_size:
                continue
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + ww), min(H, y + hh)
            if x1 <= x0 or y1 <= y0:
                continue
            cp = os.path.join(out_crops, f"{tag}__{stem}_{i}.jpg")
            img.crop((x0, y0, x1, y1)).save(cp)
            g = script_of(text) if group_by == 'script' else tag
            out.append((cp, text, g))
    return out


def rec_text(res):
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return d.get("rec_text") or ""
    except Exception:
        return ""


def predict(kwargs, paths, bs):
    model = TextRecognition(**kwargs)
    preds = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        for res in model.predict(input=chunk, batch_size=len(chunk)):
            preds.append(rec_text(res))
        print(f"    {min(i + bs, len(paths))}/{len(paths)}", end="\r")
    print()
    if len(preds) != len(paths):
        raise RuntimeError(f"{len(preds)} preds for {len(paths)} crops -- result format differs")
    return preds


def word_stats(refs, preds):
    """jiwer word-level WER + S/D/I/H, across jiwer 2.x / 3.x."""
    try:
        o = jiwer.process_words(refs, preds)                       # jiwer >= 3
        return float(o.wer), o.substitutions, o.deletions, o.insertions, o.hits
    except AttributeError:
        m = jiwer.compute_measures(refs, preds)                    # jiwer 2.x
        return float(m['wer']), m['substitutions'], m['deletions'], m['insertions'], m['hits']


def score(refs, preds):
    cer = float(jiwer.cer(refs, preds))
    wer, S, D, I, H = word_stats(refs, preds)
    return dict(n=len(refs), cer=cer, wer=wer, crr=clamp01(1 - cer), wrr=clamp01(1 - wer),
                S=S, D=D, I=I, H=H)


def per_group_scores(refs, preds, groups, order):
    """{group / '__ALL__' / '__MACRO__': metrics}. Group + __ALL__ are micro (word-pooled);
    __MACRO__ is the unweighted mean of the per-group rates over `order` ('other' excluded)."""
    by = collections.defaultdict(lambda: ([], []))
    for r, p, g in zip(refs, preds, groups):
        by[g][0].append(r)
        by[g][1].append(p)
    out = {g: score(R, P) for g, (R, P) in by.items()}
    out['__ALL__'] = score(refs, preds)
    macro_groups = [g for g in order if g != 'other' and g in out]
    if macro_groups:
        avg = lambda k: sum(out[g][k] for g in macro_groups) / len(macro_groups)
        out['__MACRO__'] = dict(n=len(macro_groups), cer=avg('cer'), wer=avg('wer'),
                                crr=avg('crr'), wrr=avg('wrr'), S='', D='', I='', H='')
    return out


def main():
    ap = argparse.ArgumentParser(description="Recogniser-only eval on GT word boxes (jiwer).")
    ap.add_argument('--labels', action='append', default=[],
                    help="rec label file '<crop_path>\\t<text>' (e.g. bharat_rec_test.txt). Repeatable.")
    ap.add_argument('--images-dir', action='append', default=[],
                    help="folder of images + matching GT JSON (e.g. real_hindi). Repeatable.")
    ap.add_argument('--group-by', choices=['language', 'script', 'none'], default='language')
    ap.add_argument('--model', nargs=2, action='append', metavar=('LABEL', 'DIR'), default=[],
                    help="a recogniser to score: LABEL and its inference dir. Repeatable.")
    ap.add_argument('--stock', action='store_true', help="also score stock PP-OCRv6_medium_rec.")
    ap.add_argument('--rec-name', default='PP-OCRv6_medium_rec',
                    help="model_name for the fine-tuned inference dirs (default PP-OCRv6_medium_rec).")
    ap.add_argument('--data-dir', default='/', help="base dir for RELATIVE label crop paths.")
    ap.add_argument('--skip-text', nargs='*', default=['NA', '###'],
                    help="GT texts to ignore (placeholder / illegible). Default: NA ###.")
    ap.add_argument('--min-size', type=int, default=6, help="drop GT boxes smaller than this (images-dir).")
    ap.add_argument('--out', default='./rec_gt_eval')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    models = list(args.model)
    if args.stock:
        models.append(("stock", None))
    if not models:
        raise SystemExit("no models -- pass --model LABEL DIR (repeatable) and/or --stock")
    if not args.labels and not args.images_dir:
        raise SystemExit("no inputs -- pass --labels FILE and/or --images-dir DIR")

    os.makedirs(args.out, exist_ok=True)
    skip, seen, crops = set(args.skip_text), set(), []
    for lf in args.labels:
        crops += crops_from_labels(lf, args.data_dir, args.group_by, skip, seen)
    for d in args.images_dir:
        crops += crops_from_images(d, os.path.join(args.out, 'crops'), args.group_by, skip, args.min_size)
    if not crops:
        raise SystemExit("no usable (crop, text) pairs -- check --labels / --images-dir")

    paths = [c[0] for c in crops]
    refs = [c[1] for c in crops]
    groups = [c[2] for c in crops]
    dist = collections.Counter(groups)

    base = LANG_ORDER if args.group_by == 'language' else (SCRIPT_ORDER if args.group_by == 'script' else [])
    extras = sorted(g for g in dist if g not in base and g != 'other')
    ORDER = [g for g in base if g != 'other' and dist.get(g)] + extras + (['other'] if dist.get('other') else [])
    print(f"[data] {len(crops)} GT crops | group-by {args.group_by} | "
          + ", ".join(f"{g}={dist[g]}" for g in ORDER) + "\n")

    SUMMARY = ['__ALL__', '__MACRO__']
    results = {}                                    # model_label -> per_group_scores
    for label, mdir in models:
        kw = dict(model_name=args.rec_name, model_dir=mdir) if mdir else dict(model_name=args.rec_name)
        print(f"[{label}] {kw}")
        try:
            preds = predict(kw, paths, args.batch_size)
        except Exception as e:
            print(f"  [failed: {e}]")
            continue
        sc = per_group_scores(refs, preds, groups, ORDER)
        results[label] = sc

        with open(os.path.join(args.out, f"{label}_preds.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["group", "ref", "pred"])
            for r, p, g in zip(refs, preds, groups):
                wr.writerow([g, r, p])
        with open(os.path.join(args.out, f"{label}_by_group.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["group", "n", "CER", "WER", "CRR", "WRR", "S", "D", "I", "H"])
            for g in ORDER + SUMMARY:
                if g in sc:
                    m = sc[g]
                    wr.writerow([g, m['n'], f"{m['cer']:.4f}", f"{m['wer']:.4f}",
                                 f"{m['crr']:.4f}", f"{m['wrr']:.4f}", m['S'], m['D'], m['I'], m['H']])

        print(f"  {'group':12s}{'n':>7s}{'CER':>9s}{'WER':>9s}{'CRR':>9s}{'WRR':>9s}")
        for g in ORDER + SUMMARY:
            if g in sc:
                m = sc[g]
                print(f"  {g:12s}{m['n']:>7d}{m['cer']:>9.4f}{m['wer']:>9.4f}{m['crr']:>9.4f}{m['wrr']:>9.4f}")

    # ---- side-by-side comparison tables + CSVs (WRR headline, CRR alongside) ----
    if results:
        labels = [l for l, _ in models if l in results]
        ref_sc = results[labels[0]]
        for metric in ('wrr', 'crr'):
            print(f"\n===== per-{args.group_by} {metric.upper()} "
                  f"({'word' if metric == 'wrr' else 'char'} recognition, higher is better) =====")
            print(f"{'group':12s}{'n':>7s}" + "".join(f"{l:>12s}" for l in labels))
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
                wr = csv.writer(fh)
                wr.writerow([args.group_by, "n"] + labels)
                for g in ORDER + SUMMARY:
                    if g in ref_sc:
                        wr.writerow([g, ref_sc[g]['n']]
                                    + [f"{results[l][g][metric]:.4f}" if g in results[l] else "" for l in labels])
            print(f"-> {comp}")


if __name__ == "__main__":
    main()
