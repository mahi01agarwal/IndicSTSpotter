#!/usr/bin/env python3
"""verify_rec_bharat.py -- INDEPENDENT re-check of recogniser WRR/CRR on Bharat GT crops.

Why this exists
---------------
The committed per-script numbers (paper_tab_rec_perscript.tex) come from eval_rec_gt.py, which
uses jiwer. This script re-derives the SAME quantities with a *completely separate* metric
implementation (own Levenshtein + exact-match WRR -- NO jiwer), so a metric bug cannot hide in
both. It ALSO probes the classic Indic gotcha: Unicode normalisation. GT and prediction can be
canonically equal but byte-different (NFC vs NFD; invisible ZWNJ/ZWJ joiners). If the committed
eval did not normalise, exact-match WRR is depressed relative to a system (e.g. IndicPhotoOCR)
that did -- which would make us look worse than we are. This tells you which it is.

What it reports
---------------
WRR (exact match) and CRR (1 - charEditDist/refChars) per majority-Unicode script -- the 10 BSTD
columns -- with macro (unweighted mean = BSTD 'Average') and micro (__ALL__), under three
normalisations:
    strip      whitespace-normalise only  (~ jiwer default -> should reproduce the committed number)
    nfc        + Unicode NFC              (the fair Indic default)
    nfc_clean  + strip ZWNJ/ZWJ           (upper bound; treats invisible joiners as equal)
Plus a side-by-side delta vs IndicPhotoOCR (+oracle TD), a "normalisation lift" line, and a
mismatch dump for eyeballing.

Input (pick ONE path)
---------------------
  A) run the recogniser yourself:
       --labels FILE ...      PaddleOCR rec label file '<crop_path>\\t<text>'
       --model-dir DIR        exported inference dir (e.g. output/v6_rec_stage2_infer)
  B) fast, no GPU / no paddle -- recompute from an existing dump:
       --preds CSV ...        a *_preds.csv from eval_rec_gt.py (needs columns 'ref','pred')

Grouping is ALWAYS by majority Unicode block of the GT (same BLOCKS as eval_rec_gt.py), so it does
not depend on how the crops/preds were originally bucketed.

Needs (mode A only): paddleocr. Mode B is pure stdlib.
"""
import os
import re
import csv
import argparse
import unicodedata
import collections

# ---- script buckets: identical to eval_rec_gt.py so grouping never differs -------------------
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
SCRIPTS_10 = ['devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya',
              'tamil', 'telugu', 'kannada', 'malayalam', 'latin']
DISPLAY = {'devanagari': 'Devanagari', 'bengali': 'Bengali', 'gurmukhi': 'Punjabi',
           'gujarati': 'Gujarati', 'oriya': 'Odia', 'tamil': 'Tamil', 'telugu': 'Telugu',
           'kannada': 'Kannada', 'malayalam': 'Malayalam', 'latin': 'English', 'other': 'other'}
# IndicPhotoOCR (+oracle TD) reference row from their BSTD-Test table, keyed by OUR script name
# (their English->latin, Odia->oriya, Punjabi->gurmukhi).
IPOCR_WRR = {'bengali': 0.75, 'latin': 0.74, 'gujarati': 0.54, 'devanagari': 0.63, 'kannada': 0.49,
             'malayalam': 0.54, 'oriya': 0.68, 'gurmukhi': 0.70, 'tamil': 0.69, 'telugu': 0.49}
IPOCR_CRR = {'bengali': 0.87, 'latin': 0.78, 'gujarati': 0.64, 'devanagari': 0.74, 'kannada': 0.68,
             'malayalam': 0.80, 'oriya': 0.84, 'gurmukhi': 0.84, 'tamil': 0.78, 'telugu': 0.73}
IPOCR_MACRO_WRR, IPOCR_MACRO_CRR = 0.62, 0.77
ZW = {'‌', '‍'}                    # ZWNJ, ZWJ
MODES = ['strip', 'nfc', 'nfc_clean']


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


def normstr(s, mode):
    if mode == 'raw':
        return s
    s = re.sub(r'\s+', ' ', s).strip()                 # mirror jiwer RemoveMultipleSpaces + Strip
    if mode == 'strip':
        return s
    s = unicodedata.normalize('NFC', s)
    if mode == 'nfc':
        return s
    if mode == 'nfc_clean':
        return ''.join(c for c in s if c not in ZW)
    return s


def edist(a, b):
    """Character-level Levenshtein (own impl -- deliberately not jiwer)."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if not m:
        return n
    if not n:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[n]


def group_metrics(triples, mode):
    """triples: [(ref, pred, script)]. -> {script/__ALL__/__MACRO__: (n, WRR, CRR)}.
    WRR = exact-match rate; CRR = 1 - sum(charEditDist)/sum(refChars) (corpus-level, BSTD eq.)."""
    buck = collections.defaultdict(list)
    for r, p, g in triples:
        buck[g].append((normstr(r, mode), normstr(p, mode)))

    def one(lst):
        n = len(lst)
        exact = sum(r == p for r, p in lst)
        ed = sum(edist(r, p) for r, p in lst)
        nc = sum(len(r) for r, p in lst)
        return n, exact / n, (clamp01(1 - ed / nc) if nc else 0.0)

    out = {g: one(lst) for g, lst in buck.items()}
    out['__ALL__'] = one([(normstr(r, mode), normstr(p, mode)) for r, p, _ in triples])
    present = [g for g in SCRIPTS_10 if g in out]
    if present:
        out['__MACRO__'] = (len(present),
                            sum(out[g][1] for g in present) / len(present),
                            sum(out[g][2] for g in present) / len(present))
    return out


# ---- input loaders ---------------------------------------------------------------------------
def load_labels(files, data_dir, skip):
    """[(abs_crop_path, ref_text)] from PaddleOCR rec label files; skip missing/placeholder."""
    pairs, seen, missing = [], set(), 0
    for lf in files:
        with open(lf, encoding='utf-8') as f:
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
                pairs.append((p, text))
    if missing:
        print(f"[warn] {missing} crops referenced in labels were missing on disk (skipped)")
    return pairs


def load_preds(files, skip):
    """[(ref, pred)] from eval_rec_gt.py *_preds.csv (columns 'ref','pred')."""
    pairs = []
    for cf in files:
        with open(cf, encoding='utf-8') as f:
            rd = csv.DictReader(f)
            if 'ref' not in rd.fieldnames or 'pred' not in rd.fieldnames:
                raise SystemExit(f"{cf}: expected columns 'ref','pred', got {rd.fieldnames}")
            for row in rd:
                r = (row.get('ref') or '').strip()
                p = row.get('pred') or ''
                if not r or r in skip:
                    continue
                pairs.append((r, p))
    return pairs


def rec_text(res):
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return d.get("rec_text") or ""
    except Exception:
        return ""


def run_rec(model_name, model_dir, paths, bs):
    from paddleocr import TextRecognition
    kw = dict(model_name=model_name, model_dir=model_dir) if model_dir else dict(model_name=model_name)
    print(f"[rec] {kw}")
    model = TextRecognition(**kw)
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


# ---- reporting -------------------------------------------------------------------------------
def print_table(title, m):
    print(f"\n{title}")
    print(f"  {'script':11s}{'N':>7s}{'WRR':>9s}{'CRR':>9s}")
    for g in SCRIPTS_10 + ['other', '__ALL__', '__MACRO__']:
        if g in m:
            n, w, c = m[g]
            print(f"  {DISPLAY.get(g, g):11s}{n:>7d}{w:>9.4f}{c:>9.4f}")


def main():
    ap = argparse.ArgumentParser(description="Independent WRR/CRR re-check on Bharat GT crops.")
    ap.add_argument('--labels', action='append', default=[], help="rec label file(s) (mode A). Repeatable.")
    ap.add_argument('--model-dir', default=None, help="exported inference dir, e.g. output/v6_rec_stage2_infer (mode A).")
    ap.add_argument('--rec-name', default='PP-OCRv6_medium_rec', help="model_name for TextRecognition.")
    ap.add_argument('--preds', action='append', default=[], help="*_preds.csv from eval_rec_gt.py (mode B). Repeatable.")
    ap.add_argument('--data-dir', default='/', help="base dir for RELATIVE label crop paths.")
    ap.add_argument('--skip-text', nargs='*', default=['NA', '###'], help="GT texts to ignore.")
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--dump', type=int, default=12, help="mismatch examples per script to dump (0=off).")
    ap.add_argument('--out', default='./rec_verify')
    args = ap.parse_args()

    if not args.preds and not (args.labels and args.model_dir):
        raise SystemExit("give EITHER --preds CSV ...  OR  --labels FILE ... --model-dir DIR")
    os.makedirs(args.out, exist_ok=True)
    skip = set(args.skip_text)

    # ---- build (ref, pred, script) triples --------------------------------------------------
    if args.preds:
        pairs = load_preds(args.preds, skip)
        triples = [(r, p, script_of(r)) for r, p in pairs]
        src = "preds-csv"
    else:
        lab = load_labels(args.labels, args.data_dir, skip)
        if not lab:
            raise SystemExit("no usable crops from --labels")
        preds = run_rec(args.rec_name, args.model_dir, [p for p, _ in lab], args.batch_size)
        triples = [(ref, pred, script_of(ref)) for (_, ref), pred in zip(lab, preds)]
        src = os.path.basename(args.model_dir.rstrip('/'))

    dist = collections.Counter(g for _, _, g in triples)
    print(f"\n[data] {len(triples)} crops from {src} | "
          + ", ".join(f"{DISPLAY.get(g, g)}={dist[g]}" for g in SCRIPTS_10 + ['other'] if dist.get(g)))

    # ---- metrics under each normalisation ---------------------------------------------------
    M = {mode: group_metrics(triples, mode) for mode in MODES}
    for mode in MODES:
        tag = {'strip': "strip  (~ jiwer / committed number)", 'nfc': "nfc  (fair Indic default)",
               'nfc_clean': "nfc_clean  (+ strip ZWNJ/ZWJ)"}[mode]
        print_table(f"===== WRR / CRR  --  {tag} =====", M[mode])

    # ---- macro/micro summary across normalisations ------------------------------------------
    print("\n===== macro / micro summary (macro = unweighted mean over 10 scripts = BSTD 'Average') =====")
    print(f"  {'mode':11s}{'macroWRR':>10s}{'macroCRR':>10s}{'microWRR':>10s}{'microCRR':>10s}")
    for mode in MODES:
        ma, mi = M[mode]['__MACRO__'], M[mode]['__ALL__']
        print(f"  {mode:11s}{ma[1]:>10.4f}{ma[2]:>10.4f}{mi[1]:>10.4f}{mi[2]:>10.4f}")

    # ---- normalisation lift (the "are we really worse?" line) -------------------------------
    strip_wrong = lift_nfc = lift_clean = 0
    for r, p, _ in triples:
        if normstr(r, 'strip') != normstr(p, 'strip'):
            strip_wrong += 1
            lift_nfc += (normstr(r, 'nfc') == normstr(p, 'nfc'))
            lift_clean += (normstr(r, 'nfc_clean') == normstr(p, 'nfc_clean'))
    print(f"\n[normalisation lift] of {strip_wrong} words wrong under 'strip': "
          f"{lift_nfc} become correct under NFC (+{lift_nfc / len(triples) * 100:.1f} WRR pts overall), "
          f"{lift_clean} under NFC+ZWJ-strip.")

    # ---- head-to-head vs IndicPhotoOCR (+oracle TD), NFC ------------------------------------
    m = M['nfc']
    print("\n===== ours (NFC) vs IndicPhotoOCR (+oracle TD) -- both oracle detection =====")
    print(f"  {'script':11s}{'ourWRR':>8s}{'ipoWRR':>8s}{'dWRR':>7s}   {'ourCRR':>8s}{'ipoCRR':>8s}{'dCRR':>7s}")
    winW = winC = tot = 0
    for g in SCRIPTS_10:
        if g not in m:
            continue
        ow, oc = m[g][1], m[g][2]
        iw, ic = IPOCR_WRR[g], IPOCR_CRR[g]
        winW += ow >= iw
        winC += oc >= ic
        tot += 1
        print(f"  {DISPLAY[g]:11s}{ow:>8.2f}{iw:>8.2f}{ow - iw:>+7.2f}   {oc:>8.2f}{ic:>8.2f}{oc - ic:>+7.2f}")
    ma = m['__MACRO__']
    print(f"  {'MACRO':11s}{ma[1]:>8.2f}{IPOCR_MACRO_WRR:>8.2f}{ma[1] - IPOCR_MACRO_WRR:>+7.2f}"
          f"   {ma[2]:>8.2f}{IPOCR_MACRO_CRR:>8.2f}{ma[2] - IPOCR_MACRO_CRR:>+7.2f}")
    print(f"  -> we match/beat IndicPhotoOCR on WRR in {winW}/{tot} scripts, CRR in {winC}/{tot}.")
    print("  NOTE fairness caveats: (1) crop sets may differ (their BSTD-Test cropping / don't-care")
    print("       filtering vs bharat_rec_test.txt); (2) they route per-script recognisers + SI, we")
    print("       use one unified model (no SI). Treat as indicative, not identical-protocol.")

    # ---- CSVs + mismatch dump ---------------------------------------------------------------
    for mode in MODES:
        with open(os.path.join(args.out, f"perscript_{mode}.csv"), 'w', newline='', encoding='utf-8') as fh:
            wr = csv.writer(fh)
            wr.writerow(['script', 'N', 'WRR', 'CRR'])
            for g in SCRIPTS_10 + ['other', '__ALL__', '__MACRO__']:
                if g in M[mode]:
                    n, w, c = M[mode][g]
                    wr.writerow([DISPLAY.get(g, g), n, f"{w:.4f}", f"{c:.4f}"])

    if args.dump:
        seen = collections.Counter()
        dump_path = os.path.join(args.out, "mismatches.csv")
        with open(dump_path, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.writer(fh)
            wr.writerow(['script', 'ref', 'pred', 'ok_nfc', 'ok_nfc_clean',
                         'ref_codepoints', 'pred_codepoints'])
            for r, p, g in triples:
                if normstr(r, 'strip') == normstr(p, 'strip') or seen[g] >= args.dump:
                    continue
                seen[g] += 1
                cps = lambda s: ' '.join(f"{ord(c):04X}" for c in s)
                wr.writerow([DISPLAY.get(g, g), r, p,
                             int(normstr(r, 'nfc') == normstr(p, 'nfc')),
                             int(normstr(r, 'nfc_clean') == normstr(p, 'nfc_clean')),
                             cps(r), cps(p)])
        print(f"\n[dump] up to {args.dump} mismatches/script -> {dump_path}")
    print(f"[out] per-script CSVs -> {args.out}/perscript_*.csv")


if __name__ == "__main__":
    main()
