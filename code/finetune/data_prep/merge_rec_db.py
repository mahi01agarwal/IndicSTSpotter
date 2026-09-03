#!/usr/bin/env python3
"""merge_rec_db.py — pool SYNTHETIC + REAL recognizer crops into one unified
train/val set for the synth+real recognizer, rebalanced so the scarce REAL crops
actually carry weight instead of being drowned by the (redundant) synthetic.

Each input line is PaddleOCR rec format:  <crop_path>\t<text>.
Real crops use ABSOLUTE paths; synthetic crops use paths RELATIVE to rec_data_25k
(crops/<script>/...). The pooled files carry no language column, so the script of a
crop is inferred from its transcription (majority Unicode block among the 11 Indic
scripts + Latin). That inference is used ONLY for reporting and for the caps.

Balancing philosophy (why this differs from a naive concat):
  The earlier synth-only recognizer already converged on all ~2.5M synthetic crops,
  so re-feeding them teaches nothing new — the ONLY new signal is the ~194k real
  crops. Improvement is governed by real's WEIGHT per gradient step, not by more
  synthetic. So we CAP SYNTHETIC PER SCRIPT (it's redundant coverage) and keep every
  real crop, optionally oversampling real. Result: real becomes ~30-40% of the mix
  (and ~40-50% on the scripts that actually have real data), and each epoch is far
  cheaper. Sparse-real scripts stay synth-dominated — correct, there's little real
  to lean on there, which is exactly why synthetic is kept at all.

    python merge_rec_db.py \
      --synth-inputs $SCR/rec_data_25k/unified_train.txt $SCR/rec_data_25k/unified_val.txt \
      --real-inputs  $SCR/rec_data_real/bharat/bharat_rec_train.txt \
                     $SCR/rec_data_real/mlt17/mlt17_rec.txt \
                     $SCR/rec_data_real/mlt19/mlt19_rec.txt \
      --synth-cap-per-script 20000 --real-oversample 1 --latin-cap 30000 \
      --out-dir $SCR/rec_data_combined

Do NOT pass bharat_rec_test.txt — it is the held-out benchmark.
"""
import argparse, os, random, collections

# script name -> inclusive Unicode block(s) (for majority-vote script inference)
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


def script_of(text):
    counts = collections.Counter()
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x0030 <= cp <= 0x0039):   # Latin letters/ext + digits
            counts['latin'] += 1; continue
        for name, ranges in BLOCKS:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1; break
    if not counts:
        return 'other'
    return counts.most_common(1)[0][0]


def read_lines(paths, seen, per_source, maxlen_box):
    """Read label files -> {script: [line, ...]}, dedup by crop path across `seen`."""
    by_script = collections.defaultdict(list)
    dup = 0
    for inp in paths:
        tag = os.path.basename(inp)
        with open(inp, encoding='utf-8') as f:
            for ln in f:
                ln = ln.rstrip('\n')
                if not ln.strip() or '\t' not in ln:
                    continue
                path, text = ln.split('\t', 1)
                if path in seen:
                    dup += 1; continue
                seen.add(path)
                by_script[script_of(text)].append(ln)
                per_source[tag] += 1
                maxlen_box[0] = max(maxlen_box[0], len(text))
    return by_script, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth-inputs', nargs='*', default=[],
                    help='synthetic rec label .txt files (per-script capped).')
    ap.add_argument('--real-inputs', nargs='*', default=[],
                    help='real rec label .txt files (kept in full; optionally oversampled).')
    ap.add_argument('--inputs', nargs='*', default=[],
                    help='DEPRECATED single pool (treated as synth). Prefer --synth-inputs/--real-inputs.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--synth-cap-per-script', type=int, default=0,
                    help='max SYNTHETIC crops to keep per inferred script (0 = keep all). '
                         'The synthetic is redundant; cap it so real carries weight.')
    ap.add_argument('--real-oversample', type=int, default=1,
                    help='duplicate each REAL crop this many times in the TRAIN split '
                         '(1 = no oversample). Val is never oversampled (no leakage).')
    ap.add_argument('--latin-cap', type=int, default=0,
                    help='max Latin/English crops in the FINAL combined set (0 = keep all). '
                         'Backstop so real Latin (huge from MLT) does not dominate Indic.')
    ap.add_argument('--val-frac', type=float, default=0.02)
    ap.add_argument('--val-max', type=int, default=5000, help='cap val set size (crops)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    seen = set()
    per_source = collections.Counter()
    maxlen_box = [0]

    synth_paths = list(args.synth_inputs) + list(args.inputs)
    synth_by_script, dup_s = read_lines(synth_paths, seen, per_source, maxlen_box)
    real_by_script, dup_r = read_lines(args.real_inputs, seen, per_source, maxlen_box)
    dup = dup_s + dup_r

    # ---- cap synthetic per script (redundant coverage) ----
    synth_before = {s: len(v) for s, v in synth_by_script.items()}
    if args.synth_cap_per_script:
        for s, lst in synth_by_script.items():
            if len(lst) > args.synth_cap_per_script:
                rng.shuffle(lst)
                synth_by_script[s] = lst[:args.synth_cap_per_script]

    # ---- combine per script, tagging real vs synth (is_real) ----
    scripts = set(synth_by_script) | set(real_by_script)
    combined = collections.defaultdict(list)               # script -> [(line, is_real)]
    for s in scripts:
        for ln in synth_by_script.get(s, []):
            combined[s].append((ln, False))
        for ln in real_by_script.get(s, []):
            combined[s].append((ln, True))

    # ---- final Latin cap on the COMBINED latin (real + synth) ----
    before_latin = len(combined.get('latin', []))
    latin_dropped = 0
    if args.latin_cap and before_latin > args.latin_cap:
        rng.shuffle(combined['latin'])
        latin_dropped = before_latin - args.latin_cap
        combined['latin'] = combined['latin'][:args.latin_cap]

    # ---- flatten, split val from UNIQUE rows, then oversample real in TRAIN only ----
    rows = [(ln, is_real) for lst in combined.values() for (ln, is_real) in lst]
    rng.shuffle(rows)
    n_val = min(int(len(rows) * args.val_frac), args.val_max)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    train = [ln for ln, _ in train_rows]
    if args.real_oversample > 1:
        extra = [ln for ln, is_real in train_rows if is_real] * (args.real_oversample - 1)
        train += extra
    val = [ln for ln, _ in val_rows]
    rng.shuffle(train)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (('unified_train.txt', train), ('unified_val.txt', val)):
        with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(data) + '\n')

    # ---- report: per-script synth vs real, real fraction ----
    def script_counts():
        out = {}
        for s in sorted(scripts, key=lambda s: -len(combined[s])):
            n_syn = sum(1 for _, ir in combined[s] if not ir)
            n_real = sum(1 for _, ir in combined[s] if ir)
            tot = n_syn + n_real
            out[s] = f"{tot} (real {n_real} = {100.0 * n_real / tot:.0f}%)"
        return out

    tot_real = sum(1 for lst in combined.values() for _, ir in lst if ir)
    tot_all = sum(len(lst) for lst in combined.values())
    rep = [f'=== merged rec -> {args.out_dir} ===',
           f'  per-source crops     : {dict(per_source)}',
           f'  duplicates dropped   : {dup}',
           f'  synth cap/script     : {args.synth_cap_per_script or "none"}',
           f'  real oversample      : x{args.real_oversample} (train only)',
           f'  latin cap (combined) : {args.latin_cap or "none"} (dropped {latin_dropped} of {before_latin})',
           f'  per-script total (real share):',
           ] + [f'      {s:12s} {v}' for s, v in script_counts().items()] + [
           f'  OVERALL real share   : {tot_real}/{tot_all} = {100.0 * tot_real / max(tot_all,1):.1f}%',
           f'  TRAIN={len(train)}  VAL={len(val)}',
           f'  longest label = {maxlen_box[0]} chars  ->  make_config.py --max-text-length {maxlen_box[0] + 5}']
    txt = '\n'.join(rep)
    print(txt)
    with open(os.path.join(args.out_dir, 'merge_rec_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
