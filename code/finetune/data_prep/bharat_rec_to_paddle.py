#!/usr/bin/env python3
"""bharat_rec_to_paddle.py
Convert the Bharat Scene Text *recognition* split -> PaddleOCR rec label files.

Input (from recognition.zip, unzipped at <bharat_root>):
  <bharat_root>/Recognition/train_recognition_data.json
  <bharat_root>/Recognition/test_recognition_data.json
    each = dict { basename: {"path": "Recognition/<split>/<lang>/<f>.jpg",
                             "language": "<lang>", "text": "<transcription>"} }
  crops already exist at <bharat_root>/<path>.

Output (<out_dir>):
  bharat_rec_train.txt    lines: <abs_crop_path>\t<text>   (for training pool)
  bharat_rec_test.txt     held-out real benchmark  (do NOT put in training)
  bharat_rec_report.txt   per-language/script counts, drops, charset, OOV

Bharat covers only our target scripts, so no script filtering by default.
Absolute crop paths are written so this merges trivially with the synthetic +
MLT label files later (PaddleOCR SimpleDataSet with data_dir=/).
"""
import argparse, json, os, sys, collections

LANG2SCRIPT = {
    'hindi': 'Devanagari', 'marathi': 'Devanagari',
    'bengali': 'Bengali', 'assamese': 'Bengali', 'manipuri': 'Bengali',
    'gujarati': 'Gujarati', 'kannada': 'Kannada', 'malayalam': 'Malayalam',
    'odia': 'Odia', 'punjabi': 'Gurmukhi', 'tamil': 'Tamil', 'telugu': 'Telugu',
    'english': 'Latin',
}
ZW = {'‌', '‍'}  # ZWNJ, ZWJ — meaningful in Indic conjuncts; keep in text


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def convert(records, bharat_root, keep_langs, check_exists):
    lines, kept, dropped, reasons = [], collections.Counter(), collections.Counter(), collections.Counter()
    for r in records:
        lang = (r.get('language') or '').strip().lower()
        text = r.get('text', '')
        rel = r.get('path', '')
        if keep_langs and lang not in keep_langs:
            dropped[lang] += 1; reasons['lang_filtered'] += 1; continue
        if text is None:
            dropped[lang] += 1; reasons['null_text'] += 1; continue
        text = text.strip()
        if text == '':
            dropped[lang] += 1; reasons['empty_text'] += 1; continue
        abspath = os.path.join(bharat_root, rel)
        if check_exists and not os.path.isfile(abspath):
            dropped[lang] += 1; reasons['missing_file'] += 1; continue
        lines.append((abspath, text, lang))
        kept[lang] += 1
    return lines, kept, dropped, reasons


def charset_report(lines):
    cs, zwj = collections.Counter(), 0
    for _, text, _ in lines:
        for ch in text:
            cs[ch] += 1
        if any(z in text for z in ZW):
            zwj += 1
    return cs, zwj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bharat-root', required=True,
                    help='dir containing Recognition/  (e.g. $SCR/bharat/recognition)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--keep-langs', default='',
                    help='comma list to restrict languages (default: keep all target scripts)')
    ap.add_argument('--dict', default='', help='optional dict file (one char per line) to check OOV')
    ap.add_argument('--no-check-exists', action='store_true',
                    help='skip verifying each crop file exists on disk')
    args = ap.parse_args()

    keep = set(x.strip().lower() for x in args.keep_langs.split(',') if x.strip())
    os.makedirs(args.out_dir, exist_ok=True)
    rec_dir = os.path.join(args.bharat_root, 'Recognition')

    report, all_lines = [], {}
    for split in ('train', 'test'):
        jf = os.path.join(rec_dir, f'{split}_recognition_data.json')
        if not os.path.isfile(jf):
            print(f'WARN: {jf} not found', file=sys.stderr); continue
        recs = list(load(jf).values())
        lines, kept, dropped, reasons = convert(recs, args.bharat_root, keep, not args.no_check_exists)
        out = os.path.join(args.out_dir, f'bharat_rec_{split}.txt')
        with open(out, 'w', encoding='utf-8') as f:
            for abspath, text, _ in lines:
                f.write(f'{abspath}\t{text}\n')
        all_lines[split] = lines
        scr = collections.Counter()
        for _, _, lang in lines:
            scr[LANG2SCRIPT.get(lang, lang)] += 1
        report += [
            f'=== {split} ===',
            f'  input records : {len(recs)}',
            f'  kept lines    : {sum(kept.values())} -> {out}',
            f'  per-language  : {dict(kept)}',
            f'  per-script    : {dict(scr)}',
            f'  dropped       : {sum(dropped.values())}  reasons={dict(reasons)}',
        ]

    if 'train' in all_lines:
        cs, zwj = charset_report(all_lines['train'])
        report += ['--- train charset ---',
                   f'  distinct chars       : {len(cs)}',
                   f'  samples with ZWJ/ZWNJ: {zwj}']
        if args.dict:
            with open(args.dict, encoding='utf-8') as f:
                dchars = set(line.rstrip('\n') for line in f)
            oov = sorted(((c, n) for c, n in cs.items() if c not in dchars and c != ' '),
                         key=lambda x: -x[1])
            report += [f'  OOV vs {args.dict}: {len(oov)} chars',
                       '  top OOV: ' + ' '.join(f'{repr(c)}:{n}' for c, n in oov[:40])]

    txt = '\n'.join(report)
    print(txt)
    with open(os.path.join(args.out_dir, 'bharat_rec_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
