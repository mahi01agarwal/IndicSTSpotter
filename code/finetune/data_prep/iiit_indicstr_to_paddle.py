#!/usr/bin/env python3
"""iiit_indicstr_to_paddle.py — IIIT-IndicSTR-Word (pre-cropped words) -> PaddleOCR rec labels.

Per language, per split the unzipped layout is:
    <lang>/<split>/<split>_images.txt   # relative image paths, one per line
    <lang>/<split>/<split>_labels.txt   # transcriptions, LINE-ALIGNED with *_images.txt
    <lang>/<split>/images/*.png         # the actual word crops
Images are already word-level, so no cropping — just emit <abs_image_path>\t<transcription>.

Uses TRAIN + VAL by default; TEST is held out as a benchmark. Absolute paths -> drops
straight into merge_rec_db --real-inputs.

    python iiit_indicstr_to_paddle.py --root $SCR/iiit_indicstr \
        --splits train val --out $SCR/rec_data_real/iiit_indicstr/iiit_indicstr_rec.txt
"""
import argparse, os, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir with per-language subfolders, each unzipped into <split>/')
    ap.add_argument('--splits', nargs='*', default=['train', 'val'],
                    help='splits to include (default train val; test held out as benchmark).')
    ap.add_argument('--langs', nargs='*', default=None, help='languages to include (default: all found).')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    langs = args.langs or sorted(d for d in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, d)))
    rows, per = [], collections.Counter()
    n_mismatch = n_missing_img = n_empty = 0

    for lang in langs:
        for split in args.splits:
            sd = os.path.join(args.root, lang, split)
            imgs_txt = os.path.join(sd, f'{split}_images.txt')
            lbls_txt = os.path.join(sd, f'{split}_labels.txt')
            if not (os.path.isfile(imgs_txt) and os.path.isfile(lbls_txt)):
                continue
            with open(imgs_txt, encoding='utf-8') as f:
                imgs = [ln.strip() for ln in f if ln.strip()]
            with open(lbls_txt, encoding='utf-8') as f:
                lbls = [ln.rstrip('\r\n') for ln in f]
            if len(imgs) != len(lbls):                 # tolerate a trailing blank line
                n_mismatch += abs(len(imgs) - len(lbls))
                m = min(len(imgs), len(lbls))
                imgs, lbls = imgs[:m], lbls[:m]
            for rel, text in zip(imgs, lbls):
                text = text.strip()
                if not text:
                    n_empty += 1; continue
                img_abs = os.path.join(sd, 'images', os.path.basename(rel))
                if not os.path.isfile(img_abs):
                    n_missing_img += 1; continue
                rows.append(f'{img_abs}\t{text}')
                per[lang] += 1

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'[ok] {len(rows)} crops -> {args.out}')
    print(f'     per-language: {dict(per)}')
    print(f'     line mismatches={n_mismatch}  missing_images={n_missing_img}  empty_labels={n_empty}')
    print(f'     splits={args.splits}  (test held out as benchmark)')


if __name__ == '__main__':
    main()
