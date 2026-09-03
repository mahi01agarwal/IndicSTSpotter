#!/usr/bin/env python3
"""indicstr12_to_paddle.py — IndicSTR12 'real' SCENES + quad GT -> recognizer crops.

Despite the project page calling them "word crops", the real.zip `verified_twice/<lang>/`
holds FULL SCENE images (<N>.jpeg) + a per-scene GT (<N>_gt.txt) with one line per word:
    idx <TAB> x1 y1 x2 y2 x3 y3 x4 y4 <TAB> transcription
So we warp-crop each quad out of the scene (PaddleOCR-style), exactly like mlt_crop_rec.py.

TRAIN/TEST SPLIT (added): IndicSTR12 ships NO split in real.zip, but the paper (Table 3)
defines a ~20K/7K train/test split of the real word-images and uses train for fine-tuning,
test as the benchmark. We reproduce a split OURSELVES, deterministically, at the SCENE
level (not the crop level): every word cropped from one signboard goes entirely to train
OR entirely to test. Splitting whole scenes is essential — near-identical words from the
same sign must not straddle train and test, or the benchmark leaks. The assignment is a
stable md5 hash of "<lang>/<scene>", so it is identical across machines and re-runs and
never depends on Python's salted hash().

Emits (into <out-dir>/):
    images/<lang>_<N>_<i>.jpg
    <tag>_rec.txt        <- TRAIN crops   (crop_path\ttext)  -> merge_rec_db --real-inputs
    <tag>_rec_test.txt   <- TEST crops    (held out as a benchmark; NEVER put in training)
Skips urdu (RTL/Arabic, out of scope) and macOS junk (__MACOSX, ._*, .DS_Store).

    python indicstr12_to_paddle.py --root $SCR/indicstr12/verified_twice \
        --out-dir $SCR/rec_data_real/indicstr12 --tag indicstr12 \
        --test-frac 0.2 --split-seed 1234
"""
import argparse, os, glob, re, collections, hashlib
import numpy as np
import cv2

IMG_EXTS = ('jpeg', 'jpg', 'png', 'JPEG', 'JPG', 'PNG')
DEFAULT_SKIP = ['urdu']


def parse_gt(path):
    """-> [(quad(4x2 float32), text)]. Tolerates tab- or whitespace-delimited."""
    out = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for ln in f:
            ln = ln.rstrip('\r\n')
            if not ln.strip():
                continue
            parts = ln.split('\t')
            if len(parts) < 10:                      # fall back to whitespace, keep text intact
                parts = re.split(r'\s+', ln.strip(), maxsplit=9)
            if len(parts) < 10:
                continue
            try:
                n = [float(x) for x in parts[1:9]]   # parts[0] is the word index
            except ValueError:
                continue
            text = parts[9].strip()
            if not text or text == '###':
                continue
            quad = np.array([[n[0], n[1]], [n[2], n[3]], [n[4], n[5]], [n[6], n[7]]], dtype='float32')
            out.append((quad, text))
    return out


def get_rotate_crop_image(img, pts):
    pts = pts.astype('float32')
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 1 or h < 1:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h * 1.0 / w >= 1.5:                            # vertical text -> upright
        crop = np.rot90(crop)
    return crop


def is_test(lang, base, frac, seed):
    """Deterministic per-SCENE split. Stable across machines/runs (md5, not salted hash()).
    All crops of one scene share this verdict, so a signboard never straddles train/test."""
    if frac <= 0:
        return False
    key = f'{seed}:{lang}/{base}'.encode('utf-8')
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 10000
    return bucket < int(round(frac * 10000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir with per-language subfolders (verified_twice/).')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='indicstr12')
    ap.add_argument('--skip-langs', nargs='*', default=DEFAULT_SKIP)
    ap.add_argument('--min-h', type=int, default=8)
    ap.add_argument('--test-frac', type=float, default=0.2,
                    help='fraction of SCENES (per language) held out as the test benchmark (0 = all train).')
    ap.add_argument('--split-seed', type=int, default=1234,
                    help='seed for the deterministic scene-level split (change to reshuffle).')
    args = ap.parse_args()
    skip = set(args.skip_langs)

    imgdir = os.path.join(args.out_dir, 'images')
    os.makedirs(imgdir, exist_ok=True)
    stats = collections.Counter()
    per_lang = {'train': collections.Counter(), 'test': collections.Counter()}      # crops
    scenes = {'train': collections.Counter(), 'test': collections.Counter()}        # scenes

    train_path = os.path.join(args.out_dir, f'{args.tag}_rec.txt')
    test_path = os.path.join(args.out_dir, f'{args.tag}_rec_test.txt')
    with open(train_path, 'w', encoding='utf-8') as trainf, \
         open(test_path, 'w', encoding='utf-8') as testf:
        for lang in sorted(os.listdir(args.root)):
            d = os.path.join(args.root, lang)
            if not os.path.isdir(d) or lang.startswith(('__', '.')) or lang in skip:
                continue
            imgs = []
            for e in IMG_EXTS:
                imgs += glob.glob(os.path.join(d, f'*.{e}'))
            for img in sorted(imgs):
                base = os.path.splitext(os.path.basename(img))[0]
                if base.startswith('._'):
                    continue
                gt = os.path.join(d, base + '_gt.txt')
                if not os.path.isfile(gt):
                    stats['gt_missing'] += 1; continue
                im = cv2.imread(img)
                if im is None:
                    stats['img_unreadable'] += 1; continue
                split = 'test' if is_test(lang, base, args.test_frac, args.split_seed) else 'train'
                out_f = testf if split == 'test' else trainf
                kept_here = 0
                for i, (quad, text) in enumerate(parse_gt(gt)):
                    crop = get_rotate_crop_image(im, quad)
                    if crop is None or crop.shape[0] < args.min_h:
                        stats['skip_small'] += 1; continue
                    name = f'{lang}_{base}_{i}.jpg'
                    cv2.imwrite(os.path.join(imgdir, name), crop)
                    out_f.write(os.path.join(imgdir, name) + '\t' + text + '\n')
                    per_lang[split][lang] += 1
                    stats[f'kept_{split}'] += 1
                    kept_here += 1
                if kept_here:
                    scenes[split][lang] += 1

    def fmt(counter):
        return dict(sorted(counter.items()))

    tr, te = stats['kept_train'], stats['kept_test']
    tot = tr + te
    rep = [f'=== {args.tag} rec crops -> {args.out_dir} ===',
           f'  split: scene-level, test-frac={args.test_frac}, seed={args.split_seed}',
           f'  TRAIN crops : {tr}  -> {train_path}',
           f'  TEST  crops : {te}  -> {test_path}   (HELD OUT — never in training)',
           f'  test share  : {100.0 * te / max(tot, 1):.1f}% of {tot} crops',
           f'  per-language TRAIN crops : {fmt(per_lang["train"])}',
           f'  per-language TEST  crops : {fmt(per_lang["test"])}',
           f'  per-language scenes (tr/te): '
           + ', '.join(f'{l}={scenes["train"][l]}/{scenes["test"][l]}'
                       for l in sorted(set(scenes["train"]) | set(scenes["test"]))),
           f'  skip_small={stats["skip_small"]}  gt_missing={stats["gt_missing"]}  '
           f'img_unreadable={stats["img_unreadable"]}  (skipped langs: {sorted(skip)})']
    print('\n'.join(rep))
    with open(os.path.join(args.out_dir, f'{args.tag}_rec_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')


if __name__ == '__main__':
    main()
