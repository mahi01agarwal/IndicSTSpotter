#!/usr/bin/env python3
"""mlt_crop_rec.py — ICDAR MLT (2017/2019) -> recognizer crops for target scripts.

Keeps only --keep-scripts (default: latin,bangla,hindi — the only Indic-relevant MLT
scripts). Skips '###'/empty. Warps each quad to an axis-aligned crop (PaddleOCR-style
perspective crop, with a rotate for vertical text).

Emits:  <out_dir>/images/<stem>_<i>.jpg   and   <out_dir>/<tag>_rec.txt  (crop_path\ttext)

GT line:  x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
"""
import argparse, glob, os, collections
import numpy as np
import cv2

IMG_EXTS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
SCRIPT_MAP = {'latin': 'Latin', 'bangla': 'Bengali', 'hindi': 'Devanagari'}


def index_images(img_root):
    idx = {}
    for e in IMG_EXTS:
        for p in glob.glob(os.path.join(img_root, '**', '*' + e), recursive=True):
            idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return idx


def parse_gt(path):
    out = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for ln in f:
            ln = ln.rstrip('\r\n')
            if not ln.strip():
                continue
            parts = ln.split(',', 9)
            if len(parts) == 9:
                parts.append('')
            if len(parts) < 10:
                continue
            try:
                n = [float(x) for x in parts[:8]]
            except Exception:
                continue
            pts = np.array([[n[0], n[1]], [n[2], n[3]], [n[4], n[5]], [n[6], n[7]]], dtype='float32')
            out.append((pts, parts[8].strip(), parts[9]))
    return out


def get_rotate_crop_image(img, pts):
    """Standard PaddleOCR perspective crop of a quad; rotates vertical text upright."""
    pts = pts.astype('float32')
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 1 or h < 1:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h * 1.0 / w >= 1.5:                 # vertical text
        crop = np.rot90(crop)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', required=True, help='parent dir of the image part(s)')
    ap.add_argument('--gt-root', required=True, help='dir with gt_*.txt files')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--keep-scripts', default='latin,bangla,hindi')
    ap.add_argument('--min-h', type=int, default=8)
    ap.add_argument('--tag', default='mlt')
    args = ap.parse_args()

    keep = set(s.strip().lower() for s in args.keep_scripts.split(',') if s.strip())
    imgdir = os.path.join(args.out_dir, 'images')
    os.makedirs(imgdir, exist_ok=True)
    imgidx = index_images(args.img_root)
    gts = glob.glob(os.path.join(args.gt_root, '**', 'gt_*.txt'), recursive=True)
    stats, scr = collections.Counter(), collections.Counter()

    with open(os.path.join(args.out_dir, f'{args.tag}_rec.txt'), 'w', encoding='utf-8') as labelf:
        for gt in gts:
            stem = os.path.basename(gt)[3:-4]
            img = imgidx.get(stem)
            if not img:
                stats['img_missing'] += 1; continue
            im = cv2.imread(img)
            if im is None:
                stats['img_unreadable'] += 1; continue
            for i, (pts, script, trans) in enumerate(parse_gt(gt)):
                s = script.lower()
                if s not in keep:
                    stats['skip_script'] += 1; continue
                t = trans.strip()
                if t in ('', '###'):
                    stats['skip_illegible'] += 1; continue
                crop = get_rotate_crop_image(im, pts)
                if crop is None or crop.shape[0] < args.min_h:
                    stats['skip_small'] += 1; continue
                name = f'{stem}_{i}.jpg'
                cv2.imwrite(os.path.join(imgdir, name), crop)
                labelf.write(os.path.join(imgdir, name) + '\t' + t + '\n')
                scr[SCRIPT_MAP.get(s, s)] += 1
                stats['kept'] += 1

    rep = [f'=== {args.tag} rec crops -> {args.out_dir} ===',
           f'  crops kept: {stats["kept"]}',
           f'  per-script: {dict(scr)}',
           f'  skipped script/illegible/small: '
           f'{stats["skip_script"]}/{stats["skip_illegible"]}/{stats["skip_small"]}',
           f'  img missing/unreadable: {stats["img_missing"]}/{stats["img_unreadable"]}']
    print('\n'.join(rep))
    with open(os.path.join(args.out_dir, f'{args.tag}_rec_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')


if __name__ == '__main__':
    main()
