#!/usr/bin/env python3
"""mlt_to_db.py — ICDAR MLT (2017/2019) detection GT -> PaddleOCR DB detector labels.

Language-agnostic: keeps ALL scripts (detection doesn't care).
GT line format:  x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
  - transcription may itself contain commas -> split(',', 9)
  - transcription '###' or empty -> ignore region (care=False)
Image <stem>.<ext>  <->  GT file  gt_<stem>.txt

Images can be spread across ImagesPart1/ImagesPart2/... — we index by stem recursively,
so just point --img-root at the parent of all image parts.
"""
import argparse, glob, json, os, collections

IMG_EXTS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')


def index_images(img_root):
    idx = {}
    for e in IMG_EXTS:
        for p in glob.glob(os.path.join(img_root, '**', '*' + e), recursive=True):
            idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return idx


def parse_gt(path):
    boxes = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for ln in f:
            ln = ln.rstrip('\r\n')
            if not ln.strip():
                continue
            parts = ln.split(',', 9)
            if len(parts) == 9:
                parts.append('')            # missing transcription
            if len(parts) < 10:
                continue
            try:
                n = [int(round(float(x))) for x in parts[:8]]
            except Exception:
                continue
            pts = [[n[0], n[1]], [n[2], n[3]], [n[4], n[5]], [n[6], n[7]]]
            boxes.append((pts, parts[8].strip(), parts[9]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', required=True, help='parent dir of the image part(s)')
    ap.add_argument('--gt-root', required=True, help='dir with gt_*.txt files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='mlt')
    args = ap.parse_args()

    imgidx = index_images(args.img_root)
    gts = glob.glob(os.path.join(args.gt_root, '**', 'gt_*.txt'), recursive=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    stats, scripts = collections.Counter(), collections.Counter()
    n = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for gt in gts:
            stem = os.path.basename(gt)[3:-4]           # strip 'gt_' and '.txt'
            img = imgidx.get(stem)
            if not img:
                stats['img_missing'] += 1; continue
            boxes = []
            for pts, script, trans in parse_gt(gt):
                scripts[script or 'none'] += 1
                t = trans.strip()
                if t in ('', '###'):
                    boxes.append({'transcription': '###', 'points': pts}); stats['ignore'] += 1
                else:
                    boxes.append({'transcription': t, 'points': pts}); stats['legible'] += 1
            if not boxes:
                stats['no_boxes'] += 1; continue
            out.write(img + '\t' + json.dumps(boxes, ensure_ascii=False) + '\n'); n += 1

    rep = [f'=== {args.tag} det -> {args.out} ===',
           f'  images written: {n}   img_missing: {stats["img_missing"]}   no_boxes: {stats["no_boxes"]}',
           f'  boxes legible: {stats["legible"]}   ignore(###): {stats["ignore"]}',
           f'  script mix: {dict(scripts)}']
    print('\n'.join(rep))
    with open(os.path.splitext(args.out)[0] + '_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')


if __name__ == '__main__':
    main()
