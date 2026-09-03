#!/usr/bin/env python3
"""mist_to_db.py — MIST (CVIT, WACV'26) detection GT -> PaddleOCR DB detector labels.

MIST ships LabelMe-style JSON, one file per image, matched to the image by identical
stem across parallel dirs:

    <root>/Images/<Split>/<stem>.png
    <root>/gt/<Split>/<stem>.json          Split in {Train, Test}

GT JSON schema (LabelMe v5.x):
    { "version": ..., "flags": {...},
      "shapes": [ { "points": [[x,y], ...],   # float polygon, absolute px (1920x1080)
                    "care": 1|0,              # 1 = keep/evaluate, 0 = do-not-care
                    "script": "hindi"|... },  # 11 scripts + english; illegible/logos -> care 0
                  ... ] }

MIST is DETECTION-ONLY (no per-word transcription in the shapes), so for the PaddleOCR DB
format we only need points + the care flag:
    care==1 -> transcription "text"  (a legible placeholder; DB ignores the string, only the
                                      "is it ### ?" test matters for shrink/border maps)
    care==0 -> transcription "###"   (ignore region, care=False, masked out of the loss)
If a shape ever carries a real "label"/"text"/"transcription" it is preserved instead of the
placeholder (harmless for detection, handy if these crops are reused for rec later).

Output (PaddleOCR DB SimpleDataSet, one line per image, ABSOLUTE image paths so data_dir=/):
    <abs_img_path>\t[{"transcription": "...", "points": [[x,y],...]}, ...]

The *_stale_* sibling dirs in the release are superseded re-annotations — pass only Train/Test.
Run once per split:
    python mist_to_db.py --root $SCR/mist --split Train --out $SCR/det_data_real/mist/mist_det_train.txt
    python mist_to_db.py --root $SCR/mist --split Test  --out $SCR/det_data_real/mist/mist_det_test.txt
Merge ONLY the train file into the detector set (Test is the held-out MIST benchmark).
"""
import argparse, glob, json, os, collections, sys

IMG_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')


def poly_area(pts):
    n = len(pts); a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def index_images(img_dir):
    """stem -> abs image path (first extension wins)."""
    idx = {}
    for e in IMG_EXTS:
        for p in glob.glob(os.path.join(img_dir, '*' + e)):
            idx.setdefault(os.path.splitext(os.path.basename(p))[0], os.path.abspath(p))
    return idx


def clean_label(s):
    lab = s.get('label') or s.get('text') or s.get('transcription')
    if lab is None:
        return None
    if isinstance(lab, float):        # JSON NaN
        return None
    lab = str(lab).strip()
    return lab or None


def to_points(shape):
    """Return integer [[x,y],...] with >=3 points, expanding a 2-pt LabelMe rectangle."""
    coords = shape.get('points')
    if not coords:
        return None
    try:
        pts = [[int(round(float(x))), int(round(float(y)))] for x, y in coords]
    except Exception:
        return None
    if shape.get('shape_type') == 'rectangle' or len(pts) == 2:
        if len(pts) < 2:
            return None
        (x0, y0), (x1, y1) = pts[0], pts[1]
        pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return pts if len(pts) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='MIST root holding Images/ and gt/ (e.g. $SCR/mist)')
    ap.add_argument('--split', default='Train', choices=['Train', 'Test'],
                    help='which split to convert (default Train)')
    ap.add_argument('--img-dir', default='',
                    help='override image dir (default <root>/Images/<split>)')
    ap.add_argument('--gt-dir', default='',
                    help='override gt dir (default <root>/gt/<split>)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-area', type=float, default=10.0,
                    help='drop polygons with area < this many px^2 (default 10)')
    ap.add_argument('--tag', default='mist')
    args = ap.parse_args()

    img_dir = args.img_dir or os.path.join(args.root, 'Images', args.split)
    gt_dir = args.gt_dir or os.path.join(args.root, 'gt', args.split)
    for d in (img_dir, gt_dir):
        if not os.path.isdir(d):
            print(f'[fatal] not a dir: {d}', file=sys.stderr); sys.exit(1)

    imgidx = index_images(img_dir)
    gts = sorted(glob.glob(os.path.join(gt_dir, '*.json')))
    print(f'[{args.tag}/{args.split}] gt files: {len(gts)}   images indexed: {len(imgidx)}',
          file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    stats = collections.Counter()
    scripts = collections.Counter()
    ptcount = collections.Counter()
    n = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for gt in gts:
            stem = os.path.splitext(os.path.basename(gt))[0]
            img = imgidx.get(stem)
            if not img:
                stats['img_missing'] += 1; continue
            try:
                d = json.load(open(gt, encoding='utf-8'))
            except Exception:
                stats['bad_json'] += 1; continue
            boxes = []
            for sh in d.get('shapes', []):
                pts = to_points(sh)
                if pts is None:
                    stats['dropped_fewpts'] += 1; continue
                if poly_area(pts) < args.min_area:
                    stats['dropped_degenerate'] += 1; continue
                scripts[str(sh.get('script') or 'unknown').lower()] += 1
                ptcount[len(pts)] += 1
                # care: 1 -> keep, 0 (or missing/false) -> ignore region
                care = sh.get('care', 1)
                keep = care not in (0, '0', False, None)
                if keep:
                    lab = clean_label(sh) or 'text'
                    boxes.append({'transcription': lab, 'points': pts})
                    stats['boxes_legible'] += 1
                else:
                    boxes.append({'transcription': '###', 'points': pts})
                    stats['boxes_ignore'] += 1
            if not boxes:
                stats['img_no_boxes'] += 1; continue
            out.write(img + '\t' + json.dumps(boxes, ensure_ascii=False) + '\n'); n += 1

    rep = [f'=== {args.tag} {args.split} det -> {args.out} ===',
           f'  images written : {n}   img_missing: {stats["img_missing"]}   '
           f'no_boxes: {stats["img_no_boxes"]}   bad_json: {stats["bad_json"]}',
           f'  boxes legible  : {stats["boxes_legible"]}   ignore(###): {stats["boxes_ignore"]}',
           f'  dropped degen/fewpts: {stats["dropped_degenerate"]}/{stats["dropped_fewpts"]}',
           f'  polygon point-counts: {dict(sorted(ptcount.items()))}',
           f'  script mix: {dict(scripts)}']
    txt = '\n'.join(rep)
    print(txt)
    with open(os.path.splitext(args.out)[0] + '_report.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
