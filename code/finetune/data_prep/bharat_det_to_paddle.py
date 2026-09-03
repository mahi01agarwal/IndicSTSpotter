#!/usr/bin/env python3
"""bharat_det_to_paddle.py
Convert Bharat Scene Text *detection* annotations -> PaddleOCR DB detector labels.

Input (from detection.zip):
  <ann> JSON = dict { image_id: {"annotations": {"polygon_k": {
                        "coordinates": [[x,y],...],           # arbitrary-length polygon
                        "text": <str|NaN>,                    # NaN = illegible
                        "script_language": <str>}}} }
  image_id 'A_image_1005' -> <img_root>/A/image_1005.jpg
     (folder = chars before first '_'; name = the rest)

Output (PaddleOCR DB SimpleDataSet format), one line per image:
  <abs_img_path>\t[{"transcription": "...", "points": [[x,y],...]}, ...]

  - ALL scripts kept (detection is language-agnostic).
  - Illegible / NaN / empty text -> transcription "###"  (ignore region, care=False).
  - Degenerate polygons (area < --min-area, <3 pts, bad coords) dropped.

Optional split: pass --test-scene-json <test_recognition_data.json> to hold out the
same scenes used by the recognition test set (keeps the Bharat benchmark consistent
and leak-free). Without it, everything goes to bharat_det_all.txt.
"""
import argparse, json, os, sys, collections


def clean_text(t):
    if t is None:
        return None
    if isinstance(t, float):        # JSON NaN parses to float('nan')
        return None
    t = str(t).strip()
    return t if t else None


def poly_area(pts):
    n = len(pts); a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


EXTS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')


def img_path_for(key, img_root):
    if '_' not in key:
        return None
    folder, name = key.split('_', 1)
    stem = os.path.join(img_root, folder, name)
    for e in EXTS:
        if os.path.isfile(stem + e):
            return stem + e
    return stem + '.jpg'   # default; flagged as missing if truly absent


def scene_of(basename):
    b = os.path.splitext(basename)[0]        # A_image_109_10
    i = b.rfind('_')
    if i > 0 and b[i + 1:].isdigit():
        return b[:i]                          # -> A_image_109
    return b


def load_test_scenes(rec_json):
    with open(rec_json, encoding='utf-8') as f:
        d = json.load(f)
    scenes = set()
    for k, v in d.items():
        base = os.path.basename(v.get('path', k)) if isinstance(v, dict) else k
        scenes.add(scene_of(base))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='BSTD_*.json annotation file')
    ap.add_argument('--img-root', required=True,
                    help='dir holding the letter folders A/ B/ ... (e.g. $SCR/bharat/detection/detection)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--test-scene-json', default='',
                    help='test_recognition_data.json; if given, split det into train/test by scene')
    ap.add_argument('--min-area', type=float, default=10.0,
                    help='drop polygons with area < this many px^2 (default 10)')
    ap.add_argument('--no-check-exists', action='store_true')
    args = ap.parse_args()

    with open(args.ann, encoding='utf-8') as f:
        d = json.load(f)                      # allow_nan=True -> NaN text -> float nan
    print(f'loaded {args.ann}: {len(d)} images', file=sys.stderr)

    test_scenes = load_test_scenes(args.test_scene_json) if args.test_scene_json else set()
    if test_scenes:
        print(f'test scenes from rec split: {len(test_scenes)}', file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    if test_scenes:
        f_train = open(os.path.join(args.out_dir, 'bharat_det_train.txt'), 'w', encoding='utf-8')
        f_test = open(os.path.join(args.out_dir, 'bharat_det_test.txt'), 'w', encoding='utf-8')
    else:
        f_train = open(os.path.join(args.out_dir, 'bharat_det_all.txt'), 'w', encoding='utf-8')
        f_test = None

    stats = collections.Counter()
    scripts = collections.Counter()
    ptcount = collections.Counter()
    written = collections.Counter()

    for key, rec in d.items():
        anns = (rec or {}).get('annotations') or {}
        img = img_path_for(key, args.img_root)
        if img is None:
            stats['bad_key'] += 1; continue
        if not args.no_check_exists and not os.path.isfile(img):
            stats['img_missing'] += 1; continue
        boxes = []
        for pk, pv in anns.items():
            coords = pv.get('coordinates')
            if not coords or len(coords) < 3:
                stats['dropped_fewpts'] += 1; continue
            try:
                pts = [[int(round(float(x))), int(round(float(y)))] for x, y in coords]
            except Exception:
                stats['dropped_badcoord'] += 1; continue
            if poly_area(pts) < args.min_area:
                stats['dropped_degenerate'] += 1; continue
            scripts[(pv.get('script_language') or 'unknown').lower()] += 1
            ptcount[len(pts)] += 1
            txt = clean_text(pv.get('text'))
            if txt is None:
                boxes.append({'transcription': '###', 'points': pts})
                stats['boxes_ignore'] += 1
            else:
                boxes.append({'transcription': txt, 'points': pts})
                stats['boxes_legible'] += 1
        if not boxes:
            stats['img_no_boxes'] += 1; continue
        line = img + '\t' + json.dumps(boxes, ensure_ascii=False) + '\n'
        if test_scenes and key in test_scenes:
            f_test.write(line); written['test'] += 1
        else:
            f_train.write(line); written['train'] += 1

    f_train.close()
    if f_test:
        f_test.close()

    lines = [f'=== bharat det -> {args.out_dir} ===',
             f'  images written : train={written["train"]} test={written["test"]}',
             f'  images missing : {stats["img_missing"]}   no-boxes: {stats["img_no_boxes"]}   bad-key: {stats["bad_key"]}',
             f'  boxes legible  : {stats["boxes_legible"]}',
             f'  boxes ignore(###): {stats["boxes_ignore"]}',
             f'  dropped degen/fewpts/badcoord: '
             f'{stats["dropped_degenerate"]}/{stats["dropped_fewpts"]}/{stats["dropped_badcoord"]}',
             f'  polygon point-counts: {dict(sorted(ptcount.items()))}',
             f'  script_language mix: {dict(scripts)}']
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, 'bharat_det_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
