#!/usr/bin/env python3
"""Detection-only evaluation of stock PaddleOCR detectors on real GT-box images.

Counterpart to eval_real_perimage.py (which is recogniser-only). This one ignores
text content entirely and asks: does the detector put a box where the GT has a word?

Protocol (ICDAR-2015 style):
  * A GT word box is VALID if it has readable text, or IGNORED if its text is the
    "NA" placeholder — the annotator marked those unreadable, so a detector that
    finds them is not wrong and one that misses them is not penalised. Ignored
    regions are removed from the recall denominator, and any prediction matching
    one is discarded rather than counted as a false positive.
  * Greedy one-to-one matching by polygon IoU, highest IoU first. Each GT box and
    each prediction can be used at most once.
  * precision = TP / (TP + FP), recall = TP / valid_GT, hmean = harmonic mean.

GT boxes are axis-aligned rects (x, y, w, h); predictions are quads. IoU is real
polygon IoU (Sutherland-Hodgman clip + shoelace), not bbox-of-quad, so rotated
text is not given a free pass.

Needs: pillow, paddleocr. No shapely.

Usage
  python eval_det_real.py --images-dir $SCR/real_marathi --out $SCR/real_marathi_det
  python eval_det_real.py --images-dir $SCR/real_marathi --models PP-OCRv6_medium_det PP-OCRv5_server_det
"""
import os
import csv
import json
import glob
import argparse
from collections import defaultdict

from PIL import Image, ImageDraw


# ------------------------------------------------------------------ geometry
def poly_area(pts):
    """Shoelace. Sign-agnostic."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clip_poly(subject, clip):
    """Sutherland-Hodgman: clip `subject` against CONVEX `clip`. Returns vertices."""
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            return []
        cx1, cy1 = clip[i]
        cx2, cy2 = clip[(i + 1) % n]
        # inside = left of the directed edge (clip must be counter-clockwise)
        inside = lambda p: (cx2 - cx1) * (p[1] - cy1) - (cy2 - cy1) * (p[0] - cx1) >= 0
        nxt, prev = [], out[-1]
        for cur in out:
            if inside(cur):
                if not inside(prev):
                    nxt.append(intersect(prev, cur, (cx1, cy1), (cx2, cy2)))
                nxt.append(cur)
            elif inside(prev):
                nxt.append(intersect(prev, cur, (cx1, cy1), (cx2, cy2)))
            prev = cur
        out = nxt
    return out


def intersect(p1, p2, q1, q2):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = q1; x4, y4 = q2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return p2
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / d,
            (a * (y3 - y4) - (y1 - y2) * b) / d)


def ccw(pts):
    """Force counter-clockwise winding (clip_poly's `inside` test assumes it)."""
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += (x2 - x1) * (y2 + y1)
    return pts[::-1] if s > 0 else pts


def iou(a, b):
    """Polygon IoU. `b` is used as the clip region, so it must be convex — GT
    rects always are."""
    inter = poly_area(clip_poly(list(a), ccw(list(b))))
    if inter <= 0:
        return 0.0
    union = poly_area(a) + poly_area(b) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------- data
def load_gt(images_dir, skip_texts, granularity="word"):
    """-> {stem: (img_path, [valid_rects], [ignored_rects])}

    granularity="line" merges every GT word sharing a `line` index into that
    line's union rect. PP-OCR's DB detector emits text-LINE polygons, so scoring
    its output against word boxes charges it 1 TP + (k-1) FN for each line it got
    perfectly right. Line granularity is the fair reading for a stock detector;
    word granularity is what matters if you intend to feed crops to a recogniser.
    """
    out = {}
    for jp in sorted(glob.glob(os.path.join(images_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = next((os.path.join(images_dir, f"{stem}.{e}")
                    for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                    if os.path.isfile(os.path.join(images_dir, f"{stem}.{e}"))), None)
        if img is None:
            print(f"[skip] no image for {jp}")
            continue
        words = []
        for w in json.load(open(jp, encoding="utf-8")).get("words", []):
            t = str(w.get("text", "")).strip()
            if not t:
                continue
            x, y, ww, hh = int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"])
            if ww <= 0 or hh <= 0:
                continue
            words.append((x, y, x + ww, y + hh, t in skip_texts, w.get("line")))

        if granularity == "line":
            groups = defaultdict(list)
            for i, (x0, y0, x1, y1, ig, ln) in enumerate(words):
                groups[ln if ln is not None else f"_w{i}"].append((x0, y0, x1, y1, ig))
            merged = []
            for g in groups.values():
                # a line counts as ignored only if EVERY word in it is a placeholder
                merged.append((min(v[0] for v in g), min(v[1] for v in g),
                               max(v[2] for v in g), max(v[3] for v in g),
                               all(v[4] for v in g)))
            words = [(a, b, c, d, e, None) for a, b, c, d, e in merged]

        valid, ignored = [], []
        for x0, y0, x1, y1, ig, _ in words:
            rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            (ignored if ig else valid).append(rect)
        out[stem] = (img, valid, ignored)
    return out


def extract_polys(res):
    """Pull quads out of a paddleocr TextDetection result, tolerating layout
    differences between versions."""
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
    except Exception:
        d = {}
    polys = d.get("dt_polys") or d.get("polys") or d.get("boxes") or []
    out = []
    for p in polys:
        pts = [(float(pt[0]), float(pt[1])) for pt in p]
        if len(pts) >= 3 and poly_area(pts) > 0:
            out.append(pts)
    return out


# ---------------------------------------------------------------- evaluation
def match(preds, valid, ignored, thr):
    """Greedy one-to-one by IoU. -> (tp, fp, fn, matched_pairs, ignored_hits)"""
    cand = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(valid):
            v = iou(p, g)
            if v >= thr:
                cand.append((v, pi, gi))
    cand.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for v, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); pairs.append((pi, gi, v))

    # unmatched predictions that land on an ignored region are discarded
    ign_hits = 0
    for pi, p in enumerate(preds):
        if pi in used_p:
            continue
        if any(iou(p, g) >= thr for g in ignored):
            ign_hits += 1
            used_p.add(pi)

    tp = len(pairs)
    fp = len(preds) - len(used_p)
    fn = len(valid) - tp
    return tp, fp, fn, pairs, ign_hits


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def draw(img_path, preds, valid, ignored, pairs, dst):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    hit = {gi for _, gi, _ in pairs}
    for i, g in enumerate(ignored):
        d.polygon(g, outline=(128, 128, 128), width=3)
    for i, g in enumerate(valid):
        d.polygon(g, outline=(0, 200, 0) if i in hit else (255, 0, 0), width=4)
    for pi, p in enumerate(preds):
        d.polygon([tuple(x) for x in p], outline=(0, 128, 255), width=2)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser(description="Detector-only eval on real GT-box images.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="./real_det_eval")
    ap.add_argument("--models", nargs="*", default=["PP-OCRv6_medium_det"],
                    help="stock paddleocr TextDetection model_name(s) to evaluate.")
    ap.add_argument("--ft-det-dir", default=None,
                    help="exported inference dir of a FINE-TUNED detector (evaluated alongside "
                         "the stock ones). Note it is trained word-level, so read its WORD "
                         "numbers against the stock models' LINE numbers with care.")
    ap.add_argument("--ft-det-name", default="PP-OCRv6_medium_det",
                    help="base model_name the fine-tuned weights belong to.")
    ap.add_argument("--ft-det-label", default="ft_det", help="name for it in the output files.")
    ap.add_argument("--det-limit-side-len", type=int, default=960,
                    help="detector inference resize cap; raise (1280/1920/2560) for dense/high-res "
                         "scenes. 960 = PP-OCR default. Applied to EVERY model in the run so the "
                         "stock-vs-ft comparison is at matched resolution.")
    ap.add_argument("--det-limit-type", default="max", choices=["max", "min"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--extra-iou", nargs="*", type=float, default=[0.3, 0.7],
                    help="additional IoU thresholds to report (sensitivity check).")
    ap.add_argument("--skip-text", nargs="*", default=["NA"],
                    help="GT texts treated as IGNORE regions, not scored either way.")
    ap.add_argument("--save-vis", action="store_true",
                    help="write annotated images (GT green=hit red=miss grey=ignored, pred blue).")
    ap.add_argument("--granularity", choices=["word", "line", "both"], default="both",
                    help="score against word boxes, merged line boxes, or both (default).")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grans = ["word", "line"] if args.granularity == "both" else [args.granularity]
    for g in grans:
        run_one(args, g)


def run_one(args, granularity):
    gt = load_gt(args.images_dir, set(args.skip_text), granularity)
    stems = sorted(gt)[:args.limit] if args.limit else sorted(gt)
    n_valid = sum(len(gt[s][1]) for s in stems)
    n_ign = sum(len(gt[s][2]) for s in stems)
    print(f"\n########## granularity = {granularity.upper()} ##########")
    print(f"[data] {len(stems)} images | {n_valid} valid GT boxes | {n_ign} ignored "
          f"('{','.join(args.skip_text)}')\n")
    if not n_valid:
        raise SystemExit("no valid GT boxes — check --images-dir / the json format")

    from paddleocr import TextDetection

    lim = dict(limit_side_len=args.det_limit_side_len, limit_type=args.det_limit_type)
    todo = [(n, dict(model_name=n, **lim)) for n in args.models]
    if args.ft_det_dir:
        todo.append((args.ft_det_label,
                     dict(model_name=args.ft_det_name, model_dir=args.ft_det_dir, **lim)))
    if not todo:
        raise SystemExit("nothing to evaluate — pass --models and/or --ft-det-dir")

    summary = []
    for name, kw in todo:
        print(f"[{name}] loading {kw}")
        try:
            model = TextDetection(**kw)
        except Exception as e:
            print(f"  [failed to load: {e}]")
            continue

        vis_dir = os.path.join(args.out, f"{name}_{granularity}_vis")
        if args.save_vis:
            os.makedirs(vis_dir, exist_ok=True)

        rows, tot = [], defaultdict(int)
        alt = {t: [0, 0, 0] for t in args.extra_iou}
        for i, stem in enumerate(stems, 1):
            img_path, valid, ignored = gt[stem]
            try:
                preds = []
                for res in model.predict(input=img_path):
                    preds.extend(extract_polys(res))
            except Exception as e:
                print(f"  [{stem}] predict failed: {e}")
                preds = []

            tp, fp, fn, pairs, ign = match(preds, valid, ignored, args.iou)
            p, r, f = prf(tp, fp, fn)
            rows.append([stem, len(valid), len(ignored), len(preds), tp, fp, fn, ign,
                         f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
                         f"{sum(v for _, _, v in pairs) / tp:.4f}" if tp else "NA"])
            tot["tp"] += tp; tot["fp"] += fp; tot["fn"] += fn
            tot["pred"] += len(preds); tot["ign"] += ign
            for t in args.extra_iou:
                a, b, c, _, _ = match(preds, valid, ignored, t)
                alt[t][0] += a; alt[t][1] += b; alt[t][2] += c
            if args.save_vis:
                draw(img_path, preds, valid, ignored, pairs,
                     os.path.join(vis_dir, f"{stem}.jpg"))
            print(f"    {i}/{len(stems)}", end="\r")
        print()

        csv_path = os.path.join(args.out, f"{name}_{granularity}_per_image.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "gt_valid", "gt_ignored", "n_pred", "TP", "FP", "FN",
                         "pred_on_ignored", "precision", "recall", "hmean", "mean_IoU_of_TP"])
            wr.writerows(rows)
            P, R, F = prf(tot["tp"], tot["fp"], tot["fn"])
            wr.writerow(["__POOLED__", n_valid, n_ign, tot["pred"], tot["tp"], tot["fp"],
                         tot["fn"], tot["ign"], f"{P:.4f}", f"{R:.4f}", f"{F:.4f}", ""])
        print(f"  -> {csv_path}")
        if args.save_vis:
            print(f"  -> {vis_dir}/")

        P, R, F = prf(tot["tp"], tot["fp"], tot["fn"])
        summary.append((name, P, R, F, tot["tp"], tot["fp"], tot["fn"], tot["pred"],
                        {t: prf(*alt[t]) for t in args.extra_iou}))

    print(f"\n===== Detection [{granularity}], pooled over {len(stems)} images "
          f"({n_valid} valid GT boxes, IoU>={args.iou}) =====")
    print(f"{'model':26s}{'P':>8s}{'R':>8s}{'F1':>8s}{'TP':>7s}{'FP':>7s}{'FN':>7s}{'preds':>7s}")
    for name, P, R, F, tp, fp, fn, npred, _ in summary:
        print(f"{name:26s}{P:>8.4f}{R:>8.4f}{F:>8.4f}{tp:>7d}{fp:>7d}{fn:>7d}{npred:>7d}")
    for t in args.extra_iou:
        print(f"\n-- IoU>={t}")
        for name, *_, extra in summary:
            p, r, f = extra[t]
            print(f"{name:26s}{p:>8.4f}{r:>8.4f}{f:>8.4f}")
    print("=" * 78)
    print("recall = fraction of real GT words the detector found; "
          "precision = fraction of its boxes that are real words.")


if __name__ == "__main__":
    main()
