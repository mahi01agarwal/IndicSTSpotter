#!/usr/bin/env python3
"""Convert the synthetic dataset annotations into PaddleOCR DETECTION label files.

PaddleOCR det (DB) trains on FULL images + a label file, one line per image:
    <relative_image_path>\t[{"transcription": "<text>", "points": [[x,y],[x,y],[x,y],[x,y]]}, ...]

Our annotations/scene_XXXXX.json already carry per-word quad (4 pts) / polygon (curved)
+ text, so this just reshapes them. Curved instances use their polygon; the rest use the
quad. Boxes smaller than --min-size are written as "###" (PaddleOCR's ignore token: the
region is masked out of the loss instead of penalising the detector for missing sub-pixel text).

Output (in --out-dir):  det_train.txt, det_val.txt
Image paths are RELATIVE to --data-dir (default: the dataset dir) -> point PaddleOCR's
Global/Train/Eval data_dir there.

Run (include curved — the detector benefits from polygons, so NO skip-curved here):
    python make_det_labels.py --dataset-dir $SCR/out_dataset --out-dir $SCR/det_data
"""
import os
import json
import glob
import random
import argparse


def _pairs(seq):
    """Accept a flat [x1,y1,x2,y2,...] or nested [[x1,y1],...] and return rounded [[x,y],...]."""
    if not seq:
        return []
    if isinstance(seq[0], (list, tuple)):
        return [[round(float(p[0])), round(float(p[1]))] for p in seq]
    return [[round(float(seq[i])), round(float(seq[i + 1]))] for i in range(0, len(seq) - 1, 2)]


def _order_quad(pts):
    """Order 4 points as TL,TR,BR,BL (clockwise) so the quad is a SIMPLE,
    non-self-intersecting polygon. DB's MakeShrinkMap runs a pyclipper offset on
    EVERY box (even the ignored '###' ones); a bowtie / mis-ordered quad makes
    that offset spin forever -> dataloader workers peg at 100% CPU, GPUs idle,
    no batch ever completes. Standard sum/diff ordering, robust for quads."""
    pts = [list(p) for p in pts[:4]]
    s = [p[0] + p[1] for p in pts]
    d = [p[0] - p[1] for p in pts]
    tl = pts[s.index(min(s))]
    br = pts[s.index(max(s))]
    tr = pts[d.index(max(d))]
    bl = pts[d.index(min(d))]
    return [tl, tr, br, bl]


def _min_edge(pts):
    """Shortest edge length — near-zero means a degenerate (collinear/sliver) quad."""
    import math
    n = len(pts)
    return min(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n))


def to_points(inst):
    # ALWAYS use the 4-point quad, NEVER the traced curved `polygon`. A curved
    # polygon can be self-intersecting / near-degenerate, which makes DB's
    # MakeShrinkMap (pyclipper offset) spin forever — the det dataloader workers
    # burn 100% CPU and no batch ever completes. Every instance has a valid quad
    # (curved words carry their axis box), so quads are clean and lossless for DB.
    q = inst.get("quad")
    if q and len(q) >= 8:
        return _order_quad(_pairs(q[:8]))
    return _pairs(inst.get("polygon", []))


def _poly_area(pts):
    """Shoelace area (no cv2 dependency)."""
    n, s = len(pts), 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def sentinel_ignore_box():
    """A single tiny '###' box, used to keep a TEXT-FREE image in the dataset.

    PaddleOCR's DetLabelEncode returns None when a line has zero boxes, and
    SimpleDataSet then silently resamples a different index -- so a hard negative
    written as `path\\t[]` never trains on anything. You would believe you had
    trained on background negatives and you would not have.

    Giving the image one '###' (ignore) box keeps the sample alive. The ignore
    region is masked out of the loss, while EVERY OTHER PIXEL is supervised as
    background -- which is exactly the hard-negative signal we want. 8x8 clears
    both our degenerate-quad filters and DB's pyclipper shrink.
    """
    return [{"transcription": "###", "points": [[0, 0], [8, 0], [8, 8], [0, 8]]}]


def main():
    ap = argparse.ArgumentParser(description="Synthetic annotations -> PaddleOCR det label files.")
    ap.add_argument("--dataset-dir", required=True, help="out_dataset/ (has images/ + annotations/).")
    ap.add_argument("--out-dir", default="./det_data")
    ap.add_argument("--data-dir", default=None,
                    help="Path the image paths are relative to (default: --dataset-dir).")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=4,
                    help="Boxes with w or h below this are marked '###' (ignored in the det loss).")
    ap.add_argument("--drop-negatives", action="store_true",
                    help="Drop text-free images instead of keeping them as hard negatives. "
                         "Keeping them is the point: they teach the detector to REJECT "
                         "real background clutter (awnings, grilles, trusses).")
    ap.add_argument("--max-negative-frac", type=float, default=0.30,
                    help="Cap negatives at this fraction of the label set, so they cannot "
                         "swamp the positives and suppress recall.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann_dir = os.path.join(args.dataset_dir, "annotations")
    data_dir = os.path.abspath(args.data_dir or args.dataset_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    lines, neg_lines = [], []
    n_img = n_box = n_ignore = n_degen = 0
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        ann = json.load(open(ann_path, encoding="utf-8"))
        img_rel = f"images/{ann['image_file']}"          # relative to data_dir
        labels = []
        for inst in ann.get("instances", []):
            text = (inst.get("text") or "").strip()
            if not text:
                continue
            pts = to_points(inst)
            if len(pts) < 4:                             # DB needs >= 4 points
                continue
            if _poly_area(pts) < 8.0 or _min_edge(pts) < 2.0:  # drop degenerate/sliver quads
                n_degen += 1                             # (these are what hang pyclipper)
                continue
            x, y, w, h = inst.get("bbox", [0, 0, 0, 0])
            ignore = (w < args.min_size or h < args.min_size)
            labels.append({"transcription": "###" if ignore else text, "points": pts})
            n_box += 1
            n_ignore += int(ignore)
        if not labels:
            # Text-free image. Either the generator marked it a hard negative, or
            # every box got filtered out. Both are usable as background-only
            # supervision; keep unless explicitly told not to.
            if not args.drop_negatives:
                neg_lines.append(
                    f"{img_rel}\t{json.dumps(sentinel_ignore_box(), ensure_ascii=False)}")
            continue
        lines.append(f"{img_rel}\t{json.dumps(labels, ensure_ascii=False)}")
        n_img += 1

    if not lines:
        raise SystemExit(f"no labelled images found under {ann_dir}")

    rng = random.Random(args.seed)

    # cap negatives so they cannot swamp the positives and suppress recall
    n_neg_kept = 0
    if neg_lines:
        cap = int(len(lines) * args.max_negative_frac / max(1e-9, 1.0 - args.max_negative_frac))
        rng.shuffle(neg_lines)
        neg_lines = neg_lines[:cap]
        n_neg_kept = len(neg_lines)
        lines.extend(neg_lines)

    random.Random(args.seed).shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_frac))
    val, train = lines[:n_val], lines[n_val:]
    for split, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"det_{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    print(f"[det] images={n_img}  boxes={n_box}  ignored(<{args.min_size}px)={n_ignore}  dropped_degenerate={n_degen}")
    if n_neg_kept:
        print(f"      hard negatives (text-free, sentinel '###' box) = {n_neg_kept} "
              f"({n_neg_kept / float(n_img + n_neg_kept) * 100:.0f}% of the set)")
    elif not args.drop_negatives:
        print("      hard negatives = 0  <- dataset has no text-free images; regenerate "
              "with generate_realistic.py --negative-frac 0.2 to add them")
    print(f"      train={len(train)}  val={len(val)}  -> {args.out_dir}/det_{{train,val}}.txt")
    print(f"      mean boxes/image (positives) = {n_box / float(max(1, n_img)):.1f} "
          f"— real scene photos carry ~11; check with det_dataset_stats.py")
    print(f"      PaddleOCR Global data_dir must be:  {data_dir}")


if __name__ == "__main__":
    main()
