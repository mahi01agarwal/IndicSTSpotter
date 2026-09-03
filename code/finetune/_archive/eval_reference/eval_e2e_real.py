#!/usr/bin/env python3
"""END-TO-END evaluation on real GT-box images: detector + recogniser.

Where the other two scripts each isolate one half —
    eval_real_perimage.py : recogniser only, fed perfect GT boxes
    eval_det_real.py      : detector only, text ignored
— this one runs the real pipeline and charges it for BOTH kinds of error, which is
what makes the result comparable to an end-to-end system like GoogleOCR.

Pipeline per image: detector -> quads -> perspective-crop each -> recogniser -> text.
Predictions are then matched one-to-one against the GT word boxes by polygon IoU
(highest first), and every box is accounted for:

  matched pair          -> char edits = levenshtein(gt_text, pred_text)
                           word       = 0 if exact else 1 substitution
  GT word never matched -> char edits = len(gt_text)     (detector MISS: deletions)
                           word       = 1 deletion
  prediction unmatched  -> char edits = len(pred_text)   (FALSE POSITIVE: insertions)
                           word       = 1 insertion

  CER = total char edits / total GT chars      CRR = 1 - CER   (clamped to [0,1])
  WER = (S + D + I)      / total GT words      WRR = 1 - WER   (clamped to [0,1])

A prediction landing on an "NA" placeholder box is discarded rather than charged as
a false positive — the annotator marked those unreadable, so finding them is not an
error. Those boxes are also excluded from the GT totals.

Reported both ways: micro (pooled over all pages) and macro (mean of per-page scores).
--no-fp-penalty additionally reports the variant that ignores false positives, which
isolates "how well did it read what it found" from "how much did it hallucinate".

Needs: pillow, opencv (ships with paddleocr), paddleocr.

Usage
  python eval_e2e_real.py --images-dir $SCR/real_marathi \\
      --rec-dir $SCR/PaddleOCR/output/v6_unified_infer --out $SCR/e2e_marathi
"""
import os
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import jiwer                 # standard WER/CER; `pip install jiwer` in the env
except ImportError:
    jiwer = None

# geometry + detector-output parsing live in the detector eval; reuse them
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_det_real import iou, poly_area, extract_polys          # noqa: E402


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# --------------------------------------------------------------------- jiwer
def reading_order_text(items):
    """items: [(rect_or_poly, text)] -> one reading-order string (lines top->bottom
    by y-centre with a height-tolerant band, words left->right within a line).
    Used symmetrically for GT and predictions so the page-level WER is a fair
    sequence comparison independent of the GT 'line' field."""
    boxes = []
    for pts, text in items:
        ys = [float(p[1]) for p in pts]
        xs = [float(p[0]) for p in pts]
        boxes.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), text))
    if not boxes:
        return ""
    med_h = sorted(b[2] for b in boxes)[len(boxes) // 2] or 1.0
    tol = 0.6 * med_h
    boxes.sort(key=lambda b: (b[0], b[1]))
    lines, cur, anchor = [], [], None
    for yc, x0, _, text in boxes:
        if anchor is None or abs(yc - anchor) <= tol:
            cur.append((x0, text))
            anchor = yc if anchor is None else anchor
        else:
            lines.append(cur); cur, anchor = [(x0, text)], yc
    if cur:
        lines.append(cur)
    return " ".join(" ".join(t for _, t in sorted(ln, key=lambda w: w[0])) for ln in lines)


def jiwer_scores(refs, hyps):
    """Corpus WER/CER via jiwer over aligned ref/hyp lists (empty refs dropped)."""
    R = [r for r, h in zip(refs, hyps) if r.strip()]
    H = [h for r, h in zip(refs, hyps) if r.strip()]
    if not R:
        return None
    wer = jiwer.wer(R, H)
    cer = jiwer.cer(R, H)
    try:                                            # jiwer >=3
        w = jiwer.process_words(R, H)
        sdi = (w.substitutions, w.deletions, w.insertions, w.hits)
    except AttributeError:                          # jiwer 2.x
        m = jiwer.compute_measures(R, H)
        sdi = (m["substitutions"], m["deletions"], m["insertions"], m["hits"])
    return dict(wer=wer, cer=cer, wrr=clamp01(1 - wer), crr=clamp01(1 - cer),
                S=sdi[0], D=sdi[1], I=sdi[2], H=sdi[3], n=len(R))


# ----------------------------------------------------------------------- data
def load_gt(images_dir, skip_texts):
    """-> {stem: (img_path, [(rect, text)], [ignored_rect])}"""
    out = {}
    for jp in sorted(glob.glob(os.path.join(images_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = next((os.path.join(images_dir, f"{stem}.{e}")
                    for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                    if os.path.isfile(os.path.join(images_dir, f"{stem}.{e}"))), None)
        if img is None:
            print(f"[skip] no image for {jp}")
            continue
        valid, ignored = [], []
        for w in json.load(open(jp, encoding="utf-8")).get("words", []):
            t = str(w.get("text", "")).strip()
            if not t:
                continue
            x, y, ww, hh = int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"])
            if ww <= 0 or hh <= 0:
                continue
            rect = [(x, y), (x + ww, y), (x + ww, y + hh), (x, y + hh)]
            if t in skip_texts:
                ignored.append(rect)
            else:
                valid.append((rect, t))
        out[stem] = (img, valid, ignored)
    return out


def crop_quad(img_bgr, quad, pad=0):
    """Perspective-warp a detected quad to an upright crop — the same thing the
    PaddleOCR pipeline does internally. Falls back to the axis-aligned bounding
    box for polygons with more than 4 points (curved text)."""
    import cv2
    pts = np.array(quad, dtype=np.float32)
    if len(pts) != 4:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(img_bgr.shape[1], int(x1) + pad), min(img_bgr.shape[0], int(y1) + pad)
        return img_bgr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None

    # order TL,TR,BR,BL
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    box = np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                    pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[3] - box[2])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h))


def rec_text(res):
    try:
        d = res.json
        d = d.get("res", d) if isinstance(d, dict) else {}
        return d.get("rec_text") or ""
    except Exception:
        return ""


# ------------------------------------------------------------------- scoring
def score_image(gt_words, preds_text, pairs, matched_pred, ignored_hits, fp_penalty=True):
    """-> dict of raw counts for this image."""
    c_edits = c_ref = 0
    subs = dele = ins = 0
    for gi, (_, gt_text) in enumerate(gt_words):
        c_ref += len(gt_text)
    matched_gt = {gi for _, gi, _ in pairs}
    for pi, gi, _ in pairs:
        gt_text = gt_words[gi][1]
        pred = preds_text[pi]
        c_edits += levenshtein(gt_text, pred)
        subs += (gt_text != pred)
    for gi, (_, gt_text) in enumerate(gt_words):
        if gi not in matched_gt:
            c_edits += len(gt_text)          # detector missed it entirely
            dele += 1
    if fp_penalty:
        for pi, txt in enumerate(preds_text):
            if pi not in matched_pred:
                c_edits += len(txt)          # hallucinated text
                ins += 1
    return dict(c_edits=c_edits, c_ref=c_ref, subs=subs, dele=dele, ins=ins,
                n_gt=len(gt_words), n_pred=len(preds_text), tp=len(pairs),
                ign=ignored_hits)


def rates(a):
    cer = a["c_edits"] / a["c_ref"] if a["c_ref"] else 0.0
    wer = (a["subs"] + a["dele"] + a["ins"]) / a["n_gt"] if a["n_gt"] else 0.0
    return cer, wer, clamp01(1 - cer), clamp01(1 - wer)


def main():
    ap = argparse.ArgumentParser(description="End-to-end (detector + recogniser) eval.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="./e2e_eval")
    ap.add_argument("--det-name", default="PP-OCRv6_medium_det")
    ap.add_argument("--det-dir", default=None, help="fine-tuned detector inference dir.")
    ap.add_argument("--det-limit-side-len", type=int, default=960,
                    help="detector inference resize cap; raise (1280/1920/2560) for dense/high-res "
                         "scenes. 960 = PP-OCR default.")
    ap.add_argument("--det-limit-type", default="max", choices=["max", "min"])
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--rec-dir", default=None,
                    help="fine-tuned recogniser inference dir (omit for the stock one).")
    ap.add_argument("--label", default=None, help="name for this run in the outputs.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--skip-text", nargs="*", default=["NA"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-fp-penalty", action="store_true",
                    help="also report the variant that does not charge false positives.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    label = args.label or f"{args.det_name}+{'ft_rec' if args.rec_dir else args.rec_name}"
    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    gt = load_gt(args.images_dir, set(args.skip_text))
    stems = sorted(gt)[:args.limit] if args.limit else sorted(gt)
    tot_gt = sum(len(gt[s][1]) for s in stems)
    print(f"[data] {len(stems)} images | {tot_gt} scoreable GT words | "
          f"{sum(len(gt[s][2]) for s in stems)} ignored\n[run ] {label}\n")

    import cv2
    from paddleocr import TextDetection, TextRecognition

    det_kw = dict(model_name=args.det_name,
                  limit_side_len=args.det_limit_side_len, limit_type=args.det_limit_type)
    if args.det_dir:
        det_kw["model_dir"] = args.det_dir
    rec_kw = dict(model_name=args.rec_name)
    if args.rec_dir:
        rec_kw["model_dir"] = args.rec_dir
    print(f"  detector  {det_kw}\n  recogniser {rec_kw}")
    det = TextDetection(**det_kw)
    rec = TextRecognition(**rec_kw)

    # ---- 1. detect + crop everything first, so recognition can batch
    per_img_polys, crop_paths, crop_owner = {}, [], []
    for i, stem in enumerate(stems, 1):
        img_path, valid, ignored = gt[stem]
        polys = []
        try:
            for res in det.predict(input=img_path):
                polys.extend(extract_polys(res))
        except Exception as e:
            print(f"  [{stem}] detect failed: {e}")
        img = cv2.imread(img_path)
        kept = []
        for k, q in enumerate(polys):
            c = crop_quad(img, q) if img is not None else None
            if c is None or c.size == 0:
                continue
            cp = os.path.join(crops_dir, f"{stem}_{k}.jpg")
            cv2.imwrite(cp, c)
            kept.append(q)
            crop_paths.append(cp)
            crop_owner.append((stem, len(kept) - 1))
        per_img_polys[stem] = kept
        print(f"    detect {i}/{len(stems)}", end="\r")
    print(f"\n[det ] {len(crop_paths)} crops from {len(stems)} images")

    # ---- 2. recognise them all
    texts = []
    for i in range(0, len(crop_paths), args.batch_size):
        chunk = crop_paths[i:i + args.batch_size]
        for res in rec.predict(input=chunk, batch_size=len(chunk)):
            texts.append(rec_text(res))
        print(f"    recognise {min(i + args.batch_size, len(crop_paths))}/{len(crop_paths)}", end="\r")
    print()
    if len(texts) != len(crop_paths):
        raise RuntimeError(f"{len(texts)} preds for {len(crop_paths)} crops")
    by_img = defaultdict(dict)
    for (stem, k), t in zip(crop_owner, texts):
        by_img[stem][k] = t

    # ---- 3. match + score
    rows, agg = [], defaultdict(int)
    agg_nofp = defaultdict(int)
    per_page_rates = []
    rec_refs, rec_hyps = [], []          # jiwer: recognition on matched boxes
    e2e_refs, e2e_hyps = [], []          # jiwer: end-to-end page reading-order
    for stem in stems:
        _, valid, ignored = gt[stem]
        polys = per_img_polys[stem]
        ptexts = [by_img[stem].get(k, "") for k in range(len(polys))]

        cand = []
        for pi, p in enumerate(polys):
            for gi, (g, _) in enumerate(valid):
                v = iou(p, g)
                if v >= args.iou:
                    cand.append((v, pi, gi))
        cand.sort(reverse=True)
        used_p, used_g, pairs = set(), set(), []
        for v, pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi); pairs.append((pi, gi, v))
        ign_hits = 0
        for pi, p in enumerate(polys):
            if pi in used_p:
                continue
            if any(iou(p, g) >= args.iou for g in ignored):
                ign_hits += 1
                used_p.add(pi)

        for pi, gi, _ in pairs:
            rec_refs.append(valid[gi][1]); rec_hyps.append(ptexts[pi])
        e2e_refs.append(reading_order_text(valid))
        e2e_hyps.append(reading_order_text(list(zip(polys, ptexts))))

        a = score_image(valid, ptexts, pairs, used_p, ign_hits, fp_penalty=True)
        b = score_image(valid, ptexts, pairs, used_p, ign_hits, fp_penalty=False)
        cer, wer, crr, wrr = rates(a)
        per_page_rates.append((cer, wer, crr, wrr))
        rows.append([stem, a["n_gt"], a["n_pred"], a["tp"], a["ign"],
                     a["subs"], a["dele"], a["ins"],
                     f"{cer:.4f}", f"{wer:.4f}", f"{crr:.4f}", f"{wrr:.4f}"])
        for k, v in a.items():
            agg[k] += v
        for k, v in b.items():
            agg_nofp[k] += v
        print(f"    score {len(rows)}/{len(stems)}", end="\r")
    print()

    csv_path = os.path.join(args.out, f"{label.replace('/', '_')}_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "gt_words", "det_boxes", "det_matched", "pred_on_ignored",
                     "subs", "deletions", "insertions", "CER", "WER", "CRR", "WRR"])
        wr.writerows(rows)
        cer, wer, crr, wrr = rates(agg)
        wr.writerow(["__MICRO__", agg["n_gt"], agg["n_pred"], agg["tp"], agg["ign"],
                     agg["subs"], agg["dele"], agg["ins"],
                     f"{cer:.4f}", f"{wer:.4f}", f"{crr:.4f}", f"{wrr:.4f}"])
        n = len(per_page_rates) or 1
        wr.writerow(["__MACRO__", "", "", "", "", "", "", "",
                     f"{sum(r[0] for r in per_page_rates)/n:.4f}",
                     f"{sum(r[1] for r in per_page_rates)/n:.4f}",
                     f"{sum(r[2] for r in per_page_rates)/n:.4f}",
                     f"{sum(r[3] for r in per_page_rates)/n:.4f}"])
    print(f"  -> {csv_path}")

    cer, wer, crr, wrr = rates(agg)
    rec_recall = 100.0 * agg["tp"] / agg["n_gt"] if agg["n_gt"] else 0.0
    print(f"\n===== END-TO-END: {label} =====")
    print(f"  GT words (ground truth)      : {agg['n_gt']}")
    print(f"  detector boxes produced      : {agg['n_pred']}")
    print(f"  of those matching a GT word  : {agg['tp']}   (detection recall {rec_recall:.2f}%)")
    print(f"  landed on ignored 'NA' boxes : {agg['ign']}")
    print(f"  substitutions/deletions/ins  : {agg['subs']} / {agg['dele']} / {agg['ins']}")
    print(f"  CER {cer:.4f}   WER {wer:.4f}   CRR {crr:.4f}   WRR {wrr:.4f}   (micro)")
    n = len(per_page_rates) or 1
    print(f"  CER {sum(r[0] for r in per_page_rates)/n:.4f}   "
          f"WER {sum(r[1] for r in per_page_rates)/n:.4f}   "
          f"CRR {sum(r[2] for r in per_page_rates)/n:.4f}   "
          f"WRR {sum(r[3] for r in per_page_rates)/n:.4f}   (macro)")
    if args.no_fp_penalty:
        c2, w2, cr2, wr2 = rates(agg_nofp)
        print(f"  -- without false-positive penalty: "
              f"CER {c2:.4f} WER {w2:.4f} CRR {cr2:.4f} WRR {wr2:.4f}")
    print("=" * 60)

    # ---- jiwer (standard WER/CER): recognition-on-matched + end-to-end page ----
    with open(os.path.join(args.out, "rec_pairs.jsonl"), "w", encoding="utf-8") as fh:
        for r, h in zip(rec_refs, rec_hyps):
            fh.write(json.dumps({"gt": r, "pred": h}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "e2e_pages.jsonl"), "w", encoding="utf-8") as fh:
        for s, r, h in zip(stems, e2e_refs, e2e_hyps):
            fh.write(json.dumps({"image": s, "gt": r, "pred": h}, ensure_ascii=False) + "\n")
    if jiwer is None:
        print("\n[jiwer] not installed — `pip install jiwer` for standard WER/CER "
              "(pairs dumped to rec_pairs.jsonl / e2e_pages.jsonl for later scoring).")
    else:
        rj, ej = jiwer_scores(rec_refs, rec_hyps), jiwer_scores(e2e_refs, e2e_hyps)
        print(f"\n===== jiwer: {label} =====")
        if rj:
            print(f"  RECOGNITION (matched boxes, {rj['n']} words):  "
                  f"WER {rj['wer']:.4f}  CER {rj['cer']:.4f}  WRR {rj['wrr']:.4f}  CRR {rj['crr']:.4f}"
                  f"   S/D/I/H {rj['S']}/{rj['D']}/{rj['I']}/{rj['H']}")
        if ej:
            print(f"  END-TO-END  (page reading-order, {ej['n']} pages):  "
                  f"WER {ej['wer']:.4f}  CER {ej['cer']:.4f}  WRR {ej['wrr']:.4f}  CRR {ej['crr']:.4f}"
                  f"   S/D/I/H {ej['S']}/{ej['D']}/{ej['I']}/{ej['H']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
