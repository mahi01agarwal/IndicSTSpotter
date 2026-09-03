#!/usr/bin/env python3
"""eval_e2e_bharat_perscript.py — END-TO-END, PER-SCRIPT eval on the Bharat DET test set,
with a false-positive penalty. The FP-penalized companion to eval_rec_perscript.py.

IndicSTR12 scenes are only partially annotated (its detector produces ~2.5x the annotated
words -> every un-annotated real word is charged as a false positive, so page-level e2e is
meaningless there). Bharat detection GT is DENSE — every word is annotated, illegible ones
marked '###' — so an unmatched detector box is a *genuine* false positive. That makes a
trustworthy FP-penalized per-script e2e possible.

Input: a PaddleOCR DET label file (bharat_det_test.txt), each line
    <img_path>\t[{"transcription": "...", "points": [[x,y],...]}, ...]
'###' (or empty) transcription = ignore region (illegible): a prediction landing on one is
DISCARDED, not charged as a false positive (matches eval_e2e_real's NA handling).

Pipeline: detector -> quads -> crop -> recogniser -> text; predictions IoU-matched to GT.
Each contribution is bucketed by script (majority Unicode block of the relevant text):
    matched pair        -> GT word's script   (substitution if misread)
    GT word not matched -> GT word's script   (deletion; detector miss)
    prediction unmatched-> PREDICTED text's script (insertion; false positive)
Per script:  WER = (S+D+I)/N_gt,  CER = (char_edits + fp_chars)/gt_chars,  CRR/WRR = 1-*.
jiwer aligns the matched+deleted pairs (real refs); false-positive text is added as pure
insertions by count — which avoids jiwer's empty-reference error while still penalising FPs.

Pass 2+ recognisers (detection is shared) to get a per-script CRR comparison in one run —
e.g. synth-only vs balanced — the A'/B tie-breaker WITH the FP penalty.

    python eval_e2e_bharat_perscript.py \
        --labels $SCR/det_data_real/bharat/bharat_det_test.txt \
        --det-dir $SCR/PaddleOCR/output/v6_det_real_infer_latest \
        --model synth    $SCR/PaddleOCR/output/v6_unified_infer \
        --model balanced $SCR/PaddleOCR/output/v6_rec_synthreal_infer \
        --out $SCR/e2e_bharat_perscript

Needs: jiwer, opencv, paddleocr.  Run in the $SCR/ocr env on gnode063.
"""
import os
import sys
import json
import csv
import argparse
import collections

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_e2e_real import crop_quad, rec_text            # noqa: E402
from eval_det_real import iou, extract_polys             # noqa: E402

import jiwer

# script inference (majority Unicode block; identical to merge_rec_db.py / eval_rec_perscript.py)
BLOCKS = [
    ('devanagari', [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ('bengali',    [(0x0980, 0x09FF)]),
    ('gurmukhi',   [(0x0A00, 0x0A7F)]),
    ('gujarati',   [(0x0A80, 0x0AFF)]),
    ('oriya',      [(0x0B00, 0x0B7F)]),
    ('tamil',      [(0x0B80, 0x0BFF)]),
    ('telugu',     [(0x0C00, 0x0C7F)]),
    ('kannada',    [(0x0C80, 0x0CFF)]),
    ('malayalam',  [(0x0D00, 0x0D7F)]),
]
SCRIPT_ORDER = ['devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya',
                'tamil', 'telugu', 'kannada', 'malayalam', 'latin', 'other']


def script_of(text):
    counts = collections.Counter()
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x0030 <= cp <= 0x0039):
            counts['latin'] += 1; continue
        for name, ranges in BLOCKS:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1; break
    if not counts:
        return 'other'
    return counts.most_common(1)[0][0]


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def word_counts(R, H):
    """-> (S, D, I, H) word counts via jiwer, tolerant of version."""
    try:
        o = jiwer.process_words(R, H)
        return o.substitutions, o.deletions, o.insertions, o.hits
    except AttributeError:
        m = jiwer.compute_measures(R, H)
        return m["substitutions"], m["deletions"], m["insertions"], m["hits"]


def load_det_labels(label_file):
    """-> [(img_path, [(pts, text)] valid, [pts] ignore)]."""
    data = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            img, js = line.split("\t", 1)
            try:
                boxes = json.loads(js)
            except json.JSONDecodeError:
                continue
            valid, ignore = [], []
            for b in boxes:
                pts = b.get("points")
                t = (b.get("transcription") or "").strip()
                if not pts:
                    continue
                if t == "###" or not t:
                    ignore.append(pts)
                else:
                    valid.append((pts, t))
            data.append((img, valid, ignore))
    return data


def per_script_metrics(bucket):
    """bucket[script] = dict(m_ref, m_hyp, d_ref, i_hyp) -> {script/'__ALL__': metrics}."""
    out = {}
    tot = dict(m_ref=[], m_hyp=[], d_ref=[], i_hyp=[])
    for s, b in bucket.items():
        for k in tot:
            tot[k].extend(b[k])
    for s in list(bucket) + ['__ALL__']:
        b = tot if s == '__ALL__' else bucket[s]
        R = list(b['m_ref']) + list(b['d_ref'])
        H = list(b['m_hyp']) + [""] * len(b['d_ref'])
        if R:
            S, D, Imd, Hh = word_counts(R, H)
            ref_words = Hh + S + D
            ref_chars = sum(len(r) for r in R)
            char_edits = jiwer.cer(R, H) * ref_chars
        else:
            S = D = Imd = Hh = ref_words = ref_chars = char_edits = 0
        fp = [p for p in b['i_hyp'] if p.strip()]
        fp_words = sum(len(p.split()) or 1 for p in fp)
        fp_chars = sum(len(p) for p in fp)
        wer = (S + D + Imd + fp_words) / ref_words if ref_words else (1.0 if fp_words else 0.0)
        cer = (char_edits + fp_chars) / ref_chars if ref_chars else (1.0 if fp_chars else 0.0)
        rec = None
        if b['m_ref']:
            rc = jiwer.cer(list(b['m_ref']), list(b['m_hyp']))
            rw = jiwer.wer(list(b['m_ref']), list(b['m_hyp']))
            rec = dict(cer=rc, wer=rw, crr=clamp01(1 - rc), wrr=clamp01(1 - rw))
        # non-FP-penalised, misses-count metrics (never floor from over-firing):
        #   recall  = matched / GT words   (detection recall)
        #   wrr_nofp = correct words / GT words   (pipeline word accuracy; == "correct/GT")
        #   crr_nofp = 1 - (char edits / GT chars), without charging FP characters
        recall = (Hh + S) / ref_words if ref_words else 0.0
        wrr_nofp = Hh / ref_words if ref_words else 0.0
        crr_nofp = clamp01(1 - (char_edits / ref_chars)) if ref_chars else 0.0
        out[s] = dict(n_gt=ref_words, n_fp=fp_words,
                      cer=cer, wer=wer, crr=clamp01(1 - cer), wrr=clamp01(1 - wer),
                      rec=rec, n_matched=len(b['m_ref']), n_correct=Hh,
                      recall=recall, wrr_nofp=wrr_nofp, crr_nofp=crr_nofp)
    # __MACRO__ = unweighted mean over the listed scripts (matches BSTD's "Avg" column).
    macro = [s for s in SCRIPT_ORDER if s != 'other' and s in out]
    if macro:
        mean = lambda k: sum(out[s][k] for s in macro) / len(macro)
        out['__MACRO__'] = dict(
            n_gt=sum(out[s]['n_gt'] for s in macro), n_fp=sum(out[s]['n_fp'] for s in macro),
            n_matched=sum(out[s]['n_matched'] for s in macro),
            n_correct=sum(out[s]['n_correct'] for s in macro),
            cer=mean('cer'), wer=mean('wer'), crr=mean('crr'), wrr=mean('wrr'),
            recall=mean('recall'), wrr_nofp=mean('wrr_nofp'), crr_nofp=mean('crr_nofp'), rec=None)
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-script FP-penalized end-to-end eval on Bharat det test.")
    ap.add_argument("--labels", required=True, help="bharat_det_test.txt (DET label file).")
    ap.add_argument("--det-dir", default=None,
                    help="fine-tuned detector inference dir; OMIT to use the STOCK --det-name.")
    ap.add_argument("--det-name", default="PP-OCRv6_medium_det")
    ap.add_argument("--det-limit-side-len", type=int, default=960,
                    help="detector inference resize cap. Raise (1280/1920/2560) to keep small "
                         "words detectable in dense high-res scenes. 960 = PP-OCR default.")
    ap.add_argument("--det-limit-type", default="max", choices=["max", "min"],
                    help="'max': longest side <= N (default). 'min': shortest side >= N.")
    ap.add_argument("--det-box-thresh", type=float, default=None,
                    help="detector box score threshold; RAISE (0.7-0.85) to cut false positives on "
                         "dense scenes. Default = model built-in (~0.6). Trades recall for precision.")
    ap.add_argument("--det-unclip-ratio", type=float, default=None,
                    help="detector box expansion ratio; LOWER (e.g. 1.5) to shrink over-merged boxes.")
    ap.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "REC_DIR"),
                    default=[], help="a recogniser: LABEL and its inference dir. Repeatable.")
    ap.add_argument("--stock", action="store_true", help="also score stock PP-OCRv6_medium_rec.")
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--out", default="./e2e_bharat_perscript")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-h", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="cap images (debug).")
    args = ap.parse_args()

    models = list(args.model)
    if args.stock:
        models.append(("stock", None))
    if not models:
        raise SystemExit("no recognisers — pass --model LABEL DIR (repeatable) and/or --stock")

    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    data = load_det_labels(args.labels)
    if args.limit:
        data = data[:args.limit]
    tot_gt = sum(len(v) for _, v, _ in data)
    print(f"[data] {len(data)} images | {tot_gt} legible GT words | "
          f"{sum(len(ig) for _, _, ig in data)} ignore boxes\n")

    from paddleocr import TextDetection, TextRecognition
    det_kw = dict(model_name=args.det_name,
                  limit_side_len=args.det_limit_side_len, limit_type=args.det_limit_type)
    if args.det_dir:
        det_kw["model_dir"] = args.det_dir
    if args.det_box_thresh is not None:
        det_kw["box_thresh"] = args.det_box_thresh
    if args.det_unclip_ratio is not None:
        det_kw["unclip_ratio"] = args.det_unclip_ratio
    print(f"[det ] {det_kw}")
    det = TextDetection(**det_kw)

    # ---- 1. detect + crop once (shared across recognisers) ----
    per_img = []                                    # dict(valid, ignore, polys)
    crop_paths, crop_owner = [], []
    for i, (img, valid, ignore) in enumerate(data, 1):
        im = cv2.imread(img)
        polys = []
        if im is not None:
            try:
                for res in det.predict(input=img):
                    polys.extend(extract_polys(res))
            except Exception as e:
                print(f"  [{os.path.basename(img)}] detect failed: {e}")
        si = len(per_img)
        kept = []
        for k, q in enumerate(polys):
            c = crop_quad(im, q) if im is not None else None
            if c is None or c.size == 0 or c.shape[0] < args.min_h:
                continue
            cp = os.path.join(crops_dir, f"{si}_{k}.jpg")
            cv2.imwrite(cp, c)
            kept.append(q)
            crop_paths.append(cp)
            crop_owner.append((si, len(kept) - 1))
        per_img.append(dict(valid=valid, ignore=ignore, polys=kept))
        print(f"    detect {i}/{len(data)}", end="\r")
    print(f"\n[det] {len(crop_paths)} crops from {len(per_img)} images")

    rec = TextRecognition  # class handle; instantiated per model below
    results = {}                                    # model label -> per_script_metrics
    for label, mdir in models:
        kw = dict(model_name=args.rec_name, model_dir=mdir) if mdir else dict(model_name=args.rec_name)
        print(f"\n[{label}] {kw}")
        model = rec(**kw)
        texts = []
        for i in range(0, len(crop_paths), args.batch_size):
            chunk = crop_paths[i:i + args.batch_size]
            for res in model.predict(input=chunk, batch_size=len(chunk)):
                texts.append(rec_text(res))
            print(f"    recognise {min(i + args.batch_size, len(crop_paths))}/{len(crop_paths)}", end="\r")
        print()
        ptexts = [[""] * len(p["polys"]) for p in per_img]
        for (si, k), t in zip(crop_owner, texts):
            ptexts[si][k] = t

        bucket = collections.defaultdict(lambda: dict(m_ref=[], m_hyp=[], d_ref=[], i_hyp=[]))
        for si, p in enumerate(per_img):
            valid, ignore, polys = p["valid"], p["ignore"], p["polys"]
            pt = ptexts[si]
            cand = []
            for pi, poly in enumerate(polys):
                for gi, (g, _) in enumerate(valid):
                    v = iou(poly, g)
                    if v >= args.iou:
                        cand.append((v, pi, gi))
            cand.sort(reverse=True)
            up, ug, pairs = set(), set(), []
            for v, pi, gi in cand:
                if pi in up or gi in ug:
                    continue
                up.add(pi); ug.add(gi); pairs.append((pi, gi))
            for pi, gi in pairs:                                   # matched -> GT script
                s = script_of(valid[gi][1])
                bucket[s]['m_ref'].append(valid[gi][1]); bucket[s]['m_hyp'].append(pt[pi])
            for gi, (_, gtext) in enumerate(valid):                # deletion -> GT script
                if gi not in ug:
                    bucket[script_of(gtext)]['d_ref'].append(gtext)
            for pi, poly in enumerate(polys):                      # unmatched pred
                if pi in up:
                    continue
                if any(iou(poly, g) >= args.iou for g in ignore):  # on ignore box -> discard
                    continue
                bucket[script_of(pt[pi])]['i_hyp'].append(pt[pi])  # false positive -> pred script
        results[label] = per_script_metrics(bucket)

        sc = results[label]
        print(f"  {'script':12s}{'gt':>7s}{'fp':>6s}{'recall':>8s}"
              f"{'e2eWRR*':>9s}{'e2eCRR*':>9s}{'e2eWRR':>8s}{'recWRR':>8s}")
        print(f"  {'':12s}{'':7s}{'':6s}{'(det)':>8s}{'(no-FP)':>9s}{'(no-FP)':>9s}{'(FP)':>8s}{'(match)':>8s}")
        for s in SCRIPT_ORDER + ['__MACRO__', '__ALL__']:
            if s not in sc:
                continue
            m = sc[s]
            rw = f"{m['rec']['wrr']:.4f}" if m['rec'] else "  -   "
            print(f"  {s:12s}{m['n_gt']:>7d}{m['n_fp']:>6d}{m['recall']:>8.4f}"
                  f"{m['wrr_nofp']:>9.4f}{m['crr_nofp']:>9.4f}{m['wrr']:>8.4f}{rw:>8s}")

    # ---- cross-model per-script e2e WRR (no-FP: correct words / GT; never floors) ----
    labels = [l for l, _ in models if l in results]
    if len(labels) >= 2:
        print("\n===== per-script e2e WRR (no-FP = correct/GT, higher is better) =====")
        print(f"{'script':12s}{'gt':>7s}" + "".join(f"{l:>12s}" for l in labels))
        base = results[labels[0]]
        for s in SCRIPT_ORDER + ['__MACRO__', '__ALL__']:
            if s not in base:
                continue
            row = f"{s:12s}{base[s]['n_gt']:>7d}"
            for l in labels:
                row += f"{results[l][s]['wrr_nofp']:>12.4f}" if s in results[l] else f"{'-':>12s}"
            print(row)
        print("=" * (19 + 12 * len(labels)))

    with open(os.path.join(args.out, "e2e_bharat_perscript.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "script", "n_gt", "n_fp", "n_matched", "n_correct", "recall",
                     "e2e_WRR_noFP", "e2e_CRR_noFP", "e2e_CER", "e2e_WER", "e2e_CRR", "e2e_WRR",
                     "rec_CRR", "rec_WRR"])
        for label in labels:
            sc = results[label]
            for s in SCRIPT_ORDER + ['__MACRO__', '__ALL__']:
                if s not in sc:
                    continue
                m = sc[s]
                wr.writerow([label, s, m['n_gt'], m['n_fp'], m['n_matched'], m['n_correct'],
                             f"{m['recall']:.4f}", f"{m['wrr_nofp']:.4f}", f"{m['crr_nofp']:.4f}",
                             f"{m['cer']:.4f}", f"{m['wer']:.4f}", f"{m['crr']:.4f}", f"{m['wrr']:.4f}",
                             f"{m['rec']['crr']:.4f}" if m['rec'] else "",
                             f"{m['rec']['wrr']:.4f}" if m['rec'] else ""])
    print(f"\n-> {os.path.join(args.out, 'e2e_bharat_perscript.csv')}")


if __name__ == "__main__":
    main()
