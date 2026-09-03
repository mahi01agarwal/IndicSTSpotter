#!/usr/bin/env python3
"""eval_e2e_perscript.py — END-TO-END, PER-SCRIPT eval on IndicSTR12 scenes (FP-penalized).

The real Hi/Mr sets only cover Devanagari, and a per-script end-to-end number needs
multi-script SCENE GT where each scene's script is known. IndicSTR12's `verified_twice/`
is organised BY LANGUAGE folder (assamese, bengali, ... telugu), so every scene is one
script — which lets us bucket the page-level jiwer score per language without having to
guess the script of a false-positive box.

Pipeline per scene: detector -> quads -> crop -> recogniser -> text. Then per LANGUAGE:
  * END-TO-END (FP-penalized): jiwer WER/CER between the scene's full GT reading-order
    string and the full predicted reading-order string. A false-positive box dumps its
    text into the hypothesis (insertions); a missed word drops from it (deletions) — so
    both detector failure modes push the score DOWN, exactly like eval_e2e_real's page
    metric, but resolved per script.
  * RECOGNITION (companion, no FP): jiwer over IoU-matched (GT word, predicted word) pairs
    — how well the recogniser reads the words the detector actually found, per script.

Neither the fine-tuned detector nor the balanced recogniser trained on IndicSTR12, so this
is a clean cross-dataset benchmark. By default it scores only the held-out TEST scenes
(same deterministic scene-level split as indicstr12_to_paddle.py, so it stays consistent
once IndicSTR12-train enters recogniser training); pass --all to score every scene (valid
right now, since the current models never saw any of it — more scenes per script).

    python eval_e2e_perscript.py --root $SCR/indicstr12/verified_twice \
        --det-dir $SCR/PaddleOCR/output/v6_det_real_infer_latest \
        --rec-dir $SCR/PaddleOCR/output/v6_rec_synthreal_infer \
        --all --out $SCR/e2e_perscript_indicstr12

Needs: jiwer, opencv, paddleocr.  Run in the $SCR/ocr env on gnode063.
"""
import os
import sys
import glob
import csv
import argparse
import collections

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data_prep"))
from eval_e2e_real import reading_order_text, jiwer_scores, crop_quad, rec_text   # noqa: E402
from eval_det_real import iou, extract_polys                                       # noqa: E402
from indicstr12_to_paddle import parse_gt, is_test                                 # noqa: E402

IMG_EXTS = ("jpeg", "jpg", "png", "JPEG", "JPG", "PNG")
# report order (language folders); each maps to a script noted in the header
LANG_ORDER = ["hindi", "marathi", "bengali", "assamese", "punjabi", "odia",
              "gujarati", "tamil", "telugu", "kannada", "malayalam", "urdu"]
LANG_SCRIPT = {"hindi": "devanagari", "marathi": "devanagari", "bengali": "bengali",
               "assamese": "bengali", "punjabi": "gurmukhi", "odia": "oriya",
               "gujarati": "gujarati", "tamil": "tamil", "telugu": "telugu",
               "kannada": "kannada", "malayalam": "malayalam", "urdu": "urdu"}


def scenes_for_lang(langdir, lang, use_all, frac, seed):
    imgs = []
    for e in IMG_EXTS:
        imgs += glob.glob(os.path.join(langdir, f"*.{e}"))
    out = []
    for img in sorted(imgs):
        base = os.path.splitext(os.path.basename(img))[0]
        if base.startswith("._"):
            continue
        gt = os.path.join(langdir, base + "_gt.txt")
        if not os.path.isfile(gt):
            continue
        if not use_all and not is_test(lang, base, frac, seed):
            continue
        out.append((img, gt, base))
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-script end-to-end eval on IndicSTR12 scenes.")
    ap.add_argument("--root", required=True, help="IndicSTR12 verified_twice/ dir.")
    ap.add_argument("--det-dir", required=True, help="fine-tuned detector inference dir.")
    ap.add_argument("--det-name", default="PP-OCRv6_medium_det")
    ap.add_argument("--rec-dir", required=True, help="recogniser inference dir.")
    ap.add_argument("--rec-name", default="PP-OCRv6_medium_rec")
    ap.add_argument("--out", default="./e2e_perscript")
    ap.add_argument("--skip-langs", nargs="*", default=["urdu"])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--all", action="store_true",
                    help="score ALL scenes (ignore the test split; valid while models are unseen).")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-h", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-per-lang", type=int, default=0, help="cap scenes/lang (debug).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    skip = set(args.skip_langs)

    langs = sorted(d for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d))
                   and not d.startswith(("__", ".")) and d not in skip)
    scenes = []                                     # (lang, img, gt, base)
    for lang in langs:
        s = scenes_for_lang(os.path.join(args.root, lang), lang, args.all,
                            args.test_frac, args.split_seed)
        if args.limit_per_lang:
            s = s[:args.limit_per_lang]
        for img, gt, base in s:
            scenes.append((lang, img, gt, base))
    mode = "ALL scenes" if args.all else f"TEST scenes (frac={args.test_frac}, seed={args.split_seed})"
    print(f"[data] {len(scenes)} scenes over {len(langs)} langs — {mode}\n")

    from paddleocr import TextDetection, TextRecognition
    det = TextDetection(model_name=args.det_name, model_dir=args.det_dir)
    rec = TextRecognition(model_name=args.rec_name, model_dir=args.rec_dir)

    # ---- 1. detect + crop every scene (so recognition can batch across scenes) ----
    scene_data = []                                 # dict(lang, gt=[(quad,text)], polys, ptexts, base)
    crop_paths, crop_owner = [], []
    for i, (lang, img, gt, base) in enumerate(scenes, 1):
        gt_items = list(parse_gt(gt))
        im = cv2.imread(img)
        polys = []
        if im is not None:
            try:
                for res in det.predict(input=img):
                    polys.extend(extract_polys(res))
            except Exception as e:
                print(f"  [{lang}/{base}] detect failed: {e}")
        kept = []
        si = len(scene_data)
        for k, q in enumerate(polys):
            c = crop_quad(im, q) if im is not None else None
            if c is None or c.size == 0 or c.shape[0] < args.min_h:
                continue
            cp = os.path.join(crops_dir, f"{lang}_{base}_{k}.jpg")
            cv2.imwrite(cp, c)
            kept.append(q)
            crop_paths.append(cp)
            crop_owner.append((si, len(kept) - 1))
        scene_data.append(dict(lang=lang, gt=gt_items, polys=kept, ptexts=None, base=base))
        print(f"    detect {i}/{len(scenes)}", end="\r")
    print(f"\n[det] {len(crop_paths)} crops from {len(scene_data)} scenes")

    # ---- 2. recognise all crops ----
    texts = []
    for i in range(0, len(crop_paths), args.batch_size):
        chunk = crop_paths[i:i + args.batch_size]
        for res in rec.predict(input=chunk, batch_size=len(chunk)):
            texts.append(rec_text(res))
        print(f"    recognise {min(i + args.batch_size, len(crop_paths))}/{len(crop_paths)}", end="\r")
    print()
    for sd in scene_data:
        sd["ptexts"] = [""] * len(sd["polys"])
    for (si, k), t in zip(crop_owner, texts):
        scene_data[si]["ptexts"][k] = t

    # ---- 3. bucket per language: page-level (FP) + matched recognition ----
    B = collections.defaultdict(lambda: dict(gt_pages=[], pred_pages=[],
                                             rec_refs=[], rec_hyps=[], nwords=0, nscene=0))
    for sd in scene_data:
        L = B[sd["lang"]]
        gt_items, polys, ptexts = sd["gt"], sd["polys"], sd["ptexts"]
        L["gt_pages"].append(reading_order_text(gt_items))
        L["pred_pages"].append(reading_order_text(list(zip(polys, ptexts))))
        L["nwords"] += len(gt_items)
        L["nscene"] += 1
        cand = []
        for pi, p in enumerate(polys):
            for gi, (g, _) in enumerate(gt_items):
                v = iou(p, g)
                if v >= args.iou:
                    cand.append((v, pi, gi))
        cand.sort(reverse=True)
        up, ug = set(), set()
        for v, pi, gi in cand:
            if pi in up or gi in ug:
                continue
            up.add(pi); ug.add(gi)
            L["rec_refs"].append(gt_items[gi][1]); L["rec_hyps"].append(ptexts[pi])

    # ---- 4. report ----
    order = [l for l in LANG_ORDER if l in B] + [l for l in sorted(B) if l not in LANG_ORDER]
    all_gtp, all_predp, all_rr, all_rh = [], [], [], []
    rows = []
    for lang in order:
        L = B[lang]
        e2e = jiwer_scores(L["gt_pages"], L["pred_pages"])
        recg = jiwer_scores(L["rec_refs"], L["rec_hyps"])
        all_gtp += L["gt_pages"]; all_predp += L["pred_pages"]
        all_rr += L["rec_refs"]; all_rh += L["rec_hyps"]
        rows.append((lang, LANG_SCRIPT.get(lang, "?"), L["nscene"], L["nwords"], e2e, recg))
    all_e2e = jiwer_scores(all_gtp, all_predp)
    all_rec = jiwer_scores(all_rr, all_rh)

    def cell(d, key):
        return f"{d[key]:.4f}" if d else "  -   "

    hdr = (f"{'language':11s}{'script':11s}{'scn':>5s}{'wrds':>7s}"
           f"{'e2e_CRR':>9s}{'e2e_WRR':>9s}{'rec_CRR':>9s}{'rec_WRR':>9s}")
    print("\n===== PER-SCRIPT — end-to-end (FP-penalized) + recognition (matched) =====")
    print(hdr)
    for lang, scr, nsc, nw, e2e, recg in rows:
        print(f"{lang:11s}{scr:11s}{nsc:>5d}{nw:>7d}"
              f"{cell(e2e,'crr'):>9s}{cell(e2e,'wrr'):>9s}{cell(recg,'crr'):>9s}{cell(recg,'wrr'):>9s}")
    nsc = sum(r[2] for r in rows); nw = sum(r[3] for r in rows)
    print(f"{'ALL':11s}{'':11s}{nsc:>5d}{nw:>7d}"
          f"{cell(all_e2e,'crr'):>9s}{cell(all_e2e,'wrr'):>9s}"
          f"{cell(all_rec,'crr'):>9s}{cell(all_rec,'wrr'):>9s}")
    print("=" * len(hdr))

    csv_path = os.path.join(args.out, "e2e_perscript.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["language", "script", "scenes", "gt_words",
                     "e2e_CER", "e2e_WER", "e2e_CRR", "e2e_WRR", "e2e_S", "e2e_D", "e2e_I",
                     "rec_CER", "rec_WER", "rec_CRR", "rec_WRR"])
        for lang, scr, nsc2, nw2, e2e, recg in rows:
            def g(d, k):
                return f"{d[k]:.4f}" if d else ""
            wr.writerow([lang, scr, nsc2, nw2,
                         g(e2e, "cer"), g(e2e, "wer"), g(e2e, "crr"), g(e2e, "wrr"),
                         e2e["S"] if e2e else "", e2e["D"] if e2e else "", e2e["I"] if e2e else "",
                         g(recg, "cer"), g(recg, "wer"), g(recg, "crr"), g(recg, "wrr")])
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
