#!/usr/bin/env python3
"""Combine OUR PaddleOCR recognisers with the ilocr benchmark sheet, on the SAME
50 real Marathi pages, into one comparison table.

Two sources, joined on PageID (= the GT json/jpg stem in real_marathi_withGT):

  1. ilocr benchmark  -> 1217.xlsx, sheet "in".  One row per (page, layout model,
     OCR model). Only word-level numbers are populated: Correct Words / Total
     Words -> WRR.  The CRR / Correct Characters / Subs / Ins / Del columns are
     ZERO for every row, so no character metric exists on this side.
  2. our eval         -> the --eval-dir written by eval_real_perimage.py:
     <label>_per_image.csv (CER/WER/CRR/WRR) and <label>_words.csv (ref, pred).

Because the two sides do not define WRR the same way, this reports BOTH:

  WRR_exact  = exact-word-match rate, NFC-normalised   -> comparable to ilocr's
               Correct Words / Total Words (our GT-box eval gives exactly one
               prediction per GT word, so the bag-match degenerates to this).
  WRR_edit   = 1 - WER (jiwer, edit distance)          -> ours only; the number
               used in the earlier handoff tables.

Only ilocr rows with Layout Model == "GT" are a fair comparison to our
recogniser-only eval; the rest bundle a detector's errors in. All rows are kept,
tagged in the "scope" column, and the default table sorts within scope.

Usage
  python build_marathi_comparison.py --xlsx 1217.xlsx --gt-dir real_marathi_withGT \
      [--eval-dir real_marathi_eval] [--out marathi_comparison.xlsx]
"""
import os
import csv
import json
import glob
import argparse
import unicodedata
from collections import defaultdict, OrderedDict

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

OUR_LABELS = OrderedDict([
    ("stock_v5", "PP-OCRv5 Devanagari specialist (stock)"),
    ("ft_v5",    "PP-OCRv5 fine-tuned, 5k synth (specdict)"),
    ("stock_v6", "PP-OCRv6 medium rec (stock)"),
    ("ft_v6",    "PP-OCRv6 unified fine-tuned, 25k synth"),
])


def nfc(s):
    return unicodedata.normalize("NFC", str(s or "").strip())


def edit_distance(a, b):
    """Levenshtein over two sequences (chars or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pooled_cer_wer(pairs):
    """True micro CER/WER over (ref, pred) pairs — same pooling jiwer uses:
    total edits / total reference length, NOT a mean of per-item rates."""
    ce = cn = we = wn = 0
    for ref, pred in pairs:
        r, p = str(ref or ""), str(pred or "")
        ce += edit_distance(r, p)
        cn += len(r)
        rw, pw = r.split(), p.split()
        we += edit_distance(rw, pw)
        wn += len(rw)
    return (ce / cn if cn else None), (we / wn if wn else None)


# ---------------------------------------------------------------- ilocr sheet
def load_ilocr(xlsx_path, sheet="in"):
    """-> {(layout, ocr): {pageid: (correct_words, total_words)}}, deduped."""
    ws = openpyxl.load_workbook(xlsx_path, data_only=True)[sheet]
    hdr = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col = {h: i + 1 for i, h in enumerate(hdr) if h}
    need = ["PageID", "Layout Model", "OCR Model", "Total Words", "Correct Words"]
    missing = [n for n in need if n not in col]
    if missing:
        raise SystemExit(f"{xlsx_path}: missing columns {missing}")

    # Every (page, layout, ocr) appears TWICE: the sheet holds two evaluation
    # passes. They are NOT copies — 282 of the 550 doubled keys disagree, and not
    # only on Correct Words but on Total Words too, so the two passes differ on the
    # ground truth itself. Keep both here; the caller picks a pass and reports how
    # far apart they are, because on some rows the choice moves WRR by 5 points.
    out = defaultdict(lambda: defaultdict(list))
    for r in range(2, ws.max_row + 1):
        pid = ws.cell(row=r, column=col["PageID"]).value
        if pid is None:
            continue
        key = (ws.cell(row=r, column=col["Layout Model"]).value,
               ws.cell(row=r, column=col["OCR Model"]).value)
        out[key][int(pid)].append(
            (ws.cell(row=r, column=col["Correct Words"]).value or 0,
             ws.cell(row=r, column=col["Total Words"]).value or 0,
             ws.cell(row=r, column=col["GT Characters"]).value or 0,
             ws.cell(row=r, column=col["Total Characters"]).value or 0))
    return out


# ------------------------------------------------------------------- our eval
def load_gt(gt_dir, skip_texts=("NA",)):
    """-> {pageid: [word, ...]} from the GT jsons (same filter as the eval)."""
    gt = {}
    for jp in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        pid = os.path.splitext(os.path.basename(jp))[0]
        data = json.load(open(jp, encoding="utf-8"))
        words = [str(w.get("text", "")).strip() for w in data.get("words", [])]
        gt[pid] = [w for w in words if w and w not in skip_texts]
    return gt


def load_ours(eval_dir):
    """-> {label: {'per_image': {pid: {...}}, 'words': {pid: [(ref, pred), ...]}}}"""
    ours = {}
    for label in OUR_LABELS:
        pi_path = os.path.join(eval_dir, f"{label}_per_image.csv")
        w_path = os.path.join(eval_dir, f"{label}_words.csv")
        if not os.path.isfile(pi_path):
            continue
        per_image = {}
        with open(pi_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["image"].startswith("__"):
                    continue
                if row["CER"] == "NA":
                    continue
                per_image[row["image"]] = {k: float(row[k]) for k in ("CER", "WER", "CRR", "WRR")}
                per_image[row["image"]]["n_words"] = int(row["n_words"])
        words = defaultdict(list)
        if os.path.isfile(w_path):
            with open(w_path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    words[row["image"]].append((row["ref"], row["pred"]))
        else:
            print(f"  [warn] {w_path} missing -> no exact-match WRR for {label} "
                  f"(re-run eval_real_perimage.py with the words-dump patch)")
        ours[label] = {"per_image": per_image, "words": dict(words)}
    return ours


# ------------------------------------------------------------------ assembly
def summarise(rows_by_page):
    """rows_by_page: {pid: (correct, total)} -> (micro%, macro%, n_pages)."""
    if not rows_by_page:
        return None, None, 0
    c = sum(v[0] for v in rows_by_page.values())
    t = sum(v[1] for v in rows_by_page.values())
    micro = 100.0 * c / t if t else 0.0
    per = [100.0 * v[0] / v[1] for v in rows_by_page.values() if v[1]]
    macro = sum(per) / len(per) if per else 0.0
    return micro, macro, len(rows_by_page)


def build(args):
    ilocr = load_ilocr(args.xlsx)
    gt = load_gt(args.gt_dir)
    ours = load_ours(args.eval_dir) if args.eval_dir and os.path.isdir(args.eval_dir) else {}
    if args.eval_dir and not ours:
        print(f"[warn] no *_per_image.csv found under {args.eval_dir} -> baselines only")

    table = []   # dicts: system, source, scope, wrr_exact_micro/macro, cer, wer, crr, n
    per_page = {}                                     # {system: {pid: wrr_exact%}}
    counts = {}                                       # {system: {pid: (correct, scored)}}

    # --- our models
    for label, pretty in OUR_LABELS.items():
        if label not in ours:
            continue
        d = ours[label]
        exact = {}
        for pid, pairs in d["words"].items():
            hit = sum(1 for ref, pred in pairs if nfc(ref) == nfc(pred))
            exact[pid] = (hit, len(pairs))
        micro, macro, n = summarise(exact)
        n_correct = sum(v[0] for v in exact.values())
        n_total = sum(v[1] for v in exact.values())
        gtc = sum(len(r) for prs in d["words"].values() for r, _ in prs)
        prc = sum(len(p) for prs in d["words"].values() for _, p in prs)
        pi = d["per_image"]
        n_pi = len(pi) or 1
        # micro CER/WER pool every word; macro CRR/WRR come straight from the eval's
        # own per-image rows (each page weighted equally).
        all_pairs = [pr for prs in d["words"].values() for pr in prs]
        cer, wer = pooled_cer_wer(all_pairs) if all_pairs else (None, None)
        table.append(dict(
            system=pretty, source="ours — exact word match, GT boxes", scope="GT boxes",
            wrr_exact_micro=micro, wrr_exact_macro=macro,
            cer=cer, wer=wer,
            cer_macro=sum(v["CER"] for v in pi.values()) / n_pi if pi else None,
            wer_macro=sum(v["WER"] for v in pi.values()) / n_pi if pi else None,
            crr=sum(v["CRR"] for v in pi.values()) / n_pi if pi else None,
            wrr_edit=sum(v["WRR"] for v in pi.values()) / n_pi if pi else None,
            char_ratio=prc / gtc if gtc else None,
            spread=None,                      # single eval run, nothing to disagree with
            correct=n_correct, total=n_total,
            n=n or len(pi)))
        per_page[pretty] = {p: 100.0 * c / t for p, (c, t) in exact.items() if t}
        counts[pretty] = dict(exact)

    # --- ilocr benchmark
    for (layout, ocr), passes in sorted(ilocr.items()):
        # LATER pass wins (the user's call: treat pass 2 as the corrected re-run).
        chosen = {p: v[-1] for p, v in passes.items()}
        earlier = {p: v[0] for p, v in passes.items()}
        micro, macro, n = summarise(chosen)
        alt_micro, _, _ = summarise(earlier)
        n_correct = sum(v[0] for v in chosen.values())
        n_total = sum(v[1] for v in chosen.values())
        gtc = sum(v[2] for v in chosen.values())
        prc = sum(v[3] for v in chosen.values())
        end2end = layout == ocr                      # GoogleOCR / EasyOCR rows
        scope = ("end-to-end (own detector)" if end2end
                 else "GT boxes" if layout == "GT" else f"det: {layout}")
        name = ocr if end2end else f"{ocr}  [layout {layout}]"
        table.append(dict(
            system=name, source="ilocr 1217.xlsx — as published", scope=scope,
            wrr_exact_micro=micro, wrr_exact_macro=macro,
            cer=None, wer=None, cer_macro=None, wer_macro=None,
            crr=None, wrr_edit=None,
            char_ratio=prc / gtc if gtc else None,
            spread=abs(micro - alt_micro) if alt_micro is not None else None,
            correct=n_correct, total=n_total, n=n))
        per_page[name] = {str(p): 100.0 * v[0] / v[1] for p, v in chosen.items() if v[1]}
        counts[name] = {str(p): (v[0], v[1]) for p, v in chosen.items()}

    # Every system is scored on a different denominator: ours sees all 554 GT word
    # boxes, while the ilocr passes were run against 540-543. Words their eval never
    # counted are still words the GT contains, so rebasing onto 554 charges them as
    # misses and puts every row on one scale.
    common = sum(len(v) for v in gt.values()) or None
    ours_total = next((r["total"] for r in table if r["source"].startswith("ours")), None)
    if ours_total:
        common = ours_total
    for r in table:
        r["wrr_rebased"] = 100.0 * r["correct"] / common if common else None
        r["common_denom"] = common

    # GT-box scope first, then the rest; best rebased WRR first within each
    order = {"GT boxes": 0, "end-to-end (own detector)": 1}
    table.sort(key=lambda r: (order.get(r["scope"], 2), -(r["wrr_rebased"] or 0)))
    return table, per_page, counts, gt


# ------------------------------------------------------------------- writers
def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


MD_COLS = [
    ("System",              lambda r: r["system"]),
    ("Source",              lambda r: r["source"]),
    ("Scope",               lambda r: r["scope"]),
    ("correct",             lambda r: str(r["correct"])),
    ("total scored",        lambda r: str(r["total"])),
    ("WRR micro %",         lambda r: fmt(r["wrr_exact_micro"])),
    ("WRR vs 554 GT %",     lambda r: fmt(r["wrr_rebased"])),
    ("WRR macro %",         lambda r: fmt(r["wrr_exact_macro"])),
    ("dup spread",          lambda r: fmt(r["spread"])),
    ("CER micro",           lambda r: fmt(r["cer"], 4)),
    ("WER micro",           lambda r: fmt(r["wer"], 4)),
    ("CER macro",           lambda r: fmt(r["cer_macro"], 4)),
    ("WER macro",           lambda r: fmt(r["wer_macro"], 4)),
    ("CRR macro",           lambda r: fmt(r["crr"], 4)),
    ("WRR=1-WER macro",     lambda r: fmt(r["wrr_edit"], 4)),
    ("pred/GT chars",       lambda r: fmt(r["char_ratio"], 3)),
    ("Pages",               lambda r: str(r["n"])),
]


def write_markdown(table, path):
    lines = ["| " + " | ".join(h for h, _ in MD_COLS) + " |",
             "|" + "|".join("---" if i < 3 else "---:" for i in range(len(MD_COLS))) + "|"]
    for r in table:
        lines.append("| " + " | ".join(f(r) for _, f in MD_COLS) + " |")
    note = ("\nMicro pools every word; macro averages the 50 pages. CRR/WRR macro are complements of the "
            "MACRO CER/WER, not of the micro ones. WRR-exact is exact word match (NFC) and is the only "
            "column defined identically on both sides.\n")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n" + note)
    return "\n".join(lines)


def write_xlsx(table, per_page, counts, gt, path):
    wb = openpyxl.Workbook()
    HEAD = Font(name="Arial", bold=True, color="FFFFFF")
    BODY = Font(name="Arial")
    FILL = PatternFill("solid", fgColor="4472C4")
    OURS = PatternFill("solid", fgColor="FFF2CC")

    # Both sheets are a snapshot of numbers computed elsewhere (the eval CSVs and
    # the ilocr sheet), so the cells hold values, not formulas: nothing here has an
    # input a reader could meaningfully edit and recalculate. Summary's macro column
    # is the mean of the matching PerPage column.
    systems = list(per_page.keys())
    pages = sorted({p for d in per_page.values() for p in d}, key=lambda x: int(x))

    ws = wb.active
    ws.title = "Summary"
    cols = ["System", "Source", "Scope", "Correct words", "Total scored",
        "WRR micro %", "WRR vs 554 GT %", "WRR macro %", "dup spread",
            "CER (micro)", "WER (micro)", "CER (macro)", "WER (macro)",
            "CRR (macro)", "WRR=1-WER (macro)", "pred/GT chars", "Pages"]
    keys = ["system", "source", "scope", "correct", "total",
        "wrr_exact_micro", "wrr_rebased", "wrr_exact_macro", "spread",
            "cer", "wer", "cer_macro", "wer_macro", "crr", "wrr_edit",
            "char_ratio", "n"]
    for i, h in enumerate(cols, 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font, c.fill = HEAD, FILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r, row in enumerate(table, 2):
        for i, k in enumerate(keys, 1):
            c = ws.cell(row=r, column=i, value=row[k])
            c.font = BODY
            if row["source"].startswith("ours"):
                c.fill = OURS
            if 6 <= i <= 9:
                c.number_format = "0.00"
            if 10 <= i <= 15:
                c.number_format = "0.0000"
            if i == 16:
                c.number_format = "0.000"
    for i, w in enumerate([44, 30, 26, 9, 9, 11, 12, 11, 10, 11, 11, 11, 11, 11, 15, 12, 8], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    n = len(table) + 3
    notes = [
        "Notes",
        "Dataset: 50 real Marathi pages (PageID 109199-109248) = real_marathi_withGT = 1217.xlsx.",
        "WRR micro pools all words (sum correct / sum total); WRR macro averages the 50 pages equally.",
        "The two sides do NOT compute WRR the same way — read the Source column. Ours is exact word match "
        "(NFC-normalised) over 554 GT crops, computed here. ilocr's is its own published Correct Words / "
        "Total Words; the sheet stores only counts, never the predicted text, so its matching rule cannot be "
        "verified. Treat cross-source gaps as indicative, not as a like-for-like result.",
        "'dup spread' = how far the WRR micro moves if the OTHER evaluation pass is used instead. Each "
        "(page, layout, ocr) appears twice in 1217.xlsx and 282 of 550 such pairs disagree; this table uses "
        "the LATER pass. A spread of 5.16 on GT/V-04.00.00.02 means its lead over our ft_v6 (64.08) is inside "
        "the sheet's own noise — those two are not separable on this data.",
        "ilocr side: Correct Words / Total Words from 1217.xlsx. Its CRR / Correct Characters / "
        "Substitutions / Insertions / Deletions columns are 0 for every row, so no character metric exists there.",
        "CRR is NOT recoverable for the ilocr rows. Bracketing it as [WRR, min(1, pred/GT chars)] was tested "
        "against our four models, where the truth is known: the bracket spans ~36 points and its lower edge "
        "is violated by stock_v6 (true CRR 0.1020 < WRR 0.1227), because that model's few exact hits are "
        "unusually short tokens. Too wide and too fragile to quote, so those cells are left blank.",
        "'pred/GT chars' is a LENGTH ratio, not an accuracy: <1 means the system emitted less text than the "
        "GT (missed/dropped words), >1 means it emitted more (over-detection, hallucinated text). It is the "
        "only character-level signal the ilocr sheet supports. Note the ilocr counts appear to include the "
        "spaces joining words, while ours count word characters only.",
        "Our side: recogniser-only on GT word boxes (eval_real_perimage.py). The CER/WER/CRR/WRR columns "
        "reproduce that script's own summary block: micro = pooled over all 554 words (its uCER/uWER), "
        "macro = mean over the 50 pages. CRR (macro) = 1 - CER (macro), NOT 1 - CER (micro).",
        "Only rows with scope 'GT boxes' are strictly comparable to ours. Rows tagged 'det: X' include that "
        "detector's errors; GoogleOCR / EasyOCR run their own detection end-to-end.",
        "'NA' placeholder GT boxes are excluded on our side.",
    ]
    for i, t in enumerate(notes):
        c = ws.cell(row=n + i, column=1, value=t)
        c.font = Font(name="Arial", bold=(i == 0), italic=(i > 0), size=10)

    ws2 = wb.create_sheet("PerPage")
    hdr = ["PageID", "GT words"] + systems
    for i, h in enumerate(hdr, 1):
        c = ws2.cell(row=1, column=i, value=h)
        c.font, c.fill = HEAD, FILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r, p in enumerate(pages, 2):
        ws2.cell(row=r, column=1, value=int(p)).font = BODY
        ws2.cell(row=r, column=2, value=len(gt.get(p, []))).font = BODY
        for i, s in enumerate(systems, 3):
            c = ws2.cell(row=r, column=i, value=per_page[s].get(p))
            c.font, c.number_format = BODY, "0.00"
    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = 10
    for i in range(3, len(systems) + 3):
        ws2.column_dimensions[get_column_letter(i)].width = 22
    ws2.freeze_panes = "C2"

    ws3 = wb.create_sheet("PerPage_Counts")
    # The point of this sheet: each system was scored against its OWN word count.
    # "GT words" is the truth (our GT jsons); "scored" is how many that system's eval
    # actually looked at; "missed by eval" is the shortfall it was never charged for.
    order3 = [r["system"] for r in table]
    hdr3 = ["PageID", "GT words"]
    for sname in order3:
        hdr3 += [f"{sname} — scored", f"{sname} — correct", f"{sname} — missed by eval"]
    for i, h in enumerate(hdr3, 1):
        c = ws3.cell(row=1, column=i, value=h)
        c.font, c.fill = HEAD, FILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r, p in enumerate(pages, 2):
        gtn = len(gt.get(p, []))
        ws3.cell(row=r, column=1, value=int(p)).font = BODY
        ws3.cell(row=r, column=2, value=gtn).font = BODY
        for k, sname in enumerate(order3):
            cor, sc = counts.get(sname, {}).get(p, (None, None))
            for off, val in enumerate((sc, cor, None if sc is None else gtn - sc)):
                c = ws3.cell(row=r, column=3 + k * 3 + off, value=val)
                c.font = BODY
    tot = len(pages) + 2
    ws3.cell(row=tot, column=1, value="TOTAL").font = Font(name="Arial", bold=True)
    ws3.cell(row=tot, column=2, value=sum(len(gt.get(p, [])) for p in pages)).font = Font(name="Arial", bold=True)
    for k, sname in enumerate(order3):
        d = counts.get(sname, {})
        sc = sum(v[1] for v in d.values()) if d else None
        cor = sum(v[0] for v in d.values()) if d else None
        gtot = sum(len(gt.get(p, [])) for p in pages)
        for off, val in enumerate((sc, cor, None if sc is None else gtot - sc)):
            c = ws3.cell(row=tot, column=3 + k * 3 + off, value=val)
            c.font = Font(name="Arial", bold=True)
    ws3.column_dimensions["A"].width = 10
    ws3.column_dimensions["B"].width = 10
    for i in range(3, 3 + len(order3) * 3):
        ws3.column_dimensions[get_column_letter(i)].width = 15
    ws3.freeze_panes = "C2"

    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="1217.xlsx", help="ilocr benchmark sheet (Marathi 50).")
    ap.add_argument("--gt-dir", default="real_marathi_withGT", help="images + GT jsons.")
    ap.add_argument("--eval-dir", default=None,
                    help="output dir of eval_real_perimage.py (the *_per_image.csv / *_words.csv).")
    ap.add_argument("--out", default="marathi_comparison.xlsx")
    ap.add_argument("--md", default=None, help="also write a markdown table here.")
    args = ap.parse_args()

    table, per_page, counts, gt = build(args)
    write_xlsx(table, per_page, counts, gt, args.out)
    md = write_markdown(table, args.md or os.path.splitext(args.out)[0] + ".md")
    print(md)
    print(f"\n-> {args.out}")
    print(f"-> {args.md or os.path.splitext(args.out)[0] + '.md'}")


if __name__ == "__main__":
    main()
