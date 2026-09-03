#!/usr/bin/env python3
"""Build a whole-image manifest.jsonl from REAL GT-box JSONs for eval_fullpage.py.

Each real image has a sidecar JSON:  {"words":[{"x","y","w","h","line","text"}, ...]}.
eval_fullpage.py needs ONE full-image reference string per image. This joins the GT
words into reading order — lines top->bottom, words left->right within a line — the
SAME layout eval_fullpage builds from the pipeline's predictions, so CER/WER compares
like with like.

Output (default <images-dir>/manifest_real.jsonl), one line per image:
    {"image": "<basename.jpg>", "text": "<line1 words>\\n<line2 words>..."}
Image paths are RELATIVE to --images-dir -> pass that dir as eval_fullpage --images-root.

Run:
    python build_real_manifest.py --images-dir $SCR/real_hindi
"""
import os
import json
import glob
import argparse


def joined_gt(words, skip):
    """Group words by their `line` index, sort left->right within a line, and order
    lines top->bottom by mean-y (robust even if line ids aren't in visual order)."""
    lines = {}
    for w in words:
        text = str(w.get("text", "")).strip()
        if not text or text in skip:
            continue
        ln = w.get("line", 0)
        lines.setdefault(ln, []).append((float(w.get("x", 0)), float(w.get("y", 0)), text))
    ordered = sorted(lines.values(), key=lambda ws: sum(y for _, y, _ in ws) / len(ws))
    return "\n".join(" ".join(t for _, _, t in sorted(ws, key=lambda z: z[0])) for ws in ordered)


def main():
    ap = argparse.ArgumentParser(description="Real GT-box JSONs -> whole-image manifest.jsonl")
    ap.add_argument("--images-dir", required=True, help="Folder with images + matching GT .json files.")
    ap.add_argument("--out", default=None, help="Output path (default: <images-dir>/manifest_real.jsonl).")
    ap.add_argument("--skip-text", nargs="*", default=["NA"],
                    help='GT text values to ignore (default: "NA" placeholder boxes).')
    args = ap.parse_args()

    out = args.out or os.path.join(args.images_dir, "manifest_real.jsonl")
    skip = set(args.skip_text)
    exts = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for jp in sorted(glob.glob(os.path.join(args.images_dir, "*.json"))):
            stem = os.path.splitext(os.path.basename(jp))[0]
            img = next((f"{stem}.{e}" for e in exts
                        if os.path.isfile(os.path.join(args.images_dir, f"{stem}.{e}"))), None)
            if img is None:
                print(f"[skip] no image for {jp}")
                continue
            data = json.load(open(jp, encoding="utf-8"))
            text = joined_gt(data.get("words", []), skip)
            if not text.strip():
                continue
            fo.write(json.dumps({"image": img, "text": text}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[manifest] {n} images -> {out}")


if __name__ == "__main__":
    main()
