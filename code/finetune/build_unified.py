#!/usr/bin/env python3
"""Merge the per-script crop data into ONE unified training set (all scripts, one dictionary).

Reads what crop_words.py produced (rec_data/<script>_{train,val}.txt + <script>_dict.txt) and writes:
    rec_data/unified_train.txt    all scripts' crops pooled + shuffled
    rec_data/unified_val.txt
    rec_data/unified_dict.txt     union of every script's characters

After this, `make_config.py --script unified` works unchanged — one recogniser that reads any script,
no `lang` needed at inference.

Run:
    python build_unified.py --rec-data /ssd_scratch/cvit/mahima/rec_data
"""
import os
import glob
import random
import argparse


def main():
    ap = argparse.ArgumentParser(description="Pool per-script crops into one unified set.")
    ap.add_argument("--rec-data", required=True, help="Output dir from crop_words.py.")
    ap.add_argument("--scripts", nargs="*", default=None, help="Subset (default: every script found).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rec = os.path.abspath(args.rec_data)
    scripts = args.scripts or sorted(
        os.path.basename(p)[: -len("_dict.txt")]
        for p in glob.glob(os.path.join(rec, "*_dict.txt"))
        if not os.path.basename(p).startswith("unified")
    )
    if not scripts:
        raise SystemExit(f"no <script>_dict.txt found in {rec} — run crop_words.py first")
    print(f"scripts ({len(scripts)}): {', '.join(scripts)}\n")

    max_len = 0
    for split in ("train", "val"):
        rows = []
        for s in scripts:
            p = os.path.join(rec, f"{s}_{split}.txt")
            if not os.path.isfile(p):
                print(f"  [skip] missing {p}")
                continue
            n = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    rows.append(line)
                    n += 1
                    text = line.partition("\t")[2]
                    max_len = max(max_len, len(text))
            print(f"  {s:11s} {split:5s} {n:7d}")
        random.Random(args.seed).shuffle(rows)
        out = os.path.join(rec, f"unified_{split}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[{split:5s}] {len(rows):7d} rows -> {out}\n")

    chars = set()
    for s in scripts:
        p = os.path.join(rec, f"{s}_dict.txt")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                chars.update(l.rstrip("\n") for l in f if l.rstrip("\n"))
    out = os.path.join(rec, "unified_dict.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(chars)) + "\n")

    print(f"[dict ] {len(chars)} unique characters -> {out}")
    print(f"[hint ] longest label = {max_len} chars  ->  pass --max-text-length {max_len + 5} to make_config.py")


if __name__ == "__main__":
    main()
