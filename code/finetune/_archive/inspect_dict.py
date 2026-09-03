#!/usr/bin/env python3
"""Inspect a PaddleOCR character dictionary — for review before training.

Lists every character with its Unicode codepoint, category and name; summarises by
Unicode script/block and category; and FLAGS the two things worth catching:

  1. pre-composed / decomposable characters ("compound characters", e.g. क़ = क + ़)
  2. characters from an unexpected script (e.g. Latin sneaking into a Devanagari dict)

Run:
    python inspect_dict.py --dict /ssd_scratch/cvit/mahima/rec_data/devanagari_dict.txt \
                           --expect DEVANAGARI
    # save a reviewable list:
    python inspect_dict.py --dict ..._dict.txt --expect BENGALI > bengali_charlist.txt
"""
import argparse
import unicodedata
from collections import Counter

CAT_NAMES = {
    "Lo": "letter", "Lu": "uppercase letter", "Ll": "lowercase letter",
    "Lm": "modifier letter",
    "Mn": "combining mark (non-spacing)", "Mc": "combining mark (spacing)",
    "Nd": "digit", "No": "number (other)",
    "Po": "punctuation (other)", "Pd": "dash", "Ps": "open bracket", "Pe": "close bracket",
    "Pi": "quote (initial)", "Pf": "quote (final)", "Pc": "connector",
    "Sm": "math symbol", "Sc": "currency", "So": "symbol (other)", "Sk": "modifier symbol",
}


def block_of(ch):
    """Coarse script/block from the character's Unicode name (DEVANAGARI, BENGALI, LATIN, ...)."""
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNNAMED"


def main():
    ap = argparse.ArgumentParser(description="Inspect a character dictionary before training.")
    ap.add_argument("--dict", required=True, help="Path to <script>_dict.txt (one char per line).")
    ap.add_argument("--expect", default=None,
                    help="Expected script block, e.g. DEVANAGARI / BENGALI — flags anything else.")
    args = ap.parse_args()

    with open(args.dict, encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f]
    chars = [c for c in chars if c != ""]

    print(f"=== {args.dict} — {len(chars)} characters ===\n")

    cats = Counter(unicodedata.category(c) if len(c) == 1 else "??" for c in chars)
    blocks = Counter(block_of(c) if len(c) == 1 else "MULTI" for c in chars)

    print("by Unicode category:")
    for cat, n in cats.most_common():
        print(f"  {cat:4s} {CAT_NAMES.get(cat, '?'):32s} {n:4d}")

    print("\nby script / block:")
    for b, n in blocks.most_common():
        flag = ""
        if args.expect and b not in (args.expect, "DIGIT", "UNNAMED"):
            flag = "   <-- not the expected script"
        print(f"  {b:16s} {n:4d}{flag}")

    # 1) pre-composed ("compound") characters
    composed = [c for c in chars if len(c) == 1 and unicodedata.normalize("NFD", c) != c]
    print(f"\npre-composed / decomposable characters — {len(composed)}:")
    if not composed:
        print("  none — the dictionary is base characters + combining marks only")
    for c in composed:
        parts = " + ".join(f"U+{ord(x):04X}" for x in unicodedata.normalize("NFD", c))
        print(f"  {c}  U+{ord(c):04X}  {unicodedata.name(c, '?')}  ->  {parts}")

    # 2) multi-codepoint entries (should never happen)
    multi = [c for c in chars if len(c) != 1]
    if multi:
        print(f"\nWARNING — {len(multi)} entries are not a single codepoint: {multi[:10]}")

    print("\nfull list:")
    for i, c in enumerate(chars, 1):
        if len(c) == 1:
            print(f"  {i:4d}  {c}\tU+{ord(c):04X}  {unicodedata.category(c)}  "
                  f"{unicodedata.name(c, '?')}")
        else:
            print(f"  {i:4d}  {c!r}\t(multi-codepoint)")


if __name__ == "__main__":
    main()
