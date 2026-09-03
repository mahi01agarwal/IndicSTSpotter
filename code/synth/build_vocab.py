#!/usr/bin/env python3
"""Build the generator's vocab/<script>.txt files from the 000_text_corpus.

The corpus has one folder per LANGUAGE, each with train/val/test word-label
files. Most are one word per line, but the Urdu files are whole verses, so
every line is split on whitespace into words. The generator is keyed by SCRIPT
(what a
glyph looks like), so several languages fold into one script:

  hindi + marathi                 -> devanagari
  bengali + assamese + manipuri   -> bengali   (all Eastern-Nagari script)
  english                         -> latin
  gujarati / punjabi / oriya / urdu -> their own (new) scripts

For each target script we merge every source language's train+val+test, drop
tokens that are PURE punctuation/symbols (kept: anything with at least one
letter or digit -- so numbers like "੪੦੮" and words like "say,'" survive),
dedup to unique tokens (first-seen order), and write vocab/<script>.txt.

Urdu is right-to-left; nothing special is needed here -- tokens are stored in
logical order and Raqm/HarfBuzz reorders them at render time.

Run:
    python build_vocab.py --corpus-dir ~/data/000_text_corpus --out-dir ./vocab
"""

import os
import argparse
import unicodedata
from collections import OrderedDict

# corpus language folder -> generator script key
LANG_TO_SCRIPT = {
    "english":   "latin",
    "hindi":     "devanagari",
    "marathi":   "devanagari",
    "telugu":    "telugu",
    "tamil":     "tamil",
    "kannada":   "kannada",
    "malayalam": "malayalam",
    "bengali":   "bengali",
    "assamese":  "bengali",
    "manipuri":  "bengali",
    "gujarati":  "gujarati",
    "punjabi":   "gurmukhi",
    "oriya":     "oriya",
    "urdu":      "urdu",
}

SPLITS = ("train_word_labels.txt", "val_word_labels.txt", "test_word_labels.txt")


def is_wordlike(tok):
    """Keep tokens with >=1 letter (L*) or number (N*); drop pure punctuation/symbols."""
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in tok)


def main():
    ap = argparse.ArgumentParser(
        description="Build vocab/<script>.txt from the 000_text_corpus word-label files.")
    ap.add_argument("--corpus-dir", required=True, help="Path to 000_text_corpus/")
    ap.add_argument("--out-dir", default="./vocab")
    ap.add_argument("--min-len", type=int, default=1,
                    help="Drop tokens shorter than this many characters.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # group source languages by their target script
    script_to_langs = OrderedDict()
    for lang, script in LANG_TO_SCRIPT.items():
        script_to_langs.setdefault(script, []).append(lang)

    grand_unique = 0
    for script, langs in script_to_langs.items():
        seen = OrderedDict()          # token -> None (dedup, keeps first-seen order)
        n_read = n_punct = n_missing = 0
        for lang in langs:
            for split in SPLITS:
                path = os.path.join(args.corpus_dir, lang, split)
                if not os.path.isfile(path):
                    n_missing += 1
                    print(f"  [skip] missing {path}")
                    continue
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        # split on whitespace so phrase-per-line corpora (the Urdu
                        # verses) become individual words like every other language
                        for tok in line.split():
                            n_read += 1
                            if len(tok) < args.min_len:
                                continue
                            if not is_wordlike(tok):
                                n_punct += 1
                                continue
                            seen[tok] = None

        out_path = os.path.join(args.out_dir, script + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for tok in seen:
                f.write(tok + "\n")
        grand_unique += len(seen)
        print(f"[{script:10s}] {'+'.join(langs):28s} "
              f"read={n_read:>9}  punct_dropped={n_punct:>7}  unique={len(seen):>8}  -> {out_path}")

    print(f"\n[done] {grand_unique} unique tokens across {len(script_to_langs)} scripts -> {args.out_dir}")


if __name__ == "__main__":
    main()
