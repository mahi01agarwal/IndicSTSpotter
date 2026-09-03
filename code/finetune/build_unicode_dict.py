#!/usr/bin/env python3
"""Build FULL Unicode-block character dictionaries for PaddleOCR rec training.

Replaces the corpus-derived ``<script>_dict.txt`` that crop_words.py emits with a dict
that covers the ENTIRE Unicode block(s) for each script (every assigned, printable
code point), UNIONed with the characters that actually occur in the crop labels
(so no training label is ever out-of-vocabulary), plus ASCII digits/letters and
common punctuation that appear on real signage.

Why: a corpus-derived dict only contains characters seen in our synthetic words, so
it can MISS characters that occur in REAL images -> those characters become
unpredictable by construction (a guaranteed error, no matter how good the model is).
The Unicode block guarantees coverage; the label-union guarantees every training
sample is representable.

Run AFTER crop_words.py (it reads the ``<script>_{train,val}.txt`` label files):

    # every script found in rec_data, plus a rebuilt unified_dict.txt:
    python build_unicode_dict.py --rec-data $SCR/rec_data

    # just one script:
    python build_unicode_dict.py --rec-data $SCR/rec_data --scripts devanagari

The original corpus dict is backed up to ``<script>_dict.corpus.txt`` (once) before
being overwritten. make_config.py / build_unified.py keep working unchanged.

NOTE (warm-start interaction): if you warm-start from a stock language specialist
(e.g. devanagari_PP-OCRv5_mobile_rec), a dict that differs from the specialist's own
dict causes PaddleOCR to REINITIALISE the CTC head (shape mismatch -> skipped on
load). You still inherit the backbone/neck. To also inherit the head, feed the
specialist's official dict verbatim instead of the Unicode dict (see --seed-dict, or
just point make_config.py --dict at it).
"""
import os
import glob
import shutil
import argparse
import unicodedata

# Full Unicode block range(s) per script (inclusive). Only ASSIGNED, printable code
# points survive the C*/Z* filter below (unassigned, control, whitespace dropped),
# so listing whole blocks is safe -- combining matras/signs (Mn/Mc) are kept.
SCRIPT_BLOCKS = {
    "latin":      [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],  # + Latin-1/Ext-A/B
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],                    # + Devanagari Extended
    "bengali":    [(0x0980, 0x09FF)],
    "gurmukhi":   [(0x0A00, 0x0A7F)],
    "gujarati":   [(0x0A80, 0x0AFF)],
    "oriya":      [(0x0B00, 0x0B7F)],
    "tamil":      [(0x0B80, 0x0BFF)],
    "telugu":     [(0x0C00, 0x0C7F)],
    "kannada":    [(0x0C80, 0x0CFF)],
    "malayalam":  [(0x0D00, 0x0D7F)],
    "urdu":       [(0x0600, 0x06FF), (0x0750, 0x077F)],                    # Arabic + Arabic Supplement
}

# Added to EVERY script dict -- real Indic signage routinely mixes these in.
COMMON_RANGES = [(0x0030, 0x0039),                       # ASCII digits 0-9
                 (0x0041, 0x005A), (0x0061, 0x007A)]     # ASCII letters A-Z a-z
# ASCII punctuation (no space -> that is handled by use_space_char) + common typographic.
COMMON_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "₹–—‘’“”…•"


def expand(ranges):
    """All assigned, printable single chars in the given inclusive code-point ranges."""
    out = set()
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            cat = unicodedata.category(ch)          # 'Cn'=unassigned, 'C*'=control, 'Z*'=separator
            if cat[0] in ("C", "Z"):
                continue
            out.add(ch)
    return out


def observed_chars(rec, script):
    """Every non-space char present in this script's train+val labels."""
    chars = set()
    for split in ("train", "val"):
        p = os.path.join(rec, f"{script}_{split}.txt")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n").partition("\t")[2]
                chars.update(c for c in text if not c.isspace())
    return chars


def load_seed(path):
    with open(path, encoding="utf-8") as f:
        return {l.rstrip("\n") for l in f if l.rstrip("\n") and not l.rstrip("\n").isspace()}


def main():
    ap = argparse.ArgumentParser(description="Full Unicode-block rec dicts for PaddleOCR.")
    ap.add_argument("--rec-data", required=True, help="Output dir from crop_words.py.")
    ap.add_argument("--scripts", nargs="*", default=None,
                    help="Subset (default: every script found in rec-data that we have a block for).")
    ap.add_argument("--seed-dict", default=None,
                    help="Optional extra dict file to union in (e.g. a specialist's official dict).")
    ap.add_argument("--no-common", action="store_true",
                    help="Do NOT add ASCII digits/letters/punctuation (script block + labels only).")
    ap.add_argument("--no-unified", action="store_true",
                    help="Skip rebuilding unified_dict.txt as the union of the selected scripts.")
    ap.add_argument("--extra-labels", nargs="*", default=None,
                    help="Extra rec label files (crop<TAB>text), e.g. the pooled REAL crops "
                         "(bharat/mlt). Their observed characters are UNIONed into unified_dict.txt "
                         "so no real transcription is out-of-vocabulary. Full Unicode blocks already "
                         "cover them, but this is a guarantee for any script/char not otherwise present.")
    args = ap.parse_args()

    rec = os.path.abspath(args.rec_data)
    found = sorted(
        os.path.basename(p)[: -len("_train.txt")]
        for p in glob.glob(os.path.join(rec, "*_train.txt"))
        if not os.path.basename(p).startswith("unified")
    )
    scripts = args.scripts or [s for s in found if s in SCRIPT_BLOCKS]
    unknown = [s for s in scripts if s not in SCRIPT_BLOCKS]
    if unknown:
        raise SystemExit(f"no Unicode block defined for: {unknown} (add it to SCRIPT_BLOCKS)")
    if not scripts:
        raise SystemExit(f"no <script>_train.txt found in {rec} -- run crop_words.py first")

    seed = load_seed(args.seed_dict) if args.seed_dict else set()
    common = set() if args.no_common else (expand(COMMON_RANGES) | set(COMMON_PUNCT))

    print(f"scripts ({len(scripts)}): {', '.join(scripts)}\n")
    per_script = {}
    for s in scripts:
        block = expand(SCRIPT_BLOCKS[s])
        obs = observed_chars(rec, s)
        chars = block | common | seed | obs
        chars = {c for c in chars if c and not c.isspace()}
        per_script[s] = chars

        dict_path = os.path.join(rec, f"{s}_dict.txt")
        backup = os.path.join(rec, f"{s}_dict.corpus.txt")
        if os.path.isfile(dict_path) and not os.path.isfile(backup):
            shutil.copyfile(dict_path, backup)                    # keep the original once
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(chars)) + "\n")

        missed = obs - block - common - seed                      # observed chars outside the block
        oov = obs - chars                                         # should always be empty now
        print(f"[{s:11s}] dict={len(chars):5d}  (block={len(block)} "
              f"observed={len(obs)} observed_outside_block={len(missed)})"
              + (f"  !! {len(oov)} still-OOV" if oov else ""))

    if not args.no_unified and len(scripts) > 1:
        allc = set().union(*per_script.values())
        if args.extra_labels:
            extra = set()
            for lp in args.extra_labels:
                if not os.path.isfile(lp):
                    print(f"  [warn] --extra-labels file not found: {lp}"); continue
                with open(lp, encoding="utf-8") as f:
                    for line in f:
                        text = line.rstrip("\n").partition("\t")[2]
                        extra.update(c for c in text if not c.isspace())
            new = extra - allc
            allc |= extra
            print(f"[extra] unioned {len(extra)} observed real chars ({len(new)} not already covered)")
        out = os.path.join(rec, "unified_dict.txt")
        if os.path.isfile(out) and not os.path.isfile(out + ".corpus"):
            shutil.copyfile(out, out + ".corpus")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(allc)) + "\n")
        print(f"\n[unified] {len(allc)} unique characters -> {out}")

    print("\n[done] dicts rebuilt. Corpus dicts backed up to <script>_dict.corpus.txt")
    print("       re-run make_config.py so the config points at the new dict size.")


if __name__ == "__main__":
    main()
