#!/usr/bin/env python3
"""Find fonts that CANNOT correctly render a script, so you can delete them.

For every script's font (discovered exactly like generate.py, via `fc-list`),
this shapes all the vocab words with HarfBuzz (uharfbuzz -- the same engine
Pillow's Raqm uses) and flags any font that emits a `.notdef` glyph (glyph id
0). A `.notdef` means the font is MISSING a glyph for that string, so it would
render a tofu box / wrong output. That's the definitive automatic signal that a
font can't render the script.

Output:
  - a per-script OK/FAIL report to the console;
  - <report>.txt              : every (font, script, failed_words);
  - <report>.delete.txt       : unique font file paths that failed (the
                                delete-list you asked for).

Optionally `--quarantine DIR` MOVES the failing font files into DIR (reversible)
instead of you deleting them by hand.

NOTE: `.notdef` catches "missing glyphs / can't render". It does NOT catch a
font that renders but with subtly wrong shaping (there's no ground truth to
diff against). For the fonts that PASS here, still eyeball a few debug images.

Requires: pip install uharfbuzz
Run:
    python check_fonts.py --vocab-dir ./vocab --report bad_fonts
"""

import os
import random
import argparse
import shutil

try:
    import uharfbuzz as hb
except ImportError:
    raise SystemExit("Install uharfbuzz first:  pip install uharfbuzz")

import generate as G   # reuse SCRIPTS, discover_fonts(), load_vocab()


def has_notdef(font_path, text, _cache={}):
    """True if shaping `text` with the font yields any .notdef (gid 0)."""
    f = _cache.get(font_path)
    if f is None:
        blob = hb.Blob.from_file_path(font_path)
        f = hb.Font(hb.Face(blob))
        _cache[font_path] = f
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(f, buf)
    return any(info.codepoint == 0 for info in buf.glyph_infos)


def main():
    ap = argparse.ArgumentParser(
        description="Flag fonts that can't render their script (.notdef).")
    ap.add_argument("--vocab-dir", default="./vocab")
    ap.add_argument("--scripts", nargs="*", default=None,
                    help="Subset of scripts to check (default: all configured).")
    ap.add_argument("--report", default="bad_fonts",
                    help="Output report basename (writes .txt and .delete.txt).")
    ap.add_argument("--fail-frac", type=float, default=0.0,
                    help="Flag a font if MORE than this fraction of words fail "
                         "(0.0 = flag on any single .notdef).")
    ap.add_argument("--quarantine", default=None,
                    help="If set, MOVE failing font files into this directory.")
    ap.add_argument("--verbose", action="store_true",
                    help="Also print every OK font.")
    ap.add_argument("--max-words", type=int, default=0,
                    help="Check each font against at most this many random words "
                         "per script (0 = all). Use a few hundred for big vocabs; "
                         "a broken font .notdefs on most words anyway.")
    args = ap.parse_args()

    fonts_idx = G.discover_fonts()
    vocab = G.load_vocab(args.vocab_dir)
    scripts = args.scripts or list(G.SCRIPTS.keys())

    report_rows = []        # (script, path, n_failed, n_total, failed_words)
    bad_paths = set()
    n_checked = 0

    for script in scripts:
        fonts = fonts_idx.get(script, [])
        words = vocab.get(script, [])
        if not fonts or not words:
            continue
        if args.max_words and len(words) > args.max_words:
            words = random.sample(words, args.max_words)   # sample for speed
        print(f"\n=== {script}: checking {len(fonts)} font(s) "
              f"against {len(words)} word(s) ===")
        for fp in fonts:
            n_checked += 1
            failed = []
            unreadable = False
            for w in words:
                try:
                    if has_notdef(fp, w):
                        failed.append(w)
                except Exception:
                    unreadable = True
                    break
            if unreadable:
                report_rows.append((script, fp, len(words), len(words),
                                    ["<font failed to load>"]))
                bad_paths.add(fp)
                print(f"  FAIL  {os.path.basename(fp)}: unreadable/corrupt")
                continue
            if failed and (len(failed) / len(words)) > args.fail_frac:
                report_rows.append((script, fp, len(failed), len(words), failed))
                bad_paths.add(fp)
                print(f"  FAIL  {os.path.basename(fp)}: "
                      f"{len(failed)}/{len(words)} words .notdef "
                      f"e.g. {failed[:3]}")
            elif args.verbose:
                print(f"  OK    {os.path.basename(fp)}")

    # write reports
    rpt = args.report + ".txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# fonts checked: {n_checked} | flagged: {len(bad_paths)}\n")
        for script, fp, nf, nt, words in report_rows:
            f.write(f"{script}\t{nf}/{nt}\t{fp}\t{' '.join(words[:10])}\n")
    delrpt = args.report + ".delete.txt"
    with open(delrpt, "w", encoding="utf-8") as f:
        for p in sorted(bad_paths):
            f.write(p + "\n")

    print(f"\n[summary] checked {n_checked} fonts, "
          f"{len(bad_paths)} cannot render their script.")
    print(f"  full report : {rpt}")
    print(f"  delete-list : {delrpt}")

    if args.quarantine and bad_paths:
        os.makedirs(args.quarantine, exist_ok=True)
        moved = 0
        for p in sorted(bad_paths):
            try:
                shutil.move(p, os.path.join(args.quarantine,
                                            os.path.basename(p)))
                moved += 1
            except Exception as e:
                print(f"  could not move {p}: {e}")
        print(f"  quarantined {moved} font(s) -> {args.quarantine}")
        print("  -> run `fc-cache -f ~/.fonts` so fontconfig forgets them.")
    elif bad_paths:
        print("\nTo delete them yourself (review first!):")
        print(f"  xargs rm < {delrpt}   &&   fc-cache -f ~/.fonts")


if __name__ == "__main__":
    main()
