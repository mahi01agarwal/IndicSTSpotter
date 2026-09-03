#!/usr/bin/env python3
"""render_paper_gallery.py

Small, display-quality synthetic renders for the ICVGIP 2026 supplementary
material. NOT training data — one prominent word per canvas on plain
"sign-panel" backgrounds (solid coloured panels with faint texture) OR on a
curated set of plain real backgrounds. Correct Indic conjunct shaping is done
via PIL + libraqm, so this script must run in an env where Pillow was built
with libraqm (on nltm: `conda activate raqm_test`).

Prereqs (nltm, one-liner sanity check):
    python -c "from PIL import features; print(features.check('raqm'))"   # True

Expected layout (defaults match ~/indic_spotter_synth on nltm):
    <vocab-dir>/<script>.txt          one word per line, UTF-8
    <font-dir>/<script>/*.ttf|*.otf|*.ttc     already hygiene-filtered
    (optional) <bg-dir>/*.png|*.jpg           curated plain backgrounds

If your vocab / font directories use different names per script (e.g. hi.txt
instead of deva.txt), symlink them or pass a JSON via --script-map. See
`DEFAULT_SCRIPT_MAP` below for the naming this script assumes.

Typical run on nltm:
    conda activate raqm_test
    cd ~/indic_spotter_synth
    python render_paper_gallery.py \
        --vocab-dir vocab --font-dir fonts \
        --out-dir ~/paper_gallery --n-per-script 8 --seed 42

    # add real plain BGs to the mix:
    python render_paper_gallery.py --bg-dir bg_paper_plain [...]

Then on nltm:
    rsync -avz ~/paper_gallery/ ~/paper_figs_pull/paper_gallery/
And on Mac:
    rsync -avz mahima@10.4.16.81:~/paper_figs_pull/paper_gallery/ \
        ~/Desktop/ReportsCVIT/paper_figs/paper_gallery/
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# ------------------------------------------------------------------------
# The ten scripts we render. Ordering here also fixes the presentation order
# in contact sheets.
SCRIPTS = ["deva", "ben", "gur", "guj", "odi", "tam", "tel", "kan", "mal", "lat"]

# Vocab file stem per script (matches nltm layout ~/indic_spotter_synth/vocab/*.txt).
# Override with --script-map '{...}' if your files use different names.
DEFAULT_SCRIPT_MAP: dict[str, str] = {
    "deva": "devanagari",
    "ben":  "bengali",
    "gur":  "gurmukhi",
    "guj":  "gujarati",
    "odi":  "oriya",
    "tam":  "tamil",
    "tel":  "telugu",
    "kan":  "kannada",
    "mal":  "malayalam",
    "lat":  "latin",
}

# fontconfig language codes used to discover fonts per script via
# `fc-list :lang=<code> file`. All Devanagari-capable fonts include Hindi;
# same trick for the other scripts' primary language.
FC_LANG: dict[str, str] = {
    "deva": "hi",  "ben":  "bn",  "gur":  "pa",  "guj":  "gu",
    "odi":  "or",  "tam":  "ta",  "tel":  "te",  "kan":  "kn",
    "mal":  "ml",  "lat":  "en",
}

# Six palettes chosen to read as plausible signage: cream board, highway
# blue, amber warning, brushed metal, olive board, red panel. (bg_rgb, ink_rgb)
SIGN_PALETTE = [
    ((248, 245, 232), (60, 40, 20)),
    ((28, 60, 128),  (255, 255, 240)),
    ((235, 220, 90), (30, 30, 30)),
    ((238, 238, 236),(30, 30, 30)),
    ((72, 90, 60),   (245, 240, 220)),
    ((186, 30, 40),  (250, 245, 235)),
]

# Display canvas — larger than training (885x497) so the gallery reads well
# when scaled down in the paper.
CANVAS_W, CANVAS_H = 1200, 600


# =========================================================================
# helpers
# =========================================================================
def check_raqm() -> None:
    if not features.check("raqm"):
        sys.exit(
            "Pillow is not built with libraqm. On nltm: `conda activate raqm_test`."
        )


def load_vocab(vocab_dir: Path, script: str, script_map: dict[str, str]) -> list[str]:
    stem = script_map.get(script, script)
    p = vocab_dir / f"{stem}.txt"
    if not p.exists():
        return []
    words = [w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]
    # 3-12 chars: long enough to show conjuncts, short enough to be legible
    # at gallery size without wrapping.
    return [w for w in words if 3 <= len(w) <= 12]


def load_fonts_fc(script: str, bad_fonts: set[str]) -> list[Path]:
    """Discover fonts for `script` via fontconfig (`fc-list :lang=... file`).

    This matches the original pipeline's per-script discovery (see PROJECT
    master doc: "fc-list per script -> HarfBuzz .notdef hygiene"). We do
    NOT re-run the HarfBuzz hygiene here — trust the existing bad_fonts.txt.
    """
    lang = FC_LANG.get(script)
    if lang is None:
        return []
    try:
        out = subprocess.run(
            ["fc-list", f":lang={lang}", "file"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[warn] fc-list failed for {script}: {e}", file=sys.stderr)
        return []
    fonts = []
    for line in out.splitlines():
        # `fc-list :lang=hi file` prints "PATH:" per line; split on ':' and
        # take the path field (also handles the bare-path variant).
        path_str = line.split(":", 1)[0].strip()
        if not path_str:
            continue
        p = Path(path_str)
        if p.suffix.lower() in {".ttf", ".otf", ".ttc"} and p.name not in bad_fonts:
            fonts.append(p)
    # dedup while preserving order
    seen: set[str] = set()
    uniq = []
    for f in fonts:
        if str(f) in seen:
            continue
        seen.add(str(f))
        uniq.append(f)
    return uniq


def load_fonts_dir(font_dir: Path, script: str, script_map: dict[str, str],
                   bad_fonts: set[str]) -> list[Path]:
    stem = script_map.get(script, script)
    d = font_dir / stem
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".ttf", ".otf", ".ttc"} and p.name not in bad_fonts
    )


def load_bgs(bg_dir: Path | None) -> list[Path]:
    if bg_dir is None or not bg_dir.exists():
        return []
    return sorted(
        p for p in bg_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def load_bad_fonts(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


# =========================================================================
# backgrounds
# =========================================================================
def make_solid_bg(rng: random.Random) -> tuple[Image.Image, tuple[int, int, int]]:
    """Solid sign-panel colour with faint noise + vignette."""
    bg_color, ink_color = rng.choice(SIGN_PALETTE)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg_color)
    noise = Image.effect_noise((CANVAS_W, CANVAS_H), 4).convert("RGB")
    img = Image.blend(img, noise, 0.08)
    # subtle vignette so it doesn't look CG-flat
    mask = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((-CANVAS_W // 2, -CANVAS_H // 2,
               CANVAS_W + CANVAS_W // 2, CANVAS_H + CANVAS_H // 2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    img = Image.composite(img, dark, mask)
    return img, ink_color


def mean_luminance(img: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = img.crop(box).convert("L").resize((1, 1))
    return crop.getpixel((0, 0))


def pick_ink_color(bg_img: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Black or off-white based on the target region's mean luminance."""
    return (30, 30, 30) if mean_luminance(bg_img, box) > 128 else (245, 245, 240)


# =========================================================================
# renderer
# =========================================================================
def render_one(word: str, font_path: Path, bg_img: Image.Image,
               ink: tuple[int, int, int], rng: random.Random
               ) -> tuple[Image.Image, tuple[int, int, int, int]]:
    W, H = bg_img.size
    target_h = int(H * rng.uniform(0.22, 0.32))
    font = ImageFont.truetype(str(font_path), size=target_h,
                              layout_engine=ImageFont.Layout.RAQM)
    tmp = Image.new("RGB", (10, 10))
    l, t, r, b = ImageDraw.Draw(tmp).textbbox((0, 0), word, font=font)
    tw, th = r - l, b - t

    # centre with a small random jitter so the gallery doesn't look robotic
    x = (W - tw) // 2 + rng.randint(-30, 30) - l
    y = (H - th) // 2 + rng.randint(-15, 15) - t
    bbox = (x + l, y + t, x + l + tw, y + t + th)

    # soft drop-shadow so text lifts off the panel
    out = bg_img.convert("RGBA")
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((x + 3, y + 3), word, font=font, fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(2.2))
    out.alpha_composite(shadow)
    ImageDraw.Draw(out).text((x, y), word, font=font, fill=ink)
    return out.convert("RGB"), bbox


def render_script(script: str, args, rng: random.Random,
                  bgs: list[Path], script_map: dict[str, str],
                  bad_fonts: set[str]) -> list[dict]:
    vocab = load_vocab(args.vocab_dir, script, script_map)
    if args.font_dir is not None:
        fonts = load_fonts_dir(args.font_dir, script, script_map, bad_fonts)
    else:
        fonts = load_fonts_fc(script, bad_fonts)
    if not vocab or not fonts:
        print(f"[skip] {script}: vocab={len(vocab)} fonts={len(fonts)}", file=sys.stderr)
        return []
    out_dir = args.out_dir / script
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    for i in range(args.n_per_script):
        word = rng.choice(vocab)
        font_path = rng.choice(fonts)
        if bgs and rng.random() < args.real_bg_prob:
            bg = Image.open(rng.choice(bgs)).convert("RGB").resize((CANVAS_W, CANVAS_H))
            probe = (int(0.2 * CANVAS_W), int(0.35 * CANVAS_H),
                     int(0.8 * CANVAS_W), int(0.65 * CANVAS_H))
            ink = pick_ink_color(bg, probe)
            bg_kind = "real"
        else:
            bg, ink = make_solid_bg(rng)
            bg_kind = "solid"
        img, bbox = render_one(word, font_path, bg, ink, rng)
        stem = f"{script}_{i:02d}"
        img.save(out_dir / f"{stem}.png", "PNG")
        meta.append({
            "file": f"{stem}.png",
            "script": script,
            "word": word,
            "font": font_path.name,
            "bbox": bbox,
            "bg": bg_kind,
        })
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok] {script}: {len(meta)} renders -> {out_dir}")
    return meta


def make_contact_sheet(out_dir: Path, script: str, meta: list[dict], cols: int = 3) -> None:
    imgs = [Image.open(out_dir / script / m["file"]) for m in meta]
    if not imgs:
        return
    cols = min(cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    thumb_w, thumb_h = 400, 200
    pad = 10
    sheet = Image.new("RGB",
                      (cols * thumb_w + (cols + 1) * pad,
                       rows * thumb_h + (rows + 1) * pad),
                      (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im.resize((thumb_w, thumb_h)),
                    (pad + c * (thumb_w + pad), pad + r * (thumb_h + pad)))
    sheet.save(out_dir / f"{script}_contact.png", "PNG")


# =========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-dir", default="~/indic_spotter_synth/vocab")
    ap.add_argument("--font-dir",  default=None,
                    help="optional: per-script fonts dir with <script>/*.ttf; "
                         "omit to auto-discover via `fc-list :lang=...`")
    ap.add_argument("--bg-dir",    default=None,
                    help="directory of curated plain backgrounds; omit for solid panels only")
    ap.add_argument("--out-dir",   default="~/paper_gallery")
    ap.add_argument("--scripts",   default=",".join(SCRIPTS))
    ap.add_argument("--n-per-script", type=int, default=6)
    ap.add_argument("--real-bg-prob", type=float, default=0.5,
                    help="fraction of renders that use a real BG (rest = solid panel)")
    ap.add_argument("--script-map", default="{}",
                    help='JSON mapping script code -> file stem, e.g. \'{"deva":"hi"}\'')
    ap.add_argument("--bad-fonts", default=None,
                    help="path to bad_fonts.txt (font filenames to exclude, one per line)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.vocab_dir = Path(args.vocab_dir).expanduser()
    args.font_dir  = Path(args.font_dir).expanduser() if args.font_dir else None
    args.out_dir   = Path(args.out_dir).expanduser()
    args.bg_dir    = Path(args.bg_dir).expanduser() if args.bg_dir else None

    check_raqm()
    script_map = DEFAULT_SCRIPT_MAP | json.loads(args.script_map)
    bad_fonts = load_bad_fonts(Path(args.bad_fonts).expanduser() if args.bad_fonts else None)
    rng = random.Random(args.seed)
    bgs = load_bgs(args.bg_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.scripts.split(","):
        s = s.strip()
        if not s:
            continue
        meta = render_script(s, args, rng, bgs, script_map, bad_fonts)
        make_contact_sheet(args.out_dir, s, meta)


if __name__ == "__main__":
    main()
