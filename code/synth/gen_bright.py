"""Bright, contrast-aware text palette — sign-painted look, still readable."""
import random, sys, runpy, importlib

LIGHT_PALETTE = [
    (255, 235, 60),   # bright yellow
    (60, 220, 220),   # bright cyan
    (100, 240, 120),  # light green
    (255, 150, 200),  # pink
    (255, 200, 100),  # peach
    (255, 255, 255),  # white
]
DARK_PALETTE = [
    (180, 20, 20),    # dark red
    (20,  50, 180),   # dark blue
    (30, 100,  30),   # dark green
    (100, 30, 130),   # purple
    (200, 100, 20),   # burnt orange
    (30,  30,  30),   # near-black
]

def _pick(*args, **kwargs):
    bg = None
    for a in args:
        if isinstance(a, (tuple, list)) and len(a) >= 3:
            try:
                bg = tuple(int(x) for x in a[:3]); break
            except Exception:
                pass
    if bg is None:
        return random.choice(LIGHT_PALETTE + DARK_PALETTE)
    r, g, b = bg
    lum = 0.299*r + 0.587*g + 0.114*b
    return random.choice(LIGHT_PALETTE if lum < 128 else DARK_PALETTE)

for mod in ("generate", "generate_transforms", "generate_grid_transforms"):
    try:
        m = importlib.import_module(mod)
        if hasattr(m, "pick_color"):
            setattr(m, "pick_color", _pick)
    except ImportError:
        pass

sys.argv[0] = "generate_grid_transforms.py"
runpy.run_module("generate_grid_transforms", run_name="__main__", alter_sys=True)
