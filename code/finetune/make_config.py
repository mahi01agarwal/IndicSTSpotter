#!/usr/bin/env python3
"""Patch a PaddleOCR PP-OCRv5 rec config for one of our scripts (no hand-editing).

Copies a base rec config and injects our paths: the per-script dictionary, train/val
label lists, data dir, pretrained weights, and a few training knobs.

Run from the cloned PaddleOCR repo, after crop_words.py:
    python /path/to/FineTune/make_config.py \
        --base configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        --script devanagari \
        --rec-data /ssd_scratch/cvit/mahima/rec_data \
        --pretrained ./pretrain/PP-OCRv5_server_rec.pdparams \
        --max-text-length 40 --out my_devanagari.yml
"""
import os
import argparse
import yaml


def main():
    ap = argparse.ArgumentParser(description="Patch a PP-OCRv5 rec config for one script.")
    ap.add_argument("--base", required=True, help="Base rec config, e.g. configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml")
    ap.add_argument("--script", required=True)
    ap.add_argument("--rec-data", required=True, help="Output dir from crop_words.py (has <script>_*.txt).")
    ap.add_argument("--pretrained", required=True, help="Path to the pretrained .pdparams. "
                    "For a warm-start from the stock Devanagari specialist, use the MOBILE base "
                    "config and devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams.")
    ap.add_argument("--dict", default=None,
                    help="Override the character dict (default: <rec-data>/<script>_dict.txt). "
                         "Point at a specialist's official dict to also transfer its CTC head.")
    ap.add_argument("--no-space-char", dest="use_space_char", action="store_false",
                    help="Set use_space_char=False. MUST match the specialist's setting when "
                         "warm-starting with its dict, or num_classes differs by 1 and the head "
                         "still reinitializes.")
    ap.set_defaults(use_space_char=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-text-length", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Per-card batch. 32 fits ~12 GB for PP-OCRv5 server_rec; raise if you have room.")
    args = ap.parse_args()

    rec = os.path.abspath(args.rec_data)
    dict_path = os.path.abspath(args.dict) if args.dict else os.path.join(rec, f"{args.script}_dict.txt")
    train_txt = os.path.join(rec, f"{args.script}_train.txt")
    val_txt = os.path.join(rec, f"{args.script}_val.txt")
    for p in (dict_path, train_txt, val_txt):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} — run crop_words.py first (is the script name '{args.script}' right?)")

    with open(args.base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg.setdefault("Global", {})
    g["use_gpu"] = True
    g["epoch_num"] = args.epochs
    g["pretrained_model"] = args.pretrained
    g["character_dict_path"] = dict_path
    g["use_space_char"] = args.use_space_char
    g["max_text_length"] = args.max_text_length
    g["save_model_dir"] = f"./output/{args.script}_rec"

    for split, txt in (("Train", train_txt), ("Eval", val_txt)):
        node = cfg.setdefault(split, {})
        ds = node.setdefault("dataset", {})
        ds["data_dir"] = rec
        ds["label_file_list"] = [txt]
        node.setdefault("loader", {})["batch_size_per_card"] = args.batch_size

    # MultiScaleSampler's first_bs actually drives the Train batch size — set it too
    samp = cfg.get("Train", {}).get("sampler")
    if isinstance(samp, dict) and "first_bs" in samp:
        samp["first_bs"] = args.batch_size

    # keep the NRTR branch's max length in sync with Global (this rec model has a CTC + NRTR head)
    try:
        heads = cfg["Architecture"]["Head"]["head_list"]
        head_iter = heads.values() if isinstance(heads, dict) else heads
        for h in head_iter:
            if isinstance(h, dict) and "NRTRHead" in h:
                h["NRTRHead"]["max_text_length"] = args.max_text_length
        if isinstance(heads, dict) and "NRTRHead" in heads:
            heads["NRTRHead"]["max_text_length"] = args.max_text_length
    except (KeyError, TypeError):
        pass

    out = args.out or f"my_{args.script}.yml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[ok] wrote {out}")
    print(f"     dict       = {dict_path}")
    print(f"     train/val  = {train_txt} / {val_txt}")
    print(f"     pretrained = {args.pretrained}")
    print(f"\nnext:  python tools/train.py -c {out}")


if __name__ == "__main__":
    main()
