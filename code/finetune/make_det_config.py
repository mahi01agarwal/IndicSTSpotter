#!/usr/bin/env python3
"""Patch a PaddleOCR DETECTION (DB) config for our synthetic data (no hand-editing).

Detection configs have no character dict and no rec head — this just injects the data
paths, the pretrained det weights, and a few training knobs.

Run from the cloned PaddleOCR repo, after make_det_labels.py:
    python make_det_config.py \
        --base configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
        --det-data $SCR/det_data --data-dir $SCR/out_dataset \
        --pretrained ./pretrain/PP-OCRv5_mobile_det.pdparams \
        --epochs 100 --batch-size 8 --out my_det.yml
"""
import os
import argparse
import yaml


def main():
    ap = argparse.ArgumentParser(description="Patch a PP-OCRv5 det (DB) config for our data.")
    ap.add_argument("--base", required=True, help="Base det config, e.g. configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")
    ap.add_argument("--det-data", required=True, help="Dir with det_train.txt/det_val.txt (from make_det_labels.py).")
    ap.add_argument("--data-dir", required=True, help="Path the image paths are relative to (e.g. out_dataset).")
    ap.add_argument("--pretrained", required=True, help="Path to the pretrained det .pdparams.")
    ap.add_argument("--out", default="my_det.yml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Det uses more memory than rec; 8 fits ~11 GB for the mobile det. Lower if OOM.")
    ap.add_argument("--eval-limit-side", type=int, default=960,
                    help="Cap the eval-time image long side (DetResizeForTest limit_side_len). "
                         "Det eval runs batch=1 near native res; on 11GB cards the training memory "
                         "pool leaves ~1.4GB, so a 12MP image OOMs at the default max_side_limit=4000. "
                         "960 keeps eval memory tiny; only used to select best_accuracy.")
    args = ap.parse_args()

    det = os.path.abspath(args.det_data)
    data_dir = os.path.abspath(args.data_dir)
    train_txt = os.path.join(det, "det_train.txt")
    val_txt = os.path.join(det, "det_val.txt")
    for p in (train_txt, val_txt):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} — run make_det_labels.py first")

    with open(args.base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg.setdefault("Global", {})
    g["use_gpu"] = True
    g["epoch_num"] = args.epochs
    g["pretrained_model"] = args.pretrained
    g["save_model_dir"] = "./output/det"

    for split, txt in (("Train", train_txt), ("Eval", val_txt)):
        node = cfg.setdefault(split, {})
        ds = node.setdefault("dataset", {})
        ds["data_dir"] = data_dir
        ds["label_file_list"] = [txt]
        # det EVAL must be batch 1: eval images keep native (variable) sizes, so
        # batching >1 fails to collate (np.stack shape mismatch). Only Train batches.
        node.setdefault("loader", {})["batch_size_per_card"] = args.batch_size if split == "Train" else 1

    # Cap eval-time resolution. Det eval keeps images near native size (batch 1); the default
    # DetResizeForTest has max_side_limit=4000, and with the training memory pool held during
    # eval there's only ~1.4GB free on an 11GB card -> a 12MP val image OOMs mid-eval. Force the
    # long side <= eval_limit_side. Val hmean is only for picking best_accuracy, so this is safe;
    # do final-benchmark eval separately (eval_det_real.py) at full res on a fresh process.
    for t in cfg.get("Eval", {}).get("dataset", {}).get("transforms", []) or []:
        if isinstance(t, dict) and "DetResizeForTest" in t:
            t["DetResizeForTest"] = {"limit_side_len": args.eval_limit_side, "limit_type": "max"}

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[ok] wrote {args.out}")
    print(f"     data_dir   = {data_dir}")
    print(f"     train/val  = {train_txt} / {val_txt}")
    print(f"     pretrained = {args.pretrained}")
    print(f"\nnext:  python tools/train.py -c {args.out}")


if __name__ == "__main__":
    main()
