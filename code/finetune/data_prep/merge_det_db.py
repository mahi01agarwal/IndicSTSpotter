#!/usr/bin/env python3
"""merge_det_db.py — merge real DB detector label files into a train/val split.

Concatenates PaddleOCR DB det label files (each line: <abs_img>\t[{"transcription","points"},...]),
dedups by image path, shuffles (seeded), and carves a real val split for early-stopping /
box_thresh tuning. Absolute image paths are preserved, so the training config uses data_dir=/.

Detector = REAL-ONLY headline set: pass the three real det files (Bharat train + MLT17 + MLT19).
Do NOT pass Bharat's test file — it's the held-out benchmark. For the synth+real ablation later,
just add the synthetic det file to --inputs and re-run into a different --out-dir.

    python merge_det_db.py \
      --inputs $SCR/det_data_real/bharat/bharat_det_train.txt \
               $SCR/det_data_real/mlt17/mlt17_det.txt \
               $SCR/det_data_real/mlt19/mlt19_det.txt \
      --out-dir $SCR/det_data_real/merged_real
"""
import argparse, json, os, random, collections


def count_boxes(rows):
    imgs = len(rows); leg = ign = 0
    for r in rows:
        try:
            boxes = json.loads(r.split('\t', 1)[1])
        except Exception:
            continue
        for b in boxes:
            if b.get('transcription') == '###':
                ign += 1
            else:
                leg += 1
    return imgs, leg, ign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='det label .txt files to merge')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--val-frac', type=float, default=0.03, help='fraction held out for val')
    ap.add_argument('--val-max', type=int, default=1500, help='cap val set size (imgs)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    seen = {}                      # img_path -> line (first wins)
    order = []
    per_src = collections.Counter()
    dup = 0
    for inp in args.inputs:
        tag = os.path.basename(inp)
        with open(inp, encoding='utf-8') as f:
            for ln in f:
                ln = ln.rstrip('\n')
                if not ln.strip():
                    continue
                img = ln.split('\t', 1)[0]
                if img in seen:
                    dup += 1; continue
                seen[img] = ln
                order.append(img)
                per_src[tag] += 1

    lines = [seen[i] for i in order]
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    n_val = min(int(len(lines) * args.val_frac), args.val_max)
    val, train = lines[:n_val], lines[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'det_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train) + '\n')
    with open(os.path.join(args.out_dir, 'det_val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val) + '\n')

    ti, tl, tg = count_boxes(train)
    vi, vl, vg = count_boxes(val)
    rep = [f'=== merged det -> {args.out_dir} ===',
           f'  inputs (imgs kept per source): {dict(per_src)}',
           f'  duplicate images dropped     : {dup}',
           f'  TRAIN: imgs={ti}  legible={tl}  ignore(###)={tg}',
           f'  VAL  : imgs={vi}  legible={vl}  ignore(###)={vg}']
    txt = '\n'.join(rep)
    print(txt)
    with open(os.path.join(args.out_dir, 'merge_det_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
