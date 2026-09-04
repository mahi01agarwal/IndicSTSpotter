#!/usr/bin/env bash
# Generate the 25k dataset on nltm, sharded across N parallel worker processes.
# Each worker produces a disjoint slice of global scene indices into the SAME
# out_dir (unique filenames by global index; per-shard manifest parts merged at
# the end). Per-scene seeding makes every image reproducible regardless of shard.
#
# Run on nltm, inside the raqm_test env, from ~/indic_spotter_synth:
#     conda activate raqm_test
#     bash generate_25k.sh
#
# Tune WORKERS to the box: a good start is (physical cores - 2). Generation is
# CPU-bound (Raqm/PIL), so more workers ~= linear speedup until cores/RAM saturate.
set -euo pipefail

CONFIG="${CONFIG:-dataset_config_25k.json}"   # override: CONFIG=dataset_config_det.json ...
OUT="${OUT:-out_dataset_25k}"                 # override: OUT=out_dataset_25k_realistic ...
# Which generator to shard. generate_grid_transforms.py tiles a dense 7x6 grid
# (~121 words/image) -- good for the RECOGNISER, wrong for the DETECTOR, which
# learns from it that text covers a third of every frame. For detector data use:
#   GEN=generate_realistic.py CONFIG=dataset_config_det.json OUT=out_dataset_det bash generate_25k.sh
GEN="${GEN:-generate_grid_transforms.py}"
TOTAL="${TOTAL:-25000}"
WORKERS="${WORKERS:-6}"       # override: WORKERS=12 bash generate_25k.sh
BG_DIR="${BG_DIR:-}"          # optional override, e.g. BG_DIR=~/data/places365 bash generate_25k.sh

per=$(( (TOTAL + WORKERS - 1) / WORKERS ))    # ceil, so the shards cover all TOTAL
echo "[gen] $TOTAL scenes, $WORKERS workers, ~$per each -> $OUT"
[ -n "$BG_DIR" ] && echo "[gen] background override: $BG_DIR"

pids=()
for w in $(seq 0 $((WORKERS - 1))); do
    start=$(( w * per ))
    [ "$start" -ge "$TOTAL" ] && break
    count=$per
    [ $(( start + count )) -gt "$TOTAL" ] && count=$(( TOTAL - start ))    # last shard trimmed
    extra=()
    [ -n "$BG_DIR" ] && extra=(--bg-dir "$BG_DIR")
    echo "[shard $w] indices $start..$(( start + count - 1 ))"
    python "$GEN" --config "$CONFIG" \
        --out-dir "$OUT" --start-index "$start" --num-scenes "$count" \
        "${extra[@]}" > "$OUT.shard_${w}.log" 2>&1 &
    pids+=($!)
done

# wait for every worker; fail loudly if any shard dies
fail=0
for p in "${pids[@]}"; do
    wait "$p" || { echo "[gen] a shard failed (pid $p) — check $OUT.shard_*.log"; fail=1; }
done
[ "$fail" -ne 0 ] && exit 1

# merge the per-shard manifest parts into one manifest.jsonl
if ls "$OUT"/manifest.part_*.jsonl >/dev/null 2>&1; then
    cat "$OUT"/manifest.jsonl "$OUT"/manifest.part_*.jsonl > "$OUT/manifest.merged.jsonl" 2>/dev/null || \
        cat "$OUT"/manifest.part_*.jsonl > "$OUT/manifest.merged.jsonl"
    mv "$OUT/manifest.merged.jsonl" "$OUT/manifest.jsonl"
fi

n_img=$(ls "$OUT/images" | wc -l)
n_ann=$(ls "$OUT/annotations" | wc -l)
n_man=$(wc -l < "$OUT/manifest.jsonl" 2>/dev/null || echo 0)
echo "[done] images=$n_img annotations=$n_ann manifest_lines=$n_man -> $OUT"
