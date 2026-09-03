#!/usr/bin/env bash
# Generic multi-connection downloader (aria2c) — installs aria2 into the active
# conda env if missing. Much faster than wget when the server throttles per-connection.
#
# Usage:  DEST=<dir> bash download_aria.sh URL [URL ...]
set -euo pipefail

DEST="${DEST:?set DEST=<target dir>}"
mkdir -p "$DEST"; cd "$DEST"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "[setup] aria2c not found — installing via conda-forge ..."
  conda install -y -c conda-forge aria2
fi

for u in "$@"; do
  echo "[aria] $u"
  aria2c -x16 -s16 -k1M --continue=true --check-certificate=false \
         --max-tries=5 --retry-wait=5 --summary-interval=30 \
         -o "$(basename "$u")" "$u"
done
echo "[aria] done. sizes:"; ls -lh "$DEST"
