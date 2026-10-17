#!/bin/bash
# Download DocILE (needs DOCILE_TOKEN in the environment) and convert it.
# Raw data goes to $DOCILE_RAW (default /content/docile_raw), fast local disk on Colab.
set -euo pipefail
: "${DOCILE_TOKEN:?set DOCILE_TOKEN (free from https://docile.rossum.ai)}"
raw="${DOCILE_RAW:-/content/docile_raw}"
if [ ! -f data/annotated-trainval.zip ]; then
  [ -d /tmp/docile_repo ] || git clone -q --depth 1 https://github.com/rossumai/docile.git /tmp/docile_repo
  # the official script prints the download URL, which contains the token: hide it
  bash /tmp/docile_repo/download_dataset.sh "$DOCILE_TOKEN" annotated-trainval data 2>&1 | sed "s/$DOCILE_TOKEN/***/g"
fi
mkdir -p "$raw" && unzip -qn data/annotated-trainval.zip -d "$raw"
python -m docext.prepare docile --docile-path "$raw" --out data
