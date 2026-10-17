#!/bin/bash
# Download DocILE (needs DOCILE_TOKEN in the environment) and convert it.
# Raw data goes to $DOCILE_RAW (default /content/docile_raw), fast local disk on Colab.
set -euo pipefail
: "${DOCILE_TOKEN:?set DOCILE_TOKEN (free from https://docile.rossum.ai)}"
raw="${DOCILE_RAW:-/content/docile_raw}"
if [ ! -f data/labeled-trainval.zip ]; then
  [ -d /tmp/docile_repo ] || git clone -q --depth 1 https://github.com/rossumai/docile.git /tmp/docile_repo
  bash /tmp/docile_repo/download_dataset.sh "$DOCILE_TOKEN" labeled-trainval data
fi
mkdir -p "$raw" && unzip -qn data/labeled-trainval.zip -d "$raw"
python -m docext.prepare docile --docile-path "$raw" --out data
