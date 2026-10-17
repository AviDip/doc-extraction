#!/bin/bash
# Download WildReceipt (public) and convert it. Usage: bash scripts/get_wildreceipt.sh
set -euo pipefail
mkdir -p data/wildreceipt_raw
[ -f data/wildreceipt.tar ] || wget -q https://download.openmmlab.com/mmocr/data/wildreceipt.tar -O data/wildreceipt.tar
tar -xf data/wildreceipt.tar -C data/wildreceipt_raw --skip-old-files
raw=$(dirname "$(find data/wildreceipt_raw -name class_list.txt | head -1)")
python -m docext.prepare wildreceipt --wildreceipt-path "$raw" --out data
