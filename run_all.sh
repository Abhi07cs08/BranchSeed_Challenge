#!/usr/bin/env bash
# Run the whole dev set, then score it. Usage: ./run_all.sh data refs
set -euo pipefail
DATA="${1:-data}"; REFS="${2:-refs}"
mkdir -p preds viz; rm -f features.csv
for d in "$DATA"/*/; do
  c=$(basename "$d")
  python run.py --image "$d"/*orig*.nii* --aorta-mask "$d"/*mask*.nii* \
                --output "preds/$c.json" --viz "viz/$c.png" --features features.csv
done
[ -d "$REFS" ] && python evaluate.py --pred preds/ --ref "$REFS" --match-mm 3,5,8 --per-case
