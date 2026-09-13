#!/usr/bin/env bash
# Run the detector over every case, then score if references are available.
#
#   ./run_all.sh [DATA_DIR] [REFS_DIR]
#
# Scoring is skipped when REFS_DIR holds no .json files -- you do not need reference
# annotations to produce predictions, only to measure them.
set -uo pipefail

PY="${PY:-python3}"
DATA="${1:-data}"
REFS="${2:-refs}"

if [ ! -d "$DATA" ]; then echo "no such directory: $DATA" >&2; exit 1; fi

mkdir -p preds viz
rm -f features.csv

n=0; ok=0; failed=()
for d in "$DATA"/*/; do
  [ -d "$d" ] || continue
  c=$(basename "$d")
  img=$(ls "$d"*orig*.nii "$d"*orig*.nii.gz 2>/dev/null | head -1)
  msk=$(ls "$d"*mask*.nii "$d"*mask*.nii.gz 2>/dev/null | head -1)
  if [ -z "$img" ] || [ -z "$msk" ]; then
    echo "[skip] $c: need one *orig*.nii* and one *mask*.nii* (found img='$img' mask='$msk')" >&2
    failed+=("$c: missing files"); n=$((n+1)); continue
  fi
  n=$((n+1))
  # one bad case must not abort the other 24
  if "$PY" run.py --image "$img" --aorta-mask "$msk" \
                  --output "preds/$c.json" --viz "viz/$c.png" --features features.csv; then
    ok=$((ok+1))
  else
    failed+=("$c: run.py failed")
  fi
done

echo ""
echo "=== $ok/$n cases produced predictions -> preds/ , visual checks -> viz/ ==="
if [ ${#failed[@]} -gt 0 ]; then
  printf '  FAILED %s\n' "${failed[@]}"
fi

shopt -s nullglob
refjson=("$REFS"/*.json)
shopt -u nullglob
if [ ${#refjson[@]} -gt 0 ]; then
  echo ""
  "$PY" evaluate.py --pred preds/ --ref "$REFS" --match-mm 3,5,8 --per-case
else
  echo ""
  echo "No reference JSONs in '$REFS/' — skipping scoring."
  echo "Predictions are still written. Inspect them with: open viz/*.png"
fi
