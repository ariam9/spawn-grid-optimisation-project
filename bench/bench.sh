#!/usr/bin/env bash
# Run spawn_sim against all 5 public test grids and report wall-clock time.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-./spawn_sim}"
SIZE="${SIZE:-32768}"          # <- change this
GENS="${GENS:-10000}"
ARGS="${ARGS:---threads=8}"

echo "Binary : $BIN"
echo "Size   : $SIZE"
echo "Gens   : $GENS"
echo "Args   : $ARGS"
echo "Date   : $(date)"
echo

for GRID in \
    "test_grids/public_1_random_low_${SIZE}.bin" \
    "test_grids/public_2_random_high_${SIZE}.bin" \
    "test_grids/public_3_structured_${SIZE}.bin" \
    "test_grids/public_4_sparse_clusters_${SIZE}.bin" \
    "test_grids/public_5_boundary_stress_${SIZE}.bin"
do
    NAME=$(basename "$GRID" .bin)
    echo "=== $NAME ==="
    "$BIN" "$GRID" /dev/null "$GENS" $ARGS
    echo
done
