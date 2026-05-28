#!/usr/bin/env bash
# Run spawn_sim against all 5 public 32K test grids and report wall-clock time.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-./spawn_sim}"
ARGS="${ARGS:---kernel=neon --threads=8}"

echo "Binary : $BIN"
echo "Args   : $ARGS"
echo "Date   : $(date)"
echo

for GRID in \
    test_grids/public_1_random_low_32768.bin \
    test_grids/public_2_random_high_32768.bin \
    test_grids/public_3_structured_32768.bin \
    test_grids/public_4_sparse_clusters_32768.bin \
    test_grids/public_5_boundary_stress_32768.bin
do
    NAME=$(basename "$GRID" .bin)
    echo "=== $NAME ==="
    "$BIN" "$GRID" /dev/null $ARGS
    echo
done
