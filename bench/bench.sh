#!/usr/bin/env bash
# Run spawn_sim against all 5 public test grids and report wall-clock time.
# When ARGS is unset, picks --threads from SIZE: <=256 -> 1, else 8.
# (The binary picks the kernel itself: scalar for width <=256, neon otherwise.)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-./spawn_sim}"
SIZE="${SIZE:-32768}"
GENS="${GENS:-10000}"

if [[ -z "${ARGS+x}" ]]; then
    if (( SIZE <= 256 )); then THREADS=1; else THREADS=8; fi
    ARGS="--threads=$THREADS"
fi

if (( SIZE <= 256 )); then KERNEL="scalar"; else KERNEL="neon"; fi

echo "Binary : $BIN"
echo "Size   : $SIZE  (kernel=$KERNEL)"
echo "Gens   : $GENS"
echo "Args   : $ARGS"
echo "Date   : $(date)"
echo

# Hint if no grids exist at the requested size (anything <512 needs generating).
if ! ls test_grids/public_1_random_low_"${SIZE}".bin >/dev/null 2>&1; then
    echo "No test grids at size $SIZE. Generate with:"
    echo "  python3 test_grids/gen_no_numpy.py $SIZE"
    echo
fi

for GRID in \
    "test_grids/public_1_random_low_${SIZE}.bin" \
    "test_grids/public_2_random_high_${SIZE}.bin" \
    "test_grids/public_3_structured_${SIZE}.bin" \
    "test_grids/public_4_sparse_clusters_${SIZE}.bin" \
    "test_grids/public_5_boundary_stress_${SIZE}.bin"
do
    NAME=$(basename "$GRID" .bin)
    echo "=== $NAME ==="
    if [[ ! -f "$GRID" ]]; then
        echo "skip (missing)"
        echo
        continue
    fi
    "$BIN" "$GRID" /dev/null "$GENS" $ARGS
    echo
done
