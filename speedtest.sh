#!/usr/bin/env bash
# speedtest.sh — median wall-clock over N runs (timing only, no correctness check)
# usage: bash speedtest.sh <input.bin> [kernel=neon] [threads=8] [runs=5] [extra args...]
set -euo pipefail

IN="${1:?usage: bash speedtest.sh <input.bin> [kernel] [threads] [runs] [extra...]}"
KERNEL="${2:-neon}"
THREADS="${3:-8}"
RUNS="${4:-5}"
shift $(( $# < 4 ? $# : 4 ))   # remaining args ($@) are passed through (e.g. --multi-gen=4)

times=()
for i in $(seq 1 "$RUNS"); do
    # program prints "<N.NNN> ms" — take the first field
    ms=$(./spawn_sim "$IN" /dev/null --kernel="$KERNEL" --threads="$THREADS" "$@" 2>/dev/null | awk '{print $1}')
    times+=("$ms")
    echo "  run $i: ${ms} ms"
done

printf '%s\n' "${times[@]}" | sort -n | awk '
{ a[NR]=$1 }
END {
    mid = (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2
    printf "  --> min %.1f  median %.1f  max %.1f ms\n", a[1], mid, a[NR]
}'
