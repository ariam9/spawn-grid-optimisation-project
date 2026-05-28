#!/usr/bin/env bash
# speedtest.sh — median wall-clock over N runs (timing only, no correctness check)
# usage: bash speedtest.sh <input.bin> [threads=8] [runs=5] [extra args...]
set -euo pipefail

IN="${1:?usage: bash speedtest.sh <input.bin> [threads] [runs] [extra...]}"
THREADS="${2:-8}"
RUNS="${3:-5}"
shift $(( $# < 3 ? $# : 3 ))   # remaining args ($@) are passed through

times=()
for i in $(seq 1 "$RUNS"); do
    ms=$(./spawn_sim "$IN" /dev/null --threads="$THREADS" "$@" 2>/dev/null | awk '{print $1}')
    times+=("$ms")
    echo "  run $i: ${ms} ms"
done

printf '%s\n' "${times[@]}" | sort -n | awk '
{ a[NR]=$1 }
END {
    mid = (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2
    printf "  --> min %.1f  median %.1f  max %.1f ms\n", a[1], mid, a[NR]
}'
