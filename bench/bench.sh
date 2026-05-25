#!/usr/bin/env bash
# bench.sh — Run spawn_sim N times, report median/p10/p90/min/max.
# Usage: bench/bench.sh <input.bin> [N=20] [binary=./spawn_sim]
set -euo pipefail

INPUT="${1:?Usage: bench.sh <input.bin> [N] [binary]}"
N="${2:-20}"
BINARY="${3:-./spawn_sim}"
WARMUP=3

echo "Binary : $BINARY"
echo "Input  : $INPUT"
echo "Runs   : $N (first $WARMUP discarded as warmup)"

sudo cpupower frequency-set -g performance 2>/dev/null \
    || echo "(cpupower not available — governor unchanged)"

TIMES=()
for ((i=1; i<=N; i++)); do
    MS=$("$BINARY" "$INPUT" /dev/null 2>/dev/null | grep -oP '[0-9]+(\.[0-9]+)?')
    echo "  Run $i/$N : $MS ms"
    TIMES+=("$MS")
done

python3 - "${WARMUP}" "${TIMES[@]}" <<'EOF'
import sys, math
warmup = int(sys.argv[1])
vals = [float(x) for x in sys.argv[2:]]
vals = vals[warmup:]   # discard warmup
vals.sort()
n = len(vals)
median = vals[n//2] if n%2==1 else (vals[n//2-1]+vals[n//2])/2
p10 = vals[int(n*0.10)]
p90 = vals[int(n*0.90)]
mn, mx = vals[0], vals[-1]
avg = sum(vals)/n
sd = math.sqrt(sum((v-avg)**2 for v in vals)/n)
print(f"\nAfter discarding {warmup} warmup runs ({n} samples):")
print(f"  median : {median:.3f} ms")
print(f"  p10    : {p10:.3f} ms")
print(f"  p90    : {p90:.3f} ms")
print(f"  min    : {mn:.3f} ms")
print(f"  max    : {mx:.3f} ms")
print(f"  cv     : {sd/avg*100:.1f}% (sd/mean)")
EOF
