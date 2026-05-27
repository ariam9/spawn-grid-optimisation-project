#!/usr/bin/env bash
# bench.sh — Run spawn_sim N times, report median/p10/p90/min/max/CV.
# Usage: bench/bench.sh <input.bin> [N=20] [binary=./spawn_sim] [extra flags...]
#   Extra flags are forwarded verbatim to the binary (e.g. --kernel=neon --threads=8).
#   Sampling stops early if CV < 1% after the first 10 valid samples.
set -euo pipefail

INPUT="${1:?Usage: bench.sh <input.bin> [N] [binary] [flags...]}"
N="${2:-20}"
BINARY="${3:-./spawn_sim}"
EXTRA_ARGS=("${@:4}")
WARMUP=3

# ── Pre-flight ─────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────" >&2
echo "Binary : $BINARY ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}" >&2
echo "Input  : $INPUT" >&2
echo "Runs   : up to $N (first $WARMUP discarded; stops early if CV<1% after 10)" >&2

FREE_MIB=$(free -m | awk '/^Mem:/{print $7}')
LOAD=$(uptime | awk -F'average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
echo "Memory : ${FREE_MIB} MiB available  |  Load(1m): ${LOAD}  |  Governor: ${GOV}" >&2

if [ "$FREE_MIB" -lt 2048 ]; then
    echo "WARNING: less than 2 GiB free — background memory pressure may skew results" >&2
fi
if awk "BEGIN{exit !($LOAD > 2.0)}"; then
    echo "WARNING: load average ${LOAD} > 2.0 — consider closing background processes" >&2
fi
echo "──────────────────────────────────────────────" >&2

# ── Run loop ───────────────────────────────────────────────────────────────────
TIMES=()
for ((i=1; i<=N; i++)); do
    MS=$("$BINARY" "$INPUT" /dev/null "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>/dev/null \
         | grep -oP '[0-9]+(\.[0-9]+)?')
    echo "  Run $i/$N : $MS ms"
    TIMES+=("$MS")

    # After warmup, check for early convergence every sample
    if [ "${#TIMES[@]}" -gt "$((WARMUP + 10))" ]; then
        CONVERGED=$(python3 - "${WARMUP}" "${TIMES[@]}" <<'PYEOF'
import sys, math
warmup = int(sys.argv[1])
vals = [float(x) for x in sys.argv[2:]][warmup:]
if len(vals) < 10:
    print("no")
else:
    avg = sum(vals)/len(vals)
    sd = math.sqrt(sum((v-avg)**2 for v in vals)/len(vals))
    print("yes" if sd/avg*100 < 1.0 else "no")
PYEOF
)
        if [ "$CONVERGED" = "yes" ]; then
            echo "  (CV < 1% — stopping early at run $i)"
            break
        fi
    fi
done

# ── Statistics ─────────────────────────────────────────────────────────────────
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
