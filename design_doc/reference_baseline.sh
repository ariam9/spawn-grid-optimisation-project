#!/usr/bin/env bash
# design_doc/reference_baseline.sh
#
# Builds the reference (naive per-cell) implementation if needed, then times
# it on public_1_random_low_512.bin to produce the headline reference number
# for the design document's speedup figure.
#
# Reference timings at 2048+ are infeasible (the 32K reference takes about
# 25 hours per run per test_grids/README.md). 512 is the largest size at
# which a 5-run sample completes in a few minutes.
#
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-5}"
REF_BIN="${REF_BIN:-./reference/spawn_sim_ref}"
OUT="design_doc/reference_baseline.txt"
TMP_OUT="/tmp/ref_out_$$.bin"
trap 'rm -f "$TMP_OUT"' EXIT

# Build the reference with a non-default name so it doesn't clobber the
# optimised spawn_sim binary at the repo root.
if [[ ! -x "$REF_BIN" ]]; then
    echo "Building reference -> $REF_BIN ..."
    OUTPUT="$REF_BIN" bash reference/build.sh
fi

INPUT="test_grids/public_1_random_low_512.bin"
EXPECTED="test_grids/public_1_random_low_512.expected.bin"

GOVERNOR="unknown"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
fi

DROP_CACHES=false
if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    DROP_CACHES=true
fi

{
    echo "==============================================================================="
    echo "Reference (naive per-cell) baseline"
    echo "==============================================================================="
    echo "Date         : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Host         : $(hostname)"
    echo "Binary       : $REF_BIN"
    echo "Input        : $INPUT"
    echo "Iterations   : $N"
    echo "CPU affinity : taskset -c 0-7"
    echo "ASLR         : disabled via setarch -R"
    echo "FS caches    : $([[ $DROP_CACHES == true ]] && echo 'dropped between runs (root)' || echo 'not dropped (no root)')"
    echo "CPU governor : $GOVERNOR"
    echo "Generations  : 10000 (binary default)"
    echo ""
} | tee "$OUT"

TIMES=()
for ((i = 1; i <= N; i++)); do
    [[ $DROP_CACHES == true ]] && echo 3 > /proc/sys/vm/drop_caches
    RAW=$(taskset -c 0-7 setarch "$(uname -m)" -R "$REF_BIN" "$INPUT" "$TMP_OUT" 2>&1)
    T=$(echo "$RAW" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    TIMES+=("$T")
    printf "  Run %d/%d : %s ms\n" "$i" "$N" "$T" | tee -a "$OUT"
done

# Correctness check.
if cmp -s "$TMP_OUT" "$EXPECTED"; then
    CORRECT="PASS (bit-exact vs $(basename "$EXPECTED"))"
else
    CORRECT="FAIL (last run's output differs from expected)"
fi

# Aggregate.
{
    echo ""
    echo "Correctness  : $CORRECT"
    echo ""
    python3 - "${TIMES[@]}" <<'PYEOF'
import sys, math, statistics
vals = [float(x) for x in sys.argv[1:]]
n = len(vals)
med = statistics.median(vals)
mn, mx = min(vals), max(vals)
mean = sum(vals)/n
sd = math.sqrt(sum((v-mean)**2 for v in vals)/n) if n > 1 else 0.0
cv = (sd/mean*100) if mean > 0 else 0.0
print(f"Statistics:")
print(f"  n          : {n}")
print(f"  median     : {med:.3f} ms")
print(f"  min        : {mn:.3f} ms")
print(f"  max        : {mx:.3f} ms")
print(f"  mean       : {mean:.3f} ms")
print(f"  std dev    : {sd:.3f} ms")
print(f"  CV         : {cv:.2f}%")
PYEOF
    echo ""
    echo "==============================================================================="
} | tee -a "$OUT"
