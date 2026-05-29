#!/usr/bin/env bash
# design_doc/full_benchmark.sh
#
# Sweeps the optimised spawn_sim binary over the full (size, pattern) matrix
# used by the grader. Produces a structured timing table at
# design_doc/full_benchmark_results.txt.
#
# Controls applied per run, matching harness/run.sh:
#   * taskset -c 0-7              (CPU affinity)
#   * setarch $(uname -m) -R      (ASLR disabled)
#   * echo 3 > /proc/sys/vm/drop_caches  (only if root; otherwise warned)
#
# For sizes where a .expected.bin is committed (512, 2048), each run is
# verified bit-exact via cmp. For 8K and 32K (no public expected output)
# the script reports timing only and relies on the unit tests for the
# correctness gate.
#
# Usage:
#   bash design_doc/full_benchmark.sh           # N=5, all sizes
#   N=10 bash design_doc/full_benchmark.sh      # override iterations
#   SIZES="512 2048" bash design_doc/full_benchmark.sh  # subset
#
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="${BIN:-./spawn_sim}"
N="${N:-5}"
SIZES="${SIZES:-512 2048 8192 32768}"
PATTERNS="${PATTERNS:-public_1_random_low public_2_random_high public_3_structured public_4_sparse_clusters public_5_boundary_stress}"
OUT="design_doc/full_benchmark_results.txt"
TMP_OUT="/tmp/spawn_out_$$.bin"
trap 'rm -f "$TMP_OUT"' EXIT

if [[ ! -x "$BIN" ]]; then
    echo "Error: $BIN not found or not executable. Run ./build.sh first." >&2
    exit 1
fi

# Try to drop FS caches (needs root).
DROP_CACHES=false
if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    DROP_CACHES=true
fi

# Capture host info for the header.
GOVERNOR="unknown"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
fi

{
    echo "==============================================================================="
    echo "Monster Spawning Grid: full benchmark sweep"
    echo "==============================================================================="
    echo "Date         : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Host         : $(hostname)"
    echo "OS           : $(uname -srm)"
    echo "Binary       : $BIN"
    echo "Iterations   : $N per (size, pattern)"
    echo "CPU affinity : taskset -c 0-7"
    echo "ASLR         : disabled via setarch -R"
    echo "FS caches    : $([[ $DROP_CACHES == true ]] && echo 'dropped between runs (root)' || echo 'not dropped (no root)')"
    echo "CPU governor : $GOVERNOR"
    echo "Generations  : 10000 (binary default)"
    echo ""
} | tee "$OUT"

# run_once <input.bin>: runs the binary once, echoes the simulation time in ms.
run_once() {
    local input="$1"
    [[ $DROP_CACHES == true ]] && echo 3 > /proc/sys/vm/drop_caches
    local raw
    raw=$(taskset -c 0-7 setarch "$(uname -m)" -R "$BIN" "$input" "$TMP_OUT" 2>&1)
    echo "$raw" | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# Aggregator (Python): given a list of ms values, print median/min/max/cv on one line.
aggregate() {
    python3 - "$@" <<'PYEOF'
import sys, math, statistics
vals = [float(x) for x in sys.argv[1:]]
if not vals:
    print("(no data)")
    sys.exit(0)
n = len(vals)
med = statistics.median(vals)
mn  = min(vals)
mx  = max(vals)
mean = sum(vals)/n
sd  = math.sqrt(sum((v-mean)**2 for v in vals)/n) if n > 1 else 0.0
cv  = (sd/mean*100) if mean > 0 else 0.0
print(f"{med:.3f} {mn:.3f} {mx:.3f} {cv:.2f}")
PYEOF
}

# Iterate. One section per size; one row per (size, pattern).
for SIZE in $SIZES; do
    {
        echo "-------------------------------------------------------------------------------"
        echo "Size: ${SIZE} x ${SIZE}"
        echo "-------------------------------------------------------------------------------"
        printf '%-30s %12s %12s %12s %8s %10s %10s\n' \
               "Pattern" "median(ms)" "min(ms)" "max(ms)" "CV(%)" "GCUPS" "Correct"
    } | tee -a "$OUT"

    for PATTERN in $PATTERNS; do
        INPUT="test_grids/${PATTERN}_${SIZE}.bin"
        EXPECTED="test_grids/${PATTERN}_${SIZE}.expected.bin"

        if [[ ! -f "$INPUT" ]]; then
            printf '%-30s %12s %12s %12s %8s %10s %10s\n' \
                   "$PATTERN" "MISSING" "-" "-" "-" "-" "-" | tee -a "$OUT"
            continue
        fi

        # N runs, capture timings.
        TIMES=()
        for ((i = 1; i <= N; i++)); do
            T=$(run_once "$INPUT")
            if [[ -z "$T" ]]; then
                echo "  ERROR on iter $i for $PATTERN ${SIZE}" >&2
                break
            fi
            TIMES+=("$T")
        done

        if [[ ${#TIMES[@]} -ne $N ]]; then
            printf '%-30s %12s %12s %12s %8s %10s %10s\n' \
                   "$PATTERN" "ERROR" "-" "-" "-" "-" "-" | tee -a "$OUT"
            continue
        fi

        # Correctness check using the LAST run's output (still on disk in $TMP_OUT).
        if [[ -f "$EXPECTED" ]]; then
            if cmp -s "$TMP_OUT" "$EXPECTED"; then
                CORRECT="PASS"
            else
                CORRECT="FAIL"
            fi
        else
            CORRECT="SKIP"
        fi

        # Aggregate stats.
        read MEDIAN MIN MAX CV < <(aggregate "${TIMES[@]}")

        # Compute GCUPS = (size^2 * 10000) / median_ms / 1e6
        GCUPS=$(python3 -c "print(f'{(($SIZE * $SIZE * 10000) / ($MEDIAN / 1000)) / 1e9:.2f}')")

        printf '%-30s %12.3f %12.3f %12.3f %8.2f %10s %10s\n' \
               "$PATTERN" "$MEDIAN" "$MIN" "$MAX" "$CV" "$GCUPS" "$CORRECT" | tee -a "$OUT"
    done

    echo "" | tee -a "$OUT"
done

{
    echo "==============================================================================="
    echo "Sweep complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "==============================================================================="
} | tee -a "$OUT"
