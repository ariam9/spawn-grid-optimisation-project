#!/usr/bin/env bash
# profile_p15.sh — Profile the NEON kernel with perf.
#
# Run on an otherwise-idle box. The program pins all 8 cores; any extra load
# oversubscribes and skews counters/timing. Close background processes before
# invoking.
#
# Produces, in the repo root:
#   perf_p15.data  — perf record (cycles, frame-pointer call graph)
#   ann_p15.txt    — perf annotate --stdio dump
#   stat_p15.txt   — perf stat (two tight <=4-event groups, repeated 3x)
#
# Auto-builds ./spawn_sim_prof if missing (same flags as build.sh, plus -g and
# -fno-omit-frame-pointer for frame-pointer call graphs).
#
# Usage: bench/profile_p15.sh [input.bin] [generations]
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${1:-test_grids/public_1_random_low_32768.bin}"
GENS="${2:-500}"
PROF=./spawn_sim_prof
ARGS=(--threads=8)
OUT=/tmp/o_prof.bin

CXX="${CXX:-g++-14}"
CXXFLAGS_PROF=(-std=c++23 -O3 -mcpu=neoverse-v2+sha3
               -fno-omit-frame-pointer -g -Wall -Wextra)

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────" >&2
LOAD=$(uptime | awk -F'average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
echo "Binary : $PROF ${ARGS[*]}" >&2
echo "Input  : $INPUT   Gens: $GENS" >&2
echo "Load(1m): ${LOAD}   Governor: ${GOV}   perf_event_paranoid: ${PARANOID}" >&2
if awk "BEGIN{exit !($LOAD > 0.5)}"; then
    echo "WARNING: load ${LOAD} > 0.5 — close background processes (incl. Claude Code) before profiling" >&2
fi

# ── Auto-build profiling binary if missing or stale ──────────────────────────
KERNEL_SRC=src/kernel_neon.cpp
if [ ! -x "$PROF" ] || [ "$KERNEL_SRC" -nt "$PROF" ]; then
    echo ">> Building $PROF (kernel changed or binary missing)" >&2
    "$CXX" "${CXXFLAGS_PROF[@]}" \
        src/main.cpp src/io.cpp src/transpose.cpp src/kernel_neon.cpp \
        -o "$PROF" -lpthread
fi
echo "──────────────────────────────────────────────" >&2

# ── 1. Hotspot record ────────────────────────────────────────────────────────
echo ">> [1/3] perf record -> perf_p15.data" >&2
perf record -o perf_p15.data -e cycles -g --call-graph fp -- \
    "$PROF" "$INPUT" "$OUT" "$GENS" "${ARGS[@]}"

# ── 2. Annotate dump ─────────────────────────────────────────────────────────
echo ">> [2/3] perf annotate -> ann_p15.txt" >&2
perf annotate -i perf_p15.data --stdio -l > ann_p15.txt 2>&1

# ── 3. perf stat (tight groups, repeated 3x; ratios are what matter) ─────────
echo ">> [3/3] perf stat -> stat_p15.txt" >&2
{
    echo "### perf stat — Phase 15B — $INPUT — $GENS gens — $(date -Iseconds)"
    for run in 1 2 3; do
        echo ""
        echo "===== run $run: GROUP A (compute) ====="
        perf stat -e cycles,instructions,stall_backend,stall_backend_mem \
            "$PROF" "$INPUT" "$OUT" "$GENS" "${ARGS[@]}" 2>&1
        echo ""
        echo "===== run $run: GROUP B (memory) ====="
        perf stat -e cycles,l1d_cache,l1d_cache_refill,l2d_cache_refill,dTLB-load-misses \
            "$PROF" "$INPUT" "$OUT" "$GENS" "${ARGS[@]}" 2>&1
    done
} > stat_p15.txt 2>&1

echo "──────────────────────────────────────────────" >&2
echo "Done. Outputs: perf_p15.data  ann_p15.txt  stat_p15.txt" >&2
