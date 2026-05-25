#!/usr/bin/env bash
# build.sh — Builds the optimised spawn_sim binary.
set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"

echo "Building with $CXX ..."
"$CXX" $CXXFLAGS \
    src/main.cpp \
    src/io.cpp \
    -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
