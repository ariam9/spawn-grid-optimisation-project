#!/usr/bin/env bash
# build.sh — Builds spawn_sim and test binaries.
set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"

echo "Building spawn_sim ..."
"$CXX" $CXXFLAGS \
    src/main.cpp \
    src/io.cpp \
    src/context.cpp \
    src/transpose.cpp \
    src/kernel_scalar.cpp \
    src/kernel_neon.cpp \
    -o "$OUTPUT" -lpthread

echo "Building test_transpose ..."
"$CXX" $CXXFLAGS \
    src/transpose.cpp \
    tests/test_transpose.cpp \
    -o tests/test_transpose

echo "Building test_kernel_scalar ..."
"$CXX" $CXXFLAGS \
    src/transpose.cpp \
    src/context.cpp \
    src/kernel_scalar.cpp \
    tests/test_kernel_scalar.cpp \
    -o tests/test_kernel_scalar

echo "Building test_kernel_neon ..."
"$CXX" $CXXFLAGS \
    src/transpose.cpp \
    src/context.cpp \
    src/kernel_scalar.cpp \
    src/kernel_neon.cpp \
    tests/test_kernel_neon.cpp \
    -o tests/test_kernel_neon

echo "Done: $OUTPUT, tests/test_transpose, tests/test_kernel_scalar, tests/test_kernel_neon"
