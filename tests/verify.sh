#!/bin/bash
# usage: verify.sh <input.bin> <expected.bin>
# Runs $BINARY (default ./spawn_sim) on input, compares output to expected.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${BINARY:-$SCRIPT_DIR/../spawn_sim}"
"$BINARY" "$1" /tmp/out.bin
cmp /tmp/out.bin "$2" && echo PASS || echo FAIL
