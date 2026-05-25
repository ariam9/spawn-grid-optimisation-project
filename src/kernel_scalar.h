#pragma once
#include "grid.h"
#include <cstddef>

// Phase 2: scalar 64-bit bit-sliced kernel.
// Simulates one generation: reads from src, writes to dst.
// row_begin/row_end: half-open row range (for threading hook, Phase 7).
void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t width, size_t height,
                   size_t row_begin, size_t row_end);
