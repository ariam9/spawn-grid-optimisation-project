#pragma once
#include "grid.h"
#include <cstddef>

// Phase 3: NEON 128-bit bit-sliced kernel.
void kernel_neon(const BitplanePair& src, BitplanePair& dst,
                 size_t width, size_t height,
                 size_t row_begin, size_t row_end);
