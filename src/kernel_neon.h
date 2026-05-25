#pragma once
#include "grid.h"
#include <cstddef>

// Phase 5: NEON 128-bit bit-sliced kernel with column tiling.
// tile_cols: column-tile width (must be a multiple of 128). 0 = full width (no tiling).
void kernel_neon(const BitplanePair& src, BitplanePair& dst,
                 size_t width, size_t height,
                 size_t row_begin, size_t row_end,
                 size_t tile_cols = 0);
