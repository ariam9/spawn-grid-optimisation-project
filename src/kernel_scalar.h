#pragma once
#include "grid.h"
#include <cstddef>

// Phase 5: scalar 64-bit bit-sliced kernel with column tiling.
// tile_cols: column-tile width (must be a multiple of 64). 0 = full width (no tiling).
void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t width, size_t height,
                   size_t row_begin, size_t row_end,
                   size_t tile_cols = 0);
