#pragma once
#include "context.h"
#include "grid.h"
#include <cstddef>

// Scalar 64-bit bit-sliced kernel with column tiling.
// `ctx` must be sized for at least `tile_cols/64` words (or `width/64` if tile_cols == 0).
// tile_cols: column-tile width (must be a multiple of 64). 0 = full width (no tiling).
void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t width, size_t height,
                   size_t row_begin, size_t row_end,
                   KernelContext& ctx,
                   size_t tile_cols = 0);
