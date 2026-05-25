#pragma once
#include "grid.h"
#include <cstddef>

// Phase 4.5: sorting-network NEON kernel (prototype, not on the perf path).
// Same signature as kernel_neon for drop-in dispatch via --kernel=sort.
void kernel_sort(const BitplanePair& src, BitplanePair& dst,
                 size_t width, size_t height,
                 size_t row_begin, size_t row_end);
