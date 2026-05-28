#pragma once
#include "grid.h"
#include "kernel_context.h"
#include <cstddef>

void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t height,
                   size_t row_begin, size_t row_end,
                   KernelContext& ctx);
