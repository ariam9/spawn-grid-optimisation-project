#pragma once
#include "grid.h"
#include <cstddef>
#include <vector>

// Per-thread context: allocated once at startup, reused across all generations.
// Holds the ring-buffer store and the persistent column-sum store.
struct NeonKernelContext {
    std::vector<uint64_t> rs_store;   // 5 * 3 * tw words
    std::vector<uint64_t> C_store;    // 5 * tw words
    size_t cap_tw = 0;
    void ensure(size_t tw) {
        if (tw <= cap_tw) return;
        rs_store.assign(5 * 3 * tw, 0);
        C_store.assign(5 * tw, 0);
        cap_tw = tw;
    }
};

void kernel_neon(const BitplanePair& src, BitplanePair& dst,
                 size_t width, size_t height,
                 size_t row_begin, size_t row_end,
                 size_t tile_cols,
                 NeonKernelContext& ctx);
