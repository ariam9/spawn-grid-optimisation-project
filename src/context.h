#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Per-thread scratch buffers reused across all kernel invocations.
// `tile_words` is the maximum tile width (in 64-bit words) the context will
// be used with — sized for that, then reused for every generation.
// Buffers are 64-byte aligned so NEON loads remain aligned.
struct KernelContext {
    uint64_t* rs_store  = nullptr; // 5 slots * 3 bitplanes * tile_words
    uint64_t* C_store   = nullptr; // 5 bitplanes * tile_words  (running cumulative count)
    uint64_t* adult_tmp = nullptr; //            tile_words
    size_t tile_words = 0;

    void alloc(size_t tw);
    void free_data();
};
