#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Per-thread kernel context: allocated once at startup, reused across all
// generations. Holds the ring-buffer store and the persistent column-sum store.
struct KernelContext {
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
