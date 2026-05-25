#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Two-bitplane representation of the grid.
// s1 = bit 1 of the cell state, s0 = bit 0.
// ADULT = s1 & s0. States: EMPTY=00, EGG=01, JUVENILE=10, ADULT=11.
// Row stride is width/64 uint64_t words (width must be a multiple of 64).
// Each bitplane is 64-byte aligned.
struct Bitplane {
    uint64_t* data = nullptr;
    size_t    words = 0;   // total uint64_t words in the plane
    size_t    row_words = 0; // words per row = width / 64

    void alloc(size_t width, size_t height) {
        words = (width / 64) * height;
        row_words = width / 64;
        // 64-byte aligned allocation
        if (posix_memalign(reinterpret_cast<void**>(&data), 64,
                           words * sizeof(uint64_t)) != 0)
            data = nullptr;
    }
    void free_data() { std::free(data); data = nullptr; }

    uint64_t* row(size_t r) { return data + r * row_words; }
    const uint64_t* row(size_t r) const { return data + r * row_words; }
};

// A pair of bitplanes (s1 and s0) for one grid buffer.
struct BitplanePair {
    Bitplane s1, s0;

    void alloc(size_t width, size_t height) {
        s1.alloc(width, height);
        s0.alloc(width, height);
    }
    void free_data() { s1.free_data(); s0.free_data(); }
};

static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");
