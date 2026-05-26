#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
        if (posix_memalign(reinterpret_cast<void**>(&data), 64,
                           words * sizeof(uint64_t)) != 0)
            data = nullptr;
    }
    void zero() { std::memset(data, 0, words * sizeof(uint64_t)); }
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

// LocalBitplanePair: same layout as BitplanePair, allocated with
// height = strip_height + 2*K.  Type alias keeps call-sites simple.
using LocalBitplanePair = BitplanePair;

// Copy a horizontal strip from src_global into dst_local, padding K ghost rows
// above and below with toroidal wrap.
//
// dst_local must be allocated with height = strip_height + 2*K rows.
// Local layout:
//   rows [0, K)                       — top ghosts
//   rows [K, K+strip_height)          — interior copy
//   rows [K+strip_height, 2K+strip_height) — bottom ghosts
inline void copy_ghost_strip(const BitplanePair& src, LocalBitplanePair& dst,
                              size_t strip_row_begin, size_t strip_row_end,
                              size_t K, size_t grid_height)
{
    const size_t rw = src.s1.row_words;
    const size_t strip_height = strip_row_end - strip_row_begin;
    const size_t row_bytes = rw * sizeof(uint64_t);

    // Top ghosts: local row i ← global row (begin - K + i) mod H
    for (size_t i = 0; i < K; ++i) {
        size_t gr = (strip_row_begin + grid_height - K + i) % grid_height;
        std::memcpy(dst.s1.row(i),     src.s1.row(gr), row_bytes);
        std::memcpy(dst.s0.row(i),     src.s0.row(gr), row_bytes);
    }
    // Interior: local row K+i ← global row begin+i
    for (size_t i = 0; i < strip_height; ++i) {
        std::memcpy(dst.s1.row(K + i), src.s1.row(strip_row_begin + i), row_bytes);
        std::memcpy(dst.s0.row(K + i), src.s0.row(strip_row_begin + i), row_bytes);
    }
    // Bottom ghosts: local row K+strip_height+i ← global row (end + i) mod H
    for (size_t i = 0; i < K; ++i) {
        size_t gr = (strip_row_end + i) % grid_height;
        std::memcpy(dst.s1.row(K + strip_height + i), src.s1.row(gr), row_bytes);
        std::memcpy(dst.s0.row(K + strip_height + i), src.s0.row(gr), row_bytes);
    }
}

// Copy the interior rows of src_local back to their positions in dst_global.
// Local row K + (r - strip_row_begin) → global row r.
inline void copy_interior_to_global(const LocalBitplanePair& src, BitplanePair& dst,
                                    size_t strip_row_begin, size_t strip_row_end,
                                    size_t K)
{
    const size_t rw = src.s1.row_words;
    const size_t row_bytes = rw * sizeof(uint64_t);
    for (size_t r = strip_row_begin; r < strip_row_end; ++r) {
        size_t lr = K + (r - strip_row_begin);
        std::memcpy(dst.s1.row(r), src.s1.row(lr), row_bytes);
        std::memcpy(dst.s0.row(r), src.s0.row(lr), row_bytes);
    }
}
