#include "transpose.h"
#include <cassert>
#include <cstring>

// Bit layout: for cell at column c in a row,
//   word index = c / 64,  bit position = c % 64  (LSB = column 0 of each 64-wide group)
//
// bytes_to_bitplanes: for each input byte b,
//   s1 gets bit 1:  (b >> 1) & 1
//   s0 gets bit 0:  b & 1
//
// We process 64 columns at a time to avoid per-cell OR-into-word overhead.

void bytes_to_bitplanes(const std::vector<uint8_t>& src,
                        BitplanePair& dst,
                        size_t width, size_t height)
{
    assert(width % 64 == 0);
    assert(dst.s1.data && dst.s0.data);
    dst.s1.zero();
    dst.s0.zero();

    for (size_t r = 0; r < height; ++r) {
        const uint8_t* src_row = src.data() + r * width;
        uint64_t* s1_row = dst.s1.row(r);
        uint64_t* s0_row = dst.s0.row(r);

        // Build one 64-bit word at a time: 64 consecutive cells per word.
        size_t nw = width / 64;
        for (size_t w = 0; w < nw; ++w) {
            const uint8_t* p = src_row + w * 64;
            uint64_t w1 = 0, w0 = 0;
            for (int b = 0; b < 64; ++b) {
                uint8_t cell = p[b];
                w1 |= (uint64_t)((cell >> 1) & 1) << b;
                w0 |= (uint64_t)(cell & 1)        << b;
            }
            s1_row[w] = w1;
            s0_row[w] = w0;
        }
    }
}

void bitplanes_to_bytes(const BitplanePair& src,
                        std::vector<uint8_t>& dst,
                        size_t width, size_t height)
{
    assert(width % 64 == 0);
    assert(src.s1.data && src.s0.data);
    dst.resize(width * height);

    for (size_t r = 0; r < height; ++r) {
        const uint64_t* s1_row = src.s1.row(r);
        const uint64_t* s0_row = src.s0.row(r);
        uint8_t* dst_row = dst.data() + r * width;

        size_t nw = width / 64;
        for (size_t w = 0; w < nw; ++w) {
            uint64_t w1 = s1_row[w];
            uint64_t w0 = s0_row[w];
            uint8_t* p = dst_row + w * 64;
            for (int b = 0; b < 64; ++b) {
                p[b] = (uint8_t)(((w1 >> b) & 1) << 1 | ((w0 >> b) & 1));
            }
        }
    }
}
