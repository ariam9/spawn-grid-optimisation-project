// Scalar bit-sliced kernel with column tiling + row-sum ring buffer.
// Outer loop over column tiles of width tile_cols (0 = full width, no tiling).
// Each tile keeps its own ring buffer, sized to fit in L1.
// All scratch buffers live in ctx — no per-call heap allocation.
#include "kernel_scalar.h"
#include <algorithm>
#include <cassert>

// Horizontal 5-wide popcount for one row tile.
// adult[0..tw-1] are the ADULT bits for words [ws, ws+tw).
// prev_word / next_word are the boundary words just outside the tile (toroidal).
static void row_sum_5_tile(const uint64_t* adult, size_t tw,
                            uint64_t prev_word, uint64_t next_word,
                            uint64_t* out2, uint64_t* out1, uint64_t* out0)
{
    for (size_t w = 0; w < tw; ++w) {
        const uint64_t prev = (w == 0)    ? prev_word : adult[w - 1];
        const uint64_t curr = adult[w];
        const uint64_t next = (w == tw-1) ? next_word : adult[w + 1];

        const uint64_t a = (curr << 2) | (prev >> 62);
        const uint64_t b = (curr << 1) | (prev >> 63);
        const uint64_t c = curr;
        const uint64_t d = (curr >> 1) | (next << 63);
        const uint64_t e = (curr >> 2) | (next << 62);

        const uint64_t s_abc = a ^ b ^ c;
        const uint64_t c_abc = (a & b) | (c & (a ^ b));
        const uint64_t c_des = (d & e) | (s_abc & (d ^ e));
        out0[w] = s_abc ^ d ^ e;
        out1[w] = c_abc ^ c_des;
        out2[w] = c_abc & c_des;
    }
}

void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t /*width*/, size_t height,
                   size_t row_begin, size_t row_end,
                   KernelContext& ctx,
                   size_t tile_cols)
{
    const size_t rw = src.s1.row_words;
    const size_t tile_words = tile_cols ? tile_cols / 64 : rw;
    assert(ctx.tile_words >= tile_words);

    for (size_t ws = 0; ws < rw; ws += tile_words) {
        const size_t we = std::min(ws + tile_words, rw);
        const size_t tw = we - ws;

        // Slot pointers stride by ctx.tile_words so they're stable across calls.
        uint64_t* rs2[5], *rs1[5], *rs0[5];
        for (int i = 0; i < 5; ++i) {
            rs2[i] = ctx.rs_store + (size_t)(3*i + 0) * ctx.tile_words;
            rs1[i] = ctx.rs_store + (size_t)(3*i + 1) * ctx.tile_words;
            rs0[i] = ctx.rs_store + (size_t)(3*i + 2) * ctx.tile_words;
        }
        uint64_t* const adult_tmp = ctx.adult_tmp;
        uint64_t* C0 = ctx.C_store + 0 * ctx.tile_words;
        uint64_t* C1 = ctx.C_store + 1 * ctx.tile_words;
        uint64_t* C2 = ctx.C_store + 2 * ctx.tile_words;
        uint64_t* C3 = ctx.C_store + 3 * ctx.tile_words;
        uint64_t* C4 = ctx.C_store + 4 * ctx.tile_words;

        auto fill_slot = [&](size_t src_row, int slot) {
            const uint64_t* sp1 = src.s1.row(src_row);
            const uint64_t* sp0 = src.s0.row(src_row);
            for (size_t w = 0; w < tw; ++w)
                adult_tmp[w] = sp1[ws + w] & sp0[ws + w];
            const uint64_t bnd_prev = sp1[ws == 0 ? rw-1 : ws-1] & sp0[ws == 0 ? rw-1 : ws-1];
            const uint64_t bnd_next = sp1[we == rw ? 0    : we  ] & sp0[we == rw ? 0    : we  ];
            row_sum_5_tile(adult_tmp, tw, bnd_prev, bnd_next,
                           rs2[slot], rs1[slot], rs0[slot]);
        };

        // Initialise ring: row-sums for (row_begin-2)..(row_begin+2), toroidal.
        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = delta < 0
                ? (row_begin + height - (size_t)(-delta)) % height
                : (row_begin + (size_t)delta) % height;
            fill_slot(sr, delta + 2);
        }
        int tail = 0;

        for (size_t r = row_begin; r < row_end; ++r) {

            // Stage 2c: vertical sum over all 5 ring slots.
            for (size_t w = 0; w < tw; ++w) {
                uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
                for (int slot = 0; slot < 5; ++slot) {
                    const uint64_t r0 = rs0[slot][w];
                    const uint64_t r1 = rs1[slot][w];
                    const uint64_t r2 = rs2[slot][w];
                    uint64_t carry, ns;
                    ns = c0 ^ r0;         carry = c0 & r0;                  c0 = ns;
                    ns = c1 ^ r1 ^ carry; carry = (c1&r1)|(carry&(c1^r1));  c1 = ns;
                    ns = c2 ^ r2 ^ carry; carry = (c2&r2)|(carry&(c2^r2));  c2 = ns;
                    ns = c3 ^ carry;      carry = c3 & carry;                c3 = ns;
                    c4 ^= carry;
                }
                C0[w]=c0; C1[w]=c1; C2[w]=c2; C3[w]=c3; C4[w]=c4;
            }

            // Stage 2d: subtract centre ADULT bit.
            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            for (size_t w = 0; w < tw; ++w) {
                uint64_t borrow = sp1[ws+w] & sp0[ws+w];
                uint64_t diff;
                diff = C0[w]^borrow; borrow = ~C0[w]&borrow; C0[w]=diff;
                diff = C1[w]^borrow; borrow = ~C1[w]&borrow; C1[w]=diff;
                diff = C2[w]^borrow; borrow = ~C2[w]&borrow; C2[w]=diff;
                diff = C3[w]^borrow; borrow = ~C3[w]&borrow; C3[w]=diff;
                C4[w] ^= borrow;
            }

            // Stages 2e+2f: predicates and next-state.
            uint64_t* d1 = dst.s1.row(r);
            uint64_t* d0 = dst.s0.row(r);
            for (size_t w = 0; w < tw; ++w) {
                const uint64_t c0=C0[w], c1=C1[w], c2=C2[w], c3=C3[w], c4=C4[w];
                const uint64_t s1w=sp1[ws+w], s0w=sp0[ws+w];
                const uint64_t born     = ~c4 & ~c3 & ((c0&c1&~c2)|(~c1&c2));
                const uint64_t survives = ~c4 & ((~c3&c2)|(c3&~c2&~c1));
                d1[ws+w] = (s0w^s1w)|(s0w&s1w&survives);
                d0[ws+w] = (~s1w&~s0w&born)|(s1w&~s0w)|(s1w&s0w&survives);
            }

            fill_slot((r + 3) % height, tail);
            tail = (tail + 1) % 5;
        }
    }
}
