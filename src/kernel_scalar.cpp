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
        // Initialise running cumulative count C0..C4 = sum of all 5 ring slots.
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

        int tail = 0;

        for (size_t r = row_begin; r < row_end; ++r) {

            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            uint64_t* d1 = dst.s1.row(r);
            uint64_t* d0 = dst.s0.row(r);

            // Fused: centre-sub into local copy, predicates+emit, sub evicted slot.
            for (size_t w = 0; w < tw; ++w) {
                uint64_t c0=C0[w], c1=C1[w], c2=C2[w], c3=C3[w], c4=C4[w];

                // Local copy of C for centre-sub; c preserved for C-update below.
                uint64_t tc0=c0, tc1=c1, tc2=c2, tc3=c3, tc4=c4;
                const uint64_t s1w = sp1[ws+w], s0w = sp0[ws+w];
                uint64_t borrow = s1w & s0w, diff;
                diff = tc0^borrow; borrow = ~tc0&borrow; tc0=diff;
                diff = tc1^borrow; borrow = ~tc1&borrow; tc1=diff;
                diff = tc2^borrow; borrow = ~tc2&borrow; tc2=diff;
                diff = tc3^borrow; borrow = ~tc3&borrow; tc3=diff;
                tc4 ^= borrow;

                // Predicates (Karnaugh-optimised, same formulas as NEON kernel).
                const uint64_t nc4=~tc4, nc3=~tc3, nc1=~tc1;
                const uint64_t born     = nc4 & nc3 & (tc2^tc1) & (nc1|tc0);
                const uint64_t survives = nc4 & (tc3^tc2) & (nc3|nc1);

                // Next state (optimised d0, ns1w eliminated).
                const uint64_t adult_surv = s1w & s0w & survives;
                d1[ws+w] = (s0w^s1w) | adult_surv;
                d0[ws+w] = (~s0w & (s1w|born)) | adult_surv;

                // Sub evicted slot from running C (before fill_slot overwrites it).
                const uint64_t o0=rs0[tail][w], o1=rs1[tail][w], o2=rs2[tail][w];
                uint64_t bw;
                diff = c0^o0; bw = ~c0&o0; c0=diff;
                {uint64_t c1xr=c1^o1; diff=c1xr^bw; bw=(~c1&o1)|(~c1xr&bw); c1=diff;}
                {uint64_t c2xr=c2^o2; diff=c2xr^bw; bw=(~c2&o2)|(~c2xr&bw); c2=diff;}
                diff = c3^bw; bw = ~c3&bw; c3=diff;
                c4 ^= bw;

                C0[w]=c0; C1[w]=c1; C2[w]=c2; C3[w]=c3; C4[w]=c4;
            }

            // Fill new slot (row r+3) into tail position (overwrites evicted slot).
            fill_slot((r + 3) % height, tail);

            // Add new slot to running C.
            for (size_t w = 0; w < tw; ++w) {
                uint64_t c0=C0[w], c1=C1[w], c2=C2[w], c3=C3[w], c4=C4[w];
                const uint64_t n0=rs0[tail][w], n1=rs1[tail][w], n2=rs2[tail][w];
                uint64_t carry, ns;
                ns = c0^n0; carry = c0&n0; c0=ns;
                {uint64_t c1xr=c1^n1; ns=c1xr^carry; carry=(c1&n1)|(carry&c1xr); c1=ns;}
                {uint64_t c2xr=c2^n2; ns=c2xr^carry; carry=(c2&n2)|(carry&c2xr); c2=ns;}
                ns = c3^carry; carry = c3&carry; c3=ns;
                c4 ^= carry;
                C0[w]=c0; C1[w]=c1; C2[w]=c2; C3[w]=c3; C4[w]=c4;
            }

            tail = (tail + 1) % 5;
        }
    }
}
