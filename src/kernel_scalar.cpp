// Phase 9: scalar kernel with three optimisations over Phase 5:
//   (1) KernelContext: rs_store and C_store allocated once, reused every generation.
//   (2) Persistent C0..C4: maintained across rows with one 3-bit sub + one 3-bit add.
//   (3) Fused inner loop: row-sum + C update + emit in one pass.
#include "kernel_scalar.h"
#include <algorithm>

static inline void row_sum_word(uint64_t prev, uint64_t curr, uint64_t next,
                                uint64_t& out2, uint64_t& out1, uint64_t& out0)
{
    const uint64_t a = (curr << 2) | (prev >> 62);
    const uint64_t b = (curr << 1) | (prev >> 63);
    const uint64_t c = curr;
    const uint64_t d = (curr >> 1) | (next << 63);
    const uint64_t e = (curr >> 2) | (next << 62);
    const uint64_t s_abc = a ^ b ^ c;
    const uint64_t c_abc = (a & b) | (c & (a ^ b));
    const uint64_t c_des = (d & e) | (s_abc & (d ^ e));
    out0 = s_abc ^ d ^ e;
    out1 = c_abc ^ c_des;
    out2 = c_abc & c_des;
}

static inline void c5_sub3_scalar(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3, uint64_t& c4,
    uint64_t r0, uint64_t r1, uint64_t r2)
{
    uint64_t b = (~c0) & r0; c0 ^= r0;
    uint64_t ax = c1^r1; uint64_t nb = (~c1&r1)|(b&~ax); c1 = ax^b; b = nb;
    ax = c2^r2; nb = (~c2&r2)|(b&~ax); c2 = ax^b; b = nb;
    nb = b&~c3; c3 ^= b; b = nb;
    c4 ^= b;
}

static inline void c5_add3_scalar(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3, uint64_t& c4,
    uint64_t r0, uint64_t r1, uint64_t r2)
{
    uint64_t carry = c0 & r0; c0 ^= r0;
    uint64_t ax = c1^r1; uint64_t nc = (c1&r1)|(carry&ax); c1 = ax^carry; carry = nc;
    ax = c2^r2; nc = (c2&r2)|(carry&ax); c2 = ax^carry; carry = nc;
    nc = c3 & carry; c3 ^= carry; carry = nc;
    c4 ^= carry;
}

static void fill_ring_slot_scalar(
    const BitplanePair& src, size_t rw,
    size_t ws, size_t we, size_t tw,
    size_t src_row, int slot,
    uint64_t* const* rs2, uint64_t* const* rs1, uint64_t* const* rs0)
{
    const uint64_t* sp1 = src.s1.row(src_row);
    const uint64_t* sp0 = src.s0.row(src_row);
    const uint64_t bnd_prev = sp1[ws == 0 ? rw-1 : ws-1] & sp0[ws == 0 ? rw-1 : ws-1];
    const uint64_t bnd_next = sp1[we == rw ? 0    : we  ] & sp0[we == rw ? 0    : we  ];
    for (size_t w = 0; w < tw; ++w) {
        const uint64_t prev = (w == 0)    ? bnd_prev : sp1[ws+w-1] & sp0[ws+w-1];
        const uint64_t curr = sp1[ws+w]   & sp0[ws+w];
        const uint64_t next = (w == tw-1) ? bnd_next : sp1[ws+w+1] & sp0[ws+w+1];
        row_sum_word(prev, curr, next, rs2[slot][w], rs1[slot][w], rs0[slot][w]);
    }
}

void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t /*width*/, size_t height,
                   size_t row_begin, size_t row_end,
                   size_t tile_cols,
                   ScalarKernelContext& ctx)
{
    const size_t rw = src.s1.row_words;
    const size_t tile_words = tile_cols ? tile_cols / 64 : rw;

    for (size_t ws = 0; ws < rw; ws += tile_words) {
        const size_t we = std::min(ws + tile_words, rw);
        const size_t tw = we - ws;

        ctx.ensure(tw);

        uint64_t* rs2[5], *rs1[5], *rs0[5];
        for (int i = 0; i < 5; ++i) {
            rs2[i] = ctx.rs_store.data() + (size_t)(3*i + 0) * tw;
            rs1[i] = ctx.rs_store.data() + (size_t)(3*i + 1) * tw;
            rs0[i] = ctx.rs_store.data() + (size_t)(3*i + 2) * tw;
        }

        uint64_t* C0 = ctx.C_store.data() + 0 * tw;
        uint64_t* C1 = ctx.C_store.data() + 1 * tw;
        uint64_t* C2 = ctx.C_store.data() + 2 * tw;
        uint64_t* C3 = ctx.C_store.data() + 3 * tw;
        uint64_t* C4 = ctx.C_store.data() + 4 * tw;

        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = delta < 0
                ? (row_begin + height - (size_t)(-delta)) % height
                : (row_begin + (size_t)delta) % height;
            fill_ring_slot_scalar(src, rw, ws, we, tw, sr, delta + 2, rs2, rs1, rs0);
        }

        // Initialise C = sum of all 5 ring slots.
        for (size_t w = 0; w < tw; ++w) {
            uint64_t c0=0, c1=0, c2=0, c3=0, c4=0;
            for (int s = 0; s < 5; ++s)
                c5_add3_scalar(c0, c1, c2, c3, c4, rs0[s][w], rs1[s][w], rs2[s][w]);
            C0[w]=c0; C1[w]=c1; C2[w]=c2; C3[w]=c3; C4[w]=c4;
        }

        int tail = 0;

        for (size_t r = row_begin; r < row_end; ++r) {
            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            uint64_t* d1 = dst.s1.row(r);
            uint64_t* d0 = dst.s0.row(r);

            const size_t new_row = (r + 3) % height;
            const uint64_t* np1 = src.s1.row(new_row);
            const uint64_t* np0 = src.s0.row(new_row);
            const uint64_t bnd_prev_n = np1[ws == 0 ? rw-1 : ws-1] & np0[ws == 0 ? rw-1 : ws-1];
            const uint64_t bnd_next_n = np1[we == rw ? 0    : we  ] & np0[we == rw ? 0    : we  ];

            for (size_t w = 0; w < tw; ++w) {
                // New row-sum for word w.
                const uint64_t n_prev = (w == 0)    ? bnd_prev_n : np1[ws+w-1] & np0[ws+w-1];
                const uint64_t n_curr = np1[ws+w] & np0[ws+w];
                const uint64_t n_next = (w == tw-1) ? bnd_next_n : np1[ws+w+1] & np0[ws+w+1];
                uint64_t new_r2, new_r1, new_r0;
                row_sum_word(n_prev, n_curr, n_next, new_r2, new_r1, new_r0);

                // Load C for current row's emit.
                uint64_t c0=C0[w], c1=C1[w], c2=C2[w], c3=C3[w], c4=C4[w];
                uint64_t e0=c0, e1=c1, e2=c2, e3=c3, e4=c4;

                // Update C: sub leaving row, add entering row.
                c5_sub3_scalar(c0, c1, c2, c3, c4, rs0[tail][w], rs1[tail][w], rs2[tail][w]);
                c5_add3_scalar(c0, c1, c2, c3, c4, new_r0, new_r1, new_r2);
                C0[w]=c0; C1[w]=c1; C2[w]=c2; C3[w]=c3; C4[w]=c4;
                rs0[tail][w]=new_r0; rs1[tail][w]=new_r1; rs2[tail][w]=new_r2;

                // Emit: subtract centre ADULT from e0..e4.
                const uint64_t adult = sp1[ws+w] & sp0[ws+w];
                uint64_t borrow = (~e0) & adult; e0 ^= adult;
                uint64_t b2 = (~e1) & borrow; e1 ^= borrow;
                uint64_t b3 = (~e2) & b2;      e2 ^= b2;
                uint64_t b4 = (~e3) & b3;      e3 ^= b3;
                e4 ^= b4;

                // Predicates (Karnaugh-simplified, same formulas as NEON kernel).
                const uint64_t s1w=sp1[ws+w], s0w=sp0[ws+w];
                const uint64_t nc4=~e4, nc3=~e3, nc1=~e1;
                const uint64_t born     = nc4 & nc3 & (e2^e1) & (nc1|e0);
                const uint64_t survives = nc4 & (e3^e2) & (nc3|nc1);
                // State encode (simplified d0, ns1w eliminated).
                const uint64_t adult_sv = s1w & s0w & survives;
                d1[ws+w] = (s0w^s1w) | adult_sv;
                d0[ws+w] = (~s0w & (s1w|born)) | adult_sv;
            }

            tail = (tail + 1) % 5;
        }
    }
}
