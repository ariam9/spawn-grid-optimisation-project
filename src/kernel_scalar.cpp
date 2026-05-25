// Phase 2: scalar bit-sliced kernel.
#include "kernel_scalar.h"
#include <vector>

// Horizontal sum-of-5 ADULT cells across one row.
// For word w: combines columns w*64-2 .. w*64+63+2 (toroidal at row ends).
// Output: 3-bit sum stored in three parallel uint64_t planes (out2, out1, out0).
static void row_sum_5(const uint64_t* adult_row, size_t rw,
                      uint64_t* out2, uint64_t* out1, uint64_t* out0)
{
    for (size_t w = 0; w < rw; ++w) {
        const uint64_t prev = adult_row[w == 0     ? rw - 1 : w - 1];
        const uint64_t curr = adult_row[w];
        const uint64_t next = adult_row[w == rw-1  ? 0      : w + 1];

        // Five column offsets: -2, -1, 0, +1, +2 relative to word w.
        // "Look left k"  = (curr << k) | (prev >> (64-k))
        // "Look right k" = (curr >> k) | (next << (64-k))
        const uint64_t a = (curr << 2) | (prev >> 62);
        const uint64_t b = (curr << 1) | (prev >> 63);
        const uint64_t c = curr;
        const uint64_t d = (curr >> 1) | (next << 63);
        const uint64_t e = (curr >> 2) | (next << 62);

        // 5-input popcount via 2 full-adders + 1 half-adder.
        // FA1: a,b,c -> s_abc, c_abc
        const uint64_t s_abc = a ^ b ^ c;
        const uint64_t c_abc = (a & b) | (c & (a ^ b));
        // FA2: s_abc,d,e -> bit0, c_des
        const uint64_t c_des = (d & e) | (s_abc & (d ^ e));
        // HA: c_abc, c_des -> bit1, bit2
        out0[w] = s_abc ^ d ^ e;
        out1[w] = c_abc ^ c_des;
        out2[w] = c_abc & c_des;
    }
}

void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t /*width*/, size_t height,
                   size_t row_begin, size_t row_end)
{
    const size_t rw = src.s1.row_words;

    // Row-sum ring: 5 slots × 3 planes × rw words.
    std::vector<uint64_t> rs_store(5 * 3 * rw);
    uint64_t* rs2[5], *rs1[5], *rs0[5];
    for (int i = 0; i < 5; ++i) {
        rs2[i] = rs_store.data() + (size_t)(3*i + 0) * rw;
        rs1[i] = rs_store.data() + (size_t)(3*i + 1) * rw;
        rs0[i] = rs_store.data() + (size_t)(3*i + 2) * rw;
    }

    // Per-word adult scratch and 5-bit count accumulator.
    std::vector<uint64_t> adult_tmp(rw);
    std::vector<uint64_t> C_store(5 * rw);
    uint64_t* C0 = C_store.data() + 0 * rw;
    uint64_t* C1 = C_store.data() + 1 * rw;
    uint64_t* C2 = C_store.data() + 2 * rw;
    uint64_t* C3 = C_store.data() + 3 * rw;
    uint64_t* C4 = C_store.data() + 4 * rw;

    for (size_t r = row_begin; r < row_end; ++r) {

        // Stage 2a+2b: horizontal row-sum for 5 source rows (r-2 .. r+2), toroidal.
        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = delta < 0
                ? (r + height - (size_t)(-delta)) % height
                : (r + (size_t)delta) % height;

            const uint64_t* sp1 = src.s1.row(sr);
            const uint64_t* sp0 = src.s0.row(sr);
            for (size_t w = 0; w < rw; ++w)
                adult_tmp[w] = sp1[w] & sp0[w];

            const int slot = delta + 2;
            row_sum_5(adult_tmp.data(), rw, rs2[slot], rs1[slot], rs0[slot]);
        }

        // Stage 2c: vertical sum of five 3-bit row-sums → 5-bit count.
        // Add each 3-bit row-sum into running (c4,c3,c2,c1,c0) via ripple-carry.
        for (size_t w = 0; w < rw; ++w) {
            uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
            for (int slot = 0; slot < 5; ++slot) {
                const uint64_t r0 = rs0[slot][w];
                const uint64_t r1 = rs1[slot][w];
                const uint64_t r2 = rs2[slot][w];
                uint64_t carry;
                uint64_t ns;
                ns = c0 ^ r0;         carry = c0 & r0;            c0 = ns;
                ns = c1 ^ r1 ^ carry; carry = (c1&r1)|(carry&(c1^r1)); c1 = ns;
                ns = c2 ^ r2 ^ carry; carry = (c2&r2)|(carry&(c2^r2)); c2 = ns;
                ns = c3 ^ carry;      carry = c3 & carry;          c3 = ns;
                c4 ^= carry;
            }
            C0[w] = c0; C1[w] = c1; C2[w] = c2; C3[w] = c3; C4[w] = c4;
        }

        // Stage 2d: subtract centre cell's ADULT bit with borrow propagation.
        const uint64_t* sp1 = src.s1.row(r);
        const uint64_t* sp0 = src.s0.row(r);
        for (size_t w = 0; w < rw; ++w) {
            uint64_t borrow = sp1[w] & sp0[w];  // 1 where centre is ADULT
            uint64_t diff;
            diff = C0[w] ^ borrow; borrow = ~C0[w] & borrow; C0[w] = diff;
            diff = C1[w] ^ borrow; borrow = ~C1[w] & borrow; C1[w] = diff;
            diff = C2[w] ^ borrow; borrow = ~C2[w] & borrow; C2[w] = diff;
            diff = C3[w] ^ borrow; borrow = ~C3[w] & borrow; C3[w] = diff;
            C4[w] ^= borrow;
        }

        // Stage 2e+2f: predicates and next-state.
        uint64_t* d1 = dst.s1.row(r);
        uint64_t* d0 = dst.s0.row(r);
        for (size_t w = 0; w < rw; ++w) {
            const uint64_t c0 = C0[w], c1 = C1[w], c2 = C2[w];
            const uint64_t c3 = C3[w], c4 = C4[w];
            const uint64_t s1w = sp1[w], s0w = sp0[w];

            const uint64_t born     = ~c4 & ~c3 & ((c0 & c1 & ~c2) | (~c1 & c2));
            const uint64_t survives = ~c4 & ((~c3 & c2) | (c3 & ~c2 & ~c1));

            d1[w] = (s0w ^ s1w) | (s0w & s1w & survives);
            d0[w] = (~s1w & ~s0w & born) | (s1w & ~s0w) | (s1w & s0w & survives);
        }
    }
}
