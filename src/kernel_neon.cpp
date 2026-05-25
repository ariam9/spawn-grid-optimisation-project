// Phase 4: 128-bit NEON bit-sliced kernel with row-sum ring buffer.
// Ring of 5 cached row-sums: 1 horizontal row-sum per output row instead of 5.
#include "kernel_neon.h"
#include <arm_neon.h>
#include <vector>

static inline uint64x2_t vnot64(uint64x2_t x)
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

static void neon_row_sum_5(const uint64_t* adult_row, size_t rw,
                           uint64_t* out2, uint64_t* out1, uint64_t* out0)
{
    const size_t nw = rw / 2;

    for (size_t vi = 0; vi < nw; ++vi) {
        uint64x2_t prev_v = vld1q_u64(adult_row + (vi == 0    ? rw - 2 : vi*2 - 2));
        uint64x2_t curr_v = vld1q_u64(adult_row + vi * 2);
        uint64x2_t next_v = vld1q_u64(adult_row + (vi == nw-1 ? 0      : vi*2 + 2));

        const uint64x2_t prev_adj = vextq_u64(prev_v, curr_v, 1);
        const uint64x2_t next_adj = vextq_u64(curr_v, next_v, 1);

        const uint64x2_t a = vorrq_u64(vshlq_n_u64(curr_v, 2), vshrq_n_u64(prev_adj, 62));
        const uint64x2_t b = vorrq_u64(vshlq_n_u64(curr_v, 1), vshrq_n_u64(prev_adj, 63));
        const uint64x2_t c = curr_v;
        const uint64x2_t d = vorrq_u64(vshrq_n_u64(curr_v, 1), vshlq_n_u64(next_adj, 63));
        const uint64x2_t e = vorrq_u64(vshrq_n_u64(curr_v, 2), vshlq_n_u64(next_adj, 62));

        const uint64x2_t axb   = veorq_u64(a, b);
        const uint64x2_t s_abc = veorq_u64(axb, c);
        const uint64x2_t c_abc = vorrq_u64(vandq_u64(a, b), vandq_u64(c, axb));
        const uint64x2_t dxe   = veorq_u64(d, e);
        const uint64x2_t c_des = vorrq_u64(vandq_u64(d, e), vandq_u64(s_abc, dxe));

        vst1q_u64(out0 + vi*2, veorq_u64(veorq_u64(s_abc, d), e));
        vst1q_u64(out1 + vi*2, veorq_u64(c_abc, c_des));
        vst1q_u64(out2 + vi*2, vandq_u64(c_abc, c_des));
    }
}

void kernel_neon(const BitplanePair& src, BitplanePair& dst,
                 size_t /*width*/, size_t height,
                 size_t row_begin, size_t row_end)
{
    const size_t rw = src.s1.row_words;
    const size_t nw = rw / 2;

    // Ring buffer: 5 slots × 3-bit row-sum planes × rw words.
    // slot[(tail + delta + 2) % 5] holds row-sum for row (r + delta), delta in -2..+2.
    std::vector<uint64_t> rs_store(5 * 3 * rw);
    uint64_t* rs2[5], *rs1[5], *rs0[5];
    for (int i = 0; i < 5; ++i) {
        rs2[i] = rs_store.data() + (size_t)(3*i + 0) * rw;
        rs1[i] = rs_store.data() + (size_t)(3*i + 1) * rw;
        rs0[i] = rs_store.data() + (size_t)(3*i + 2) * rw;
    }

    std::vector<uint64_t> adult_tmp(rw);

    auto fill_slot = [&](size_t src_row, int slot) {
        const uint64_t* sp1 = src.s1.row(src_row);
        const uint64_t* sp0 = src.s0.row(src_row);
        for (size_t vi = 0; vi < nw; ++vi)
            vst1q_u64(adult_tmp.data() + vi*2,
                      vandq_u64(vld1q_u64(sp1 + vi*2), vld1q_u64(sp0 + vi*2)));
        neon_row_sum_5(adult_tmp.data(), rw, rs2[slot], rs1[slot], rs0[slot]);
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

        // Stages 2c+2d+2e+2f fused: vertical sum (all 5 ring slots) → subtract →
        // predicates → next-state. c0..c4 stay in NEON registers throughout.
        const uint64_t* sp1 = src.s1.row(r);
        const uint64_t* sp0 = src.s0.row(r);
        uint64_t* d1 = dst.s1.row(r);
        uint64_t* d0 = dst.s0.row(r);

        for (size_t vi = 0; vi < nw; ++vi) {
            const uint64x2_t s1w = vld1q_u64(sp1 + vi*2);
            const uint64x2_t s0w = vld1q_u64(sp0 + vi*2);

            // Stage 2c: ripple-carry accumulation over all 5 ring slots.
            uint64x2_t c0 = vdupq_n_u64(0), c1 = vdupq_n_u64(0), c2 = vdupq_n_u64(0);
            uint64x2_t c3 = vdupq_n_u64(0), c4 = vdupq_n_u64(0);
            for (int slot = 0; slot < 5; ++slot) {
                const uint64x2_t r0 = vld1q_u64(rs0[slot] + vi*2);
                const uint64x2_t r1 = vld1q_u64(rs1[slot] + vi*2);
                const uint64x2_t r2 = vld1q_u64(rs2[slot] + vi*2);
                uint64x2_t carry;

                const uint64x2_t ns0 = veorq_u64(c0, r0);
                carry = vandq_u64(c0, r0); c0 = ns0;

                const uint64x2_t c1xr1 = veorq_u64(c1, r1);
                const uint64x2_t ns1 = veorq_u64(c1xr1, carry);
                carry = vorrq_u64(vandq_u64(c1, r1), vandq_u64(carry, c1xr1)); c1 = ns1;

                const uint64x2_t c2xr2 = veorq_u64(c2, r2);
                const uint64x2_t ns2 = veorq_u64(c2xr2, carry);
                carry = vorrq_u64(vandq_u64(c2, r2), vandq_u64(carry, c2xr2)); c2 = ns2;

                const uint64x2_t ns3 = veorq_u64(c3, carry);
                carry = vandq_u64(c3, carry); c3 = ns3;
                c4 = veorq_u64(c4, carry);
            }

            // Stage 2d: subtract centre ADULT bit.
            uint64x2_t borrow = vandq_u64(s1w, s0w);
            uint64x2_t diff;
            diff = veorq_u64(c0, borrow); borrow = vandq_u64(vnot64(c0), borrow); c0 = diff;
            diff = veorq_u64(c1, borrow); borrow = vandq_u64(vnot64(c1), borrow); c1 = diff;
            diff = veorq_u64(c2, borrow); borrow = vandq_u64(vnot64(c2), borrow); c2 = diff;
            diff = veorq_u64(c3, borrow); borrow = vandq_u64(vnot64(c3), borrow); c3 = diff;
            c4 = veorq_u64(c4, borrow);

            // Stage 2e: predicates.
            const uint64x2_t nc4 = vnot64(c4), nc3 = vnot64(c3);
            const uint64x2_t nc2 = vnot64(c2), nc1 = vnot64(c1);

            const uint64x2_t born =
                vandq_u64(nc4, vandq_u64(nc3,
                    vorrq_u64(vandq_u64(vandq_u64(c0, c1), nc2),
                              vandq_u64(nc1, c2))));

            const uint64x2_t survives =
                vandq_u64(nc4,
                    vorrq_u64(vandq_u64(nc3, c2),
                              vandq_u64(vandq_u64(c3, nc2), nc1)));

            // Stage 2f: next-state.
            const uint64x2_t ns1w  = vnot64(s1w);
            const uint64x2_t ns0w  = vnot64(s0w);
            const uint64x2_t adult = vandq_u64(s1w, s0w);

            vst1q_u64(d1 + vi*2,
                vorrq_u64(veorq_u64(s0w, s1w), vandq_u64(adult, survives)));

            vst1q_u64(d0 + vi*2,
                vorrq_u64(
                    vorrq_u64(vandq_u64(vandq_u64(ns1w, ns0w), born),
                              vandq_u64(s1w, ns0w)),
                    vandq_u64(adult, survives)));
        }

        // Advance ring: overwrite oldest slot (tail) with row-sum for (r+3), then rotate.
        fill_slot((r + 3) % height, tail);
        tail = (tail + 1) % 5;
    }
}
