// Phase 10: NEON kernel. Builds on Phase 9 (KernelContext, persistent C, fused loop)
// with x2 unrolling of the inner vi loop. Adjacent word-accumulators (vi and vi+1)
// are independent; interleaving their carry chains lets the OOO engine overlap the
// two serial dependency chains and hide load-use latency on the new-row adult loads.
#include "kernel_neon.h"
#include <algorithm>
#include <arm_neon.h>

static inline uint64x2_t vnot64(uint64x2_t x)
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

// 3-bit horizontal row-sum for a window of 5 adjacent ADULT columns.
// prev_v/curr_v/next_v are consecutive NEON pairs (2×uint64 each).
static inline void neon_row_sum_3bit(
    uint64x2_t prev_v, uint64x2_t curr_v, uint64x2_t next_v,
    uint64x2_t& out2, uint64x2_t& out1, uint64x2_t& out0)
{
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

    out0 = veorq_u64(veorq_u64(s_abc, d), e);
    out1 = veorq_u64(c_abc, c_des);
    out2 = vandq_u64(c_abc, c_des);
}

// Subtract 3-bit (r2,r1,r0) from 5-bit (c4..c0), ripple borrow.
// Uses vbicq (a & ~b) to avoid explicit vnot.
static inline void c5_sub3_neon(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    uint64x2_t b = vbicq_u64(r0, c0);
    c0 = veorq_u64(c0, r0);
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nb = vorrq_u64(vbicq_u64(r1, c1), vbicq_u64(b, ax));
    c1 = veorq_u64(ax, b);
    b = nb;
    ax = veorq_u64(c2, r2);
    nb = vorrq_u64(vbicq_u64(r2, c2), vbicq_u64(b, ax));
    c2 = veorq_u64(ax, b);
    b = nb;
    nb = vbicq_u64(b, c3);
    c3 = veorq_u64(c3, b);
    b = nb;
    c4 = veorq_u64(c4, b);
}

// Add 3-bit (r2,r1,r0) to 5-bit (c4..c0), ripple carry.
static inline void c5_add3_neon(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    uint64x2_t carry = vandq_u64(c0, r0);
    c0 = veorq_u64(c0, r0);
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nc = vorrq_u64(vandq_u64(c1, r1), vandq_u64(carry, ax));
    c1 = veorq_u64(ax, carry);
    carry = nc;
    ax = veorq_u64(c2, r2);
    nc = vorrq_u64(vandq_u64(c2, r2), vandq_u64(carry, ax));
    c2 = veorq_u64(ax, carry);
    carry = nc;
    nc = vandq_u64(c3, carry);
    c3 = veorq_u64(c3, carry);
    carry = nc;
    c4 = veorq_u64(c4, carry);
}

// Fill one ring slot from a source row.
// pw0/pw1: indices of the two words immediately left of the tile (toroidal).
// nw0/nw1: indices of the two words immediately right of the tile (toroidal).
static void fill_ring_slot_neon(
    const BitplanePair& src,
    size_t ws, size_t tnw,
    size_t pw0, size_t pw1, size_t nw0, size_t nw1,
    size_t src_row, int slot,
    uint64_t* const* rs2, uint64_t* const* rs1, uint64_t* const* rs0)
{
    const uint64_t* sp1 = src.s1.row(src_row);
    const uint64_t* sp0 = src.s0.row(src_row);

    const uint64x2_t bnd_prev = vcombine_u64(
        vcreate_u64(sp1[pw0] & sp0[pw0]), vcreate_u64(sp1[pw1] & sp0[pw1]));
    const uint64x2_t bnd_next = vcombine_u64(
        vcreate_u64(sp1[nw0] & sp0[nw0]), vcreate_u64(sp1[nw1] & sp0[nw1]));

    uint64x2_t adult_prev = bnd_prev;
    uint64x2_t adult_curr = vandq_u64(vld1q_u64(sp1 + ws), vld1q_u64(sp0 + ws));

    for (size_t vi = 0; vi < tnw; ++vi) {
        uint64x2_t adult_next = (vi == tnw - 1)
            ? bnd_next
            : vandq_u64(vld1q_u64(sp1 + ws + vi * 2 + 2),
                        vld1q_u64(sp0 + ws + vi * 2 + 2));

        uint64x2_t r2, r1, r0;
        neon_row_sum_3bit(adult_prev, adult_curr, adult_next, r2, r1, r0);
        vst1q_u64(rs0[slot] + vi * 2, r0);
        vst1q_u64(rs1[slot] + vi * 2, r1);
        vst1q_u64(rs2[slot] + vi * 2, r2);

        adult_prev = adult_curr;
        adult_curr = adult_next;
    }
}

void kernel_neon(const BitplanePair& src, BitplanePair& dst,
                 size_t height,
                 size_t row_begin, size_t row_end,
                 KernelContext& ctx)
{
    const size_t rw         = src.s1.row_words;
    const size_t tile_words = rw;

    for (size_t ws = 0; ws < rw; ws += tile_words) {
        const size_t we  = std::min(ws + tile_words, rw);
        const size_t tw  = we - ws;
        const size_t tnw = tw / 2;

        const size_t pw0 = (ws == 0) ? rw - 2 : ws - 2;
        const size_t pw1 = (ws == 0) ? rw - 1 : ws - 1;
        const size_t nw0 = (we == rw) ? 0 : we;
        const size_t nw1 = (we == rw) ? 1 : we + 1;

        ctx.ensure(tw);

        uint64_t* rs2[5], *rs1[5], *rs0[5];
        for (int i = 0; i < 5; ++i) {
            rs2[i] = ctx.rs_store.data() + (size_t)(3 * i + 0) * tw;
            rs1[i] = ctx.rs_store.data() + (size_t)(3 * i + 1) * tw;
            rs0[i] = ctx.rs_store.data() + (size_t)(3 * i + 2) * tw;
        }

        uint64_t* C0 = ctx.C_store.data() + 0 * tw;
        uint64_t* C1 = ctx.C_store.data() + 1 * tw;
        uint64_t* C2 = ctx.C_store.data() + 2 * tw;
        uint64_t* C3 = ctx.C_store.data() + 3 * tw;
        uint64_t* C4 = ctx.C_store.data() + 4 * tw;

        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = (delta < 0)
                ? (row_begin + height - (size_t)(-delta)) % height
                : (row_begin + (size_t)delta) % height;
            fill_ring_slot_neon(src, ws, tnw, pw0, pw1, nw0, nw1,
                                sr, delta + 2, rs2, rs1, rs0);
        }

        // Initialise C = sum of all 5 ring slots.
        for (size_t vi = 0; vi < tnw; ++vi) {
            uint64x2_t c0 = vdupq_n_u64(0), c1 = vdupq_n_u64(0), c2 = vdupq_n_u64(0);
            uint64x2_t c3 = vdupq_n_u64(0), c4 = vdupq_n_u64(0);
            for (int s = 0; s < 5; ++s)
                c5_add3_neon(c0, c1, c2, c3, c4,
                             vld1q_u64(rs0[s] + vi * 2),
                             vld1q_u64(rs1[s] + vi * 2),
                             vld1q_u64(rs2[s] + vi * 2));
            vst1q_u64(C0 + vi * 2, c0); vst1q_u64(C1 + vi * 2, c1);
            vst1q_u64(C2 + vi * 2, c2); vst1q_u64(C3 + vi * 2, c3);
            vst1q_u64(C4 + vi * 2, c4);
        }

        int tail = 0;

        for (size_t r = row_begin; r < row_end; ++r) {
            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            uint64_t*       d1  = dst.s1.row(r);
            uint64_t*       d0  = dst.s0.row(r);

            const size_t new_row = (r + 3) % height;
            const uint64_t* np1 = src.s1.row(new_row);
            const uint64_t* np0 = src.s0.row(new_row);

            const uint64x2_t bnd_prev = vcombine_u64(
                vcreate_u64(np1[pw0] & np0[pw0]), vcreate_u64(np1[pw1] & np0[pw1]));
            const uint64x2_t bnd_next = vcombine_u64(
                vcreate_u64(np1[nw0] & np0[nw0]), vcreate_u64(np1[nw1] & np0[nw1]));

            uint64x2_t adult_prev = bnd_prev;
            uint64x2_t adult_curr = vandq_u64(vld1q_u64(np1 + ws),
                                              vld1q_u64(np0 + ws));

            // x2 unrolled: process NEON pairs vi and vi+1 together.
            // The two accumulators (c*_0, c*_1) are independent; the OOO engine can
            // overlap their carry chains, hiding the serial sub3+add3 latency.
            // Loop condition is vi+2 < tnw (not <=), so adult_next_1 is NEVER the
            // boundary case here — its load is unconditional. The even-tnw boundary
            // (vi+2 == tnw) is handled by a peeled x2 iteration after the loop;
            // this keeps `bnd_next` off the loop's hot path (it was being
            // speculatively reloaded from [sp] every iteration via the ternary).
            size_t vi = 0;
            for (; vi + 2 < tnw; vi += 2) {
                const uint64x2_t adult_next_0 =
                    vandq_u64(vld1q_u64(np1 + ws + (vi + 1) * 2),
                              vld1q_u64(np0 + ws + (vi + 1) * 2));
                const uint64x2_t adult_next_1 =
                    vandq_u64(vld1q_u64(np1 + ws + (vi + 2) * 2),
                              vld1q_u64(np0 + ws + (vi + 2) * 2));

                uint64x2_t new_r2_0, new_r1_0, new_r0_0;
                uint64x2_t new_r2_1, new_r1_1, new_r0_1;
                neon_row_sum_3bit(adult_prev,   adult_curr,    adult_next_0,
                                  new_r2_0, new_r1_0, new_r0_0);
                neon_row_sum_3bit(adult_curr,   adult_next_0,  adult_next_1,
                                  new_r2_1, new_r1_1, new_r0_1);

                uint64x2_t c0_0 = vld1q_u64(C0 + vi * 2);
                uint64x2_t c1_0 = vld1q_u64(C1 + vi * 2);
                uint64x2_t c2_0 = vld1q_u64(C2 + vi * 2);
                uint64x2_t c3_0 = vld1q_u64(C3 + vi * 2);
                uint64x2_t c4_0 = vld1q_u64(C4 + vi * 2);
                uint64x2_t c0_1 = vld1q_u64(C0 + (vi + 1) * 2);
                uint64x2_t c1_1 = vld1q_u64(C1 + (vi + 1) * 2);
                uint64x2_t c2_1 = vld1q_u64(C2 + (vi + 1) * 2);
                uint64x2_t c3_1 = vld1q_u64(C3 + (vi + 1) * 2);
                uint64x2_t c4_1 = vld1q_u64(C4 + (vi + 1) * 2);

                const uint64x2_t s1w_0 = vld1q_u64(sp1 + ws + vi * 2);
                const uint64x2_t s0w_0 = vld1q_u64(sp0 + ws + vi * 2);
                const uint64x2_t s1w_1 = vld1q_u64(sp1 + ws + (vi+1) * 2);
                const uint64x2_t s0w_1 = vld1q_u64(sp0 + ws + (vi+1) * 2);

                // Emit directly from C (= A_full, INCLUDING the centre cell).
                // born only fires on EMPTY cells (centre=0 -> A=A_full) and
                // survives only on ADULT cells (centre=1 -> A=A_full-1, folded
                // into the range). No centre subtraction => no depth-5 borrow
                // chain and no C snapshot; emit happens before C is rolled.
                //   born     = A_full in {3,4,5}:  ~c4&~c3&(c2^c1)&(~c1|c0)
                //   survives = A_full in {5..10}:  ~c4 & ((~c3&c2&(c1|c0)) | (c3&~c2&(~c1|~c0)))
                {
                    // Predicates on C = A_full; expressed via BIC/ORN/De-Morgan so the
                    // 5 complements ~c4..~c0 never materialise — saves 3 ops/position
                    // and (more importantly) 5 register live-ranges per position. Same
                    // formulas: born = ~c4&~c3&(c2^c1)&(~c1|c0); survives = ~c4 &
                    // ((c2&~c3&(c1|c0)) | (c3&~c2&~(c1&c0))).
                    const uint64x2_t born_hi = vnot64(vorrq_u64(c4_0, c3_0));  // ~c4 & ~c3
                    const uint64x2_t born =
                        vandq_u64(vandq_u64(born_hi, veorq_u64(c2_0, c1_0)),
                                  vornq_u64(c0_0, c1_0));                     // c0 | ~c1
                    const uint64x2_t surv_lo =
                        vandq_u64(vbicq_u64(c2_0, c3_0),  vorrq_u64(c1_0, c0_0));
                    const uint64x2_t surv_hi =
                        vandq_u64(vbicq_u64(c3_0, c2_0),  vnot64(vandq_u64(c1_0, c0_0)));
                    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_0);
                    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_0, s0w_0), survives);
                    vst1q_u64(d1 + ws + vi * 2,
                        vorrq_u64(veorq_u64(s0w_0, s1w_0), adult_sv));
                    vst1q_u64(d0 + ws + vi * 2,
                        vorrq_u64(vandq_u64(vnot64(s0w_0), vorrq_u64(s1w_0, born)), adult_sv));
                }
                {
                    const uint64x2_t born_hi = vnot64(vorrq_u64(c4_1, c3_1));
                    const uint64x2_t born =
                        vandq_u64(vandq_u64(born_hi, veorq_u64(c2_1, c1_1)),
                                  vornq_u64(c0_1, c1_1));
                    const uint64x2_t surv_lo =
                        vandq_u64(vbicq_u64(c2_1, c3_1),  vorrq_u64(c1_1, c0_1));
                    const uint64x2_t surv_hi =
                        vandq_u64(vbicq_u64(c3_1, c2_1),  vnot64(vandq_u64(c1_1, c0_1)));
                    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_1);
                    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_1, s0w_1), survives);
                    vst1q_u64(d1 + ws + (vi+1) * 2,
                        vorrq_u64(veorq_u64(s0w_1, s1w_1), adult_sv));
                    vst1q_u64(d0 + ws + (vi+1) * 2,
                        vorrq_u64(vandq_u64(vnot64(s0w_1), vorrq_u64(s1w_1, born)), adult_sv));
                }

                // Roll C to the next row: subtract the leaving row, add the
                // entering row. C is dead for emit at this point. Two positions
                // interleaved to expose independent carry chains to the OOO engine.
                const uint64x2_t old_r0_0 = vld1q_u64(rs0[tail] + vi * 2);
                const uint64x2_t old_r1_0 = vld1q_u64(rs1[tail] + vi * 2);
                const uint64x2_t old_r2_0 = vld1q_u64(rs2[tail] + vi * 2);
                const uint64x2_t old_r0_1 = vld1q_u64(rs0[tail] + (vi + 1) * 2);
                const uint64x2_t old_r1_1 = vld1q_u64(rs1[tail] + (vi + 1) * 2);
                const uint64x2_t old_r2_1 = vld1q_u64(rs2[tail] + (vi + 1) * 2);

                c5_sub3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, old_r0_0, old_r1_0, old_r2_0);
                c5_sub3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, old_r0_1, old_r1_1, old_r2_1);
                c5_add3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, new_r0_0, new_r1_0, new_r2_0);
                c5_add3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, new_r0_1, new_r1_1, new_r2_1);

                vst1q_u64(C0 + vi * 2, c0_0);       vst1q_u64(C0 + (vi+1)*2, c0_1);
                vst1q_u64(C1 + vi * 2, c1_0);       vst1q_u64(C1 + (vi+1)*2, c1_1);
                vst1q_u64(C2 + vi * 2, c2_0);       vst1q_u64(C2 + (vi+1)*2, c2_1);
                vst1q_u64(C3 + vi * 2, c3_0);       vst1q_u64(C3 + (vi+1)*2, c3_1);
                vst1q_u64(C4 + vi * 2, c4_0);       vst1q_u64(C4 + (vi+1)*2, c4_1);
                vst1q_u64(rs0[tail] + vi * 2,     new_r0_0);
                vst1q_u64(rs0[tail] + (vi+1) * 2, new_r0_1);
                vst1q_u64(rs1[tail] + vi * 2,     new_r1_0);
                vst1q_u64(rs1[tail] + (vi+1) * 2, new_r1_1);
                vst1q_u64(rs2[tail] + vi * 2,     new_r2_0);
                vst1q_u64(rs2[tail] + (vi+1) * 2, new_r2_1);

                adult_prev = adult_next_0;
                adult_curr = adult_next_1;
            }

            // Peeled x2 boundary iteration: runs when tnw is even and >= 2, i.e.
            // exactly one full pair remains and its right-edge wraps to bnd_next.
            // Same body as the main loop above, but adult_next_1 is bnd_next
            // directly (no load, no ternary), so the compiler keeps bnd_next in
            // a register across the row instead of speculatively reloading from
            // [sp] every iteration.
            if (vi + 2 == tnw) {
                const uint64x2_t adult_next_0 =
                    vandq_u64(vld1q_u64(np1 + ws + (vi + 1) * 2),
                              vld1q_u64(np0 + ws + (vi + 1) * 2));
                const uint64x2_t adult_next_1 = bnd_next;

                uint64x2_t new_r2_0, new_r1_0, new_r0_0;
                uint64x2_t new_r2_1, new_r1_1, new_r0_1;
                neon_row_sum_3bit(adult_prev,   adult_curr,    adult_next_0,
                                  new_r2_0, new_r1_0, new_r0_0);
                neon_row_sum_3bit(adult_curr,   adult_next_0,  adult_next_1,
                                  new_r2_1, new_r1_1, new_r0_1);

                uint64x2_t c0_0 = vld1q_u64(C0 + vi * 2);
                uint64x2_t c1_0 = vld1q_u64(C1 + vi * 2);
                uint64x2_t c2_0 = vld1q_u64(C2 + vi * 2);
                uint64x2_t c3_0 = vld1q_u64(C3 + vi * 2);
                uint64x2_t c4_0 = vld1q_u64(C4 + vi * 2);
                uint64x2_t c0_1 = vld1q_u64(C0 + (vi + 1) * 2);
                uint64x2_t c1_1 = vld1q_u64(C1 + (vi + 1) * 2);
                uint64x2_t c2_1 = vld1q_u64(C2 + (vi + 1) * 2);
                uint64x2_t c3_1 = vld1q_u64(C3 + (vi + 1) * 2);
                uint64x2_t c4_1 = vld1q_u64(C4 + (vi + 1) * 2);

                const uint64x2_t s1w_0 = vld1q_u64(sp1 + ws + vi * 2);
                const uint64x2_t s0w_0 = vld1q_u64(sp0 + ws + vi * 2);
                const uint64x2_t s1w_1 = vld1q_u64(sp1 + ws + (vi+1) * 2);
                const uint64x2_t s0w_1 = vld1q_u64(sp0 + ws + (vi+1) * 2);

                {
                    const uint64x2_t born_hi = vnot64(vorrq_u64(c4_0, c3_0));
                    const uint64x2_t born =
                        vandq_u64(vandq_u64(born_hi, veorq_u64(c2_0, c1_0)),
                                  vornq_u64(c0_0, c1_0));
                    const uint64x2_t surv_lo =
                        vandq_u64(vbicq_u64(c2_0, c3_0),  vorrq_u64(c1_0, c0_0));
                    const uint64x2_t surv_hi =
                        vandq_u64(vbicq_u64(c3_0, c2_0),  vnot64(vandq_u64(c1_0, c0_0)));
                    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_0);
                    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_0, s0w_0), survives);
                    vst1q_u64(d1 + ws + vi * 2,
                        vorrq_u64(veorq_u64(s0w_0, s1w_0), adult_sv));
                    vst1q_u64(d0 + ws + vi * 2,
                        vorrq_u64(vandq_u64(vnot64(s0w_0), vorrq_u64(s1w_0, born)), adult_sv));
                }
                {
                    const uint64x2_t born_hi = vnot64(vorrq_u64(c4_1, c3_1));
                    const uint64x2_t born =
                        vandq_u64(vandq_u64(born_hi, veorq_u64(c2_1, c1_1)),
                                  vornq_u64(c0_1, c1_1));
                    const uint64x2_t surv_lo =
                        vandq_u64(vbicq_u64(c2_1, c3_1),  vorrq_u64(c1_1, c0_1));
                    const uint64x2_t surv_hi =
                        vandq_u64(vbicq_u64(c3_1, c2_1),  vnot64(vandq_u64(c1_1, c0_1)));
                    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_1);
                    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_1, s0w_1), survives);
                    vst1q_u64(d1 + ws + (vi+1) * 2,
                        vorrq_u64(veorq_u64(s0w_1, s1w_1), adult_sv));
                    vst1q_u64(d0 + ws + (vi+1) * 2,
                        vorrq_u64(vandq_u64(vnot64(s0w_1), vorrq_u64(s1w_1, born)), adult_sv));
                }

                const uint64x2_t old_r0_0 = vld1q_u64(rs0[tail] + vi * 2);
                const uint64x2_t old_r1_0 = vld1q_u64(rs1[tail] + vi * 2);
                const uint64x2_t old_r2_0 = vld1q_u64(rs2[tail] + vi * 2);
                const uint64x2_t old_r0_1 = vld1q_u64(rs0[tail] + (vi + 1) * 2);
                const uint64x2_t old_r1_1 = vld1q_u64(rs1[tail] + (vi + 1) * 2);
                const uint64x2_t old_r2_1 = vld1q_u64(rs2[tail] + (vi + 1) * 2);

                c5_sub3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, old_r0_0, old_r1_0, old_r2_0);
                c5_sub3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, old_r0_1, old_r1_1, old_r2_1);
                c5_add3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, new_r0_0, new_r1_0, new_r2_0);
                c5_add3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, new_r0_1, new_r1_1, new_r2_1);

                vst1q_u64(C0 + vi * 2, c0_0);       vst1q_u64(C0 + (vi+1)*2, c0_1);
                vst1q_u64(C1 + vi * 2, c1_0);       vst1q_u64(C1 + (vi+1)*2, c1_1);
                vst1q_u64(C2 + vi * 2, c2_0);       vst1q_u64(C2 + (vi+1)*2, c2_1);
                vst1q_u64(C3 + vi * 2, c3_0);       vst1q_u64(C3 + (vi+1)*2, c3_1);
                vst1q_u64(C4 + vi * 2, c4_0);       vst1q_u64(C4 + (vi+1)*2, c4_1);
                vst1q_u64(rs0[tail] + vi * 2,     new_r0_0);
                vst1q_u64(rs0[tail] + (vi+1) * 2, new_r0_1);
                vst1q_u64(rs1[tail] + vi * 2,     new_r1_0);
                vst1q_u64(rs1[tail] + (vi+1) * 2, new_r1_1);
                vst1q_u64(rs2[tail] + vi * 2,     new_r2_0);
                vst1q_u64(rs2[tail] + (vi+1) * 2, new_r2_1);

                vi += 2;
            }

            // Tail: single remaining pair when tnw is odd.
            if (vi < tnw) {
                const uint64x2_t adult_next = bnd_next;

                uint64x2_t new_r2, new_r1, new_r0;
                neon_row_sum_3bit(adult_prev, adult_curr, adult_next,
                                  new_r2, new_r1, new_r0);

                uint64x2_t c0 = vld1q_u64(C0 + vi * 2);
                uint64x2_t c1 = vld1q_u64(C1 + vi * 2);
                uint64x2_t c2 = vld1q_u64(C2 + vi * 2);
                uint64x2_t c3 = vld1q_u64(C3 + vi * 2);
                uint64x2_t c4 = vld1q_u64(C4 + vi * 2);

                const uint64x2_t s1w = vld1q_u64(sp1 + ws + vi * 2);
                const uint64x2_t s0w = vld1q_u64(sp0 + ws + vi * 2);

                // Emit directly from C (= A_full); see unrolled body for rationale.
                {
                    const uint64x2_t born_hi = vnot64(vorrq_u64(c4, c3));
                    const uint64x2_t born =
                        vandq_u64(vandq_u64(born_hi, veorq_u64(c2, c1)),
                                  vornq_u64(c0, c1));
                    const uint64x2_t surv_lo =
                        vandq_u64(vbicq_u64(c2, c3), vorrq_u64(c1, c0));
                    const uint64x2_t surv_hi =
                        vandq_u64(vbicq_u64(c3, c2), vnot64(vandq_u64(c1, c0)));
                    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4);
                    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w, s0w), survives);
                    vst1q_u64(d1 + ws + vi * 2,
                        vorrq_u64(veorq_u64(s0w, s1w), adult_sv));
                    vst1q_u64(d0 + ws + vi * 2,
                        vorrq_u64(vandq_u64(vnot64(s0w), vorrq_u64(s1w, born)), adult_sv));
                }

                const uint64x2_t old_r0 = vld1q_u64(rs0[tail] + vi * 2);
                const uint64x2_t old_r1 = vld1q_u64(rs1[tail] + vi * 2);
                const uint64x2_t old_r2 = vld1q_u64(rs2[tail] + vi * 2);
                c5_sub3_neon(c0, c1, c2, c3, c4, old_r0, old_r1, old_r2);
                c5_add3_neon(c0, c1, c2, c3, c4, new_r0, new_r1, new_r2);

                vst1q_u64(C0 + vi * 2, c0); vst1q_u64(C1 + vi * 2, c1);
                vst1q_u64(C2 + vi * 2, c2); vst1q_u64(C3 + vi * 2, c3);
                vst1q_u64(C4 + vi * 2, c4);
                vst1q_u64(rs0[tail] + vi * 2, new_r0);
                vst1q_u64(rs1[tail] + vi * 2, new_r1);
                vst1q_u64(rs2[tail] + vi * 2, new_r2);
            }

            tail = (tail + 1) % 5;
        }
    }
}
