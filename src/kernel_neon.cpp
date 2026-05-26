// Phase 9: NEON kernel with three optimisations over Phase 5:
//   (1) KernelContext: rs_store and C_store allocated once, reused every generation.
//   (2) Persistent C0..C4: maintained across rows with one 3-bit sub + one 3-bit add
//       per row instead of a 5-slot ripple rebuild (~25 ops vs ~50).
//   (3) Fused inner loop: emit + new-row-sum + C update in one vi pass; new row's
//       ADULT bits carried in sliding NEON registers, never touch memory.
#include "kernel_neon.h"
#include <algorithm>
#include <arm_neon.h>

static inline uint64x2_t vnot64(uint64x2_t x)
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

// Inline 3-bit row-sum from three consecutive ADULT pairs (prev, curr, next).
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
static inline void c5_sub3_clean(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    // bit 0
    uint64x2_t b = vbicq_u64(r0, c0);
    c0 = veorq_u64(c0, r0);
    // bit 1
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nb = vorrq_u64(vbicq_u64(r1, c1), vbicq_u64(b, ax));
    c1 = veorq_u64(ax, b);
    b = nb;
    // bit 2
    ax = veorq_u64(c2, r2);
    nb = vorrq_u64(vbicq_u64(r2, c2), vbicq_u64(b, ax));
    c2 = veorq_u64(ax, b);
    b = nb;
    // bit 3 (r3=0)
    nb = vbicq_u64(b, c3);
    c3 = veorq_u64(c3, b);
    b = nb;
    // bit 4 (r4=0)
    c4 = veorq_u64(c4, b);
}

// Add 3-bit (r2,r1,r0) to 5-bit (c4..c0), ripple carry.
static inline void c5_add3(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    // bit 0
    uint64x2_t carry = vandq_u64(c0, r0);
    c0 = veorq_u64(c0, r0);
    // bit 1
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nc = vorrq_u64(vandq_u64(c1, r1), vandq_u64(carry, ax));
    c1 = veorq_u64(ax, carry);
    carry = nc;
    // bit 2
    ax = veorq_u64(c2, r2);
    nc = vorrq_u64(vandq_u64(c2, r2), vandq_u64(carry, ax));
    c2 = veorq_u64(ax, carry);
    carry = nc;
    // bit 3 (r3=0)
    nc = vandq_u64(c3, carry);
    c3 = veorq_u64(c3, carry);
    carry = nc;
    // bit 4 (r4=0)
    c4 = veorq_u64(c4, carry);
}

// Fill one ring slot from a source row, computing adult bits inline (no temp buffer).
static void fill_ring_slot(
    const BitplanePair& src, size_t rw,
    size_t ws, size_t we, size_t tnw,
    size_t src_row, int slot,
    uint64_t* const* rs2, uint64_t* const* rs1, uint64_t* const* rs0)
{
    const uint64_t* sp1 = src.s1.row(src_row);
    const uint64_t* sp0 = src.s0.row(src_row);

    const size_t pw0 = (ws == 0) ? rw - 2 : ws - 2;
    const size_t pw1 = (ws == 0) ? rw - 1 : ws - 1;
    const size_t nw0 = (we == rw) ? 0 : we;
    const size_t nw1 = (we == rw) ? 1 : we + 1;

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
                 size_t /*width*/, size_t height,
                 size_t row_begin, size_t row_end,
                 size_t tile_cols,
                 NeonKernelContext& ctx)
{
    const size_t rw        = src.s1.row_words;
    const size_t tile_words = tile_cols ? tile_cols / 64 : rw;

    for (size_t ws = 0; ws < rw; ws += tile_words) {
        const size_t we  = std::min(ws + tile_words, rw);
        const size_t tw  = we - ws;
        const size_t tnw = tw / 2;

        ctx.ensure(tw);

        // Ring-buffer pointer arrays into ctx.rs_store.
        uint64_t* rs2[5], *rs1[5], *rs0[5];
        for (int i = 0; i < 5; ++i) {
            rs2[i] = ctx.rs_store.data() + (size_t)(3 * i + 0) * tw;
            rs1[i] = ctx.rs_store.data() + (size_t)(3 * i + 1) * tw;
            rs0[i] = ctx.rs_store.data() + (size_t)(3 * i + 2) * tw;
        }

        // Persistent C store.
        uint64_t* C0 = ctx.C_store.data() + 0 * tw;
        uint64_t* C1 = ctx.C_store.data() + 1 * tw;
        uint64_t* C2 = ctx.C_store.data() + 2 * tw;
        uint64_t* C3 = ctx.C_store.data() + 3 * tw;
        uint64_t* C4 = ctx.C_store.data() + 4 * tw;

        // Fill ring for rows (row_begin-2)..(row_begin+2).
        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = (delta < 0)
                ? (row_begin + height - (size_t)(-delta)) % height
                : (row_begin + (size_t)delta) % height;
            fill_ring_slot(src, rw, ws, we, tnw, sr, delta + 2,
                           rs2, rs1, rs0);
        }

        // Initialise C = sum of all 5 ring slots.
        for (size_t vi = 0; vi < tnw; ++vi) {
            uint64x2_t c0 = vdupq_n_u64(0), c1 = vdupq_n_u64(0), c2 = vdupq_n_u64(0);
            uint64x2_t c3 = vdupq_n_u64(0), c4 = vdupq_n_u64(0);
            for (int s = 0; s < 5; ++s)
                c5_add3(c0, c1, c2, c3, c4,
                        vld1q_u64(rs0[s] + vi * 2),
                        vld1q_u64(rs1[s] + vi * 2),
                        vld1q_u64(rs2[s] + vi * 2));
            vst1q_u64(C0 + vi * 2, c0); vst1q_u64(C1 + vi * 2, c1);
            vst1q_u64(C2 + vi * 2, c2); vst1q_u64(C3 + vi * 2, c3);
            vst1q_u64(C4 + vi * 2, c4);
        }

        int tail = 0;  // ring slot holding the oldest row-sum (leaving the window)

        for (size_t r = row_begin; r < row_end; ++r) {
            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            uint64_t*       d1  = dst.s1.row(r);
            uint64_t*       d0  = dst.s0.row(r);

            // New row entering the window (r+3).
            const size_t new_row = (r + 3) % height;
            const uint64_t* np1 = src.s1.row(new_row);
            const uint64_t* np0 = src.s0.row(new_row);

            const size_t pw0 = (ws == 0) ? rw - 2 : ws - 2;
            const size_t pw1 = (ws == 0) ? rw - 1 : ws - 1;
            const size_t nw0 = (we == rw) ? 0 : we;
            const size_t nw1 = (we == rw) ? 1 : we + 1;

            const uint64x2_t bnd_prev_new = vcombine_u64(
                vcreate_u64(np1[pw0] & np0[pw0]), vcreate_u64(np1[pw1] & np0[pw1]));
            const uint64x2_t bnd_next_new = vcombine_u64(
                vcreate_u64(np1[nw0] & np0[nw0]), vcreate_u64(np1[nw1] & np0[nw1]));

            // Sliding adult window for new row: prev carries forward each vi.
            uint64x2_t adult_new_prev = bnd_prev_new;
            uint64x2_t adult_new_curr = vandq_u64(vld1q_u64(np1 + ws),
                                                  vld1q_u64(np0 + ws));

            for (size_t vi = 0; vi < tnw; ++vi) {
                // Peek one pair ahead for row-sum computation.
                const uint64x2_t adult_new_next = (vi == tnw - 1)
                    ? bnd_next_new
                    : vandq_u64(vld1q_u64(np1 + ws + vi * 2 + 2),
                                vld1q_u64(np0 + ws + vi * 2 + 2));

                // New row-sum (row entering window).
                uint64x2_t new_r2, new_r1, new_r0;
                neon_row_sum_3bit(adult_new_prev, adult_new_curr, adult_new_next,
                                  new_r2, new_r1, new_r0);

                // Load persistent C (correct for current row r's emit).
                uint64x2_t c0 = vld1q_u64(C0 + vi * 2);
                uint64x2_t c1 = vld1q_u64(C1 + vi * 2);
                uint64x2_t c2 = vld1q_u64(C2 + vi * 2);
                uint64x2_t c3 = vld1q_u64(C3 + vi * 2);
                uint64x2_t c4 = vld1q_u64(C4 + vi * 2);

                // Copy for emit (old C before update).
                uint64x2_t e0 = c0, e1 = c1, e2 = c2, e3 = c3, e4 = c4;

                // Update persistent C: subtract leaving row, add entering row.
                const uint64x2_t old_r0 = vld1q_u64(rs0[tail] + vi * 2);
                const uint64x2_t old_r1 = vld1q_u64(rs1[tail] + vi * 2);
                const uint64x2_t old_r2 = vld1q_u64(rs2[tail] + vi * 2);
                c5_sub3_clean(c0, c1, c2, c3, c4, old_r0, old_r1, old_r2);
                c5_add3      (c0, c1, c2, c3, c4, new_r0, new_r1, new_r2);

                // Store updated C and new ring slot.
                vst1q_u64(C0 + vi * 2, c0); vst1q_u64(C1 + vi * 2, c1);
                vst1q_u64(C2 + vi * 2, c2); vst1q_u64(C3 + vi * 2, c3);
                vst1q_u64(C4 + vi * 2, c4);
                vst1q_u64(rs0[tail] + vi * 2, new_r0);
                vst1q_u64(rs1[tail] + vi * 2, new_r1);
                vst1q_u64(rs2[tail] + vi * 2, new_r2);

                // Emit row r: subtract center ADULT from e0..e4 (1-bit borrow chain).
                const uint64x2_t s1w   = vld1q_u64(sp1 + ws + vi * 2);
                const uint64x2_t s0w   = vld1q_u64(sp0 + ws + vi * 2);
                const uint64x2_t adult = vandq_u64(s1w, s0w);

                uint64x2_t borrow = vbicq_u64(adult, e0);
                e0 = veorq_u64(e0, adult);
                uint64x2_t b2 = vbicq_u64(borrow, e1); e1 = veorq_u64(e1, borrow);
                uint64x2_t b3 = vbicq_u64(b2,     e2); e2 = veorq_u64(e2, b2);
                uint64x2_t b4 = vbicq_u64(b3,     e3); e3 = veorq_u64(e3, b3);
                e4 = veorq_u64(e4, b4);

                // Predicates.
                const uint64x2_t nc4 = vnot64(e4), nc3 = vnot64(e3);
                const uint64x2_t nc2 = vnot64(e2), nc1 = vnot64(e1);

                const uint64x2_t born =
                    vandq_u64(nc4, vandq_u64(nc3,
                        vorrq_u64(vandq_u64(vandq_u64(e0, e1), nc2),
                                  vandq_u64(nc1, e2))));
                const uint64x2_t survives =
                    vandq_u64(nc4,
                        vorrq_u64(vandq_u64(nc3, e2),
                                  vandq_u64(vandq_u64(e3, nc2), nc1)));

                // Next-state.
                const uint64x2_t ns1w  = vnot64(s1w), ns0w = vnot64(s0w);
                const uint64x2_t adlt2 = vandq_u64(s1w, s0w);

                vst1q_u64(d1 + ws + vi * 2,
                    vorrq_u64(veorq_u64(s0w, s1w),
                              vandq_u64(adlt2, survives)));
                vst1q_u64(d0 + ws + vi * 2,
                    vorrq_u64(
                        vorrq_u64(vandq_u64(vandq_u64(ns1w, ns0w), born),
                                  vandq_u64(s1w, ns0w)),
                        vandq_u64(adlt2, survives)));

                // Advance sliding adult window.
                adult_new_prev = adult_new_curr;
                adult_new_curr = adult_new_next;
            }

            tail = (tail + 1) % 5;
        }
    }
}
