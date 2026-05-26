// NEON 128-bit bit-sliced kernel with column tiling + persistent running sum.
//
// (1) Per-call scratch buffers live in ctx (no malloc/free per generation).
// (2) C0..C4 is a 5-bitplane running cumulative count of the 5-row vertical
//     window, maintained across rows. Per row, sub the evicted slot's
//     row-sum and add the newly computed one — two 3-bit-into-5-bit ripple
//     chains instead of five 3-bit adds.
// (3) Single fused inner-loop pass: emit, plus new-row sliding-window adult
//     compute + new row-sum + C update + slot store, all in one pass over
//     tw words. The intermediate adult_tmp buffer is gone; new-row ADULT
//     bits live entirely in registers (prev/curr/next sliding window).
//
// `tile_cols = 0` means full width (no tiling).
#include "kernel_neon.h"
#include <algorithm>
#include <arm_neon.h>
#include <cassert>

static inline uint64x2_t vnot64(uint64x2_t x)
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

// Horizontal 5-wide popcount for one row tile (used only at initialisation).
static void neon_row_sum_5_tile(const uint64_t* adult_tile, size_t tnw,
                                 const uint64_t* bnd_prev,
                                 const uint64_t* bnd_next,
                                 uint64_t* out2, uint64_t* out1, uint64_t* out0)
{
    for (size_t vi = 0; vi < tnw; ++vi) {
        uint64x2_t prev_v = (vi == 0)     ? vld1q_u64(bnd_prev)
                                          : vld1q_u64(adult_tile + vi*2 - 2);
        uint64x2_t curr_v = vld1q_u64(adult_tile + vi * 2);
        uint64x2_t next_v = (vi == tnw-1) ? vld1q_u64(bnd_next)
                                          : vld1q_u64(adult_tile + vi*2 + 2);

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
                 size_t row_begin, size_t row_end,
                 KernelContext& ctx,
                 size_t tile_cols)
{
    const size_t rw = src.s1.row_words;
    const size_t tile_words = tile_cols ? tile_cols / 64 : rw;
    assert(ctx.tile_words >= tile_words);

    for (size_t ws = 0; ws < rw; ws += tile_words) {
        const size_t we = std::min(ws + tile_words, rw);
        const size_t tw  = we - ws;
        const size_t tnw = tw / 2;

        uint64_t* rs2[5], *rs1[5], *rs0[5];
        for (int i = 0; i < 5; ++i) {
            rs2[i] = ctx.rs_store + (size_t)(3*i + 0) * ctx.tile_words;
            rs1[i] = ctx.rs_store + (size_t)(3*i + 1) * ctx.tile_words;
            rs0[i] = ctx.rs_store + (size_t)(3*i + 2) * ctx.tile_words;
        }
        uint64_t* const adult_tmp = ctx.adult_tmp;
        uint64_t* const C0 = ctx.C_store + 0 * ctx.tile_words;
        uint64_t* const C1 = ctx.C_store + 1 * ctx.tile_words;
        uint64_t* const C2 = ctx.C_store + 2 * ctx.tile_words;
        uint64_t* const C3 = ctx.C_store + 3 * ctx.tile_words;
        uint64_t* const C4 = ctx.C_store + 4 * ctx.tile_words;

        // Tile-relative toroidal column boundary word indices.
        const size_t pw0 = (ws == 0) ? rw-2 : ws-2;
        const size_t pw1 = (ws == 0) ? rw-1 : ws-1;
        const size_t nw0 = (we == rw) ? 0 : we;
        const size_t nw1 = (we == rw) ? 1 : we+1;

        auto fill_slot = [&](size_t src_row, int slot) {
            const uint64_t* sp1 = src.s1.row(src_row);
            const uint64_t* sp0 = src.s0.row(src_row);
            for (size_t vi = 0; vi < tnw; ++vi)
                vst1q_u64(adult_tmp + vi*2,
                          vandq_u64(vld1q_u64(sp1 + ws + vi*2),
                                    vld1q_u64(sp0 + ws + vi*2)));
            uint64_t bnd_prev[2] = { sp1[pw0] & sp0[pw0], sp1[pw1] & sp0[pw1] };
            uint64_t bnd_next[2] = { sp1[nw0] & sp0[nw0], sp1[nw1] & sp0[nw1] };
            neon_row_sum_5_tile(adult_tmp, tnw, bnd_prev, bnd_next,
                                rs2[slot], rs1[slot], rs0[slot]);
        };

        // Initialise the 5-slot ring for rows (row_begin-2)..(row_begin+2), toroidal.
        for (int delta = -2; delta <= 2; ++delta) {
            const size_t sr = delta < 0
                ? (row_begin + height - (size_t)(-delta)) % height
                : (row_begin + (size_t)delta) % height;
            fill_slot(sr, delta + 2);
        }

        // Compute initial running cumulative sum C0..C4 = sum of all 5 slot row-sums.
        for (size_t vi = 0; vi < tnw; ++vi) {
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
            vst1q_u64(C0 + vi*2, c0);
            vst1q_u64(C1 + vi*2, c1);
            vst1q_u64(C2 + vi*2, c2);
            vst1q_u64(C3 + vi*2, c3);
            vst1q_u64(C4 + vi*2, c4);
        }

        int tail = 0;

        for (size_t r = row_begin; r < row_end; ++r) {

            const uint64_t* sp1 = src.s1.row(r);
            const uint64_t* sp0 = src.s0.row(r);
            uint64_t* d1 = dst.s1.row(r);
            uint64_t* d0 = dst.s0.row(r);

            const size_t new_row = (r + 3) % height;
            const uint64_t* sp1n = src.s1.row(new_row);
            const uint64_t* sp0n = src.s0.row(new_row);

            // New row toroidal column boundaries — used at vi=0 (prev) and vi=tnw-1 (next).
            const uint64_t bnd_prev_arr[2] = { sp1n[pw0] & sp0n[pw0], sp1n[pw1] & sp0n[pw1] };
            const uint64_t bnd_next_arr[2] = { sp1n[nw0] & sp0n[nw0], sp1n[nw1] & sp0n[nw1] };
            const uint64x2_t bnd_prev_v = vld1q_u64(bnd_prev_arr);
            const uint64x2_t bnd_next_v = vld1q_u64(bnd_next_arr);

            // Sliding window for new-row ADULT bits, initialised at vi=0.
            uint64x2_t adult_prev = bnd_prev_v;
            uint64x2_t adult_curr = vandq_u64(vld1q_u64(sp1n + ws),
                                              vld1q_u64(sp0n + ws));

            uint64_t* const rs0t = rs0[tail];
            uint64_t* const rs1t = rs1[tail];
            uint64_t* const rs2t = rs2[tail];

            for (size_t vi = 0; vi < tnw; ++vi) {
                // ===== Emit phase (uses centre-subtracted local copy of C) =====
                const uint64x2_t s1w = vld1q_u64(sp1 + ws + vi*2);
                const uint64x2_t s0w = vld1q_u64(sp0 + ws + vi*2);

                uint64x2_t c0 = vld1q_u64(C0 + vi*2);
                uint64x2_t c1 = vld1q_u64(C1 + vi*2);
                uint64x2_t c2 = vld1q_u64(C2 + vi*2);
                uint64x2_t c3 = vld1q_u64(C3 + vi*2);
                uint64x2_t c4 = vld1q_u64(C4 + vi*2);

                // Local copies tc for centre-sub; c (untouched) is reused below for C update.
                uint64x2_t tc0 = c0, tc1 = c1, tc2 = c2, tc3 = c3, tc4 = c4;

                {
                    uint64x2_t borrow = vandq_u64(s1w, s0w);
                    uint64x2_t diff;
                    diff = veorq_u64(tc0, borrow); borrow = vandq_u64(vnot64(tc0), borrow); tc0 = diff;
                    diff = veorq_u64(tc1, borrow); borrow = vandq_u64(vnot64(tc1), borrow); tc1 = diff;
                    diff = veorq_u64(tc2, borrow); borrow = vandq_u64(vnot64(tc2), borrow); tc2 = diff;
                    diff = veorq_u64(tc3, borrow); borrow = vandq_u64(vnot64(tc3), borrow); tc3 = diff;
                    tc4 = veorq_u64(tc4, borrow);
                }

                const uint64x2_t nc4 = vnot64(tc4), nc3 = vnot64(tc3);
                const uint64x2_t nc2 = vnot64(tc2), nc1 = vnot64(tc1);
                const uint64x2_t born =
                    vandq_u64(nc4, vandq_u64(nc3,
                        vorrq_u64(vandq_u64(vandq_u64(tc0, tc1), nc2),
                                  vandq_u64(nc1, tc2))));
                const uint64x2_t survives =
                    vandq_u64(nc4,
                        vorrq_u64(vandq_u64(nc3, tc2),
                                  vandq_u64(vandq_u64(tc3, nc2), nc1)));

                const uint64x2_t ns1w   = vnot64(s1w);
                const uint64x2_t ns0w   = vnot64(s0w);
                const uint64x2_t adult_ = vandq_u64(s1w, s0w);

                vst1q_u64(d1 + ws + vi*2,
                    vorrq_u64(veorq_u64(s0w, s1w), vandq_u64(adult_, survives)));
                vst1q_u64(d0 + ws + vi*2,
                    vorrq_u64(
                        vorrq_u64(vandq_u64(vandq_u64(ns1w, ns0w), born),
                                  vandq_u64(s1w, ns0w)),
                        vandq_u64(adult_, survives)));

                // ===== New-row row-sum (sliding-window ADULT bits) =====
                const uint64x2_t adult_next = (vi == tnw - 1)
                    ? bnd_next_v
                    : vandq_u64(vld1q_u64(sp1n + ws + (vi+1)*2),
                                vld1q_u64(sp0n + ws + (vi+1)*2));

                const uint64x2_t prev_adj = vextq_u64(adult_prev, adult_curr, 1);
                const uint64x2_t next_adj = vextq_u64(adult_curr, adult_next, 1);

                const uint64x2_t a  = vorrq_u64(vshlq_n_u64(adult_curr, 2), vshrq_n_u64(prev_adj, 62));
                const uint64x2_t b  = vorrq_u64(vshlq_n_u64(adult_curr, 1), vshrq_n_u64(prev_adj, 63));
                const uint64x2_t cc = adult_curr;
                const uint64x2_t d_ = vorrq_u64(vshrq_n_u64(adult_curr, 1), vshlq_n_u64(next_adj, 63));
                const uint64x2_t e_ = vorrq_u64(vshrq_n_u64(adult_curr, 2), vshlq_n_u64(next_adj, 62));

                const uint64x2_t axb   = veorq_u64(a, b);
                const uint64x2_t s_abc = veorq_u64(axb, cc);
                const uint64x2_t c_abc = vorrq_u64(vandq_u64(a, b), vandq_u64(cc, axb));
                const uint64x2_t dxe   = veorq_u64(d_, e_);
                const uint64x2_t c_des = vorrq_u64(vandq_u64(d_, e_), vandq_u64(s_abc, dxe));

                const uint64x2_t r0n = veorq_u64(veorq_u64(s_abc, d_), e_);
                const uint64x2_t r1n = veorq_u64(c_abc, c_des);
                const uint64x2_t r2n = vandq_u64(c_abc, c_des);

                // Advance sliding window for next iteration.
                adult_prev = adult_curr;
                adult_curr = adult_next;

                // ===== Update C: C += r_new - r_old (3-bit ripple sub then add) =====
                const uint64x2_t r0o = vld1q_u64(rs0t + vi*2);
                const uint64x2_t r1o = vld1q_u64(rs1t + vi*2);
                const uint64x2_t r2o = vld1q_u64(rs2t + vi*2);

                {
                    uint64x2_t borrow, ns;
                    ns = veorq_u64(c0, r0o); borrow = vandq_u64(vnot64(c0), r0o); c0 = ns;
                    {
                        const uint64x2_t c1xr = veorq_u64(c1, r1o);
                        ns = veorq_u64(c1xr, borrow);
                        borrow = vorrq_u64(vandq_u64(vnot64(c1), r1o),
                                           vandq_u64(vnot64(c1xr), borrow));
                        c1 = ns;
                    }
                    {
                        const uint64x2_t c2xr = veorq_u64(c2, r2o);
                        ns = veorq_u64(c2xr, borrow);
                        borrow = vorrq_u64(vandq_u64(vnot64(c2), r2o),
                                           vandq_u64(vnot64(c2xr), borrow));
                        c2 = ns;
                    }
                    ns = veorq_u64(c3, borrow); borrow = vandq_u64(vnot64(c3), borrow); c3 = ns;
                    c4 = veorq_u64(c4, borrow);
                }

                {
                    uint64x2_t carry, ns;
                    ns = veorq_u64(c0, r0n); carry = vandq_u64(c0, r0n); c0 = ns;
                    {
                        const uint64x2_t c1xr = veorq_u64(c1, r1n);
                        ns = veorq_u64(c1xr, carry);
                        carry = vorrq_u64(vandq_u64(c1, r1n), vandq_u64(carry, c1xr));
                        c1 = ns;
                    }
                    {
                        const uint64x2_t c2xr = veorq_u64(c2, r2n);
                        ns = veorq_u64(c2xr, carry);
                        carry = vorrq_u64(vandq_u64(c2, r2n), vandq_u64(carry, c2xr));
                        c2 = ns;
                    }
                    ns = veorq_u64(c3, carry); carry = vandq_u64(c3, carry); c3 = ns;
                    c4 = veorq_u64(c4, carry);
                }

                vst1q_u64(C0 + vi*2, c0);
                vst1q_u64(C1 + vi*2, c1);
                vst1q_u64(C2 + vi*2, c2);
                vst1q_u64(C3 + vi*2, c3);
                vst1q_u64(C4 + vi*2, c4);
                vst1q_u64(rs0t + vi*2, r0n);
                vst1q_u64(rs1t + vi*2, r1n);
                vst1q_u64(rs2t + vi*2, r2n);
            }

            tail = (tail + 1) % 5;
        }
    }
}
