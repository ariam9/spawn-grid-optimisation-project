// NEON bit-sliced kernel for the Monster Spawning Grid.
//
// Each NEON register holds 2×uint64, processing 128 cells per operation.
// A persistent 5-bit column accumulator C is maintained across rows via a
// sliding-window subtract/add instead of recomputing from all 5 ring slots.
// The inner vi loop is x2-unrolled so the OOO engine can overlap the two
// independent carry chains and hide load-use latency.
#include "kernel_neon.h"
#include <arm_neon.h>

// 3-bit horizontal row-sum for a window of 5 adjacent ADULT columns.
// prev_v/curr_v/next_v are consecutive NEON pairs (2×uint64 each).
//
// The 5 shifted ADULT lanes a,b,c,d,e fuse shift-and-OR into single
// vsriq/vsliq ("shift-right/left-and-insert") instructions, dropping
// 4 vorrqs per call. vsri/vsli are 2-input — no register-pressure cost.
static inline void neon_row_sum_3bit(
    uint64x2_t prev_v, uint64x2_t curr_v, uint64x2_t next_v,
    uint64x2_t& out2, uint64x2_t& out1, uint64x2_t& out0)
{
    const uint64x2_t prev_adj = vextq_u64(prev_v, curr_v, 1);
    const uint64x2_t next_adj = vextq_u64(curr_v, next_v, 1);

    // a = (curr << 2) | (prev_adj >> 62): top 62 bits from curr<<2,
    //                                      low  2 bits from prev_adj>>62 (via SRI)
    const uint64x2_t a = vsriq_n_u64(vshlq_n_u64(curr_v, 2), prev_adj, 62);
    const uint64x2_t b = vsriq_n_u64(vshlq_n_u64(curr_v, 1), prev_adj, 63);
    const uint64x2_t c = curr_v;
    // d = (curr >> 1) | (next_adj << 63): low 63 bits from curr>>1,
    //                                      bit 63 from next_adj<<63 (via SLI)
    const uint64x2_t d = vsliq_n_u64(vshrq_n_u64(curr_v, 1), next_adj, 63);
    const uint64x2_t e = vsliq_n_u64(vshrq_n_u64(curr_v, 2), next_adj, 62);

    // The full-adder carries c_abc and c_des are MAJ functions; each folds to
    // one vbslq. The 3-way XOR `out0 = s_abc ^ d ^ e` emits as one veor3q
    // (SHA3). axb/dxe are kept live as both BSL masks and XOR inputs.
    const uint64x2_t axb   = veorq_u64(a, b);
    const uint64x2_t s_abc = veorq_u64(axb, c);
    const uint64x2_t c_abc = vbslq_u64(axb, c, a);
    const uint64x2_t dxe   = veorq_u64(d, e);
    const uint64x2_t c_des = vbslq_u64(dxe, s_abc, d);

    out0 = veor3q_u64(s_abc, d, e);
    out1 = veorq_u64(c_abc, c_des);
    out2 = vandq_u64(c_abc, c_des);
}

// Subtract 3-bit (r2,r1,r0) from 5-bit (c4..c0), ripple borrow.
// Inner borrow stages fold (~ci & ri) | (~(ci^ri) & b) into one vbslq:
//   when ci==ri: borrow passes through (b);  when ci!=ri: borrow := ri.
// ci update uses veor3q so it doesn't wait on ax (reduces store-buffer pressure).
static inline void c5_sub3_neon(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    uint64x2_t b = vbicq_u64(r0, c0);
    c0 = veorq_u64(c0, r0);
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nb = vbslq_u64(ax, r1, b);
    c1 = veor3q_u64(c1, r1, b);
    b = nb;
    ax = veorq_u64(c2, r2);
    nb = vbslq_u64(ax, r2, b);
    c2 = veor3q_u64(c2, r2, b);
    b = nb;
    // c4 ^= (b & ~c3) folds into one vbcaxq (SHA3). c3 read before overwrite.
    c4 = vbcaxq_u64(c4, b, c3);
    c3 = veorq_u64(c3, b);
}

// Add 3-bit (r2,r1,r0) to 5-bit (c4..c0), ripple carry.
// Inner carry stages fold (ci & ri) | (carry & (ci^ri)) — the MAJ function —
// into one vbslq: when ci==ri: carry := ci;  when ci!=ri: carry passes through.
// ci update uses veor3q so it doesn't wait on ax (reduces store-buffer pressure).
static inline void c5_add3_neon(
    uint64x2_t& c0, uint64x2_t& c1, uint64x2_t& c2,
    uint64x2_t& c3, uint64x2_t& c4,
    uint64x2_t r0, uint64x2_t r1, uint64x2_t r2)
{
    uint64x2_t carry = vandq_u64(c0, r0);
    c0 = veorq_u64(c0, r0);
    uint64x2_t ax = veorq_u64(c1, r1);
    uint64x2_t nc = vbslq_u64(ax, carry, c1);
    c1 = veor3q_u64(c1, r1, carry);
    carry = nc;
    ax = veorq_u64(c2, r2);
    nc = vbslq_u64(ax, carry, c2);
    c2 = veor3q_u64(c2, r2, carry);
    carry = nc;
    nc = vandq_u64(c3, carry);
    c3 = veorq_u64(c3, carry);
    carry = nc;
    c4 = veorq_u64(c4, carry);
}

// Fill one ring slot from a source row.
// pw0/pw1: indices of the two words immediately left of the row (toroidal wrap).
// nw0/nw1: indices of the two words immediately right of the row (toroidal wrap).
static void fill_ring_slot_neon(
    const BitplanePair& src,
    size_t tnw,
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
    uint64x2_t adult_curr = vandq_u64(vld1q_u64(sp1), vld1q_u64(sp0));

    for (size_t vi = 0; vi < tnw; ++vi) {
        uint64x2_t adult_next = (vi == tnw - 1)
            ? bnd_next
            : vandq_u64(vld1q_u64(sp1 + vi * 2 + 2),
                        vld1q_u64(sp0 + vi * 2 + 2));

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
    const size_t rw  = src.s1.row_words;
    const size_t tnw = rw / 2;

    // Toroidal column-boundary word indices (the two words that wrap around
    // the left and right edges of the full-width row).
    const size_t pw0 = rw - 2;
    const size_t pw1 = rw - 1;
    const size_t nw0 = 0;
    const size_t nw1 = 1;

    ctx.ensure(rw);

    uint64_t* rs2[5], *rs1[5], *rs0[5];
    for (int i = 0; i < 5; ++i) {
        rs2[i] = ctx.rs_store.data() + (size_t)(3 * i + 0) * rw;
        rs1[i] = ctx.rs_store.data() + (size_t)(3 * i + 1) * rw;
        rs0[i] = ctx.rs_store.data() + (size_t)(3 * i + 2) * rw;
    }

    uint64_t* C0 = ctx.C_store.data() + 0 * rw;
    uint64_t* C1 = ctx.C_store.data() + 1 * rw;
    uint64_t* C2 = ctx.C_store.data() + 2 * rw;
    uint64_t* C3 = ctx.C_store.data() + 3 * rw;
    uint64_t* C4 = ctx.C_store.data() + 4 * rw;

    for (int delta = -2; delta <= 2; ++delta) {
        const size_t sr = (delta < 0)
            ? (row_begin + height - (size_t)(-delta)) % height
            : (row_begin + (size_t)delta) % height;
        fill_ring_slot_neon(src, tnw, pw0, pw1, nw0, nw1,
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
        uint64x2_t adult_curr = vandq_u64(vld1q_u64(np1), vld1q_u64(np0));

        // x2 unrolled: process NEON pairs vi and vi+1 together.
        // The two accumulators (c*_0, c*_1) are independent; the OOO engine can
        // overlap their carry chains, hiding the serial sub3+add3 latency.
        // Loop condition is vi+2 < tnw (not <=), so adult_next_1 is NEVER the
        // boundary case here — its load is unconditional. The even-tnw boundary
        // (vi+2 == tnw) is handled by a peeled x2 iteration after the loop;
        // this keeps `bnd_next` off the loop's hot path.
        size_t vi = 0;
        for (; vi + 2 < tnw; vi += 2) {
            const uint64x2_t adult_next_0 =
                vandq_u64(vld1q_u64(np1 + (vi + 1) * 2),
                          vld1q_u64(np0 + (vi + 1) * 2));
            const uint64x2_t adult_next_1 =
                vandq_u64(vld1q_u64(np1 + (vi + 2) * 2),
                          vld1q_u64(np0 + (vi + 2) * 2));

            uint64x2_t new_r2_0, new_r1_0, new_r0_0;
            uint64x2_t new_r2_1, new_r1_1, new_r0_1;
            neon_row_sum_3bit(adult_prev,  adult_curr,   adult_next_0,
                              new_r2_0, new_r1_0, new_r0_0);
            neon_row_sum_3bit(adult_curr,  adult_next_0, adult_next_1,
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

            const uint64x2_t s1w_0 = vld1q_u64(sp1 + vi * 2);
            const uint64x2_t s0w_0 = vld1q_u64(sp0 + vi * 2);
            const uint64x2_t s1w_1 = vld1q_u64(sp1 + (vi + 1) * 2);
            const uint64x2_t s0w_1 = vld1q_u64(sp0 + (vi + 1) * 2);

            // Emit directly from C = A_full (includes centre). This works because
            // born fires only on EMPTY (centre=0, so A==A_full) and survives only
            // on ADULT (centre=1, so A==A_full-1, which maps the {4..9} window to
            // {5..10} in A_full). No centre subtraction needed.
            //   born     = A_full in {3,4,5}
            //   survives = A_full in {5..10}
            // The inner factor of born is a MUX on c1, saving one op vs eor+orn+and.
            // d1 and d0 emit via veor3q/vbcaxq (SHA3) because adult_sv is disjoint
            // from the other terms so OR reduces to XOR.
            {
                const uint64x2_t born =
                    vbicq_u64(vbslq_u64(c1_0, vbicq_u64(c0_0, c2_0), c2_0),
                              vorrq_u64(c4_0, c3_0));
                const uint64x2_t surv_lo =
                    vandq_u64(vbicq_u64(c2_0, c3_0), vorrq_u64(c1_0, c0_0));
                const uint64x2_t surv_hi =
                    vbicq_u64(vbicq_u64(c3_0, c2_0), vandq_u64(c1_0, c0_0));
                const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_0);
                const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_0, s0w_0), survives);
                vst1q_u64(d1 + vi * 2,
                    veor3q_u64(s0w_0, s1w_0, adult_sv));
                vst1q_u64(d0 + vi * 2,
                    vbcaxq_u64(adult_sv, vorrq_u64(s1w_0, born), s0w_0));
            }
            {
                const uint64x2_t born =
                    vbicq_u64(vbslq_u64(c1_1, vbicq_u64(c0_1, c2_1), c2_1),
                              vorrq_u64(c4_1, c3_1));
                const uint64x2_t surv_lo =
                    vandq_u64(vbicq_u64(c2_1, c3_1), vorrq_u64(c1_1, c0_1));
                const uint64x2_t surv_hi =
                    vbicq_u64(vbicq_u64(c3_1, c2_1), vandq_u64(c1_1, c0_1));
                const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_1);
                const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_1, s0w_1), survives);
                vst1q_u64(d1 + (vi + 1) * 2,
                    veor3q_u64(s0w_1, s1w_1, adult_sv));
                vst1q_u64(d0 + (vi + 1) * 2,
                    vbcaxq_u64(adult_sv, vorrq_u64(s1w_1, born), s0w_1));
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

            vst1q_u64(C0 + vi * 2, c0_0);       vst1q_u64(C0 + (vi + 1) * 2, c0_1);
            vst1q_u64(C1 + vi * 2, c1_0);       vst1q_u64(C1 + (vi + 1) * 2, c1_1);
            vst1q_u64(C2 + vi * 2, c2_0);       vst1q_u64(C2 + (vi + 1) * 2, c2_1);
            vst1q_u64(C3 + vi * 2, c3_0);       vst1q_u64(C3 + (vi + 1) * 2, c3_1);
            vst1q_u64(C4 + vi * 2, c4_0);       vst1q_u64(C4 + (vi + 1) * 2, c4_1);
            vst1q_u64(rs0[tail] + vi * 2,       new_r0_0);
            vst1q_u64(rs0[tail] + (vi + 1) * 2, new_r0_1);
            vst1q_u64(rs1[tail] + vi * 2,       new_r1_0);
            vst1q_u64(rs1[tail] + (vi + 1) * 2, new_r1_1);
            vst1q_u64(rs2[tail] + vi * 2,       new_r2_0);
            vst1q_u64(rs2[tail] + (vi + 1) * 2, new_r2_1);

            adult_prev = adult_next_0;
            adult_curr = adult_next_1;
        }

        // Peeled x2 boundary iteration: runs when tnw is even and >= 2, i.e.
        // exactly one full pair remains and its right-edge wraps to bnd_next.
        // Same body as the main loop above, but adult_next_1 is bnd_next
        // directly (no load, no ternary), so the compiler keeps bnd_next in
        // a register across the row instead of speculatively reloading it.
        if (vi + 2 == tnw) {
            const uint64x2_t adult_next_0 =
                vandq_u64(vld1q_u64(np1 + (vi + 1) * 2),
                          vld1q_u64(np0 + (vi + 1) * 2));

            uint64x2_t new_r2_0, new_r1_0, new_r0_0;
            uint64x2_t new_r2_1, new_r1_1, new_r0_1;
            neon_row_sum_3bit(adult_prev,  adult_curr,   adult_next_0,
                              new_r2_0, new_r1_0, new_r0_0);
            neon_row_sum_3bit(adult_curr,  adult_next_0, bnd_next,
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

            const uint64x2_t s1w_0 = vld1q_u64(sp1 + vi * 2);
            const uint64x2_t s0w_0 = vld1q_u64(sp0 + vi * 2);
            const uint64x2_t s1w_1 = vld1q_u64(sp1 + (vi + 1) * 2);
            const uint64x2_t s0w_1 = vld1q_u64(sp0 + (vi + 1) * 2);

            {
                const uint64x2_t born =
                    vbicq_u64(vbslq_u64(c1_0, vbicq_u64(c0_0, c2_0), c2_0),
                              vorrq_u64(c4_0, c3_0));
                const uint64x2_t surv_lo =
                    vandq_u64(vbicq_u64(c2_0, c3_0), vorrq_u64(c1_0, c0_0));
                const uint64x2_t surv_hi =
                    vbicq_u64(vbicq_u64(c3_0, c2_0), vandq_u64(c1_0, c0_0));
                const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_0);
                const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_0, s0w_0), survives);
                vst1q_u64(d1 + vi * 2,
                    veor3q_u64(s0w_0, s1w_0, adult_sv));
                vst1q_u64(d0 + vi * 2,
                    vbcaxq_u64(adult_sv, vorrq_u64(s1w_0, born), s0w_0));
            }
            {
                const uint64x2_t born =
                    vbicq_u64(vbslq_u64(c1_1, vbicq_u64(c0_1, c2_1), c2_1),
                              vorrq_u64(c4_1, c3_1));
                const uint64x2_t surv_lo =
                    vandq_u64(vbicq_u64(c2_1, c3_1), vorrq_u64(c1_1, c0_1));
                const uint64x2_t surv_hi =
                    vbicq_u64(vbicq_u64(c3_1, c2_1), vandq_u64(c1_1, c0_1));
                const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_1);
                const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_1, s0w_1), survives);
                vst1q_u64(d1 + (vi + 1) * 2,
                    veor3q_u64(s0w_1, s1w_1, adult_sv));
                vst1q_u64(d0 + (vi + 1) * 2,
                    vbcaxq_u64(adult_sv, vorrq_u64(s1w_1, born), s0w_1));
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

            vst1q_u64(C0 + vi * 2, c0_0);       vst1q_u64(C0 + (vi + 1) * 2, c0_1);
            vst1q_u64(C1 + vi * 2, c1_0);       vst1q_u64(C1 + (vi + 1) * 2, c1_1);
            vst1q_u64(C2 + vi * 2, c2_0);       vst1q_u64(C2 + (vi + 1) * 2, c2_1);
            vst1q_u64(C3 + vi * 2, c3_0);       vst1q_u64(C3 + (vi + 1) * 2, c3_1);
            vst1q_u64(C4 + vi * 2, c4_0);       vst1q_u64(C4 + (vi + 1) * 2, c4_1);
            vst1q_u64(rs0[tail] + vi * 2,       new_r0_0);
            vst1q_u64(rs0[tail] + (vi + 1) * 2, new_r0_1);
            vst1q_u64(rs1[tail] + vi * 2,       new_r1_0);
            vst1q_u64(rs1[tail] + (vi + 1) * 2, new_r1_1);
            vst1q_u64(rs2[tail] + vi * 2,       new_r2_0);
            vst1q_u64(rs2[tail] + (vi + 1) * 2, new_r2_1);

            vi += 2;
        }

        // Tail: single remaining pair when tnw is odd.
        if (vi < tnw) {
            uint64x2_t new_r2, new_r1, new_r0;
            neon_row_sum_3bit(adult_prev, adult_curr, bnd_next,
                              new_r2, new_r1, new_r0);

            uint64x2_t c0 = vld1q_u64(C0 + vi * 2);
            uint64x2_t c1 = vld1q_u64(C1 + vi * 2);
            uint64x2_t c2 = vld1q_u64(C2 + vi * 2);
            uint64x2_t c3 = vld1q_u64(C3 + vi * 2);
            uint64x2_t c4 = vld1q_u64(C4 + vi * 2);

            const uint64x2_t s1w = vld1q_u64(sp1 + vi * 2);
            const uint64x2_t s0w = vld1q_u64(sp0 + vi * 2);

            // Emit directly from C (= A_full); see unrolled body for rationale.
            {
                const uint64x2_t born =
                    vbicq_u64(vbslq_u64(c1, vbicq_u64(c0, c2), c2),
                              vorrq_u64(c4, c3));
                const uint64x2_t surv_lo =
                    vandq_u64(vbicq_u64(c2, c3), vorrq_u64(c1, c0));
                const uint64x2_t surv_hi =
                    vbicq_u64(vbicq_u64(c3, c2), vandq_u64(c1, c0));
                const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4);
                const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w, s0w), survives);
                vst1q_u64(d1 + vi * 2,
                    veor3q_u64(s0w, s1w, adult_sv));
                vst1q_u64(d0 + vi * 2,
                    vbcaxq_u64(adult_sv, vorrq_u64(s1w, born), s0w));
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
