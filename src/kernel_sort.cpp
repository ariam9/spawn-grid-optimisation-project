// Phase 4.5: sorting-network bit-sliced NEON kernel (prototype).
//
// Idea: replace the full-adder popcount + threshold network with a
// sorting network. For binary inputs sorted ascending, position k of the
// sorted output equals 1 iff at least (N-k) inputs were 1 — thermometer
// code. The four predicates we need (born = 3≤A≤5, survives = 4≤A≤9 on
// 24 neighbours) become reads of specific sorted positions.
//
// This is a prototype to MEASURE the approach against the Phase 4 NEON
// adder kernel. We expect it to lose on ALU op count by ~3–4×; the value
// is empirical evidence for the design-doc "what didn't work" section.
//
// Structure: per output row, per NEON vector column (128 cells):
//   1. Compute 25 ADULT bit vectors covering the 5×5 neighbourhood,
//      including the centre cell.
//   2. Pad to 32 vectors with zero, sort all 32 ascending using a
//      Batcher bitonic sort (~191 compare-exchanges = ~382 NEON ALU ops).
//   3. Read positions 21,22,25,26,27,28,29 → A_incl thresholds for k ∈
//      {11,10,7,6,5,4,3}. (Position 32−k of the 32-sorted array is the
//      k-th largest, which equals "popcount ≥ k".)
//   4. Mux on the centre ADULT bit to derive A_excl thresholds for
//      k ∈ {3,4,6,10}, since the centre is in the 25 sorted inputs but
//      must be excluded from the neighbour count.
//   5. born = A_excl_3 AND NOT A_excl_6; survives = A_excl_4 AND NOT A_excl_10.
//   6. Standard next-state assembly.
//
// No row caching. Ring-buffer-style amortisation in the sort regime would
// require caching 5 thermometer planes per row (vs 3 binary in Phase 4),
// pushing the 32 768-wide ring to ~100 KiB and out of L1d. We document
// that constraint; the prototype itself just sorts per-vi.

#include "kernel_sort.h"
#include <arm_neon.h>

static inline uint64x2_t vnot64(uint64x2_t x)
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

// Compare-exchange for ascending order on bit-sliced binary inputs:
// min(a,b) = a AND b at the lower index, max(a,b) = a OR b at the higher.
static inline void cx_asc(uint64x2_t& a, uint64x2_t& b)
{
    const uint64x2_t lo = vandq_u64(a, b);
    const uint64x2_t hi = vorrq_u64(a, b);
    a = lo;
    b = hi;
}

// Bitonic sort on 32 vectors, output ascending.
// Iterative Batcher construction: ~191 compare-exchanges = ~382 NEON ops.
static inline void bitonic_sort32(uint64x2_t* x)
{
    for (int k = 2; k <= 32; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < 32; ++i) {
                const int l = i ^ j;
                if (l > i) {
                    // ascending sub-direction iff bit-k of i is 0
                    if ((i & k) == 0) {
                        // ensure x[i] <= x[l]
                        const uint64x2_t lo = vandq_u64(x[i], x[l]);
                        const uint64x2_t hi = vorrq_u64(x[i], x[l]);
                        x[i] = lo; x[l] = hi;
                    } else {
                        // ensure x[i] >= x[l]
                        const uint64x2_t hi = vorrq_u64(x[i], x[l]);
                        const uint64x2_t lo = vandq_u64(x[i], x[l]);
                        x[i] = hi; x[l] = lo;
                    }
                }
            }
        }
    }
}

// Load the 5 column-shifted ADULT vectors at offsets -2..+2 for a row,
// with toroidal column wrap. Same shift pattern as kernel_neon::neon_row_sum_5.
static inline void load5_cols(const uint64_t* adult_row, size_t rw, size_t vi,
                              uint64x2_t out[5])
{
    const size_t nw = rw / 2;
    const uint64x2_t prev_v = vld1q_u64(adult_row + (vi == 0    ? rw - 2 : vi*2 - 2));
    const uint64x2_t curr_v = vld1q_u64(adult_row + vi * 2);
    const uint64x2_t next_v = vld1q_u64(adult_row + (vi == nw-1 ? 0      : vi*2 + 2));

    const uint64x2_t prev_adj = vextq_u64(prev_v, curr_v, 1);
    const uint64x2_t next_adj = vextq_u64(curr_v, next_v, 1);

    out[0] = vorrq_u64(vshlq_n_u64(curr_v, 2), vshrq_n_u64(prev_adj, 62));  // col -2
    out[1] = vorrq_u64(vshlq_n_u64(curr_v, 1), vshrq_n_u64(prev_adj, 63));  // col -1
    out[2] = curr_v;                                                          // col  0
    out[3] = vorrq_u64(vshrq_n_u64(curr_v, 1), vshlq_n_u64(next_adj, 63));  // col +1
    out[4] = vorrq_u64(vshrq_n_u64(curr_v, 2), vshlq_n_u64(next_adj, 62));  // col +2
}

void kernel_sort(const BitplanePair& src, BitplanePair& dst,
                 size_t /*width*/, size_t height,
                 size_t row_begin, size_t row_end)
{
    const size_t rw = src.s1.row_words;
    const size_t nw = rw / 2;

    // Per-output-row scratch: 5 ADULT bitplane rows (r-2..r+2).
    // 5 × rw × 8 bytes = 5 KiB for w=8192, 20 KiB for w=32768. Fits L1d.
    static thread_local uint64_t adult_buf_storage[5 * 32768 / 64];
    uint64_t* adult_rows[5];
    for (int i = 0; i < 5; ++i)
        adult_rows[i] = adult_buf_storage + (size_t)i * rw;

    auto fill_adult = [&](size_t src_row, int slot) {
        const uint64_t* sp1 = src.s1.row(src_row);
        const uint64_t* sp0 = src.s0.row(src_row);
        for (size_t vi = 0; vi < nw; ++vi)
            vst1q_u64(adult_rows[slot] + vi * 2,
                      vandq_u64(vld1q_u64(sp1 + vi*2), vld1q_u64(sp0 + vi*2)));
    };

    for (size_t r = row_begin; r < row_end; ++r) {
        // Fill 5 ADULT rows for r-2..r+2 (toroidal).
        for (int dr = -2; dr <= 2; ++dr) {
            const size_t sr = dr < 0
                ? (r + height - (size_t)(-dr)) % height
                : (r + (size_t)dr) % height;
            fill_adult(sr, dr + 2);
        }

        const uint64_t* sp1_r = src.s1.row(r);
        const uint64_t* sp0_r = src.s0.row(r);
        uint64_t* d1 = dst.s1.row(r);
        uint64_t* d0 = dst.s0.row(r);

        for (size_t vi = 0; vi < nw; ++vi) {
            // Build 25 ADULT vectors covering 5×5 neighbourhood including centre.
            uint64x2_t x[32];
            for (int dr = 0; dr < 5; ++dr) {
                load5_cols(adult_rows[dr], rw, vi, &x[dr * 5]);
            }
            // Pad positions 25..31 with zero.
            const uint64x2_t zero = vdupq_n_u64(0);
            x[25] = zero; x[26] = zero; x[27] = zero;
            x[28] = zero; x[29] = zero; x[30] = zero; x[31] = zero;

            // Sort all 32 ascending.
            bitonic_sort32(x);

            // After ascending sort of 32 inputs (7 are zero pads),
            // position (32-k) is "the k-th largest input is 1" = "popcount ≥ k".
            // popcount of the 32-input array == popcount of the 25 real inputs
            //   == A_incl (count of ADULTs in the 5×5 including centre).
            const uint64x2_t Ai_3  = x[29];  // A_incl ≥ 3
            const uint64x2_t Ai_4  = x[28];  // A_incl ≥ 4
            const uint64x2_t Ai_5  = x[27];  // A_incl ≥ 5
            const uint64x2_t Ai_6  = x[26];  // A_incl ≥ 6
            const uint64x2_t Ai_7  = x[25];  // A_incl ≥ 7
            const uint64x2_t Ai_10 = x[22];  // A_incl ≥ 10
            const uint64x2_t Ai_11 = x[21];  // A_incl ≥ 11

            // Centre ADULT bit (used to convert A_incl thresholds to A_excl).
            const uint64x2_t s1w = vld1q_u64(sp1_r + vi*2);
            const uint64x2_t s0w = vld1q_u64(sp0_r + vi*2);
            const uint64x2_t c   = vandq_u64(s1w, s0w);
            const uint64x2_t nc  = vnot64(c);

            // A_excl ≥ k = (NOT c AND A_incl_k) OR (c AND A_incl_{k+1}).
            const uint64x2_t Ae_3  = vorrq_u64(vandq_u64(nc, Ai_3),  vandq_u64(c, Ai_4));
            const uint64x2_t Ae_4  = vorrq_u64(vandq_u64(nc, Ai_4),  vandq_u64(c, Ai_5));
            const uint64x2_t Ae_6  = vorrq_u64(vandq_u64(nc, Ai_6),  vandq_u64(c, Ai_7));
            const uint64x2_t Ae_10 = vorrq_u64(vandq_u64(nc, Ai_10), vandq_u64(c, Ai_11));

            // Predicates.
            const uint64x2_t born     = vandq_u64(Ae_3, vnot64(Ae_6));
            const uint64x2_t survives = vandq_u64(Ae_4, vnot64(Ae_10));

            // Next state — identical formulas to kernel_neon.
            const uint64x2_t ns1w  = vnot64(s1w);
            const uint64x2_t ns0w  = vnot64(s0w);

            vst1q_u64(d1 + vi*2,
                vorrq_u64(veorq_u64(s0w, s1w), vandq_u64(c, survives)));

            vst1q_u64(d0 + vi*2,
                vorrq_u64(
                    vorrq_u64(vandq_u64(vandq_u64(ns1w, ns0w), born),
                              vandq_u64(s1w, ns0w)),
                    vandq_u64(c, survives)));
        }
    }
}
