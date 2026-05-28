// Scalar bit-sliced kernel for the Monster Spawning Grid.
//
// Used for narrow grids (width <= 256) where NEON's 128-cell-per-op vector
// width has too few iterations to amortise setup overhead. Two paths:
//   - "wide" (row_words >= 2): cross-word boundary shifts, same as the
//     pre-cleanup kernel_scalar at adc1956~1:src/kernel_scalar.cpp.
//   - "narrow" (row_words == 1, width <= 64): the entire row lives in one
//     uint64_t with only the low W bits valid. Horizontal wrap is intra-word
//     cyclic rotation masked to W bits.
#include "kernel_scalar.h"

static inline void five_sum(
    uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e,
    uint64_t& out2, uint64_t& out1, uint64_t& out0)
{
    const uint64_t axb   = a ^ b;
    const uint64_t s_abc = axb ^ c;
    const uint64_t c_abc = (a & b) | (c & axb);
    const uint64_t dxe   = d ^ e;
    const uint64_t c_des = (d & e) | (s_abc & dxe);
    out0 = s_abc ^ d ^ e;
    out1 = c_abc ^ c_des;
    out2 = c_abc & c_des;
}

// Wide-path: row-sum for the cells in `curr`, drawing the 2-cell horizontal
// halo from the adjacent words `prev` and `next`.
static inline void row_sum_word_wide(
    uint64_t prev, uint64_t curr, uint64_t next,
    uint64_t& out2, uint64_t& out1, uint64_t& out0)
{
    const uint64_t a = (curr << 2) | (prev >> 62);
    const uint64_t b = (curr << 1) | (prev >> 63);
    const uint64_t c = curr;
    const uint64_t d = (curr >> 1) | (next << 63);
    const uint64_t e = (curr >> 2) | (next << 62);
    five_sum(a, b, c, d, e, out2, out1, out0);
}

// Narrow-path: row of W cells (8 <= W <= 64) packed in the low W bits of `r`.
// Horizontal halo wraps cyclically within the same word.
static inline void row_sum_word_narrow(
    uint64_t r, unsigned W, uint64_t mask,
    uint64_t& out2, uint64_t& out1, uint64_t& out0)
{
    const uint64_t a = ((r << 2) | (r >> (W - 2))) & mask;
    const uint64_t b = ((r << 1) | (r >> (W - 1))) & mask;
    const uint64_t c = r;
    const uint64_t d = ((r >> 1) | (r << (W - 1))) & mask;
    const uint64_t e = ((r >> 2) | (r << (W - 2))) & mask;
    five_sum(a, b, c, d, e, out2, out1, out0);
}

static inline void c5_sub3_scalar(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3, uint64_t& c4,
    uint64_t r0, uint64_t r1, uint64_t r2)
{
    uint64_t b = (~c0) & r0; c0 ^= r0;
    uint64_t ax = c1 ^ r1; uint64_t nb = (~c1 & r1) | (b & ~ax); c1 = ax ^ b; b = nb;
    ax = c2 ^ r2; nb = (~c2 & r2) | (b & ~ax); c2 = ax ^ b; b = nb;
    nb = b & ~c3; c3 ^= b; b = nb;
    c4 ^= b;
}

static inline void c5_add3_scalar(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3, uint64_t& c4,
    uint64_t r0, uint64_t r1, uint64_t r2)
{
    uint64_t carry = c0 & r0; c0 ^= r0;
    uint64_t ax = c1 ^ r1; uint64_t nc = (c1 & r1) | (carry & ax); c1 = ax ^ carry; carry = nc;
    ax = c2 ^ r2; nc = (c2 & r2) | (carry & ax); c2 = ax ^ carry; carry = nc;
    nc = c3 & carry; c3 ^= carry; carry = nc;
    c4 ^= carry;
}

// Predicate + next-state emit. Operates on whole uint64_t words; for narrow
// widths the unused high bits are zero in inputs and stay zero in outputs.
static inline void emit_word(
    uint64_t s1w, uint64_t s0w,
    uint64_t e0, uint64_t e1, uint64_t e2, uint64_t e3, uint64_t e4,
    uint64_t& d1, uint64_t& d0)
{
    // Subtract the centre ADULT from the A_full snapshot.
    const uint64_t adult = s1w & s0w;
    uint64_t borrow = (~e0) & adult; e0 ^= adult;
    uint64_t b2 = (~e1) & borrow;    e1 ^= borrow;
    uint64_t b3 = (~e2) & b2;        e2 ^= b2;
    uint64_t b4 = (~e3) & b3;        e3 ^= b3;
    e4 ^= b4;

    const uint64_t nc4 = ~e4, nc3 = ~e3, nc1 = ~e1;
    const uint64_t born     = nc4 & nc3 & (e2 ^ e1) & (nc1 | e0);
    const uint64_t survives = nc4 & (e3 ^ e2) & (nc3 | nc1);
    const uint64_t adult_sv = adult & survives;
    d1 = (s0w ^ s1w) | adult_sv;
    d0 = (~s0w & (s1w | born)) | adult_sv;
}

static void kernel_scalar_narrow(const BitplanePair& src, BitplanePair& dst,
                                 size_t height,
                                 size_t row_begin, size_t row_end)
{
    const unsigned W    = (unsigned)src.s1.width;
    const uint64_t mask = (W == 64) ? ~0ULL : ((1ULL << W) - 1);

    // 5 ring slots of row-sums; one word each.
    uint64_t rs2[5], rs1[5], rs0[5];

    auto load_adult_row = [&](size_t r) -> uint64_t {
        return src.s1.row(r)[0] & src.s0.row(r)[0];
    };

    for (int delta = -2; delta <= 2; ++delta) {
        const size_t sr = (delta < 0)
            ? (row_begin + height - (size_t)(-delta)) % height
            : (row_begin + (size_t)delta) % height;
        row_sum_word_narrow(load_adult_row(sr), W, mask,
                            rs2[delta + 2], rs1[delta + 2], rs0[delta + 2]);
    }

    uint64_t C0 = 0, C1 = 0, C2 = 0, C3 = 0, C4 = 0;
    for (int s = 0; s < 5; ++s)
        c5_add3_scalar(C0, C1, C2, C3, C4, rs0[s], rs1[s], rs2[s]);

    int tail = 0;
    for (size_t r = row_begin; r < row_end; ++r) {
        const uint64_t s1w = src.s1.row(r)[0];
        const uint64_t s0w = src.s0.row(r)[0];

        const size_t new_row = (r + 3) % height;
        uint64_t new_r2, new_r1, new_r0;
        row_sum_word_narrow(load_adult_row(new_row), W, mask,
                            new_r2, new_r1, new_r0);

        // Snapshot C before update (A_full at the centre row).
        uint64_t e0 = C0, e1 = C1, e2 = C2, e3 = C3, e4 = C4;

        c5_sub3_scalar(C0, C1, C2, C3, C4, rs0[tail], rs1[tail], rs2[tail]);
        c5_add3_scalar(C0, C1, C2, C3, C4, new_r0, new_r1, new_r2);
        rs0[tail] = new_r0; rs1[tail] = new_r1; rs2[tail] = new_r2;

        uint64_t d1, d0;
        emit_word(s1w, s0w, e0, e1, e2, e3, e4, d1, d0);
        dst.s1.row(r)[0] = d1 & mask;
        dst.s0.row(r)[0] = d0 & mask;

        tail = (tail + 1) % 5;
    }
}

static void kernel_scalar_wide(const BitplanePair& src, BitplanePair& dst,
                               size_t height,
                               size_t row_begin, size_t row_end,
                               KernelContext& ctx)
{
    const size_t rw = src.s1.row_words;
    const size_t pw = rw - 1;  // boundary word indices
    const size_t nw = 0;

    ctx.ensure(rw);

    uint64_t* rs2[5]; uint64_t* rs1[5]; uint64_t* rs0[5];
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

    auto fill_slot = [&](size_t sr, int slot) {
        const uint64_t* sp1 = src.s1.row(sr);
        const uint64_t* sp0 = src.s0.row(sr);
        const uint64_t bnd_prev = sp1[pw] & sp0[pw];
        const uint64_t bnd_next = sp1[nw] & sp0[nw];
        for (size_t w = 0; w < rw; ++w) {
            const uint64_t prev = (w == 0)      ? bnd_prev : sp1[w - 1] & sp0[w - 1];
            const uint64_t curr = sp1[w] & sp0[w];
            const uint64_t next = (w == rw - 1) ? bnd_next : sp1[w + 1] & sp0[w + 1];
            row_sum_word_wide(prev, curr, next,
                              rs2[slot][w], rs1[slot][w], rs0[slot][w]);
        }
    };

    for (int delta = -2; delta <= 2; ++delta) {
        const size_t sr = (delta < 0)
            ? (row_begin + height - (size_t)(-delta)) % height
            : (row_begin + (size_t)delta) % height;
        fill_slot(sr, delta + 2);
    }

    for (size_t w = 0; w < rw; ++w) {
        uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
        for (int s = 0; s < 5; ++s)
            c5_add3_scalar(c0, c1, c2, c3, c4, rs0[s][w], rs1[s][w], rs2[s][w]);
        C0[w] = c0; C1[w] = c1; C2[w] = c2; C3[w] = c3; C4[w] = c4;
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
        const uint64_t bnd_prev_n = np1[pw] & np0[pw];
        const uint64_t bnd_next_n = np1[nw] & np0[nw];

        for (size_t w = 0; w < rw; ++w) {
            const uint64_t n_prev = (w == 0)      ? bnd_prev_n : np1[w - 1] & np0[w - 1];
            const uint64_t n_curr = np1[w] & np0[w];
            const uint64_t n_next = (w == rw - 1) ? bnd_next_n : np1[w + 1] & np0[w + 1];
            uint64_t new_r2, new_r1, new_r0;
            row_sum_word_wide(n_prev, n_curr, n_next, new_r2, new_r1, new_r0);

            uint64_t c0 = C0[w], c1 = C1[w], c2 = C2[w], c3 = C3[w], c4 = C4[w];
            uint64_t e0 = c0, e1 = c1, e2 = c2, e3 = c3, e4 = c4;

            c5_sub3_scalar(c0, c1, c2, c3, c4, rs0[tail][w], rs1[tail][w], rs2[tail][w]);
            c5_add3_scalar(c0, c1, c2, c3, c4, new_r0, new_r1, new_r2);
            C0[w] = c0; C1[w] = c1; C2[w] = c2; C3[w] = c3; C4[w] = c4;
            rs0[tail][w] = new_r0; rs1[tail][w] = new_r1; rs2[tail][w] = new_r2;

            uint64_t dd1, dd0;
            emit_word(sp1[w], sp0[w], e0, e1, e2, e3, e4, dd1, dd0);
            d1[w] = dd1;
            d0[w] = dd0;
        }

        tail = (tail + 1) % 5;
    }
}

void kernel_scalar(const BitplanePair& src, BitplanePair& dst,
                   size_t height,
                   size_t row_begin, size_t row_end,
                   KernelContext& ctx)
{
    if (src.s1.row_words == 1)
        kernel_scalar_narrow(src, dst, height, row_begin, row_end);
    else
        kernel_scalar_wide(src, dst, height, row_begin, row_end, ctx);
}
