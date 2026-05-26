// Phase 2 unit + end-to-end tests for kernel_scalar.
// Tests:
//   1. popcount5      – all 32 5-bit input patterns
//   2. predicates     – all 25 neighbour counts 0..25
//   3. end-to-end     – kernel vs slow per-cell reference, 1/10/100 gens
#include "../src/grid.h"
#include "../src/transpose.h"
#include "../src/kernel_scalar.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Inline copies of the math helpers (must stay in sync with kernel_scalar.cpp)
// ---------------------------------------------------------------------------

static void popcount5_ref(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e,
                          uint64_t& bit2, uint64_t& bit1, uint64_t& bit0)
{
    const uint64_t s_abc = a ^ b ^ c;
    const uint64_t c_abc = (a & b) | (c & (a ^ b));
    const uint64_t c_des = (d & e) | (s_abc & (d ^ e));
    bit0 = s_abc ^ d ^ e;
    bit1 = c_abc ^ c_des;
    bit2 = c_abc & c_des;
}

// ---------------------------------------------------------------------------
// Test 1: popcount5 – all 32 patterns of (a,b,c,d,e) as uniform 64-bit lanes
// ---------------------------------------------------------------------------
static bool test_popcount5()
{
    bool ok = true;
    for (int p = 0; p < 32; ++p) {
        uint64_t a = (p>>0)&1 ? ~0ULL : 0ULL;
        uint64_t b = (p>>1)&1 ? ~0ULL : 0ULL;
        uint64_t c = (p>>2)&1 ? ~0ULL : 0ULL;
        uint64_t d = (p>>3)&1 ? ~0ULL : 0ULL;
        uint64_t e = (p>>4)&1 ? ~0ULL : 0ULL;
        uint64_t bit2, bit1, bit0;
        popcount5_ref(a, b, c, d, e, bit2, bit1, bit0);

        const int expected = __builtin_popcount((unsigned)p);
        const int got = (bit2 ? 4 : 0) | (bit1 ? 2 : 0) | (bit0 ? 1 : 0);
        if (got != expected) {
            std::fprintf(stderr, "  popcount5 FAIL: inputs=0b%05d expected=%d got=%d\n",
                         p, expected, got);
            ok = false;
        }
    }
    std::printf("  popcount5:    %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: predicates – all counts 0..25 as uniform lanes
// ---------------------------------------------------------------------------
static bool test_predicates()
{
    bool ok = true;
    for (int A = 0; A <= 25; ++A) {
        const uint64_t c0 = ((A>>0)&1) ? ~0ULL : 0ULL;
        const uint64_t c1 = ((A>>1)&1) ? ~0ULL : 0ULL;
        const uint64_t c2 = ((A>>2)&1) ? ~0ULL : 0ULL;
        const uint64_t c3 = ((A>>3)&1) ? ~0ULL : 0ULL;
        const uint64_t c4 = ((A>>4)&1) ? ~0ULL : 0ULL;

        const uint64_t born     = ~c4 & ~c3 & ((c0 & c1 & ~c2) | (~c1 & c2));
        const uint64_t survives = ~c4 & ((~c3 & c2) | (c3 & ~c2 & ~c1));

        const bool born_val = (born     != 0);
        const bool surv_val = (survives != 0);
        const bool born_exp = (A >= 3 && A <= 5);
        const bool surv_exp = (A >= 4 && A <= 9);

        if (born_val != born_exp || surv_val != surv_exp) {
            std::fprintf(stderr, "  predicates FAIL: A=%d born=%d(exp %d) surv=%d(exp %d)\n",
                         A, born_val, born_exp, surv_val, surv_exp);
            ok = false;
        }
    }
    std::printf("  predicates:   %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Slow per-cell reference for one generation
// State machine: EMPTY→EGG(born) | EGG→JUV | JUV→ADULT | ADULT→ADULT(surv)|EMPTY
// born=(3≤A≤5), survives=(4≤A≤9), A = #ADULT neighbours in 5×5 minus centre.
// ---------------------------------------------------------------------------
static void step_ref(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                     size_t W, size_t H)
{
    for (size_t r = 0; r < H; ++r) {
        for (size_t c = 0; c < W; ++c) {
            int A = 0;
            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int ir = (int)r + dr; if (ir < 0) ir += (int)H; else if (ir >= (int)H) ir -= (int)H;
                    int ic = (int)c + dc; if (ic < 0) ic += (int)W; else if (ic >= (int)W) ic -= (int)W;
                    if (src[(size_t)ir * W + (size_t)ic] == 3) ++A;
                }
            }
            switch (src[r * W + c]) {
                case 0: dst[r * W + c] = (A >= 3 && A <= 5) ? 1 : 0; break;
                case 1: dst[r * W + c] = 2; break;
                case 2: dst[r * W + c] = 3; break;
                case 3: dst[r * W + c] = (A >= 4 && A <= 9) ? 3 : 0; break;
                default: dst[r * W + c] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: kernel_scalar vs slow reference
// ---------------------------------------------------------------------------
static bool run_end_to_end(const char* label, const std::vector<uint8_t>& init,
                           size_t W, size_t H, int gens)
{
    // Slow reference
    std::vector<uint8_t> ref = init;
    std::vector<uint8_t> ref2(W * H);
    for (int g = 0; g < gens; ++g) {
        step_ref(ref, ref2, W, H);
        std::swap(ref, ref2);
    }

    // kernel_scalar
    BitplanePair buf[2];
    buf[0].alloc(W, H);
    buf[1].alloc(W, H);
    if (!buf[0].s1.data || !buf[1].s1.data) {
        std::fprintf(stderr, "  alloc failed\n");
        return false;
    }
    bytes_to_bitplanes(init, buf[0], W, H);

    ScalarKernelContext ctx;
    int src = 0, dst_idx = 1;
    for (int g = 0; g < gens; ++g) {
        kernel_scalar(buf[src], buf[dst_idx], W, H, 0, H, 0, ctx);
        int t = src; src = dst_idx; dst_idx = t;
    }

    std::vector<uint8_t> result;
    bitplanes_to_bytes(buf[src], result, W, H);

    buf[0].free_data();
    buf[1].free_data();

    bool ok = (result == ref);
    if (!ok) {
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i] != ref[i]) {
                std::fprintf(stderr,
                    "    first diff at byte %zu (row=%zu col=%zu): ref=%u got=%u\n",
                    i, i / W, i % W, (unsigned)ref[i], (unsigned)result[i]);
                break;
            }
        }
    }
    std::printf("  %-48s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_end_to_end()
{
    bool ok = true;

    // Build test grids
    constexpr size_t W = 128, H = 128;
    std::vector<uint8_t> grid_rand(W * H);
    std::vector<uint8_t> grid_all_adult(W * H, 3);
    std::vector<uint8_t> grid_sparse(W * H, 0);

    // Random grid with LCG (seed 42)
    uint32_t rng = 42;
    for (size_t i = 0; i < W * H; ++i) {
        rng = rng * 1664525u + 1013904223u;
        grid_rand[i] = (uint8_t)(rng & 3);
    }

    // Sparse: one adult at centre
    grid_sparse[(H/2) * W + (W/2)] = 3;

    ok &= run_end_to_end("128×128 random  seed=42  1 gen",   grid_rand,       W, H,   1);
    ok &= run_end_to_end("128×128 random  seed=42  10 gens", grid_rand,       W, H,  10);
    ok &= run_end_to_end("128×128 random  seed=42  100 gens",grid_rand,       W, H, 100);
    ok &= run_end_to_end("128×128 all-ADULT  1 gen",         grid_all_adult,  W, H,   1);
    ok &= run_end_to_end("128×128 sparse-centre  1 gen",     grid_sparse,     W, H,   1);

    // Second random seed for better coverage
    std::vector<uint8_t> grid_rand2(W * H);
    rng = 123;
    for (size_t i = 0; i < W * H; ++i) {
        rng = rng * 1664525u + 1013904223u;
        grid_rand2[i] = (uint8_t)(rng & 3);
    }
    ok &= run_end_to_end("128×128 random  seed=123  10 gens",grid_rand2,      W, H,  10);

    return ok;
}

int main()
{
    bool all_pass = true;

    std::printf("=== test_kernel_scalar ===\n");
    std::printf("--- popcount5 ---\n");
    all_pass &= test_popcount5();

    std::printf("--- predicates ---\n");
    all_pass &= test_predicates();

    std::printf("--- end-to-end vs reference ---\n");
    all_pass &= test_end_to_end();

    std::printf("=== %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
