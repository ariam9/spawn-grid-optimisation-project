// Phase 3 tests for kernel_neon.
// 1. End-to-end: NEON output vs slow per-cell reference.
// 2. Cross-check: NEON output byte-identical to scalar output.
#include "../src/context.h"
#include "../src/grid.h"
#include "../src/transpose.h"
#include "../src/kernel_scalar.h"
#include "../src/kernel_neon.h"
#include <cstdio>
#include <cstring>
#include <vector>

// Slow per-cell reference (same as in test_kernel_scalar.cpp).
// EMPTY→EGG(born), EGG→JUV, JUV→ADULT, ADULT→ADULT(surv)|EMPTY
// born=(3≤A≤5), surv=(4≤A≤9), A=#ADULT in 5×5 minus centre.
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
                case 0: dst[r*W+c] = (A>=3 && A<=5) ? 1 : 0; break;
                case 1: dst[r*W+c] = 2; break;
                case 2: dst[r*W+c] = 3; break;
                case 3: dst[r*W+c] = (A>=4 && A<=9) ? 3 : 0; break;
                default: dst[r*W+c] = 0;
            }
        }
    }
}

using KernelFn = void(*)(const BitplanePair&, BitplanePair&, size_t, size_t, size_t, size_t,
                         KernelContext&, size_t);

static std::vector<uint8_t> run_kernel(const std::vector<uint8_t>& init,
                                       size_t W, size_t H, int gens, KernelFn kfn)
{
    BitplanePair buf[2];
    buf[0].alloc(W, H); buf[1].alloc(W, H);
    bytes_to_bitplanes(init, buf[0], W, H);

    KernelContext ctx;
    ctx.alloc(W / 64);

    int src = 0, dst = 1;
    for (int g = 0; g < gens; ++g) {
        kfn(buf[src], buf[dst], W, H, 0, H, ctx, 0);
        int t = src; src = dst; dst = t;
    }

    std::vector<uint8_t> result;
    bitplanes_to_bytes(buf[src], result, W, H);
    buf[0].free_data(); buf[1].free_data();
    ctx.free_data();
    return result;
}

// Test NEON vs slow reference.
static bool test_neon_vs_ref(const char* label, const std::vector<uint8_t>& init,
                              size_t W, size_t H, int gens)
{
    std::vector<uint8_t> ref = init;
    std::vector<uint8_t> tmp(W * H);
    for (int g = 0; g < gens; ++g) { step_ref(ref, tmp, W, H); std::swap(ref, tmp); }

    auto got = run_kernel(init, W, H, gens, kernel_neon);
    bool ok = (got == ref);
    if (!ok)
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "    first diff byte %zu (row=%zu col=%zu): ref=%u neon=%u\n",
                             i, i/W, i%W, (unsigned)ref[i], (unsigned)got[i]);
                break;
            }
    std::printf("  neon-vs-ref   %-40s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

// Cross-check: NEON must be byte-identical to scalar.
static bool test_neon_vs_scalar(const char* label, const std::vector<uint8_t>& init,
                                 size_t W, size_t H, int gens)
{
    auto sc = run_kernel(init, W, H, gens, kernel_scalar);
    auto nv = run_kernel(init, W, H, gens, kernel_neon);
    bool ok = (sc == nv);
    if (!ok)
        for (size_t i = 0; i < sc.size(); ++i)
            if (sc[i] != nv[i]) {
                std::fprintf(stderr, "    first diff byte %zu (row=%zu col=%zu): scalar=%u neon=%u\n",
                             i, i/W, i%W, (unsigned)sc[i], (unsigned)nv[i]);
                break;
            }
    std::printf("  neon-vs-scalar %-39s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    bool all = true;
    constexpr size_t W = 128, H = 128;

    // Build test grids
    std::vector<uint8_t> rand42(W*H), rand123(W*H), all_adult(W*H, 3), sparse(W*H, 0);
    sparse[(H/2)*W + W/2] = 3;
    uint32_t rng = 42;
    for (auto& x : rand42)  { rng = rng*1664525u+1013904223u; x = rng & 3; }
    rng = 123;
    for (auto& x : rand123) { rng = rng*1664525u+1013904223u; x = rng & 3; }

    std::printf("=== test_kernel_neon ===\n");
    std::printf("--- NEON vs slow reference ---\n");
    all &= test_neon_vs_ref("128x128 rand42  1 gen",   rand42,     W, H,   1);
    all &= test_neon_vs_ref("128x128 rand42  10 gens", rand42,     W, H,  10);
    all &= test_neon_vs_ref("128x128 rand42  100 gens",rand42,     W, H, 100);
    all &= test_neon_vs_ref("128x128 all-ADULT 1 gen", all_adult,  W, H,   1);
    all &= test_neon_vs_ref("128x128 sparse  1 gen",   sparse,     W, H,   1);
    all &= test_neon_vs_ref("128x128 rand123 10 gens", rand123,    W, H,  10);

    std::printf("--- NEON vs scalar cross-check ---\n");
    all &= test_neon_vs_scalar("128x128 rand42  100 gens", rand42,    W, H, 100);
    all &= test_neon_vs_scalar("128x128 all-ADULT 1 gen",  all_adult, W, H,   1);
    all &= test_neon_vs_scalar("128x128 rand123 10 gens",  rand123,   W, H,  10);

    std::printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
