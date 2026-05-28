// Tests for kernel_neon.
// 1. Exhaustive predicate check: Karnaugh born/survives vs truth table A=0..25.
// 2. End-to-end: NEON output vs slow per-cell reference.
#include "../src/grid.h"
#include "../src/transpose.h"
#include "../src/kernel_neon.h"
#include "test_utils.h"
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<uint8_t> run_neon(const std::vector<uint8_t>& init,
                                     size_t W, size_t H, int gens)
{
    BitplanePair buf[2];
    buf[0].alloc(W, H); buf[1].alloc(W, H);
    bytes_to_bitplanes(init, buf[0], W, H);

    KernelContext ctx;
    int src = 0, dst = 1;
    for (int g = 0; g < gens; ++g) {
        kernel_neon(buf[src], buf[dst], H, 0, H, ctx);
        int t = src; src = dst; dst = t;
    }

    std::vector<uint8_t> result;
    bitplanes_to_bytes(buf[src], result, W, H);
    buf[0].free_data(); buf[1].free_data();
    return result;
}

static bool test_neon_vs_ref(const char* label, const std::vector<uint8_t>& init,
                              size_t W, size_t H, int gens)
{
    std::vector<uint8_t> ref = init;
    std::vector<uint8_t> tmp(W * H);
    for (int g = 0; g < gens; ++g) { step_ref(ref, tmp, W, H); std::swap(ref, tmp); }

    auto got = run_neon(init, W, H, gens);
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

// Exhaustive check of the Karnaugh-simplified born/survives formulas against
// the truth table for all A in 0..25.
static bool test_predicates()
{
    bool ok = true;
    for (int A = 0; A <= 25; ++A) {
        const uint64_t c0 = (A>>0)&1 ? ~0ULL : 0ULL;
        const uint64_t c1 = (A>>1)&1 ? ~0ULL : 0ULL;
        const uint64_t c2 = (A>>2)&1 ? ~0ULL : 0ULL;
        const uint64_t c3 = (A>>3)&1 ? ~0ULL : 0ULL;
        const uint64_t c4 = (A>>4)&1 ? ~0ULL : 0ULL;
        const uint64_t nc4=~c4, nc3=~c3, nc1=~c1;
        const uint64_t born_got     = nc4 & nc3 & (c2^c1) & (nc1|c0);
        const uint64_t survives_got = nc4 & (c3^c2) & (nc3|nc1);
        const bool born_exp = (A >= 3 && A <= 5);
        const bool surv_exp = (A >= 4 && A <= 9);
        if ((born_got != 0) != born_exp || (survives_got != 0) != surv_exp) {
            std::fprintf(stderr, "  predicates FAIL: A=%d born=%d(exp %d) surv=%d(exp %d)\n",
                         A, (born_got!=0), born_exp, (survives_got!=0), surv_exp);
            ok = false;
        }
    }
    std::printf("  predicates:   %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    bool all = true;
    std::printf("--- predicates ---\n");
    all &= test_predicates();
    // 512 is the smallest width that dispatches to NEON (width <= 256 -> scalar).
    constexpr size_t W = 512, H = 512;

    std::vector<uint8_t> rand42(W*H), rand123(W*H), all_adult(W*H, 3), sparse(W*H, 0);
    sparse[(H/2)*W + W/2] = 3;
    uint32_t rng = 42;
    for (auto& x : rand42)  { rng = rng*1664525u+1013904223u; x = rng & 3; }
    rng = 123;
    for (auto& x : rand123) { rng = rng*1664525u+1013904223u; x = rng & 3; }

    std::printf("=== test_kernel_neon ===\n");
    std::printf("--- NEON vs slow reference ---\n");
    all &= test_neon_vs_ref("512x512 rand42  1 gen",   rand42,     W, H,   1);
    all &= test_neon_vs_ref("512x512 rand42  10 gens", rand42,     W, H,  10);
    all &= test_neon_vs_ref("512x512 rand42  100 gens",rand42,     W, H, 100);
    all &= test_neon_vs_ref("512x512 all-ADULT 1 gen", all_adult,  W, H,   1);
    all &= test_neon_vs_ref("512x512 sparse  1 gen",   sparse,     W, H,   1);
    all &= test_neon_vs_ref("512x512 rand123 10 gens", rand123,    W, H,  10);

    std::printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
