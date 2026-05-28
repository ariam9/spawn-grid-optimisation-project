// Tests for kernel_scalar across widths 8..512 (narrow path covers <=64,
// wide path covers >=128) against the slow per-cell reference.
#include "../src/grid.h"
#include "../src/transpose.h"
#include "../src/kernel_scalar.h"
#include "test_utils.h"
#include <cstdio>
#include <vector>

static std::vector<uint8_t> run_scalar(const std::vector<uint8_t>& init,
                                       size_t W, size_t H, int gens)
{
    BitplanePair buf[2];
    buf[0].alloc(W, H); buf[1].alloc(W, H);
    bytes_to_bitplanes(init, buf[0], W, H);

    KernelContext ctx;
    int src = 0, dst = 1;
    for (int g = 0; g < gens; ++g) {
        kernel_scalar(buf[src], buf[dst], H, 0, H, ctx);
        int t = src; src = dst; dst = t;
    }

    std::vector<uint8_t> result;
    bitplanes_to_bytes(buf[src], result, W, H);
    buf[0].free_data(); buf[1].free_data();
    return result;
}

static bool one(const char* label, const std::vector<uint8_t>& init,
                size_t W, size_t H, int gens)
{
    std::vector<uint8_t> ref = init;
    std::vector<uint8_t> tmp(W * H);
    for (int g = 0; g < gens; ++g) { step_ref(ref, tmp, W, H); std::swap(ref, tmp); }

    auto got = run_scalar(init, W, H, gens);
    bool ok = (got == ref);
    if (!ok) {
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "    first diff byte %zu (row=%zu col=%zu): ref=%u scl=%u\n",
                             i, i/W, i%W, (unsigned)ref[i], (unsigned)got[i]);
                break;
            }
    }
    std::printf("  %-32s : %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

static std::vector<uint8_t> random_grid(size_t W, size_t H, uint32_t seed)
{
    std::vector<uint8_t> g(W * H);
    uint32_t rng = seed;
    for (auto& x : g) { rng = rng * 1664525u + 1013904223u; x = rng & 3; }
    return g;
}

int main()
{
    bool all = true;
    const size_t sizes[] = {8, 16, 32, 64, 128, 256, 512};
    const int gens_list[] = {1, 10, 50};

    std::printf("=== test_kernel_scalar ===\n");
    for (size_t S : sizes) {
        for (int G : gens_list) {
            auto g42  = random_grid(S, S, 42);
            auto g123 = random_grid(S, S, 123);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%zux%zu rand42  %d gens",  S, S, G);
            all &= one(buf, g42,  S, S, G);
            std::snprintf(buf, sizeof(buf), "%zux%zu rand123 %d gens",  S, S, G);
            all &= one(buf, g123, S, S, G);
        }
        // Edge case: single live ADULT at the centre, 1 gen — exercises wrap-around
        // for narrow grids especially.
        std::vector<uint8_t> sparse(S * S, 0);
        sparse[(S / 2) * S + S / 2] = 3;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%zux%zu sparse  1 gen",  S, S);
        all &= one(buf, sparse, S, S, 1);
    }

    std::printf("=== %s ===\n", all ? "ALL PASS" : "SOME FAILED");
    return all ? 0 : 1;
}
