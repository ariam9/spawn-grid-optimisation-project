// Stage 8a unit test for copy_ghost_strip / copy_interior_to_global.
//
// Grid: 64 rows × 256 bits (4 words/row).
// Pattern: s1 row r filled with word value (r | (r<<8) | ... | (r<<56)),
//          s0 row r filled with complement (~s1 row).
// This makes every row uniquely identifiable by its byte values.
//
// For each K ∈ {1, 2, 4, 8} and a set of strips (including wrap-around cases):
//   1. copy_ghost_strip  → local buffer
//   2. Verify local buffer row content matches expected global rows.
//   3. copy_interior_to_global → fresh global buffer
//   4. cmp interior rows of fresh global against original.
#include "../src/grid.h"
#include <cassert>
#include <cstdio>
#include <cstring>

static const size_t GRID_H   = 64;
static const size_t GRID_W   = 256;   // bits
static const size_t ROW_WORDS = GRID_W / 64;  // = 4

// Fill bitplane row with a pattern derived from the row index.
static void fill_row(uint64_t* dst, size_t row_idx)
{
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v |= (row_idx & 0xFFULL) << (b * 8);
    for (size_t w = 0; w < ROW_WORDS; ++w) dst[w] = v;
}

// Build a global BitplanePair with the known pattern.
static void fill_global(BitplanePair& g)
{
    for (size_t r = 0; r < GRID_H; ++r) {
        fill_row(g.s1.row(r), r);
        fill_row(g.s0.row(r), ~r & 0xFF);
    }
}

// Return the expected uint64 word value for row r in s1.
static uint64_t expected_s1(size_t r)
{
    uint64_t v = 0;
    size_t rv = r & 0xFF;
    for (int b = 0; b < 8; ++b) v |= (rv << (b * 8));
    return v;
}
static uint64_t expected_s0(size_t r)
{
    uint64_t v = 0;
    size_t rv = (~r) & 0xFF;
    for (int b = 0; b < 8; ++b) v |= (rv << (b * 8));
    return v;
}

// Check a local buffer row against what we expect from a given global row.
static bool check_local_row(const LocalBitplanePair& loc, size_t local_r,
                             size_t expected_global_r, const char* ctx)
{
    bool ok = true;
    uint64_t es1 = expected_s1(expected_global_r);
    uint64_t es0 = expected_s0(expected_global_r);
    for (size_t w = 0; w < ROW_WORDS; ++w) {
        if (loc.s1.row(local_r)[w] != es1) {
            std::fprintf(stderr, "  FAIL %s: local_r=%zu word=%zu s1 got=0x%016llx exp=0x%016llx (global_r=%zu)\n",
                         ctx, local_r, w, (unsigned long long)loc.s1.row(local_r)[w],
                         (unsigned long long)es1, expected_global_r);
            ok = false;
        }
        if (loc.s0.row(local_r)[w] != es0) {
            std::fprintf(stderr, "  FAIL %s: local_r=%zu word=%zu s0 got=0x%016llx exp=0x%016llx (global_r=%zu)\n",
                         ctx, local_r, w, (unsigned long long)loc.s0.row(local_r)[w],
                         (unsigned long long)es0, expected_global_r);
            ok = false;
        }
    }
    return ok;
}

struct TestCase { size_t begin, end; const char* name; };

static bool run_k(const BitplanePair& global, size_t K,
                  const TestCase* cases, size_t ncases)
{
    bool all_ok = true;
    for (size_t ci = 0; ci < ncases; ++ci) {
        const size_t rb = cases[ci].begin;
        const size_t re = cases[ci].end;
        const size_t sh = re - rb;
        const size_t local_h = sh + 2 * K;
        char ctx[128];

        // Allocate local src buffer.
        LocalBitplanePair loc;
        loc.alloc(GRID_W, local_h);

        copy_ghost_strip(global, loc, rb, re, K, GRID_H);

        // Verify top ghosts
        for (size_t i = 0; i < K; ++i) {
            size_t expected_gr = (rb + GRID_H - K + i) % GRID_H;
            std::snprintf(ctx, sizeof(ctx), "K=%zu strip=[%zu,%zu) top_ghost i=%zu", K, rb, re, i);
            if (!check_local_row(loc, i, expected_gr, ctx)) all_ok = false;
        }
        // Verify interior
        for (size_t i = 0; i < sh; ++i) {
            std::snprintf(ctx, sizeof(ctx), "K=%zu strip=[%zu,%zu) interior i=%zu", K, rb, re, i);
            if (!check_local_row(loc, K + i, rb + i, ctx)) all_ok = false;
        }
        // Verify bottom ghosts
        for (size_t i = 0; i < K; ++i) {
            size_t expected_gr = (re + i) % GRID_H;
            std::snprintf(ctx, sizeof(ctx), "K=%zu strip=[%zu,%zu) bot_ghost i=%zu", K, rb, re, i);
            if (!check_local_row(loc, K + sh + i, expected_gr, ctx)) all_ok = false;
        }

        // Round-trip: copy interior back to a fresh global, compare interior rows.
        BitplanePair fresh;
        fresh.alloc(GRID_W, GRID_H);
        fresh.s1.zero(); fresh.s0.zero();

        copy_interior_to_global(loc, fresh, rb, re, K);

        for (size_t r = rb; r < re; ++r) {
            if (std::memcmp(fresh.s1.row(r), global.s1.row(r), ROW_WORDS * 8) != 0 ||
                std::memcmp(fresh.s0.row(r), global.s0.row(r), ROW_WORDS * 8) != 0) {
                std::fprintf(stderr, "  FAIL K=%zu strip=[%zu,%zu) roundtrip: global row %zu mismatch\n",
                             K, rb, re, r);
                all_ok = false;
            }
        }

        loc.free_data();
        fresh.free_data();
    }
    return all_ok;
}

int main()
{
    BitplanePair global;
    global.alloc(GRID_W, GRID_H);
    fill_global(global);

    // Strips: interior, starting at 0 (top-wrap), ending at H (bottom-wrap),
    // single-row, and full-grid.
    static const TestCase cases[] = {
        { 16, 32, "middle" },
        {  0, 16, "top_wrap" },        // top ghosts wrap from bottom
        { 48, 64, "bottom_wrap" },     // bottom ghosts wrap from top
        {  0,  8, "single_small_top" },
        { 56, 64, "single_small_bot" },
        {  4, 60, "large_interior" },
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);

    static const size_t Ks[] = {1, 2, 4, 8};
    bool overall = true;

    for (size_t Ki = 0; Ki < 4; ++Ki) {
        size_t K = Ks[Ki];
        bool ok = run_k(global, K, cases, ncases);
        std::printf("  K=%zu: %s\n", K, ok ? "PASS" : "FAIL");
        overall = overall && ok;
    }

    global.free_data();
    std::printf("test_ghost_copy: %s\n", overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}
