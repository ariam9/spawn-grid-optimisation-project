#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Slow per-cell reference for one generation. Used by test_kernel_scalar and
// test_kernel_neon for end-to-end correctness checks.
// States: EMPTY(0) EGG(1) JUVENILE(2) ADULT(3).
// born=(3<=A<=5), survives=(4<=A<=9), A = #ADULT in 5x5 neighbourhood minus centre.
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
