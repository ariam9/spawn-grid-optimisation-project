#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

static constexpr size_t BUF_BYTES = 256UL * 1024 * 1024; // 256 MiB each
static constexpr int    ITERS     = 8;

int main()
{
    void* a = std::aligned_alloc(64, BUF_BYTES);
    void* b = std::aligned_alloc(64, BUF_BYTES);
    if (!a || !b) { std::fputs("alloc failed\n", stderr); return 1; }

    std::memset(a, 0xAB, BUF_BYTES);
    std::memset(b, 0xCD, BUF_BYTES);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        std::memcpy(b, a, BUF_BYTES);
        // prevent dead-store elimination
        asm volatile("" : : "r"(b) : "memory");
    }
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double bytes_moved = (double)BUF_BYTES * ITERS;  // bytes read
    double gbps = bytes_moved / elapsed_s / 1e9;

    std::printf("memcpy: %.0f MiB x %d iters in %.3f s = %.2f GB/s\n",
                BUF_BYTES / (1024.0 * 1024.0), ITERS, elapsed_s, gbps);

    std::free(a);
    std::free(b);
    return 0;
}
