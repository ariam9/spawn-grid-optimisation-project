#include "grid.h"
#include "io.h"
#include "timing.h"
#include "transpose.h"
#include "kernel_neon.h"
#include "kernel_scalar.h"
#include <barrier>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <input.bin> <output.bin> [generations] [--threads=N]\n"
            "  generations default: 10000\n"
            "  --threads default:   1 for width<=256, else 8\n",
            argv[0]);
        return 1;
    }

    int generations = 10000;
    int num_threads = 0;  // 0 = auto-pick from width after the grid is loaded

    for (int i = 3; i < argc; ++i) {
        if (std::strncmp(argv[i], "--threads=", 10) == 0) {
            char* end;
            long t = std::strtol(argv[i] + 10, &end, 10);
            if (*end != '\0' || t < 1) {
                std::fprintf(stderr, "Error: invalid --threads value '%s'\n", argv[i] + 10);
                return 1;
            }
            num_threads = (int)t;
        } else {
            char* end;
            long g = std::strtol(argv[i], &end, 10);
            if (*end != '\0' || g <= 0) {
                std::fprintf(stderr, "Error: expected generations or flag, got '%s'\n", argv[i]);
                return 1;
            }
            generations = (int)g;
        }
    }

    uint64_t width, height;
    auto cells_in = read_grid(argv[1], width, height);

    const size_t W = (size_t)width;
    const size_t H = (size_t)height;

    if (num_threads == 0) num_threads = (W <= 256) ? 1 : 8;

    if ((size_t)num_threads > H) {
        std::fprintf(stderr, "Warning: --threads=%d > height=%zu; clamping to %zu\n",
                     num_threads, H, H);
        num_threads = (int)H;
    }

    BitplanePair buf[2];
    buf[0].alloc(W, H);
    buf[1].alloc(W, H);
    if (!buf[0].s1.data || !buf[0].s0.data || !buf[1].s1.data || !buf[1].s0.data) {
        std::fprintf(stderr, "Error: bitplane allocation failed\n");
        return 1;
    }

    bytes_to_bitplanes(cells_in, buf[0], W, H);
    cells_in.clear();
    cells_in.shrink_to_fit();

    // NEON's 128-cell vector width has too few iterations to pay off at small
    // widths; the scalar kernel wins there.
    const bool use_scalar = (W <= 256);

    Timer timer;

    if (num_threads == 1) {
        KernelContext ctx;
        timer.start();
        int src = 0, dst = 1;
        for (int g = 0; g < generations; ++g) {
            if (use_scalar)
                kernel_scalar(buf[src], buf[dst], H, 0, H, ctx);
            else
                kernel_neon(buf[src], buf[dst], H, 0, H, ctx);
            int tmp = src; src = dst; dst = tmp;
        }
        const double ms = timer.elapsed_ms();
        std::printf("%.3f ms\n", ms);

        std::vector<uint8_t> cells_out;
        bitplanes_to_bytes(buf[src], cells_out, W, H);
        buf[0].free_data(); buf[1].free_data();
        write_grid(argv[2], width, height, cells_out);
    } else {
        const size_t strip = H / (size_t)num_threads;
        std::barrier<> setup_done(num_threads + 1);
        std::barrier<> sync(num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(t, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

                const size_t rb = (size_t)t * strip;
                const size_t re = (t == num_threads - 1) ? H : rb + strip;
                int local_src = 0, local_dst = 1;
                KernelContext ctx;

                setup_done.arrive_and_wait();

                for (int g = 0; g < generations; ++g) {
                    if (use_scalar)
                        kernel_scalar(buf[local_src], buf[local_dst],
                                      H, rb, re, ctx);
                    else
                        kernel_neon(buf[local_src], buf[local_dst],
                                    H, rb, re, ctx);
                    sync.arrive_and_wait();
                    int tmp = local_src; local_src = local_dst; local_dst = tmp;
                }
            });
        }

        timer.start();
        setup_done.arrive_and_wait();
        for (auto& th : threads) th.join();

        const double ms = timer.elapsed_ms();
        std::printf("%.3f ms\n", ms);

        const int final_src = generations % 2;
        std::vector<uint8_t> cells_out;
        bitplanes_to_bytes(buf[final_src], cells_out, W, H);
        buf[0].free_data(); buf[1].free_data();
        write_grid(argv[2], width, height, cells_out);
    }

    return 0;
}
