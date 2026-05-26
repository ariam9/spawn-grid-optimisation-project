#include "io.h"
#include "timing.h"
#include "transpose.h"
#include "kernel_scalar.h"
#include "kernel_neon.h"
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
            "Usage: %s <input.bin> <output.bin> [generations]"
            " [--kernel=scalar|neon] [--threads=N] [--tile-cols=N]\n",
            argv[0]);
        return 1;
    }

    int generations = 10000;
    bool use_neon = false;
    size_t tile_cols = 0;
    int num_threads = 1;

    for (int i = 3; i < argc; ++i) {
        if (std::strncmp(argv[i], "--kernel=", 9) == 0) {
            const char* kname = argv[i] + 9;
            if (std::strcmp(kname, "neon") == 0)       use_neon = true;
            else if (std::strcmp(kname, "scalar") != 0) {
                std::fprintf(stderr, "Error: unknown kernel '%s'\n", kname);
                return 1;
            }
        } else if (std::strncmp(argv[i], "--tile-cols=", 12) == 0) {
            char* end;
            long t = std::strtol(argv[i] + 12, &end, 10);
            if (*end != '\0' || t < 0) {
                std::fprintf(stderr, "Error: invalid --tile-cols value '%s'\n", argv[i] + 12);
                return 1;
            }
            tile_cols = (size_t)t;
        } else if (std::strncmp(argv[i], "--threads=", 10) == 0) {
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

    // Compute per-thread row strips. Each strip boundary is a row boundary,
    // which is always 64-byte aligned (row_words * 8 is a multiple of 64 for
    // all supported widths). No false sharing between strips.
    const size_t strip = H / (size_t)num_threads;
    std::vector<size_t> row_begin_v(num_threads), row_end_v(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        row_begin_v[t] = (size_t)t * strip;
        row_end_v[t]   = (t == num_threads - 1) ? H : row_begin_v[t] + strip;
    }

    Timer timer;
    timer.start();

    if (num_threads == 1) {
        int src = 0, dst = 1;
        for (int g = 0; g < generations; ++g) {
            if (use_neon)
                kernel_neon(buf[src], buf[dst], W, H, 0, H, tile_cols);
            else
                kernel_scalar(buf[src], buf[dst], W, H, 0, H, tile_cols);
            int tmp = src; src = dst; dst = tmp;
        }
        // result is in buf[src]
        const double ms = timer.elapsed_ms();
        std::printf("%.3f ms\n", ms);

        std::vector<uint8_t> cells_out;
        bitplanes_to_bytes(buf[src], cells_out, W, H);
        buf[0].free_data(); buf[1].free_data();
        write_grid(argv[2], width, height, cells_out);
    } else {
        // Multi-threaded: T threads each own [row_begin, row_end) for all generations.
        // std::barrier synchronises between generations so no thread reads a buffer
        // another thread is still writing to.
        std::barrier<> sync(num_threads);
        int final_src = 0;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                // Pin this thread to CPU core t. Gracefully ignored if the core
                // doesn't exist (e.g. on a 1-vCPU VM).
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(t, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

                const size_t rb = row_begin_v[t];
                const size_t re = row_end_v[t];
                int local_src = 0, local_dst = 1;

                for (int g = 0; g < generations; ++g) {
                    if (use_neon)
                        kernel_neon(buf[local_src], buf[local_dst],
                                    W, H, rb, re, tile_cols);
                    else
                        kernel_scalar(buf[local_src], buf[local_dst],
                                      W, H, rb, re, tile_cols);

                    // Wait for every thread to finish writing this generation
                    // before any thread flips to the next.
                    sync.arrive_and_wait();
                    int tmp = local_src; local_src = local_dst; local_dst = tmp;
                }

                // Thread 0 records where the final result landed.
                if (t == 0)
                    final_src = local_src;
            });
        }

        for (auto& th : threads) th.join();

        const double ms = timer.elapsed_ms();
        std::printf("%.3f ms\n", ms);

        std::vector<uint8_t> cells_out;
        bitplanes_to_bytes(buf[final_src], cells_out, W, H);
        buf[0].free_data(); buf[1].free_data();
        write_grid(argv[2], width, height, cells_out);
    }

    return 0;
}
