#include "io.h"
#include "timing.h"
#include "transpose.h"
#include "kernel_scalar.h"
#include "kernel_neon.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <input.bin> <output.bin> [generations] [--kernel=scalar|neon]\n",
            argv[0]);
        return 1;
    }

    int generations = 10000;
    bool use_neon = false;

    for (int i = 3; i < argc; ++i) {
        if (std::strncmp(argv[i], "--kernel=", 9) == 0) {
            const char* kname = argv[i] + 9;
            if (std::strcmp(kname, "neon") == 0)       use_neon = true;
            else if (std::strcmp(kname, "scalar") != 0) {
                std::fprintf(stderr, "Error: unknown kernel '%s'\n", kname);
                return 1;
            }
        } else {
            char* end;
            long g = std::strtol(argv[i], &end, 10);
            if (*end != '\0' || g <= 0) {
                std::fprintf(stderr, "Error: expected generations or --kernel=, got '%s'\n", argv[i]);
                return 1;
            }
            generations = (int)g;
        }
    }

    uint64_t width, height;
    auto cells_in = read_grid(argv[1], width, height);

    const size_t W = (size_t)width;
    const size_t H = (size_t)height;

    BitplanePair buf[2];
    buf[0].alloc(W, H);
    buf[1].alloc(W, H);
    if (!buf[0].s1.data || !buf[0].s0.data || !buf[1].s1.data || !buf[1].s0.data) {
        std::fprintf(stderr, "Error: bitplane allocation failed\n");
        return 1;
    }

    bytes_to_bitplanes(cells_in, buf[0], W, H);
    // Free input bytes before simulation to reduce peak RSS.
    cells_in.clear();
    cells_in.shrink_to_fit();

    Timer timer;
    timer.start();

    int src = 0, dst = 1;
    for (int g = 0; g < generations; ++g) {
        if (use_neon)
            kernel_neon(buf[src], buf[dst], W, H, 0, H);
        else
            kernel_scalar(buf[src], buf[dst], W, H, 0, H);
        int tmp = src; src = dst; dst = tmp;
    }

    const double ms = timer.elapsed_ms();
    std::printf("%.3f ms\n", ms);

    std::vector<uint8_t> cells_out;
    bitplanes_to_bytes(buf[src], cells_out, W, H);

    buf[0].free_data();
    buf[1].free_data();

    write_grid(argv[2], width, height, cells_out);
    return 0;
}
