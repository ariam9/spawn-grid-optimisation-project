#include "io.h"
#include "timing.h"
#include <cstdio>
#include <string>

// Phase 0 placeholder kernel: copies input to output (does not simulate).
// Replace this with the real kernel in Phase 2.
static void kernel_stub(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                        uint64_t /*width*/, uint64_t /*height*/, int /*generations*/)
{
    dst = src;
}

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = (int)g;
    }

    uint64_t width, height;
    auto cells_in = read_grid(argv[1], width, height);

    std::vector<uint8_t> cells_out(cells_in.size());

    Timer timer;
    timer.start();
    kernel_stub(cells_in, cells_out, width, height, generations);
    double ms = timer.elapsed_ms();

    std::printf("%.3f ms\n", ms);

    write_grid(argv[2], width, height, cells_out);
    return 0;
}
