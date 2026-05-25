#include "io.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

std::vector<uint8_t> read_grid(const std::string& path, uint64_t& width, uint64_t& height)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "Error: cannot open '%s'\n", path.c_str());
        std::exit(2);
    }
    if (std::fread(&width,  sizeof(uint64_t), 1, f) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, f) != 1) {
        std::fprintf(stderr, "Error: '%s' too short (no header)\n", path.c_str());
        std::fclose(f); std::exit(3);
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr, "Error: grid must be square and non-zero, got %" PRIu64
                     "x%" PRIu64 "\n", width, height);
        std::fclose(f); std::exit(3);
    }
    size_t N = (size_t)width * height;
    std::vector<uint8_t> cells(N);
    if (std::fread(cells.data(), 1, N, f) != N) {
        std::fprintf(stderr, "Error: '%s' cell data truncated\n", path.c_str());
        std::fclose(f); std::exit(4);
    }
    std::fclose(f);
    return cells;
}

void write_grid(const std::string& path, uint64_t width, uint64_t height,
                const std::vector<uint8_t>& cells)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "Error: cannot open '%s' for writing\n", path.c_str());
        std::exit(5);
    }
    if (std::fwrite(&width,  sizeof(uint64_t), 1, f) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, f) != 1 ||
        std::fwrite(cells.data(), 1, cells.size(), f) != cells.size()) {
        std::fprintf(stderr, "Error: write error on '%s'\n", path.c_str());
        std::fclose(f); std::exit(6);
    }
    std::fclose(f);
}
