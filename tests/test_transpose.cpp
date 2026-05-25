// Phase 1 round-trip test: bytes → bitplanes → bytes, verify byte-identical.
#include "../src/transpose.h"
#include "../src/grid.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Read a grid file; returns cell data. Sets width/height.
static std::vector<uint8_t> load_grid(const char* path, size_t& width, size_t& height)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    uint64_t w, h;
    if (std::fread(&w, 8, 1, f) != 1 || std::fread(&h, 8, 1, f) != 1) {
        std::fprintf(stderr, "bad header in %s\n", path); std::exit(1);
    }
    if (w != h || w == 0 || (w & (w-1)) != 0 || w % 64 != 0) {
        std::fprintf(stderr, "width %llu not a power-of-two multiple of 64\n",
                     (unsigned long long)w);
        std::exit(1);
    }
    width = (size_t)w; height = (size_t)h;
    std::vector<uint8_t> cells(width * height);
    if (std::fread(cells.data(), 1, cells.size(), f) != cells.size()) {
        std::fprintf(stderr, "truncated data in %s\n", path); std::exit(1);
    }
    std::fclose(f);
    return cells;
}

static bool run(const char* path)
{
    size_t width, height;
    std::vector<uint8_t> original = load_grid(path, width, height);

    BitplanePair planes;
    planes.alloc(width, height);
    if (!planes.s1.data || !planes.s0.data) {
        std::fprintf(stderr, "alloc failed for %s\n", path);
        return false;
    }

    bytes_to_bitplanes(original, planes, width, height);

    std::vector<uint8_t> recovered;
    bitplanes_to_bytes(planes, recovered, width, height);

    planes.free_data();

    bool ok = (original.size() == recovered.size() &&
               std::memcmp(original.data(), recovered.data(), original.size()) == 0);

    std::printf("  %-52s : %s\n", path, ok ? "PASS" : "FAIL");

    if (!ok) {
        for (size_t i = 0; i < original.size(); ++i) {
            if (original[i] != recovered[i]) {
                size_t row = i / width, col = i % width;
                std::fprintf(stderr,
                    "    first diff at byte %zu [row=%zu col=%zu]: "
                    "original=%u recovered=%u\n",
                    i, row, col, (unsigned)original[i], (unsigned)recovered[i]);
                break;
            }
        }
    }
    return ok;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_transpose <grid.bin>...\n");
        return 1;
    }

    bool all_pass = true;
    std::printf("Round-trip test (bytes -> bitplanes -> bytes):\n");
    for (int i = 1; i < argc; ++i)
        if (!run(argv[i]))
            all_pass = false;

    std::printf("%s\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
