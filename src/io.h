#pragma once
#include <cstdint>
#include <vector>
#include <string>

// Read a grid file: 8-byte width, 8-byte height (must be equal), then width*height bytes.
// Returns cell data (1 byte per cell). Exits on error.
std::vector<uint8_t> read_grid(const std::string& path, uint64_t& width, uint64_t& height);

// Write a grid file with the same format.
// Exits on error.
void write_grid(const std::string& path, uint64_t width, uint64_t height,
                const std::vector<uint8_t>& cells);
