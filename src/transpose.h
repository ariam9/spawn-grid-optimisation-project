#pragma once
#include "grid.h"
#include <cstdint>
#include <vector>

// Convert flat byte array (1 byte/cell) to two bitplanes.
// s1 = (byte >> 1) & 1, s0 = byte & 1.
void bytes_to_bitplanes(const std::vector<uint8_t>& src,
                        BitplanePair& dst,
                        size_t width, size_t height);

// Convert two bitplanes back to flat byte array.
void bitplanes_to_bytes(const BitplanePair& src,
                        std::vector<uint8_t>& dst,
                        size_t width, size_t height);
