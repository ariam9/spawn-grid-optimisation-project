#!/usr/bin/env python3
"""Generate test grids without numpy. Uses bytearray for memory efficiency."""
import random
import struct
import sys
from pathlib import Path

EMPTY, EGG, JUVENILE, ADULT = 0, 1, 2, 3


def write_grid(path, n, cells: bytearray):
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, n))
        f.write(cells)


def gen_random_low(r: random.Random, n):
    cells = bytearray(n * n)
    for i in range(n * n):
        if r.random() < 0.05:
            cells[i] = ADULT
    return cells


def gen_random_high(r: random.Random, n):
    cells = bytearray(n * n)
    for i in range(n * n):
        if r.random() < 0.50:
            cells[i] = ADULT
    return cells


def gen_structured(r: random.Random, n):
    cells = bytearray(n * n)
    period, block = 10, 3
    for y in range(n):
        for x in range(n):
            if (y % period < block) and (x % period < block):
                cells[y * n + x] = ADULT
    return cells


def gen_sparse_clusters(r: random.Random, n):
    cells = bytearray(n * n)
    num_clusters = max(4, int(50 * (n / 32768) ** 2))
    radius = max(4, int(20 * n / 32768))
    density = 0.80
    offsets = [(dy, dx)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dy * dy + dx * dx <= radius * radius]
    for _ in range(num_clusters):
        cy = r.randint(0, n - 1)
        cx = r.randint(0, n - 1)
        for (dy, dx) in offsets:
            if r.random() < density:
                ny = (cy + dy) % n
                nx = (cx + dx) % n
                cells[ny * n + nx] = ADULT
    return cells


def gen_boundary_stress(r: random.Random, n):
    cells = bytearray(n * n)
    density, border = 0.75, 4
    for y in range(0, border):
        for x in range(n):
            if r.random() < density:
                cells[y * n + x] = ADULT
    for y in range(n - border, n):
        for x in range(n):
            if r.random() < density:
                cells[y * n + x] = ADULT
    for y in range(n):
        for x in range(0, border):
            if r.random() < density:
                cells[y * n + x] = ADULT
    for y in range(n):
        for x in range(n - border, n):
            if r.random() < density:
                cells[y * n + x] = ADULT
    return cells


PUBLIC_GRIDS = [
    ('public_1_random_low',      gen_random_low),
    ('public_2_random_high',     gen_random_high),
    ('public_3_structured',      gen_structured),
    ('public_4_sparse_clusters', gen_sparse_clusters),
    ('public_5_boundary_stress', gen_boundary_stress),
]


def main():
    sizes = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [512, 2048]
    out_dir = Path(__file__).parent
    r = random.Random(42)
    for size in sizes:
        print(f"  {size}x{size}:")
        for name, gen_fn in PUBLIC_GRIDS:
            path = out_dir / f'{name}_{size}.bin'
            cells = gen_fn(r, size)
            write_grid(path, size, cells)
            adult_pct = cells.count(ADULT) / (size * size) * 100
            print(f"    {path.name}  ({adult_pct:.1f}% adult)")


if __name__ == '__main__':
    main()
