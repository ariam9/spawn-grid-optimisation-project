# Monster Spawning Grid — Implementation Plan

## You are Claude Code, running on the dev EC2 box

You have a shell, the source tree, and direct access to build, run, and test. The human is Anurag, who owns the design decisions and reviews each phase before you proceed. Your job: implement, test, measure, and report. Anurag's job: design judgment, code review, defending choices to the panel.

**Hard rule:** stop at every acceptance-criteria checkpoint, report results, and wait for Anurag to say "continue" before starting the next phase. Do not chain phases on your own — the value of this work is that Anurag understands every decision, which requires him to review your output at each gate.

## Hardware reality

**This box (dev):** 1 vCPU Graviton4 (Neoverse-V2, SVE2), 1.8 GiB RAM total, 64 KiB L1d, 2 MiB L2, 36 MiB L3.

**Final grading box:** 8 vCPU c8g.2xlarge, 16 GiB RAM, same Neoverse-V2 microarchitecture. Threading work happens there, not here.

**Critical RAM consideration:** you (Claude Code) are running on this 1.8 GiB box and consuming some of it yourself — likely a few hundred MiB. Effective RAM available for `spawn_sim` is more like **1.2–1.5 GiB**. Run `free -h` before every large test and abort if available memory is under 200 MiB. We absolutely cannot OOM-kill this machine.

Two constraints follow:

1. **No multithreading on this box.** Structure code so threading can be added later (`kernel(row_begin, row_end, ...)` signatures, no global mutable state), but do not spawn threads here.
2. **The 32768 grid is on the edge of what fits.** Primary development testing on 512, 2048, 8192. Only attempt 32768 after Phase 5's memory analysis confirms it'll fit. If it doesn't, defer to the 8-core box and say so.

## The design is fixed — do not propose changes

Anurag has worked out the algorithm. Implement exactly this:

- **Internal representation:** two bitplanes (s1 plane, s0 plane), packed 1-bit-per-cell.
- **ADULT plane is derived on the fly** as `s1 AND s0`. Never stored.
- **Neighbour count A is computed by a bit-sliced full-adder network** on the ADULT plane.
- **The 5×5 sum is separable:** horizontal sum-of-5, then vertical sum-of-5 of row-sums. Row-sums cached in a sliding window.
- **The count is never materialized.** Only the predicates `born = (3 ≤ A ≤ 5)` and `survives = (4 ≤ A ≤ 9)` are derived from the count bitplanes.
- **Next state from boolean formula:**
  ```
  next_s1 = (s0 XOR s1) OR (s0 AND s1 AND survives)
  next_s0 = (NOT s1 AND NOT s0 AND born)
         OR (s1 AND NOT s0)
         OR (s1 AND s0 AND survives)
  ```
- **Toroidal boundaries** via 2 ghost rows/cols of padding refreshed once per generation. No conditional logic in the inner loop.

If you find yourself wanting to change the design, stop and ask Anurag.

## Correctness gating

The reference (`reference/spawn_sim.cpp`) is the oracle. Every phase diffs its output against `.expected.bin` files using `cmp`. **If `cmp` reports any difference, stop and report — do not work around it.** All bugs are findable; silent compensation in later phases destroys the audit trail.

## Repository layout to create

```
spawn-grid/
├── PLAN.md                    # this file
├── NOTES.md                   # design decisions log, append-only
├── build.sh                   # builds release executable
├── src/
│   ├── main.cpp               # CLI: read, simulate, write, print time
│   ├── io.h, io.cpp           # binary read/write per spec
│   ├── timing.h               # steady_clock wrapper
│   ├── transpose.h, .cpp      # byte ↔ bitplane
│   ├── kernel_scalar.h, .cpp  # Phase 2 bit-sliced, uint64
│   ├── kernel_neon.h, .cpp    # Phase 3+ bit-sliced, NEON 128-bit
│   └── grid.h                 # buffer types, ghost-row helpers
├── tests/
│   ├── test_transpose.cpp
│   └── verify.sh
└── bench/
    └── bench.sh
```

A `--kernel=scalar|neon` flag lets `main.cpp` dispatch to either kernel under the same I/O code.

## Conventions

- Append every non-trivial decision to `NOTES.md` with date, what, and why. **Anurag writes the design doc from this file — do not write design-doc prose yourself, only factual notes Anurag can build on.**
- Compiler: g++-14. Flags: `-O3 -mcpu=neoverse-v2 -std=c++23 -Wall -Wextra`. Add `-fno-omit-frame-pointer` while profiling. No `-ffast-math`.
- `static_assert` for bit-layout invariants.
- Width is always a power of two, ≥ 512. Hardcode nothing else.
- After each phase, run `free -h` and record peak RSS of `spawn_sim` on the largest test grid attempted. Use `/usr/bin/time -v ./spawn_sim …` to capture peak RSS automatically.
- Commit to git after each passing phase. If something later breaks, we can bisect.

## How to talk to Anurag at checkpoints

When you hit an acceptance-criteria checkpoint, report exactly:

1. What you built (1-2 sentences).
2. Which tests passed (paste the actual `cmp` exit codes / `verify.sh` output).
3. Timing numbers from the runs (ms).
4. Peak RSS observed.
5. Any decisions you had to make that aren't in this plan, with a one-line justification.
6. The string: **"Ready for Phase N review. Awaiting approval to proceed."**

Do not start the next phase until Anurag types "continue" or equivalent.

---

## Phase 0 — Environment, oracle, baselines

### Tasks

1. Run and record:
   ```
   uname -a
   lscpu
   free -h
   g++-14 --version
   ```
2. Verify NEON header compiles:
   ```
   echo '#include <arm_neon.h>' | g++-14 -x c++ -std=c++23 -mcpu=neoverse-v2 -c - -o /dev/null
   ```
3. Build the reference per `reference/build.sh`. Time it on `public_1_random_low_512.bin` and `public_1_random_low_2048.bin`. Record both. Do not run on 8192 or 32768 yet.
4. Confirm reference output matches `.expected.bin` for sizes 512 and 2048 using `cmp`.
5. Write `bench/stream.cpp` (≈50 lines): allocate two 256-MiB buffers (no larger — we will OOM), time `memcpy` between them in a loop, report GB/s. Compile and run.
6. Write `tests/verify.sh`:
   ```bash
   #!/bin/bash
   # usage: verify.sh <input.bin> <expected.bin>
   ./spawn_sim "$1" /tmp/out.bin
   cmp /tmp/out.bin "$2" && echo PASS || echo FAIL
   ```
7. Stub `src/main.cpp` and `src/io.cpp` so `./spawn_sim in out` reads the binary, calls a placeholder kernel that copies input to output, writes the binary, prints a dummy ms time. This exercises the I/O code path early.
8. `git init` (if not already) and commit.

### Acceptance criteria

- Reference builds and matches `.expected.bin` for 512 and 2048.
- `verify.sh` on the stubbed kernel returns FAIL (the stub doesn't simulate) — confirms the harness can catch wrong output.
- `NOTES.md` contains: reference timings (512, 2048), measured DRAM bandwidth (`memcpy` GB/s), available RAM after building.

---

## Phase 1 — Bitplane representation and transpose

### Tasks

1. Define `Bitplane` in `grid.h`: `width*height/8` bytes, 64-byte aligned (`posix_memalign`). Row stride `width/8` bytes.
2. Implement `bytes_to_bitplanes(src, s1, s0, w, h)` — for each input byte b, s1 takes `(b >> 1) & 1`, s0 takes `b & 1`. Scalar version, no optimisation.
3. Implement the inverse.
4. `tests/test_transpose.cpp` loads a test grid, round-trips through bitplanes, `cmp`s against the original buffer.

### Acceptance criteria

- Round-trip byte-identical on all four 512 grids.
- Round-trip byte-identical on all four 2048 grids.
- Round-trip byte-identical on `public_1_random_low_8192.bin`. Record peak RSS.

---

## Phase 2 — Scalar bit-sliced kernel (uint64)

The bit-sliced algorithm at 64-bit width — easy to debug, easy to unit-test, translates 1:1 to NEON in Phase 3. Work in stages.

### Stage 2a — horizontal row-sum-of-5

1. Function: five adjacent 64-bit words (column offsets −2..+2) → three 64-bit words (3-bit sum, max 5).
2. Small adder network: pairs of half-adders combined.
3. Unit test: enumerate all 32 5-bit input patterns, verify all 64 lanes of output against a slow per-lane reference.

### Stage 2b — full row with horizontal torus

1. Slide the 5-input window across one bitplane row.
2. Wraparound: leftmost two output words read from the row's rightmost end. Load wrap-around words once at row start.
3. Output: 3 bitplane-rows per input row.
4. Test against a slow per-cell reference on a small grid.

### Stage 2c — vertical sum-of-5 of row-sums

1. Five row-sums (3 bitplanes each) → 5-bitplane count.
2. Schoolbook addition bit-by-bit with carries.
3. Test against slow per-cell count on a small grid.

### Stage 2d — subtract centre

1. The count above includes the centre cell. Subtract its adult bit from the count's bit-0 plane with borrow propagation.
2. Verify count exactly matches the literal 24-neighbour count for every cell.

### Stage 2e — predicates

1. `born = NOT b4 AND NOT b3 AND ((b0 AND b1 AND NOT b2) OR (NOT b1 AND b2))`
2. `survives = NOT b4 AND ((NOT b3 AND b2) OR (b3 AND NOT b2 AND NOT b1))`
3. Enumerate all 25 possible counts and verify both predicates.

### Stage 2f — next-state assembly

1. Compute `next_s1`, `next_s0` per the design formulas.
2. Single generation, no caching, no tiling. Wraparound loads for the top/bottom 2 rows.

### Stage 2g — multi-gen loop

1. Ping-pong between two bitplane buffer pairs.
2. 10,000 generations. Convert to bytes once at the end. Write.
3. Time only the simulation, not I/O.

### Acceptance criteria

- `verify.sh` PASS on all four 512 grids.
- `verify.sh` PASS on all four 2048 grids.
- `verify.sh` PASS on `public_1_random_low_8192.bin`.
- Timing recorded in `NOTES.md` per size. Expect meaningful speedup over reference already (bit-slicing alone = 64× throughput per word).
- Peak RSS for 8192 run recorded.

---

## Phase 3 — NEON port

### Tasks

1. Copy `src/kernel_scalar.cpp` to `src/kernel_neon.cpp`.
2. Replace 64-bit ops with NEON intrinsics:
   - `^` → `veorq_u64`
   - `&` → `vandq_u64`
   - `|` → `vorrq_u64`
   - `~` → no native 64-bit `vnotq_u64`. Use `vmvnq_u32` reinterpreted, or maintain inverses via XOR with an all-ones constant kept in a register (often cheaper).
   - Loads/stores: `vld1q_u64` / `vst1q_u64`. All bitplane allocations must be 16-byte aligned.
3. Each half-/full-adder becomes 2–5 NEON instructions over 128-bit registers — 128 cells per call.
4. Horizontal sliding window: NEON bit-shifts (`vshlq_n_u64`, `vshrq_n_u64`) are per-64-bit-lane, not across the full 128-bit register. To shift the 128-cell window left/right by 1 bit you must combine two adjacent vectors using `vextq_u64` for the byte-level part and `vshlq`/`vshrq` for the within-lane part. Get this right with a focused unit test before integrating.
5. The toroidal wrap rows are still handled by the ghost-padding approach.
6. Keep the scalar kernel building and runnable (`--kernel=scalar`) for cross-checks.

### Acceptance criteria

- `verify.sh` PASS on all four 512 and 2048 grids with `--kernel=neon`.
- `verify.sh` PASS on `public_1_random_low_8192.bin` with NEON.
- Both `--kernel=scalar` and `--kernel=neon` still produce byte-identical output for at least one size (cross-check that the NEON port preserves semantics).
- Timing recorded. Expect roughly 2× over scalar.

---

## Phase 4 — Row-sum caching (sliding window)

### Tasks

1. Ring buffer of 5 row-sums (3 bitplanes each = 15 bitplane-rows).
2. To produce output row y, ring holds row-sums for input rows y−2..y+2.
3. Advance y by 1: evict the oldest, compute the newest, rotate the indices.
4. At width 32768 the ring is ~60 KiB — tight for the 64 KiB L1d. **Measure** L1 miss rates with `perf stat -e L1-dcache-load-misses` after wiring it up. If the ring spills, we tile by columns in Phase 5.

### Acceptance criteria

- `verify.sh` PASS on 512, 2048, 8192 (all four pattern variants per size).
- Timing recorded. Vertical pass should be roughly 3–5× faster than Phase 3.
- Peak RSS for 8192 recorded.
- `perf stat` output for the 8192 run captured in `NOTES.md`.

---

## Phase 5 — Column tiling and first 32768 attempt

### Memory check first — do this before any tiling code

Compute and record in `NOTES.md` the expected peak RSS for 32768:
- Input bytes: 1 GiB (loaded then freed before simulation — **confirm we free it**)
- Bitplanes: 2 planes × 2 buffers × 128 MiB = 512 MiB
- Row-sum ring: < 1 MiB
- Output bytes during write-back: 1 GiB
- Claude Code's own resident set: estimate from `ps -o rss= -p $$` before starting

Conservative peak estimate: **~1.5 GiB**, plus Claude Code's footprint. This is on the edge.

Mitigations to apply in order:
- Free the input byte buffer immediately after `bytes_to_bitplanes` returns.
- Allocate the output byte buffer **just before** `bitplanes_to_bytes`, after the simulation completes.
- If still tight, stream output to disk in chunks during the inverse transpose rather than holding the whole byte buffer in memory.

If after these you still can't fit 32768 here, **skip it on this box, document the memory math, and defer to the 8-core instance.** Do not attempt to oversubscribe RAM.

### Tasks

1. Process the grid in column tiles of width W. For each tile, walk all rows top to bottom maintaining the row-sum ring for that tile only.
2. Tile boundaries need 2 columns of overlap on each side, drawn from neighbouring tiles or the toroidal wrap at the left/right edges.
3. Sweep `W ∈ {1024, 2048, 4096, 8192, full-width}` on the 8192 grid. Measure each. Pick the best. Record the full sweep in `NOTES.md`.
4. Attempt 32768 only after the sweep on 8192 has chosen a W. Run on `public_1_random_low_32768.bin` and verify. Watch `free -h` in another shell if you can.

### Acceptance criteria

- `verify.sh` PASS on 512, 2048, 8192 with the chosen tile width.
- 32768: either PASS, or a documented "deferred — peak RSS would have been X" entry in `NOTES.md`.
- Chosen W justified in `NOTES.md` with a paragraph referencing L1/L2 working-set math and the measured sweep.

---

## Phase 6 — Benchmark methodology

### Tasks

1. `bench/bench.sh`:
   - Try `sudo cpupower frequency-set -g performance 2>/dev/null || echo "(governor not set, may have variance)"` — likely no root; document.
   - Run the binary N times (default 20) on a given input, discard first 3 as warmup.
   - Report median, p10, p90, min, max in ms.
2. `bench/perf.sh`: wrap a single run in `perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses`, write output to a log.
3. Run on 512, 2048, 8192 (and 32768 if it fit in Phase 5). Record all numbers in `NOTES.md`.

### Acceptance criteria

- `bench/bench.sh` produces median + percentiles for at least one size.
- Variance `(p90 − p10) / median` under ~5% on at least one size, or documented why not.
- `perf stat` output for at least one size captured.

---

## Phase 7 — Threading (DEFERRED, do not run here)

This runs on the 8-core box, not this one. Documented so the hooks are in place.

### Plan (for the 8-core box)

1. Decompose the grid into 8 horizontal strips, one per thread.
2. Each thread owns `[row_begin, row_end)` and produces output for those rows.
3. Each thread reads 2 rows of ghost data above and below its strip from neighbouring strips (or the toroidal wrap) at the start of each generation.
4. `std::thread` spawned once at startup, pinned with `pthread_setaffinity_np`. `std::barrier` between generations.
5. Pad strip boundaries to 64-byte cache-line alignment to avoid false sharing.

### Hooks you must put in place now (Phases 2–5)

- All kernel functions take `(row_begin, row_end, ...)` and the single-threaded path calls `kernel(0, height, ...)`.
- No globals. All state owned by a per-thread context struct, even if there's only one "thread" today.

---

## Phase 8 — Stretch: multi-generation temporal tiling

**Do not attempt on this box.** Only attempt on the 8-core box after Phase 7 lands and you've measured that the implementation is DRAM-bandwidth-bound. Skip otherwise.

### Sketch

- Process a tile through K generations before moving to the next tile.
- The dependency cone shrinks by 2 cells per side per generation (5×5 stencil), so a tile of size T yields valid output of size T − 2K after K generations.
- Trade tile efficiency for DRAM savings. The risk is boundary corruption — do not attempt without a known-good single-gen impl to diff against.

---

## Failure protocol

If at any phase `verify.sh` fails:

1. **Stop. Do not work around it.**
2. `cmp -l /tmp/out.bin EXPECTED | head -20` — first 20 differing byte offsets.
3. Convert offsets to (row, col): `row = offset / (width); col = offset % width`. Report.
4. Bisect by generation count. Add a hack to write intermediate state every 1000 gens; find the earliest divergent generation. Report that number.
5. Most likely causes, in order of likelihood:
   - Toroidal boundary handling at rows 0, 1, height−2, height−1 or symmetrically for columns.
   - Centre-cell subtraction sign / borrow propagation.
   - A misaligned NEON load (alignment fault would crash, but unaligned can silently read wrong bytes if `vld1q_u8` is used instead of `vld1q_u64` on a non-16-byte-aligned address — be precise).
   - A half-adder or full-adder formula typo (especially the carry expression).
6. Report findings to Anurag with the bisected generation number, the (row, col) of the first divergence, and your hypothesis. Wait for direction.

## Things that are not your job

- Writing prose for the design doc. You write factual notes in `NOTES.md`; Anurag writes the doc.
- Choosing algorithmic alternatives. The design at the top is fixed.
- Threading on this box.
- Anything on the 32768 grid that would exceed available RAM.

If you find yourself wanting to do any of these, stop and ask.
