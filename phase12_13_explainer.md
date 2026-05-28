# Phases 12 & 13: Closing the Gap

## TL;DR

| Configuration | Time (32K, 8 threads, NEON) |
|---|---|
| Phase 11 baseline | ~123.5 s |
| After Phase 12 (predicate refactor)   | ~121.2 s |
| After Phase 13 (boundary peel)        | **~120.06 s** |
| Combined improvement                  | **−2.8 %** |

Other teams were reportedly at ~120 s. **Gap effectively closed.**

Two small, tightly-related changes — neither alters what is computed, only *how*
the compiler renders it. They were chosen by reading the `perf annotate` profile of
the Phase 11 build (the artifacts `perf_p11.data` / `ann_p11.txt` / `stat_p11.txt`
sit in the repo root) rather than guessing.

---

## Background you'll want for the panel

### What the kernel computes (one paragraph)

Each cell is one of EMPTY / EGG / JUVENILE / ADULT, stored as two bitplanes
`s1`, `s0` (1 bit per cell, 64 cells per `uint64`; `ADULT = s1 & s0`). Per
generation, a cell's fate depends on **A** = ADULT-count in its 5 × 5
neighbourhood (post-Phase-11, including itself — `A_full`):

- `born`     fires on EMPTYs with `A_full ∈ {3,4,5}`     → EMPTY becomes EGG
- `survives` fires on ADULTs with `A_full ∈ {5..10}`     → ADULT stays ADULT
- EGG → JUVENILE and JUVENILE → ADULT unconditionally

The NEON kernel processes 128 cells per `uint64x2_t` register. A persistent
5-bit accumulator `C` per column holds `A_full` and slides one row at a time
via `c5_sub3_neon` (subtract the leaving row's contribution) + `c5_add3_neon`
(add the entering row's). The inner loop is x2-unrolled — two columns
processed together so their independent carry chains overlap.

### Where Phase 11 left us, and what profiling said

Phase 11 (the centre-subtract elimination) had brought us to ~123 s. Profiling
the 123 s build with `perf` revealed:

- **IPC 3.41** on Neoverse-V2 (theoretical SIMD-ALU ceiling is 4 across 4 pipes).
- **`stall_backend_mem` only ~5.7 %** of cycles → memory is *not* the bottleneck.
- **52 % of cycles in pure boolean SIMD ops** (`vand`, `veor`, `vorr`, `vnot`,
  `vbic`) → throughput-bound on the boolean ALU.
- Two specific lines stood out:
  - `arm_neon.h:17537` (the `mvn` instruction, i.e. `vnot64`) at **6.3 %** of cycles.
  - `kernel_neon.cpp:210` (a `ldr q,[sp]` — a per-iteration **stack reload**) at **2.1 %**.

Both pointed at the same underlying issue: **register pressure**. The kernel
runs out of registers in the x2 unrolled body, the compiler has to either
recompute or reload values, and that costs cycles.

We also confirmed register pressure by testing the obvious counter-move (more
parallelism via x3 unroll): the third interleaved chain spilled 22 q-registers
inside the hot loop and regressed by 6 %. So the right strategy was to
*reduce* per-position register live-ranges, not *add* parallelism.

---

## Phase 12 — Predicate refactor with `vbicq` / `vornq`

### The problem

The predicate block as of Phase 11 began with **five `vnot` ("complement")
operations** per position, one for each bit of `C`:

```cpp
const uint64x2_t nc4 = vnot64(c4_0), nc3 = vnot64(c3_0);
const uint64x2_t nc2 = vnot64(c2_0), nc1 = vnot64(c1_0), nc0 = vnot64(c0_0);
// ...then use them in born / survives:
const uint64x2_t surv_lo = vandq_u64(vandq_u64(nc3, c2_0), vorrq_u64(c1_0, c0_0));
const uint64x2_t surv_hi = vandq_u64(vandq_u64(c3_0,  nc2), vorrq_u64(nc1,  nc0));
const uint64x2_t survives = vandq_u64(nc4, vorrq_u64(surv_lo, surv_hi));
// ...etc.
```

Each `nc_i` costs (i) one `mvn` instruction to compute and (ii) **one register**
that has to be kept alive from the moment it's computed until its last use. Two
positions in the x2 body = **10 of these "complement registers" live
simultaneously** on top of the 10 `c_i` registers and all the row-sum and store
temporaries. That's the register-pressure problem we measured.

### The two AArch64 intrinsics we hadn't used

AArch64 NEON has *fused* logical-with-NOT instructions, one machine
instruction each:

- **`vbicq_u64(a, b)`** computes `a & ~b` in a single op.
- **`vornq_u64(a, b)`** computes `a | ~b` in a single op.

Every place the old code computed `nc_i = ~c_i` and then used it in `vandq` or
`vorrq`, that's a `vbicq` / `vornq` waiting to happen — *at the use site*,
without ever materialising `nc_i` in a register.

### The transformation, walked through

Take `surv_lo`: `~c3 & c2 & (c1 | c0)`.
- Old: precompute `nc3 = ~c3`, then `(nc3 & c2) & (c1 | c0)`.
- New: `vbicq(c2, c3)` *is* `c2 & ~c3` directly. No `nc3` needed.

Take `surv_hi`: `c3 & ~c2 & (~c1 | ~c0)`.
- Old: precompute `nc2`, `nc1`, `nc0`; combine.
- New (two steps): `vbicq(c3, c2)` for `c3 & ~c2`, and *De Morgan* on the
  right factor: `~c1 | ~c0` ≡ `~(c1 & c0)`. So `surv_hi = vbicq(c3, c2) &
  vnot(c1 & c0)`. `nc1` and `nc0` are gone.

Take `born`: `~c4 & ~c3 & (c2 ^ c1) & (~c1 | c0)`.
- The `~c4 & ~c3` pair becomes `~(c4 | c3)` (De Morgan): one `vorrq` + one `vnot`
  (2 ops total) replaces two `vnot`s + one `vandq` (3 ops total).
- The `~c1 | c0` factor becomes `vornq(c0, c1)` directly.
- Result: `born = ~(c4|c3) & (c2 ^ c1) & vornq(c0, c1)`. No precomputed `nc`s
  needed.

Take `survives`: `~c4 & (surv_lo | surv_hi)`.
- Becomes `vbicq(surv_lo | surv_hi, c4)`. No `nc4`.

After all six substitutions, **all five `nc4..nc0` complements are gone.** The
formulas are algebraically identical to the Phase 11 versions — verified
mechanically (`vbicq`/`vornq` are exact substitutions of their definitions) and
empirically (see Correctness, below).

### Op-count accounting (per position)

|                       | Phase 11 | Phase 12 |
|-----------------------|---------:|---------:|
| `mvn` (NOT)           |        6 |        3 |
| `vand`                |       11 |        8 |
| `vorr`                |        7 |        6 |
| `vbic` (fused AND-NOT)|        0 |        3 |
| `vorn` (fused OR-NOT) |        0 |        1 |
| `veor`                |        2 |        2 |
| **Total ops/position**|   **26** |   **23** |

Three operations per position × two positions × three predicate sites = **18
fewer ops per unrolled iteration**, on a body that totals ~168 ops. And —
critically — **five fewer register live-ranges per position**.

### What actually happened in the binary

`objdump -d`, before vs after, inside `kernel_neon`:

| Static instruction | Phase 11 | Phase 12 |
|---|---:|---:|
| `mvn`  | 18 | 9 |
| `bic`  | 18 (from `c5_sub3` only) | 27 (+ 9 in predicates) |
| `orn`  | 0  | 3  |

Dynamically (`ann_p12.txt`):

- The `mvn` share **fell from 6.34 % to 2.52 %** of cycles — exactly the predicted
  ~3.8 percentage points. That's where the wall-clock saving came from.

### Surprise: a new stack reload appeared

Even though Phase 12 freed five register live-ranges per position, the
compiler **made different allocation choices** with the new instruction mix and
introduced one new per-iteration stack reload in the hot loop (`ldr q9, [sp]`
at the `(vi+2 == tnw) ? bnd_next : ...` ternary at line 209). It costs ~1.47 %
of cycles. The net gain (mvn savings minus the new reload) still came out to
+1.9 % wall-clock. But it set up Phase 13.

### Phase 12 wall-clock result

| | 32K T=8 | CV |
|---|---|---|
| Phase 11 baseline | 123.5 s | (single run) |
| Phase 12          | **121.2 s** | 0.1 % (5 idle-box runs) |

---

## Phase 13 — Peeling the boundary iteration

### Where the spill came from

Inside the x2 unrolled loop, the right-edge cell `adult_next_1` wraps around
on the *last* iteration of each row (toroidal grid). The Phase 11 code
expressed this as a conditional inside the loop:

```cpp
for (; vi + 2 <= tnw; vi += 2) {
    const uint64x2_t adult_next_1 = (vi + 2 == tnw)
        ? bnd_next                                          // last iter: wrap
        : vandq_u64(vld1q_u64(np1 + ws + (vi + 2) * 2),     // normal: real load
                    vld1q_u64(np0 + ws + (vi + 2) * 2));
    /* ... 80 lines of body ... */
}
```

`bnd_next` is computed once per row (it's the boundary value), then *only used
on one iteration*. But the compiler — which can't know that `vi+2 == tnw`
fires exactly once per row — generates code that **speculatively loads
`bnd_next` from `[sp]` every iteration**, then either keeps it or overwrites
it with the real load depending on the condition.

That speculative reload is the 1.47 % we measured at `kernel_neon.cpp:210` in
`ann_p12.txt`.

### The fix: peel the boundary iteration

If the boundary iteration is handled *outside* the loop, the ternary
disappears from the loop body, and the in-loop `adult_next_1` is an
unconditional real load. `bnd_next` is no longer referenced inside the loop at
all — the compiler keeps it in a register until the peeled block uses it.

Loop condition changed from `vi + 2 <= tnw` to `vi + 2 < tnw`:

```cpp
size_t vi = 0;
for (; vi + 2 < tnw; vi += 2) {
    // adult_next_1 is unconditional — never the boundary case here.
    const uint64x2_t adult_next_1 =
        vandq_u64(vld1q_u64(np1 + ws + (vi + 2) * 2),
                  vld1q_u64(np0 + ws + (vi + 2) * 2));
    /* ... body ... */
}

// Peeled iteration: runs only when tnw is even (so vi+2 == tnw remains true here).
if (vi + 2 == tnw) {
    /* identical body, but with adult_next_1 = bnd_next directly */
    vi += 2;
}

// Single-pair tail (unchanged): runs only when tnw is odd.
if (vi < tnw) { /* ... */ }
```

### Correctness reasoning by `tnw` parity

I traced every case to convince myself coverage was preserved exactly:

| `tnw` | Main loop iters | Peel fires? | Tail fires? | Pairs covered |
|------:|----------------:|:-----------:|:-----------:|--------------:|
| 256 (production 32 K) | 127 | yes (once) | no  | 128 ✓ |
| 16 (production 2 K)   |   7 | yes (once) | no  |   8 ✓ |
| 8                     |   3 | yes (once) | no  |   4 ✓ |
| 5 (odd)               |   2 | no         | yes (once) |   3 ✓ |
| 3 (odd)               |   1 | no         | yes (once) |   2 ✓ |
| 2                     |   0 | yes (once) | no  |   1 ✓ |
| 1                     |   0 | no         | yes (once) |   1 ✓ |
| 0                     |   0 | no         | no  |   0 ✓ |

In production (`--tile-cols=0`), `tnw` is always even (`32 768 / 64 / 2 =
256`), so the peel always fires exactly once per row. The `--tile-cols` flag
forces odd `tnw` in our regression tests but isn't used in the graded run.

### What the binary looks like now

`objdump` of the production binary:

| | Phase 12 | Phase 13 |
|---|---:|---:|
| `q`-register stack ops *inside the x2 hot loop* | 1 | **0** |
| Stack ops outside the hot loop (one-time save/restore) | — | 4 (negligible: O(rows), not O(iters)) |

The reload at `:210` is gone.

### Phase 13 wall-clock result

| | 32K T=8 | CV |
|---|---|---|
| Phase 12          | 121.2 s | 0.1 % |
| Phase 13          | **120.06 s** | 0.1 % (5 idle-box runs) |

1.47 % profile share converted to 0.9 % wall-clock — typical erosion (some of
the cycles the reload had cost were already being hidden by the out-of-order
engine, so not every recovered cycle is observable in wall time, but most are).

---

## Correctness (both phases together)

- **Unit tests**: `test_kernel_neon` (which includes a NEON-vs-scalar cross-check on
  random data, 10 generations) and `test_kernel_scalar` both `ALL PASS`. The
  scalar kernel was unchanged through Phases 12 and 13, so the NEON-vs-scalar
  match is independent evidence that the NEON refactor preserves behaviour.

- **All 5 public 2048 grids match `.expected.bin` at 10 000 generations** (the
  authoritative byte-for-byte reference) under both Phase 12 and Phase 13.

- **Tail-path stress**: ran with `--tile-cols=256` (forces `tnw=2`, peel fires
  alone), `--tile-cols=384` (`tnw=3`, peel skipped, tail handles one pair),
  `--tile-cols=640` (`tnw=5`, two main iters + one tail), `--tile-cols=512`
  (`tnw=4`, one main + one peel), `--tile-cols=1024` (`tnw=8`). All `PASS`.

- **Pre-existing kernel limitation**: `--tile-cols` values that produce
  *odd* `tile_words` (e.g. 64, 832) leave one word of each tile unprocessed
  because the kernel's `tnw = tw/2` floors the count. This pre-dates Phases
  12 and 13 (confirmed by rebuilding the HEAD-baseline binary), is not
  exercised by the graded path (default `--tile-cols=0`), and is documented
  in `HANDOFF.md` rather than fixed here.

---

## Things we tried this session that didn't work

These are documented in detail under "Tried and rejected" in `HANDOFF.md`. The
short versions, for the panel:

### x3 unroll (attempted, reverted)

The natural reaction to a register-pressure finding is "give the compiler
more parallelism — 3 interleaved chains instead of 2." Phase 11 had freed
the five `e0..e4` snapshot registers, so we expected x3 to fit. It didn't:
the compiler spilled **22 q-registers** inside the hot loop and the build
**regressed by 6 %** (130.9 s vs 123.5 s). The kernel is *register-bound* at
x2 — adding parallelism makes things worse, not better, until per-position
register pressure comes down first.

This is also why x4 had been rejected previously — the same wall, just hit
earlier.

### Delta-fusion of `sub3` + `add3` (analysed, not implemented)

Each row of the kernel does `C = C - old_row + new_row` as two depth-5 ripple
chains (`c5_sub3` then `c5_add3`). The tempting move is to compute `delta =
new_row - old_row` once, then `C += delta` as a single chain — looks like a
big saving on paper. Working through the bit-level arithmetic showed the
3-bit signed subtract needed for `delta` costs roughly as much as one of the
two chains it would replace (≈ 12–15 ops). And the loop-carried-on-`C`
latency it would shorten is already fully hidden by the ~256-deep recurrence
across independent C columns. So the change is **break-even at best with real
correctness risk** — we skipped it.

---

## Where we stand and what's left

We landed at ~120 s, matching the reported other-team baseline. The current
profile says:

- The kernel remains **compute-bound on the boolean SIMD ALU** (~52 % of cycles
  in boolean ops); memory is still a non-issue (~5.7 % of cycles in memory
  stalls).
- After Phase 13, the **`c5_sub3` / `c5_add3` carry chains** become the
  dominant cluster (~30 %+ of cycles around the chain opcodes).
- The two obvious op-count-reduction techniques for carry chains
  (**Wallace-tree** rebuild, **Kogge-Stone** parallel-prefix) are in the
  "Tried and rejected" list for documented reasons: Wallace-tree ~doubles
  instruction count to shorten a latency that is already hidden by the
  column-recurrence distance, and Kogge-Stone doubles op count to cut depth
  in a kernel that is already throughput-bound, not latency-bound.

My honest read of the data is that **we're near the practical floor for this
bitwise-bitplane representation.** A meaningfully faster kernel would likely
need to change *how it counts neighbours* (different representation), not
shave further at the boolean level. That's a research question rather than a
phase.

If continuing: **re-profile the 120 s build first** (the peel shifted
instruction layout; the new hottest lines are not predictable from the Phase
12 annotate), then consider re-attempting x3 unroll now that `bnd_next` no
longer takes a stack slot. The script `bench/profile_p12.sh` can be adapted
to a `p13` variant.

---

## Files touched

- `src/kernel_neon.cpp` — predicate blocks rewritten with `vbicq` / `vornq`
  (Phase 12); main loop peeled and a duplicated boundary block added before
  the existing single-pair tail (Phase 13).

No other files changed. The scalar kernel, the row-sum, the C-init loop, the
ring buffer, and the multithreading driver are all untouched.
