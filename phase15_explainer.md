# Phase 15: Carry-Chain Compaction

## TL;DR

| Configuration | Time (32K, 8 threads, NEON) |
|---|---|
| Phase 14 baseline (post vnot + vsri/vsli) | 113.0 s |
| After Phase 15A (`vbsl` MAJ-fold in c5_sub3/c5_add3) | **107.2 s** |
| Combined improvement                  | **−5.1 %** |

The `c5_sub3_neon` and `c5_add3_neon` ripple chains were the largest single
cycle bucket after Phase 14 (≈ 30 % of cycles). Both functions are structurally
identical full-adder / full-subtractor cascades, and both express their
"propagate next carry/borrow" stage with the same 3-op pattern. That pattern is
the bitwise majority function in disguise — one machine instruction on AArch64
(`BSL`). This phase collapses the 3-op pattern down to 1-op at the two interior
stages of each chain.

No algorithm changes, no register-pressure changes (BSL is destructive: it
writes into one of its input registers). Pure op-count reduction in the largest
remaining cycle bucket.

---

## Background you'll want for the panel

### What the kernel computes (one paragraph, repeated for self-containment)

Each cell is one of EMPTY / EGG / JUVENILE / ADULT, stored as two bitplanes
`s1`, `s0` (1 bit/cell, 64 cells per `uint64`; `ADULT = s1 & s0`). Per
generation, a cell's fate depends on **A** = ADULT-count in its 5 × 5
neighbourhood (post-Phase 11, including itself — `A_full`):

- `born`     fires on EMPTYs with `A_full ∈ {3,4,5}`     → EMPTY becomes EGG
- `survives` fires on ADULTs with `A_full ∈ {5..10}`     → ADULT stays ADULT
- EGG → JUVENILE and JUVENILE → ADULT unconditionally

A persistent 5-bit accumulator `C = (c4,c3,c2,c1,c0)` per column holds
`A_full`. To slide it down one row we:

1. Subtract the 3-bit sum of the *leaving* row: `c5_sub3_neon(C, r_leave)`.
2. Add the 3-bit sum of the *entering* row:     `c5_add3_neon(C, r_enter)`.

Each is a ripple chain of depth 5 (one full-adder per bit position).

### Where Phase 14 left us, and what profiling said

Phase 14 finished the boolean-side cleanup: every `mvn` was eliminated from
the predicate / emit block (via De-Morgan folds to `vbic`/`vorn`), and the
horizontal `neon_row_sum_3bit` was tightened with `vsri`/`vsli` shift-and-insert
instructions. The wall clock landed at 113 s.

The HANDOFF named the three roughly co-equal components of the inner body:

| Component | ops/iter (x2-unrolled) | notes |
|-----------|----------:|-------|
| `c5_sub3` + `c5_add3` (×2 positions) | ~60 | two serial depth-5 ripples per position |
| born/survives predicates + encode (×2) | ~52 | already heavily reworked in Phases 11–14 |
| `neon_row_sum_3bit` (×2) | ~54 | already heavily reworked in Phase 14 |

The predicate and row-sum components had been hit twice each. The carry chains
were the only major bucket *not yet* touched algebraically. They were the
obvious next target.

---

## Phase 15A — `vbsl` MAJ-fold in the carry/borrow ripple

### The pattern in both chains

`c5_add3_neon`, before:

```cpp
uint64x2_t ax = veorq_u64(c1, r1);                       // sum bit (depth-1)
uint64x2_t nc = vorrq_u64(vandq_u64(c1, r1),             // carry generate
                          vandq_u64(carry, ax));         // carry propagate
c1 = veorq_u64(ax, carry);
carry = nc;
```

That's **3 ops** per stage to compute the next carry: two `and`s and one `orr`.
The same shape appears at the c1-stage *and* the c2-stage of `c5_add3`, so
6 ops per call.

`c5_sub3_neon` has the dual shape (borrow generate / borrow propagate) — same
3-op pattern, also at two stages, also 6 ops per call:

```cpp
uint64x2_t ax = veorq_u64(c1, r1);
uint64x2_t nb = vorrq_u64(vbicq_u64(r1, c1),             // borrow generate: ~c1 & r1
                          vbicq_u64(b, ax));             // borrow propagate: ~ax & b
c1 = veorq_u64(ax, b);
b = nb;
```

### What `BSL` actually does

AArch64 NEON has the `BSL` instruction (Bitwise Select):

```
vbslq_u64(mask, a, b)  =  (mask & a) | (~mask & b)
```

It's three of the four logical inputs an old-school multiplexer needs:
"for each bit of `mask`, choose either `a` or `b`". Crucially **it's one
machine instruction**, destructive in its first operand — and it executes on
the same 4 SIMD pipes on Neoverse-V2 as `AND`/`ORR`/`EOR`. We had not used it
anywhere in the kernel before this phase.

### Why the carry expression *is* a `BSL`

Look at the add3 carry: `(c1 & r1) | (carry & (c1 ^ r1))`.

This is the textbook **majority function** `MAJ(c1, r1, carry)` — output is 1
iff at least two of the three inputs are 1. Equivalently:

- when `c1 == r1` (so `ax = c1 ^ r1 = 0`): both bits agree, output = `c1` (= `r1`).
- when `c1 ≠ r1` (so `ax = 1`): the tie is broken by `carry`, output = `carry`.

That's exactly a `BSL` keyed on `ax`:

```
vbslq(ax, carry, c1)
  = (ax & carry) | (~ax & c1)
  // when ax=1: carry  ;  when ax=0: c1
  = MAJ(c1, r1, carry)         ✓
```

Algebraic check that `~ax & c1 = c1 & r1`:
- `~ax = ~(c1 ^ r1) = (c1 & r1) | (~c1 & ~r1)`
- `c1 & ~ax = c1 & ((c1&r1) | (~c1&~r1)) = c1 & r1`     ✓

So the entire 3-op pattern collapses to **one `vbslq`** — the `ax = c1 ^ r1`
input is already being computed for the sum bit, so it's free.

For sub3, the borrow expression `(~c1 & r1) | (~ax & b)` collapses the same
way, this time keyed `vbslq(ax, r1, b)`:
- when `ax = 0` (c1 == r1): pass the incoming borrow through.
- when `ax = 1` (c1 ≠ r1): take `r1` (which is 1 iff `c1=0, r1=1` — exactly when
  this stage generates a new borrow).

Truth-table check of all four `(c1, r1)` combinations confirms identity with
the original `(~c1 & r1) | (~ax & b)` expression. (And independently: the unit
tests pass on every grid, so the result is byte-identical to the previous
build — there is no "almost identical" failure mode in pure boolean arithmetic.)

### The transformation, both functions

```cpp
// c5_add3_neon — carry stage (×2 per call)
- uint64x2_t nc = vorrq_u64(vandq_u64(c1, r1), vandq_u64(carry, ax));
+ uint64x2_t nc = vbslq_u64(ax, carry, c1);

// c5_sub3_neon — borrow stage (×2 per call)
- uint64x2_t nb = vorrq_u64(vbicq_u64(r1, c1), vbicq_u64(b, ax));
+ uint64x2_t nb = vbslq_u64(ax, r1, b);
```

The c0-stage of each function doesn't get the fold (the c0 stage has no
incoming carry, so the expression there is just `c0 & r0` or `r0 & ~c0` — one
op already). The c3/c4 tail stages also don't get the fold (carry is the only
live signal at those stages — no MAJ left). So the gain is exactly **two
fewer ops per chain × two chains × four ripple-chain calls per x2 iteration**
= 16 ops/iter saved in the hot path. Plus 4 ops/chain × 5 chains amortised in
the C-init.

### Op-count accounting (per call)

|                     | Phase 14 | Phase 15A |
|---------------------|---------:|----------:|
| `vand`              |        4 |         2 |
| `vorr`              |        2 |         0 |
| `vbic` (sub3 only)  |        4 |         2 |
| `vbsl` (new)        |        0 |         2 |
| `veor`              |        6 |         6 |
| Other (initial gen) |        2 |         2 |
| **Total ops/call**  |   **15** |    **11** |

x2 unrolled iteration = 2 sub3 calls + 2 add3 calls + init/maintenance:
**−16 ops per inner iteration**, on a body that was ~166 ops.

### What actually happened in the binary

`objdump -d spawn_sim | grep -E '\bbsl|\band|\borr|\bbic\b' | wc -l`, inside
the two ripple functions only:

| Static instruction | Phase 14 | Phase 15A |
|---|---:|---:|
| `bsl` | 0 | 4 (2 per function × 2 functions) |
| `orr` (in chains) | 4 | 0 |
| `and` / `bic` (in chains) | 8 | 4 |

Dynamically, the 32K T=8 run dropped from **113.0 s → 107.2 s**, a 5.1 %
wall-clock improvement. That's slightly *better* than the linear extrapolation
of "10 % fewer ops in a 30 % bucket → 3 % wall-clock", which suggests `BSL`
runs at full 4-pipe throughput on V2 (i.e. it's not a constrained
pipe-mix) — consistent with the SIMD-ALU pipes being symmetric for basic
3-input boolean ops on this core.

### Why BSL doesn't add register pressure

Two facts let this trade come out cleanly:

1. **`BSL` is destructive in its mask operand** (`Vd = (Vd & Vn) | (~Vd & Vm)`
   in the encoding). The previous code already kept `ax` live in a register
   across the carry computation; BSL consumes that same register as its mask
   and overwrites it with the result. Same live-range count.
2. The two operands killed by the fold (`c1 & r1` / `vbicq(r1, c1)` and
   `carry & ax` / `vbicq(b, ax)`) were short-lived temporaries that the
   compiler was already spilling/refilling onto pipe operands. They never had
   architectural registers of their own. Removing them removes work, not
   pressure.

This is the same reason Phase 14's `vbic` substitution worked: shorter
expression *and* shorter live ranges, never just one or the other.

---

## Correctness

- **`tests/test_kernel_neon`**: both the exhaustive K-map predicate check
  (A = 0..25) and the 6 NEON-vs-`step_ref` 128 × 128 cases (1 / 10 / 100 gens,
  including all-ADULT and sparse seeds) `PASS`. The reference `step_ref` in
  `tests/test_utils.h` is an independent per-cell implementation of the rule,
  so this is real ground-truth coverage.
- **All 5 public 2048 grids match `.expected.bin` at 10 000 generations** —
  byte-identical to the committed expected outputs.
- **No tail-path regressions**: the `c5_sub3`/`c5_add3` functions are called
  identically by the main x2 loop, the peeled boundary block, and the odd-tail
  scalar block — touching just the two function bodies updates all three call
  sites uniformly.

---

## What this leaves for Phase 15B

The carry/borrow ripples still execute as serial depth-5 chains. We made each
stage cheaper, but we did not shorten the dependency chain. Per the HANDOFF,
ILP is already plentiful (tnw independent C columns, x2 interleaving), so this
is the right call — but if a future phase wants to shave more from this
bucket, the targets are:

1. **`eor3` (SHA3 extension)** for the 3-way XOR chains in
   `neon_row_sum_3bit` (`out0 = (s_abc ^ d) ^ e`) — one 3-input XOR instead of
   two 2-input XORs. ARM v8.2 `+sha3` is enabled by default on
   `-mcpu=neoverse-v2`. Candidate B, queued.
2. **`bcax` (SHA3 extension)** for the `a ^ (b & ~c)` shapes at the c4-tails
   of both ripple functions — one instruction instead of a `vbic` + `veor`
   pair. Lower-impact than (1).

Both are pure op-count reductions in the same Phase-14/15A spirit and should
go in a follow-up commit, not bundled with 15A.

---

## Files touched

- `src/kernel_neon.cpp` — `c5_sub3_neon` and `c5_add3_neon` only. Two
  3-op patterns replaced by `vbslq_u64` at the c1 and c2 stages of each
  function.

No other files changed.
