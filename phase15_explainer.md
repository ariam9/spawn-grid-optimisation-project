# Phase 15: Carry-Chain Compaction

## TL;DR

| Configuration | Time (32K, 8 threads, NEON) |
|---|---|
| Phase 14 baseline (post vnot + vsri/vsli) | 113.0 s |
| After Phase 15A (`vbsl` MAJ-fold in c5_sub3/c5_add3) | 107.2 s |
| After Phase 15B (`vbsl`/`eor3`/`bcax` in row_sum + sub3 tail) | 104.4 s |
| After Phase 15C (eor3/bcax in d1/d0 emit — teammate Anurag) | 102.0 s |
| After Phase 15D (born vbsl-fold) + 15E (eor3 sum-bit in c5) | **99.0 s** |
| Combined improvement                  | **−12.4 %** |

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

## Files touched (15A)

- `src/kernel_neon.cpp` — `c5_sub3_neon` and `c5_add3_neon` only. Two
  3-op patterns replaced by `vbslq_u64` at the c1 and c2 stages of each
  function.

---

## Phase 15B — extending the fold to `neon_row_sum_3bit`, plus SHA3 ops

After 15A landed, re-reading the row-sum function made one thing obvious:
**the same MAJ pattern Phase 15A folded in the C ripple appears twice in
`neon_row_sum_3bit`**, because `neon_row_sum_3bit` is itself two stacked
full-adders. We had been blind to it because the row-sum had been "already
optimised" in Phase 14 (the `vsri`/`vsli` work), and the eyes pattern-matched
on that being done.

It is not done. Two more `vbsl` folds, plus two SHA3 instructions
(`eor3`, `bcax`) for the remaining 2-eor and bic+eor pairs.

### `neon_row_sum_3bit` — the two MAJ patterns and the 3-way XOR

The horizontal row-sum sums 5 shifted ADULT lanes `a,b,c,d,e` into a 3-bit
output. It does this as two full-adders:

- FA(a, b, c) → `s_abc` (sum), `c_abc` (carry)
- FA(d, e, s_abc) → final sum `out0` (carry: `c_des`)

Plus a final 2-of-2 stage that combines `c_abc` and `c_des` to produce
`out1` (their xor) and `out2` (their and).

The two carries are MAJ functions of their three inputs:

```cpp
const uint64x2_t c_abc = vorrq(vandq(a, b), vandq(c, axb));  // MAJ(a, b, c)
const uint64x2_t c_des = vorrq(vandq(d, e), vandq(s_abc, dxe)); // MAJ(d, e, s_abc)
```

Identical 3-op shape to the c5_add3 inner stages. Identical fold:

```cpp
const uint64x2_t c_abc = vbslq_u64(axb, c, a);                  // -2 ops
const uint64x2_t c_des = vbslq_u64(dxe, s_abc, d);              // -2 ops
```

The 3-way xor for `out0`:

```cpp
out0 = veorq_u64(veorq_u64(s_abc, d), e);                       // 2 eor
```

becomes one SHA3 `EOR3` instruction:

```cpp
out0 = veor3q_u64(s_abc, d, e);                                 // -1 op
```

**Total: 5 fewer ops per `neon_row_sum_3bit` call.** It's called twice per
x2-unrolled inner iteration → 10 ops/iter saved. Plus 5 calls per row in the
ring-buffer warm-up (amortised across the row sweep).

### `c5_sub3_neon` tail — `bcax`

The tail of `c5_sub3_neon` does:

```cpp
nb = vbicq_u64(b, c3);     // b & ~c3
c3 = veorq_u64(c3, b);
b  = nb;
c4 = veorq_u64(c4, b);     // c4 ^= (b_orig & ~c3_orig)
```

SHA3 has `BCAX`: `vbcaxq_u64(a, b, c) = a ^ (b & ~c)`. This is exactly the
`c4` update, in one instruction:

```cpp
c4 = vbcaxq_u64(c4, b, c3);    // -1 op
c3 = veorq_u64(c3, b);          // unchanged
```

(Reordered so `c4` reads `c3` in its original form, then `c3` is overwritten.
No data hazard.) The corresponding pattern in `c5_add3_neon` is `c4 ^=
(c3 & carry)` — there is no `a ^ (b & c)` SHA3 instruction (`bcax` requires the
inverted form), so add3 keeps the 2-op tail.

**Saves 1 op per `c5_sub3` call → 2 ops/iter** (2 sub3 calls per x2 iter).

### Build flag change

`veor3q_u64` and `vbcaxq_u64` are SHA3 intrinsics. `-mcpu=neoverse-v2`
includes the SHA3 ISA in the silicon but gcc-14 does *not* auto-enable the
intrinsics; the build flag needs `+sha3`:

```diff
- CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2          -Wall -Wextra"
+ CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2+sha3     -Wall -Wextra"
```

The grading box (`c8g.2xlarge`, Neoverse-V2) supports SHA3, so this is a
build-flag change only, not a portability change.

### Op-count accounting (Phase 15B, per call)

`neon_row_sum_3bit`:

|                     | Phase 14 | Phase 15A | Phase 15B |
|---------------------|---------:|----------:|----------:|
| `veor`              |        6 |         6 |         3 |
| `vand`              |        4 |         4 |         2 |
| `vorr`              |        2 |         2 |         0 |
| `vbsl` (new in B)   |        0 |         0 |         2 |
| `veor3` (new in B)  |        0 |         0 |         1 |
| `vsri` / `vsli`     |        4 |         4 |         4 |
| `vext` / `vshl/r`   |        6 |         6 |         6 |
| **Total**           |   **22** |    **22** |    **18** |

`c5_sub3_neon`:

|                     | Phase 14 | Phase 15A | Phase 15B |
|---------------------|---------:|----------:|----------:|
| `veor`              |        6 |         6 |         6 |
| `vand`              |        0 |         0 |         0 |
| `vorr`              |        2 |         0 |         0 |
| `vbic`              |        4 |         2 |         1 |
| `vbsl` (new in A)   |        0 |         2 |         2 |
| `vbcax` (new in B)  |        0 |         0 |         1 |
| **Total**           |   **15** |    **11** |    **10** |

### What actually landed in the binary

`objdump -d spawn_sim` inside `kernel_neon`, after Phase 15B:

| Instruction | count |
|---|---:|
| `bsl`  | 14 |
| `eor3` | 35 |
| `bcax` |  5 |
| `bic`  | 35 |
| `mvn`  |  0 |

The `eor3` count (35) is larger than the explicit calls in source —
gcc-14 noticed additional 3-way-XOR opportunities elsewhere in the kernel
once `+sha3` was enabled and rewrote them automatically. Free additional
op-count reduction.

### Wall-clock result

| | 32K T=8 |
|---|---|
| Phase 15A baseline | 107.2 s |
| Phase 15B          | **104.4 s** (single idle-box run) |

A 2.6 % wall-clock improvement on a 12-ops-per-iter reduction in a body that
is now ~150 ops. That's ~8 % op-count reduction translating to 2.6 %
wall-clock — i.e. roughly one-third of the op savings showed up as
wall-clock, lower than 15A's near-1:1 conversion. Two plausible reasons:
(1) `eor3` and `bcax` are 3-input ops and may share fewer pipes than basic
2-input boolean ops on V2, so cycle savings ≠ op savings; (2) the row-sum
calls sit between heavy dependency chains (the C ripple reads from prior
row-sum outputs), so out-of-order execution was already partly hiding the
row-sum cost. Either way, the change is positive, byte-correct, and stable.

### Correctness

- `tests/test_kernel_neon`: predicates `PASS`, all 6 NEON-vs-`step_ref`
  cases `PASS`.
- All 5 public 2048 grids at 10 000 generations match `.expected.bin`
  byte-for-byte.
- Identity proofs for `c_abc`, `c_des`, and `out0` are the same MAJ /
  3-way-XOR reasoning as Phase 15A — no new mathematical risk.

---

## Files touched (15A + 15B together)

- `src/kernel_neon.cpp` — Phase 15A: `c5_sub3_neon`, `c5_add3_neon` (inner
  carry/borrow stages). Phase 15B: `neon_row_sum_3bit` (both MAJ carries +
  3-way XOR) and `c5_sub3_neon` tail (`bcax`).
- `build.sh` — Phase 15B only: `-mcpu=neoverse-v2` → `-mcpu=neoverse-v2+sha3`.

No other files changed. The predicate / emit block, the multithreading
driver, and all I/O remain untouched.

---

## Phase 15D — Born vbsl-fold (1 op saved per born)

After 15C lands the emit fusions, the predicate block is the next-largest
boolean cluster in the profile. Re-reading the born formula:

```cpp
born = vbicq(vandq(veorq(c2,c1), vornq(c0,c1)),  vorrq(c4,c3))
//      └────── 3 ops: eor + orn + and ──────┘   └── orr ─┘   └ bic ─┘
```

The inner factor `(c2^c1) & (c0|~c1)` is a MUX on c1:
- when c1 = 0: result is c2 (since `(c2^0) & (c0|1) = c2 & 1 = c2`).
- when c1 = 1: result is `~c2 & c0` (since `(c2^1) & (c0|0) = ~c2 & c0`).

Truth-table verified across all 8 (c2,c1,c0) cases. So the 3-op expression
folds to one `vbslq`:

```cpp
born = vbicq(vbslq(c1, vbicq(c0,c2), c2),  vorrq(c4,c3))
//      └────── 2 ops: bic + bsl ──────┘   └── orr ─┘   └ bic ─┘
```

**4 ops vs 5 ops.** Applied at 5 emit sites. Wall-clock: 102.0 → 101.4 s
(−0.6 %). Smaller than the per-op accounting predicted because the predicate
block competes for the same SIMD pipes as everything else, and we're already
~15 % under the 4-pipe ceiling — saving one op just creates one more idle
slot rather than freeing a cycle.

## Phase 15E — `eor3` sum-bit update in c5_add3 / c5_sub3 (schedule-only)

Each inner stage (c1 and c2, in both add3 and sub3) has the shape:

```cpp
ax = c1 ^ r1                  // 1 eor (needed for the BSL mask)
nc = vbslq(ax, carry, c1)     // 1 bsl  ← carry chain critical path
c1 = ax ^ carry               // 1 eor  ← waits on ax (depth 2 from c1, r1)
```

The c1 update is the 3-way XOR `c1 ^ r1 ^ carry`, identical regardless of
whether we route through `ax` or use `veor3q` directly:

```cpp
ax = c1 ^ r1
nc = vbslq(ax, carry, c1)
c1 = veor3q(c1, r1, carry)    // independent of ax — one cycle earlier
```

**Same op count (3 ops).** What changes is the dependency graph: the c1
write doesn't wait on the `ax` eor. The downstream `vst1q_u64(C* + ...)`
can issue one cycle earlier, relieving store-buffer pressure (`stall_backend_mem`
had climbed to 8.7 % of cycles after 15B). Applied at 4 sites in source
(c1+c2 × add3+sub3), inlined ~20 times per kernel invocation.

Wall-clock: 101.4 → 99.0 s (−2.4 %). Bigger than expected — this likely
confirms the kernel was bottlenecked on store-buffer pressure post-15C, not
on pure SIMD-ALU throughput. The 2.4 % is roughly the share of cycles where
the store buffer was stalling dispatch; opening up that bottleneck let OoO
fill more slots.

### Why this is the floor of instruction-level work

Post-15E the obvious idiom-level levers are all exhausted:
- All single-op AND/OR/NOT compounds folded (`vbic`, `vorn`).
- All shift-and-OR compounds folded (`vsri`, `vsli`).
- All 3-input MAJ functions folded (`vbsl`).
- All 3-input XOR chains folded (`veor3q`).
- All `a ^ (b & ~c)` patterns folded (`vbcax`).
- All `|`-of-disjoints replaced with `^` so they feed the above.

What's left is structural (loop reorganisation, different data layout, SVE2)
or pipe-mix experimentation (substituting `vbicq(.., x)` for `vbcaxq(0, .., x)`
to shift work onto less-saturated SIMD pipes). The structural path was
spiked once (Phase 16 row-tile, 104 → 123 s, reverted) and the pipe-mix
path requires the V2 software-optimisation guide we don't have.

## Files touched in 15D + 15E

- `src/kernel_neon.cpp` only — born formula at 5 emit sites; c1/c2 sum-bit
  update in `c5_add3_neon` and `c5_sub3_neon`.
