# The x2-Unrolled Inner Loop

This document walks through the hot loop in `kernel_neon.cpp` — the body that
runs ~32K times per row × 32K rows × 10 000 generations and accounts for
essentially all of the runtime at 32 K grids. Everything we do here is in
service of that loop's throughput.

The discussion is in three parts:

1. **What the loop does**, step-by-step.
2. **Why it's structured this way** (x2 unroll, peeled tail, etc.).
3. **The dependency graph**, which explains the choice of ordering.

---

## 1. Anatomy of one iteration

Each iteration processes **two NEON pairs**. One NEON pair is a
`uint64x2_t` = 2 × 64-bit lanes = **128 cells**, so an x2 iteration handles
**256 cells**. The two positions are referred to as suffix `_0` (columns
`vi*2 .. vi*2+1` of the row in u64-word indices) and `_1`
(columns `(vi+1)*2 .. (vi+1)*2+1`).

The loop variable advances by **2** each iteration:

```cpp
size_t vi = 0;
for (; vi + 2 < tnw; vi += 2) {
```

The bound is `vi + 2 < tnw` (strict less-than). This means the loop does
**not** process the last pair when the right-neighbour pair `vi+2` would not
exist as a real load — that case is handled by the **peeled boundary
iteration** further down, so the hot loop never has to test for "are we at
the right edge?" The cost of the test was a speculative reload of `bnd_next`
from the stack every iteration, which the peel cleanly removes.

### Phase A — Load the entering row's ADULT bits

```cpp
const uint64x2_t adult_next_0 =
    vandq_u64(vld1q_u64(np1 + (vi + 1) * 2),
              vld1q_u64(np0 + (vi + 1) * 2));
const uint64x2_t adult_next_1 =
    vandq_u64(vld1q_u64(np1 + (vi + 2) * 2),
              vld1q_u64(np0 + (vi + 2) * 2));
```

`np1` / `np0` point at row `(r + 3)` — the row about to **enter** the 5-row
window. Each NEON pair holds the ADULT bitmask (`s1 & s0`) for one NEON
pair's worth of columns.

We need three consecutive pairs to feed the two `neon_row_sum_3bit` calls:
`(prev, curr, next_0)` for the first call and `(curr, next_0, next_1)` for
the second. `prev` and `curr` are held across iterations as
`adult_prev` / `adult_curr` — only the two `next` pairs need to be loaded
fresh each iteration.

### Phase B — Compute new row-sums

```cpp
neon_row_sum_3bit(adult_prev,  adult_curr,   adult_next_0,
                  new_r2_0, new_r1_0, new_r0_0);
neon_row_sum_3bit(adult_curr,  adult_next_0, adult_next_1,
                  new_r2_1, new_r1_1, new_r0_1);
```

Each call produces a **3-bit horizontal sum** for one NEON pair (= 128
columns). Output is in bitplane form: `new_rN_*` holds bit-N of every
column's 5-cell horizontal sum, packed into a `uint64x2_t`.

Internally `neon_row_sum_3bit` uses a shift-and-insert network (`vsliq` /
`vsriq`) to align 5 shifted copies of the centre row, then a 5-input
full-adder network to reduce them. After Phase 15B that compressed using
SHA3's `veor3q` (3-way XOR) and a `vbslq`-based MAJ fold, the call costs
about **18 NEON ops** for 128 cells.

### Phase C — Load the persistent column-sum bitplanes

```cpp
uint64x2_t c0_0 = vld1q_u64(C0 + vi * 2);     // bitplane 0, position 0
uint64x2_t c1_0 = vld1q_u64(C1 + vi * 2);
uint64x2_t c2_0 = vld1q_u64(C2 + vi * 2);
uint64x2_t c3_0 = vld1q_u64(C3 + vi * 2);
uint64x2_t c4_0 = vld1q_u64(C4 + vi * 2);
uint64x2_t c0_1 = vld1q_u64(C0 + (vi + 1) * 2);   // bitplane 0, position 1
uint64x2_t c1_1 = vld1q_u64(C1 + (vi + 1) * 2);
uint64x2_t c2_1 = vld1q_u64(C2 + (vi + 1) * 2);
uint64x2_t c3_1 = vld1q_u64(C3 + (vi + 1) * 2);
uint64x2_t c4_1 = vld1q_u64(C4 + (vi + 1) * 2);
```

Ten loads — five bitplanes × two positions. `C` holds the **full 5×5
neighborhood count** `A_full ∈ {0..25}`, stored as 5 bitplanes. Loading
both positions' C up front lets the predicate emit and the C-roll
sub/add chains use them right away without forwarding stalls.

`C` lives in `KernelContext::C_store` (heap, ~20 KB total) rather than
in registers, because the kernel doesn't have enough NEON registers to
keep both positions' C live across iterations on top of all the other
state. Reload-per-row is a deliberate trade.

### Phase D — Load the centre-row source bits

```cpp
const uint64x2_t s1w_0 = vld1q_u64(sp1 + vi * 2);
const uint64x2_t s0w_0 = vld1q_u64(sp0 + vi * 2);
const uint64x2_t s1w_1 = vld1q_u64(sp1 + (vi + 1) * 2);
const uint64x2_t s0w_1 = vld1q_u64(sp0 + (vi + 1) * 2);
```

`sp1` / `sp0` point at the **centre row** `r` (the row being emitted).
We need the actual cell states (EMPTY / EGG / JUVENILE / ADULT) — they
gate whether `born` fires (only on EMPTY) and whether `survives` matters
(only on ADULT), and they form most of the output `d0` / `d1` bits.

### Phase E — Predicate and emit (per position)

```cpp
{
    const uint64x2_t born =
        vbicq_u64(vbslq_u64(c1_0, vbicq_u64(c0_0, c2_0), c2_0),
                  vorrq_u64(c4_0, c3_0));
    const uint64x2_t surv_lo =
        vandq_u64(vbicq_u64(c2_0, c3_0), vorrq_u64(c1_0, c0_0));
    const uint64x2_t surv_hi =
        vbicq_u64(vbicq_u64(c3_0, c2_0), vandq_u64(c1_0, c0_0));
    const uint64x2_t survives = vbicq_u64(vorrq_u64(surv_lo, surv_hi), c4_0);
    const uint64x2_t adult_sv = vandq_u64(vandq_u64(s1w_0, s0w_0), survives);
    vst1q_u64(d1 + vi * 2,
        veor3q_u64(s0w_0, s1w_0, adult_sv));
    vst1q_u64(d0 + vi * 2,
        vbcaxq_u64(adult_sv, vorrq_u64(s1w_0, born), s0w_0));
}
```

Each of the two `{ ... }` blocks emits one position. The block is the
densest part of the kernel.

#### Born

`born = A_full ∈ {3, 4, 5}`. The Karnaugh-minimised expression is:

```
born = ~(c4 | c3) & ((c2 ^ c1) & (~c1 | c0))
     = ~(c4 | c3) & (~c1 ? c2 : (c2 & c0))
```

The MUX form is what the `vbslq_u64(c1_0, vbicq_u64(c0_0, c2_0), c2_0)`
does — `vbsl(mask, a, b) = (a & mask) | (b & ~mask)`. The inner
`vbicq(c0, c2) = c0 & ~c2` plus the bsl gives `(c1 ? (c0 & ~c2) : c2)`, and
the outer `vbicq` with `c4 | c3` kills any cell where the high bits are
set (i.e. `A_full ≥ 8`).

#### Survives

`survives = A_full ∈ {5..10}`. Split into two halves:

- `surv_lo` (A_full = 5..7): `c3 = 0, c2 = 1, (c1 | c0)`
- `surv_hi` (A_full = 8..10): `c3 = 1, c2 = 0, ~(c1 & c0)`

Then `survives = (surv_lo | surv_hi) & ~c4` — the outer `& ~c4` rules out
`A_full ≥ 16`. (Survives at exactly `A_full = 5` overlaps with `born`,
which is fine because the cell-state gating ensures only one of `born` /
`survives` is used per cell.)

#### Adult-survives mask

```
adult_sv = (s1 & s0) & survives
```

This is 1 only on ADULT cells that survive — i.e. cells that stay ADULT
into the next generation. Critically, `adult_sv` is bit-disjoint from the
other terms in the d0/d1 expressions, which lets the next two emits use
XOR-based SHA3 instructions instead of OR.

#### d1 fused emit (SHA3 `eor3`)

```cpp
vst1q_u64(d1 + vi * 2,
    veor3q_u64(s0w_0, s1w_0, adult_sv));
```

Algebraically: `d1 = (s0 ^ s1) | adult_sv`. Why is this OR ≡ XOR?

- `(s0 ^ s1) = 1` requires `s0 ≠ s1`, which forces `s1 & s0 = 0`, so
  `adult_sv = 0`.
- `adult_sv = 1` requires `s0 = s1 = 1`, so `s0 ^ s1 = 0`.

The two terms cannot both be 1 in the same bit position — OR and XOR
produce identical results. `eor3` collapses the original `eor + orr`
sequence into a single SHA3 op.

#### d0 fused emit (SHA3 `bcax`)

```cpp
vst1q_u64(d0 + vi * 2,
    vbcaxq_u64(adult_sv, vorrq_u64(s1w_0, born), s0w_0));
```

Algebraically: `d0 = ((s1 | born) & ~s0) | adult_sv`. Same disjointness
argument:

- `(s1 | born) & ~s0` requires `s0 = 0`, so `adult_sv = 0`.
- `adult_sv = 1` requires `s0 = 1`, so the masked expression is 0.

`vbcaxq(a, b, c) = a ^ (b & ~c)`. Mapping `a = adult_sv, b = s1|born,
c = s0` gives exactly `adult_sv ^ ((s1|born) & ~s0)`, which equals the
OR form. One `bcax` replaces the original `orr + bic + orr` triple.

### Phase F — C-roll (rolling the column-sum window)

```cpp
const uint64x2_t old_r0_0 = vld1q_u64(rs0[tail] + vi * 2);
const uint64x2_t old_r1_0 = vld1q_u64(rs1[tail] + vi * 2);
const uint64x2_t old_r2_0 = vld1q_u64(rs2[tail] + vi * 2);
const uint64x2_t old_r0_1 = vld1q_u64(rs0[tail] + (vi + 1) * 2);
const uint64x2_t old_r1_1 = vld1q_u64(rs1[tail] + (vi + 1) * 2);
const uint64x2_t old_r2_1 = vld1q_u64(rs2[tail] + (vi + 1) * 2);

c5_sub3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, old_r0_0, old_r1_0, old_r2_0);
c5_sub3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, old_r0_1, old_r1_1, old_r2_1);
c5_add3_neon(c0_0, c1_0, c2_0, c3_0, c4_0, new_r0_0, new_r1_0, new_r2_0);
c5_add3_neon(c0_1, c1_1, c2_1, c3_1, c4_1, new_r0_1, new_r1_1, new_r2_1);
```

The row-emit is finished, so `C` is free for mutation. We slide the
5-row window by one:

1. Load the **leaving row's** row-sum from slot `tail`. (Slot `tail`
   currently holds row `r − 2`, the row about to fall off the bottom of
   the window.)
2. **Subtract** that 3-bit row-sum from the 5-bit `C` via `c5_sub3_neon`
   (a 5-stage ripple borrow).
3. **Add** the new row's row-sum (from Phase B) via `c5_add3_neon` (a
   5-stage ripple carry).

After this, `C` represents `A_full(r + 1, ·)` — ready for the next row's
emit.

**Why the two positions' sub/add calls are interleaved** in this order
(`sub_0, sub_1, add_0, add_1`) instead of `sub_0, add_0, sub_1, add_1`:

Each `c5_sub3` / `c5_add3` is a ripple chain — a 5-stage
loop-carried dependency on C. The two positions' chains are **fully
independent** (different `c*_0` vs `c*_1` registers). The interleaving
exposes both chains to the OOO scheduler at the same time, doubling
ILP and roughly halving the chain's critical-path latency contribution.

### Phase G — Stores (10 + 6)

```cpp
vst1q_u64(C0 + vi * 2, c0_0);       vst1q_u64(C0 + (vi + 1) * 2, c0_1);
...
vst1q_u64(C4 + vi * 2, c4_0);       vst1q_u64(C4 + (vi + 1) * 2, c4_1);
vst1q_u64(rs0[tail] + vi * 2,       new_r0_0);
vst1q_u64(rs0[tail] + (vi + 1) * 2, new_r0_1);
...
vst1q_u64(rs2[tail] + (vi + 1) * 2, new_r2_1);
```

- **10 C stores** — bitplane × position pair, persisting the rolled C
  for the next row.
- **6 ring stores** — `rs0/rs1/rs2` × position pair, overwriting the
  ring slot at `tail` with the new row-sum (`new_r*_*` computed back
  in Phase B). The slot is now the *newest* in the ring; the old
  contents (which we already used as `old_r*` in Phase F) are gone.
  The slot's identity flips mid-iteration: at the top it represents
  row `r − 2`, at the bottom it represents row `r + 3`. The
  read-before-write order is what makes this safe — the leaving
  row's data lives in registers (`old_r*_*`) across the C-roll, so
  the in-place overwrite can't race with the subtract.

The `tail` pointer itself is advanced at the bottom of the per-row
block: `tail = (tail + 1) % 5;`.

### Phase H — Slide the row-sum's `prev`/`curr`

```cpp
adult_prev = adult_next_0;
adult_curr = adult_next_1;
```

The horizontal sliding-trio shifts by two columns (one for each of the
two positions we processed). Next iteration's `adult_next_0` will be
two further columns to the right.

---

## 2. The peeled boundary iteration

```cpp
if (vi + 2 == tnw) {
    const uint64x2_t adult_next_0 =
        vandq_u64(vld1q_u64(np1 + (vi + 1) * 2),
                  vld1q_u64(np0 + (vi + 1) * 2));
    // ...
    neon_row_sum_3bit(adult_curr, adult_next_0, bnd_next, ...);
    // ...
}
```

Runs when the row has an even `tnw ≥ 2` and the main loop stopped one
pair short of the end. The body is **identical to the main loop's**, with
exactly two differences:

1. `adult_next_1` is **not loaded** — it's replaced by `bnd_next`, the
   precomputed wrap-around right-edge ADULT vector.
2. `vi` is not incremented after, because the next thing that runs is
   either the odd-tail block or the end of the row.

The peel removes the `vi + 2 == tnw ? bnd_next : load(...)` ternary that
would otherwise sit inside the main loop. With that ternary, the
compiler kept `bnd_next` on the stack and **speculatively reloaded it
every iteration** to hedge the branch — visible as a hot stack-load in
perf annotate. Peeling moves the only use to its own block, lets
`bnd_next` live in a register across the row, and shrinks the main loop
body by a handful of instructions.

---

## 3. Why x2, not x1 or x4

The C-roll is a loop-carried dependency chain about 10 stages deep
(5-stage borrow + 5-stage carry, with the next iteration's ripple
depending on this iteration's resolved C). On a single C, the OOO
engine cannot overlap two consecutive iterations — it can only execute
one carry-chain at a time.

**x2** introduces a *second*, independent C accumulator (the `_1`
variants). The chains run in lock-step on the same hardware but with
fully independent data, so the scheduler can interleave their µops on
the SIMD pipes. Effective IPC during the C-roll roughly doubles.

**x4** would expose four chains but bring register pressure past the
breaking point. With x2 we already spill one vector to `[sp]` (the
disassembly shows `str q12, [sp]` / `ldr q0, [sp]` around addresses
`5b1c` / `5b4c`). x4 would multiply that and the spill cost would
exceed the ILP gain.

---

## 4. Live registers across one iteration

Rough inventory of NEON values live at the **peak** of the iteration —
right between the emit and the C-roll, when `adult_*`, all of `C*_*`,
`s*w_*`, and the freshly-computed `new_r*_*` are simultaneously alive:

| Group                         | Count | Notes                                  |
|-------------------------------|-------|----------------------------------------|
| `adult_prev`, `adult_curr`    | 2     | Carried across iterations              |
| `adult_next_0`, `adult_next_1`| 2     | Live during phase B                    |
| `new_r2/r1/r0` × 2 positions  | 6     | Used by C-roll add and ring store      |
| `c0..c4` × 2 positions        | 10    | Live for emit + C-roll                 |
| `s1w_0/s0w_0/s1w_1/s0w_1`     | 4     | Used by emit only                      |
| Misc scratch (`born`, `surv*`, `adult_sv`) | a few transient | Recycled within emit |

Total: ~24 "must stay live" registers plus a transient pool. NEON has
32 architectural registers (`v0`–`v31`). The compiler manages to keep
almost everything in registers, with one observable spill.

---

## 5. Op count, ops-per-cell

The x2-iteration body is **roughly 230 NEON instructions**, processing
**256 cells** (2 positions × 128-bit pair × 64 bits/lane × 2 lanes — wait,
that's not right; let me recount). Each NEON pair = 128 bits = 128 cells;
the iteration processes **two pairs** = **256 cells**. So:

```
~230 insns / 256 cells ≈ 0.9 NEON insns per cell
```

This was 1.05 before the Phase 15B SHA3 fusions and ~0.92 after the
final d0 / d1 fusions added in this session. The SIMD throughput floor
on Neoverse-V2 (4-wide SIMD issue) is around 0.5–0.6 ops/cell for an
ideal stream, so we have shrinking but real headroom.

---

## 6. Critical-path dependency graph (one iteration, one position)

The arrows below approximate the **loop-carried** part of the critical
path (the chain that bounds back-to-back row iterations):

```
adult_next_* loaded
       │
       ▼
neon_row_sum_3bit  (~7 cycles internal)
       │
       ▼
new_r0/1/2_* ────────────────────┐
                                  │
C0..C4 loaded (10 lds in parallel)│
       │                          │
       ▼                          │
[emit predicate compute] ─→ d0/d1 store
       │                          │
       ▼                          │
c5_sub3 (5 stages × 2 cyc) ─→ C   │
       │                          │
       ▼                          │
c5_add3 (5 stages × 2 cyc) ─→ C ──┘
       │
       ▼
C0..C4 stored
```

The visible loop-carried chain through `C` is roughly **20 cycles** of
latency (10 stages × 2-cycle NEON ops). x2 unrolling overlaps two
copies, so the effective per-iteration cost is closer to **10 cycles**
for the C-roll plus what the scheduler hides in the row-sum compute and
the predicate emit.

---

## 7. Summary

The x2-unrolled body is the kernel's hottest loop, and every line in it
has a job:

- **Loads up front** (adult, C, source) so the scheduler can begin all
  three computation streams in parallel.
- **Row-sum compute and emit** run alongside each other; emit only
  needs `C` (which has already been loaded) and the centre row source.
- **C-roll** (sub + add) happens *after* the emit because the emit
  needs the un-rolled `C`. Doing it after lets the chain's latency
  overlap with the predicate compute of the *next* iteration via OOO
  scheduling.
- **Stores** of `C` and the ring slot at `tail` happen last, after the
  values are fully resolved.
- **Two-position interleaving** is the lever that doubles ILP on the
  ripple-borrow / ripple-carry critical path.
- **The peeled boundary** keeps the hot loop free of a `bnd_next`
  ternary that would force a stack reload.
- **SHA3 fusions** (`veor3q`, `vbcaxq`, `vbslq` for MAJ) collapse
  classic two- and three-op gate patterns into single instructions
  wherever bit-disjointness or three-input identities make it sound.
