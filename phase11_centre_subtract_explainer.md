# Phase 11: Centre-Subtract Elimination — What We Did and Why

## Result

| Configuration | Time (32K, 8 threads, NEON) |
|---|---|
| Before (Phase 10) | ~130 s |
| After (Phase 11)  | ~123 s |
| Improvement       | ~5.4% |

Cumulative since Phase 9: 142 s → 123 s (~13%).

Committed as `be61d59`, merged to `main` via PR #3.

---

## The rule we're computing

Each cell's fate depends on **A** = the number of ADULT cells in its 5×5 neighbourhood, **excluding the cell itself**:

```
EMPTY    → EGG       if 3 ≤ A ≤ 5
EGG      → JUVENILE  (always)
JUVENILE → ADULT     (always)
ADULT    → ADULT     if 4 ≤ A ≤ 9,  else EMPTY
```

So we need two predicates:
- **born**     = (3 ≤ A ≤ 5)   — decides EMPTY → EGG
- **survives** = (4 ≤ A ≤ 9)   — decides ADULT → ADULT-or-die

---

## What we were doing before (the wasteful part)

The kernel maintains a running 5-bit accumulator `C` per cell. But `C` is the sum over the full 5×5 window **including the centre cell** — call it `A_full`. Since A must *exclude* the centre, the old code did, every cell, every generation:

1. Snapshot `C` into a temporary `e0..e4`
2. Subtract the centre cell's own ADULT bit from `e` — a **5-bit ripple borrow chain** (depth 5, 9 ops)
3. Evaluate born/survives on the corrected `e`

That borrow chain (`e0..e4` with `vbicq`/`veorq`) was pure overhead repeated for all 1 billion+ cell-updates.

---

## The insight

We never actually need to subtract the centre, because **born and survives apply to mutually exclusive cell types whose centre value is already known:**

- **born** only ever affects an **EMPTY** cell. An EMPTY cell is not ADULT, so its centre contributes **0**. Therefore for any cell where born matters, `A = A_full` exactly — no subtraction needed.

- **survives** only ever affects an **ADULT** cell. An ADULT cell's centre contributes **1**. Therefore `A = A_full − 1`. Rather than subtract 1 from the number, we **shift the range test by 1**:

  ```
  survives: 4 ≤ A ≤ 9   ⟺   4 ≤ A_full − 1 ≤ 9   ⟺   5 ≤ A_full ≤ 10
  ```

The output encoding already gates born by "cell was EMPTY" and survives by "cell was ADULT", so this substitution is exact. (For EGG/JUVENILE cells the predicates are computed but masked away — same as before.)

---

## The new predicates (evaluated directly on `C` = A_full)

**born — unchanged.** The set {3,4,5} is identical whether you call it A or A_full:
```
born = ~c4 & ~c3 & (c2 ^ c1) & (~c1 | c0)
```

**survives — new range {5,6,7,8,9,10}.** Derived by Karnaugh map over the 5-bit value (c4..c0), with 26–31 as don't-cares:
```
survives = ~c4 & ( (~c3 & c2 & (c1 | c0))      // covers 5,6,7
                 | ( c3 & ~c2 & (~c1 | ~c0)) ) // covers 8,9,10
```
Hand-verified true on {5..10}, false on {0..4, 11..25}.

---

## What changed in the code

`src/kernel_neon.cpp`, inner row loop (both the x2-unrolled body and the odd tail):

1. **Deleted** the `e0..e4` snapshot and the 5-deep borrow chain (the centre subtraction).
2. **Reordered** to *emit-before-update*: compute born/survives from `C` first, write the output, **then** roll `C` to the next row with `c5_sub3`/`c5_add3`. Previously we snapshotted because the roll overwrote `C` before the emit needed it; emitting first removes that need.
3. **Swapped** the survives formula for the `{5..10}` version above.

Net effect per cell: removed ~9 borrow ops + 5 snapshot copies, added ~6 ops to the survives predicate ≈ **−8 ops/cell (~11% of the inner loop)**.

---

## Secondary win: the register spill is gone

Phase 10's x2 unroll was right at the 32-register NEON limit; the compiler spilled one value to the stack each iteration (`str q12,[sp]` + `ldr q0,[sp]` — confirmed in `objdump`). Dropping the `e0..e4` snapshot frees 5 live registers, so peak register pressure falls below 32 and the spill round-trip disappears. Verified: the post-Phase-11 unrolled loop has no `str q … [sp]` inside it.

---

## Correctness

- All 18 unit tests pass (`test_kernel_neon`, `test_kernel_scalar`).
- **NEON-vs-scalar cross-check:** the scalar kernel still does the old centre-subtract; NEON now does the A_full form. They produce byte-identical output over 100 generations of random data — independent proof the new formula is equivalent.
- **All 10 public grids** (512 + 2048, all 5 patterns) match the gen-10,000 `.expected.bin` reference exactly.

---

## Why this was the right lever (and Wallace-tree wasn't)

We considered replacing the rolling `C` accumulator with a from-scratch Wallace-tree sum of the 5 ring slots, computing predicates on a carry-save form. Rejected: the rolling accumulator already *avoids* summing 5 registers (it does 1 sub + 1 add of row deltas); recomputing from scratch would roughly **double** the instruction count to shorten a carry chain that is already hidden by the row-length recurrence distance. The centre-subtract elimination, by contrast, removes work that was on the live path of every cell with no downside.
