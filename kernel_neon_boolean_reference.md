# `kernel_neon.cpp` — Boolean Logic and Instruction Mapping

A bit-sliced cellular-automaton kernel. Cells are packed across two bitplanes and
processed 128 at a time in `uint64x2_t` registers. Every transition rule is expressed
as Boolean algebra over SIMD lanes. This document states the Boolean identity behind
each construct and the NEON instruction(s) it lowers to.

Notation: `⊕` XOR, `∧` AND, `∨` OR, `¬` NOT.

---

## 1. Cell encoding

Each cell holds a 2-bit state across bitplanes `s1` and `s0`. A `uint64x2_t` packs
128 cells (2 × 64). Only ADULT cells contribute to neighbour counts; this is the
single property that defines the automaton.

| State      | (s1, s0) | Counts as neighbour |
|------------|----------|---------------------|
| EMPTY      | 0, 0     | no                  |
| JUVENILE   | 0, 1     | no                  |
| ADOLESCENT | 1, 0     | no                  |
| ADULT      | 1, 1     | yes                 |

The "adult" test is therefore `adult = s1 ∧ s0`, emitted as `vandq_u64(sp1, sp0)`
wherever a row is loaded.

---

## 2. Horizontal row-sum — `neon_row_sum_3bit`

For each cell, sum the ADULT bits of a 5-wide window `{a, b, c, d, e}` centred on it.
Result is a 3-bit count `(out2, out1, out0)`, value range `0..5`.

### 2.1 Window lane construction

Each lane is the centre word shifted by ±1 or ±2 columns. Bits shifted off a 64-bit
word boundary are pulled from the adjacent word (`prev_adj` / `next_adj`, assembled
with `EXT`). The shift and the OR-in of wrapped bits fuse into one shift-and-insert.

| Lane | Boolean meaning                                  | Intrinsics → instructions |
|------|--------------------------------------------------|---------------------------|
| `a`  | `(curr ≪ 2)`, low 2 bits from `prev_adj ≫ 62`    | SHL + SRI                 |
| `b`  | `(curr ≪ 1)`, low 1 bit from `prev_adj ≫ 63`     | SHL + SRI                 |
| `c`  | `curr` (centre)                                  | —                         |
| `d`  | `(curr ≫ 1)`, bit 63 from `next_adj ≪ 63`        | USHR + SLI                |
| `e`  | `(curr ≫ 2)`, top 2 bits from `next_adj ≪ 62`    | USHR + SLI                |

`SRI` / `SLI` keep the destination's surviving bits and overwrite only the vacated
ones, replacing a shift followed by an OR with a single 2-input instruction. This
removes four `vorrq` per call.

### 2.2 Full-adder tree

Two chained 3-input full adders. A full adder is:

```
sum   = x ⊕ y ⊕ z
carry = maj(x, y, z) = (x ∧ y) ∨ (z ∧ (x ⊕ y))
```

The majority function collapses to one `BSL`, since `BSL(m, p, q) = (m ∧ p) ∨ (¬m ∧ q)`:

```
maj(x, y, z) = BSL(x ⊕ y, z, x)
```

When `x = y`, mask `x⊕y = 0` selects `x` (= `x∧y`). When `x ≠ y`, it selects `z`.

Stage 1 sums `a + b + c`; stage 2 sums that with `d + e`.

| Code line | Boolean                       | Instruction      |
|-----------|-------------------------------|------------------|
| `axb`     | `a ⊕ b`                       | EOR              |
| `s_abc`   | `a ⊕ b ⊕ c`                   | EOR              |
| `c_abc`   | `maj(a, b, c)`                | BSL              |
| `dxe`     | `d ⊕ e`                       | EOR              |
| `c_des`   | `maj(s_abc, d, e)`            | BSL              |
| `out0`    | `s_abc ⊕ d ⊕ e`               | EOR3             |
| `out1`    | `c_abc ⊕ c_des`               | EOR              |
| `out2`    | `c_abc ∧ c_des`               | AND              |

Derivation of the high bits: total `= s_abc + d + e + 2·c_abc = out0 + 2·c_des + 2·c_abc`.
Since `c_abc + c_des ∈ {0,1,2}`, its bit 0 is `out1` (XOR) and its bit 1 is `out2` (AND).
`axb` and `dxe` are reused as both BSL masks and XOR operands.

---

## 3. Five-bit column accumulator

`C = (c4 c3 c2 c1 c0)` holds the running sum of five adjacent row-sums
(5 rows × max 5 = 25, fits in 5 bits). Per row, the kernel subtracts the row leaving
the window and adds the row entering it. Both operands are 3-bit (`r2, r1, r0`), so the
ripple chain is short. The same `add3` routine, run five times against a zeroed `C`,
performs initialisation.

### 3.1 Subtract — `c5_sub3_neon`

Per-bit: `diff = c ⊕ r ⊕ borrow_in`, and

```
borrow_out = (¬c ∧ r) ∨ (¬(c ⊕ r) ∧ borrow_in) = BSL(c ⊕ r, r, borrow_in)
```

When `c ≠ r`, borrow becomes `r`; when `c = r`, it passes through.

| Stage          | borrow update              | diff update                  |
|----------------|----------------------------|------------------------------|
| bit 0 (no in)  | `b = ¬c0 ∧ r0` → BIC       | `c0 = c0 ⊕ r0` → EOR         |
| bit 1          | `BSL(c1⊕r1, r1, b)`        | `c1 = c1 ⊕ r1 ⊕ b` → EOR3    |
| bit 2          | `BSL(c2⊕r2, r2, b)`        | `c2 = c2 ⊕ r2 ⊕ b` → EOR3    |
| bits 3–4       | borrow into 4 = `b ∧ ¬c3`  | `c4 = c4 ⊕ (b ∧ ¬c3)` → BCAX; `c3 = c3 ⊕ b` → EOR |

With `r3 = r4 = 0`, the bit-4 update `c4 ⊕ (b ∧ ¬c3)` is exactly `BCAX(c4, b, c3)`
(`a ⊕ (b ∧ ¬c)`), a single SHA3 instruction. `c3` is read before being overwritten.

### 3.2 Add — `c5_add3_neon`

Per-bit: `sum = c ⊕ r ⊕ carry_in`, and `carry_out = maj(c, r, carry_in) = BSL(c⊕r, carry_in, c)`.

| Stage          | carry update                | sum update                     |
|----------------|-----------------------------|--------------------------------|
| bit 0          | `carry = c0 ∧ r0` → AND     | `c0 = c0 ⊕ r0` → EOR           |
| bit 1          | `BSL(c1⊕r1, carry, c1)`     | `c1 = c1 ⊕ r1 ⊕ carry` → EOR3  |
| bit 2          | `BSL(c2⊕r2, carry, c2)`     | `c2 = c2 ⊕ r2 ⊕ carry` → EOR3  |
| bit 3 (r3=0)   | `carry = c3 ∧ carry` → AND  | `c3 = c3 ⊕ carry` → EOR        |
| bit 4 (r4=0)   | —                           | `c4 = c4 ⊕ carry` → EOR        |

---

## 4. Transition rule

After the slide, `C = A_full`: the count of adult cells in the full 5×5 window,
including the centre. Both predicates are 5-bit value tests.

### 4.1 `born` — true for `A_full ∈ {3, 4, 5}`

```
born = BSL(c1, c0 ∧ ¬c2, c2) ∧ ¬c3 ∧ ¬c4
```

The `BIC` against `(c4 ∨ c3)` enforces `value ≤ 7`. The inner `BSL(c1, …)` is a MUX on
bit 1: select `c0 ∧ ¬c2` when `c1 = 1`, else `c2`. Over values `0..7` this yields exactly
`{3, 4, 5}`. Instructions: BIC + BSL + ORR + BIC.

### 4.2 `survives` — true for `A_full ∈ {5..10}`

```
surv_lo  = (c2 ∧ ¬c3) ∧ (c1 ∨ c0)      // {5, 6, 7}
surv_hi  = (c3 ∧ ¬c2) ∧ ¬(c1 ∧ c0)     // {8, 9, 10}
survives = (surv_lo ∨ surv_hi) ∧ ¬c4   // exclude ≥ 16
```

`surv_lo` is the `c2`-band `{4,5,6,7}` minus 4; `surv_hi` is the `c3`-band `{8,9,10,11}`
minus 11; union is `{5..10}`. Instructions: AND / BIC / ORR, final BIC against `c4`.

### 4.3 Output bits

No centre subtraction is required: `born` is only consequential on EMPTY (centre 0, so
`A = A_full`), and `survives` only on ADULT (centre 1, which shifts the survival window
`{4..9}` to `{5..10}` in `A_full`).

```
adult_sv = (s1 ∧ s0) ∧ survives
d1 = s0 ⊕ s1 ⊕ adult_sv                 // EOR3
d0 = adult_sv ⊕ ((s1 ∨ born) ∧ ¬s0)     // ORR + BCAX
```

Evaluating both formulas over all four input states reproduces the state machine:

| In (s1,s0)   | adult_sv   | d1         | d0         | Out         | Behaviour                  |
|--------------|------------|------------|------------|-------------|----------------------------|
| EMPTY 0,0    | 0          | 0          | `born`     | (0, born)   | spawns to JUVENILE if born |
| JUVENILE 0,1 | 0          | 1          | 0          | 1,0         | ages → ADOLESCENT          |
| ADOLESCENT 1,0 | 0        | 1          | 1          | 1,1         | ages → ADULT               |
| ADULT 1,1    | `survives` | `survives` | `survives` | mirrors     | survives, else dies → EMPTY |

State machine summary: an EMPTY cell with 3–5 adult neighbours spawns a JUVENILE; the
spawn matures over two generations (JUVENILE → ADOLESCENT → ADULT); an ADULT persists
only with 5–10 adults in the wider window, otherwise dies. `d1` is one EOR3; `d0` is
ORR then BCAX.

---

## 5. Instruction reference

| Intrinsic        | Instruction | Boolean function        | Role                                   |
|------------------|-------------|-------------------------|----------------------------------------|
| `veorq_u64`      | EOR         | `a ⊕ b`                 | XOR, sum bits                          |
| `vandq_u64`      | AND         | `a ∧ b`                 | adult mask, carries                    |
| `vorrq_u64`      | ORR         | `a ∨ b`                 | predicate bands                        |
| `vbicq_u64`      | BIC         | `a ∧ ¬b`                | range exclusion, masking               |
| `vbslq_u64`      | BSL         | `(m ∧ x) ∨ (¬m ∧ y)`    | majority (carry/borrow), MUX           |
| `veor3q_u64`     | EOR3 (SHA3) | `a ⊕ b ⊕ c`             | 3-way sum in one op                    |
| `vbcaxq_u64`     | BCAX (SHA3) | `a ⊕ (b ∧ ¬c)`          | fused top-bit carry/borrow + output    |
| `vsriq_n_u64`    | SRI         | shift-right-and-insert  | fused shift-OR, window lanes           |
| `vsliq_n_u64`    | SLI         | shift-left-and-insert   | fused shift-OR, window lanes           |
| `vshlq_n_u64`    | SHL         | logical left shift      | window shifts                          |
| `vshrq_n_u64`    | USHR        | logical right shift     | window shifts                          |
| `vextq_u64`      | EXT         | cross-lane extract      | word-boundary wrap                     |

Three identities account for most of the density:

- majority → `BSL` (every carry and borrow)
- 3-input XOR → `EOR3`
- XOR of an AND-NOT → `BCAX`

These let a full adder, a ripple stage, and the rule's output bit each compress to one
or two instructions.
