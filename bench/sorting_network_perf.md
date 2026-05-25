# Sorting-network kernel vs Phase 4 NEON — performance comparison

Standalone summary of the Phase 4.5 excursion on branch `sorting`. Full
implementation in `src/kernel_sort.cpp`; full context (theory, design,
correctness, build log) in `NOTES.md`.

## Hardware

Dev box, single vCPU Graviton4 (Neoverse-V2), `-O3 -mcpu=neoverse-v2`,
g++-14. Dedicated CPU (no concurrent processes).

## Headline

The sorting-network kernel is **~21× slower** than the Phase 4 NEON
adder kernel on all measured grid sizes. Outputs are byte-identical to
the reference oracle. The slowdown is consistent across grid sizes and
input patterns, which means the cost is in the kernel's per-cell work,
not in any size-dependent effect (cache, etc.).

## Timings (10 000 generations, median over the 5 public test patterns)

| Size | Phase 4 NEON (ms) | kernel_sort (ms) | Slowdown |
|---|---|---|---|
| 512  | 385     | 7 950   | **20.6×** |
| 2048 | 5 500   | 117 000 | **21.3×** |
| 8192 | 100 128 | n/a (~35 min projected) | — |

Per-pattern detail at 2048:

| Pattern | NEON (ms) | sort (ms) | ratio |
|---|---|---|---|
| public_1_random_low      | 5 813 | 123 918 | 21.3× |
| public_2_random_high     | 5 358 | 116 648 | 21.8× |
| public_3_structured      | 5 467 | 117 955 | 21.6× |
| public_4_sparse_clusters | 5 352 | 116 166 | 21.7× |
| public_5_boundary_stress | 5 671 | 116 328 | 20.5× |

Variance across patterns is small (~1.3×). The slowdown is intrinsic to
the kernel structure, not input-dependent.

## Op-count accounting (per 128-cell NEON vector per output row)

| Stage | Phase 4 NEON | kernel_sort |
|---|---|---|
| Horizontal popcount (amortised 1/row in Phase 4) | ~22 | folded into sort |
| Vertical popcount / sort | ~55 (5 ripple-carry FA chains) | ~382 (bitonic sort 32) |
| Centre handling | ~15 (borrow propagation) | ~13 (mux on A_incl thresholds) |
| Predicates | ~13 | ~4 |
| Next-state assembly | ~13 | ~13 |
| ADULT recompute per vi | folded into row-sum | ~50 (5 rows × 5 col-shifts) |
| **Total ALU ops** | **~118** | **~462** |
| **Theoretical ratio** | — | **~3.9×** |

The measured 21× is **5× worse** than the op-count ratio of ~4× would
predict. Likely contributors (not investigated further — prototype is
for evidence, not for optimisation):

- Bitonic sort has depth ~15 levels of dependent compare-exchanges per
  cell-vector. The OoO core overlaps work across `vi` but the
  in-vector critical path is long.
- 32 NEON vectors held live through the sort. Architectural register
  file is exactly 32 × 128-bit on Neoverse-V2; with the AND/OR
  temporaries the compiler likely spills to stack, adding load/store
  traffic.
- ADULT bits are recomputed per output row instead of being amortised
  across 5 rows as in Phase 4 — the sort framing doesn't admit the
  same row-sum ring buffer at width 32 768 (see below).

## Why the cached version isn't an obvious fix

In principle the sort kernel could cache 5-cell-sorted thermometer
rows in a ring buffer (analogous to Phase 4's 3-bit row-sums) and then
do a vertical-merge of 5 sorted-of-5 thermometers per output row. That
would amortise the horizontal sort work the way Phase 4 amortises
horizontal popcount.

Problem: the cached row representation grows from 3 planes to 5 planes
per row. The ring buffer width grows correspondingly:

- Phase 4 ring (3 planes × 5 rows): 60 KiB at width 32 768. Already
  tight in 64 KiB L1d.
- Sort ring (5 planes × 5 rows): 100 KiB at width 32 768. Out of L1.

So even the structural optimisation that would close part of the 5×
"hidden" gap pushes the workload out of L1d at the target grid width.
The two negatives (op count + L1 fit) compound.

## Why this happens (theoretical case)

The four predicates we need — `born = (3 ≤ A ≤ 5)` and
`survives = (4 ≤ A ≤ 9)` on the 24-cell neighbour count — are
**symmetric Boolean functions** of 24 binary inputs: they depend only
on popcount.

Classical synthesis result: symmetric functions on N inputs are
gate-optimally computed by `popcount → small threshold circuit on
⌈log₂ N⌉ output bits`. The Phase 4 NEON kernel is that construction.

A sorting network is the right tool when you need *many order-statistic
outputs at once* and when the cost model penalises XOR (i.e., hardware
synthesis where XOR is a multi-gate primitive). Neither applies here:

- We need 4 thresholds, not many. Selection networks for 4 specific
  thresholds on 24 inputs share enough structure with a near-full sort
  that the savings vs full-sort don't materialise.
- NEON XOR is single-cycle, identical to AND and OR. The hardware-Life
  win — moria.us, 35 gates sort vs 49 gates adder — comes from XOR
  costing several gates in FPGA synthesis. That cost model evaporates
  in software bit-sliced.

In addition, Spawning Grid's parameter regime (N=24 inputs, k_max=10
≈ N/2.4) is worse for sorting than Conway's Life (N=8, k_max=N/2)
because selection cost peaks at the median; the further k stays from
the median, the cheaper selection gets relative to popcount. Our
worst-case threshold k=10 is mid-range and pulls cost up.

## Correctness

Every code path was validated before timing. See `NOTES.md` for the
full check matrix. Most stringent check:

| Grid | Reference time | kernel_sort time | sort vs reference |
|---|---|---|---|
| public_1_random_low_512  | 30 197 ms  | 7 703 ms   | **byte-identical** |
| public_1_random_low_2048 | 452 741 ms | 116 154 ms | **byte-identical** |

This is sort kernel output `cmp`'d directly against the
`reference/spawn_sim.cpp` oracle on the only two grids where
`.expected.bin` truth is available.

## Decision

Sort-network kernel **not the perf path**. Phase 4 NEON adder kernel
retained. Branch `sorting` retained as a research record but **not
merged** to `main`.

## What this carries forward to the design doc

Three independently sufficient reasons it didn't work:

1. **Cost-model mismatch.** The hardware-Life win for sorting vs adders
   is XOR-cost-driven; that disappears in software bit-sliced on
   NEON, where XOR is single-cycle.
2. **Wrong tool for symmetric functions.** Popcount-plus-threshold is
   gate-optimal for symmetric Boolean outputs; sorting networks pay
   O(N log² N) for what costs O(N) with adders.
3. **Cache.** Even with sort-network row caching, the ring buffer
   wouldn't fit in 64 KiB L1d at width 32 768. The two penalties
   compound.

Measured factor: **21× slower** at both 512 and 2048. Empirical
penalty matches the theoretical prediction in direction and exceeds it
in magnitude.
