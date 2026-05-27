# Phase 10: x2 Loop Unrolling — What We Did and Why

## Result

| Configuration | Time (32K, 8 threads, NEON) |
|---|---|
| Before (Phase 9) | ~142 s |
| After (Phase 10) | ~130 s |
| Improvement | ~8.5% |

---

## Background: How the kernel works

The grid stores cell state as two bitplanes (`s1`, `s0`). One NEON register holds 128 bits = 128 cells side by side. The inner loop processes one "NEON pair" (`vi`) per iteration — two consecutive 64-bit words, 128 cells total.

Per iteration, the kernel does:
1. Load the adult bits for the incoming row (the new row about to enter the 5-row window)
2. Compute the 5-wide horizontal row sum for that row using a full-adder network
3. Load the 5-bit running column accumulator `C` from memory
4. Subtract the outgoing row's contribution, add the new row's contribution (persistent-C trick from Phase 9)
5. Write `C` back to memory
6. Subtract the current cell's own adult status from `C` (the centre cell is not part of its own neighbourhood)
7. Evaluate the born/survives predicates
8. Encode and write the output state

---

## What the profiler showed

We ran `perf record` then `perf annotate` on the NEON binary at 32K, 8 threads. The annotated output (`ann.txt`) showed nearly all cycles in one 350-instruction hot region in `kernel_neon`. Three instructions stood out:

### Bottleneck 1 — Load-use latency (offset 0x5950–0x5954, ~14% of all cycles)

```
7.63%   ldr  q31, [x2, x0]          ← loads s1 for new row (np1[vi])
6.27%   and  v25.16b, v25.16b, v31.16b  ← adult = np1 & np0; consumes v31 immediately
```

On Neoverse-V2, a NEON load takes 4 cycles before the result is usable. There is zero distance between this load and the AND that consumes it. The AND stalls for 3 cycles waiting for the data. perf attributes those idle cycles to both instructions. Together: ~14% of runtime.

### Bottleneck 2 — Serial carry-chain stalls (offsets 0x5a24, 0x5a50, 0x5a60, ~18% combined)

```
7.57%   eor  v17.16b, v19.16b, v29.16b   ← stall: waiting for carry from earlier in chain
7.07%   and  v21.16b, v22.16b, v6.16b    ← stall: waiting for chain through 5a38→5a48
4.09%   and  v22.16b, v28.16b, v17.16b   ← convergence: needs BOTH v17 (5a24) and v28 (5a50)
```

These are in the `c5_sub3_neon` / `c5_add3_neon` functions. These perform a 5-bit ripple operation:

```
bit0 → carry → bit1 → carry → bit2 → carry → bit3 → carry → bit4
```

Each stage depends on the carry from the previous stage. The CPU cannot start stage N+1 until stage N is done. This is a serial dependency chain — no amount of out-of-order execution can shorten it within a single accumulator. Two such chains run back-to-back per `vi` iteration (one sub3, one add3), giving ~10 dependent operations on the critical path.

### Overall picture

- ~27% memory (loads/stores)
- ~70% arithmetic (bitwise logic in the carry chains and predicates)
- IPC: 3.14 — high, but the critical path forces a minimum latency per `vi` regardless

---

## Why x2 unrolling fixes this

The key insight: the `C` accumulator for position `vi` and the `C` accumulator for position `vi+1` are **completely independent**. They cover different columns of the grid. Their carry chains never interact.

In the original loop, the CPU processes `vi`, then `vi+1`, then `vi+2` — one at a time. The serial carry chain for `vi` sits on the critical path, and `vi+1`'s work cannot begin until `vi` finishes (because the loop body is one sequential block).

By unrolling x2, both positions' work is visible to the CPU in the same instruction window:

```
Sub chain for vi   ─────────────────────►
                         ↕ (independent)
Sub chain for vi+1 ─────────────────────►
```

The out-of-order engine can dispatch instructions from both chains simultaneously to different execution units. The stalls at 0x5a24, 0x5a50, and 0x5a60 shrink because while the CPU is waiting for a carry in the vi chain, it issues instructions from the vi+1 chain instead of sitting idle.

The same benefit applies to the load-use stall at 0x5950: the unrolled iteration has two independent new-row adult loads, and the CPU can be retiring the second load's chain while the first is still completing.

---

## What the code change looks like

Before (schematically):
```
for vi in 0..tnw:
    carry_chain(C[vi])   ← serial: vi+1 must wait for vi to finish
```

After:
```
for vi in 0, 2, 4, ...:
    carry_chain(C[vi])   ─┐ issued together; OOO engine overlaps them
    carry_chain(C[vi+1]) ─┘
tail handler for odd tnw (no-op for power-of-2 widths like 32768)
```

The loop stride doubles; the boundary case (`adult_next` wrapping at end of tile) only applies to the second position in each pair, checked with `vi + 2 == tnw`.

---

## What we did NOT change

- The carry-chain logic itself (`c5_sub3_neon`, `c5_add3_neon`) — still identical
- `fill_ring_slot_neon` — not in the perf hot path (called only 5× at row startup)
- The scalar kernel — not the graded path
- Thread count, memory layout, tiling strategy — unchanged

---

## Remaining gap

Other teams are reportedly at ~120 s. We are now at ~130 s (~8% gap remaining). Possible directions:

1. **x4 unrolling** — more ILP at the cost of higher register pressure (32 NEON registers is already tight at x2)
2. **Carry-chain restructuring** — Brent-Kung or Kogge-Stone style parallel prefix instead of ripple
3. **Prefetch hints** — `__builtin_prefetch` or ARM `prfm` ahead of the C_store loads to hide memory latency earlier
4. **Predicate fusion** — the born/survives computation still has a depth-4 AND tree; some savings possible
