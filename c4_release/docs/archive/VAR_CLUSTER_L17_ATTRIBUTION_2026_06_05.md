# var_cluster L17 -0.391 OUTPUT_LO[0] attribution (2026-06-05)

Date: 2026-06-05
HEAD: `8d999b53` (`docs(mem-phase4): composite scaffolding design`)
Author: read-only block + per-head attribution agent

## TL;DR

Block 26 (compiler L17 attn) contributes **-0.391** to OUTPUT_LO[0] / OUTPUT_HI[0]
at the failing MEM_addr1 prediction row of `var_simple_0` / `if_var_0`, and
**-0.159** for `var_three_0`. Per-head decomposition shows that
**100% of the contribution comes from head 1 = `layer14_mem_generation.head_1`**.
All other 7 heads (0, 2..7) contribute exactly 0.000 to OUTPUT_LO[0]/HI[0]
at the failing slot.

The op is the **legitimate L3-cancel-and-emit-correct-byte writer**, not an
accidental fire. The -0.391 / -0.159 is the realized magnitude of an
attention-weighted -1.0 CONST cancel of L3's baseline. The
underpower is **attention-mass leak across keys** — the head's slot-0/1/2
predicate selects multiple key candidates and dilutes the cancel mass.

## Top-3 ops by |contribution to OUTPUT_LO[0]| at block 26 (var_simple_0)

| Rank | Op | head_idx | dLO0 | dHI0 |
|---:|---|---:|---:|---:|
| 1 | `layer14_mem_generation.head_1` (MEM addr byte 1, d=1) | 1 | **-0.3909** | **-0.3909** |
| 2 | `layer14_mem_generation.head_0` (MEM addr byte 0, d=0) | 0 | 0.0000 | 0.0000 |
| 3 | `layer14_mem_generation.head_2` (MEM addr byte 2, d=2) | 2 | 0.0000 | 0.0000 |

(Heads 3..7 also contribute 0.000.) The op is declared at
`neural_vm/unified_compiler/ops/l14_ops.py:693` (`make_layer14_mem_generation_op`).
Head 1 spec is constructed in the `for h in range(4)` loop at
`l14_ops.py:432-525` with `h=1`, `head_idx=1`, position predicate
`(BD.L1H1+MEM_I, BD.L1H0+MEM_I)`.

## Cross-program magnitudes

| Program | logit_pos | head 1 dLO0 | dominant key | attn weight | v_to_LO0 |
|---|---:|---:|---:|---:|---:|
| var_simple_0 | 118 | -0.3909 | key=93 | 0.8909 | -0.5000 |
| var_three_0 | 246 | -0.1587 | key=232 | 0.3175 | -0.5000 |
| if_var_0 | 174 | -0.3909 | key=149 | 0.8909 | -0.5000 |

`v_to_LO0=-0.5` reflects head_dim 0 (= CONST=+1.0) projected by
`AO(OUTPUT_LO+0, 0, -1.0)` PLUS head_dim 1 (= CLEAN_EMBED_LO+0 + OUTPUT_LO+0)
projected by `AO(OUTPUT_LO+0, 1, +1.0)`. At the selected keys the byte is
0x00 so CLEAN_EMBED_LO+0 = +1.0 ⇒ contribution = -1.0 + 0.5 = -0.5 per key.
(The +0.5 keys 116 / 96 / 172 / 152 reflect alternative source byte = 0xff
positions where the byte-flip side of the cancel writes +0.5.)

## Predicate audit — does it fire on the legitimate condition?

Head 1's position predicate (`addr_pos[1]`) is

```python
(BD.L1H1 + MEM_I, BD.L1H0 + MEM_I)  # d=1: predicts addr_b1
```

i.e. selects rows where `L1H1+MEM > L1H0+MEM` (one position past the MEM
marker; addr_b0 token, predicting addr_b1). Slot-33 enforces the same
predicate at high strength (±500). Slot-34 gates on `MEM_STORE=1`. Slot-38
blocks PC/AX/BP/STACK0 markers. All gates are appropriate for
`step0:MEM_addr1` prediction, and the divergence row IS the legitimate
addr_b0 row — **so this is the intended legitimate silencer, not an
accidental fire**.

The op WAS designed to cancel L3's `mem_byte_0_default` +0.940 baseline
(see `O` writes at `l14_ops.py:513-514`: explicit `AO(OUTPUT_LO+0, 0, -1.0)`
with comment "cancel L3 default at byte 0"). The intended algebra is
+0.940 (L3) + -1.0 * attn_mass (L14 head 1) + emit_byte_via_clean_embed
→ correct byte. But attention mass is only ~0.89 (var_simple) or ~0.32
(var_three), so the cancel falls short.

## Why var_three_0 attention mass is only 0.32

For var_simple_0 / if_var_0, attention concentrates at the symbolic
addr_b0 key (key=93 / 149) with w=0.89, leaving 0.10 / 0.01 on near
neighbours. For var_three_0 only 0.32 lands on the legitimate key (232)
and the rest goes to softmax1's implicit "no-key" floor (0.68 of mass
sinks into the +1 denominator). Slot-33's ±500 gate weight is set up
for clean PSH-store rows; the var_three_0 row's predicate sums fall
below the dominance threshold against the softmax1 normaliser.

## Recommendation

**Strengthen the legitimate silencer**, not block it. Two options:

1. **Bump slot-33 position-gate K weight from `+5.0` to e.g. `+50.0`**
   (`l14_ops.py:484`). The position gate is the limiting factor — making
   it dominate softmax1 by ~10x pulls attention mass at the legitimate
   key to ~0.99 across the board, restoring full -1.0 cancel. This is
   minimally invasive and zero-sum safe because slot-33 ALREADY enforces
   the same predicate the rest of head 1 needs.

2. **Add an O-write magnitude bump on the CONST cancel slot**
   (`l14_ops.py:513-514`): change `-1.0` to `-2.0`. Equivalent algebraic
   effect (doubles the per-key cancel mass) but skews the residual emit
   path if the byte is not 0x00. Less safe — option 1 is preferred.

Block-the-op patches (e.g. zero out head 1) are wrong: this op IS the
legitimate addr_b1 byte predictor for PSH stores. Silencing it would
break every working PSH/SI store, not just var_simple.

## Constraints met

* No code changes.
* Warm disk cache (`compile_full_vm_dynamic` reused across 3 programs).
* Doc-only.

## Artifacts

* `/tmp/var_l17_attribution.py` — block + per-head probe (read-only)
* `/tmp/var_l17_attribution.log` — full per-block + per-head output
* `neural_vm/unified_compiler/ops/l14_ops.py:693`
  (`make_layer14_mem_generation_op`)
* `neural_vm/unified_compiler/ops/l14_ops.py:432-525` (heads 0-3 spec loop)
