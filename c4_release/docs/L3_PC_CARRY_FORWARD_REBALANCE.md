# L3 PC Carry-Forward Fix (2026-05-11)

## TL;DR

`Primitives.carry_forward_attention` is **correctly implemented**. The bug
diagnosed in `PHASE_4_BZ_BNZ_INSTRUMENTATION.md` (L3 PC carry attends to
self instead of the previous step's PC byte 0) had a **different root
cause** than that doc identifies: the L1 threshold attention's K weights
were being silently overwritten by the L0 block op, corrupting L1H0 /
L1H1 / L1H2 with L0's thresholds (3.5 / 4.5 / 7.5 instead of 0.5 / 1.5
/ 2.5). This made the source-position discrimination `K = L1H1 - L1H0`
collapse to zero at the previous step's PC byte 0 (both flags fire there
with corrupted L0-like semantics).

The fix is in the compiler layer-assignment pipeline, not in
`Primitives.carry_forward_attention`. **The primitive's score budget did
not need re-tuning.**

After the fix:

* L1H0[PC] semantics: fires only at PC marker itself (d <= 0.5). Confirmed.
* L1H1[PC] semantics: fires at d <= 1.5. Confirmed.
* L3 head 0 source pos 37 (step-0 PC byte 0) score raw 41.7 (was 14.06,
  same as anywhere else). Attention weight 0.9999 (was ~0). Self pos 71
  attention weight 0.0 (was 0.393).
* `EMBED_LO[10] = 1.0` at step-1 PC marker after L3 attn (was
  `EMBED_LO[0] = 1.0`).

## Root cause: head 0 K-weight collision between L0 (block op) and L1 (attn op)

`LayerCompiler._assign_layers` places `kind="attn"` ops at the earliest
layer whose inbound deps (dim writes) are satisfied. `kind="block"` ops
are not part of that analysis — they are pinned to `layer_idx`.

Pre-fix layout (after `derive_layout()`):

```
Layer 0:
  - layer1_threshold_attn   (kind=attn, deps: IS_MARK, MARK_SE_ONLY, CONST)
  - layer0_threshold_attn   (kind=block, layer_idx=0, runs LAST)

Layer 1:
  - layer2_threshold_attn   (kind=attn, deps: IS_MARK, CONST)

Layer 2:
  - layer3_carry_forward_attn  (kind=attn, deps include L1H0/L1H1)
```

Both `layer1_threshold_attn` and `layer0_threshold_attn` target
`block[0].attn`. They both set head 0 K weights:

```
layer1 (head 0): W_k[0, IS_MARK] = 0.5     # detect "marker within 0.5"
layer0 (head 0): W_k[0, IS_MARK] = 3.5     # detect "marker within 3.5"
```

The block op runs **after** the attn op, overwriting L1's 0.5 with L0's
3.5. Output projection (W_o) still routes head 0's output to `L1H0`
because the W_o entries don't collide, but the K threshold now matches
L0 semantics: `L1H0` fires within distance 3.5 of any marker (not 0.5).

Same conflict on heads 1, 2 (L1H1 / L1H2). Heads 3, 4 (HAS_SE,
L1H4) avoid the conflict because their K dims (MARK_SE_ONLY, IS_MARK
at threshold 6.5) differ from L0's K writes for those slots.

The downstream `make_layer3_carry_forward_attn_op` sets, for head 0:

```
W_k[0, L1H1+0] = +L
W_k[0, L1H0+0] = -L
```

intended to fire at the previous step's PC byte 0 (d=1 from PC marker,
where L1H1=1 AND L1H0=0). With L1H0 corrupted to fire wherever L0H0
fires (within 3.5 of any marker), the `L1H1 - L1H0` discriminator
zeroes out exactly at the intended source position. Empirically:

```
pre-fix:  pos 37 K-score = +14.06 (gate-only)
          pos 71 K-score = +14.06 (gate-only)
          → top attn weight goes to pos 40 (the EMBED-zero byte), softmax1
            anchor absorbs most mass elsewhere

post-fix: pos 37 K-score = +41.72 - 17 (ALiBi) = +24.72
          pos 71 K-score = +14.06
          → pos 37 attn weight = 0.9999, EMBED_LO[10]=1.0 carries forward
```

## Fix

Convert `layer1_threshold_attn`, `layer2_threshold_attn`, and
`layer3_carry_forward_attn` from `kind="attn"` to `kind="block"`, each
pinned to `layer_idx={1,2,3}` respectively. Add three companion
`_layer{1,2,3}*_dep_anchor` no-op `kind="attn"` ops with identical
reads/writes so the LayerCompiler's longest-chain length is preserved
and downstream `kind="attn"` ops remain at their original block indices.

Post-fix attn-weight inventory (non-zero block.attn entries):

```
Block 0:  8 Q / 8 K   (L0 thresholds, alibi=10)
Block 1:  5 Q / 5 K   (L1 thresholds, alibi=10, head3 slope=0)
Block 2:  1 Q / 1 K   (L2 threshold 5.5)
Block 3:  25 Q / 19 K (L3 carry-forward, 7 heads, alibi=0.5)
Block 4:  8 Q / 5 K   (L4 PC relay, unchanged)
... downstream blocks unchanged
```

## Impact on tests

`test_smoke.py` (handler mode): unchanged — no smoke test regressed
(verified up through `test_simple_function`; the pre-existing slow
tests `test_jmp_forward`, `test_jmp_backward`, `test_memcmp`,
`test_div_by_zero` also fail on baseline with the same timeout).

`test_pure_neural_pc.py` Phase 1 tests: still 0/13 on this branch. The
L3 PC carry-forward fix is necessary but not sufficient. Single-step
IMM tests like `test_imm_then_exit` return `0x04040404` because:

* `test_imm_byte_values[42]`: pre-fix returned 690563371 (0x2929292B),
  post-fix returns 690563369 (0x29292929). The "off-by-1 between bytes"
  has been eliminated (consistent with the carry-forward now being
  correct), but the model still emits AX bytes 1/2/3 incorrectly as
  copies of byte 0, and byte 0 is still off-by-1.

Phase 1 (and downstream Phases 2, 4) needs at least two more fixes
beyond this one to start passing:

1. AX byte 0 production by the IMM execution path (off-by-1 even on
   single-step programs).
2. AX bytes 1-3 not being suppressed — the model emits the same byte
   value four times in a row instead of zeros for the high bytes.

Both are upstream of the BZ/BNZ Phase-4 fall-through issue and need to
be triaged with single-step IMM in isolation before BZ/BNZ.

## Files modified

* `c4_release/neural_vm/unified_compiler/migrated_ops.py`:
  `make_layer1_threshold_attn_op`, `make_layer2_threshold_attn_op`,
  `make_layer3_carry_forward_attn_op` converted to block ops with
  explicit `layer_idx`. Three companion `*_dep_anchor` ops added.
  `all_core_ops()` updated to register the dep anchors.

## Diagnostic scripts (added under scripts/debug/)

* `trace_block3_scores.py` — dumps raw L3 head 0 Q*K scores at the
  step-1 PC marker for the BZ_not_taken[1] program; confirms the
  source-position weight is now ~1.0.
* `check_l1h0_pos37.py` — dumps L1H0[PC] / L1H1[PC] / L1H2[PC] at the
  step-0 PC marker and surrounding byte positions; confirms the
  fine-threshold semantics are now correct.
* `dump_all_blocks.py` — prints per-block attn non-zero W_q / W_k
  counts and alibi slopes; useful for verifying layer-assignment
  changes don't shift other ops to the wrong block.
* `inspect_layer_assignment.py` — prints the LayerCompiler's
  ops-per-layer assignment given the current `all_core_ops` set.
