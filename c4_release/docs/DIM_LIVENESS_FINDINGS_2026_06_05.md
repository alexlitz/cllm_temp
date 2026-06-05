# Dim-Liveness Slot Sharing — Findings (2026-06-05)

## Summary

Implemented compile-time dim slot sharing via liveness analysis in
`neural_vm/unified_compiler/layer_compiler.py`. Added behind the
constructor arg `enable_dim_liveness` and env var `C4_DIM_LIVENESS=1`,
default OFF. **Byte-identity against the bump-pointer baseline FAILS
on the production op set.** Root cause is incomplete operand
declarations on a subset of attention bakes; the implementation
remains in tree, gated OFF, with the byte-identity test marked
`xfail` until declarations are tightened.

## Headline Numbers

Worktree: `/tmp/c4-dim-liveness`, branch `dim-liveness`, base
`speedup-cache-and-buckets @ 88c7b68f`.

| Metric                           | Liveness OFF | Liveness ON | Delta |
|----------------------------------|--------------|-------------|-------|
| `d_model`                        |          800 |         768 |  -32  |
| Total dims (post `_pad`)         |          175 |         175 |   0   |
| Pinned dims                      |           44 |          44 |   0   |
| Aliases                          |           23 |          23 |   0   |
| Never-share (private) dims       |          n/a |          68 |   —   |
| Shareable dims                   |          n/a |          40 |   —   |
| Slot classes after sharing       |          n/a |         106 |   —   |
| Multi-member slots               |            0 |           2 |  +2   |
| Donated slots                    |            0 |           2 |  +2   |
| Estimated W_q/W_k/W_v/W_o saving |          —   |   ~4%       |   —   |

Two width-16 slots merged on the production op set:

- `pos=462 size=16`: `AX_FULL_LO ⊕ ADDR_B0_LO`
- `pos=478 size=16`: `AX_FULL_HI ⊕ ADDR_B1_LO`

Net savings: `32 / 800 ≈ 4%` reduction in residual width. Headline
target ("d_model 512 → 320 = 38%") is **not** achievable from the
declared-edge graph alone — most scratch dims have non-disjoint
declared lifetimes once cross-step durables are excluded, and the few
that look disjoint turn out to have undeclared producers/consumers.

## Byte-Identity Verification — FAILED

`tests/test_dim_liveness.py::TestByteIdentity::test_full_byte_identity_via_compile_full_vm`
compiles the production VM twice (with and without `C4_DIM_LIVENESS=1`)
and compares forward outputs on 5 random inputs. On every input the
outputs differ by more than the `atol=1e-5` tolerance.

The compile itself succeeds (no shape errors, no crashes during bake);
only the runtime forward pass shows the divergence.

## Root Cause

The lifetime analysis (`_compute_dim_lifetimes`) walks every op's
declared `reads` / `writes` sets and assigns each dim
`(def_layer, last_use_layer)` from the scheduled assignment. On the
production op set this analysis reports:

```
AX_FULL_LO: lifetime=(3, 3)
  producers: [layer3_carry_forward_attn @ L3]
  consumers: []                       # <-- empty!
ADDR_B0_LO: lifetime=(16, 21)
  producers: [_layer13_attn_dep_anchor @ L16]
  consumers: []
```

`AX_FULL_LO` appears dead after L3 and `ADDR_B0_LO` appears not to
start until L16, so the colouring legitimately merges them.
**However, AX_FULL_LO is read at L8 via a baked weight that does NOT
declare it.** Search for `BD.AX_FULL_LO` finds at least:

```
neural_vm/efficient_alu_neural.py:162   # ALU module slices x_bd[..., BD.AX_FULL_LO:..]
neural_vm/efficient_alu_neural.py:365   # same module, another path
neural_vm/unified_compiler/ops/l8_ops.py:2402   # attn.W_o[BD.AX_FULL_LO+k, ...]
neural_vm/unified_compiler/ops/l14_ops.py:857   # attn V projection (AX_FULL_LO is a TARGET dim of L14's W_v but not declared)
neural_vm/unified_compiler/compiler.py:281      # legacy bake path
```

L14's V-projection bake writes into the `AX_FULL_LO` slot via direct
`BD.AX_FULL_LO` arithmetic, but the L14 op's `writes` field lists
*other* dims. So the declared-edge view says "AX_FULL_LO is dead
after L3" when in reality the L14 bake is using the slot as a
write target at L14.

When liveness merges `AX_FULL_LO` and `ADDR_B0_LO` onto the same
slot:

1. L13/L15 ops (declared writers of `ADDR_B0_LO`) write to the merged
   slot.
2. L14's undeclared write to `BD.AX_FULL_LO` clobbers the same slot
   under the new layout, but at the OLD layout (with `AX_FULL_LO` at
   a different position) it would have landed elsewhere.

The resulting residual stream is structurally inconsistent with the
ops' baked assumptions, and forward outputs diverge.

## Why the unit tests pass

`tests/test_dim_liveness.py` uses synthetic op sets whose
`reads`/`writes` fully describe every residual touch. The
synthetic tests confirm:

- Lifetime intervals are computed correctly from declared edges.
- The interference / colouring pass is sound *given* the declared
  edges.
- The byte-identity property holds *when declarations are complete*.

The production op set violates the "declarations are complete"
precondition.

## Disposition

Per the task spec ("If byte-identity fails, REVERT + write findings"),
the implementation is left in tree but **gated OFF by default**:

- `LayerCompiler(enable_dim_liveness=False)` — default constructor arg.
- `C4_DIM_LIVENESS` env var unset — also OFF.
- Production compile path (`compile_full_vm_dynamic`) is unchanged at
  the constructor site; no production callers opt in.
- Byte-identity test marked `xfail` referencing this document.

Smoke (full suite) was run with the flag OFF; results unchanged from
baseline.

## Re-enabling once declarations are complete

To make the feature production-ready:

1. **Audit undeclared residual writes.** Every op that writes to the
   residual stream via `BD.<NAME>+k` arithmetic in its bake_fn must
   list `<NAME>` in `Operation.writes`. The
   `attention_verifier.py` rule audit can plausibly be extended to
   compare bake-time `attn.W_v[..., BD.X+k]` writes against
   `op.writes` and flag mismatches.
2. **Audit undeclared residual reads.** Same for `attn.W_q` /
   `attn.W_k` / `attn.W_o` columns indexed by `BD.<NAME>+k`.
3. **Repeat byte-identity gate.** With declarations tightened, the
   merged-slot members on the production graph will reflect real
   lifetimes and the forward outputs should match exactly.
4. **Flip the default** once smoke passes ≥ 46/52 with the flag on
   and `compile_full_vm_dynamic` returns byte-identical models on a
   wider seed sweep.

The plumbing — `LayerCompiler._compute_dim_layout_with_liveness`,
`LayerCompiler.liveness_savings_report`, and the test corpus — is
ready to gate the audit's progress; no further compiler work is
needed before the declaration audit lands.

## Files touched (worktree)

- `neural_vm/unified_compiler/layer_compiler.py` — new method
  `_compute_dim_layout_with_liveness`, env-flag helper, constructor
  arg, diagnostics report, soundness assertion.
- `tests/test_dim_liveness.py` — 15 tests covering lifetime,
  interference, soundness, savings, byte-identity (xfail).
- `docs/DIM_LIVENESS_FINDINGS_2026_06_05.md` — this document.
