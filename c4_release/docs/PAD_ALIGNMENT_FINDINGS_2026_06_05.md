# d_model Pad Alignment — Findings (2026-06-05)

## Summary

Investigated lowering the residual-stream pad alignment so the
liveness-pass savings (commit `471c1c08`) translate to a real `d_model`
reduction. **Conclusion: not feasible without an attention-module
rewrite. The current 800-padded / 793-inner gap is structural and
falls 1 dim above the next-lower multiple of `n_heads=8`.**

## Headline Numbers (with `C4_DIM_LIVENESS=1`)

| Metric                          | Value |
|---------------------------------|------:|
| Inner `d_model` (pre-pad)       |   793 |
| `_pad` slot width               |     7 |
| Final `d_model` (post-pad)      |   800 |
| `n_heads`                       |     8 |
| `head_dim = d_model // n_heads` |   100 |
| Next-lower multiple of 8        |   792 |
| Sum of all declared dim sizes   |  1170 |
| Number of declared dims         |   175 |

The last "real" dim sits at `pos=777, size=16` (`SP_VIA_LEV_DETECTOR`),
ending at 793; `_pad` of size 7 brings the residual stream to 800.

## The Constraint

Pad alignment is computed in
`c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py:2220-2223`:

```python
if layout.d_model % n_heads != 0:
    pad = n_heads - (layout.d_model % n_heads)
    compiler.declare_dim("_pad", pad)
    layout = compiler.compile()
```

The constraint is `d_model % n_heads == 0`. It is enforced by the
attention module assertion at `vm_step.py:144`:

```python
assert num_heads * self.head_dim == dim, ...
```

Uniform head_dim is also required by the RoPE cache
(`precompute_rope_cache(head_dim, ...)`), the ALiBi slopes
(`[H]`-shaped, applied per-head identically), and the SDPA
`view(B, T, num_heads, head_dim)` reshape in `forward()`.

## Why `n_heads` Cannot Be Reduced

Grep over `c4_release/neural_vm/unified_compiler/ops/` shows allocated
`head_idx` values 0-8 plus 12-13:

- `l3_ops.py:1487`         — `head_idx=7`
- `l14_ops.py:862`         — `head_idx=8`
- `l15_ops.py:38, 670, 779`— `head_idx=12, 13`
- `model_ops.py:761`       — `head_idx=7`

The allocator default is `DEFAULT_LAYER_MAX_HEADS=8`
(`neural_vm/attention_head_allocator.py:72`), but layers already pin
heads beyond that index. Reducing `n_heads` below 8 (let alone below
the per-layer max of 14) is infeasible without a per-layer attention
re-bake. The `vm_step.py:55` comment explicitly notes:

```python
DEFAULT_N_HEADS = 8  # HD=64; HD=32 broke attention score budgets (LEV).
```

## Why Non-Uniform `head_dim` Is a Big Rewrite

Allowing different heads to have different widths would let us pad
fewer dims (e.g., 7 heads of width 100 + 1 head of width 93 → 793). It
would touch:

- `AutoregressiveAttention.__init__` — `W_q/W_k/W_v/W_o` no longer
  `(dim, dim)` square; need block-diagonal Q/K/V layouts.
- `forward()` — `view(B, T, num_heads, head_dim)` cannot share one
  `head_dim`; per-head `split + cat` required (kills SDPA path).
- RoPE — `precompute_rope_cache(head_dim, ...)` per-head.
- ALiBi — slopes already per-head, but the bias shape would diverge.
- Every IR helper that names a Q/K/V column by absolute index across
  a square `(dim, dim)` matrix.

This is a multi-day refactor with high blast radius. Scoped out.

## The 1-Dim Gap

We are precisely 1 dim above the next-lower multiple of 8 (793 vs 792).
If the liveness pass could share *one more* shareable dim, `d_model`
would drop from 800 → 792. The current pass shares two width-16 slots
(see `DIM_LIVENESS_FINDINGS_2026_06_05.md`); the next-best candidate
would have to be either a width-1 dim with a disjoint declared lifetime,
or a width-2+ dim that closes the 793-→-792 gap.

Quick path: audit `STACK0_BYTE1/2/3` and similar width-1 durables to
see whether they have disjoint lifetimes with any existing slot. If
*any* width-≥1 dim can share, we save 7 dims (the full pad), not 1,
because the pad shrinks to 0.

## Alternative Savings Paths

1. **`d_model_packing=True`** (already implemented in
   `full_vm_compiler_dynamic.py:2233-2261` and gated by
   `compile_full_vm_dynamic(d_model_packing=True)`). Best-fit-decreasing
   repack of unpinned dims; rounds up to `n_heads` at the end. May
   already capture some of the 7-dim slack — worth measuring with the
   liveness pass ON. Test surface: `tests/test_d_model_packing.py`.

2. **Sparse W_q/W_k/W_v storage** — many residual dims are read by
   only a handful of heads. A masked / sparse-projection attention
   could keep `d_model=800` but cut FLOPs and params. Out of scope
   for "lower d_model" but a strict win for inference cost.

3. **One more shared slot in the liveness pass** — see "The 1-Dim Gap"
   above. Lowest-effort path; quantify with the
   `_compute_dim_lifetimes` debug print at `layer_compiler.py:1453`.

4. **Drop one width-1 durable** — audit `dim_registry_dynamic.py` for
   width-1 dims that the liveness pass marks "never-share". If any is
   actually dead in the current op set (post L0-L7 declaration audit),
   removing it drops inner d_model 793 → 792 and the pad cancels.

## Recommendation

Do not attempt non-uniform `head_dim`. Pursue path (3) or (4): find
one more dim that can be shared or removed. If the liveness pass can
be tightened to share even one more durable, the pad collapses from
7 → 0 and `d_model` drops from 800 → 792 in one shot.

Document only; no code change in this commit.
