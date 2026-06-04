# RuntimeAttentionFragment migration gaps (2026-06-04)

## Summary

The plan in `docs/IR_DSL_DESIGN.md` Wave W7 calls for removing the
`RuntimeAttentionFragment` escape hatch by migrating each call site to
one of the V2 attention primitives in `building_blocks_dsl.py`:

- `efficient_exp_attention` — softmax1+ALiBi-0 `e^N` approximation.
- `memory_load_attention` — address-keyed memory load (one-hot Q/K).
- `fetch_byte_attention` — PC+offset code-byte fetch via slot-index
  permutation.

Or, for shape-dependent heads, a shape-parameterized
`DeclarativeAttentionHeadSpec` builder per `IR_DSL_DESIGN.md` §3.4.

Wave W7 partially landed in commit `5b147cb5`: the per-fragment
`runtime_predicate` lambda was replaced with compile-time Python
`if num_heads >= N` branching at the IR builder level. That leaves the
**bodies** of the five fragments still imperative — direct writers into
`attn.W_q.data`, `attn.W_k.data`, `attn.W_v.data`, `attn.W_o.data`.

This document records why those bodies cannot currently be migrated to
the V2 primitives and what shape the missing primitive(s) would need.

## Sites (all in `neural_vm/unified_compiler/ops/l15_ops.py`)

| Line | Fragment name | Body writer | Shape gate |
|---:|---|---|---|
| 450 | `layer15_memory_lookup.heads_0_3`              | `vm_step._set_layer15_memory_lookup_heads_0_3`         | always-on |
| 461 | `layer15_memory_lookup.lev_heads_4_11`         | `vm_step._set_layer15_memory_lookup_lev_heads_4_11`    | `num_heads >= 12` |
| 473 | `layer15_memory_lookup.suppress_heads_0_3`     | `l15_ops._suppress_l15_lookup_heads_0_3`               | always-on |
| 489 | `layer15_memory_lookup.suppress_lev_blockers_4_11` | `l15_ops._suppress_l15_lookup_lev_blockers_4_11`   | `num_heads > 4` |
| 503 | `layer15_memory_lookup.suppress_pop_d8_head_9` | `l15_ops._suppress_l15_lookup_pop_d8_head_9`           | `num_heads > 9` |

## Why none of the V2 primitives fit

### `memory_load_attention` — closest match, still wrong

`memory_load_attention` builds a single `DeclarativeAttentionHeadSpec`
with:

- Q: `AP(i, addr_query_dims[i], query_weight)` per addr slot
  (default `query_weight=15.0`).
- K: `AP(i, addr_key_dims[i], key_weight)` per addr slot
  (default `key_weight=15.0`).
- V: `AP(addr_w+j, value_dims[j], 1.0)` per value byte.
- O: `AO(output_dims[j], addr_w+j, 1.0)` per value byte.

That is a clean **one-hot** address match at full key strength.
The L15 lookup heads write a fundamentally different score shape:

1. **Binary 24-bit address encoding, not one-hot.** Heads 0-3 cover the
   address with 24 binary bits (3 bytes × 2 nibbles × 4 bits, dims
   `4..27` per head, scale `±10`). LEV heads 4-11 use 48-cell one-hot
   nibble encoding (dims `4..35`, scale `50.0`). A single primitive
   can model one or the other but not both.
2. **Per-head Q discriminators beyond the address.** Head 0 fires on
   `OP_LI_RELAY ∨ OP_LC_RELAY ∨ (CMP+3)` for the pop-group dual-role;
   heads 1-3 fire on `L1H4[BP] − H1[BP]` (BP-threshold byte positions).
   `memory_load_attention` has no Q-gate mechanism.
3. **Many auxiliary suppression rows.** Each head writes per-marker
   suppression bias on `MARK_PC` (−25000), `MARK_SP` (−100000),
   `OP_LEV` (−1000), `H1[SP]` (−50000), `H1[BP]` (−50000), and PC-byte
   /AX-byte block rows at offsets `+29`, `+30`, `+31`, `+32`, `+33`.
   These are zero-or-negative non-target queries with positive K terms,
   producing large negative scores. There is no V2 helper for this.
4. **Per-head byte selection.** Dim 3 picks a different MEM_VAL_Bn
   per head (head 0 → `L2H0[MEM] − H1[MEM]`, heads 1-3 →
   `MEM_VAL_B{1,2,3}`). The byte selection weight (`BS=60`) is
   distinct from the address weight (`scale=10`).
5. **Per-head position gates.** Dim 28 (and the migrated dim 36 for
   LEV heads, to avoid ADDR_KEY aliasing) gates each head to its
   target byte position (`MARK_AX ∨ MARK_STACK0` for head 0,
   `BYTE_INDEX_{0,1,2}` for heads 1-3).
6. **Dozens of bespoke discriminator rows in the suppress helper.**
   `_suppress_l15_lookup_heads_0_3` adds row writes at offsets 34, 42,
   58, 59, 60, 61, plus full Q/K wipes at rows 59-61 to break stale
   value-lane miss terms. The pop-d8 head 9 helper wipes the entire
   head and reconstructs one bespoke discriminator row at slot 63.
   None of this has a V2 template.

`docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md` §3 explicitly flags this:

> L15 score budgets are tight (`_set_layer15_memory_lookup` has very
> carefully tuned constants — see the comment block around
> `vm_step.py:6359`). A change in K-side encoding from 48-dim one-hot
> to 4×16-dim per-byte one-hot must preserve the score margins.

That document treats a K-side encoding swap alone as multi-day work.
Replacing the whole imperative body with declarative specs is a
larger undertaking.

### `efficient_exp_attention` — wrong role

This primitive synthesises a single head that approximates `e^N` via
softmax1+ALiBi-0 against an empty BOS row. The L15 fragments are
address-keyed loads / suppressors, not exponential approximations.

### `fetch_byte_attention` — wrong role

This primitive fetches the byte at `PC + pc_offset` via a slot-index
permutation. The L15 fragments are address-keyed value loads, not
PC-relative code fetches.

### Shape-parameterized `DeclarativeAttentionHeadSpec` builders

This is the right *shape*: a Python function takes `num_heads` and
returns a list of specs. The blocker is the spec contents, not the
dispatch: a faithful migration would need to emit roughly
**~250 `AP` entries per head × 12 heads ≈ 3000 `AP` lines** for the
load fragments, plus another ~500 lines for the suppress fragments,
and every single weight magnitude is load-bearing. The
imperative writers are not just "shape-dependent" — they are dense
per-head conditional weight maps with many distinct discrimination
axes (Q gates, K gates, position gates, suppression rows, value
lanes, bespoke discriminator rows for one-word push/pop, early-ENT
STACK0, AX LI/LC at 0xffe8, etc.).

A clean migration would need one of:

- A **higher-level V2 primitive** for "binary-encoded address match
  with per-head byte selection, Q opcode gating, marker suppression,
  and N-bespoke discriminator rows" — i.e. a `binary_address_lookup_attention`
  helper plus a `marker_suppression_rows` helper plus an
  `auxiliary_discriminator_row` helper.
- Or a **per-spec extension mechanism** that lets a base
  `memory_load_attention` spec be augmented with Q-side gate writes,
  per-row K wipes, and bespoke discriminator rows. The base
  `DeclarativeAttentionHeadSpec` lowering already supports arbitrary
  `AP` lists, so the gap is helper coverage rather than spec
  expressivity.

Both options are explicitly out of scope for the present V2 surface
(see `docs/BUILDING_BLOCKS_DSL.md` §2.8 — the three attention helpers
are the entire current attention surface).

## Status

- **Sites found:** 5 (all in `l15_ops.py`).
- **Sites migrated this pass:** 0.
- **Sites with no matching V2 primitive:** 5.

The Wave W7 commit `5b147cb5` already replaced the runtime predicate
with compile-time `num_heads` branching, which removes the *dispatch*
half of the escape hatch. The five sites remaining are the *content*
half: they are `RuntimeAttentionFragment` instances wrapping imperative
weight writers, and no current V2 primitive can replace those writers
without losing byte identity.

## Recommended next steps (out of scope for this pass)

1. Design a `binary_address_lookup_attention` helper modeled on
   `memory_load_attention` but with `±scale` per-bit Q/K writes and
   optional per-head Q gate / suppression-row parameters.
2. Add an `attention_head_extension` shape that takes a base spec and
   layers additional `AP`/`AO` rows on top, so the L15 heads can be
   built as `memory_load_attention(...)` + N extension rows. The
   lowering would concatenate Q/K/V/O tuples into the same head.
3. Migrate `_set_layer15_memory_lookup_heads_0_3` first (the largest
   fragment, always-on) using the new helpers. Validate via the
   neural-byte-identity gate already used for `memory_load_attention`
   in `tests/test_building_blocks_dsl.py`.
4. Repeat for the LEV branch and the three suppress fragments. Each
   migration is its own commit, gated by a `Primitives.lower_attention`
   `torch.allclose` comparison against the imperative baseline.

Until those helpers exist, the five sites remain as
`RuntimeAttentionFragment` entries. They are documented and contained:
the body fns live in named helpers, the shape branching is already
declarative, and `_L15_HEAD_LAYOUT` exposes the head axis to tooling.
