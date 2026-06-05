# Dead FFN Unit Audit — 2026-06-05

Per-layer audit of FFN hidden units allocated-but-unused. A "dead unit"
is any row `u` of `block.ffn.W_up` where
`W_up[u, :].abs().sum() == 0 AND W_gate[u, :].abs().sum() == 0`. Such
units consume `2 * d_model + 1` params apiece (W_up row + W_gate row +
b_up scalar; the b_gate scalar is symmetric and folded by the
right-sizer) but contribute nothing at inference.

L6 and L7 already ship defensive `dead_unit_zero` ops (`model_ops.py`
lines 1389 / 1502). This audit asks the same question for every other
layer: where else has the FFN unit allocator over-budgeted?

## Methodology

1. `compile_full_vm_dynamic(S=100, alu_mode="lookup", disk_cache=False,
   strict=False)` — warm cache, post-bake, pre-rightsize.
   `_right_size_ffns` (phase=1200) was monkey-patched to a no-op so the
   allocated 4096-unit footprints survive into the inspection step.
2. Walked every `block.ffn` (and post-op sub-FFNs) recursively. For
   each leaf `PureFFN`-shaped module, counted rows where both `W_up`
   and `W_gate` were entirely zero.
3. Aggregated by `block.attn.layer_idx` so the numbers map back to the
   source `lN_ops.py` files (35 expanded blocks ↔ 17 logical layers
   plus a few ALU wrapper blocks).
4. Param savings computed as `dead_units * (2 * d_model + 1)` with
   `d_model = 800`, then `× 4 bytes` for float32 MB.

Raw data: `/tmp/dead_unit_by_layer_idx.json`.

## Per-layer table (logical `layer_idx`)

| Layer | Allocated | Used | Dead | %Dead |
|------:|----------:|-----:|-----:|------:|
|   L0  |       7   |    7 |    0 |  0.0% |
|   L1  |       5   |    5 |    0 |  0.0% |
|   L2  |      10   |   10 |    0 |  0.0% |
| **L3**| **4096**  |**134**|**3962**|**96.7%** |
|   L4  |     544   |  543 |    1 |  0.2% |
| **L5**| **4096**  | **88**|**4008**|**97.9%** |
|   L6  |    2328   | 1664 |  664 | 28.5% |
| **L7**| **4096**  |   **0**|**4096**|**100.0%** |
| **L8**| **4096**  |   **0**|**4096**|**100.0%** |
|   L9  |    2056   | 2056 |    0 |  0.0% |
|   L10 |    3622   | 3622 |    0 |  0.0% |
| **L11**|**4096**  |   **0**|**4096**|**100.0%** |
| **L12**|**4313**  | **217**|**4096**|**95.0%** |
|   L13 |    1846   | 1846 |    0 |  0.0% |
|   L14 |    4096   | 4096 |    0 |  0.0% |
|   L15 |    4096   | 4096 |    0 |  0.0% |
|   L16 |    5632   | 5632 |    0 |  0.0% |
|   L17 |    1883   | 1882 |    1 |  0.1% |
|   L18 |    1578   | 1578 |    0 |  0.0% |
|   L19 |    1304   | 1304 |    0 |  0.0% |
|   L20 |    2074   | 2074 |    0 |  0.0% |
| **L21**|**4608**  | **704**|**3904**|**84.7%** |
|   L22 |    1536   | 1536 |    0 |  0.0% |
|   L23 |     192   |  192 |    0 |  0.0% |
|   L26 |    4624   | 4592 |   32 |  0.7% |
|   L28 |    1176   | 1176 |    0 |  0.0% |
|   L33 |    2059   | 2059 |    0 |  0.0% |
| **TOTAL** | **70069** | **41113** | **28956** | **41.3%** |

(Layers >L23 are wrapper / ALU expansion blocks created by
`_expand_wrapper_blocks`; their attention `layer_idx` does not roll
back to a source `lN_ops.py` so they are reported as-is.)

## Top-3 reduction candidates

### 1. L7 — 4096 dead, 100% (`l7_ops.py`)

The file's own docstring acknowledges this: *"L7 is attention-only:
every op in this file is `kind="block"` and writes attention weights
(Q/K/V/O), not FFN hidden units. There is no `_set_layer7_ffn` helper
in `vm_step` and no `ffn_units_used` annotation on any L7 op"*
(`l7_ops.py:74-83`). The `_L7_FFN_UNIT_LAYOUT` table records four
1-unit placeholders for auditability, but none are ever written.

**Recommendation**: add a no-op `Operation` at L7 with
`kind="ffn"`, `ffn_units_used=0`, so `LayerCompiler` pre-sizes the L7
block FFN to `hidden_dim=0` instead of 4096. Alternatively, instantiate
`block[7].ffn = nn.Identity()`. Savings: 4096 × 1601 = **6.55 M params
(25.0 MB float32)**.

### 2. L8 — 4096 dead, 100% (`l8_ops.py`, primary block)

Same pattern as L7: L8 is attention-routing-heavy. The actual FFN work
lives in `make_layer8_multibyte_routing_op` (`ffn_units_used=2055`),
but that op is `kind="block"` pinned to `target_op_name=
_layer8_multibyte_routing_dep_anchor` — its bake writes into a
different block's FFN, leaving the L8 primary `block.ffn` untouched.

**Recommendation**: same as L7 — annotate `_layer8_ffn_dep_anchor` (or
add one) with `ffn_units_used=0`. Savings: 4096 × 1601 =
**6.55 M params (25.0 MB float32)**.

### 3. L3 + L5 — 3962 + 4008 dead, 96.7% / 97.9% (`l3_ops.py`,
`l5_ops.py`)

These are real FFN bakes — L3 writes 134 PC/SP/BP default-rule units,
L5 writes 88 fetch / opcode-decode units — but neither
`make_layer3_ffn_op` nor `make_opcode_decode_ffn_op` declare
`ffn_units_used`. Both fall through to `DEFAULT_LAYER_MAX_UNITS=4096`.

**Recommendation**: annotate both. L3 should set `ffn_units_used=136`
(134 + 2 byte-1 emission rules added by
`_add_layer3_pc_byte1_output_rules`). L5 should set `ffn_units_used=88`.
The byte-identity rule lowering walks a monotonic cursor independent
of the allocator, so the change is forward-only. Combined savings:
(3962 + 4008) × 1601 = **12.76 M params (48.7 MB float32)**.

## Other notable layers

* **L11 + L12 (MUL partial / combine, 8192 dead total)**: The L11
  `layer11_mul_partial` op claims to write 4096 rules
  (`_L11_MUL_PARTIAL_UNIT_LAYOUT` totals 16 × 256), and L12 has
  `ffn_units_used=4096`. Yet at the primary `block[layer_idx=11/12].ffn`
  in `alu_mode="lookup"` we observe **zero** non-zero rows. The real
  MUL work lands in the `FlattenedALUMul` post-op that L12 attaches
  (and L11's `make_l11_alu_postop_attach_op` was deliberately removed
  in 2026-06-03 to fix L17 tail MUL double-fire). The primary L11/L12
  FFNs appear to be vestigial duplicates of the post-op pipeline.
  Suggested follow-up: confirm by running the byte-identity gate with
  L11/L12 primary FFNs forced to zero — if smoke is clean, drop them
  the same way `make_l11_alu_postop_attach_op` was dropped.

* **L21 (3904 dead, 84.7%)**: the layer writes only 192 units of its
  4096 budget. Owner not immediately identifiable from grep; likely a
  tail / epilogue FFN. Needs an `ffn_units_used` annotation.

* **L6 (664 dead)** and **L26 (32 dead)** already ship corrective
  `dead_unit_zero` passes; the residual count here is the static count
  after those passes (because they zero W_up *rows*, the rows still
  count as dead by our definition — the surgical zero op is doing its
  job but doesn't help the param budget).

## Aggregate savings if every dead unit is eliminated

* Dead units: **28,956**
* Params per unit (d_model=800): `2 * 800 + 1 = 1601`
* Savings: 28,956 × 1601 = **46,358,556 params**
* float32 footprint: **~176.84 MB**
* float16 / bfloat16: **~88.42 MB**

This is purely from over-allocated FFN hidden-dim budgets that
`_right_size_ffns` already trims at runtime — so the inference
footprint already reflects the post-trim numbers. The 176 MB is the
*storage / pre-trim* savings (relevant to disk-cache size, weight
export, ONNX bundles), plus a small inference win in environments that
skip `_right_size_ffns`.

## Suggested follow-up patches

1. Add `ffn_units_used=` to: `make_layer3_ffn_op` (136),
   `make_opcode_decode_ffn_op` (88), and an explicit
   `_layer7_ffn_dep_anchor` / `_layer8_ffn_dep_anchor` (0). Single-file
   edits; byte-identity preserved.
2. Investigate the L11/L12 vestigial primary-FFN duplication.
3. Add an L21 owner annotation.

After (1)+(3) the right-size pass becomes pure verification rather
than a load-bearing trim.
