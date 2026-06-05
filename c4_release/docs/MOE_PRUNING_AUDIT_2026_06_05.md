# MoE Expert Routing — Param Reduction Audit

**Date:** 2026-06-05
**Scope:** `neural_vm/unified_compiler/ops/`
**Model:** d_model=1280, n_layers=16, ffn_hidden=4096. 152M dense / 530K nonzero (99.7% sparse).
**Background reading:** [`MOE_ROUTING_AUDIT.md`](MOE_ROUTING_AUDIT.md) (2026-05-11).

## 1. Method

Task brief named `building_blocks_dsl.py` / `opcode_expert_rules` — neither exists in
this codebase. The actual MoE routing surface is the `opcodes={"OP_X", ...}`
field on `Operation` (`neural_vm/unified_compiler/layer_compiler.py:271-279`),
plus `compaction_safe` (`layer_compiler.py:234-246`). Runtime partitioning of
FFN hidden units into per-opcode "experts" happens in
`neural_vm/vm_step.py::_partition_compact_ffn_by_opcode` (line 1784), which
labels a hidden unit as opcode-X when `W_up[unit, OP_X] > 0.5` or
`|W_gate[unit, OP_X]| > 0.5`.

Lint/IR tooling named in the brief (`lint_raw_ffn_rule.py`, `dsl_interpreter.py`)
also doesn't exist; instead I AST-walked all `Operation(...)` calls and
cross-checked the per-opcode unit counts against the `_set_layerN_alu`
bake bodies' `=== OP_X (N units) ===` comments.

Per-FFN-unit param cost = W_up + W_gate + W_down = `3 × 1280 = 3840` params.

## 2. Inventory

134 `Operation()` constructors. 25 declare `opcodes=...` (23 non-empty + 2
explicit `set()` sentinel). 14 carry an `ffn_units_used` annotation (only
those numbers contribute to FFN width).

### Opcode-gated ops with `ffn_units_used` (FFN consumers)

| Layer | Op | `ffn_units_used` | Gating opcodes |
|---|---|---:|---|
| L9  | `layer9_alu`                       | 3405 | ADD, SUB, AND, OR, XOR, EQ, NE, LT, GT, LE, GE |
| L10 | `layer10_alu`                      | 1846 | AND, OR, XOR, DIV, MOD, EQ, NE, LT, GT, LE, GE |
| L14 | `layer14_alu_nocarry_ax_bytes_zero` | 1311 | AND, OR, XOR, SHR |
| L16 | `layer16_lev_routing`              |  121 | LEV |

L8's ALU op (`layer8_alu`, opcodes ADD/SUB/LEA/EQ..GE) has no
`ffn_units_used` annotation but holds ~753 units per its bake comment;
`layer8_multibyte_routing` (2055 units, not opcode-gated) sets L8's width.

### Per-opcode unit accounting (from `_set_layerN_alu` bake comments)

`vm_step.py` source-of-truth comments at the head of each cluster:

| Layer | Opcode | Unit count |
|---|---|---:|
| L8  | OP_ADD     | 256 lo + 120 carry = 376 |
| L8  | OP_LEA     | 256 lo + 120 carry = 376 |
| L8  | OP_SUB     | 256 lo + 120 borrow = 376 |
| L8  | OP_ADJ     | 256 lo + 120 carry = 376 |
| L8  | OP_ENT     | 256 lo + 256 borrow = 512 |
| L8  | shared (CMP_GROUP, LEV relay, passthrough) | ~103 |
| L9  | OP_ADD/LEA/ADJ/SUB/ENT hi nibble | 512 each (×5 = 2560) |
| L9  | shared (CMP flags, carry, clearing) | ~144 |
| L10 | OP_OR / OP_XOR / OP_AND cross-product | 512 each (×3 = 1536) |
| L10 | shared (CMP combine, AX passthrough, DIV/MOD) | ~268 |
| L14 | OP_AND / OP_OR / OP_XOR / OP_SHR (clear chain) | ~1311 total |
| L16 | OP_LEV | 121 (only opcode) |

## 3. Per-opcode expert clusters in the same layer

Every layer above is a "1-of-N opcode → 1 expert cluster" pattern with the
expert clusters living as **disjoint hidden-unit ranges in the same
`PureFFN`**. The bake author lays them out sequentially and each unit's
`W_gate[unit, OP_X] = 1.0` selects which opcode value-gates it. Only one
fires per step. The other clusters' weights stay resident (`W_up`/`W_down`
rows zero out at compile but are still allocated columns in dense storage).

## 4. Param savings if mutually-exclusive experts share hidden-unit ranges

Each cluster currently dedicates its own unit range. Because only one opcode
is ever active in a given step, the W_up/W_gate/W_down rows of all
non-firing clusters could be **overlapped in the same hidden-unit slots**
(routed by a tile index ∈ [0..max_cluster)), reducing the per-layer FFN
hidden_dim to `max(per-opcode-cluster size)` + shared.

| Layer | Sum (per-opcode) | Max (overlapped) | Units saved | Params saved |
|---|---:|---:|---:|---:|
| L8  | 2016 |  512 | 1504 |  5,775,360 |
| L9  | 2560 |  512 | 2048 |  7,864,320 |
| L10 | 1536 |  512 | 1024 |  3,932,160 |
| L14 | 1311 |  ~328 |  983 |  3,774,720 |
| **Total** | **7423** | **~1864** | **5559** | **~21.3M params** |

That is **~14% of the 152M dense footprint** — but only ~40× the current
nonzero count (530K). Since the model is already 99.7% sparse, the
overlapping gain accrues to **dense storage / dense compute paths only**:
sparse-tensor and `compact_moe` paths already drop zero rows, so the
existing ONNX/sparse export already realizes most of this. Concrete wins
live in:

- dense `PureFFN.forward` (`W_up @ x`, `W_gate @ x`, `W_down @ h`) — the
  hidden_dim shrinks by 5559 → ~30% smaller matmul per layer at L8–L14;
- ONNX dense export of the small experts (each W_up row is one expert
  weight), reducing exported file size proportionally;
- `compact_moe`'s `SoftMoEFFN` construction — overlapping clusters cuts
  the per-expert PureFFN width and the number of parallel expert forwards.

## 5. Implementation sketch

Two viable approaches, in increasing invasiveness:

**(A) Slot-share annotation on `Operation`.** Add an optional
`hidden_slot_group: Optional[str]` field. Ops declaring the same group
share a unit range; the compiler's width allocator
(`build_model_from_layout` → `ModelLayout.ffn_widths`) tracks
`max(units_used_per_group)` instead of `sum`. Each gated unit gets an extra
silu-gated path on `OP_X` (already there); on routing, only the active
group's rows fire. Requires that the bake's `unit_index → output dim`
mapping is deterministic per-cluster (true for L8/L9/L10 — each takes a
disjoint OUTPUT_LO/HI range per opcode). **Touch points:**
`layer_compiler.py` width accounting, `_partition_compact_ffn_by_opcode`
(stays unchanged — still detects opcode via `W_up`/`W_gate` columns).

**(B) Rule rewrite at compile time.** Run a post-bake pass that detects
"all units gated by exactly one of OP_X, OP_Y, ... and these sets
write to disjoint output dims" and rewrites the W_up/W_gate/W_down
columns to reuse a single tile slot. This requires zero op-author changes
but adds rewrite complexity and a fixed-point verification step. Reuses
the existing partition logic but inverts it (group, then collapse).

**Recommendation:** start with (A) on L9 (largest single win at 7.86M
params). Annotate `layer9_alu` with one slot_group per opcode cluster,
verify smoke parity (L9 ALU exercises ADD/SUB/AND/OR/XOR/CMP) via the
existing `verify_compaction_safety` detector
(`decl_verifier.py:2458`), then propagate to L8 / L10 / L14.

## 6. Caveats

- L8 `layer8_alu` lacks `ffn_units_used`, so the compiler currently
  over-allocates that layer to default 4096. Filling in the annotation
  (likely 753, from the bake comment) is an orthogonal win — the
  pre-trim default-4096 path keeps 3343 zeroed-out units alive.
- L14's 4-opcode cluster (`AND/OR/XOR/SHR`) is a "clear chain" not a
  cross-product, so per-opcode sub-cluster sizes need bake re-inspection
  before the overlap estimate hardens; ~328 units/opcode is a guess.
- The L7 attention op (`layer7_operand_gather`, 13 opcodes) is the largest
  opcode-gated op overall but is `kind="block"` with no FFN cost — its
  experts live in `W_v`/`W_q` of `attn`, where the same overlap argument
  applies and is *not* covered by `_partition_compact_ffn_by_opcode`.
  Out of scope here; flag for a separate attn-MoE audit.

---

## CORRECTION (2026-06-05 later)

The L9 estimate in this audit was wrong. A follow-up implementation attempt verified:

- `l9_ops.py` has exactly ONE FFN op (`make_layer9_alu_op`), not 5 disjoint 512-unit sub-experts.
- The op declares `ffn_units_used=3405` and gates on `{OP_ADD, OP_SUB, OP_AND, OP_OR, OP_XOR, OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE}`. No `OP_LEA`, `OP_ADJ`, or `OP_ENT` gating.
- The allocator at `layer_compiler.py:664-667` already takes `max` per layer, not `sum`. There is no `sum`-vs-`max` win to extract for L9.

The "L9: 5×512=2560 → save 2048 units / 7.86M params" claim was based on misreading the bake-body source comments. The actual L9 bake (`_set_layer9_alu`) is a single ~3398-unit ADD/LEA/SUB/AND/OR/XOR/CMP cross-product cluster plus 7 marker-suppress units.

**L10/L14/L16 estimates in this audit have NOT been verified against actual op layout** and may have the same fabrication risk. Re-audit those before implementation. Use the per-opcode dim closure + head/layer activity audits as ground truth.
