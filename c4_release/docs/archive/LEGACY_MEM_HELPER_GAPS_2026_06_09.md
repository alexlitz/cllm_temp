# Legacy Memory Helper Gaps (2026-06-09)

Diagnostic audit of the three legacy imperative helpers cited by the xfail
decorators on the SI/LI/SC/LC pure-neural smoke tests:

- `c4_release/neural_vm/vm_step.py:5354 _set_layer7_memory_heads`
- `c4_release/neural_vm/vm_step.py:6702 _set_layer14_mem_generation`
- `c4_release/neural_vm/vm_step.py:7698 _set_layer15_memory_lookup`
  (umbrella; splits into `_set_layer15_memory_lookup_heads_0_3` at 7228 and
  `_set_layer15_memory_lookup_lev_heads_4_11` at 7417, plus
  `_suppress_l15_lookup_during_current_store_generation` at l15_ops.py:1675)

Tests blocked (xfail strict=False):
- `tests/test_smoke_pure_neural.py:464 test_si_li_roundtrip`
- `tests/test_smoke_pure_neural.py:489 test_sc_lc_roundtrip`
- `tests/test_smoke_pure_neural.py:521 test_si_li_multiple_stores`
- `tests/test_smoke_pure_neural.py:542 test_si_li_overwrite`
- `tests/test_smoke_pure_neural.py:566 test_si_li_16bit_value`
- `tests/test_l13_mem_addr_gather.py:645 test_l13_e2e_int_assign_and_read_roundtrip`

Also note `run_vm.py:2363/2370/2378` carries three `TODO(phase-7)` injection
sites for LC / SI / SC that route around the model's memory path. Those
overrides mask the legacy helper gaps in handler-mode runs and disappear once
the helpers are byte-identical.

---

## Per-helper status

### 1. `_set_layer7_memory_heads` — FULLY MIGRATED (no live caller)

Verified via `grep -rn _set_layer7_memory_heads` over `c4_release/`: the only
hits are the definition at `vm_step.py:5354` and a docstring cross-reference
at `vm_step.py:7074` / `l14_ops.py:2812`. Nothing calls it.

The declarative replacement is `_layer7_memory_head_specs(BD)` at
`unified_compiler/ops/l7_ops.py:440`, lowered by `make_layer7_memory_heads_op()`
at `l7_ops.py:291`. The specs mirror the legacy heads cell-for-cell:

- Head 7 (MEM flag broadcast: MEM_STORE / MEM_ADDR_SRC / OP_JSR / OP_ENT)
- Heads 2-4 (prev-AX byte gather into ADDR_B0/B1/B2_LO/HI)
- Head 5 (LI/LC/LEA/AND/OR/XOR/JSR/SHR/SI/SC/ADD/SUB/ENT flag relay)
- Head 6 (PSH/ENT/JSR/POP relay from STACK0 marker; PSH_AT_SP from SP)

Gap: **none for SI/LI/SC/LC**. The declarative op covers every imperative
write the helper made. The dim-contracts audit reports zero L7-attributed
failures.

Migration path: delete the imperative helper body (and the docstring
cross-references at vm_step.py:7074 / l14_ops.py:2812 / building_blocks_dsl.py:1026).
This is a pure cleanup, not a behaviour change.

### 2. `_set_layer14_mem_generation` — FULLY MIGRATED (no live caller)

Verified via grep: only references are the definition at `vm_step.py:6702`,
the docstring cross-reference at `l14_ops.py:429/765`, the `diag_l14_jsr_mem.py`
diagnostic (informational), and the `TODO(phase-7)` notes in `run_vm.py:2370/2378`
(which are about removing handler-mode overrides, not callers of the helper).

The declarative replacement is `_layer14_mem_generation_head_specs(BD)` at
`unified_compiler/ops/l14_ops.py:417`, lowered by
`make_layer14_mem_generation_op()` at `l14_ops.py:759`. Specs cover all 8
heads (0-3 MEM addr bytes via SP/STACK0, 4-7 MEM val bytes via AX/STACK0).

The op also calls `_clear_l14_mem_generation_overbroad_sp_suppression(attn, BD, HD)`
at `l14_ops.py:1012` — an imperative POST-bake patch that overrides the
helper's H1[SP] suppression rows. This patch is part of the declarative bake
itself (called from `make_layer14_mem_generation_op.bake`) so it is not a
gap — but it IS an undeclared imperative write the declarative spec system
does not model.

Gap: **dim-contracts audit reports 2 declarations failures**
(`PYTHONPATH=... python c4_release/tools/dim_contracts_audit.py`):
- `addr_b0_lo_prev_step_l15_to_l14`: consumer `layer14_mem_generation` does
  not declare `ADDR_B0_LO` in `reads` (declared as `ADDR_B0_LO.*.-1` only —
  the cross-step alias). Same-step read at numeric slot 206 is undeclared.
- `addr_b0_hi_prev_step_l15_to_l14`: same for `ADDR_B0_HI` at slot 206.

These are cross-step liveness contract failures (producer in L19/L15 layer
slot, consumer in L18/L14). The reads happen via the spec's V projection at
slots `base + 1 + k` reading `CLEAN_EMBED + OUTPUT`, so the contract is
declared as a numeric alias but not as a same-step `ADDR_B0_*` read.

Migration path: extend `reads` on `make_layer14_mem_generation_op` to
include `"ADDR_B0_LO"`, `"ADDR_B0_HI"` (not just the `.*.-1` alias). This
is a declaration-only fix — no model behaviour change. Then delete the
imperative helper body and the `_clear_l14_mem_generation_overbroad_sp_suppression`
post-bake patch can be re-expressed as a declarative spec row override on
heads 0-7 dim 33 / 37, but that's a follow-up cleanup not a gap blocker.

### 3. `_set_layer15_memory_lookup` — IMPERATIVE BODIES STILL LIVE (THIS IS THE GAP)

`_layer15_memory_lookup_ir` at `unified_compiler/ops/l15_ops.py:401` wraps the
imperative helper bodies in `RuntimeAttentionFragment` instances:

- `layer15_memory_lookup.heads_0_3` → calls `_set_layer15_memory_lookup_heads_0_3`
  (at vm_step.py:7228)
- `layer15_memory_lookup.lev_heads_4_11` → calls
  `_set_layer15_memory_lookup_lev_heads_4_11` (at vm_step.py:7417, num_heads >= 12)
- `layer15_memory_lookup.suppress_heads_0_3` → calls
  `_suppress_l15_lookup_heads_0_3` (at l15_ops.py:896)
- `layer15_memory_lookup.suppress_lev_blockers_4_11` → calls
  `_suppress_l15_lookup_lev_blockers_4_11` (at l15_ops.py:1473)
- `layer15_memory_lookup.suppress_pop_d8_head_9` → calls
  `_suppress_l15_lookup_pop_d8_head_9` (at l15_ops.py:1632)

Every one of these fragments writes directly to `attn.W_q.data[row, dim]` /
`attn.W_k.data[...]` / `attn.W_v[...]` / `attn.W_o[...]` rather than via
`DeclarativeAttentionHeadSpec(q=..., k=..., v=..., o=...)`. This is the
load-bearing gap for SI/LI/SC/LC: the symbolic verification, dim-contracts
audit, and gate audit all see the wrapping `Operation` declaration but
cannot inspect the actual Q/K/V/O writes — every row write is opaque to
the spec system.

Specific imperative writes per fragment:

**heads_0_3** (vm_step.py:7258-7414, ~150 lines):
- Dim 0 bias / OP_LI_RELAY / OP_LC_RELAY / CMP[3] activations + LEV/MARK_PC
  / MARK_SP / H1[SP] / H1[BP] suppression.
- Dim 1 (store anchor), dim 2 (ZFOD), dim 3 (byte selection) — these match
  the building-blocks DSL `lookup_table_rules` pattern but are not lowered
  through it (see `building_blocks_dsl.py:1026` referencing this code).
- Dims 4-27 24-bit binary ADDR_B0/B1/B2_LO/HI address encoding (`scale=10.0`).
- Dims 28-33 per-head position gates and AX-marker suppression.
- V slots 32..47 / 48..62 CLEAN_EMBED projections.

**lev_heads_4_11** (vm_step.py:7417-7700): symmetric heads for LEV
saved_bp / return_addr — not on the SI/LI/SC/LC critical path.

**suppress_heads_0_3** (l15_ops.py:896-1473, ~580 lines): the largest body.
Adds local-stack-slot binary rows (rows 4..43), one-hot rows (rows 43..58),
non-load suppression (`non_load_suppression = -1000000.0` at OP_JSR / OP_ENT
/ OP_LEA / OP_IMM), and pop-store equality blockers (row 42, e0). These
writes interact with the heads_0_3 lookup head Q/K rows by overlaying
identical row indices — a `DeclarativeAttentionHeadSpec` would need an
"append" mode to express this overlay because the spec's `q=(...)` tuple is
applied as a wholesale replace by `Primitives.generate_attention_head`.

Gap: every Q/K/V/O write in the five fragments above is imperative. The
SI/LI/SC/LC failures land here because:

1. The heads_0_3 head 0 binary address dims (4-27) must match the store's
   ADDR_B0/B1/B2 encoding for an LI/LC roundtrip. With dim_flow_audit
   showing `layer14_mem_generation` is the only ADDR_B0_LO/HI writer on the
   value path (slot 206), but L15 reads CLEAN_EMBED at MEM val byte
   positions — the encoding round-trips through the KV cache and is not
   expressible as a producer/consumer claim against the symbolic IR.
2. The suppress_heads_0_3 non-load suppression rows (OP_JSR / OP_ENT /
   OP_LEA / OP_IMM at -1000000) are not declared anywhere in the
   `Operation.reads` set, so the dim_contracts_audit cannot detect when an
   upstream change (e.g. OP_IMM repurposing for a new op family) silently
   breaks the load query gating.
3. The pop-store equality blocker rows (row 42, `0xe0` low/high
   coordinates) hard-code stack-frame byte 0 = 0xe0 — this is an artifact
   of the `_set_layer14_jsr_mem_default_suppress` family at
   `setup_helpers_l14.py:269` and is mechanically tied to that helper's
   layout.

Migration path (incremental):

(a) Express heads_0_3 dims 0-27 as `DeclarativeAttentionHeadSpec` with the
    `building_blocks_dsl.lookup_table_rules` helper for the binary address
    encoding (the helper already exists per `building_blocks_dsl.py:1026`
    referencing exactly this code pattern). Match scale=10.0, the spec's
    `q=` tuple receives 4 per-bit AP() entries per of 24 dims = 96 rows;
    `k=` mirrors. Byte-identity gate via `compare_symbolic_to_lowered_attn`.

(b) Express the suppress_heads_0_3 binary local-slot rows (rows 4-43) as
    a `DeclarativeAttentionHeadSpec` appendable overlay. This requires an
    IR feature: per-`AP(row, dim, weight)` entries should accumulate when
    two specs target the same head index (current behaviour replaces). The
    `RuntimeAttentionFragment` exists precisely to escape this; the
    migration target is to grow `AttentionHeadSpec` with an `append` flag.

(c) The pop-store equality blocker (row 42) and the `non_load_suppression`
    rows can move to a standalone `DeclarativeAttentionHeadSpec` per
    suppress fragment. The structure of those rows is mostly K[CONST] + Q[
    CONST,MARK_STACK0,ADDR_B0_LO+low,ADDR_B0_HI+high] one-hots — straightforward
    AP/AO listings, no overlay needed.

(d) Once (a)-(c) land, delete the imperative bodies and the
    `RuntimeAttentionFragment` wrappers. The IR builder becomes plain
    `AttentionOp` with `DeclarativeAttentionHeadSpec` entries.

---

## Recommended priority

1. **HIGHEST: migrate `_suppress_l15_lookup_heads_0_3`'s non-load suppression
   rows** to declarative AP() form (path (c) above). These are ~30 row writes
   that currently fully invisible to `dim_contracts_audit` and `verify_claims_static`.
2. **HIGH: migrate `_set_layer15_memory_lookup_heads_0_3` dims 0-3** to a
   `DeclarativeAttentionHeadSpec` with `lookup_table_rules` for dims 4-27.
   This makes the SI/LI/SC/LC critical path auditable end-to-end.
3. **MEDIUM: declare `ADDR_B0_LO` / `ADDR_B0_HI` in
   `layer14_mem_generation.reads`** — fixes the 2 known dim_contracts failures
   without behaviour change.
4. **LOW: delete `_set_layer7_memory_heads` / `_set_layer14_mem_generation`
   helper bodies** (cleanup; no live callers).

The 5 SI/LI/SC/LC smoke xfails will not flip green from any of these steps
in isolation — they cumulatively unblock the verifier path that currently
declares L15 memory lookup's load-side correctness opaque. The actual
roundtrip failure (per `diag_l14_jsr_mem.py` analysis and the run_vm.py:2370
SI override) is upstream in L6 STACK0 write units, not in these three
helpers; but the helpers' opacity blocks the verifier from confirming a fix
once it lands.
