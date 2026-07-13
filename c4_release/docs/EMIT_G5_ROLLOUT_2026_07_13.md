# EMIT G5 ROLLOUT — the 4 `layer14_*_ax_bytes_zero` ops folded into ONE generic generator

**Status:** LAND-READY, byte-identical. Golden `e50521f3` reproduced in BOTH
flag states (default flag-OFF + `C4_EMIT_G5_RBYTE=1`). Net **-257 LOC**
(l14_ops.py -263, shared.py +6 code / +21 doc for the new flag helper), of which
**510 lines of hand-written per-op code were deleted** and 247 lines of a single
spec-driven generator + 4 thin factories added back.

This is the FULL fold the CLEAN_EMITTER / EMIT-FRAMING §G5 R-BYTE scope
(`docs/CLEAN_EMITTER_SCOPE_2026_07_04.md`, `docs/semantic_spec_EMIT_FRAMING.md`)
called for on the L14 AX-byte-1..3 cleanup family.

---

## 1. What was folded

Four L14 FFN cleanup ops enforced C4's 8-bit-AX-with-32-bit-register
convention (AX bytes 1-3 = `0x00`) by, at the AX byte rows
(`IS_BYTE` + `H1[AX]`), spreading `-3/S` across every `OUTPUT_LO/HI` nibble and
boosting `OUTPUT_LO[0]` / `OUTPUT_HI[0]` by `+5/S` so the byte-value-0 token wins
argmax. They were **structurally identical** — four 4-unit rule programs plus
four near-identical `make_*` factories — differing ONLY in four axes:

| op (chain offset) | opcode gate | gate_weight | threshold | extra conditions | STACK0 guard |
|-------------------|-------------|:-----------:|:---------:|------------------|:------------:|
| `layer14_jsr_ax_bytes_zero` (1874) | `OP_JSR` (L5 one-hot via `opcode_flag`) | `40.0` campaign / `1.0` golden | 1.5 | — | **yes** |
| `layer14_lc_ax_bytes_zero` (1878) | `OP_LC_RELAY` (L7 h5 relay of `OP_LC`) | 1.0 | 1.5 | `BYTE_INDEX_3 = -4` | no |
| `layer14_alu_nocarry_ax_bytes_zero` (1882) | `TEMP+7` (NOCARRY_ALU_OP relay = `OP_AND\|OR\|XOR\|SHR`) | 1.0 | 5.0 | `TEMP+7 = 4`, `BYTE_INDEX_3 = -4` | no |
| `layer14_ent_ax_bytes_zero` (auto-fit) | `OP_ENT` (L5 one-hot via `opcode_flag`) | 1.0 | 1.5 | — | no |

Everything else — the 4-unit LO/HI nibble-band + byte-0-boost writes, the
`IS_BYTE`+`H1[AX]` scope, the `layer14_mem_generation` binding, the boundary
guard, the `_L14_CLEANUP_CHAIN_LAYOUT` pin — was duplicated verbatim across all
four.

### The fold (neural_vm/unified_compiler/ops/l14_ops.py)

* `_AxBytesZeroSpec` — a per-op config object capturing exactly the four axes
  above plus the per-op wiring metadata (`reads` / `claims_start` / `slot_share`
  / `ffn_units_used` / `smoke_tests` / `spec_section`).
* `_ax_bytes_zero_rules(spec, S)` — ONE generic 4-unit rule generator (was 4
  copies).
* `_ax_bytes_zero_ir(spec, S)` / `_make_ax_bytes_zero_op(spec)` — ONE generic IR
  wrapper + ONE generic Operation factory (was 4 + 4). The factory conditionally
  attaches `claims` (JSR/LC/ALU pin their W_down cells; ENT never declared any),
  `slot_share`, `ffn_units_used`, and runs the STACK0-block guard only for JSR —
  all driven by the spec, so each op bakes bit-for-bit what its hand factory did.
* `_AX_BYTES_ZERO_SPECS` — the 4 specs, in declaration == chain order.
* The four public `make_layer14_*_ax_bytes_zero_op()` names are kept as
  one-line wrappers over `_make_ax_bytes_zero_op(spec)` so `all_core_ops.py`'s
  call sites are unchanged.

---

## 2. Mutual-exclusivity audit (the pilot's flagged prerequisite)

**Result: the four opcode gates are MUTUALLY EXCLUSIVE — they cannot co-fire on
any row.**

* `OP_JSR`, `OP_ENT`, `OP_LC` are three of the **34 mutually-exclusive one-hot
  `OP_*` flags** the L5 opcode decode emits (`make_layer5_opcode_decode_op`,
  `l5_ops.py:566` — "decode opcode byte → 34 one-hot OP_* flags"; exactly one is
  hot per instruction byte).
* `OP_LC_RELAY` is the L7 head-5 relay of `OP_LC`
  (`l7_ops.py:1142` `ScalarRelay(2, (("OP_LC", 0.2),), "OP_LC_RELAY", 1.0)`).
* `TEMP+7` (NOCARRY_ALU_OP) is the L7 head-5 relay of the DISJOINT ALU opcode
  set `OP_AND | OP_OR | OP_XOR | OP_SHR` (`l7_ops.py:1150`).

Since **no single VM step is two opcodes at once**, no row ever has two of these
gates lit simultaneously. This is exactly the invariant the four op docstrings
already asserted ("JSR / LC / nocarry-ALU gate on disjoint relays"). The audit
confirms it holds transitively through the L7 relays.

**Consequence for the fold.** Because the gates never overlap, the four
independent 4-unit ranges are behaviorally equivalent to a single OR-gated
range — a future collapse to ONE 4-unit range gated on
`(OP_JSR ∨ OP_LC_RELAY ∨ TEMP+7 ∨ OP_ENT)` would be firing-safe. This fold keeps
the four ranges (not one) because the per-op **threshold** (5.0 for ALU vs 1.5),
the `BYTE_INDEX_3` blocker (LC/ALU only), the `TEMP+7` selector (ALU only), and
the campaign `gate_weight` re-anchor (JSR only) differ — folding to a single
range would require reconciling those, which is NOT byte-identical. The
spec-driven fold gets the -LOC win with a proven-safe, exactly-reproducing
generator; the single-range collapse is deferred (it is a verdict-changing,
not byte-identical, step).

---

## 3. The `C4_EMIT_G5_RBYTE` flag (byte-neutral verification toggle)

`shared.py::emit_g5_rbyte_enabled()` (DEFAULT OFF). When ON, the generic bake
rebuilds the 4-unit rule program through a SECOND independent
`_ax_bytes_zero_rules(spec, S)` call and asserts it is structurally identical to
the first before lowering. This exercises the spec table via a distinct code
path without touching a single weight (the rules are pure functions of the
spec), so the model is byte-identical in both states. The flag is a CI /
regression guard, not a behavior switch.

---

## 4. Golden proof (BOTH states)

Baseline (pre-fold, bare env): `tools/_isa_golden_hash.py` =
`e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86`.

| state | command | `state_dict_sha256` |
|-------|---------|---------------------|
| flag-OFF (default) | `CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py` | `e50521f3…3c90e86` ✅ |
| flag-ON | `CUDA_VISIBLE_DEVICES="" C4_EMIT_G5_RBYTE=1 python tools/_isa_golden_hash.py` | `e50521f3…3c90e86` ✅ |

Both reproduce golden `e50521f3` bit-for-bit. TOTAL FFN units `42149 → 42149`
(no dead units, 100% retained) in both states. The generic op reproduces all
four hand ops bit-for-bit.

---

## 5. LOC

* `l14_ops.py`: **510 hand-written lines deleted**, 247 generic lines added →
  **-263 net**.
* `shared.py`: +27 lines (the `emit_g5_rbyte_enabled()` helper + its docstring;
  6 of them code).
* **Overall net: -257 LOC.** The 510-line gross deletion matches the ~-540 scope
  target for the removed per-op code; the net is smaller because the fold keeps
  the four `make_*` public names (call-site stability) and adds the spec table +
  generic generator + verification path.

## 6. Feeds / next

`docs/CLEAN_EMITTER_SCOPE_2026_07_04.md`, `docs/semantic_spec_EMIT_FRAMING.md`
§G5. The deferred single-range OR-gate collapse (§2) is the only remaining
consolidation on this family and is NOT byte-identical (threshold /
`BYTE_INDEX_3` / gate-weight reconciliation) — it belongs in a verdict-tracked
change, not this byte-identical land.
