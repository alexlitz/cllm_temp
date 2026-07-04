# IMM — the FIRST fully-derived opcode (task #392)

**Status:** LANDED. IMM is 100% derived from spec — decode + relay +
value-route — with ZERO hand-authored IMM rules.
**Flag:** `C4_DERIVE_IMM` (default OFF; IMPLIES `C4_DERIVE_DECODE`).
**Golden gate:** `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb`
(`81557d21`) — UNCHANGED both flag-OFF **and** flag-ON.

This closes the ONE lowering gap the DECODE pilot named
(`docs/DERIVE_DECODE_PILOT_2026_07_03.md` §G-IMM-RELAY) plus wires the
near-closed value route (§G-IMM-ROUTE), so every IMM opcode-specific stage now
derives from a generic engine. IMM is the first opcode whose FULL frame is a
generic lowering with no per-opcode exceptions.

## What was proven

IMM (`AX = imm`) has exactly three opcode-specific stages across its frame
(the DECODE pilot classified fetch/emit/ENT-frame as SHARED substrate, not
IMM's to derive):

| stage | mechanism | derived by | status |
|-------|-----------|------------|--------|
| 1 decode → `OP_IMM` | L5 main-at-AX + first-step-PC nibble-AND rules | `decode_band` (task #391) | DERIVED, byte-identical |
| 3 relay `OP_IMM` → byte positions | L8 head-4 marker-broadcast attention head | **`marker_broadcast` (NEW)** | DERIVED, byte-identical |
| 4 value route imm → OUTPUT | L8 32-rule per-nibble `AX_CARRY`→`OUTPUT` copy | **`value_route` (NEW)** | DERIVED, byte-identical |

`C4_DERIVE_IMM=1` routes all three through the generic engines. The whole-model
`state_dict` SHA256 is byte-for-byte identical to the hand-authored golden
(`81557d21`), so IMM is 100% derived with zero weight drift, by construction.

## GAP-PRIMITIVE #1 — the `marker_broadcast` head generator

`neural_vm/unified_compiler/isa_semantics_dsl.py`, new generator (sibling of
`decode_band` / `cam_lookup` / `cross_step_carry`).

### The gap it closes

Every multi-byte opcode decodes its `OP_<NAME>` marker onto the STEP's marker
row (e.g. `MARK_AX`), but the later multi-byte routing / value-route FFN gates
on that flag at the op's BYTE-POSITION rows. Something must BROADCAST the flag
from the marker row forward to the byte positions. This is NOT a `cam_lookup`
(there is no content ADDRESS key — the head attends from a byte position to ITS
OWN step's marker row, a fixed marker-bank slot). It is a
"copy-my-flag-to-my-own-byte-positions" positional relay. It had no generator;
it was hand-built once (the L8 head-4 `layer8_op_imm_relay`) and would be
re-hand-built for every future multi-byte opcode.

### API

- **`MarkerBroadcastSpec`** — `name`, `fire_slot_dim` (the resolved
  `H1+<marker_bank_index>` byte-position anchor), `source_marker` (the marker
  row the flag was decoded onto), `broadcast_bands` (the flag band(s) to copy),
  `weight` (the shared Q/K projection weight), the optional CONST-anchored
  confirm slot (`gate_slot` + `gate_*_weight`), `alibi_slope` (recency pin), and
  `step_window`. Every field is a VARYING datum; the head STRUCTURE is fixed.
- **`MarkerBroadcastBand`** — one `source_band → target_band` copy block
  (`width`, `v_slot_base`, `o_scale`). IMM: one 1-wide `OP_IMM → OP_IMM`.
- **`MarkerBroadcastBundle`** — `head_spec_builder(dim_positions, head_idx)`
  (reproduces the head EXACTLY) + structural `head_reads` / `head_writes`.

### What it re-expresses byte-identically

`marker_broadcast` re-expresses the hand-authored L8 head-4 OP_IMM relay
(`l8_ops._layer8_op_imm_relay_head_spec`) cell-for-cell:

- **fire-site Q** (slot 0): `IS_BYTE`@20, `H1+AX_I`@20, `CONST`@−30 — fire at
  the AX byte positions.
- **source-select K** (slot 0): `MARK_AX`@20, `IS_BYTE`@−200, `CONST`@10 —
  attend BACK to the step's own AX marker row.
- **confirm slot** (slot 1): Q `IS_BYTE`@500 + `CONST`@−500; K `CONST`@5.
- **broadcast V/O**: V `OP_IMM`@1.0 (slot 0) → O `OP_IMM`@1.0.
- `alibi_slope=0.5`, `step_window=CURRENT_STEP_ONLY`, `head_idx` from the L8
  layout.

The re-expression is gated by `test_marker_broadcast_reexpresses_l8_op_imm_relay`
(cell-for-cell against a fresh hand reconstruction) +
`test_marker_broadcast_reexpresses_live_l8_relay_spec` (against the LIVE
`l8_ops` spec) + the whole-model golden hash flag-ON.

## The `value_route` generator (§G-IMM-ROUTE, near-closed → closed)

Same module. A gated per-nibble VALUE copy: the fetched immediate (staged per
nibble in `AX_CARRY_LO/HI`) is copied into `OUTPUT_LO / OUTPUT_HI_THIS_STEP` at
the AX byte positions, under a shared AND-context, gated per cell.

- **`ValueRouteSpec`** — `name`, `conditions` (the shared AND-context),
  `threshold`, `channels`.
- **`ValueRouteChannel`** — one `source_band → target_band` per-nibble channel
  (`width`, `write_scale`, `rule_name_fn`). IMM: `AX_CARRY_LO → OUTPUT_LO` (16)
  + `AX_CARRY_HI → OUTPUT_HI_THIS_STEP` (16).
- **`ValueRouteBundle`** — `rules_builder()` + structural `reads` / `writes`.

Re-expresses `l8_ops._layer8_multibyte_routing_rules` — all **32/32** rules
field-for-field identical (`test_value_route_reexpresses_live_l8_multibyte_routing`).
Role-meaningful family conditions (`OP_IMM`, `MARK_AX`) authored via `dim_ref`
(Phase 7.E), so the semantic-dim-ref ratchet stays at baseline.

## Wiring (flag `C4_DERIVE_IMM`)

- **decode**: `l5_ops._derive_decode_enabled()` returns True when
  `C4_DERIVE_DECODE` OR `C4_DERIVE_IMM` — IMM's decode stage derives through
  `decode_band` (the whole ISA table derives byte-identically, so implying it is
  golden-neutral).
- **relay**: `l8_ops._layer8_op_imm_relay_head_spec` routes through
  `marker_broadcast(_op_imm_relay_marker_broadcast_spec())` when
  `derive_imm_enabled()`; the hand path is the default.
- **value route**: `l8_ops._layer8_multibyte_routing_rules` routes through
  `value_route(_layer8_multibyte_routing_value_route_spec(S))` when
  `derive_imm_enabled()`; the hand path is the default.
- `C4_DERIVE_IMM` registered in BOTH `full_vm_compiler_dynamic.py` cache-key
  snapshots (ON/OFF never share a memo / disk entry).

The only two ops declaring `opcodes={"OP_IMM"}` are exactly the relay head and
the value-route FFN — both now derived; IMM's decode derives via `decode_band`.
There is no other IMM-specific hand-authored rule family.

## Gates

1. **GATE 1 (flag-OFF golden)**: `C4_DERIVE_IMM=0 tools/_isa_golden_hash.py` =
   `81557d21` (default path untouched).
2. **GATE 2 (derived == hand-authored, byte-identical)**:
   `C4_DERIVE_IMM=1 tools/_isa_golden_hash.py` = `81557d21`. A byte-identical
   `state_dict` ⇒ identical model ⇒ identical verdict on EVERY program (a
   strictly stronger equivalence than any per-program run).

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py                  # 81557d21 (flag-OFF)
CUDA_VISIBLE_DEVICES="" C4_DERIVE_IMM=1 python tools/_isa_golden_hash.py  # 81557d21 (derived)

# unit + re-expression tests (fast):
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_isa_semantics_dsl.py \
    -k "marker_broadcast or value_route"

# whole-model byte-identity gate (slow, captures ON==OFF in one session):
CUDA_VISIBLE_DEVICES="" C4_ISA_HASH_TEST=1 PYTHONHASHSEED=0 python -m pytest \
    tests/test_isa_semantics_dsl.py::test_whole_model_hash_derive_imm_byte_identical \
    --runslow
```

## Bottom line

The gap-primitive `marker_broadcast` — the ONE lowering gap the pilot named —
is BUILT as a general generator and proven to re-express the hand-authored relay
head byte-identically. With `value_route` and the already-merged `decode_band`,
the FULL IMM opcode now derives from spec with zero hand-authored IMM rules,
byte-for-byte identical to the golden. **IMM is the first 100%-derived opcode.**
`marker_broadcast` is reused by every future multi-byte opcode's relay.
