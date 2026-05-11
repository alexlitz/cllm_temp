# V7 Heap Ops Neural Plan (LI/LC, SI/SC, LEA, AX-Merge)

Goal: Eliminate the four V7-REQUIRED handler-mode overrides in
`c4_release/neural_vm/run_vm.py::_dispatch_step` so `pure_neural=True`
produces correct results without Python fallback. The handler-mode
overrides in scope are:

| Block | Op(s) | Handler lines | Phase | Files |
|-------|-------|---------------|-------|-------|
| 10 | LI, LC | 1047-1060 | 7 | `_set_layer15_memory_lookup`, `_set_layer7_memory_heads` |
| 11 | SI, SC | 1061-1075 | 7 | `_set_layer14_mem_generation`, `_set_layer7_memory_heads` |
| 12 | LEA | 1081-1089 | 5 | `_set_layer7_operand_gather`, `_set_layer8_alu` |
| 13 | AX merge | (inside binary-pop block) | 3 | `CarryPropagationPostOp`, `BitwiseBytePropagationPostOp` |

Status (2026-05-11): All four blocks have **partial** neural-side
bakes; the remaining gaps are described per-block below. This doc
captures the precise layer/dim wirings needed to close each gap and
which handler line can be deleted once the bake works.

## 0. BD layout reference (compact dims)

The relevant compact dims used throughout this plan:

| Symbol | Dims | Purpose |
|--------|------|---------|
| `OP_LI`, `OP_LC`, `OP_SI`, `OP_SC`, `OP_LEA`, `OP_PSH` | one each | Opcode one-hots from L5 decode |
| `OP_LI_RELAY`, `OP_LC_RELAY` | one each | LI/LC flag broadcast from AX marker to AX bytes (L7 head 5) |
| `MEM_STORE`, `MEM_ADDR_SRC` | one each | Store-op marker (PSH/SI/SC=1) and addr source (0=SP for PSH, 1=STACK0 for SI/SC) |
| `ADDR_KEY` | 48 dims (3 nibbles × 16 one-hot) | Address keys for memory lookup |
| `ADDR_B0_LO/HI`, `ADDR_B1_LO/HI`, `ADDR_B2_LO/HI` | 16 each | Resolved address nibbles at query positions |
| `MARK_AX`, `MARK_STACK0`, `MARK_BP` | one each | Marker token flags |
| `OUTPUT_LO/HI` | 16+16 | Output byte per token position |
| `H1+AX_I`, `H1+SP_I`, `H1+BP_I`, `H1+MEM_I` | one each | Distance half-spaces from each marker |
| `CMP+0`, `CMP+3`, `CMP+7` | one each | PSH flag (CMP+0), POP-group flag (CMP+3), OP_LEA relay (CMP+7) |
| `IS_BYTE`, `BYTE_INDEX_0/1/2/3` | one each | Token-position flags inside register sections |

Constants: `S = 100.0` global scale, `HD = 64` head dim, `L = 15.0` typical attention strength.

## 1. LEA (block 12) — Phase 5

### Current state

The neural-side bake for LEA already exists across L7 and L8:

- **L7 head 1** (`_set_layer7_operand_gather`, `setup_helpers.py:710-729`)
  fires on `MARK_AX + OP_LEA` and gathers BP's value from the BP marker
  into `ALU_LO/HI` at the current AX marker. Distance from current AX
  marker to prev-step BP marker ≈ 25 tokens; ALiBi slope 0.5 gates this.

- **L8 ALU LEA lo nibble** (`_set_layer8_alu`, `vm_step.py:4847-4856`)
  256 units gated on `OP_LEA + MARK_AX` compute `ALU_LO[a] + FETCH_LO[b]
  → OUTPUT_LO[(a+b) mod 16]`. FETCH_LO is the immediate operand from
  L5 FETCH-rotation.

- **L8 LEA carry detection** (`vm_step.py:4890-4901`)
  120 units fire on overflow pairs (a+b≥16) and write `CARRY+0`.

- **L9 ALU LEA hi nibble** (`_set_layer9_alu`, `vm_step.py:5200-5220`)
  512 units (with and without carry-in) compute hi nibble of byte 0.

- **L8 LEA first-step AX byte 2** (`_set_layer8_alu`, `vm_step.py:5091-5104`)
  Hard-codes the byte-2 output for first-step LEA (BP = 0x10000 →
  byte 2 = 0x01) gated on `CMP+7 + H1[AX_I] + IS_BYTE + BYTE_INDEX_1
  + NOT HAS_SE`.

### Gap

The handler-mode override at `run_vm.py:1081-1089` runs only when
`pure_neural=False`. In `pure_neural=True` mode the bake above is
exercised. The two known failure modes:

1. **ENT-imm doesn't establish BP correctly in pure_neural**, so the
   prev-step BP marker's `OUTPUT_LO/HI` are stale. L7 head 1 then
   reads garbage and `OUTPUT_LO[0]` at AX marker → `imm` not
   `BP + imm`. This is the dependency tracked in
   `test_smoke_pure_neural.py::test_lea_basic` (xfail strict=False).
   **Not in this plan's scope** — owned by the parallel ENT/LEV agent.

2. **LEA byte 1-3 only handled for first-step** (the hard-coded
   `_set_layer8_alu` units 5091-5104). For non-first-step LEA where
   BP has a runtime value, bytes 1-3 of AX are not populated.

### Fix plan (deferred, depends on ENT/LEV)

- **Phase 5a (ENT-imm fix, separate agent)**: Make ENT/LEV produce
  correct BP byte 0-3 outputs in pure_neural so prev-step BP marker
  has valid `OUTPUT_LO/HI`.

- **Phase 5b (this scope)**: Extend `_set_layer7_operand_gather`
  head 1 to also gather BP bytes 1-2 (currently only byte 0) into
  `ALU_LO/HI` at AX bytes 1-2 positions. Mirror the
  `_set_layer8_sp_gather` pattern (heads 0-2 for SP bytes 0-2). Then
  the existing L8/L9 LEA ALU machinery handles all four bytes
  uniformly. Concrete wiring:
  ```
  base_h = j * HD  for j in {1, 2}  (skip head 0 — operand-A gather)
  Q: MARK_AX (initial) — REPLACE with H1[AX_I] + BYTE_INDEX_J + OP_LEA
  K: MARK_BP — KEEP, but add BYTE_INDEX_J on K side
  V: CLEAN_EMBED_LO/HI from BP byte J's token (already in V at base+1+k)
  O: ALU_LO/HI at AX byte J position (already in O at base+1+k)
  ```
  Update ALiBi slope[1] to favor the BP-byte-J distance (≈25 + j tokens).

- **Phase 5c**: Once 5a+5b land, delete `run_vm.py:1081-1089` (the
  handler override) and remove the `xfail` mark on `test_lea_basic`.

### Bake hand-off

Add a new compiler op `make_layer7_operand_gather_bp_bytes_op` at
phase=7.1 (after `layer7_operand_gather` at phase=7) that extends
heads 1-2 with the per-byte BP gather. Keep head 1's existing
byte-0 gather intact for backward compat with ADJ/ENT.

## 2. AX merge (block 13) — Phase 3

### Current state

The neural-side AX merge for multi-byte ALU results is **already
fully baked** in `l10_post_ops_combined` and `l10_post_op_attach`
(`unified_compiler/ops/l10_ops.py:288-348` and 351+):

- **CarryPropagationPostOp** × 3 (byte 0 no-cascade, byte 1 cascade,
  byte 2 cascade): ADD/SUB carry-out from byte N propagates into
  byte N+1 of `OUTPUT_LO/HI` (`vm_step.py:659-779`).

- **BitwiseBytePropagationPostOp**: AND/OR/XOR per-byte computation
  at AX byte positions 1-3 reading prev OUTPUT (current AX byte)
  and ALU (stack byte) via TEMP[4..6] op-relay flags
  (`vm_step.py:781-852`).

- **L10 attention head 4** broadcasts STACK0 byte values to ALU_LO/HI
  at AX byte positions 1-3 (so `BitwiseBytePropagationPostOp` reads
  the stack operand). See `make_layer10_attn_op` in `l10_ops.py`.

### Gap

The handler-mode override is the per-op block at
`run_vm.py:1026-1046` (binary-pop ALU). In `pure_neural=True` the
bake above is exercised, but two scenarios still fail per
`test_pure_neural_multibyte.py`:

1. **ADD/SUB byte-1 carry beyond byte 0** for results > 255
   (e.g., 200+100=300 should produce AX byte 1 = 1). The bake
   exists but the `CarryPropagationPostOp` units fire on
   `CARRY+add_carry_in`/`CARRY+sub_carry_in` at byte 1 positions.
   Current xfail in `test_add_overflow_300` /
   `test_add_overflow_510` cites
   "_set_layer9_alu carry handling + post_op CarryPropagationPostOp
   are not yet wired for multi-byte ADD results."

2. **SHL/SHR multi-byte high bytes**. SHL/SHR currently aren't
   handled by any post_op — they need a dedicated unit similar to
   `BitwiseBytePropagationPostOp` but with the shifted-byte
   computation (with carry across nibbles).

### Fix plan

- **Phase 3a (ADD/SUB carry)**: Audit
  `CarryPropagationPostOp._bake_weights` against the actual L10
  output values. The unit threshold (`-S * 9.5`) and the
  `CARRY+add_carry_in/sub_carry_in` discriminator may be off by
  ~0.5–1.0. Concrete debug step:
  1. Run `test_add_overflow_300` with `pure_neural=True`.
  2. Capture the unit activations at the AX byte-1 position
     (`H1[AX_I]=1 + IS_BYTE=1 + BYTE_INDEX_1=1`).
  3. Check whether `OUTPUT_LO/HI` at byte 1 reflects `+1` from the
     ADD carry-out at byte 0 (CARRY[1] from L9).
  4. If `CARRY[1]` is ≈0 at byte 1 (not propagated by L7), the
     issue is upstream — wire L7 head 5 (or another L8 broadcast
     head) to relay `CARRY+1`/`CARRY+2` from AX marker to byte
     positions.

- **Phase 3b (SHL/SHR high bytes)**: New `ShiftBytePropagationPostOp`
  modeled on `BitwiseBytePropagationPostOp`. For each byte position
  1..3, compute `(stack_byte << ax_byte_0) & 0xFF` (SHL) or
  `(stack_byte >> ax_byte_0) & 0xFF` (SHR) reading the shift-amount
  from the AX byte 0 value. Note that for shifts ≥ 8, the result
  needs cross-byte movement which a single byte-local unit can't
  do — defer the shift-amount-byte-aligned case to a later bake.

- **Phase 3c**: Delete the `_BINARY_POP_OPS` handler block at
  `run_vm.py:1026-1046` and re-enable the relevant xfails.

### Bake hand-off

The bake infrastructure already exists. Phase 3a is a tuning task
on `CarryPropagationPostOp` (no new ops needed). Phase 3b adds one
new PureFFN subclass + a `make_l10_shift_byte_propagation_op` at
phase=10.6.

## 3. LI/LC (block 10) — Phase 7

### Current state

The L15 memory lookup is heavily baked
(`_set_layer15_memory_lookup` in `vm_step.py:6178-6342`):

- **4 heads (0-3)** each select one output byte (head h → byte h).
- **Head 0** is gated by `OP_LI_RELAY + OP_LC_RELAY + CMP+3`
  (POP group). For LI/LC, OP_LI/LC_RELAY ≈1 at AX marker.
- **Dim 3** byte-selection ensures head h attends to MEM val byte h
  (at d=5,6,7,8 from MEM marker).
- **Dims 4-27** binary address encoding (24 bits, 3 bytes × 2
  nibbles × 4 bits). Q reads `ADDR_B*_LO/HI`, K reads `ADDR_KEY`.
- **V/O** copies `CLEAN_EMBED_LO/HI` of the matched MEM val byte
  to `OUTPUT_LO/HI`.

**For LI**: All 4 heads fire (each writing one byte) → AX = 4-byte
word.  **For LC**: Only byte 0 should be written (char is 1 byte);
bytes 1-3 must be cleared.

The address comes from prev-step AX (the pointer): L7 heads 2-4
gather prev AX bytes 0-2 into `ADDR_B0_LO/HI` ... `ADDR_B2_LO/HI`
at current AX byte positions
(`_set_layer7_memory_heads` `vm_step.py:4614-4643`).

### Gap

1. **LC byte 1-3 must be zero** — currently heads 1-3 fire for
   both LI and LC because the Q gate at dim 0 uses
   `BD.OP_LI_RELAY` *only* on head 0, but for heads 1-3 the gate
   is `L1H4[BP] - H1[BP]` (STACK0 area). At AX marker, neither
   of these is high, so the head doesn't fire — but if L7 also
   relays OP_LC to AX byte positions, the LC heads 1-3 contribution
   to OUTPUT must be zero. Verify by `test_pure_neural_heap_div.py::
   test_sc_then_lc`.

2. **LI word-wide address gather** — L7 heads 2-4 already gather
   prev AX bytes 0-2 into ADDR_B*. But:
   - The ALiBi slope from current AX position to prev AX byte
     position is ≈34 tokens. With slope 0.5, score ≈ −17. Combined
     with Q·K bonus of ≈15²·0.125 = 28, the head wins, but only
     marginally.
   - Need to verify the K-side `BYTE_INDEX_J + H1[AX_I]` matches
     uniquely at the prev AX byte position (no false matches at
     current AX bytes).

3. **MEM_VAL_B0/1/2/3 flags** — L0 FFN or token embedding writes
   one of `MEM_VAL_B*` based on the distance from MEM marker for
   each byte position. Verify this flag is present at the MEM val
   byte tokens (needed for head h's byte selection at dim 3).

### Fix plan

- **Phase 7a (LC byte clearing)**: Add an L15 head 0 V/O override
  that zeros `OUTPUT_LO/HI` for bytes 1-3 when `OP_LC_RELAY` is
  active at AX bytes 1-3. Or, more cleanly, add an L16 FFN unit
  that fires on `OP_LC_RELAY + (BYTE_INDEX_1|2|3)` and writes
  `-OUTPUT_LO/HI` (clearing). Concrete unit:
  ```
  for k in 16:
    W_up[u, OP_LC_RELAY] = S
    W_up[u, IS_BYTE] = S
    W_up[u, H1[AX_I]] = S
    W_up[u, BYTE_INDEX_0] = -S * 4  # block at byte 0
    b_up[u] = -S * 2.5
    W_gate[u, OUTPUT_LO + k] = 1.0
    W_down[OUTPUT_LO + k, u] = -2.0 / S
  ```

- **Phase 7b (LI word-wide address)**: No change needed if Phase 7a
  works for the byte-0 case. Verify with
  `test_pure_neural_heap_div.py::test_si_then_li`.

- **Phase 7c**: Delete the LI/LC handler-mode block at
  `run_vm.py:1047-1060` once the above tests pass.

### Bake hand-off

Add `make_l16_lc_byte_clear_op` at phase=16.1 as `kind="block"`
with `layer_idx=16` and `migrated=True`. The op writes 16+16
clearing units into `block.ffn` and reads from `OP_LC_RELAY`,
`IS_BYTE`, `H1`, `BYTE_INDEX_*`, `OUTPUT_LO/HI`.

## 4. SI/SC (block 11) — Phase 7

### Current state

The L14 MEM generation is fully baked for both SI/SC and PSH paths
(`_set_layer14_mem_generation` in `vm_step.py:5844-6097`):

- **Heads 0-3** (addr generation) write the destination address
  byte j to the MEM addr_b{j} token position.
- **Heads 4-7** (val generation) write the source value byte j to
  the MEM val_b{j} token position.
- **MEM_ADDR_SRC** flag (0=SP for PSH, 1=STACK0 for SI/SC)
  controls whether the address comes from the SP marker (PSH) or
  the STACK0 byte 0 position (SI/SC).
- **OP_JSR/OP_ENT** flags (also relayed by L7 head 7 to MEM marker)
  switch the val heads from AX to STACK0 source (for JSR=return-addr,
  ENT=saved-BP).

### Gap

The L14 MEM generation requires:

1. The neural-side **MEM token sequence emission** at the end of a
   SI/SC step (i.e., the autoregressive loop must actually generate
   the [MEM, a0, a1, a2, a3, v0, v1, v2, v3] tokens). This is
   already handled by the `_inject_mem_section` shim in
   `run_vm.py:1067` (SI) and `run_vm.py:1062-1068` (handler-mode).
   In `pure_neural`, the MEM persistence shim at
   `run_vm.py:867-890` extracts the MEM section from neural output
   and persists it. But for SI/SC, this only works if the model
   actually emits the MEM section autoregressively. Verify
   `_MEM_STORE_OPS` includes SI/SC (it does, at line 65).

2. **MEM_ADDR_SRC must be set to 1 for SI/SC** at the MEM marker so
   the heads 0-3 use the STACK0-source dim-2 bonus. This is done
   via L4/L5 FFN setting `MEM_ADDR_SRC` based on OP_SI/OP_SC. Need
   to verify the L5 opcode decode sets this correctly.

3. **STACK0 byte 0 contains the destination address** for SI/SC:
   the C4 convention is `pop addr; *addr = AX`. So before the SI
   step, PSH put the address onto the stack. At the SI step,
   STACK0_byte0 = address.

### Fix plan

- **Phase 11a**: Verify L5 opcode decode writes `MEM_STORE = 1`
  and `MEM_ADDR_SRC = 1` for OP_SI/OP_SC at PC marker.
- **Phase 11b**: Verify L6 attn head 6 (PSH/store flag relay)
  also relays `MEM_STORE/MEM_ADDR_SRC` from the PC marker to the
  MEM marker for SI/SC steps (currently it relays for PSH).
- **Phase 11c**: Once 11a+11b pass and the model emits a valid
  MEM section with correct addr/val bytes, delete the SI/SC
  handler-mode block at `run_vm.py:1061-1075`.

### Bake hand-off

If L5/L6 already set `MEM_STORE/MEM_ADDR_SRC` for SI/SC, no new
op is needed. If not, extend the L5 opcode decode op and the
L6 relay head to include OP_SI/OP_SC alongside OP_PSH. Phase=5
and phase=6 respectively.

## 5. Coordination

- **AX merge (block 13)** has overlap risk with the parallel
  Phase 1 multi-IMM agent (which uses
  `CarryPropagationPostOp`). If Phase 2 byte-0 lands first, the
  AX-merge work simplifies because the discriminator weights are
  re-tuned for the new byte-0 baseline.
- **LEA (block 12)** depends on the parallel ENT/LEV agent's BP
  establishment. Phase 5a is owned by that agent; Phase 5b
  (BP byte 1-2 gather) is in this plan's scope but should land
  *after* 5a.
- **LI/LC and SI/SC** are independent of the parallel work.

## 6. Test coverage

| Test | Status | Blocked by |
|------|--------|------------|
| `test_smoke_pure_neural.py::test_lea_basic` | xfail | Phase 5a (ENT BP) |
| `test_smoke_pure_neural.py::test_adj_sp` | xfail | Phase 2 ADJ migration |
| `test_pure_neural_heap_div.py::test_si_then_li` | xfail | Phase 7b |
| `test_pure_neural_heap_div.py::test_sc_then_lc` | xfail | Phase 7a |
| `test_pure_neural_multibyte.py::test_add_overflow_300` | xfail | Phase 3a |
| `test_pure_neural_multibyte.py::test_add_overflow_510` | xfail | Phase 3a |

When all four blocks land, the four corresponding handler-mode
override blocks in `run_vm.py::_dispatch_step` can be deleted:
- Lines 1047-1060 (LI/LC)
- Lines 1061-1075 (SI/SC)
- Lines 1081-1089 (LEA)
- Lines 1026-1046 (AX merge / binary-pop)

## 7. Rollout order

1. **Block 13 Phase 3a (AX merge ADD/SUB)** — simplest, no new
   bakes, just `CarryPropagationPostOp` weight tuning. Land first.
2. **Block 12 Phase 5b (BP byte 1-2 gather)** — additive, doesn't
   touch existing bakes. Land after the parallel ENT/LEV agent
   completes Phase 5a.
3. **Block 10 Phase 7a (LC byte clearing)** — additive L16 FFN
   unit. Independent. Land any time.
4. **Block 11 Phase 11a-b (SI/SC MEM_STORE/ADDR_SRC verification)**
   — verification + small fix. Land after Block 10 (so the L15
   readback path is verified first).
5. **Block 13 Phase 3b (SHL/SHR high-byte)** — new PureFFN class.
   Optional unless smoke tests require multi-byte shifts.

Each landing deletes its corresponding handler-mode block and
removes the xfail marks.

## 8. Source pointers

Core files for this work:

- `c4_release/neural_vm/run_vm.py::_dispatch_step` — handler-mode
  overrides to delete.
- `c4_release/neural_vm/vm_step.py`:
  - `_set_layer7_operand_gather` — LEA BP byte-0 gather.
  - `_set_layer7_memory_heads` — LI/LC address gather, op flag relay.
  - `_set_layer8_alu` — LEA byte-0 ALU + first-step byte-2 hard-code.
  - `_set_layer9_alu` — LEA byte-0 hi nibble.
  - `_set_layer14_mem_generation` — SI/SC/PSH MEM token generation.
  - `_set_layer15_memory_lookup` — LI/LC + LEV memory read.
  - `CarryPropagationPostOp`, `BitwiseBytePropagationPostOp` —
    AX merge byte propagation.
- `c4_release/neural_vm/setup_helpers.py::_set_layer7_operand_gather`
  — L7 head 1 LEA/ADJ/ENT BP/SP gather (canonical location).
- `c4_release/neural_vm/unified_compiler/ops/`:
  - `l7_ops.py::make_layer7_memory_heads_op` — LI/LC relay bake.
  - `l8_ops.py::make_layer8_alu_op` — LEA ALU bake.
  - `l10_ops.py::make_l10_post_ops_combined`,
    `make_l10_post_op_attach_op` — AX merge bakes.
  - `l14_ops.py`, `l15_ops.py` — MEM gen + memory lookup bakes.
- `c4_release/tests/test_pure_neural_heap_div.py` — LI/LC/SI/SC
  parity tests.
- `c4_release/tests/test_pure_neural_multibyte.py` — multi-byte
  ALU AX merge tests.
- `c4_release/tests/test_smoke_pure_neural.py::TestSmokePureNeuralAddress`
  — LEA smoke.
