# B6-G: L7-L9 Structural-Signal Audit

**Branch base:** `speedup-cache-and-buckets @ 4d069f7`
**Worktree:** `agent-a59967b126a1dc09c`
**Deliverable:** Design audit — names what upstream signals L10 needs, and what L7-L9 currently
fail to provide. **No code changes**; companion B4-H plan (`proposal/l10-tail-correction-family`)
covers the L10 side of the same wishlist.

---

## TL;DR

The L10 tail-correction family (rules `tail_mem_store_addr0_*` and `tail_sp_marker_*` at
`l10_ops.py:3450-4100`) is structurally weak because the **upstream structural signals**
produced by L7/L8/L9 do not carry enough invariant information to discriminate the
contradictory cases that B3-gamma, B5-J, B2-A, B2-B, and B3-eta each independently surfaced.

Concretely, the L10 family must today reverse-engineer four pieces of information that
**should be one-hot bits emitted upstream**:

1. **"Is this MARK_SP row's byte 0 actually 0xF8?"** — needed by rule A. Currently inferred
   from `OUTPUT_LO+8 / OUTPUT_HI+15` residue, which is a leaky proxy (B5-J).
2. **"Is the staged ADDR_B0 a valid in-step gather (vs. residue from prior steps)?"** — needed
   by rules C/E/F/G/H/K/L/M/N/O/P/Q/R. There is no in-step `ADDR_B0_*_VALID` flag, so the L10
   family discriminates via ALU/CMP/PSH witness combinations (B4-H).
3. **"Which SP byte 0 value should the MARK_SP row carry-forward this step?"** — needed
   structurally by all SP marker rows. The L3 carry-forward attention SP head (head 2) writes
   `EMBED_LO/HI` from the previous SP byte, but there is no in-step *correction* signal
   distinguishing "SP unchanged" vs. "SP incremented by ADJ" vs. "SP decremented by PSH" vs.
   "SP restored by LEV".
4. **"Did the most recent SP delta complete?"** — `HAS_SE` answers "has any step ended" but not
   "is the SP byte-0 column finalized for *this* step". B5-J's promotion of `HAS_SE` from -100
   to -1e9 confirmed this is a lifecycle gap.

The recommended fix path is to teach L7-L9 to emit four small structural one-hot dims
(detailed in Section 2 below). With those landed, the L10 tail-correction family shrinks from
~18 rules to ~5 declarative dispatchers that read the upstream evidence directly.

---

## Section 1: Existing structural signals

Tables enumerate every dim L7/L8/L9 currently writes. Columns:
- **Dim** — the `_SetDim` name (see `vm_step.py:2215-2470`).
- **Producer op** — `make_*_op` factory + phase + head/unit when relevant.
- **Encodes** — what the value means.
- **Position** — where in the 35-token step layout the write lands (AX/SP/BP/STACK0/MEM marker,
  or byte 0/1/2/3 of those, or "broadcast").
- **Reliability** — `solid` / `partial` / `proxy` / `stale-prone`.

### 1.1 L7 outputs (`l7_ops.py`)

| Dim | Producer op | Encodes | Position | Reliability |
|-----|-------------|---------|----------|-------------|
| `ALU_LO+k`, `ALU_HI+k` (16+16) | `layer7_operand_gather` head 0 (phase=7) | Operand A nibble copied from `CLEAN_EMBED_LO/HI` at prev STACK0 byte 0 | AX marker | solid for ADD/SUB/MUL/etc.; relies on previous STACK0 byte 0 being the operand |
| `ALU_LO+k`, `ALU_HI+k` (alt) | `layer7_operand_gather` head 1 (phase=7) | LEA/ADJ/ENT destination addr copied from BP/SP `OUTPUT_LO/HI` | AX marker | solid; key for LEA semantics |
| `OP_LI_RELAY`, `OP_LC_RELAY` | `layer7_memory_heads` head 5 (phase=7) | LI/LC opcode flag broadcast to byte rows | AX byte positions | solid |
| `CMP+0..7` (scalar relays) | `layer7_memory_heads` head 5/6 (phase=7) | PSH (CMP+0), ADJ (CMP+2), ENT (CMP+3), JSR (CMP+4), POP-group flags | STACK0 / SP marker | solid for primary use; **`CMP+3` POP flag is the discriminator used by L10 tail rules** but it does not carry SP-delta semantics |
| `PSH_AT_SP` | `layer7_memory_heads` head 6 (phase=7) | PSH flag relayed to SP/STACK0 marker | SP marker | solid; used by L9 ALiBi and L10 tail rules |
| `TEMP+3..9` | `layer7_memory_heads` head 5 (phase=7) | AND/OR/XOR/NOCARRY/ADD/SUB byte propagation flags | AX byte positions | solid |
| `ADDR_B0_LO/HI..ADDR_B2_LO/HI` (96 dims) | `layer7_memory_heads` heads 2-4 (phase=7) | Previous-AX bytes 0/1/2 gathered into staging dims for the multi-byte fetch path | AX byte positions | solid for IMM multibyte; **does NOT encode current step's SP byte 0** |
| `OP_LI_RELAY`, `OP_LC_RELAY` (V slot 9 NOCARRY) | head 5 | NOCARRY_ALU_OP relay to TEMP[7] | AX byte | solid |
| `OP_JSR`, `OP_SI`, `OP_SC` | head 5 V slot 8/10/11 (back-written) | broadcast opcode flag to AX byte positions | AX byte | solid |
| `MEM_STORE`, `MEM_ADDR_SRC`, `OP_JSR`, `OP_ENT` (V slots at MEM marker) | head 7 (phase=7) | MEM flag broadcast across MEM section bytes | MEM marker + MEM byte positions | solid |

**Notable absences**: L7 does NOT produce any signal that says "this row's SP byte 0 = 0xF8 (or
the actual SP value)". L7 also does not produce a `STEP_IS_FRESH` lifecycle flag at SP/BP/MEM
marker rows.

### 1.2 L8 outputs (`l8_ops.py`)

| Dim | Producer op | Encodes | Position | Reliability |
|-----|-------------|---------|----------|-------------|
| `OUTPUT_LO+k`, `OUTPUT_HI+k` (legacy ALU result) | `layer8_alu` (phase=8.2) | Lo nibble of ADD/SUB/CMP/LEA result | AX byte positions | solid for the legacy lookup ALU path |
| `CARRY+0` | `layer8_alu` (phase=8.2) | Intra-byte ADD carry/SUB borrow | AX byte positions | solid (consumed by L9) |
| `CMP_GROUP` | `layer8_alu` (phase=8.2) | EQ/NE/LT/GT/LE/GE-active gate at AX marker | AX marker | solid |
| `OUTPUT_LO/HI` (multibyte route) | `layer8_multibyte_routing` (phase=8.3) | Route `AX_CARRY_LO/HI` -> `OUTPUT_LO/HI` at AX byte positions when IMM is active | AX byte positions | solid |
| `OP_IMM` relay | `layer8_op_imm_relay` (phase=8.4) | IMM flag broadcast to AX byte positions | AX byte | solid |
| `AX_CARRY_LO/HI` (32 dims) | `layer8_multibyte_fetch_bake` (phase=8.1) | Multi-byte IMM fetch result | AX marker (op rebakes also at AX byte positions) | solid; the head-6 refresh op (`layer8_head6_ax_carry_refresh`, phase=8.05) is wired but disabled by default in the production path |
| `ADDR_B0_LO/HI..ADDR_B2_LO/HI` (96 dims) | `layer8_sp_gather_bake` heads 0-2 (phase=8.0) | SP bytes 0/1/2 gathered into ADDR_B* staging for LI/LC/SI/SC | STACK0 marker | partial — the gather only fires at the STACK0 marker, NOT at MARK_SP or MARK_MEM. **L10 rules that consume `ADDR_B0_LO+8 / ADDR_B0_HI+15` at MARK_MEM rows are reading the value via residual leakage from STACK0** |
| `ALU_LO/HI` (mem->ALU at AX) | `layer8_mem_to_alu` head 5 (phase=8.45, disabled by default) | Direct mem[SP] lookup -> ALU operand B at AX marker | AX marker | gated `enable=False` in production |
| `AX_FULL_LO/HI` | `layer8_mem_to_alu` head 7 (phase=8.45, disabled by default) | MEM val byte 1 -> AX_FULL at AX marker for SHL/MUL/SHR wide ops | AX marker | gated `enable=False` |

**Notable absences**:
- L8 ADDR_B0 gather (phase=8.0) fires at `MARK_STACK0` only — the L10 family that reads
  `ADDR_B0_LO+lo` / `ADDR_B0_HI+hi` at `MARK_MEM` rows is reading a residual leak, not a
  fresh in-step write.
- L8 does not emit a signal saying "the ADDR_B0 staging is valid for this step at this
  marker row" — L10 cannot tell whether `ADDR_B0_LO+8 / ADDR_B0_HI+15` is the *current*
  step's SP-derived address or a stale one.
- L8 does not produce a "SP_BYTE0_IS_<X>" one-hot at the SP marker row. SP byte 0 only
  appears as a `CLEAN_EMBED` value at the SP byte 0 token position, not as a marker-row
  flag that L10 can read.

### 1.3 L9 outputs (`l9_ops.py`)

| Dim | Producer op | Encodes | Position | Reliability |
|-----|-------------|---------|----------|-------------|
| `OUTPUT_HI+k` (legacy ALU) | `layer9_alu` (phase=9) | Hi nibble of ADD/SUB/LEA/CMP result | AX byte positions | solid for legacy lookup; `_suppress_l9_legacy_addsub_writes` zeros ADD/SUB outputs in efficient mode |
| `CARRY+1`, `CARRY+2` | `layer9_alu` (phase=9) | Inter-byte carry for ADD/SUB | AX byte positions | solid in lookup mode; zeroed in efficient mode |
| `OUTPUT_LO`, `OUTPUT_HI` (bitwise + CMP) | `layer9_alu` (phase=9) | AND/OR/XOR byte 0, EQ/NE/LT/GT/LE/GE byte 0, marker suppression | AX byte / marker positions | solid |
| `CMP+0..7` | `layer9_alu` (phase=9) | Comparison group output | AX marker | solid |
| `ADDR_B0_LO/HI` (LEV relay) | `layer9_lev_addr_relay` head 0 (phase=9.0) | BP byte 0 -> ADDR_B0 at SP marker for LEV | SP marker | solid for LEV path |
| `ADDR_B0_LO/HI` (LEV BP->PC relay) | `layer9_lev_bp_to_pc_relay` head 1 (phase=9.1) | BP byte 0 -> ADDR_B0 at PC marker for LEV return | PC marker | solid for LEV path |
| `OUTPUT_LO/HI` (ALiBi mem) | `layer9_alibi_mem_attn` head 2 (phase=9.2, disabled by default) | ALiBi-recency mem load from PSH'd OUTPUT positions | MEM val byte 0 position | gated `enable=False` |

**Notable absences**:
- L9 also does not produce an SP-byte-0 marker-row flag.
- L9 does not produce a `STEP_BOUNDARY_VALID` lifecycle flag that distinguishes "the L8 SP
  gather completed for this step's MARK_SP row" vs. "the residual from a prior step's
  STACK0 gather is leaking through".

---

## Section 2: Missing signals L10 needs

These are the structural one-hot dims that, if produced by L7/L8/L9, would let the L10
tail-correction family shrink to ~5 declarative dispatchers (vs. today's ~18 strength-
escalating heuristic patches). Each signal entry lists:
- **Name** — proposed dim name.
- **Condition** — when the dim should be 1.0.
- **Producer** — recommended layer + op that should set it.
- **Consumer (L10)** — which existing tail rule(s) would simplify.
- **Why upstream** — why this cannot be computed at L10 with bounded strength.

### 2.1 `SP_BYTE0_IS_F8` (B5-J's recommendation, structurally)

- **Name candidate**: `SP_BYTE0_IS_F8` (1 dim, broadcast to SP marker row).
- **Condition**: at MARK_SP, set to 1.0 iff SP byte 0 (at SP byte-0 token position 11) =
  0xf8.
- **Producer**: L7 attention head — extend `layer7_memory_heads` head 6 (currently writes
  `PSH_AT_SP` from SP marker), add a sub-head that attends from MARK_SP to the SP byte-0
  position and uses a hard byte-equals-0xF8 K-side pattern (Q reads `CLEAN_EMBED_LO+8 *
  CLEAN_EMBED_HI+15` AND-gate at the byte-0 source, writes to a single SP marker output
  dim).
- **Consumer (L10)**: rule `tail_sp_marker_byte0_f8_from_initial_stack_exact` (line 3465)
  could replace its `OUTPUT_LO+8 / OUTPUT_HI+15` proxy evidence with a single
  `(SP_BYTE0_IS_F8, +1e6)` positive gate. The `HAS_SE` lifecycle hammer (B5-J's
  promotion to -1e9) becomes unnecessary because the upstream signal *is* the
  in-step fact.
- **Why upstream**: B5-J found the L10 OUTPUT-residue evidence cannot distinguish
  "SP byte IS 0xF8 (correct by carry-forward)" from "SP residue says 0xF8 (leak from a
  previous step that pushed F8)". The discrimination must be done at the moment SP byte 0
  is read from the embedding, before any layer's carry-forward residual has accumulated.

### 2.2 `SP_BYTE0_IS_<value>` family (extends 2.1 to ALL relevant values)

Generalize 2.1 to a small one-hot family for each SP byte 0 value the tail rules need.
Today's L10 family discriminates between 0x00 / 0xE0 / 0xE8 / 0xF0 / 0xF8 for the
MEM-store address byte family. A symmetric set of "is this byte equal to X" gates at
MARK_SP, MARK_BP, and MARK_MEM would let L10 compress 18 rules into 5.

- **Name candidates**: `SP_BYTE0_LO_NIBBLE_<k>` (16 dims) + `SP_BYTE0_HI_NIBBLE_<k>` (16
  dims) at the SP marker row, equivalent to the existing `ADDR_B0_LO/HI` 32-dim one-hot
  but **emitted at MARK_SP instead of MARK_STACK0** and **with an in-step `_VALID` gate**.
- **Producer**: extend `layer8_sp_gather_bake` (phase=8.0) to ALSO write to a new band at
  the MARK_SP query position (not just MARK_STACK0). This is the same SP byte-0 attention
  the L8 op already implements; only the Q-side gate needs to fire at MARK_SP as well as
  MARK_STACK0.
- **Consumer (L10)**: replaces `("OUTPUT_LO+8", 1.0), ("OUTPUT_HI+15", 1.0)` proxy
  patterns in rules A, B, D, I, J with direct one-hot reads.

### 2.3 `ADDR_B0_VALID` lifecycle flag (B4-H's recommendation)

- **Name candidate**: `ADDR_B0_VALID` (1 dim, broadcast to MARK_MEM row).
- **Condition**: at MARK_MEM, set to 1.0 iff the L8/L13 address gather for *this* step has
  written ADDR_B0 (i.e. the staged value is fresh, not residual).
- **Producer**: L13. Extend `layer13_mem_addr_gather` (phase=13) to write a single bit
  alongside `ADDR_B0_LO/HI`: when the gather attention fires at MARK_MEM, also write
  `ADDR_B0_VALID+0 = 1.0` via a sentinel V/O pair. This is one extra dim per the L13
  gather head, no architectural change required.
- **Consumer (L10)**: replaces the 18 strength-escalating discriminators in the
  `tail_mem_store_addr0_*` family. The replacement template (per B4-H 3.2):

  ```
  tail_mem_store_addr0_from_ADDR_B0_<value>:
    conditions:
      MARK_MEM           +1.0
      ADDR_B0_VALID      +50.0   # lifecycle gate
      MEM_STORE          +5.0
      ADDR_B0_LO+<lo>    +50.0
      ADDR_B0_HI+<hi>    +50.0
      ADDR_B0_LO+other   -200.0  (winner-take-all)
      ADDR_B0_HI+other   -200.0
    writes: byte_writes(value, strength=10_000)
  ```

  Strength bounded at 10k because the ADDR_B0 one-hot IS the address — no need to
  outvote prior siblings.
- **Why upstream**: B3-gamma's MARK_SP/OUTPUT inversion experiment confirmed that L10 cannot
  retroactively determine whether the ADDR_B0 staging is valid for the current step. The
  L13 op KNOWS when its gather fires; making that fact a structural bit is trivial.

### 2.4 `STEP_BOUNDARY_VALID` / `IN_STEP_FRESH` lifecycle bit

- **Name candidate**: `IN_STEP_FRESH` (1 dim, broadcast across the current step's 35-token
  band).
- **Condition**: set to 1.0 from the FIRST token of the current step through the SE
  marker; transitions from 0->1 at PC marker, 1->0 at SE.
- **Producer**: L1 attention extension. L1 already produces `HAS_SE` ("any prior STEP_END
  exists") via head 3 (`l1_ops.py:107-115`). Add a sibling head that attends from each
  token to the *next* STEP_END (forward ALiBi) and writes a "before next SE" flag. The
  two combined give "this step has begun AND hasn't ended" -> `IN_STEP_FRESH`.
- **Consumer (L10)**: replaces ad-hoc `HAS_SE` lifecycle hammers with a positive in-step
  evidence gate. Rule A's B5-J fix becomes a 1-term positive `IN_STEP_FRESH * +1e6` (vs.
  the current `HAS_SE * -1e9` negative hammer that incidentally regressed -5 cases on
  programs where SP byte 0 legitimately IS 0xF8 across multiple steps).
- **Why upstream**: B5-J's regressions (-5/269 tests) came from `HAS_SE` being a
  one-directional life-counter (1 if any step has ended), not an in-step freshness flag.
  The correct lifecycle bit must be the conjunction "step has begun" AND "step has not
  ended", which only L1 attention can compute cheaply.

### 2.5 `MARK_SP_FINALIZED` / `SP_GATHERED_THIS_STEP` (composite of 2.3 + 2.4)

- **Name candidate**: `SP_GATHERED_THIS_STEP` (1 dim, broadcast to MARK_SP row).
- **Condition**: 1.0 at the MARK_SP row when the L8 SP gather has written to ADDR_B0_HI
  / ADDR_B1_HI / ADDR_B2_HI for this step.
- **Producer**: L8. Extend `layer8_sp_gather_bake` (phase=8.0) to write a single sentinel
  bit at the MARK_SP row when its gather attention fires (analogous to 2.3 for L13).
- **Consumer (L10)**: gates all `tail_sp_marker_*` rules so they only fire after the
  upstream SP gather is known complete. Combined with 2.2's SP byte-0 one-hot, this
  eliminates the need for `OUTPUT_LO+8 / OUTPUT_HI+15` evidence on SP marker rows
  entirely.

---

## Section 3: L7-L9 correctness bugs (if any)

### 3.1 L8 SP gather only fires at MARK_STACK0 (correctness gap, not bug)

`layer8_sp_gather_bake` (`l8_ops.py:443-546`) heads 0-2 attend from MARK_STACK0 -> SP byte
positions. Per the bake spec (heads 0-2 Q gate at line 520: `AP(0, BD.MARK_STACK0, L)`),
the gather DOES NOT fire at MARK_SP. The L10 tail rules at MARK_SP rows that consume
`ADDR_B0_LO+lo / ADDR_B0_HI+hi` therefore depend on the residual leak from MARK_STACK0,
not on a fresh gather at MARK_SP.

**Recommendation**: extend the L8 SP gather to fire at both MARK_STACK0 AND MARK_SP, so
the L10 rules have a fresh in-step source at both marker rows. This is a structural
change but small — replicate the existing Q gate with one extra MARK_SP entry.

### 3.2 L9 `_suppress_l9_legacy_addsub_writes` is mode-coupled (technical debt, not bug)

`_suppress_l9_legacy_addsub_writes` (`l9_ops.py:86-109`) zeros L9's legacy ADD/SUB hi-nibble
output units and CARRY+1/+2 in efficient mode. This is correct (the efficient
AddSub5StageBlock takes over), but it leaves a partial L9 ALU FFN: in efficient mode,
L9 still owns CMP/AND/OR/XOR but ADD/SUB are silent. Any future structural-signal
consumer that expects "L9 always produces ALU result" will be surprised by the mode
asymmetry.

**Recommendation**: not a correctness bug; document explicitly in `consumes_fresh` /
`produces` metadata of `layer9_alu` that ADD/SUB output is only produced in lookup mode.

### 3.3 L7 head 5 K-scale 2.0 vs. legacy row-multiply (latent fragility)

`_layer7_memory_head_specs` head 5 (`l7_ops.py:311-349`) uses `k=(AP(0, BD.MARK_AX, L *
2.0),)` to preserve a softmax-sharpness fix that previously ran as a post-helper row
multiply (per docstring). This is a working fix but it means head 5's V slot reads of
TEMP+3..9 are *softmax-sharper* than other L7 heads. Any future op that adds a new V
slot to head 5 will get the same sharper K — surprising but documented.

**Recommendation**: not a bug; keep the comment but consider migrating to an explicit
per-V-slot K-scale spec parameter so the implicit dependency is visible.

### 3.4 No correctness bugs found in L7/L8/L9 carry-forward semantics themselves

The fundamental claim that "L10 tail rules are downstream patches papering over weak
upstream structural signals" is correct, but the cause is **missing signals**, not
**broken signals**. L7/L8/L9 correctly produce everything they claim to produce; the
gap is in what they do NOT produce.

---

## Section 4: Concrete implementation tasks

Each task is sized to be one follow-up unit, with explicit deliverables, files touched,
test expectations, and proposed branch name. Tasks 1-3 are independent and can land in
parallel; tasks 4-5 depend on 1-3.

### Task B7-1: Add `IN_STEP_FRESH` lifecycle dim at L1

**Goal**: produce a 1-dim signal that is 1.0 from "current step PC marker" through
"current step SE marker" exclusive, replacing ad-hoc `HAS_SE` lifecycle hammers in
L10 rules.

**Files**:
- `c4_release/neural_vm/vm_step.py:2215-2470` — allocate `IN_STEP_FRESH = <next free dim>`
  in `_SetDim` class.
- `c4_release/neural_vm/unified_compiler/ops/l1_ops.py` — add a 9th L1 attention head
  (sibling to head 3 HAS_SE) that uses forward ALiBi (negative slope) to attend to
  the NEXT MARK_SE_ONLY, writing 1.0 to `IN_STEP_FRESH` until that next SE fires.
- Update `make_layer1_attn_op`'s `writes={...}` and `_claims`.

**Tests**:
- `c4_release/tests/test_l1_attn.py` — add unit test verifying `IN_STEP_FRESH = 1.0` at
  positions 0..34 (step 0), 35..69 (step 1), and so on; transitions to 0 at SE positions
  34, 69, ...
- Update layer claims registry tests if any catch the new V/O slots.

**Acceptance**: sentinel 3/32 preserved; new unit test passes.

**Estimated complexity**: medium (forward ALiBi requires reversing slope sign — verify
behavior with the existing softmax1 anchor).

### Task B7-2: Promote L8 SP gather to also fire at MARK_SP (`SP_GATHERED_THIS_STEP`)

**Goal**: extend `layer8_sp_gather_bake` (`l8_ops.py:443`) so heads 0-2 fire at MARK_SP
as well as MARK_STACK0, ensuring `ADDR_B0_LO/HI..ADDR_B2_LO/HI` carry a fresh in-step
SP-derived address at MARK_SP rows.

**Files**:
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:502-546` — modify
  `_layer8_sp_gather_head_specs` to add `AP(0, BD.MARK_SP, L)` to each head's Q gate.
  Verify the K gate (BYTE_INDEX_n + H1[SP_I]) still selects SP byte n correctly when
  the Q fires at MARK_SP (it should, since the K side is unchanged).
- Optionally: write a 1-dim sentinel `SP_GATHERED_THIS_STEP` flag at MARK_SP from a
  new V/O pair on one of the three heads (or a 4th head).

**Tests**:
- `c4_release/tests/test_l8_sp_gather.py` (new) — verify `ADDR_B0_LO/HI` content at
  MARK_SP rows matches SP byte 0 across a 5-step trace.
- Run sentinel ids 0-32 to ensure 3/32 baseline preserved.

**Acceptance**: 3/32 sentinel preserved; new unit test shows ADDR_B0 is fresh at MARK_SP
not residual.

**Estimated complexity**: small (4-line Q-spec extension).

### Task B7-3: Add `ADDR_B0_VALID` lifecycle bit at L13

**Goal**: extend `layer13_mem_addr_gather` (`l13_ops.py:7-51`) to write a 1-bit
`ADDR_B0_VALID` flag at the MARK_MEM row when its gather attention fires, so L10 can
gate the `tail_mem_store_addr0_*` family on actual freshness rather than residual.

**Files**:
- `c4_release/neural_vm/vm_step.py:2215-2470` — allocate `ADDR_B0_VALID = <next free dim>`.
- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:7-51` (or the helper
  `_set_layer13_mem_addr_gather` in `vm_step.py`) — add a 4th head (or extend an
  existing one) that fires at MARK_MEM with the same Q gate as the existing gather
  heads and writes 1.0 to `ADDR_B0_VALID` via a V/O sentinel.

**Tests**:
- `c4_release/tests/test_l13_addr_gather.py` — verify `ADDR_B0_VALID = 1.0` at MARK_MEM
  rows for SI/SC/LI/LC steps, and 0.0 (or near 0) at MARK_MEM rows on steps where the
  gather should not fire (PSH, ADJ, etc.).
- Sentinel 3/32 preserved.

**Estimated complexity**: small to medium (one extra V/O pair plus dim allocation).

### Task B7-4: Add `SP_BYTE0_VALUE` one-hot at MARK_SP (depends on B7-2)

**Goal**: extend the L8 SP gather (or add a sibling L7/L8 head) to emit a 32-dim one-hot
(`SP_BYTE0_LO_NIBBLE_<k>` + `SP_BYTE0_HI_NIBBLE_<k>`, k=0..15) at MARK_SP, encoding
the actual SP byte 0 value. This generalizes B5-J's recommended `SP_BYTE0_IS_F8` to
the full byte space.

**Files**:
- `c4_release/neural_vm/vm_step.py:2215-2470` — allocate 32 dims for the one-hot pair.
  (May need to evict TEMP space or use existing CLEAN_EMBED aliasing.)
- L7 or L8 op extension that gathers SP byte 0 -> these dims at MARK_SP.

**Tests**:
- New unit test verifying `SP_BYTE0_LO_NIBBLE_8 = 1.0 AND SP_BYTE0_HI_NIBBLE_15 = 1.0` at
  MARK_SP rows iff SP byte 0 = 0xF8.
- Convert the failing `tail_sp_marker_byte0_f8_from_initial_stack_exact` rule (B3-gamma
  rejected) to use this signal as its dominant positive evidence.

**Acceptance**: sentinel 3/32 preserved; B5-J's -5 regression on ids 0-249 reverts (rule
A no longer needs the HAS_SE -1e9 hammer once SP_BYTE0_HI_NIBBLE_15 is the positive
evidence).

**Estimated complexity**: medium (dim allocation in a near-saturated 512-dim layout is the
hard part; the attention extension itself is small).

### Task B7-5: Refactor L10 `tail_mem_store_addr0_*` family to use ADDR_B0_VALID (depends on B7-3)

**Goal**: implement B4-H's plan 3 using the new `ADDR_B0_VALID` gate. Replace 18 strength-
escalating rules with 5 declarative dispatchers (one per byte value 0x00, 0xE0, 0xE8,
0xF0, 0xF8).

**Files**:
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3450-4100` — implement
  `addr_from_l13_rules` helper from B4-H 4, replace existing rules C/E/F/G/H/K/L/M/N/
  O/P/Q/R, retain rule R as legacy compatibility shim.
- Per B4-H 3.5, also migrate byte-1/2/3 cousins (rules D, I, J) to use `ADDR_B1_*` /
  `ADDR_B2_*` with the same template.

**Tests**:
- Run sentinel ids 0-32 (target: 3/32 preserved).
- Run B5-J's `ids 200-249` regression set (target: 25/50 vs. baseline 25/50).
- Run B3-eta's ENT-main regression case (target: no spurious 0xE0 fire on ENT steps).

**Acceptance**: sentinel preserved; per-rule unit tests in `test_l10_tail_correction.py`
all pass; rule strengths bounded at <= 10,000.

**Estimated complexity**: medium (template helper is straightforward; the risk is in
re-validating the regression sets identified by B2-A / B2-B / B3-alpha / B3-eta).

### Task B7-6: Refactor L10 `tail_sp_marker_*` family to use SP_BYTE0_* one-hot (depends on B7-4)

**Goal**: replace rule A's `OUTPUT_LO+8 / OUTPUT_HI+15` proxy evidence with direct
`SP_BYTE0_HI_NIBBLE_15 / SP_BYTE0_LO_NIBBLE_8` one-hot reads. Similar treatment for
rule B's byte-1 sibling.

**Files**:
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3465-3535` — rewrite rule A and
  B to use the new one-hot evidence at threshold ~50 and bounded active_value ~10_000.
  Drop the `HAS_SE -1e9` hammer (no longer needed once the upstream signal is direct).

**Tests**:
- Revert B5-J's regression (-5 on ids 0-249) and confirm sentinel 3/32 still preserved.
- Existing per-rule unit tests for rule A (`test_tail_sp_marker_byte0_f8_from_initial_
  stack_exacts_nibbles` and `test_tail_sp_marker_byte0_f8_blocks_ax_imm_ff_residue`)
  rewritten against the new evidence.

**Acceptance**: B5-J's -5 regression reverts; sentinel preserved; new unit tests pass.

**Estimated complexity**: small once Task B7-4 lands.

### Task B7-7 (optional): Document `produces` / `consumes_fresh` metadata for new dims

**Goal**: extend the staleness-invariant scanner (Phase 3 / Agent G of
`ARCH_LEAKAGE_FIX_PLAN.md`) to track the new signals. Add `produces` annotations to
B7-1/2/3/4 ops and `consumes_fresh` annotations to refactored L10 rules.

**Files**: all ops touched by B7-1..6, plus the scanner registry.

**Acceptance**: scanner reports no staleness violations on the new signals.

**Estimated complexity**: small (metadata-only).

---

## Cross-references

- **B4-H plan**: `.agent-logs/l10-tail-family-refactor/PLAN.md` (commit e672177 on
  `proposal/l10-tail-correction-family`) — the L10-side companion to this audit. Tasks
  B7-5 and B7-6 are its concrete sub-tasks once the upstream signals (B7-1..4) land.
- **B3-gamma rejection**: commit 5ef58b0 on `investigation/b2b-mark-sp-inversion-rejected`
  + inline comment at `l10_ops.py:3464-3502`. Documents why L10-only fixes to rule A
  cannot work (contradictory evidence under output-dominant formulation).
- **B5-J investigation**: commit d0065e8 on `investigation/l10-sp-marker-lifecycle-gate`.
  Demonstrates that promoting `HAS_SE -100 -> -1e9` regresses -5 cases; the structural fix
  must be an in-step positive signal, not a negative life-counter.
- **ARCH_LEAKAGE_FIX_PLAN.md** Phase 3 / Agent G — the staleness invariant framework that
  catches mismatches between `produces` and `consumes_fresh` annotations. Tasks B7-1..4
  should each declare `produces` for their new signals.

---

## Summary table: 7 follow-up units identified

| Unit | Depends on | Risk | Sentinel-preserving? |
|------|-----------|------|----------------------|
| B7-1 IN_STEP_FRESH at L1 | none | low | yes (additive dim) |
| B7-2 L8 SP gather at MARK_SP | none | low | yes (Q gate extension) |
| B7-3 ADDR_B0_VALID at L13 | none | low | yes (one V/O pair) |
| B7-4 SP_BYTE0_VALUE one-hot | B7-2 | medium | yes if dim allocation succeeds |
| B7-5 L10 tail_mem_store_addr0_* refactor | B7-3 | medium | requires regression set validation |
| B7-6 L10 tail_sp_marker_* refactor | B7-4 | low once B7-4 lands | reverts B5-J's -5 regression |
| B7-7 Staleness metadata | B7-1..4 | low | metadata only |

**Net effect once all 7 land**: L10 tail-correction family shrinks from ~18 strength-
escalating rules (strengths up to 5e9) to ~5 bounded-strength dispatchers (strength
<= 10_000) reading upstream structural signals directly. Eliminates the regressions
identified by B2-A, B2-B, B3-alpha, B3-gamma, B3-eta, and B5-J without sentinel cost.
