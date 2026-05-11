# STACK0 via Memory Attention — Migration Plan

**Status**: Plan only. No code changes in this slice.
**Owner**: stack0-via-mem-attention agent.
**Related**: `docs/INJECT_MEM_STORE_MIGRATION_PLAN.md`,
`docs/archive/PSH_STACK0_ROOT_CAUSE_ANALYSIS.md`,
`alibi-mem-attention` branch (ALiBi memory-attention upgrade in flight).

## 1. Framing

The user's framing: *"Make that stack0 change to just get it into the
activations via the normal memory attention."*

Concretely: the **5 STACK0 token positions** (positions 20–24 of every
35-token step) are a staging buffer between AX/SP and MEM. Each
downstream consumer that reads STACK0 today can — in principle — read
the same byte value from the prior step's MEM section (or the current
step's MEM section, depending on the consumer) via attention keyed by
`SP` (the live address) plus `MEM_STORE` (the store anchor). That is
exactly the pattern `_set_layer15_memory_lookup` already uses for `*SP`
loads at the STACK0 marker for the POP group (`vm_step.py:5979-6142`,
head 0 K = `STACK0_BYTE0` today, but its address path is identical to
the `mem[SP]` lookup we want).

So this is **not** "remove STACK0 token slot + all downstream wiring"
(the previous agent's framing, ~540 refs, multi-day). It is
"redirect each downstream STACK0 consumer to read MEM via attention,
keep the activations they produce identical". The 5-token slot can be
removed at the very end once all reads route through MEM; the
write-side STACK0 mirroring (PSH→STACK0, JSR→STACK0, ENT→STACK0) is
already duplicated to MEM in `run_vm.py:700-723` and `:786-815`, so it
becomes dead the moment the reads are converted.

## 2. STACK0 consumers (and their replacements)

The five jobs STACK0 does today, in the order they fire in a step:

### Consumer 1: **L7 ALU operand 2** (binary ops read prev `*SP`)

- **Today**: L6 attn heads 6/7 copy `AX_CARRY_LO/HI` (from the AX
  marker, distance 15) into `ALU_LO/HI` at the STACK0 marker
  (`vm_step.py:4347-4382`). L7 head 0 then reads STACK0 byte 0's
  `CLEAN_EMBED_LO/HI` via the `STACK0_BYTE0` key and writes
  `ALU_LO/HI` at the AX marker (`setup_helpers.py:642-677`).
  Downstream L7-L10 ALU consumes `OUTPUT` (operand 1 = AX) +
  `ALU_LO/HI` (operand 2 = `*SP`).
- **Why STACK0 exists for this**: PSH writes byte 0 of `*SP` into
  STACK0 byte 0 (CLEAN_EMBED). That gives L7 a token whose
  `CLEAN_EMBED_LO/HI` is the operand byte and whose `STACK0_BYTE0`
  flag is the anchor.
- **Replacement (Phase 1)**: a new L7 attention head that reads
  `mem[SP]` byte 0 via the same address-encoding path L15 head 0
  already uses for POP. Specifically:
  - **Q at AX marker**: gated by binary-op activator (the same set
    today routed via the `CMP[3]` POP-group flag; binary-op opcodes
    activate this), plus `ADDR_B*` carrying `SP` as the address.
    `SP` is already gathered to `ADDR_B0/B1/B2` at the STACK0 marker
    by `_set_layer8_sp_gather` — for L7 we instead need `SP` at the
    AX marker. Two options:
    1. **(Preferred)** Add an earlier head (L6 or new L7 head) that
       relays `SP`'s `OUTPUT_LO/HI` to `ADDR_B0` at the AX marker.
       The L6 opcode relay head (`vm_step.py:6836-6937`) already
       relays AX → SP/STACK0; we add a symmetric SP → AX relay.
    2. **(Alternative)** Promote `_set_layer8_sp_gather` to L6/L7 so
       SP is already in `ADDR_B*` at the AX marker by L7. Note this
       conflicts with the existing L7 head 2-4 use of `ADDR_B0..2`
       for LI/LC address gather — would need a different lane.
  - **K side**: same as L15 head 0 — `MEM_STORE * 100` anchor,
    `ADDR_KEY` 24-bit address match, `L2H0[MEM] - H1[MEM]` byte-0
    gate. Anti-leakage gate at non-AX positions (`MARK_AX`
    requirement, `OP_LEA`/`OP_LI`/`OP_LC` suppression).
  - **V/O**: copy `CLEAN_EMBED_LO/HI` → `ALU_LO/HI` (exactly what
    L7 head 0 writes today, exactly what L7-L10 ALU expects).
- **Net effect on downstream**: L7-L10 ALU code is untouched.
  `ALU_LO/HI` at AX marker has the same value, just sourced from
  `mem[SP]` via MEM attention instead of via STACK0 byte 0.
- **What can be deleted once this lands**:
  - L7 head 0 (`setup_helpers.py:642-677`) — replaced by the new
    head.
  - L6 attn heads 6 LO + 7 HI (`vm_step.py:4347-4382`) — the
    AX → STACK0 staging for PSH's ALU lane is now unused because
    nothing reads STACK0 ALU. Note these are also used to set up
    PSH's `OUTPUT = AX` at STACK0 (consumer 2 below), so keep
    until consumer 2 is converted.
  - L4 FFN ALU clear at AX marker (`vm_step.py:4309-4327`) — exists
    only to defeat L4's PC+2 residual before L7 head 0 writes. New
    head must either still need this defeat, or be amplified enough
    to overwrite. Likely keep for now.
- **First-step special case**: step 0 has no prior MEM section. L7
  head 0 today reads `STACK0_BYTE0` from STACK0 byte 0 of the
  *current* step; on step 0 the STACK0 carry attn writes EMBED at
  the STACK0 marker (zero by default). The new MEM-attention head
  will get 0 from softmax1 fall-through if no MEM section matches —
  same effective value, no regression.

### Consumer 2: **PSH `STACK0 ← AX` staging** (`vm_step.py:3855-3876`)

- **Today**: L6 FFN, at the STACK0 marker, gated by `PSH_AT_SP`,
  routes `ALU_LO/HI` (= AX value, copied by consumer-1's L6 heads
  6/7) into `OUTPUT_LO/HI`. That output becomes the STACK0 byte 0
  token via the standard byte-0 emission path.
- **Why STACK0 exists for this**: to materialize a STACK0 token
  whose byte 0 equals AX, so consumer 1 (in the *next* step) can
  read it as operand 2.
- **Replacement (Phase 2)**: once Consumer 1 is converted, the
  PSH-STACK0 byte tokens are no longer read by anything. The MEM
  section already gets the correct value (PSH's L14 val heads 4-7
  write the AX-source bytes into the MEM section's val bytes; see
  `vm_step.py:5796-5898`). PSH's `STACK0 ← AX` writes become a
  no-op as long as we keep emitting tokens at positions 20-24
  (i.e., before the 35→30 token format change). When we drop the
  slot, this code is deleted entirely.
- **What can be deleted**:
  - The PSH STACK0 FFN block `vm_step.py:3855-3876` (32 units).
  - The duplicate `_override_register_in_last_step(..., STACK0,
    self._last_ax)` in `run_vm.py:789`.

### Consumer 3: **JSR `STACK0 ← (PC+8)` return address**
(`vm_step.py:4286-4302, 7290-7430`)

- **Today**: L6 attn head 7 copies the PC value from the PC marker
  to `AX_CARRY_LO/HI` at the STACK0 marker, then L14 val heads 4-7
  use the "STACK0 source bonus" (Q[2] = `OP_JSR`) to attend to
  STACK0 byte positions, gathering the 4 bytes of the return
  address into the MEM section. JSR also writes a STACK0 byte-0
  marker output (`vm_step.py:7357-7416`).
- **Replacement (Phase 2/3)**: the MEM section already contains the
  return address (L14 val heads + the runner-side shim in
  `run_vm.py:797-802`). For any downstream consumer that wants the
  return address (the LEV PC reload path: `_set_layer9_lev_*` and
  L15 heads 8-11 in 17-layer mode), route through MEM attention
  keyed on the saved-BP + offset. The 16-layer baseline already
  reads the return address from MEM at LEV time
  (`_extract_register` + handler shim). Once nothing reads STACK0
  on the JSR step, the L14 STACK0 source bonus (dim 2) collapses
  to "AX source only", which is the SI/SC/PSH path.
- **What can be deleted**:
  - L6 head 7 PC→STACK0 relay (`vm_step.py:7290-7322`).
  - L14 val-head STACK0 source dim 2 (`vm_step.py:5834-5859`).
  - JSR STACK0 byte FFN block (`vm_step.py:7357-7416`).
  - Runner override `run_vm.py:802`.

### Consumer 4: **ENT `STACK0 ← old_BP`** (`compiler.py:779-783`,
`vm_step.py:7246-7259, 7533-7546`)

- **Today**: L6 head 5 copies BP's `EMBED` → `TEMP` at the STACK0
  marker. L6 FFN, gated by `CMP[2]` (ENT) + `MARK_STACK0`, cancels
  the identity carry and writes `TEMP` → `OUTPUT`. L14 val heads
  use the same STACK0 source bonus as JSR to gather the bytes into
  the MEM section.
- **Replacement (Phase 2/3)**: identical to JSR. MEM section already
  has `old_BP` from the L14 val heads + runner shim
  (`run_vm.py:813-814`). Once Consumer 1 is converted and nothing
  downstream reads the STACK0 byte tokens, the ENT-specific FFN
  block becomes dead.
- **What can be deleted**: L6 head 5
  (`vm_step.py:7246-7259`), ENT STACK0 FFN block
  (`vm_step.py:7533-7546`), runner override `run_vm.py` ENT branch.

### Consumer 5: **Distance-from-BP gates** (L1/L3/L6/L7/L14)

Not a value-read consumer — these gates exist *because* STACK0
occupies a fixed range of distances from BP. Listed here so the
reader knows the full set of weight sites that change when the
slot is finally removed.

- **Today**: many FFNs and attention heads use `H1[BP]`/`H2[BP]`/
  `H3[BP]`/`H4[BP]`/`L1H4[BP]` thresholds to gate behavior at "BP
  area" (d≤4) vs "STACK0 area" (d=5..9) vs "outside BP/STACK0".
  Distinctive examples:
  - `vm_step.py:2086-2208` — nibble-copy suppression at STACK0
    area (`H4[BP] = d≤9.5`).
  - `vm_step.py:2576-2611` — STACK0 default bytes 1-3 = 0.
  - `vm_step.py:3711-3715` — SP/BP/STACK0 identity carry.
  - `vm_step.py:4286-4307` — MEM_STORE leakage cancellation at
    SP/STACK0/BP markers.
  - `vm_step.py:4404-4519` — L7 head 6 PSH/ENT/JSR flag broadcast
    that has both SP and STACK0 ranges in Q.
  - `vm_step.py:4522-4560` — L8 SP gather Q fires at STACK0 area.
  - `vm_step.py:5648+` — L14 attention addr/val heads that use
    `H4[BP]` to gate STACK0 area.
- **Replacement (Phase 4, only after token slot removal)**: once
  positions 20-24 are gone, BP→MEM distance shifts from 10 to 5
  tokens. **Every threshold that today reads "STACK0 area" or
  "≤9.5 from BP"** must be retargeted to:
  - "MEM area" (`H3[MEM]` etc.) where it was gating on the *token
    after* STACK0 (only L8 SP-gather, L14 val heads),
  - **deleted** where it was gating on STACK0 itself (nibble-copy
    suppression, default zero writes, identity carry, MEM_STORE
    leakage cancellation, L7 head 6 broadcast Q gate),
  - **shrunk** (`H4[BP] → H1[BP]` style) where it was using STACK0
    as a stand-in for "BP+5" but really just needed "near BP".
- **Scope estimate**: ~30-40 distinct gate sites. Each is small (one
  or two `W_q`/`W_up` weights) but they need to be enumerated and
  tested as a unit — they all interact.

## 3. Order of operations

### Phase 1 — L7 ALU operand 2 (this PR / next PR)
1. Add SP-to-AX-marker relay (a new head in L6 attn or a new lane
   in `_set_layer7_operand_gather`).
2. Add the new L7 mem-attention head: Q = `MARK_AX + binary-op
   activator + ADDR_B*`, K = `MEM_STORE + ADDR_KEY`, V/O = same as
   today (`CLEAN_EMBED → ALU_LO/HI`).
3. Disable old L7 head 0 (zero out its weights) and old L6 heads
   6/7 STACK0 staging.
4. Run smoke `test_add_basic` + `test_imm_exit`. Both must pass.
   `test_add_basic` is the canonical proof — ADD reads operand 2
   from `*SP` and that path is now MEM-attention sourced.
5. Hold off on touching consumers 2-5 until Phase 1 is green.

### Phase 2 — PSH/JSR/ENT STACK0 byte tokens become content-free
1. Bake the PSH/JSR/ENT FFN blocks that write into STACK0 byte
   tokens to no-ops (set their `W_down` to zero). The slot still
   exists in the token format; it just emits zero bytes.
2. Drop the runner-side `_override_register_in_last_step(...,
   STACK0, ...)` calls in `run_vm.py:789, 802, 815`.
3. Re-run smoke + `test_jsr*`, `test_ent*` (xpasses today). Should
   stay xpassing/xfailing as-is; no regression.

### Phase 3 — Drop the slot from the token format
1. Update `token_layout.py` (positions 20-24 collapse, MEM marker
   moves from 25→20, END from 34→29).
2. Update `constants.py:TOKENS_PER_STEP = 30`,
   `TOKENS_PER_REGISTER * 4` (PC/AX/SP/BP only) in the assertion.
3. Update `speculative.py:DraftVM.draft_tokens` — remove the
   `Token.STACK0` + 4-byte block (`speculative.py:390-408`).
4. Update `run_vm.py` `_build_context` (`:1370+`) to skip STACK0
   emission.
5. Mass-update distance-from-BP gates (Consumer 5). This is the
   single highest-risk step. Enumerated list lives at the end of
   this doc.

### Phase 4 — Cleanup
1. Drop `Token.STACK0` enum.
2. Drop all `MARK_STACK0`, `STACK0_BYTE0`, `NEXT_STACK0` dim
   allocations and the FFN units that write them.
3. Drop the now-orphaned L6/L7 broadcast lanes for the PSH/ENT/JSR
   flags into STACK0 byte positions (L7 head 6).
4. Drop `_set_stack0_carry_attn` (`setup_helpers.py:214-252`).

## 4. Compatibility with ALiBi memory-attention agent

The `alibi-mem-attention` branch is upgrading the memory-attention
path (today's L15 + various memory heads gain proper ALiBi slopes
plus L8 SP-gather lifted into attention from `_inject_mem_store`).
Phase 1 of this STACK0 plan reuses that same attention machinery:
- The new L7 head's address-matching is identical to L15 head 0's
  binary-bit address encoding + `MEM_STORE` anchor + ZFOD pattern.
- If ALiBi slope tuning lands first, the new head should set its
  slope to the same value L15 head 0 uses for "attend to current
  step's prior MEM" (probably `≈ 0.5` for short-range, or `0` for
  position-independent).
- We do **not** invent new infrastructure here — we re-use the L15
  recipe one layer earlier and with a different Q target.

If the ALiBi branch lands before Phase 1, this plan's L7 head is
just a copy of the new ALiBi L15 head 0 with Q gated to AX instead
of STACK0. If it lands after, both branches must update the same
L7 head's slope/scale together.

## 5. Test impact

### Tests that should NOT regress at Phase 1
- `tests/test_smoke.py::TestSmokeBasic::test_imm_exit` — touches no
  stack; pure passthrough. Pre-flight check that Phase 1 didn't
  break first-step generation.
- `tests/test_smoke.py::TestSmokeBasic::test_add_basic` — the
  canonical exerciser. ADD = pop operand 2 from `*SP`. Must pass.
- `tests/test_runtime_vanilla.py` — vanilla mode (non-pure-neural)
  uses runner-side handlers; STACK0 changes shouldn't touch this
  path because the handler reads `_memory` directly. Sanity gate
  against accidentally breaking the embedding/forward shape.
- `tests/test_layer_idx_consistency.py` — ensures L7's `layer_idx`
  pin remains correct after the new head is added.
- `tests/test_compile_determinism.py` — same compile output
  byte-for-byte across runs. Should pass trivially.

### Tests that MAY xpass on Phase 1
- The "PSH+ADD pure_neural" xfail family. Today these xfail because
  AX_CARRY → STACK0 → ALU is fragile (see
  `PSH_STACK0_ROOT_CAUSE_ANALYSIS.md`). Reading `mem[SP]` directly
  bypasses the AX_CARRY corruption chain. If they start xpassing,
  flip them to `expected_pass=True`.

### Tests that MUST be touched at Phase 2/3
- `tests/test_neural_only.py` — STACK0 reads on the output side.
- DraftVM speculation tests (`tests/test_speculation*`) — depend on
  the 35-token format.
- All tests that grep for `Token.STACK0` or positional offsets >= 20
  in the step buffer. List with
  `grep -rn 'STACK0\|position.*2[0-4]\|tokens\[2[0-4]\]' tests/`.

## 6. Enumerated distance-from-BP gates (Phase 3 work)

| File:line | Today's gate | Reason | Phase-3 action |
|-----------|--------------|--------|----------------|
| `vm_step.py:2086-2208` | `H4[BP] = -S` in nibble-copy suppression LO/HI | exclude STACK0 area | delete (no STACK0 to exclude) |
| `vm_step.py:2576-2611` | STACK0 default byte 0 = 0 unit | first-step empty stack | delete |
| `vm_step.py:3711-3715` | identity carry list includes `MARK_STACK0` | passthrough at marker | delete entry |
| `vm_step.py:4159, 4170` | `MARK_STACK0 = -S*10` suppression | block leaks at STACK0 | delete |
| `vm_step.py:4286-4307` | MEM_STORE/ADDR_SRC leakage cancel at `MARK_STACK0` | block stray writes | delete |
| `vm_step.py:4309-4327` | ALU clear at AX marker | defeat L4 PC+2 residual before L7 head 0 | keep (new L7 head still benefits) |
| `vm_step.py:4347-4382` | L6 heads 6/7 STACK0 staging | feed L7 head 0 | delete after Phase 1 |
| `vm_step.py:4404-4519` | L7 head 6 Q includes `MARK_STACK0 + H4[BP]` | broadcast flags to STACK0 bytes | delete those Q clauses; keep SP broadcast |
| `vm_step.py:4522-4560` | L8 SP gather Q fires at STACK0 area | put SP into ADDR at STACK0 | delete (already moved to AX marker in Phase 1) |
| `vm_step.py:5707, 5735` | L14 attn `STACK0_BYTE0` K paths | gather STACK0 byte 0 source | delete |
| `vm_step.py:5814-5817, 5866-5872` | L14 val-head MARK_STACK0/H4[BP] suppression | cancel STACK0-position garbage | delete |
| `vm_step.py:5816-5860` | L14 STACK0 source bonus dim 2 (JSR/ENT) | gather stack0 bytes into MEM | delete (val heads always use AX source) |
| `vm_step.py:5952, 5965` | L14 `MARK_STACK0 = -S*10` | block PC-output writes at STACK0 marker | delete |
| `vm_step.py:6027-6041` | L15 head 0 STACK0/POP-group activator | POP reads `*SP` at STACK0 marker | retarget Q to `MARK_AX + CMP[3]` (POP) |
| `vm_step.py:6038-6070` | L15 heads 1-3 `L1H4[BP]` STACK0-area Q | POP byte 1-3 reads | retarget to `BYTE_INDEX_{1..3} + MARK_AX` |
| `vm_step.py:6093-6133` | L15 byte-q-flag list includes MARK_STACK0 | byte 0 source for POP | replace MARK_STACK0 with the equivalent AX-side flag |
| `vm_step.py:6689-6764` | NEXT_STACK0 suppression at PSH-related units | propagate STACK0 carry | delete |
| `vm_step.py:6831-6937` | L6 opcode relay head `MARK_STACK0 + L1H4[BP]` Q | relay opcode to STACK0 area | drop STACK0 lanes; keep SP lane |
| `vm_step.py:7025-7060` | L7 format-pointer extraction reads `MARK_STACK0` | conversational-I/O path | retarget to `MARK_SP` or directly to `mem[SP]` via attention |
| `vm_step.py:7246-7322` | L6 head 5 (BP→STACK0 TEMP) + L6 head 7 (PC→STACK0 AX_CARRY) | ENT/JSR staging | delete after Phase 2 |
| `vm_step.py:7357-7546` | L6 FFN JSR/ENT STACK0 byte/marker writes | materialize STACK0 token | delete after Phase 2 |
| `setup_helpers.py:214-252` | `_set_stack0_carry_attn` (STACK0 marker EMBED carry) | fill STACK0 marker with byte 0 carry | delete after Phase 2 |
| `setup_helpers.py:642-677` | `_set_layer7_operand_gather` head 0 | the prime consumer | **rewritten in Phase 1** |
| `compiler.py:646-647, 779-783, 839-853` | per-op handler bindings | bake the above ops | re-bind after Phase 1 |

Total: 22 sites to touch, ~150-200 lines net delete. Conservative
estimate: 3-4 worker-sessions for Phase 1 alone (operand gather +
SP-to-AX relay + test-fix sweep), then 1-2 per subsequent phase.

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| New L7 head's softmax1 picks up wrong MEM section (e.g., a historical store at the same SP from before a POP-PSH cycle) | The L15 head-0 K-side already handles this with `MEM_STORE` anchor + per-byte gate; the most-recent store wins via ALiBi recency. Re-use that exact recipe. |
| First-step ADD has no prior MEM section | First-step ADD is undefined C anyway (read of uninitialized stack); softmax1 fall-through to 0 matches today's STACK0-empty behavior. |
| SP-to-AX relay introduces a new layer dependency or breaks an existing ADDR_B0 lane | Add as new dimension lane (TEMP/ADDR_B0 swap) or as a fresh `STACK_ADDR_Bn` dim band. Document at compile time via the dim_registry. |
| Distance-from-BP gates over-fire when STACK0 area collapses (Phase 3) | One-at-a-time conversion with the e2e gate after each. |
| ALiBi branch and this branch land out of order | Phase 1's new L7 head is a copy of the L15 recipe — both branches end up rewriting that lane once. Coordinate via the dim_registry / op-graph. |

## 8. Out of scope for this plan

- Per-byte stride/word-size MEM layouts (`STACK_ALIGNMENT` stays at
  8 throughout).
- Removing the duplicate runner-side `_mem_store_word` writes (they
  remain authoritative for the host-side stdout/stdin shim).
- Changing PC_OFFSET, INSTR_WIDTH, or the 5-byte instruction layout.

## 9. Acceptance criteria for Phase 1 PR (when it lands)

1. `test_imm_exit` and `test_add_basic` pass under the smoke recipe
   in this branch's worker prompt.
2. `test_runtime_vanilla.py`, `test_layer_idx_consistency.py`,
   `test_compile_determinism.py` pass.
3. The new L7 head is registered as a `migrated=True` op in
   `unified_compiler/ops/l7_ops.py` with explicit
   `reads={"MARK_AX", "MEM_STORE", "ADDR_KEY", "ADDR_B0_LO",
   "ADDR_B0_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"}` and
   `writes={"ALU_LO", "ALU_HI"}`.
4. Old `_set_layer7_operand_gather` head 0 weights are explicitly
   zeroed (not just unused) — verified by a small unit test that
   asserts `attn.W_q[0:HD].abs().sum() == 0` for L7 attn head 0.
5. `purity_guard.py` does not regress.
