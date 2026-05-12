# V18 Conversational-I/O THINK-Protocol Neural Plan

Goal: Replace the Python ``run_vm._handle_thinking_end`` block (the
``if self.conversational_io and next_token == Token.THINKING_END:``
branch at ``c4_release/neural_vm/run_vm.py:534-583``) with a baked
neural attention/FFN pipeline that emits the
``THINKING_END → output bytes → THINKING_START → next step`` cycle
autoregressively, no Python intervention required, per the canonical
spec at ``BLOG_SPEC.md:851``:

> Output works via the think-tag protocol: all VM computation
> happens inside THINK_START/THINK_END tags (hidden from the user),
> and when a printf executes, the model exits the think block,
> emits a character byte token (visible to the user), then
> re-enters the think block and continues execution.

Status: scoping. The convo-IO neural pipeline is **already largely
baked** (L2/L3/L5/L6/L7/L8/L9/L10/L15 — see §2). What is **missing**
is (a) end-to-end exercise of the existing bakes with
``enable_conversational_io=True`` against the V18 handler-mode
fallback as a reference, and (b) the step-resumption bake that
re-emits a full 35-token VM step after ``LAST_WAS_THINKING_START``
(the neural equivalent of ``_inject_synthetic_step``). Phase 0
(this PR) ships the design doc, a parity test that exercises the
existing convo-IO weights, and identifies the precise residual
work for Phase 1.

## Coordination with V9 PUTCHAR

The parallel V9 PUTCHAR work
(branch ``v9-neural-io-think-protocol-putchar``) targets a different
opcode trigger:

- **V9 PUTCHAR** fires at the AX marker of a PUTCHAR step, routes
  ``AX_CARRY_LO/HI → OUTPUT_LO/HI`` (units 1500-1532 of L6 FFN, see
  ``PHASE_6_PUTCHAR_BAKE_SPEC.md``). It does **not** use the
  THINK_END/THINK_START tag protocol — PUTCHAR's byte is emitted
  inline in the existing 35-token step at the AX-value byte slot.
- **V18 PRTF/READ** fires at NEXT_SE of a PRTF/READ step, triggers
  ``NEXT_THINKING_END`` (suppressing STEP_END), then in the
  following autoregressive cycle walks the format string via
  L7/L8/L9, emits bytes via L15 OUTPUT_BYTE→OUTPUT routing, and
  finally fires ``NEXT_THINKING_START`` (via L10 null-terminator
  detection) to resume execution.

The two paths share **no attention heads** (PUTCHAR is L6 FFN units
1500-1532 only; V18 uses L6 attn heads 4-5, L6 FFN units 1400-1401,
L7 head 7, L8 FFN units 600-615, L9 head 0, L10 FFN unit 1864,
L15 FFN units 1200-1232). They share the BD layout (``OUTPUT_LO/HI``
at 174-205), which is read-only here.

This plan does NOT modify any L6 FFN unit in the range 1500-1532
(V9 PUTCHAR routing) and does NOT touch the lm_head's
``OUTPUT_LO/HI`` decode rows.

Source files referenced:

- ``c4_release/docs/BLOG_SPEC.md`` (line 851, canonical for neural I/O).
- ``c4_release/docs/PURITY_AUDIT_2026_05_11.md`` §V18.
- ``c4_release/docs/PHASE_6_PUTCHAR_BAKE_SPEC.md`` (V9 PUTCHAR scope).
- ``c4_release/docs/CONVERSATIONAL_IO_FINAL_STATUS.md`` (prior status).
- ``c4_release/neural_vm/run_vm.py:534-583`` (V18 handler block).
- ``c4_release/neural_vm/setup_helpers.py:1677-1886`` (existing bakes).
- ``c4_release/neural_vm/vm_step.py:7120-7292`` (additional bakes).
- ``c4_release/neural_vm/unified_compiler/ops/l2_ops.py`` (L2 lookback).
- ``c4_release/neural_vm/unified_compiler/ops/l3_ops.py`` (L3 state init).
- ``c4_release/neural_vm/unified_compiler/ops/l7_ops.py``
  (``make_format_pointer_extraction_op``).
- ``c4_release/neural_vm/unified_compiler/ops/l8_ops.py``
  (``make_format_position_counter_op``).
- ``c4_release/neural_vm/unified_compiler/ops/l9_ops.py``
  (``make_format_string_fetch_head_op``).
- ``c4_release/neural_vm/unified_compiler/ops/flag_gated_ops.py``
  (the six ``make_convo_io_*`` and ``make_null_terminator_detection_op`` and
  ``make_conversational_io_output_routing_op``).

---

## 1. What ``run_vm.py:534-583`` does today

When ``self.conversational_io == True`` and the model autoregressively
emits ``Token.THINKING_END``, the runner takes over for several
operations that should be the model's job per BLOG_SPEC.md:851:

```python
if self.conversational_io and next_token == Token.THINKING_END:
    exec_pc = self._exec_pc()
    fmt_ptr = self._extract_register(context, Token.STACK0)
    if not fmt_ptr:
        fmt_ptr = self._mem_load_word(self._last_sp)
    if fmt_ptr:
        fmt_str = []
        addr = fmt_ptr
        while len(fmt_str) < 256:
            byte_val = self._memory.get(addr, 0)
            if byte_val == 0:
                break
            fmt_str.append(byte_val)
            addr += 1
        for byte_val in fmt_str:
            context.append(byte_val)
            output.append(chr(byte_val))
        context.append(Token.THINKING_START)
        new_pc = exec_pc + INSTR_WIDTH
        new_sp = (self._last_sp + 8) & 0xFFFFFFFF
        stack0_val = self._mem_load_word(new_sp) if new_sp else 0
        self._inject_synthetic_step(
            context, new_pc, self._last_ax, new_sp, self._last_bp, stack0_val
        )
        self._last_pc = new_pc
        self._last_sp = new_sp
```

The Python is doing five separate things, each of which the model
SHOULD do autoregressively per BLOG_SPEC.md:851:

1. **Extract format pointer** from the just-emitted REG_STACK0 marker
   in the current step's tail.
2. **Walk the format string** in memory until it hits a null
   terminator.
3. **Emit each byte** as a token (the "character byte token
   visible to the user").
4. **Emit THINKING_START** to mark "re-enter think block".
5. **Re-emit a full 35-token VM step** with PC advanced past the
   PRTF, SP popped (the printf argument removed), and the rest of
   the register state preserved.

The handler also handles `fmt_ptr == 0` (single-byte case where the
model puts the byte directly into a hidden buffer — used by the
no-format printf), but in practice STACK0 always holds the pointer.

---

## 2. What's already baked neurally

The convo-IO neural pipeline is largely in place, gated by
``enable_conversational_io=True`` (default ``False``). Each piece
has a registered compiler op:

| Layer | Op | Function (in setup_helpers/vm_step) | Purpose |
|---|---|---|---|
| Embed | (embedding) | ``Token.THINKING_START/END`` (272/273), ``IO_STATE_EMIT_*`` (274/275) | Token IDs reserved. |
| L2 attn | ``layer2_lookback_detection_head`` (l2_ops.py:52) | ``_set_lookback_detection_head`` | Detect previous token: ``LAST_WAS_THINKING_END``, ``LAST_WAS_THINKING_START``, ``LAST_WAS_BYTE``. |
| L3 FFN | ``layer3_convo_io_state_init`` (l3_ops.py:143) | ``_set_conversational_io_state_init`` | Set ``IO_IN_OUTPUT_MODE=1`` when ``LAST_WAS_THINKING_END``. |
| L5 FFN | ``convo_io_opcode_decode`` (flag_gated_ops.py:67) | ``_set_conversational_io_opcode_decode`` | Detect PRTF/READ opcodes → ``IO_IS_PRTF``/``IO_IS_READ`` at AX marker. |
| L6 attn | ``convo_io_relay_heads`` (flag_gated_ops.py:155) | ``_set_conversational_io_relay_heads`` | Relay ``IO_IS_PRTF/READ`` from AX → SE position via heads 4-5, writing ``CMP[5]`` / ``CMP[6]``. |
| L6 FFN | ``convo_io_state_machine`` (flag_gated_ops.py:244) | ``_set_conversational_io_state_machine`` | Units 1400-1401: ``CMP[5]`` (PRTF) ∧ ``NEXT_SE`` → emit ``NEXT_THINKING_END``, suppress ``NEXT_SE``, set ``IO_STATE=1``. |
| L7 attn | ``format_pointer_extraction`` (l7_ops.py:66) | ``_set_format_pointer_extraction`` | Head 7: at ``IO_IN_OUTPUT_MODE`` position, attend back to STACK0 marker and copy ``EMBED_LO/HI`` (pointer byte 0) → ``FORMAT_PTR_LO/HI``. |
| L8 FFN | ``format_position_counter`` (l8_ops.py:37) | ``_set_format_position_counter`` | Units 600-615: increment ``IO_FORMAT_POS`` by 1 on every ``LAST_WAS_BYTE ∧ IO_IN_OUTPUT_MODE``. |
| L9 attn | ``format_string_fetch_head`` (l9_ops.py:126) | ``_set_format_string_fetch_head`` | Head 0: query ``FORMAT_PTR + FORMAT_POS`` against ``ADDR_KEY``, copy ``EMBED_LO/HI`` of matched byte → ``OUTPUT_BYTE_LO/HI``. |
| L10 FFN | ``null_terminator_detection`` (flag_gated_ops.py:309) | ``_set_null_terminator_detection`` | Unit 1864: ``OUTPUT_BYTE==0 ∧ IO_IN_OUTPUT_MODE`` → set ``IO_OUTPUT_COMPLETE``, clear ``IO_IN_OUTPUT_MODE``, emit ``NEXT_THINKING_START``. |
| L15 FFN | ``conversational_io_output_routing`` (flag_gated_ops.py:359) | ``_set_conversational_io_output_routing`` | Units 1200-1231: when ``IO_IN_OUTPUT_MODE`` set, copy ``OUTPUT_BYTE_LO/HI → OUTPUT_LO/HI`` so lm_head decodes the fetched byte. |
| lm_head | ``setup_head_weights`` (shared.py:447-449) | n/a | Maps ``NEXT_THINKING_START → Token.THINKING_START``, ``NEXT_THINKING_END → Token.THINKING_END``. |

State flow (with ``enable_conversational_io=True``):

```
PRTF executed (step N), AX marker:        ACTIVE_OPCODE_PRTF=1
  → L5 FFN:                                IO_IS_PRTF=5
NEXT_SE position (last token of step N):
  → L6 attn head 4 relays IO_IS_PRTF →     CMP[5]=1
  → L6 FFN unit 1400 (CMP[5] ∧ NEXT_SE):   NEXT_THINKING_END=1, NEXT_SE=0
  → lm_head emits Token.THINKING_END instead of STEP_END

Next token after THINKING_END:
  → L2 head 1 detects t-1 was THINKING_END: LAST_WAS_THINKING_END=1
  → L3 FFN unit 1034:                       IO_IN_OUTPUT_MODE=1
  → L7 head 7: attend STACK0, copy ptr:     FORMAT_PTR_LO/HI = ptr byte 0
  → L9 head 0: fetch byte at PTR+POS:       OUTPUT_BYTE_LO/HI = format[0]
  → L15 FFN units 1200+: in output mode:    OUTPUT_LO/HI = OUTPUT_BYTE
  → lm_head decodes OUTPUT_LO/HI → byte token (first format char)

Each subsequent token:
  → L2 head 1 detects LAST_WAS_BYTE
  → L8 FFN units 600+: increment FORMAT_POS
  → L7 head 7 + L9 head 0: fetch byte at PTR + new POS
  → L15 FFN: emit that byte

When OUTPUT_BYTE == 0 (null terminator):
  → L10 FFN unit 1864: IO_OUTPUT_COMPLETE=1, IO_IN_OUTPUT_MODE=0,
                       NEXT_THINKING_START=1
  → lm_head emits Token.THINKING_START

Next token after THINKING_START:
  → L2 head 1 detects LAST_WAS_THINKING_START
  → ???  No baked handler for this transition.
```

**This is the gap.** The model has wiring up to "emit
THINKING_START at end-of-output". The Python handler then injects
a synthetic step with new PC/SP/etc. The model has nothing yet that
makes the post-THINKING_START token be ``Token.REG_PC`` with the
PC advanced past the PRTF instruction.

---

## 3. The missing bake: step resumption after THINKING_START

In handler mode, after THINKING_START the runner appends a fresh
35-token step into the context (``_inject_synthetic_step``,
``run_vm.py:1574``). The state-machine docstring in
``_set_conversational_io_state_machine`` (setup_helpers.py:1769-1777)
explicitly enumerates the missing transitions:

```
1. Normal execution (IO_STATE=0)
2. PRTF/READ detected → set NEXT_THINKING_END, IO_STATE=1   [DONE]
3. After THINKING_END → generate output, IO_STATE=2          [partial]
4. After output complete → set NEXT_THINKING_START, IO_STATE=3 [DONE via L10]
5. After THINKING_START → resume normal, IO_STATE=0          [MISSING]
```

Two concrete missing pieces:

### 3a. "Resume normal" after THINKING_START

A small FFN unit at, say, L3 or L11 that:

- Detects ``LAST_WAS_THINKING_START`` (from L2 head 1).
- Clears ``IO_STATE`` to 0.
- Forces the next lm_head decode to be ``Token.REG_PC`` (i.e.,
  sets a transient ``NEXT_REG_PC`` flag).

The lm_head already has the structural ``NEXT_REG_PC →
Token.REG_PC`` routing for the normal start-of-step transition
(at STEP_END's t+1). So the bake here is just "treat
``LAST_WAS_THINKING_START`` like ``LAST_WAS_STEP_END`` for purposes
of starting a new step": route the lookback flag into the same
``NEXT_REG_PC`` set-bit that the normal STEP_END → REG_PC path
uses.

Sketch:

```python
# L3 FFN extension (or a new layer slot):
unit = (next-free-after-1035)
ffn.W_up[unit, BD.LAST_WAS_THINKING_START] = S
ffn.b_up[unit] = -S * 0.5
ffn.b_gate[unit] = 1.0
ffn.W_down[BD.NEXT_REG_PC, unit] = 2.0 / S     # start a new step
ffn.W_down[BD.IO_STATE, unit] = -2.0 / S       # clear IO_STATE
```

### 3b. PC/SP advancement across the THINKING_END→OUTPUT→THINKING_START interlude

When the model resumes after THINKING_START, the new step's PC must
be ``exec_pc + INSTR_WIDTH`` (skip past the PRTF instruction) and
the new SP must be ``old_sp + 8`` (pop the format-string pointer
argument).

In normal execution this happens at the AX marker via L5 PC
increment + L4/L8 SP adjustment ops. After PRTF, the same ops
should fire — but the relevant cues (``ACTIVE_OPCODE_PRTF``,
``MARK_AX``) are already several tokens in the past by the time we
emit THINKING_START.

The handler mode masks this by recomputing PC/SP from
``self._last_pc / self._last_sp`` and writing them into the
synthetic step's value bytes. The neural mode needs to:

- Cache ``exec_pc + 4`` and ``last_sp + 8`` into residual
  dimensions during the PRTF step (NEXT_PC_AFTER_PRTF, etc.), so
  they survive the output-emission interlude.
- Replay them at the new REG_PC / REG_SP value-byte positions of
  the resumed step.

This is the largest missing chunk: roughly an L5 / L6 FFN pair of
"latch PC+SP into post-PRTF cache dims at the PRTF AX marker", plus
an L9 attention head that, at the resumed REG_PC value-byte
positions, copies the cached values out.

The same structure is needed for AX preservation (the value of the
``print`` call's return) and BP preservation (unchanged across the
PRTF, but the resumed step must still emit it correctly).

### 3c. Why not just delete the handler today

Without 3a + 3b, flipping ``enable_conversational_io=True`` and
removing the V18 handler would:

- Emit THINKING_END correctly.
- Emit some output bytes (existing L7/L8/L9/L15 path).
- Emit THINKING_START correctly.
- Then enter an uncontrolled autoregressive regime — no
  guarantees about which token comes after THINKING_START, no
  guarantees about PC/SP advancement.

The end-to-end test (``test_conversational_io.py:test_prtf_sequence``)
would likely produce gibberish after THINKING_START.

The handler at V18 stays as the production path until 3a + 3b land
and the end-to-end test passes neurally. This matches the V9 PUTCHAR
story (``PHASE_6_PUTCHAR_BAKE_SPEC.md``: "PUTCHAR-specific work is
NOT a single weight-bake op... PUTCHAR cannot be fixed in isolation
without first fixing the upstream IMM AX-writeback and step-boundary
emission").

---

## 4. Two design options for the missing bake

### Option A — Phase 0 staging (this PR)

**Don't add new bakes.** Instead, ship:

1. This design doc.
2. A parity / inventory test
   (``tests/test_v18_convo_io_neural_parity.py``) that locks in
   the surface contract Phase 1 must reproduce:
   - The V18 handler block is present at its known location in
     ``run_vm.py`` (guard-string check).
   - All 10 convo-IO compiler ops (table in §2) are registered in
     ``all_core_ops()``.
   - ``Token.THINKING_START/END`` and the BD dim names used by the
     bakes are stable.
   - ``_inject_synthetic_step`` keeps its signature contract.

This option **does not change production behavior**: the V18 handler
stays the active path (``conversational_io=False`` by default in
the runner, and the only callers that pass ``True`` are the
``test_conversational_io_*`` tests). It does NOT delete the handler.
It does NOT touch attention heads (V9 PUTCHAR territory or
otherwise). It documents the missing bake (3a + 3b) for a future PR.

**Pros:**
- Zero risk to gate suites (no production weight changes).
- Provides a baseline diff target so the Phase 1 bake (3a + 3b)
  has a clear pass/fail signal.
- Coordinates cleanly with V9 PUTCHAR (different opcode trigger,
  no shared FFN units).

**Cons:**
- Doesn't actually emit the bake yet. Phase 1 still has to do 3a
  + 3b. Pure documentation + test infrastructure delivery.

### Option B — Phase 1 bake (future PR)

Add the two missing bake ops (3a "resume normal" FFN + 3b "latch
PC/SP across interlude" attention/FFN pair), gated by
``enable_conversational_io_step_resume=False`` initially. Flip
once parity test passes.

Out of scope for this PR.

### Recommendation

Phase 0 (this PR): Option A. Ship the design doc and the parity
test. Don't touch production weights.

Phase 1 (separate): Option B. Add the two missing bakes, gated
off; flip when parity test passes end-to-end.

Phase 2 (separate): delete the V18 handler at run_vm.py:534-583
once the neural path is end-to-end stable.

---

## 5. Verification plan

### Phase 0 — gated parity test (this PR)

1. Add ``tests/test_v18_convo_io_neural_parity.py`` with the
   following coverage:
   - Static check: V18 handler guard string present in
     ``run_vm.py``.
   - Inventory check: all 10 convo-IO ops registered in
     ``all_core_ops()`` (so Phase 1 has stable hooks).
   - Token-ID check: ``Token.THINKING_START/END`` reserved.
   - BD-dim check: residual-stream dim names used by the bakes
     are present in ``_SetDim``.
   - Synthetic-step signature check: ``_inject_synthetic_step``
     keeps its ``(context, pc, ax, sp, bp, stack0=0)`` contract;
     the V18 handler calls it after appending ``THINKING_START``.
2. Constraint gates: smoke + runtime_vanilla + layer_idx_consistency
   + compile_determinism still pass (no production weight
   changes).
3. The handler-mode convo-IO test
   (``tests/test_conversational_io.py``) keeps passing — V18 stays
   active.

### Phase 1 — flip the gate (future PR)

1. Add ``make_convo_io_step_resume_op(enable=False)`` to
   ``flag_gated_ops.py`` (3a: ``LAST_WAS_THINKING_START`` →
   ``NEXT_REG_PC``, clear ``IO_STATE``).
2. Add ``make_convo_io_pc_sp_latch_op(enable=False)`` (3b: stash
   ``exec_pc + 4`` and ``last_sp + 8`` into post-PRTF cache dims at
   the PRTF AX marker; replay at the resumed REG_PC / REG_SP
   value-byte positions).
3. With ``enable=True`` on both, re-run the parity test from
   Phase 0; expected to pass.
4. Once parity, gate the V18 handler to also check ``pure_neural``
   and prefer the neural path.

### Phase 2 — delete the V18 handler

1. Delete the ``if self.conversational_io and next_token ==
   Token.THINKING_END:`` block at ``run_vm.py:534-583``.
2. Delete the related shadow-memory walks
   (``self._mem_load_word(self._last_sp)``,
   ``self._memory.get(addr, 0)`` for format-string read) IF no
   other callers remain — but the same dict is used by
   ``_neural_prtf_emit`` etc., so it stays.
3. Run all test gates.

---

## 6. Risks

| Concern | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| Breaks compile determinism | None | None | None |
| Breaks gate suites | None | Low (gated off) | Low (handler removal only after parity) |
| Conflicts with V9 PUTCHAR attention heads | None — different layers | None — different layers | None — V9 is on L6 FFN units 1500+ (PUTCHAR routing); V18 is on L6 FFN units 1400-1401 (state machine) and L6 attn heads 4-5 | None |
| Format-string fetch fails when format addr is at a memory address not yet in ``_mem_history`` | Documented but unchanged | Likely fail mode for Phase 1 | Resolved when ``_mem_history`` becomes neural per V10 |
| PC/SP latch (3b) drifts across the output-emission interlude (token positions ~ +N where N = format string length) | n/a | Score-budget concern; ALiBi distance vs. signal strength | n/a |

The biggest unknown for Phase 1 is 3b: latching the post-PRTF
PC/SP through what could be tens of token positions of output
emission. The cleanest design likely uses a dedicated residual
band (e.g., ``POST_PRTF_PC`` 32 dims + ``POST_PRTF_SP`` 32 dims)
that's set at the PRTF AX marker and replayed at the resumed step.
Carries through KV-cache without per-position recomputation.

---

## 7. What this PR ships

- This design doc.
- No new weight bake ops. The existing ``enable_conversational_io``
  bake set (10 ops, table in §2) is unchanged.
- A parity / inventory test
  (``tests/test_v18_convo_io_neural_parity.py``) that locks in
  the surface contract (handler location, registered op names,
  reserved token IDs, BD dim names, synthetic-step signature)
  that Phase 1 must reproduce. Does not run the model.
- No changes to ``run_vm.py``. The V18 handler at lines 534-583
  stays as the production path.
- No changes to V9 PUTCHAR attention heads or L6 FFN units 1500+.

Future work (separate PRs):

- Phase 1: bake ``make_convo_io_step_resume_op`` and
  ``make_convo_io_pc_sp_latch_op`` (gated, ``enable=False``);
  flip after parity test passes neurally.
- Phase 2: delete the V18 handler block once the neural path is
  end-to-end stable.

## 8. Phase 1c addendum: positional attention transport bake (2026-05-11)

Phase 1b shipped the capture-side bake
(``_set_convo_io_prtf_capture`` at L7 FFN units 800-863). That bake
writes ``POST_PRTF_PC_LO/HI`` and ``POST_PRTF_SP_LO/HI`` cache dims at
the PRTF AX marker, where the 3b replay band (``_set_convo_io_pc_sp_latch``
at L6 FFN units 1402-1465) reads them at the post-THINKING_START position.

What was missing: **the residual stream does not propagate dim values
across positions without an attention head**. Captured PC/SP nibbles
sat attached to the PRTF AX marker's position; the L6 FFN replay band
at the post-THINKING_START position saw zero. Phase 1c closes this gap.

The transport bake (``_set_convo_io_prtf_transport`` at L4 attn head 4,
factory ``make_convo_io_prtf_transport_op`` in ``flag_gated_ops.py``):

- Q (post-THINKING_START position, gated by ``LAST_WAS_THINKING_START``).
- K (matches PRTF AX marker via ``ACTIVE_OPCODE_PRTF`` AND ``MARK_AX``).
- V/O copy the four POST_PRTF_PC/SP nibble groups across HD=64 head slots.
- ALiBi slope = 0.1 (shallow) so the head can reach back ~80 tokens
  across the variable-length output-byte interlude.

The chain (capture 3c → transport 3d → replay 3b) is byte-aligned at
the cache-dim level: the V18 plan §3b's POST_PRTF cache dims now have
a complete writer-reader pair across the variable-length interlude.

Status: Phase 1c shipped (this PR). All three bakes (capture, transport,
replay) registered in ``all_core_ops()`` with double-gating
(``enable_conversational_io`` AND per-bake ``enable=False``). Phase 2
(deleting the V18 handler) is gated on flipping the three ``enable``
switches together and re-running the convo-IO end-to-end test.

## 9. Phase 2 entry, attempt 1 (2026-05-12): enables flipped, handler retained

Branch ``v18-phase2-delete-handler`` flipped all four V18 inner ``enable``
switches to ``True`` in ``all_core_ops.py`` (step_resume, pc_sp_latch,
prtf_capture, prtf_transport). The 38 bake unit tests
(``tests/test_v18_convo_io_neural_bakes.py``) and the 7 parity tests
(``tests/test_v18_convo_io_neural_parity.py``) still pass. The full
constraint gate (smoke + runtime_vanilla + layer_idx_consistency +
compile_determinism) is green.

What blocks Phase 2 handler deletion: an end-to-end smoke loop running
``model.forward`` autoregressively against a ``printf("Hi");`` program
with ``enable_conversational_io=True`` did NOT emit ``Token.THINKING_END``
within 300 generated tokens. The model emitted normal step tokens (REG_PC,
REG_AX, REG_SP, REG_BP, STACK0, MEM, STEP_END) but never triggered the
L6 FFN state-machine bake (units 1400-1401) that should fire
``NEXT_THINKING_END`` at the PRTF step's NEXT_SE position.

Per the Phase 2 spec the handler at ``run_vm.py:776-825`` is retained;
deletion is gated on the smoke loop producing the full
``THINKING_END → bytes → THINKING_START → REG_PC`` chain autoregressively.
The bake gate flips themselves are safe (and bake unit tests + parity
tests verify the unit-layout invariants).

Likely root cause to investigate next: either (a) the convo-IO opcode
decode L5 path needs ``IO_IS_PRTF`` to be set during the PRTF step's AX
marker — verify that runs with the new gate flip; or (b) the L6 relay
head's CMP[5] write doesn't propagate to the L6 FFN state-machine units
1400-1401 in the production residual layout; or (c) the embedding-level
opcode decode for PRTF isn't asserting ``ACTIVE_OPCODE_PRTF`` early
enough in the step. The bake unit tests cover the FFN weight pattern
but not the upstream CMP/IO_IS_PRTF data flow.
