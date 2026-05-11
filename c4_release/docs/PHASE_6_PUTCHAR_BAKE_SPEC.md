# Phase 6 PUTCHAR Pure-Neural Bake Spec

Spec for the neural-side weight bake needed to make
`tests/test_pure_neural_io.py::TestPureNeuralPutchar::test_putchar_one_char`
pass in `pure_neural=True` mode.

## TL;DR

The test currently xfails. The runner-side hook to emit a byte for PUTCHAR
ALREADY exists in `run_vm.py:_dispatch_step` (lines 639-640):

```python
if exec_op == Opcode.PUTCHAR and neural_ax is not None:
    output.append(chr(neural_ax & 0xFF))
```

and the neural-side AX→OUTPUT routing weight ALREADY exists in
`vm_step._set_io_putchar_routing` (vm_step.py:8715-8754), installed by the
compiler op `make_io_putchar_routing_op` (migrated_ops.py:2772).

The PUTCHAR test does NOT fail because `make_io_putchar_routing_op` is
missing or wrong. It fails because **the model's autoregressive generation
degenerates after step 0**: the AX value bytes the runner needs to read
back are never produced.

Diagnostic trace (Opcode.IMM 'A' → PUTCHAR → EXIT):

```
[STEP 0 tokens] ... 257 65 0 0 0 258 66 64 64 64 64 64 64 64 64 64 ...
                       ^REG_PC PC bytes  ^REG_AX  ^AX bytes
[STEP 1 tokens] 64 64 64 64 64 64 ... (degenerated to token 64)
[STEP 2 tokens] 64 64 64 64 64 64 ...
```

- Step 0 emits AX bytes `[66, 64, 64, 64]` (= 0x40404042). IMM 'A' should
  yield AX = 0x41. **The IMM AX-write path is off by one (writes 0x42 LSB,
  pads remaining bytes with 0x40).** This is an upstream IMM weight bug.
- After step 0, the model emits the constant byte token 64 forever, never
  reaching another STEP_END or REG_AX. `_extract_register(context, REG_AX)`
  returns `None` at the PUTCHAR step, so the runner skips the `output.append`.

## Token sequence the network needs at the PUTCHAR step

For the program `IMM 'A' / PUTCHAR / EXIT` to emit `"A"`, the model must
produce, at step 2 (PUTCHAR), a 35-token block of the form:

```
REG_PC  + 4 PC-value bytes (LE: 0x08, 0x00, 0x00, 0x00 — instr 1 × 8)
REG_AX  + 4 AX-value bytes (LE: 0x41, 0x00, 0x00, 0x00)    ← runner reads byte 0
REG_SP  + 4 bytes
REG_BP  + 4 bytes
STACK0  + 4 bytes
MEM     + 4 addr + 4 value
STEP_END
```

The runner already knows `exec_op == PUTCHAR` (via bytecode walking + the
pure_neural exec_idx counter) and only needs **`REG_AX byte 0`** of the
just-completed step to equal `0x41`.

## Which BD dim drives the byte tokens?

For each of the four AX-value byte positions in a step, the lm_head decodes
the byte using the residual stream's `OUTPUT_LO` (16 nibbles, BD 174-189)
and `OUTPUT_HI` (16 nibbles, BD 190-205) dims at that position:

```
head.weight[byte_b, OUTPUT_LO + (b & 0xF)] = 5.0
head.weight[byte_b, OUTPUT_HI + (b >> 4)]  = 5.0
head.bias[byte_b] = -5.0
```

(`migrated_ops.setup_head_weights`, lines 2725-2734).

So at each AX-byte position the model must have `OUTPUT_LO[lo] ≈ 1` and
`OUTPUT_HI[hi] ≈ 1` where `lo, hi = b & 0xF, b >> 4` of the AX byte to emit.

Note: `OUTPUT_LO/HI` is the right surface, NOT `NEXT_BYTE` (which is part
of the conversational-IO TOOL_CALL path baked by
`_set_tool_call_*`/`_set_emit_byte_*`; that path is for PRTF, not PUTCHAR).

## What's currently baked vs. what's missing

### Currently baked (correctly, for PUTCHAR itself)

`_set_io_putchar_routing(ffn6, S, BD)` at L6 FFN units 1500-1532:

- Unit 1500: AND(`OP_PUTCHAR`, `MARK_AX`) → `IO_IS_PUTCHAR` (informational).
- Units 1501-1516: AND(`OP_PUTCHAR`, `MARK_AX`) AND `AX_CARRY_LO[k]` →
  `OUTPUT_LO[k]` for k in 0..15.
- Units 1517-1532: AND(`OP_PUTCHAR`, `MARK_AX`) AND `AX_CARRY_HI[k]` →
  `OUTPUT_HI[k]` for k in 0..15.

This fires only at the `MARK_AX` token position of the PUTCHAR step, where
the lm_head is predicting the **REG_AX value bytes that follow**. It is
the same template EXIT/NOP routing uses for `AX_CARRY → OUTPUT_LO/HI`.

### Missing / upstream broken

1. **IMM AX writeback is off by one** (separate bug — out of PUTCHAR scope).
   `IMM 'A'` writes `OUTPUT_LO/HI` such that byte 0 of AX = 0x42 instead of
   0x41, and bytes 1-3 = 0x40 instead of 0x00.

2. **Autoregressive context degenerates after step 0.** Once the model
   starts emitting token 64 ('@' = 0x40) it never recovers — STEP_END is
   never produced, so `_dispatch_step` is only reached via the
   `synthetic_step_end` fallback in `run_vm.py:452`. By the PUTCHAR step,
   the context window contains no REG_AX marker and
   `_extract_register(context, REG_AX)` returns `None`.

The PUTCHAR bake itself is fine; **PUTCHAR cannot be fixed in isolation
without first fixing the upstream IMM AX-writeback and step-boundary
emission**. The runner already has the byte-append; the model just has to
deliver the right `REG_AX 0x41 0x00 0x00 0x00` sequence.

## Proposed compiler op signature (if a fresh bake were needed)

If the existing `_set_io_putchar_routing` were missing, the spec would be:

```python
def make_io_putchar_emit_op() -> Operation:
    """Bake L6 FFN PUTCHAR AX→OUTPUT routing: at MARK_AX of a PUTCHAR step,
    copy AX_CARRY_LO/HI nibble flags into OUTPUT_LO/HI so the lm_head
    decodes the byte 0 of AX as the next REG_AX value-byte tokens.

    Reads:  OP_PUTCHAR, MARK_AX, AX_CARRY_LO[0..15], AX_CARRY_HI[0..15]
    Writes: OUTPUT_LO[0..15], OUTPUT_HI[0..15], IO_IS_PUTCHAR
    Layer:  model.blocks[6].ffn  (L6 FFN)
    Units:  33 (1 flag + 16 lo + 16 hi)  starting at unit 1500
    Phase:  998 (before legacy_bake/right_size at 999/1200)
    Kind:   "model" so bake_fn can resolve ffn6 from the model handle.
    """
    def bake(model, dim_positions, S):
        from ..vm_step import _set_io_putchar_routing
        _set_io_putchar_routing(
            model.blocks[6].ffn, S, _as_setdim_proxy(dim_positions),
        )
    return Operation(
        name="io_putchar_routing",
        reads=set(), writes=set(),
        kind="model", bake_fn=bake,
        phase=998, migrated=True,
    )
```

This op already exists in `migrated_ops.py:2772-2799` and is registered in
`all_core_ops()` (migrated_ops.py:4560). **No new op needs to be added.**

## What the actual single-agent fix would look like

PUTCHAR-specific work is NOT a single weight-bake op. The fix list, in
order of dependency:

1. **Fix IMM AX-writeback** (separate spec; pure_neural produces wrong AX
   byte 0 even on a single-instruction program). Without this, PUTCHAR has
   no correct byte to emit anyway.

2. **Fix step-boundary emission** so the model autoregressively reaches
   STEP_END at position 34 in pure_neural mode without the
   `synthetic_step_end` fallback. (The fallback masks the underlying
   degeneration but lets dispatch run; with degeneration, `_extract_register`
   still returns None because the context tail is all 64s, not register
   markers.) This likely requires inspecting the NEXT_* flags at byte
   positions in the last AX-byte token to drive REG_SP / REG_BP / etc.

3. (Already done.) PUTCHAR-specific routing — keep as-is.

Once (1) and (2) land, the existing PUTCHAR routing should suffice and the
xfail can be removed. The runner side (`_dispatch_step` PUTCHAR branch) is
already in place.

## Estimated complexity

| Component | Status | Complexity |
|---|---|---|
| PUTCHAR L6 FFN AX→OUTPUT routing | DONE (units 1500-1532) | n/a |
| `make_io_putchar_routing_op` compiler op | DONE (phase 998) | n/a |
| `_neural_putchar_emit` runner hook | DONE (run_vm.py:639) | n/a |
| **IMM AX-writeback fix** | NOT DONE (off by one) | Medium — requires audit of L0-L5 IMM path |
| **Step-boundary emission stability** | NOT DONE (degenerates to 0x40) | High — affects all pure_neural ops, not just PUTCHAR |

**Single-agent PUTCHAR-only bake: not feasible** because PUTCHAR's bake is
already in place. The xfail is gated on two upstream issues (IMM correctness
and pure_neural step-boundary stability), neither of which is a single
PUTCHAR weight write.

**Recommendation:** keep the xfail marker on the PUTCHAR tests. Open a
separate spec for the IMM AX-writeback bug and the step-boundary degenera-
tion. PUTCHAR will likely xpass for free once those are fixed; if not,
revisit this spec — the routing op is correct in principle.

## Verification once upstream lands

After IMM+step-boundary fix:

1. Run a one-instruction probe (`IMM 'A' / EXIT`) under `pure_neural=True`
   and assert step 0's REG_AX byte 0 == 0x41. If yes, IMM is fixed.
2. Run `tests/test_pure_neural_io.py::TestPureNeuralPutchar::test_putchar_one_char`
   without xfail. With the existing `_set_io_putchar_routing`, the OUTPUT_LO/HI
   path at the PUTCHAR step's MARK_AX should activate, REG_AX value bytes
   should be 0x41 0x00 0x00 0x00, and the runner appends 'A' to output.
3. If still failing, attach `_dispatch_step` trace (see diag at the top of
   this doc) — the symptom should now be a wrong AX byte, not `neural_ax is None`.

## References

- Runner hook: `c4_release/neural_vm/run_vm.py:618-640`
- Existing bake: `c4_release/neural_vm/vm_step.py:8715-8754` (`_set_io_putchar_routing`)
- Compiler op: `c4_release/neural_vm/unified_compiler/migrated_ops.py:2772-2799`
- lm_head byte decode: `c4_release/neural_vm/unified_compiler/migrated_ops.py:2725-2734`
- BD dim layout: `c4_release/neural_vm/vm_step.py:1580-1581` (`OUTPUT_LO=174`, `OUTPUT_HI=190`),
  `c4_release/neural_vm/vm_step.py:1693-1694` (`AX_CARRY_LO=328`, `AX_CARRY_HI=344`)
- Test: `c4_release/tests/test_pure_neural_io.py:48-54`
- Phase 6 status: `c4_release/docs/PHASE_6_STATUS_2026_05_10.md`
