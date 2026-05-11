# Neural I/O via THINK-tag Protocol — Migration Plan

Status: design + Phase 1 PUTCHAR stub bake landed (gated `enable=False`).

## Canonical reference

`c4_release/docs/BLOG_SPEC.md` line 851:

> The Neural VM supports two I/O modes. In tool calling mode, I/O opcodes
> (PRTF, OPEN, READ, CLOS) emit a TOOL_CALL token at the end of the VM
> step, which the external runner intercepts to perform the actual I/O —
> reading files, formatting output, etc. — before resuming execution.
> **In neural I/O mode (the default), stdin and stdout are handled
> purely through transformer attention with no runner intervention.**
> Output works via the think-tag protocol: all VM computation happens
> inside THINK_START/THINK_END tags (hidden from the user), and when a
> printf executes, the model exits the think block, emits a character
> byte token (visible to the user), then re-enters the think block and
> continues execution. Input bytes are injected into the token stream
> between USER_INPUT_START/END markers, and the VM reads them via
> attention — position-tracking heads locate the markers, and a nibble
> cascade extracts the offset to index into the input buffer.

(See also BLOG_SPEC:853 — runtime library functions like `malloc/free/
memset/memcmp` are *not* tool calls; they are compiled into bytecode and
run entirely neurally. Only true I/O syscalls cross the host boundary.)

## Why this matters

Today (post-Phase 6, branch `f-phase8-scope`) the `pure_neural` runner
mode still hosts five Python shims that violate the canonical spec:

1. `_neural_prtf_emit`   (run_vm.py:1378)
2. `_neural_open_emit`   (run_vm.py:1450)
3. `_neural_clos_emit`   (run_vm.py:1480)
4. `_neural_read_emit`   (run_vm.py:1497)
5. `_inject_getchar`     (run_vm.py:1208)

Each of these does what the canonical neural-I/O mode is supposed to do
inside the transformer: format strings via Python `%`-substitution,
fetch bytes from `_stdin_buffer`, perform `os.open`/`os.read`/`os.close`,
and override the model's AX register. This is fine for the
`enable_tool_calling=True` path (the boundary is by-design external), but
it is *the wrong default* per BLOG_SPEC:851.

Existing `pure_neural` PUTCHAR is partially neural — `_set_io_putchar_routing`
(vm_step.py:6952) at the L6 FFN routes `AX_CARRY → OUTPUT_LO/HI`, so the
output head's next-token prediction is the byte. But the runner still reads
that byte off the just-completed step's `REG_AX` tokens
(run_vm.py:827–828) rather than collecting it from a model-emitted token
stream of the form `THINKING_END, byte, THINKING_START` interleaved with
the normal register sequence.

## Vocabulary already in place

The infrastructure for the protocol exists in the codebase — both tokens
and dim-positions are wired up. From `c4_release/neural_vm/vm_step.py:47`:

```python
THINKING_START      = 272  # <think>
THINKING_END        = 273  # </think>
IO_STATE_EMIT_BYTE  = 274  # internal state: emit output byte next
IO_STATE_EMIT_THINKING = 275  # internal state: emit THINKING_START next
USER_INPUT_START    = 269
USER_INPUT_END      = 270
TOOL_CALL           = 271
```

Embedding bake (`c4_release/neural_vm/unified_compiler/compiler.py:134-147`):
- `THINKING_START` → `MARK_THINKING_START`, `IS_MARK`, `CONST`, `TEMP+1`
- `THINKING_END`   → `MARK_THINKING_END`, `IS_MARK`, `CONST`, `TEMP+2`
- `IO_STATE_EMIT_BYTE`, `IO_STATE_EMIT_THINKING` → `IS_MARK`, `CONST`

L3 FFN op already detects `LAST_WAS_THINKING_END` and sets
`IO_IN_OUTPUT_MODE` (l3_ops.py:146-178). L10 FFN
(`_set_null_terminator_detection`) clears it when a null byte fires.
L15 FFN routes `OUTPUT_BYTE → OUTPUT` when `IO_IN_OUTPUT_MODE` is set
(`_set_conversational_io_output_routing`). All of this is gated by
`enable_conversational_io`, which is *off* in the `pure_neural_runner`
fixture today.

The path that already works (just not as the default) is the
`enable_conversational_io=True` runner branch (run_vm.py:534-583): the
model emits a THINKING_END token; the runner walks the format string
from shadow memory; emits byte tokens and a closing THINKING_START into
context; injects a synthetic step. It is a **hybrid**: model decides
*when* to emit, runner decides *what* bytes. The canonical mode requires
the model to decide *what* too.

## Target token stream

For PUTCHAR (one byte, AX byte 0) the neural-I/O step emits:

```
… [REG_PC pc0 pc1 pc2 pc3]   (normal start of step)
  [REG_AX ax0 ax1 ax2 ax3]
  [REG_SP sp0 sp1 sp2 sp3]
  [REG_BP bp0 bp1 bp2 bp3]
  [STACK0 s0 s1 s2 s3]
  [MEM addr0..3 val0..3]
  [THINKING_END]              (← replaces STEP_END for output ops)
  byte                        (← user-visible character)
  [THINKING_START]            (← re-enters think block)
  [STEP_END]                  (← step finally terminates)
```

The runner's `for i in range(max_steps * Token.STEP_TOKENS)` loop
(run_vm.py:499) must accept the 3 extra tokens inserted into the step
without breaking position-modulo arithmetic (`pos_in_step = i %
Token.STEP_TOKENS`). Two strategies:

  - **A (per-step variable)**: track a counter of "extra emitted
    THINKING tokens" per step and adjust `pos_in_step`. Simpler.
  - **B (re-define STEP_TOKENS)**: leave STEP_TOKENS at 35 but treat
    THINKING_END..THINKING_START runs as outside the step's 35-token
    contract. The forced-STEP_END override at pos 34 (run_vm.py:521-523)
    is already disabled when `pure_neural=True`, so this works as long
    as the model is trained to emit STEP_END at the *true* end of step,
    which it already is in `pure_neural` mode.

We prefer **B** because it matches what the spec actually says: "all VM
computation happens inside THINK_START/THINK_END tags". The 35-token
step continues to be the canonical VM-step structure; THINK tokens are
"out-of-band" emissions inserted between the computation tokens.

For PRTF (variable length, N bytes from format string), the same pattern
loops N times within a single semantic VM step. The model is the
authority over the loop termination (null terminator in the format
string). Phase 2 territory.

For GETCHAR (one byte from `USER_INPUT_START..USER_INPUT_END` block in
the context prefix), no THINKING tag emission is needed because the
*input* bytes are present in context up-front. Instead the model needs
to:
  - locate `USER_INPUT_START` via attention (position-tracking head)
  - track an "input position" counter across steps
  - read `context[USER_INPUT_START + 1 + pos]` into AX byte 0
  - advance the counter on each GETCHAR step

For READ (count bytes into a buffer in shadow memory), GETCHAR loops
into a multi-byte SC chain. Phase 3.

## Per-opcode emission contract

| Op       | Tokens emitted per step (beyond normal 35)             | Phase |
|----------|--------------------------------------------------------|-------|
| PUTCHAR  | THINKING_END, byte, THINKING_START                     | **1** |
| PRTF     | THINKING_END, byte_0, byte_1, ..., THINKING_START      | 2     |
| GETCHAR  | none (read from USER_INPUT_START block)                | 2     |
| READ     | none for runner-visible output; multi-byte memory write| 3     |
| OPEN     | TOOL_CALL (tool-call mode only) — neural mode N/A      | n/a   |
| CLOS     | TOOL_CALL (tool-call mode only) — neural mode N/A      | n/a   |

OPEN/CLOS are *always* tool calls — file descriptors and host syscalls
cannot be modeled inside the transformer. They are the only ops that
genuinely cross the host boundary (BLOG_SPEC:853).

## Bake layout for Phase 1 (PUTCHAR)

The model needs four pieces of weight-baked logic:

### B1. L5 FFN: detect ACTIVE_OPCODE_PUTCHAR + MARK_AX → IO_IS_PUTCHAR
**Status**: already done by `_set_io_putchar_routing` at L6 FFN unit
1500 (vm_step.py:6968-6975). No new bake needed.

### B2. L6 FFN: AX_CARRY → OUTPUT_BYTE_LO/HI when IO_IS_PUTCHAR
**Status**: already done by `_set_io_putchar_routing` at L6 FFN units
1501-1532 (vm_step.py:6977-6991). The bake writes `AX_CARRY → OUTPUT_LO/HI`
directly. For the THINK-tag protocol we want it to route to
`OUTPUT_BYTE_LO/HI` instead and let the THINK-tag state machine emit it
between THINKING_END and THINKING_START. Action: extend the bake to also
write the AX byte to `OUTPUT_BYTE_LO/HI` (separate from `OUTPUT_LO/HI`),
gated by the new `enable_neural_io_think_protocol` flag so the existing
runner-reads-AX path keeps working when the flag is False.

### B3. Step-end routing: emit THINKING_END instead of STEP_END at the end of a PUTCHAR step
This is the new piece. At the MEM marker position of a PUTCHAR step, the
model should predict THINKING_END as the next-token logit (not STEP_END).
The existing `convo_io_state_machine` bake
(setup_helpers.py:1789-1817) already does this for PRTF/READ via CMP[5]
and CMP[6]; we replicate the pattern with `IO_IS_PUTCHAR` (no relay
needed because IO_IS_PUTCHAR is set at the AX marker just like
IO_IS_PRTF, and the same `_set_conversational_io_relay_heads` pattern
copies it via attention to the SE position).

  Concretely: extend `_set_conversational_io_opcode_decode` (L5 FFN
  units 410-411) with a third unit that decodes PUTCHAR → `IO_IS_PUTCHAR`
  (it already exists at L6 FFN unit 1500; we re-emit at L5 so it
  participates in the AX→SE relay chain, OR re-route the L6
  relay-detection to use the existing flag).

  Then: at L6 FFN unit 1402 (next free unit above the 1400-1401
  convo-io state-machine units), AND `IO_IS_PUTCHAR` (relayed via CMP+7)
  with `NEXT_SE` → emit `NEXT_THINKING_END`, suppress `NEXT_SE`,
  set `IO_STATE`.

### B4. After THINKING_END: emit byte at next position
The L3 FFN op (l3_ops.py:158-178) already sets `IO_IN_OUTPUT_MODE` after
seeing `LAST_WAS_THINKING_END`. L15 FFN routing
(`_set_conversational_io_output_routing`) routes `OUTPUT_BYTE → OUTPUT`
when `IO_IN_OUTPUT_MODE`. For PUTCHAR the byte was stashed into
`OUTPUT_BYTE_LO/HI` at L6 (B2 above). Action: ensure that the OUTPUT_BYTE
slot is *populated* and *non-zero* (so the L10 null-terminator check
doesn't fire spuriously after a single PUTCHAR byte). The null-terminator
detection is gated on `IO_IN_OUTPUT_MODE > 5.0`, so as long as we don't
re-assert `IO_IN_OUTPUT_MODE` after the byte is emitted, we exit output
mode naturally on the next step.

### B5. After byte: emit THINKING_START, then STEP_END
This is the trickiest piece. After the byte is emitted, the model must
emit THINKING_START and then STEP_END. Two options:

  - **5a (count-based)**: maintain a counter dim (`IO_BYTES_EMITTED`)
    that increments on each output byte. When `IO_BYTES_EMITTED == 1`
    (single-byte PUTCHAR), set `NEXT_THINKING_START`. For PRTF, this
    counter compares against the format string length.

  - **5b (state-machine via IO_STATE)**: the existing convo-io state
    machine has 5 states (setup_helpers.py:1770-1774). We add a state
    transition `EMIT_BYTE → EMIT_THINKING_START → STEP_END`.

Phase 1 implements **5a** restricted to PUTCHAR (always exactly one
byte). The counter logic is trivial: increment on each emitted byte,
fire NEXT_THINKING_START when the counter equals the AX value of 1 byte
(always the case for PUTCHAR). PRTF will extend with comparison against
the format string length (via the `argc` mechanism or by walking until
null).

## Runner-side change

The runner's `run()` loop (run_vm.py:499) must:

1. **Recognize** THINKING_END token: track that we are "in output mode"
   for the next byte token. (Currently this is done only when
   `conversational_io=True` at line 535.)
2. **Collect** byte tokens emitted between THINKING_END and
   THINKING_START into `output`. These tokens are byte values
   (0-255), and the runner should append `chr(byte)` to output.
3. **Recognize** THINKING_START token: stop collecting bytes; expect the
   normal STEP_END to follow.

For Phase 1, the runner adds a `_neural_io_think_collector` state that
the loop consults when the new `enable_neural_io_think_protocol` flag is
True (constructor arg, defaults to False). When True, the PUTCHAR shim
in `_dispatch_step` (run_vm.py:827-828) is bypassed because the byte was
already collected via the token stream.

## Acceptance criteria for Phase 1

- `test_putchar_one_char` and `test_putchar_two_chars` pass when
  `enable_neural_io_think_protocol=True` on `pure_neural_runner`, and the
  output is collected from the model's emitted token stream (not from
  `_extract_register(context, Token.REG_AX)`).
- The existing path (`enable_neural_io_think_protocol=False`, default)
  produces byte-identical output for the same tests.
- No new failures in: `test_smoke.py::test_imm_exit`,
  `test_runtime_vanilla.py`, `test_layer_idx_consistency.py`,
  `test_compile_determinism.py`.
- Phase 1 deletes `_inject_getchar` and the PUTCHAR `output.append(chr(...))`
  shim only when the flag flips to default-True.

Phase 1 deliverable in this commit: the gated `enable_neural_io_think_protocol`
flag (defaulting False), the design doc you are reading, and the
scaffolding bake op (`make_putchar_think_protocol_op`) that exposes the
state-machine wiring needed. The actual weight bakes that flip the
contract to "neural emits THINKING_END/byte/THINKING_START" are stubbed
behind the flag and will be filled in by a follow-up bake commit. This
matches the migration pattern used by `make_tool_call_*_op` and
`make_convo_io_*_op`: register unconditionally, no-op when flag is False,
flip the flag to True on the worker that lands the actual weight wiring.

## Phase 2/3 follow-ups

### Phase 2: PRTF (variable-length output) and GETCHAR (single-byte input)

**PRTF** — the same `THINKING_END, byte_0, ..., byte_N, THINKING_START`
shape, but N comes from walking the format string in memory. The model
needs:
  - A byte-position counter (FORMAT_PTR + offset) that advances on each
    emitted byte.
  - The L15 memory lookup head to fetch byte at `FORMAT_PTR + offset`.
  - A null-byte detection that fires `NEXT_THINKING_START` when the
    fetched byte is zero (already exists at L10 FFN unit 1864 via
    `_set_null_terminator_detection`).
  - Format-specifier (`%d`, `%s`, etc.) handling: this is the hard part.
    `%d` requires an int-to-decimal-string subroutine baked in. `%s`
    requires nested format-string walking (fetch arg pointer from stack,
    follow it to a new string, emit until null, return). These are
    multi-step subroutines that may need their own attention heads or
    bytecode-level emulation.

For Phase 2 we restrict to **literal format strings only** (no
specifiers). `%d` + friends move to Phase 2b.

**GETCHAR** — the spec describes "position-tracking heads locate the
markers, and a nibble cascade extracts the offset to index into the
input buffer". Concretely:
  - An attention head at L7/L8 reads the `USER_INPUT_START` marker's
    position. Output a one-hot in some dim.
  - A counter dim (`USER_INPUT_POS`) increments on each GETCHAR.
  - L15 memory lookup head reads
    `context[USER_INPUT_START + 1 + USER_INPUT_POS]` via the nibble
    cascade (same mechanism as MEM ADDR_KEY at l15_ops.py).
  - The byte goes into AX byte 0.

This needs a new dim allocation (`USER_INPUT_POS`) and a new attention
head pinned to L7 or L8 (since L15 needs the input position to compute
ADDR_KEY).

### Phase 3: READ (multi-byte input)

Same as GETCHAR in a loop: read count bytes from
`USER_INPUT_START..USER_INPUT_END`, write each to
`buf_ptr + i` via the SC mechanism (`_set_layer14_mem_generation`).

The interesting wrinkle: READ must complete the loop within a single
semantic VM step (the model can't easily emit multiple STEP_END tokens
for a single READ). Either:
  - Loop unrolling: emit N synthetic intermediate steps (one per byte),
    each with its own SC. The runner accepts this without re-running
    `_dispatch_step`.
  - In-step iteration: use the existing memory-write head 12 times in
    parallel. This requires the count to be known statically.

For most C programs `READ(fd, buf, count)` has a fixed `count`, so the
in-step approach is viable. We defer the bake until Phase 3.

## File-by-file change inventory for Phase 1

  - `c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md` — this file.
  - `c4_release/neural_vm/unified_compiler/ops/l6_ops.py` — new
    `make_putchar_think_protocol_op()` factory.
  - `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py` —
    register the new op.
  - `c4_release/neural_vm/run_vm.py` — add
    `enable_neural_io_think_protocol` constructor flag and runner-side
    THINKING_END/byte/THINKING_START token collector (gated by the
    flag). Existing PUTCHAR shim path remains the default.

## Notes on dim allocation

The protocol does NOT require new dims for Phase 1 — everything reuses
existing slots:
  - `IO_IS_PUTCHAR`        — already in BD (vm_step.py / dim_registry).
  - `IO_IN_OUTPUT_MODE`    — already in BD.
  - `IO_STATE`             — already in BD.
  - `IO_OUTPUT_COMPLETE`   — already in BD.
  - `NEXT_THINKING_END`    — already in BD.
  - `NEXT_THINKING_START`  — already in BD.
  - `OUTPUT_BYTE_LO/HI`    — already in BD (16+16 = 32 nibble slots).
  - `LAST_WAS_THINKING_END`— already in BD.

For Phase 2 (PRTF byte counter) we may need:
  - `FORMAT_PTR_LO/HI`     — already in BD
    (`_set_format_pointer_extraction` uses them).
  - `FORMAT_OFFSET`        — new dim, 4 nibbles, for the byte-position
    counter that increments on each emitted byte.

For Phase 2 GETCHAR we need:
  - `USER_INPUT_POS`       — new dim, 4 nibbles, byte-position counter
    that increments on each GETCHAR.
  - `USER_INPUT_BASE_LO/HI`— new dims, capturing `USER_INPUT_START`
    position via an attention head.

The new dim allocations are deferred to the phase that uses them.

## Why the work is gated `enable=False`

Three reasons:
1. **Bake validation requires GPU runs**, and the existing tests in
   `test_pure_neural_io.py` already cover PUTCHAR via the AX-readoff
   path. We can land the design + scaffolding without flipping the
   contract.
2. **Avoid double-bake conflicts** with the existing
   `_set_io_putchar_routing` units (1500-1532 on L6 FFN). The new path
   writes to `OUTPUT_BYTE_LO/HI` (different slots), and the runner-side
   handler is mode-conditional, so both paths can coexist behind the
   flag while we validate.
3. **Phase 2/3 follow-up work** (PRTF format-string walking, GETCHAR
   nibble cascade) is non-trivial. Phase 1 is just the THINK-tag
   protocol skeleton. We need to land the skeleton before the deeper
   bakes, but we shouldn't flip the protocol on until those deeper
   bakes also work.
