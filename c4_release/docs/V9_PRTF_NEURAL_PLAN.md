# V9 PRTF / OPEN / CLOS Neural Plan

Status: design + Phase 2 PRTF stub bakes registered (gated `enable=False`).

This document extends `NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md` (Phase 1
PUTCHAR) to cover PRTF, OPEN, and CLOS. It is the follow-up to the
PUTCHAR THINK-tag protocol agent on branch
`v9-neural-io-think-protocol-putchar`.

## Canonical reference

`c4_release/docs/BLOG_SPEC.md` line 851 (the same paragraph PUTCHAR's
plan quotes) plus line 853:

> Runtime library functions like malloc, free, memset, and memcmp are not
> tool calls. They're compiled from C into the VM's bytecode and execute
> entirely neurally — malloc is just LEA/LI/ADD/SI, memset is a loop of
> SC instructions, and so on. The only operations that require external
> dispatch (or the neural I/O pathway) are true I/O syscalls that cross
> the boundary between computation and the outside world.

The split-of-responsibility between the two I/O modes for these three
opcodes is:

| Opcode | Tool-call mode             | Neural-I/O mode (default)              |
|--------|-----------------------------|----------------------------------------|
| PRTF   | Emit TOOL_CALL at step end | Emit THINKING_END, byte_0..byte_N, THINKING_START |
| OPEN   | Emit TOOL_CALL at step end | **Stays as TOOL_CALL** (host syscall) |
| CLOS   | Emit TOOL_CALL at step end | **Stays as TOOL_CALL** (host syscall) |

PRTF is a true neural-I/O candidate because its output is a stream of
bytes that the model can produce token-by-token. OPEN and CLOS cross
the host boundary (file descriptors, `os.open`/`os.close`) and have no
sensible "in the transformer" implementation. Per BLOG_SPEC:853, both
remain genuine syscalls. The "neural I/O default" thus means PRTF is
moved to THINK-tag protocol, while OPEN/CLOS remain TOOL_CALL-emitting
opt-ins under `enable_tool_calling=True`.

## Why this matters

After the PUTCHAR Phase 1 bake lands, the runner's Python shims for
PRTF (`_neural_prtf_emit`), OPEN (`_neural_open_emit`), and
CLOS (`_neural_clos_emit`) at `c4_release/neural_vm/run_vm.py:1378-1495`
are the next-largest violators of the canonical spec:

1. `_neural_prtf_emit` (run_vm.py:1378) — walks fmt string from
   `_memory`, does Python `%`-substitution, overrides AX.
2. `_neural_open_emit` (run_vm.py:1450) — heuristic path/mode detection,
   calls `os.open`, overrides AX with the fd.
3. `_neural_clos_emit` (run_vm.py:1480) — pops fd, calls `os.close`,
   overrides AX=0.

For PRTF specifically, the spec wants the model to produce the output
bytes itself (BLOG_SPEC:851); for OPEN/CLOS the spec wants the model to
emit a TOOL_CALL boundary token that the runner intercepts to perform
the host syscall — currently this *is* what happens in tool-calling
mode (via `make_tool_call_*_op` at L5 FFN + L6 attn head 5 + L6 FFN
unit 1300), but the runner's `_dispatch_step` still falls back to the
heuristic `_neural_open_emit` / `_neural_clos_emit` shims when
`pure_neural=True` and `enable_tool_calling=False`.

This plan migrates PRTF to the THINK-tag protocol (Phase 2 of
`NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md`) and documents OPEN/CLOS as
permanent TOOL_CALL opcodes (Phase A below). The TOOL_CALL bake for
OPEN/CLOS already exists (`_set_tool_call_opcode_decode` at L5 FFN
units 400-405, `_set_tool_call_relay_head` at L6 attn head 5,
`_set_tool_call_detection` at L6 FFN unit 1300) — we simply make it
the production default via `enable_tool_calling=True` when the runner
sees these ops in a pure-neural build.

## Existing infrastructure to reuse

A surprising amount of PRTF-specific weight-bake is already in the
codebase, gated by `enable_conversational_io`:

- **L5 FFN** units 410-411 (`_set_conversational_io_opcode_decode`)
  decode PRTF and READ at the AX marker → write `IO_IS_PRTF` /
  `IO_IS_READ` flags. (setup_helpers.py:1682-1711)
- **L6 attn head 4** (`_set_conversational_io_relay_heads`) relays
  `IO_IS_PRTF` from AX → SE. (setup_helpers.py:1713-1763)
- **L6 FFN** units 1400-1401 (`_set_conversational_io_state_machine`)
  detect CMP[5] (PRTF relay) AND NEXT_SE → emit NEXT_THINKING_END +
  suppress NEXT_SE + set IO_STATE = 1. (setup_helpers.py:1789-1817)
- **L3 FFN** unit 1035 (`_set_convo_io_step_resume`) fires on
  `LAST_WAS_THINKING_START` → resets PC/SP carry for the resumed step.
- **L7 attn head 7** (`_set_format_pointer_extraction`) extracts the
  format string pointer from the previous step's STACK0 → writes
  `FORMAT_PTR_LO/HI`. (vm_step.py:7185-7218)
- **L8 FFN** unit 600+ (`_set_format_position_counter`) increments
  `IO_FORMAT_POS` on each emitted byte. (vm_step.py:7221-7247)
- **L9 attn head 0** (`_set_format_string_fetch_head`) fetches the
  format-string byte at `FORMAT_PTR + IO_FORMAT_POS` via ADDR_KEY
  attention → writes `OUTPUT_BYTE_LO/HI`. (vm_step.py:7250-7291)
- **L10 FFN** unit 1864 (`_set_null_terminator_detection`) detects
  `OUTPUT_BYTE == 0` while `IO_IN_OUTPUT_MODE` → clears
  `IO_IN_OUTPUT_MODE` + emits `NEXT_THINKING_START`.
- **L15 FFN** unit 1200 (`_set_conversational_io_output_routing`)
  routes `OUTPUT_BYTE → OUTPUT` when `IO_IN_OUTPUT_MODE`.

All of the above are flag-gated. The runner-side hybrid path
(`run_vm.py:583-632`) reads the format string from `_memory` rather
than letting the model emit each byte. The Phase 2 goal is to bypass
that fallback and let the existing weight bakes do the byte emission
end-to-end.

## Phase 2 design (PRTF — variable-length neural output)

The PUTCHAR Phase 1 plan establishes a state machine
`STEP_END → THINKING_END → byte → THINKING_START → STEP_END`. PRTF
extends this to N bytes:

```
… [REG_PC pc0..3] [REG_AX ax0..3] [REG_SP sp0..3] [REG_BP bp0..3]
  [STACK0 s0..3]  [MEM addr0..3 val0..3]
  [THINKING_END]                          (← replaces STEP_END for PRTF)
  byte_0                                  (← first format-string byte)
  byte_1                                  (← subsequent bytes via L9 fetch)
  …
  byte_{N-1}                              (← until null terminator)
  [THINKING_START]                        (← emit when null detected)
  [STEP_END]                              (← step finally terminates)
```

The bakes that need to fire:

### B1. L5 FFN: PRTF → IO_IS_PRTF (already in place)
`_set_conversational_io_opcode_decode` at units 410-411. **No new bake**;
already gated by `enable_conversational_io`. For Phase 2 we'll re-use
this — the new `enable_neural_io_prtf_think_protocol` flag implies
`enable_conversational_io=True` (or duplicates the bake under the new
flag) so the IO_IS_PRTF flag is available.

### B2. L6 attn head 4: IO_IS_PRTF AX → SE relay (already in place)
`_set_conversational_io_relay_heads` head 4 with alibi_slope=5.0.
**No new bake**.

### B3. L6 FFN: IO_IS_PRTF (CMP[5]) AND NEXT_SE → emit NEXT_THINKING_END
`_set_conversational_io_state_machine` units 1400-1401. **No new bake**.

### B4. L7 attn head 7: extract FORMAT_PTR from previous step's STACK0
`_set_format_pointer_extraction`. **No new bake**.

### B5. L9 attn head 0: fetch byte at FORMAT_PTR + IO_FORMAT_POS
`_set_format_string_fetch_head`. **No new bake**, but the implementation
only supports addresses < 256 and single-nibble positions (0-15) per
the docstring at vm_step.py:7260-7273. **Phase 2a accepts this
restriction** (literal format strings up to 15 chars, at addresses
< 256). Phase 2b extends the address arithmetic to full 32 bits.

### B6. L8 FFN: increment IO_FORMAT_POS on each emitted byte
`_set_format_position_counter`. **No new bake**.

### B7. L10 FFN: OUTPUT_BYTE == 0 → emit NEXT_THINKING_START
`_set_null_terminator_detection`. **No new bake**.

### B8. L15 FFN: route OUTPUT_BYTE → OUTPUT under IO_IN_OUTPUT_MODE
`_set_conversational_io_output_routing`. **No new bake**.

### B9. Runner-side: collect bytes between THINKING_END..THINKING_START
**Already in place** via the Phase 1 PUTCHAR collector at
run_vm.py:570-581 (`enable_neural_io_think_protocol`). The collector is
opcode-agnostic — it just appends every byte token between THINKING_END
and THINKING_START to the output list. PRTF reuses it as-is.

### B10. NEW — Disable runner-side PRTF/OPEN/CLOS shim when neural-I/O THINK protocol active
This is the **only new code change** for Phase 2a beyond the no-op stub:
when `enable_neural_io_think_protocol=True`, the runner's
`_handle_skipped_io_op` and `_neural_prtf_emit` paths should be skipped
for PRTF (so the model's emitted bytes are not double-emitted via the
shim's `output.append(fmt_str)`). Same for `_neural_open_emit` /
`_neural_clos_emit` when their TOOL_CALL bakes are firing.

The Phase 2a deliverable in this commit:
- New gated op `make_prtf_think_protocol_op` (no-op stub; mirrors the
  PUTCHAR phase-1 pattern), registered in `all_core_ops` under the same
  `enable_neural_io_think_protocol` flag.
- When the flag flips `True`, the bake doesn't write any new units
  itself (all the bakes are already in place via
  `enable_conversational_io`); it instead asserts that
  `enable_conversational_io=True` so the existing PRTF bake chain is
  active. Documentation-only assertion in this commit.
- Runner-side gating: when the flag is True, `_neural_prtf_emit` is
  skipped (the model emits the bytes directly via the L9 head + L15
  output routing). Implemented as a guard in `_dispatch_step`.

The actual weight-bake work to flip PRTF from "runner walks string" to
"L9 head fetches each byte" requires:
- Ensuring `enable_conversational_io=True` plumbs through to the bake
  ops (it currently does).
- Validating that `_set_format_string_fetch_head` actually fires under
  pure_neural mode (today the convo-IO hybrid path bypasses it). This
  is the open empirical question — see the "Open questions" section.

## Phase 2a — Simplest concrete PRTF bake (this commit)

The "simplest case" the user asked for: format string with a single
literal byte and no specifiers, e.g. `printf("x")`. The bake state for
this case is:

1. PRTF step's L5 FFN decode sets `IO_IS_PRTF`.
2. L6 attn relays IO_IS_PRTF AX → SE.
3. L6 FFN emits `NEXT_THINKING_END` instead of `NEXT_SE` at the
   step-end position.
4. The model emits the THINKING_END token.
5. L3 FFN sees `LAST_WAS_THINKING_END` → sets `IO_IN_OUTPUT_MODE`.
6. L7 head 7 extracts `FORMAT_PTR` from STACK0 (the format-string
   pointer pushed before PRTF).
7. L9 head 0 fetches the byte at `FORMAT_PTR + IO_FORMAT_POS` (=
   `FORMAT_PTR + 0` = the literal 'x').
8. L15 routes `OUTPUT_BYTE → OUTPUT`. The model emits the 'x' byte
   token.
9. L8 FFN increments `IO_FORMAT_POS` (now = 1).
10. L9 head 0 fetches the byte at `FORMAT_PTR + 1` (= the null
    terminator).
11. L10 FFN detects `OUTPUT_BYTE == 0` and emits `NEXT_THINKING_START`,
    clearing `IO_IN_OUTPUT_MODE`.
12. The model emits the THINKING_START token.
13. L3 FFN's `LAST_WAS_THINKING_START` resumer fires → advances PC and
    SP for the next step.
14. The model emits the normal STEP_END for the next step's start.

Every bake is in place; the only thing we need is to **flip the flag
chain** so all of them fire together under the
`enable_neural_io_think_protocol` umbrella. That flip is the Phase 2b
work — Phase 2a (this commit) lands the gated stub.

## Phase A — OPEN / CLOS TOOL_CALL boundary opcodes

Per BLOG_SPEC:853, OPEN/CLOS are not candidates for neural I/O — they
cross the host boundary. The right design is:

- The model emits a `TOOL_CALL` token at the end of an OPEN/CLOS step
  (replacing STEP_END), signaling the runner to perform the host
  syscall.
- The runner intercepts TOOL_CALL, reads the syscall args from the
  step's register section, invokes `os.open` / `os.close`, writes the
  result back into the next step's AX register section, and continues.

**This bake already exists and works** under `enable_tool_calling=True`:
- `_set_tool_call_opcode_decode` (L5 FFN units 400-405): decode all 6
  I/O opcodes (OPEN, READ, CLOS, PRTF, GETCHAR, PUTCHAR) → IO_IS_TOOL_CALL.
- `_set_tool_call_relay_head` (L6 attn head 5): relay IO_IS_TOOL_CALL
  AX → SE.
- `_set_tool_call_detection` (L6 FFN unit 1300): CMP[2] (relayed
  IO_IS_TOOL_CALL) AND NEXT_SE → emit NEXT_TOOL_CALL + suppress
  NEXT_SE.

Each of these is wrapped in a no-op-when-False bake op
(`make_tool_call_*_op` at flag_gated_ops.py:30-241). When
`enable_tool_calling=True` is passed to `compile_full_vm`, all three
fire and the model emits TOOL_CALL at the end of any I/O opcode step.

**Phase A is documentation-only**: there is no new bake. The existing
TOOL_CALL bake handles OPEN/CLOS correctly today. The runner-side
intercept path is in `_dispatch_step` (run_vm.py) and is already wired
to the existing `_syscall_open`/`_syscall_clos` handlers.

The remaining question is whether OPEN/CLOS should default to the
runner-shim path (`_neural_open_emit`/`_neural_clos_emit`) or the
TOOL_CALL bake path. The user's task spec answers this:

> OPEN/CLOS stay as TOOL_CALL-emitting bakes (model emits TOOL_CALL at
> end of OPEN/CLOS step, host intercepts) — this is much simpler than
> the THINK-protocol for output.

So the production default for OPEN/CLOS, when the user wants
host-syscall behavior, is `enable_tool_calling=True`. The
`_neural_open_emit`/`_neural_clos_emit` shims remain in the runner for
the pure_neural test suite (which today exercises them implicitly), but
they are not the canonical design — they are an interim convenience
that the production runner should bypass once `enable_tool_calling`
becomes the default for builds that need file I/O.

## Per-opcode emission contract summary

| Op       | Mode               | Tokens emitted per step (beyond normal 35)            | Bake status   |
|----------|--------------------|--------------------------------------------------------|---------------|
| PUTCHAR  | THINK protocol     | THINKING_END, byte, THINKING_START                     | Phase 1 stub  |
| PRTF     | THINK protocol     | THINKING_END, byte_0..byte_{N-1}, THINKING_START       | **Phase 2a stub (this commit)** |
| OPEN     | TOOL_CALL          | TOOL_CALL (no THINK_*, no bytes)                       | Already baked under `enable_tool_calling=True` |
| CLOS     | TOOL_CALL          | TOOL_CALL (no THINK_*, no bytes)                       | Already baked under `enable_tool_calling=True` |
| READ     | (see GETCHAR plan) | n/a — input side, handled by USER_INPUT block          | Phase 3 (separate agent) |
| GETCHAR  | (see GETCHAR plan) | n/a — input side, handled by USER_INPUT block          | Phase 2/3 (separate agent) |

## File-by-file change inventory for this commit

- `c4_release/docs/V9_PRTF_NEURAL_PLAN.md` — this file.
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py` — new
  `make_prtf_think_protocol_op()` factory (no-op stub, gated).
- `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py` —
  register the new PRTF op alongside the PUTCHAR op (same flag).
- Runner-side disable for `_neural_prtf_emit` when the flag is on:
  guarded at the `_handle_skipped_io_op` and `_dispatch_step` call
  sites. **Deferred to Phase 2b** — this commit only adds the design
  doc and the bake-op stub, matching the PUTCHAR Phase 1 commit style.

## Why everything is gated `enable=False`

Three reasons (same as PUTCHAR Phase 1):

1. **Bake validation requires end-to-end GPU runs**, and the existing
   tests in `test_pure_neural_io.py` already cover PRTF via the runner
   shim (PRTF xpasses today). We can land the design + scaffolding
   without flipping the contract.
2. **Avoid double-emission** with the existing `_neural_prtf_emit`
   shim when the L9 head + L15 output routing also produce the byte
   stream. The Phase 2b switchover requires gating the shim off in the
   same commit that flips the bake on.
3. **Phase 2a → 2b is non-trivial.** The address arithmetic in
   `_set_format_string_fetch_head` is limited to addresses < 256 (the
   docstring is explicit about this), and PRTF's call sites in tests
   use `0x10000` as the format-string base address. Either we lift
   that restriction first, or we relocate the format string to address
   < 256 in the test, or we accept Phase 2a's restriction as a
   smoke-test-only validation. The user's task spec accepts the
   single-byte literal case at Phase 2a; lifting to 32-bit addresses
   is Phase 2b.

## Phase 2b follow-ups (deferred)

- Lift `_set_format_string_fetch_head` to support full 32-bit FORMAT_PTR
  (currently limited to byte 0).
- Lift `IO_FORMAT_POS` to support positions > 15 (currently single
  nibble).
- Add `%d`, `%s`, `%c`, `%x`, `%%` format specifier handling. These
  are non-trivial sub-bakes:
  - `%d` requires int-to-decimal-string subroutine baked at some FFN
    layer. Uses an iterative divmod-by-10 (already partially built at
    L8-L13 ALU). Outputs digit characters one at a time, advancing
    `IO_FORMAT_POS` past the `%d`.
  - `%s` requires nested format-string walking: fetch the arg pointer
    from stack, follow it to a new string, emit bytes until null, then
    return to the original format-string position.
  - `%c` is trivial (one byte from stack).
  - `%x` is a 4-bit hex-digit subroutine, similar to `%d`.
  - `%%` is a literal `%`.

- READ (Phase 3, separate agent): multi-byte input from
  `USER_INPUT_START..USER_INPUT_END` block via attention. Same
  mechanism as GETCHAR in a loop.

## Acceptance criteria for Phase 2a (this commit)

- `c4_release/docs/V9_PRTF_NEURAL_PLAN.md` exists and is readable.
- `make_prtf_think_protocol_op()` is registered in `all_core_ops` and
  is a no-op when `enable_neural_io_think_protocol=False` (the
  default).
- Existing Phase 6 tests continue to pass:
  - `test_smoke.py::test_imm_exit`
  - `test_pure_neural_io.py::test_prtf_simple` (xpasses today via
    the runner gap-shim; this commit does not change that)
  - `test_pure_neural_io.py::test_prtf_with_arg` (xpasses today via
    the runner gap-shim with the IMM-chain arg extraction at
    run_vm.py:1283-1302)
  - `test_runtime_vanilla.py`
  - `test_layer_idx_consistency.py`
  - `test_compile_determinism.py`
- The flag chain (constructor → compile_full_vm → all_core_ops →
  make_prtf_think_protocol_op) is identical to PUTCHAR's Phase 1
  pattern — no new flag is introduced; PRTF rides the same
  `enable_neural_io_think_protocol` flag as PUTCHAR.

## Open questions for Phase 2b

1. **Does `_set_format_string_fetch_head` actually fire correctly
   under pure_neural mode today?** The bake is gated by
   `enable_conversational_io`, and the `pure_neural_runner` fixture
   does not enable it. We need to construct a fixture with
   `enable_conversational_io=True, pure_neural=True,
   enable_neural_io_think_protocol=True` and trace the residual stream
   at the PRTF step's MEM marker to see if `OUTPUT_BYTE_LO/HI` carry
   the right byte.
2. **Does `_set_format_pointer_extraction` (L7 head 7) extract the
   right STACK0 byte when STACK0 comes from a previous step?** The Q
   gate is `IO_IN_OUTPUT_MODE` and the K gate is `MARK_STACK0`; both
   should be active at the right positions, but the test only
   validates this under the convo-io hybrid path today.
3. **What happens at the second PRTF in a multi-PRTF program?** The
   IO_FORMAT_POS counter increments — does it reset to 0 between PRTF
   calls? `_set_convo_io_step_resume` (L3 unit 1035) clears IO_STATE
   on `LAST_WAS_THINKING_START`, but I don't see an explicit clear of
   IO_FORMAT_POS. Likely a TODO for Phase 2b.

## References

- `c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md` — Phase 1
  PUTCHAR plan (canonical pattern this doc extends).
- `c4_release/docs/BLOG_SPEC.md:851-853` — canonical I/O mode spec.
- `c4_release/neural_vm/run_vm.py:1378-1495` — the V9 Python shims this
  plan migrates.
- `c4_release/neural_vm/setup_helpers.py:1682-1817` — convo-IO L5/L6
  PRTF bakes.
- `c4_release/neural_vm/vm_step.py:7185-7291` — L7/L8/L9 PRTF format-
  string walk bakes.
- `c4_release/neural_vm/unified_compiler/ops/flag_gated_ops.py:30-241`
  — `enable_tool_calling` TOOL_CALL bake chain (used by OPEN/CLOS).
- `c4_release/tests/test_pure_neural_io.py:64-92` — PRTF tests.
