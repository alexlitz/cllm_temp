# IO Neural Bake Queue — 2026-06-09

Wave B of the override removal plan (per
`VANILLA_RESTORE_INVENTORY_2026_06_09.md`) deleted every runner-side
Python IO shim:

- `_inject_getchar` (run_vm.py:2511) — GETCHAR stdin shim
- `_handle_skipped_io_op` (run_vm.py:2590) — PRTF/READ/OPEN/CLOS/PUTCHAR
  bytecode-walking shim
- `_handle_pure_neural_stall` (run_vm.py:2761) — IO synthesis on neural
  PC freeze
- `_neural_prtf_emit` / `_neural_open_emit` / `_neural_clos_emit` /
  `_neural_read_emit` (run_vm.py:2802-2995) — pure_neural per-op AX
  override shims
- `_syscall_prtf` / `_syscall_open` / `_syscall_read` / `_syscall_clos`
  (run_vm.py:3242-3391) — handler-mode (pure_neural=False) syscall
  shims

The `_syscall_handlers` dispatch table is now empty; the
`_handle_skipped_io_op` / `_handle_pure_neural_stall` paths in
`_dispatch_step` are deleted; the batched runner's PRTF/OPEN/CLOS/READ
defer-to-serial block (batched_pure_neural.py:2157-2177) is deleted.

IO-dependent tests will now fail honestly (model emits AX=0 / no output
bytes / no shadow-memory writes) until each item below lands. **No new
Python override code may be added** — every replacement is a declarative
bake brief.

## Queue

### 1. V9 phase-2 user_input gather + getchar routing — replaces `_inject_getchar`

**Stubs:**
- `c4_release/neural_vm/unified_compiler/ops/user_input_ops.py:60` —
  `make_layer5_user_input_gather_op` (raises `NotImplementedError` when
  `enable=True`).
- `c4_release/neural_vm/unified_compiler/ops/user_input_ops.py:134` —
  `make_layer6_getchar_routing_op` (raises `NotImplementedError` when
  `enable=True`).

**What the bake must do:**

L5 attention head pair (heads 8 + 9 once L5 widens):
- Head A fires at `OP_GETCHAR & MARK_AX`, attends to
  `Token.USER_INPUT_START` via `IS_MARK` + dedicated
  `MARK_USER_INPUT_START` dim.
- Head B uses Head A's output position + cursor offset to gather the
  byte at `start + 1 + cursor` into `STDIN_BYTE_LO/HI` nibble dims via
  ALiBi position-bias arithmetic.

L6 FFN units 1600-1632 (past the PUTCHAR routing block at 1500-1531):
- Unit 1600: `OP_GETCHAR & MARK_AX -> IO_IS_GETCHAR` flag.
- Units 1601-1616: `STDIN_BYTE_LO[k] -> AX_CARRY_LO[k]` (k=0..15).
- Units 1617-1632: `STDIN_BYTE_HI[k] -> AX_CARRY_HI[k]` (k=0..15).
- Bytes 1..3 of AX stay zero via existing AX-carry zero-fill.

**Hard dependencies:** items 4 (token 269/270 embedding) + 5 (L5 head
widening 8→10) must land first.

**Plan doc:** `docs/V9_GETCHAR_READ_NEURAL_PLAN.md` §3.2-§3.3.

### 2. `prtf_think_protocol` bake — replaces `_neural_prtf_emit` + `_handle_skipped_io_op` (PRTF branch) + `_syscall_prtf`

**Stub:** `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4387`
(`make_prtf_think_protocol_op`, bake body is `return` no-op gated on
`enable_neural_io_think_protocol=False`).

**What the bake must do:**

Emit `Token.THINKING_END` at PRTF step boundary, walk format string in
attention-addressed shadow memory, emit format-substituted bytes one at
a time as raw byte tokens between `THINKING_END` and `THINKING_START`.

Building blocks needed:
- L8 format-position counter (cursor advance per byte emitted) —
  attends to `OP_PRTF & MARK_AX`, increments per `THINKING_END..STEP_END`
  span.
- L9 format-byte fetch — attends to `fmt_ptr + cursor` via ALiBi-encoded
  KV-absolute reads (same machinery as user_input gather in item 1).
- L10 null-terminator chain — detect `byte == 0` to gate the
  `THINKING_START` emit at end-of-string.
- L10 format-specifier handler — `%d/%s/%c/%x/%%` substitution from
  stack args (sp[1..argc-1]); reuses the multi-byte ALU for `%d`/`%x`.

**Plan doc:** `docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md` +
`docs/V9_PRTF_NEURAL_PLAN.md`.

### 3. `putchar_think_protocol` bake — completes neural PUTCHAR boundary

**Stub:** `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4278`
(`make_putchar_think_protocol_op`, bake body is no-op stub gated on
`enable_neural_io_think_protocol=False`).

**What the bake must do:**

At PUTCHAR step end, emit `Token.THINKING_END`, then AX byte 0 as a raw
byte token, then `Token.THINKING_START`. Reuses
`_set_io_putchar_routing` (vm_step.py:6987) for the AX_CARRY → OUTPUT
dim copy; the new bake just toggles the marker-token gates.

Currently `run_vm.py` reads AX byte 0 directly when
`enable_neural_io_think_protocol=False` (run_vm.py:2133-2136, NOT a
shim — it's a pure passthrough of the model's emitted AX). That direct
read stays until the think-protocol bake replaces it.

**Plan doc:** `docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md` §2.

### 4. IO marker tokens 269/270 embedding bakes

**Required by:** items 1, 2, 3.

The token-embedding bake must write:
- Token 269 (`USER_INPUT_START`) → `MARK_USER_INPUT_START = 1.0` (new
  dim, allocate in dim registry alongside `STDIN_BYTE_LO/HI`).
- Token 270 (`USER_INPUT_END`) → `MARK_USER_INPUT_END = 1.0`.
- Token 271 (`TOOL_CALL`) → existing `IS_MARK` already covers it; check
  whether OPEN/CLOS-specific marker dims are wanted (see
  `make_open_clos_tool_call_op` at l6_ops.py:4419 — dep-graph anchor
  for a future Phase A bake).

Use `TokenEmbeddingRule` (Phase 7.D vehicle); see
`docs/DECLARATIVE_BAKE_VISION.md` for the model-level bake pattern.

### 5. L5 head widening 8 → 10 for user_input gather

**Required by:** item 1.

`c4_release/neural_vm/unified_compiler/compile_full_vm_dynamic.py` (or
its config) currently sets L5 `num_heads=8`. Heads 0-7 are taken by
`_set_layer5_fetch`. User_input gather needs heads 8/9. Widen at
compile time (model rebuild needed) and re-pin existing L5 head
allocations through `attention_head_allocator` to confirm no overlap.

Worked example for adding heads to a layer:
`docs/ATTENTION_HEAD_IR_MIGRATION_PATTERN.md` (`_LN_HEAD_LAYOUT`
tables).

### 6. argv_setup neural subroutine (follow-up, not Wave B-blocking)

**Stub:** No bake exists for `__argv_setup` (BLOG_SPEC §764-786 C
source unimplemented). `IO_ARGC` / `IO_ARGV_INDEX` / `IO_NEED_ARGV`
dims declared at `embedding.py:105-110` but no op writes them.

**What's needed:** rewrite `_build_context` to emit the `<args>` table
per §755 (4-byte little-endian argc header + null-terminated strings),
then bake a neural subroutine that reads argc/argv into stack memory
at program entry.

Not blocking deletion: tests don't exercise argv end-to-end (no
`test_argv*` files; see IO_SUBSYSTEM_STATUS_2026_06_09.md TL;DR).

## Ratchet

When each item lands, the corresponding test cluster must move from
"fails after Wave B deletion" → "passes via neural bake". Track on the
inventory table in `VANILLA_RESTORE_INVENTORY_2026_06_09.md`.

## Pre-deletion baseline (Wave B start)

`CUDA_VISIBLE_DEVICES=1 python -m pytest c4_release/tests/test_smoke.py
--tb=no -q` → **46 passed, 5 failed** (pre-existing SI/LI memory
failures, unrelated to IO).

Post-deletion smoke must hold this baseline for non-IO tests. IO-shaped
tests (`test_pure_neural_io.py`, `test_io_speculation.py`,
`test_conversational_io_*`) are expected to fail honestly until the
bakes above land — they previously "passed" only via the deleted Python
shims.
