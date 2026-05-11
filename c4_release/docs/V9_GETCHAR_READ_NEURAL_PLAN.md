# V9 — Neural GETCHAR / READ via USER_INPUT_START/END markers

**Status:** Phase 1 (scaffold). The L5 attention-head bake is *registered but
disabled* (`enable=False`). The runner-side `_inject_getchar` and
`_neural_read_emit` shims still own the byte transfer. Flipping the flag is
gated on the L6 routing weights that take the USER_INPUT-derived byte and
drive it onto `AX_CARRY_LO/HI` (so the model emits the correct AX byte 0 at
the next AX marker) being baked in and verified end-to-end.

Owner: f-phase8-scope branch (V9 series).

## 1. The spec contract

`BLOG_SPEC.md:851`:

> Input bytes are injected into the token stream between USER_INPUT_START/END
> markers, and the VM reads them via attention — position-tracking heads
> locate the markers, and a nibble cascade extracts the offset to index into
> the input buffer.

## 2. The current Python state we are replacing

| Location | Behaviour | Branch |
|----------|-----------|--------|
| `run_vm.py::_inject_getchar` | reads `_stdin_buffer[_stdin_pos]`, overrides REG_AX bytes in the last completed step, advances `_stdin_pos`. | V11 stdin buffer + V9 AX override shim. |
| `run_vm.py::_neural_read_emit` | same `_stdin_buffer` consumption, writes bytes into shadow `_memory`, also `_inject_mem_section` to surface MEM tokens. | V11 + memory. |
| `run_vm.py::_build_context` | already emits `[USER_INPUT_START, *stdin_bytes, USER_INPUT_END]` in the prefix when `stdin != ""`. | Already correct per spec. |
| `embedding`/`compiler.py` | `Token.USER_INPUT_START=269`, `Token.USER_INPUT_END=270` already have `BD.IS_MARK=1.0`. | Already correct per spec. |

So the **token-stream side of the spec is already in place**: the runner
builds the USER_INPUT block, and the embedding table marks the two marker
tokens as `IS_MARK`. What is missing is the neural-network side: no attention
head reads a byte out from inside the USER_INPUT block at GETCHAR / READ
steps, and no FFN unit routes that byte into AX.

## 3. The neural read architecture

### 3.1 Cursor representation

We need a **read-cursor** that advances by 1 byte per GETCHAR and by `count`
per READ. The cursor lives in the residual stream of the *current* step.
There are two viable encodings:

1. **Offset-from-USER_INPUT_START (preferred for V9 phase 1):** carry the
   running offset as a nibble cascade through TEMP slots, incremented by 1 at
   each GETCHAR AX marker. This is symmetric with how PC is carried forward
   across steps.
2. **Mark-the-consumed-bytes (alternate):** rewrite the consumed byte in the
   token stream to a sentinel (e.g. 0xFF) after the read. Lighter weight at
   runtime but breaks immutability of the prefix.

V9 picks option (1). The cursor lives at `STDIN_CURSOR_LO`/`STDIN_CURSOR_HI`
nibble dims, allocated alongside `AX_CARRY_LO/HI` (see §5 for the dim
allocation plan).

### 3.2 Position-tracking heads (the spec's "position-tracking heads")

We need at least **two attention heads at L5** that fire only at the AX
marker of a GETCHAR step:

- **Head A — USER_INPUT_START locator.** Q reads `OP_GETCHAR ∧ MARK_AX`.
  K reads `USER_INPUT_START` (`Token.USER_INPUT_START` has `IS_MARK=1` but
  not a dedicated `MARK_USER_INPUT_START` dim; we add one — see §5). V copies
  the *position* of the start marker into a `USER_INPUT_START_POS_*` nibble
  cascade.

- **Head B — Byte gather.** Q reads `(OP_GETCHAR ∧ MARK_AX) ∧ POS = start+1+cursor`
  using ALiBi position bias plus the cursor cascade. K reads `IS_BYTE` over
  the entire prefix. V copies the `EMBED_LO/HI` nibbles of the matched byte
  into `STDIN_BYTE_LO/HI` (a new dim pair — see §5). This is structurally
  identical to the L5 fetch head that reads opcode bytes out of the CODE
  section; we re-use `P.threshold_attention` style scoring with a position
  computed by an FFN combining `USER_INPUT_START_POS + cursor`.

In practice the position math is the trickiest part. The simplest staging is
a **two-step bake**:

1. **Phase 1 (this commit, disabled):** assume cursor = 0 always. Head B
   becomes a single fixed-offset gather: position `USER_INPUT_START + 1`,
   i.e. the *first* byte after the marker. That suffices for the
   `test_getchar_returns_byte` smoke (single GETCHAR, stdin="x").

2. **Phase 2 (follow-up):** plumb the nibble cascade cursor. An L4 FFN
   increments the cursor at every GETCHAR AX marker. Head B's K-side position
   matching uses the cursor cascade.

3. **Phase 3:** READ — same bytes, count > 1. The runner's `READ` shim
   stays in place until READ-into-memory is migrated (a separate bake; see
   §6).

### 3.3 Routing into AX (the L6 side)

Head B writes `STDIN_BYTE_LO/HI` at the AX marker position. We then need a
small L6 FFN unit pair (mirroring the L6 PUTCHAR routing units 1500+) that:

- Fires only when `OP_GETCHAR ∧ MARK_AX`.
- Routes `STDIN_BYTE_LO/HI → AX_CARRY_LO/HI`.
- The existing AX byte-0 emit pipeline then turns AX_CARRY into the byte
  token emitted at AX byte 0.

Bytes 1..3 of AX must be 0 (GETCHAR returns the byte zero-extended; EOF is
-1 / 0xFFFFFFFF handled separately — see §4). Mirror the PUTCHAR routing
loop structure (`for k in range(16):`).

### 3.4 EOF handling

When the cursor ≥ stdin length, the model should produce -1 (0xFFFFFFFF) in
AX. Two ways:

- **A.** A second attention head fires at AX marker only when the gather
  head returns 0 weights (i.e. all USER_INPUT bytes are exhausted). This
  requires a "no-match" sentinel. The simplest sentinel is to also match
  USER_INPUT_END: if the matched position is past END, route 0xFF to all 4
  AX bytes.

- **B.** Keep EOF in the Python shim. Pragmatic until READ neural bake is
  ready.

V9 phase 1 leaves EOF to Python (the smoke test only exercises non-EOF).

## 4. Read-cursor advancement

For the disabled-by-default phase 1 bake, cursor stays at 0. When the bake is
flipped on for tests beyond `test_getchar_returns_byte`, we need cursor
state that survives across VM steps.

Candidate carriers:
- **L3 carry-forward attention:** existing pattern; adds a head that copies
  STDIN_CURSOR_* from the previous step's AX marker into the current step's
  AX marker. Increment unit lives at L4 FFN, gated on `OP_GETCHAR`.
- **A dedicated cursor token in the step layout** — heavier surgery (alters
  STEP_TOKENS) so deferred.

For phase 2 we go with the L3 carry-forward + L4 increment approach.

## 5. Dim allocations needed (registered, not yet wired)

```
MARK_USER_INPUT_START  // 1 dim, set on Token.USER_INPUT_START only
MARK_USER_INPUT_END    // 1 dim, set on Token.USER_INPUT_END only
STDIN_BYTE_LO          // 16 dims, nibble of the gathered stdin byte (low)
STDIN_BYTE_HI          // 16 dims, nibble of the gathered stdin byte (high)
STDIN_CURSOR_LO        // 16 dims, cursor low nibble (offset into USER_INPUT block)
STDIN_CURSOR_HI        // 16 dims, cursor high nibble
USER_INPUT_START_POS_* // optional: 8 dims per nibble for the position of the START marker
```

The phase 1 bake only writes STDIN_BYTE_LO/HI; the rest are reserved (the
op declares them in `writes={}` so the dim registry contracts stay green).

## 6. READ vs GETCHAR

READ is GETCHAR × N plus *writing into shadow memory*. The byte-gather side
is the same head structure; the routing side has to:

- Generate `count` MEM section tokens (`[MEM, addr_b0..3, val_b0..3]`) at
  contiguous addresses starting at `buf_ptr`.
- That's a sequence emission, not a single AX override — beyond V9 phase 1.

Phase 1 keeps `_neural_read_emit` in place; the `test_read_from_stdin` xfail
remains xfail.

## 7. The runner side

`AutoregressiveVMRunner.run` already builds the right context: see
`run_vm.py:1156-1160`. No change needed to the prefix builder.

The runner-side `_inject_getchar` and `_neural_read_emit` stay in place
until the matching neural bakes are validated. They are gated by
`exec_op == Opcode.GETCHAR` / `Opcode.READ` so they only fire on those
specific opcodes — once the neural routing is verified, the calls can be
dropped (or kept under a flag for the EOF fallback).

`_stdin_buffer` itself remains in the runner for the **tool-calling mode**
where the host owns the buffer (the spec's "tool calling mode"). This is
explicitly acceptable per the V11 conversational-IO policy.

## 8. The first bake (this commit)

`make_layer5_user_input_gather_op(enable=False)`:

- `kind="block"`, `layer_idx=5`, `phase=5.7` (after `layer5_fetch` at 5,
  before L5 FFN opcode decode if needed — actually phase 5.7 sits between
  L5 fetch and the model-level bakes, mirroring the convo-IO ops).
- Body (when `enable=True`): allocates one L5 attention head pair (Head A
  + Head B per §3.2) with cursor=0, gather pos = `USER_INPUT_START + 1`.
  Writes `STDIN_BYTE_LO/HI`.
- Body (when `enable=False`): no-op. Registered to keep the dep graph
  topology stable across the flag.

`make_layer6_getchar_routing_op(enable=False)`:

- `kind="model"`, `phase=998.9` (right after `io_putchar_routing` at 998,
  before `legacy_bake` at 999).
- Body (when `enable=True`): adds an L6 FFN unit pair starting at unit 1600
  (well past the PUTCHAR routing at 1500-1531) that routes
  `STDIN_BYTE_LO/HI → AX_CARRY_LO/HI` gated on `OP_GETCHAR ∧ MARK_AX`.
- Body (when `enable=False`): no-op.

When both are flipped to `enable=True`, the runner's `_inject_getchar` call
becomes redundant for stdin reads inside the USER_INPUT block (EOF still
falls through to the Python shim). Verifying byte-identity for
`test_getchar_returns_byte` will be the gate for that flip.

## 9. Verification plan

- `test_pure_neural_io.py::TestPureNeuralGetchar::test_getchar_returns_byte`:
  with bake enabled, AX byte 0 must be `'x'` (0x78) without the Python
  override running. Today it is `'x'` via `_inject_getchar`.
- `test_pure_neural_io.py::TestPureNeuralReadStdin::test_read_from_stdin`:
  xfail today; remains xfail after V9 phase 1 (READ neural routing not
  baked). Phase 3 of this plan flips it.
- Smoke gate (`test_smoke.py::TestSmokeBasic::test_imm_exit`,
  `test_runtime_vanilla.py`, etc.) must stay green with the bakes
  registered-but-disabled: i.e. zero behavioural drift from
  `enable=False` no-op bodies.
