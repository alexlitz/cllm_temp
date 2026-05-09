# Pure-Neural Mode: What Python is Allowed To Do

This is the canonical policy for `AutoregressiveVMRunner(pure_neural=True)`. It defines what counts as "100% autoregressive / pure neural" for this codebase.

## Allowed Python-side operations

These are I/O boundary concerns — they do not compute or change any value the model is supposed to produce. Anything in this list is fair game.

1. **Result extraction.** Reading the final register value from the model's output context to return it to the caller. The model emits the result tokens; Python decodes them.
   - `_extract_register(context, Token.REG_AX)` after the loop terminates.
   - `_decode_exit_code(context)` to convert tokens to an integer.

2. **Detecting when to give the user / external system input.** Identifying that the model has emitted a TOOL_CALL token, a request-input marker, or a step that needs user-supplied stdin. The detection is fine; the runner must NOT compute any value that goes back into the model — only relay user input.
   - Detecting `Token.TOOL_CALL` to surface a tool request to the host.
   - Routing user-supplied bytes from stdin into the context as `byte` tokens.

3. **Showing the user non-thinking tokens.** Filtering the model's emitted tokens so that thinking-mode tokens stay internal and only user-facing output bytes are surfaced.
   - Skipping tokens between `THINKING_START` and `THINKING_END`.
   - Decoding output bytes (e.g., from PRTF) and printing them.

4. **Loop termination on EXIT.** Reading the bytecode to know when an EXIT instruction has executed so the runner can stop the generation loop. This does not change any tensor value the model produces — it just decides when to stop calling `generate_next`.

## Forbidden Python-side operations

Anything that **changes or forces** a value the model is supposed to compute is forbidden in pure-neural mode. The list below is exhaustive — if it isn't here, it's likely forbidden.

- ❌ Overriding any register byte in context (`_override_register_in_last_step` for REG_PC / REG_AX / REG_SP / REG_BP / STACK0).
- ❌ Computing ALU results in Python (`_compute_alu`).
- ❌ Injecting MEM section values (`_inject_mem_section`, `_mem_store_word`).
- ❌ Tracking memory writes in a Python `_memory` dict and feeding them back to the model.
- ❌ Forcing STEP_END at the end of every 35-token window.
- ❌ Calling `set_active_opcode` to inject opcode hints into the embedding or swap MoE weights.
- ❌ Overriding REG_AX on HALT.
- ❌ Running syscall handlers (PRTF, READ, OPEN, CLOS, GETCHAR, PUTCHAR, MALC, FREE, MSET, MCMP).
- ❌ Running func-call handlers (IMM, JMP, JSR, ENT, PSH, ADD, SUB, MUL, DIV, MOD, OR, XOR, AND, SHL, SHR, EQ–GE).

If the model emits the wrong byte in a register slot, the runner just records that and moves on. The result will be wrong; that's the honest signal.

## Architectural constraints

Beyond the runtime policy, two structural rules apply to the model itself:

1. **Compiler determines network depth and width.** No hardcoded `ffn_hidden=4096` independent of what's actually programmed. Each layer's FFN width = the number of programmed units. Dead/uninitialized units must not exist (they fire on residual noise and break correctness).

2. **Standard FFNs only.** Every block has exactly **one** FFN module, of the form `output = x + W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down`. This is `PureFFN` (`base_layers.py:57`). All wrapper / composite FFNs (HybridALUBlock, PureNeuralALU, EfficientDivMod_Neural, post_ops with custom forward) must be **compiled down to this same structure**. The compiler is allowed to allocate as many blocks as needed — multi-stage operations become multiple blocks, each with its own single PureFFN — but no block can contain a multi-FFN wrapper or non-FFN logic. Operations that don't compile to a SwiGLU FFN (e.g., `argmax`, `softmax`, hand-rolled step functions) must be replaced with smooth FFN equivalents or moved into attention.

3. **Vanilla attention only.** Standard multi-head attention, no custom attention variants beyond what a vanilla transformer paper would describe.

## How to verify pure-neural mode is real

```bash
python tests/run_1000_tests.py --limit 5
```

This script is wired to `AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)` with empty handler dicts. The reported pass rate is the honest pure-neural pass rate.

```bash
python -m pytest tests/test_pure_neural_pc.py -v
```

The Phase 1 gate. Currently `test_imm_then_exit` passes; `test_two_imms` and beyond fail. Phase 1 closes when all tests in this file pass.

## Reading the rest of the docs

- `PURE_NEURAL_GAP_ANALYSIS.md` — what's actually broken right now, why, and the multi-week phased plan.
- `NEURAL_VM_STATUS.md` — older status doc; predates this policy and is partly out of date.
- `TESTING_CHECKLIST.md` — the project's headline checklist; should be re-read against this policy.
