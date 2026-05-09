# Pure-Neural Gap Analysis & Plan

**Date:** 2026-05-09
**Status:** Honest assessment of where the project actually stands vs the stated checklist.

---

## TL;DR

The project's headline claim — *"1096/1096 tests passing, 100% autoregressive vanilla transformer"* — is not supported by the code as it stands. The 1096-test pass rate is achieved by a Python interpreter (`FastLogicalVM`) that bypasses the transformer entirely. When the actual transformer runs in pure-neural mode (no Python overrides), it passes **0/5** of the simplest compiled-C arithmetic tests and fails on the second instruction of a hand-written `IMM, PSH, IMM, ADD, EXIT` program.

Closing this gap is multi-week-to-multi-month architectural work, not a refactor.

---

## What "1096/1096 passing" actually runs

Trace from `tests/run_1000_tests.py`:

```
BakedC4Transformer(use_speculator=True)
  → SpeculativeVM(transformer_vm=..., validate_ratio=0.0)
    → SpeculativeVM.run() → fast_vm.run()  (always; 0% validation)
      → FastLogicalVM   ← src/speculator.py:35
```

`FastLogicalVM.run()` is a Python `while` loop over `if op == 0: ... elif op == 1: ...`. There is no neural network involvement in any of the 1096 tests as currently configured.

`AutoregressiveVMRunner` is a separate code path that does invoke the transformer. It is the path neural_only / pure_neural pytest fixtures use — not the 1096-test runner.

## What `AutoregressiveVMRunner` does in default mode

`AutoregressiveVMRunner(trust_neural_alu=False)` (the default) runs the transformer but applies extensive Python corrections each step:

- `_compute_alu(op, stack_val, ax_val)` — Python computes ADD/SUB/MUL/DIV/MOD/OR/XOR/AND/SHL/SHR/EQ/NE/LT/GT/LE/GE results and overrides the model's AX prediction (`run_vm.py:476-517`, called at `:667`).
- 4 syscall handlers (CLOS/OPEN/READ/PRTF) that handle I/O on the Python side.
- 18 func-call handlers documented in `NEURAL_VM_STATUS.md` (IMM, JMP, JSR, ENT, PSH, OR, XOR, AND, comparisons, SHL, SHR, ADD, SUB, MUL, DIV, MOD).
- Dozens of `_override_register_in_last_step(context, Token.REG_*, value)` calls that inject correct PC/SP/BP/AX/STACK0 values into the model's context after each step.
- `_inject_mem_section(addr, value)` and `_mem_store_word(addr, value)` calls that bypass the network's MEM-token writing.
- `_track_memory_write(context, op)` that maintains a Python-side `_memory` dict.

`run_vm.py` contains **61 grep hits** for these override/computation primitives.

## What `pure_neural=True` mode does

`AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)` short-circuits `_dispatch_step` (`run_vm.py:531`) so none of the overrides above run. The runner becomes a pure observer — it just reads model outputs and tracks PC/AX/SP/BP for the EXIT result.

This mode is **not exercised by any test in the codebase**. The `pure_neural_runner` fixture (`tests/conftest.py:199`) is defined but no test references it.

## Observed pure-neural behavior

**Hand-written `IMM 5, PSH, IMM 3, ADD, EXIT` (expected: 8). Actual trace:**

```
exec_op=IMM:  model_PC=10  AX=5  SP=65536  STACK0=0
exec_op=PUSH: model_PC=0   AX=5  SP=248    STACK0=5
exec_op=IMM:  model_PC=0   AX=5  SP=248    STACK0=5     ← stuck reading PC=0 forever
... 7 more identical steps ...
final result=0
```

Three independent failures in the first two transitions:

1. **PC arithmetic broken.** Model emits `next_pc=0` instead of `next_pc=16`. Without `_override_register_in_last_step(context, Token.REG_PC, exec_pc + 8)`, PC drifts to 0 immediately and the program loops on the first instruction forever.
2. **SP arithmetic broken.** Model emits `SP=248` after PSH (expected `65528 = 65536-8`). Without `_override_register_in_last_step(context, Token.REG_SP, last_sp - 8)`, the stack pointer becomes garbage.
3. **AX persistence broken.** Once the model loops back to the wrong instruction, AX collapses to 0. The byte-passthrough mechanism in L10 depends on the previous step having a coherent AX value.

**Compiled C `int main() { return 654 + 114; }` (expected: 768). Actual: 5/5 fail, all return 0.** This program compiles to 213 instructions including ENT/LEV stack frame setup, multi-byte values, and function call wrappers — every one of these subsystems depends on overrides.

## Inventory of Python paths that need to become neural

Listed roughly in dependency order — the first items are foundations the later items rely on.

### Tier 1 — Per-step register arithmetic (currently broken)

| Concern | Where Python does it | What network must learn |
|---|---|---|
| `next_pc = pc + 8` for non-branching opcodes | `run_vm.py:580-581` | Embed PC width as a literal, add 8 each step unless opcode is in JMP/BZ/BNZ/JSR/LEV/ENT |
| Branch target PC for JMP/BZ/BNZ | `run_vm.py:631-650` | Read immediate from current instruction, conditionally route to PC based on AX==0 / AX!=0 |
| JSR target & return-address push | `run_vm.py:594-603` | Resolve target index, write `pc+8` to MEM at SP-8, set PC to target |
| ENT frame setup (`SP-=8; mem[SP]=BP; BP=SP; SP=BP+imm*4`) | `run_vm.py:604-616` | Multi-step state machine implemented in attention/FFN |
| LEV frame teardown (read saved BP/return-addr from `mem[BP]`/`mem[BP+8]`, restore) | `run_vm.py:618-627` | Memory load via attention to MEM positions, register restore |
| ADJ (`SP += imm`) | `run_vm.py:651-657` | Read signed imm from instruction, add to SP |

### Tier 2 — Memory operations (currently broken)

| Concern | Where Python does it | What network must learn |
|---|---|---|
| PSH writes AX to `mem[SP-8]` | `run_vm.py:587-592` | MEM token sequence at PSH step must encode `addr=SP-8, value=AX` correctly |
| POP reads `mem[SP]` for binary ops | `run_vm.py:659-661` | Attention head from AX position to most-recent MEM-write at SP needs to retrieve full multi-byte value |
| SI writes AX (4 bytes) to `mem[STACK0]` | `run_vm.py:680-685` | MEM token sequence with addr from STACK0 and value from AX |
| LI loads 4-byte word from `mem[AX]` into AX | `run_vm.py:670-674` | Memory attention with addr=AX |
| LC loads 1-byte from `mem[AX]` (Python-side `_memory` dict) | `run_vm.py:675-679` | Same as LI but byte-granular |
| SC writes byte to `mem[STACK0]` | `run_vm.py:686-690` | Same as SI but byte-granular |

The `_memory` dict (`run_vm.py:213`) is currently the source of truth for stored values. The network's KV cache should be the source of truth instead.

### Tier 3 — ALU multi-byte completeness

(Earlier session notes — these are partly working.)

- ADD/SUB single-byte work in `trust_neural_alu=True`. Multi-byte carry/borrow is partially fixed (CarryPropagationPostOp).
- Multi-byte SUB borrow (256−1=255) currently fails.
- Cross-byte SHL/SHR (shift ≥ 8) currently fails — fundamental: `OUTPUT_LO`/`OUTPUT_HI` only encode byte 0; bytes 1-3 of the result need to be routed to AX byte 1/2/3 sequence positions.
- MUL multi-byte: not yet verified end-to-end.
- DIV/MOD: handlers active in default mode; pure-neural state unknown.

### Tier 4 — I/O & syscall (PRTF / READ / OPEN / CLOS)

- Currently 4 syscall handlers in `_syscall_handlers`. PRTF reads format string from Python `_memory` dict (`run_vm.py:362`).
- `conversational_io=True` mode partially routes I/O through THINKING_END/THINKING_START tokens but still falls back to `_memory` lookup.
- Pure-neural I/O requires the model to (a) detect PRTF/READ opcode, (b) emit format-string bytes from MEM autoregressively, (c) route OUTPUT_BYTE tokens correctly.

### Tier 5 — Heap / MALC / FREE / MSET / MCMP

- Bump allocator state (`_heap_base`, `_heap_ptr`, `_alloc_sizes`) lives in Python.
- These are listed in `NEURAL_VM_STATUS.md` as needing handlers; pure-neural status unknown.

---

## Phased plan

Each phase ends with a test target: a specific class of program that must run end-to-end through the pure-neural runner. Phases are sequential — later phases assume earlier ones land.

### Phase 1 (≈ 2-3 weeks): PC + AX coherence

**Goal:** `IMM N, EXIT` and `IMM N, IMM M, ... EXIT` work for any sequence of single-byte IMMs.

- Add neural circuits that compute `next_pc = current_pc + 8` for non-branching opcodes (most of L4-L5 already does PC fetch — extend to PC update).
- Verify AX persists correctly across non-AX-modifying opcodes (currently relies on L10 passthrough; verify it works without overrides).
- Add a pytest target `tests/test_pure_neural_pc.py` that exercises this exit gate.

### Phase 2 (≈ 2-3 weeks): SP arithmetic + PSH MEM write

**Goal:** `IMM A, PSH, IMM B, ADD, EXIT` works in pure-neural mode for single-byte A, B.

- Neural circuit for `next_sp = sp - 8` on PSH and `sp + 8` on POP-style opcodes.
- PSH must produce a correct MEM token sequence: `MARK_MEM, ADDR_LO, ADDR_HI, ..., VAL_B0, VAL_B1, VAL_B2, VAL_B3` with `addr = SP-8` and `val = AX`.
- POP attention head from binary opcodes (ADD/SUB/etc.) must retrieve the most recent MEM-write at SP, not zero.
- Multi-byte AX is allowed to be broken in this phase — only byte 0 needs to roundtrip.

### Phase 3 (≈ 2-4 weeks): Multi-byte AX + ALU completeness

**Goal:** `IMM 200, PSH, IMM 100, ADD, EXIT` returns 300 in pure-neural mode.

- Wire byte 1/2/3 of OUTPUT through sequence-position routing (currently only byte 0 has `OUTPUT_LO`/`OUTPUT_HI` dims).
- Fix multi-byte SUB borrow (already in flight).
- Fix cross-byte SHL/SHR.
- Verify MUL multi-byte.
- Likely requires extending the BD format with `OUTPUT_BYTE_1/2/3` dims or using sequence positions.

### Phase 4 (≈ 2-3 weeks): Control flow

**Goal:** `if`/`while` programs run pure-neural. JMP, BZ, BNZ targets resolved by network.

- L5 fetch already reads instructions; extend to read branch immediate and route to PC conditionally on AX zero/non-zero.
- The L5 hard-coded address 3 issue noted in `NEURAL_VM_STATUS.md` ("Priority: MEDIUM") becomes blocking here.
- Test target: simple `while (n > 0) { n = n - 1; } return n;` programs.

### Phase 5 (≈ 3-4 weeks): Function calls (JSR/ENT/LEV)

**Goal:** A compiled `int main() { return f(N); }` style program runs pure-neural.

- JSR neural circuit: push `pc+8` to mem[SP-8], set PC to target.
- ENT: `sp-=8; mem[sp]=bp; bp=sp; sp=bp+imm*4`. Multi-cycle state, possibly across multiple step tokens.
- LEV: load saved BP from `mem[bp]`, return-addr from `mem[bp+8]`, restore.
- This is the hardest phase — unlocks most of the 1096-test suite.

### Phase 6 (≈ 2-3 weeks): I/O syscalls

**Goal:** `printf` and `read` work without `_syscall_*` handlers.

- Conversational I/O routing already partially exists (`conversational_io=True`). Make it work without `_memory` fallback.
- PRTF: emit format-string bytes from MEM autoregressively, support `%d`/`%c`/etc. format specifiers via neural decoding.
- READ: extract bytes from USER_INPUT context section into MEM/AX.

### Phase 7 (≈ 2-3 weeks): Heap (MALC/FREE/MSET/MCMP) and DIV/MOD parity

**Goal:** Programs using malloc/free run pure-neural. DIV/MOD multi-byte verified.

### Phase 8 (≈ 1-2 weeks): Switch headline test runner

**Goal:** `tests/run_1000_tests.py` runs through `AutoregressiveVMRunner(pure_neural=True)` and reports honest numbers. Probably nowhere near 1096/1096 even after all the above; the residual delta becomes the next backlog.

---

## Total estimate

**16-25 weeks of focused work** across all phases, depending on how many subtle bugs surface during implementation. Phase 5 (function calls) and Phase 3 (multi-byte AX routing) are the highest-risk; both involve nontrivial sequence-position work in the transformer's attention patterns.

This estimate assumes a single engineer working full-time and that the existing layer architecture is mostly preserved. A from-scratch redesign of the layer plan could be faster or slower depending on approach.

---

## Phase 1 investigation findings (2026-05-09)

Established gate test: `tests/test_pure_neural_pc.py` — passes `IMM, EXIT`, fails `IMM, IMM, EXIT`. The minimum failing program: `[(IMM, 5), (IMM, 7), EXIT]` should return 7, returns 0.

Per-layer residual inspection at step 2's MARK_PC position (the first failing emission):

```
L3:  EMBED_LO[10]=+1.00  OUTPUT_LO[2]=+1.00 OUTPUT_HI[1]=+1.00   ← L3 correctly computes PC=18
L4:  EMBED_LO[10]=+1.00  OUTPUT_LO[2]=+1.00 OUTPUT_HI[1]=+1.00
L5:  EMBED_LO[10]=+1.00  OUTPUT_LO[2]=+1.00 OUTPUT_HI[1]=+1.00
L6:  OUTPUT_LO[2]=-3.00 OUTPUT_HI[1]=-3.00 OUTPUT_HI[0]=+164.01  ← L6 destroys it
```

L6 FFN delta at OUTPUT_HI[0]: **+164.01**. Two units are responsible:

| Unit | Source code section | Fires because | Contribution |
|---|---|---|---|
| 176 | JMP PC override target HI (line 4653) | `CMP[0] ≈ 5` (IS_JMP) at MARK_PC | +41 |
| 304 | First-step JMP target HI (line 4702) | `OP_JMP ≈ 5` at MARK_PC | +123 |

Both units should only fire on JMP opcodes, but `OP_JMP` and `CMP[0]` are spuriously active at MARK_PC of step 2 even though our bytecode is `[IMM 5, IMM 7, EXIT]` — no JMP anywhere.

**Open questions for Phase 1 fix:**
- Why is `OP_JMP=5` set at MARK_PC of step 2 when active_opcode=IMM and bytecode at PC=10 is `IMM 7`?
- Why is `CMP[0]=5` (IS_JMP) set when L6 attn head 0 should only set it for actual JMP opcodes?
- The L5 fetch reads the opcode from the bytecode via attention to PC. Either L5 misreads, or some upstream layer leaks a stale flag from step 1 into step 2's MARK_PC.

The same kind of investigation will recur for SP arithmetic (step 2 SP byte 1 also wrong) and BP carry-forward in Phase 2/3. Each Phase 1-3 layer fix likely requires this depth of per-layer residual analysis.

**Tools added for future phases:**
- `tests/test_pure_neural_pc.py` — the gate test for Phase 1
- `tests/run_1000_tests.py` rewired to `AutoregressiveVMRunner(pure_neural=True)` so the headline number reflects honest pure-neural state once Phase 1+ lands.

**Patch attempts (2026-05-09):**
- Added `OPCODE_BLOCK_MAP` defensive gate to L0–L16 FFNs that suppresses any unit with `MARK_PC > 50` and a JMP/LEV/BZ/BNZ/CMP[0] trigger weight by injecting `-S` blockers for non-target opcodes (`vm_step.py:2192+`).
- Removed `set_active_opcode` calls from `run_vm.py` when `pure_neural=True`. The runner no longer peeks at the bytecode to inject the active opcode hint into the embedding or to swap MoE expert weights.
- After both fixes: `test_two_imms` STILL fails. Step 2 still emits PC=0 instead of PC=18.

**Why the patches aren't enough:**

The architecture has too many tangled layers (L0–L16) all writing to OUTPUT at MARK_PC with various weakly-gated conditions. The L3 increment IS correct in the residual stream up through L15, but L16 then writes its own OUTPUT (LEV-related units 80, 96 at `vm_step.py blocks[16].ffn`) that cancels the correct value. Adding more defensive gates is whack-a-mole. There are likely additional layers that also need defensive gating, and even after that, the underlying issue is that the architecture was designed assuming runner overrides would inject correct values each step.

**Architectural observations from this work:**

1. **Hardcoded sizing.** `d_model=512`, `n_layers=17`, `n_heads=8`, `ffn_hidden=4096` are baked into `run_vm.py:140-145`. The compiler doesn't size the network to what's actually needed by the program. Layers' FFNs use only a fraction of their 4096 hidden units; the rest are zero-initialized but can spuriously fire on residual noise. This is what motivates the existing "BUG FIX 2026-04-09" patch in `vm_step.py:2150+` and the new patches added today.

2. **No opcode-isolation in the layer plan.** Many L6/L16 units fire on `MARK_PC + opcode_flag` combinations without exhaustive blockers for the other 32 opcodes. When the runner used to override register values, the spurious unit firing didn't matter — overrides won. Without overrides, every spurious activation breaks something.

3. **DEPRECATED-but-still-wired heads.** L5 attention head 6 is documented as "DEPRECATED: OP_* flags were removed from embeddings (2026-04-13)" but is still wired up. It leaks opcode flags into MARK_PC. Similar dead-but-wired code likely exists elsewhere.

**Phase 1 + Phase 2 results: 26/29 gate tests pass.**

After strengthening ComparisonCombine's MARK_PC blocker from -10*S to -50*S, simple PSH+ADD/SUB/AND/OR tests pass pure-neural. The blocker fights leaked CMP[0..3] residuals (which can reach ~15 at MARK_PC due to cumulative writes by L6/L7/L8 attention heads in pure-neural mode without runner overrides).

Phase 2 (`tests/test_pure_neural_psh_add.py`): 14/16 passing
- `test_imm_psh_exit` ✅
- All `test_add_small` except `[0-5-5]` (ADD 0+5 hits the multi-byte broadcast bug)
- All `test_sub_small` ✅
- All `test_and_small` ✅
- `test_or_small[170-85-255]`, `[0-0-0]` ✅; `[240-15-255]` ❌

**Phase 1 results (this session): 12/13 gate tests pass.**

The remaining failure (`test_five_imms`) is a multi-byte AX broadcast bug. Bytecode `IMM 1, IMM 2, IMM 3, IMM 4, IMM 5, EXIT` returns `0x04040404` after 4 IMMs (byte 0 broadcast across all 4 bytes). Investigation findings:

1. **Step-count threshold.** First 3 IMMs work; 4th IMM triggers the broadcast.
2. **Position-dependent.** `IMM 5` followed by 3 NOPs returns 5 correctly. 3 NOPs followed by `IMM 5` returns garbage. The IMM-at-step-≥4 specifically is broken.
3. **Bug locus.** L8 multi-byte fetch (head 3, `vm_step.py:5857`) reads code bytes at PC+1, PC+2, PC+3, PC+4 for AX byte 0, 1, 2, 3 respectively. The fetched values land in `AX_CARRY_LO/HI` and L8 multi-byte routing copies them to OUTPUT. At step ≥ 4, AX byte 1-3 fetches return the byte 0 value (the lo byte of the immediate) instead of 0x00.
4. **Verified.** L8 multi-byte routing for byte 0 fires correctly (OUTPUT_LO[4]=8 for IMM 4), but the same routing at byte 1-3 reads AX_CARRY containing 4 instead of 0.

**Hypotheses for the L8 head 3 misread at step ≥ 4:**
- ALiBi penalty growing with context length might cause attention to spread across multiple code positions, summing the V values into AX_CARRY.
- L4 FFN's PC+K computation (nibble rotation) might break for PC values that have hi nibble = 1 (PC ≥ 16). Steps 1-3 have PC = 2, 10, 18 (mixed). Step 4 has PC = 26.
- Anti-leakage gate on L8 head 3 might fail at certain PC values.

This is Phase 3 work — fixing it likely requires rewriting the multi-byte fetch as a proper PureFFN-friendly mechanism rather than the current ALiBi-based attention. It may also be solved by Phase 0's wrapper-elimination since the multi-byte fetch is part of the broader "compile to single PureFFN per block" agenda.

**Realistic next steps for Phase 1 completion:**

The whack-a-mole approach scales poorly. Two cleaner options:

a) **Per-layer audit + opcode-isolated rewrite.** For each layer that writes to OUTPUT_LO/HI at MARK_PC, list all units, classify by intended opcode, and add comprehensive blockers. Probably 1-2 weeks of careful work.

b) **Right-size the network in the compiler.** Replace the hardcoded `d_model=512`, `ffn_hidden=4096` with values computed from the actual circuits being baked. With no zero-initialized units, no spurious firings. Bigger refactor (3-4 weeks) but eliminates the entire class of bug.

Option (b) also addresses the user's question about whether the compiler determines network size — it currently does not, and doing so would solve a major class of problems.

---

## Recommendations

1. **Update the headline docs** so they distinguish `FastLogicalVM` (1096/1096), `AutoregressiveVMRunner` default (most opcodes work via Python overrides), and `AutoregressiveVMRunner(pure_neural=True)` (currently 0/5 on compiled C). The current docs conflate these.

2. **Define what "pure neural" means** for the project. Is it (a) "the transformer's forward pass is the *only* thing computing semantics," or (b) "the runner is allowed to do bookkeeping but no semantic computation"? Different definitions lead to different scope.

3. **Pick a realistic interim target.** "Pure-neural Phase 2 (PSH+ADD works)" is a far more credible 1-month milestone than "all 1096 tests pass pure-neural in 6 months."

4. **Add CI coverage for pure-neural mode now**, even with mostly-failing tests. Otherwise progress is invisible and regressions are silent.

---

## Appendix — files referenced

- `tests/run_1000_tests.py` — 1096-test runner; was wired to `BakedC4Transformer`/`FastLogicalVM`, now switched to `AutoregressiveVMRunner(pure_neural=True)`.
- `src/speculator.py:35` — `FastLogicalVM` (Python interpreter).
- `src/speculator.py:194` — `SpeculativeVM` (defaults to `validate_ratio=0.0`).
- `src/archive/baked_c4.py` — `BakedC4Transformer`.
- `neural_vm/run_vm.py:476-517` — `_compute_alu`.
- `neural_vm/run_vm.py:531-549` — `pure_neural` short-circuit.
- `neural_vm/run_vm.py:587-690` — per-opcode override block.
- `neural_vm/efficient_alu_neural.py` — efficient ALU (the work from prior sessions).
- `tests/conftest.py:188-210` — `pure_neural_runner` fixture (currently unused by tests).
