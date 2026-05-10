# Agent context — read this before working on the Neural VM

If you're a sub-agent spawned to work on this Neural VM codebase, read this file before doing anything else. It captures recurring mistakes other agents have made.

## Repo layout

- Working directory: `/home/alexlitz/Documents/misc/c4_release/c4_release`
- The inner package is `neural_vm/` and tests live at `tests/`.
- Production main branch: `migrate-l0-ops` (HEAD currently). `origin/main` is behind.

## CRITICAL: untracked support modules

`c4_release/neural_vm/unified_compiler/` contains several modules that are UNTRACKED in git but PRESENT in the working tree: `builder.py`, `ir.py`, `opcodes.py`, `allocator.py`, `auto_allocator.py`, `codegen.py`, `dim_bridge.py`, `primitives_logical.py`.

The `__init__.py` imports from them. If your worktree base doesn't have these, **`from neural_vm.run_vm import AutoregressiveVMRunner` fails with `ModuleNotFoundError`**.

**Two options:**
1. Copy them in: `cp /home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/{builder,ir,opcodes,allocator,auto_allocator,codegen,dim_bridge,primitives_logical}.py <your_worktree>/c4_release/neural_vm/unified_compiler/`. DON'T commit them.
2. Wrap the imports in `try/except ImportError` in `__init__.py`. (This fix is being applied to main.)

## Pure-neural vs default mode

Two execution modes exist:

```python
runner = AutoregressiveVMRunner()              # pure_neural=False (default)
runner = AutoregressiveVMRunner(pure_neural=True)  # pure_neural=True
```

| | `pure_neural=False` | `pure_neural=True` |
|---|---|---|
| `set_active_opcode` Python hint | runs (sets OP_* dim from bytecode) | SKIPPED — opcodes set autoregressively via L5 |
| `_dispatch_step` Python overrides | runs (force PC, write registers, etc.) | SKIPPED |
| Syscall handlers (PRTF/READ) | run | partial — only TOOL_CALL boundary |
| Memory injection from `_inject_mem_section` | runs | runs (memory needs Python injection still) |

**Smoke tests in default mode tell you NOTHING about pure_neural correctness.** Always specify which mode you're testing.

## Bytecode encoding

```python
INSTR_WIDTH = 8        # bytes per instruction (c4_release/neural_vm/constants.py)
# Each 32-bit word: (imm << 8) | opcode
```

**Branch opcodes (JMP/BZ/BNZ/JSR) are special**: the runtime does `PC = imm` directly. So the `imm` must already be a PC value, not an instruction index.

```python
# C4 compiler convention: imm = idx * INSTR_WIDTH + PC_OFFSET (PC_OFFSET = 2)
imm_for_jmp_to_idx_2 = 2 * 8 + 2  # = 18
```

**Several test files use raw idx** instead. Those tests will fail until either:
- The test fixture (`_make_bc`) converts idx → PC, or
- The neural code multiplies the imm by INSTR_WIDTH

The neural code does the former. Tests that use raw idx are incorrectly encoded.

## ALU mode dispatch

```python
runner = AutoregressiveVMRunner(trust_neural_alu=True)  # → alu_mode='efficient'
runner = AutoregressiveVMRunner()                       # → alu_mode='lookup'
```

In `efficient` mode (which is what `pure_neural=True` uses):

| Block | `_set_layerN_*` | Replaced by |
|---|---|---|
| L8 | `_set_layer8_alu` (lookup) | `ALUAddSub` (`efficient_alu_neural.py:481`) |
| L9 | `_set_layer9_alu` | `ALUAddSub` |
| L10 | `_set_layer10_alu` | `ALUAndOrXor` (`efficient_alu_neural.py:487`) |
| L11/L12 | `_set_layer11/12_*` | `ALUMul` (`efficient_alu_neural.py:493`) |
| L13 | `_set_layer13_shifts` | `ALUShift` (`efficient_alu_neural.py:499`) |

So `_set_layer10_alu` is NOT on the pure_neural path. **The opcode-handler-map I gave earlier in this session was wrong about this.** Verify before assuming.

## Opcode values

Authoritative source: `c4_release/neural_vm/embedding.py`. Search for `class Opcode`. Don't trust prompt values; check the file.

## Compiler / migration architecture

- `LayerCompiler` in `unified_compiler/layer_compiler.py` manages the bake.
- Each migrated `_set_layerN_*` function has an `Operation` factory in `unified_compiler/migrated_ops.py`.
- `Operation` has `migrated: bool` (default False). When True, `build_model_from_layout` dispatches its `bake_fn` even when `legacy_bake` is present.
- `legacy_bake` is a model-level Operation that wraps `set_vm_weights`. As ops migrate, set_vm_weights' inline calls are removed and the corresponding `Operation.migrated` flag is set to True.
- Block-scoped ops (`kind="block"`, `layer_idx=N`) target a specific layer.
- Model-scoped ops (`kind="model"`) operate on the whole model.

## Test files (per phase)

| Phase | Test file | Branch (if not on main) | Status |
|---|---|---|---|
| 1 | `tests/test_pure_neural_pc.py` | main | 13/13 ✓ |
| 2 | `tests/test_pure_neural_psh_add.py` | main | 10/16 |
| 2-ext | `tests/test_pure_neural_phase2_ext.py` | branch (not on main) | new |
| 3 | `tests/test_pure_neural_multibyte.py` | `phase3-multibyte-pure-neural-tests` | 9 xfail |
| 4 | `tests/test_pure_neural_jmp_bz.py` | `worktree-agent-af8a628b11c3ef028` | 12 xfail (likely test-encoding bug) |
| 5 | `tests/test_pure_neural_jsr_ent_lev.py` | `phase5-pure-neural-jsr-ent-lev` | 7 xfail |
| 6 | `tests/test_pure_neural_io.py` | `phase6-io-pure-neural-tests` | 7 xfail |
| 7 | `tests/test_pure_neural_heap_div.py` | `worktree-agent-a6893efcef9837dd3` | 25 xfail (8 actually pass — DIV/MOD already work) |

To get a phase test on your branch:
```bash
git show <branch>:c4_release/tests/test_pure_neural_<phase>.py > c4_release/tests/test_pure_neural_<phase>.py
```

## Pure-neural fixture

```python
# c4_release/tests/conftest.py defines:
@pytest.fixture(scope='session')
def pure_neural_runner():
    return AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
```

Session-scoped — model build is cached across tests. ~6.8s cold, ~0s warm.

## Known good / known bad

**Pure-neural confirmed working:**
- IMM 5 → 5
- IMM, IMM, IMM (1-5 sequential) — Phase 1 indentation fix landed
- ADD/SUB with non-zero operands
- DIV (4 cases), MOD (4 cases)
- MUL and SHR likely also work (under-tested)

**Pure-neural confirmed NOT working:**
- 6+ sequential IMMs (root in different layer than `_set_layer4_ffn` carry-aware)
- Bitwise AND/OR/XOR (root deeper than threshold; bitwise threshold theory disproved)
- Zero-edge: ADD(0, 5) — bug is in PSH(0) STACK0 byte 0 emission
- LEV semantics — OP_LEV relay disabled at L6 head 6, so all L9/L15/L16 LEV gates never fire
- JSR semantics — multiple bugs: L6 SP-decrement missing, L14 val routes wrong source, STEP_END not emitted
- JMP/BZ/BNZ via raw idx (test encoding bug, not neural bug)

## Constants and dim values

- `S = 100.0` (SwiGLU scale; in `vm_step.py:1748`). **NOT 10.** Many docstrings use S=10 as illustrative.
- `OP_LEV` activation value: ≈ 5 at AX marker (from L5 FFN write). Was ≈ 10 elsewhere (via L6 relay) until that relay was disabled. Many comments still reference 10.
- `MARK_PC`, `MARK_AX`, etc.: 1.0 at the marker token, 0 elsewhere.
- `H1[REG_I]`: 1.0 at the marker token AND ITS BYTE POSITIONS (i.e., AX marker + AX bytes 1, 2, 3 all have H1[AX_I]=1).

## CPU / GPU contention

If your test runs are 10x slower than expected (~5min for a 30s test), CPU contention from sibling agents is the most likely cause. Mitigations:

- Run targeted tests, not whole suites
- Use `--timeout=120` instead of `60` to tolerate slowness
- Run smoke tests instead of pytest where possible (less startup overhead)

False-negatives are common: a test that times out under contention may pass under normal load.

**This machine has 2 GPUs and many parallel agents. Pick the less-loaded GPU dynamically — don't hardcode.** Before any heavy run, query both and pick the one with lower utilization:

```bash
PICK_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$PICK_GPU timeout 120 python -m pytest ...
```

Or as a one-liner inside a Python invocation:
```bash
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1) python ...
```

Re-pick before each new run (a single agent's runs may span minutes; load shifts). For very short probes, GPU 0 is usually fine since nothing pins to it; use the dynamic selector for any pytest run or anything > 30s.

If both GPUs are saturated (>90% util on both), wait 30s and re-check rather than starting a slow run.

## Common mistakes to avoid

1. Testing in default mode and concluding pure_neural works.
2. Assuming `_set_layer10_alu` runs in pure_neural (it doesn't — replaced by `ALUAndOrXor`).
3. Trusting threshold math with S=10 (it's S=100).
4. Assuming `set_active_opcode` propagates OP_* in pure_neural (it doesn't — opcodes must come autoregressively from L5).
5. Reporting test_X "fails" when the test file isn't on your branch (it's missing, not failing).
6. Using raw idx for JMP/BZ/BNZ/JSR imm (must be `idx * INSTR_WIDTH + PC_OFFSET`).
7. Treating CPU-contention timeouts as test failures.
8. Assuming the OP_LEV relay is active (it's been disabled since the `_inject_active_opcode` fix).
9. Ignoring that worker branches may have stale code — always check `git log -1` before trusting your worktree.

## Verify-then-act checklist

Before reporting "fix didn't work" or "function is broken":

1. ✅ Confirm test file exists on your branch (`git ls-files | grep test_pure_neural_X`)
2. ✅ Confirm pure_neural fixture exists in `conftest.py` (search for `pure_neural_runner`)
3. ✅ Run smoke test in default mode AND pure_neural mode separately
4. ✅ Check whether the function you're examining actually runs in your test mode (lookup vs efficient)
5. ✅ Check S value (search `S = ` in vm_step.py — it's 100, not 10)
6. ✅ Check opcode values (read `c4_release/neural_vm/embedding.py`)
7. ✅ Check CPU load (`top` or `nvidia-smi`) before declaring a timeout = real failure

## Patterns discovered (2026-05 sessions)

### Block-op pinning is required during migration

When the compiler-assigned layer differs from the legacy `model.blocks[N]` layer, migrated `_set_layerN_*` ops MUST use `kind="block"` with an explicit `layer_idx=N` until the corresponding `set_vm_weights` references are removed. If you forget to pin, the bake silently writes weights to the wrong block — no error, just incorrect output.

Rule of thumb: if there is still a legacy reference to the old block index in `set_vm_weights`, the migrated `Operation` factory must declare `kind="block", layer_idx=N`.

See ADR-001 (`docs/adrs/ADR-001-target-op-name.md`) for the binding rationale.

### Untracked module imports use try/except guards

`c4_release/neural_vm/unified_compiler/__init__.py` guards imports for several files that aren't tracked in git (`builder.py`, `ir.py`, `opcodes.py`, `allocator.py`, `auto_allocator.py`, `codegen.py`, `dim_bridge.py`, `primitives_logical.py`). The `try/except ImportError` pattern lets the package import even when those files are absent in a worktree.

If you add a new untracked helper module, follow the same pattern: wrap the import and don't rely on it at top level.

### Pure-neural fixture state matters

Many agents have wasted hours because their worktree branch lacked either:
- the `pure_neural_runner` session-scoped fixture in `tests/conftest.py`, or
- the `pure_neural=True` constructor option on `AutoregressiveVMRunner`.

Before running any phase test, confirm both exist on your branch. If not, fetch the fixture from main or copy the relevant files over.

### Branch commit verification before exit

Multiple agents have reported "done" when no commit ever landed. Required exit ritual:

```bash
git checkout -b <named-branch>      # explicit name, not detached HEAD
# ...changes...
git commit -am "<msg>"
git log --oneline -3                # confirm your commit is HEAD
git branch                          # confirm the branch is checked out and starred
```

Both `git log` and `git branch` must be inspected. A commit that doesn't appear in `git log --oneline` did not happen.

### Compiler primitives layer

`c4_release/neural_vm/unified_compiler/primitives.py` exposes a `Primitives` class with declarative weight-setting helpers. Use these instead of raw `ffn.W_up[unit, dim] = value` patches:

- `register_decrement_unit`
- `marker_write_unit`
- `opcode_gated_pc_override`
- `byte_passthrough_chain`
- `threshold_attention_head`
- `carry_forward_attention_head`
- `nibble_rotation_chain`
- `memory_addr_head`

Raw weight patches make migration brittle and resist re-baking. New ops should call into `Primitives`. See ADR-003 for rationale.

### Composite ALU pattern (current state)

`ALUMul`, `ALUShift`, `ALUAndOrXor`, and `ALUAddSub` are now compiler-installed composites — no more runtime instantiation in `efficient_alu_neural.py`. They still wrap multiple sub-FFNs internally, however.

True "blocks of attn+FFN at runtime" (the future-pure architecture) requires residual-stream BD↔GE conversion, which is unfinished. Until then, the composite is the right factoring: declarative install-time, multi-FFN bake. See ADR-004.
