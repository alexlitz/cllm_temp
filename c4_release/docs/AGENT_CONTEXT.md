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
- `compile_full_vm` (in `unified_compiler/full_vm_compiler.py`) is the **sole bake authority**. `make_legacy_bake_op` was deleted (2026-05-11); there is no `legacy_bake` bridge op anymore. `set_vm_weights` still exists in `vm_step.py` but is **backward-compat only** for direct external callers — the compiler does not invoke it.
- Each migrated `_set_layerN_*` function has an `Operation` factory in `unified_compiler/ops/lN_ops.py` (per-layer modules). `unified_compiler/migrated_ops.py` is now a 30-line re-export shim — import from `ops/` directly for new code.
- `Operation` has `migrated: bool` (default False). All compiler-installed ops are now `migrated=True` since `legacy_bake` is gone.
- Block-scoped ops (`kind="block"`, `layer_idx=N`) target a specific layer.
- Model-scoped ops (`kind="model"`) operate on the whole model.
- Phase ordering still references the historical 998/999/1001/1002 slots in many op docstrings/phases — these used to bracket `legacy_bake@999`. The numbers persist as ordering anchors even though no bake op runs at 999 anymore.

## Test files (per phase)

| Phase | Test file | Branch (if not on main) | Status |
|---|---|---|---|
| 1 | `tests/test_pure_neural_pc.py` | main | 13/13 ✓ (closed 2026-05-11) |
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
- Phase 1 (PC/AX coherence): 13/13 — all IMM/NOP sequences, including 5+ sequential IMMs (CarryProp clamp + L14 OUTPUT restore landed 2026-05-11)
- ADD/SUB with non-zero operands
- DIV (4 cases), MOD (4 cases)
- MUL and SHR likely also work (under-tested)

**Pure-neural confirmed NOT working:**
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

When the compiler-assigned layer differs from the legacy `model.blocks[N]` layer, migrated `_set_layerN_*` ops MUST use `kind="block"` with an explicit `layer_idx=N`. If you forget to pin, the bake silently writes weights to the wrong block — no error, just incorrect output.

**As of 2026-05-11**, `LayerCompiler._assign_layers` also honors `layer_idx` for `kind="attn"` and `kind="ffn"` ops (commit `b101578` / `293f189`): a pinned attn/ffn op lands at exactly its `layer_idx`, and the dep-graph invariant (every read dim's producer is at a strictly-earlier layer, OR same-layer with lower phase, OR attn-then-ffn at same phase) is enforced as a hard check. Previously dep-graph placement could silently shift migrated attn/ffn ops to the wrong block (see `docs/MODEL_REGRESSION_BISECT.md`). New rule: any migrated attn/ffn op that targets a specific semantic block should declare `layer_idx=N`.

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

`HybridALUBlock` (the lookup-mode wrapper that ran a structural neural ALU on top of a lookup-table FFN) was **deleted 2026-05-11** (commit `45b1f14`). ALU modules that previously needed the wrapper are now attached as `block.post_ops` and split out by `_expand_wrapper_blocks` at runtime. Some `make_*_hybrid_alu_wrap_op` factories and doc comments still reference "HybridALUBlock" — those are historical naming; the actual wrapper class is gone.

True "blocks of attn+FFN at runtime" (the future-pure architecture) requires residual-stream BD↔GE conversion, which is unfinished. Until then, the composite is the right factoring: declarative install-time, multi-FFN bake. See ADR-004.

## Compiler migration patterns (2026-05-10/11)

15+ migration units in the 2026-05-10/11 window moved inline bakes out of `set_vm_weights` into discrete `Operation` instances. The op factories now live in per-layer modules under `c4_release/neural_vm/unified_compiler/ops/lN_ops.py` (split out 2026-05-11, commit `c3d8427`); `migrated_ops.py` is a 30-line re-export shim. As of 2026-05-11, `make_legacy_bake_op` has been deleted (commit `0b77e62`) and the compiler is the sole bake authority. Three patterns crystallized; each was rediscovered by multiple agents who debugged their way to it. Pick the right pattern up front to avoid that pain.

### Pattern 1: In-place placeholder → migrated (CLEANEST)

When an op already exists in `all_core_ops()` as a `kind="attn"` or `kind="ffn"` placeholder (i.e., not yet migrated), convert it in place:

- Change `kind="attn"` → `kind="block", layer_idx=N`
- Change `kind="ffn"` → `kind="block", layer_idx=N`
- Set `migrated=True`
- Update `bake_fn` signature from `(ffn, ...)` or `(attn, ...)` to `(block, ...)`, and access `block.attn` / `block.ffn` inside
- **The dep-graph stays stable** because the op count in `all_core_ops()` is unchanged
- **No `dep_anchor` companion needed.** This is the cleanest pattern — use it whenever there's an existing placeholder.

**Concrete example**: `migrate-l9-lev-attn-bakes` (Unit E today, commit on main) converted `make_layer9_lev_addr_relay_op` and `make_layer9_lev_bp_to_pc_relay_op` from `kind="attn"` placeholders to migrated block ops without any extra companion.

### Pattern 2: New op + `dep_anchor` companion

When you want to ADD a new migrated op (no existing placeholder to convert), and the dep-graph cares about the count of ops at certain kinds:

- Add a new `kind="block", layer_idx=N, migrated=True` op factory
- KEEP a (possibly hypothetical) `kind="attn"` / `kind="ffn"` placeholder as a NO-OP companion — it preserves the dep-graph's longest-chain length so downstream layer assignments don't shift
- Otherwise: removing a `kind="ffn"` node from the dep-graph shrinks the longest chain by 1, shifting all downstream block indices and breaking weight placement

**Concrete examples**:
- Unit A (`migrate-l3-ffn-op`): added `make_layer3_ffn_dep_anchor_op` as a no-op companion to the new block-pinned `make_layer3_ffn_op`.
- Unit G (`migrate-l14-l15-ffn-bakes`): kept the existing `make_nibble_copy_ffn_op` (`kind="ffn"`, not migrated) as a layout placeholder. Removing it shifted Block 20/22 assignments and broke smoke.
- Units D, F (`migrate-l8-attn-bakes`, `migrate-l10-attn-bakes`): same pattern.

### Pattern 3: `kind="model"` for override-ordering

When a bake intentionally OVERRIDES earlier writes from other model ops (e.g., two functions both write the same attention head weights, and the later write is supposed to win), the migrated op must run AFTER the override target. Since `kind="block"` ops dispatch BEFORE `kind="model"` ops in `build_model_from_layout`, use:

- `kind="model"` with a phase in the 998/1001/1002 ordering anchors (these slots used to bracket `legacy_bake@999`, which is now deleted; the numbers persist as ordering anchors).
- Phases 998.X sit before the phase-999 residual-alibi-slopes bake; phases 1002+ sit after embedding/head bakes.

**Concrete examples**:
- Unit B (`migrate-l6-attn-bakes`): L6 `_set_layer6_relay_heads` head-7 writes override `_set_function_call_weights` head-7 writes. Used `kind="model"` phases 998.5 / .6 / .7.
- Unit 6 (`migrate-everything-unit6`, `_set_opcode_relay_head`): used `kind="model", phase=1002` because the bake's `alibi_slopes[6/7]=5.0` writes were clobbered by an earlier inline `attn6.alibi_slopes.fill_(0.0)` previously at phase=999 in `legacy_bake`.

### Flag-gated ops

For ops gated by `enable_conversational_io` or `enable_tool_calling` flags:

- Register UNCONDITIONALLY in `all_core_ops()` to keep the dep graph stable across flag states
- Inside the `bake_fn`, check the flag and return early (no-op) when off
- Thread the flag through `all_core_ops(*, enable_conversational_io=False, enable_tool_calling=False)` keyword args
- Forward from `compile_full_vm` to `all_core_ops` (see `c4_release/neural_vm/unified_compiler/full_vm_compiler.py`)

**Concrete examples**: Units 1, 2, 3, 4, 5 today all use this pattern.

### Worktree-stale + main-checkout coordination

The `isolation: "worktree"` mechanism creates worktrees from a base commit that may be 100+ commits stale. Migration agents should:

- First try `git checkout main` inside the isolated worktree
- If `main` isn't a local ref, fall back to creating a dedicated worktree off main HEAD:
  ```bash
  git worktree add /tmp/<unit>-wt -b <branch-name> main
  ```
- Avoid working directly in `/home/alexlitz/Documents/misc/c4_release/` — that's the coordinator's worktree and concurrent agents will trample each other's edits.
