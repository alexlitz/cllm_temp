# Phase 8: Switch Headline Test Runner to Pure Neural

**Date:** 2026-05-11  
**Author:** Research Agent  
**Status:** Scope Definition (Research Complete)

---

## Executive Summary

Phase 8's goal is to flip the headline test runner (`test_suite_1096_pytest.py` and `test_smoke.py`) from the default handler-based runner to pure-neural mode. Currently, tests use `AutoregressiveVMRunner()` (handler mode), which delegates most computation to Python overrides. The change requires:

1. **Understand what "headline runner" is** → Fixtures in `conftest.py` that drive `test_smoke.py`, `test_suite_1096_pytest.py`, and other integration tests
2. **Map current handler vs neural split** → Which opcodes use Python, which use neural, which fail in pure-neural
3. **Identify blockers from Phases 1-7** → What must pass before Phase 8 can succeed
4. **Decide implementation strategy** → Flip fixture default? Create a separate pure_neural suite? Parametrize?
5. **Execute flipping with risk assessment** → Step-by-step plan with per-step risk

---

## 1. What is the "Headline Test Runner"?

### Definition
The "headline" runner is the set of fixtures used by the **main integration test suites**:

| Test Suite | Fixture | Scope | Count |
|---|---|---|---|
| `test_smoke.py` | `quick_runner` | function-scoped | ~300 quick tests |
| `test_suite_1096_pytest.py` | class-scoped `runner` | class-scoped | 1096 tests |
| `run_1000_tests.py` (standalone) | Manual `AutoregressiveVMRunner()` | N/A | ~1096 tests |

### Current Implementation (Handler Mode)

**conftest.py fixtures:**

```python
@pytest.fixture
def quick_runner():
    """Quick runner with limited steps for unit tests."""
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    return runner

@pytest.fixture(scope="class")
def runner(self):
    """Shared runner for all tests in class."""
    return AutoregressiveVMRunner()  # test_suite_1096_pytest.py:60
```

**Key:** Both create `AutoregressiveVMRunner()` **with no kwargs**. This means:
- `pure_neural=False` (default) → All Python overrides enabled
- `trust_neural_alu=False` (default) → ALU lookup mode
- Handlers active: `_dispatch_step`, `_compute_alu_legacy`, `_syscall_*`

### Pure-Neural Alternative

**conftest.py already defines:**

```python
@pytest.fixture
def pure_neural_runner(_pure_neural_runner_model):
    """Runner with NO Python overrides at all (100% neural when supported)."""
    runner = _pure_neural_runner_model
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner
```

**Key:** This runner is **session-scoped model + function-scoped state reset**, creates via:
```python
AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
```

**Usage:** Currently only referenced by `test_pure_neural_*.py` files. **Never used by smoke or 1096 suite.**

---

## 2. Current Handler vs Neural Split

### Handler Mode (`pure_neural=False`)

**Architecture:** `_dispatch_step` runs for every step. Pseudo-code:

```python
def _dispatch_step(self, context, bytecode, exec_idx, prefix_len, output):
    exec_op = bytecode[exec_idx] & 0xFF
    
    if self.pure_neural:
        # Pure-neural branch: skip all overrides (see section 3)
        ...
    else:
        # Handler mode (current smoke/1096 path)
        if exec_op in self._func_call_handlers:
            self._func_call_handlers[exec_op](...)  # Python override
        elif exec_op in self._syscall_handlers:
            self._syscall_handlers[exec_op](...)    # Python override
        elif exec_op in _RUNNER_ALU_OPS:
            result = self._compute_alu_legacy(op, stack_val, ax_val)
            self._override_register_in_last_step(..., result)  # Python override
        # Also:
        # - _override_register_in_last_step() for PC/SP/BP/AX
        # - _track_memory_write() for memory side-effects
        # - _inject_mem_section() for MEM token recovery
```

**Handlers (run every step, except explicitly guarded):**

| Op | Handler | Type | Location | Issue |
|---|---|---|---|---|
| IMM | `_func_call_IMM` | func | run_vm.py:1195 | PC+AX injection |
| JMP/BZ/BNZ | `_func_call_JMP/BZ/BNZ` | func | run_vm.py:1211-1262 | PC override |
| JSR/LEV | `_func_call_JSR/_LEV` | func | run_vm.py:1277-1340 | PC/SP/BP cascades |
| ENT | `_func_call_ENT` | func | run_vm.py:1363-1398 | SP/BP frame setup |
| PSH | `_func_call_PSH` | func | run_vm.py:1407-1413 | SP decrement + MEM write |
| LI/LC/SI/SC | (no handler) | fallback | _track_memory_write | Python dict lookups |
| ADD/SUB/MUL/DIV/MOD | `_compute_alu_legacy` | ALU | run_vm.py:505-554 | Python compute |
| OR/XOR/AND/EQ/NE/LT/GT/LE/GE/SHL/SHR | `_compute_alu_legacy` | ALU | run_vm.py:505-554 | Python compute |
| PRTF/OPEN/READ/CLOS | `_syscall_*` | syscall | run_vm.py:350-420 | File I/O |
| MALC/FREE/MSET/MCMP | (none) | fallback | heap allocator | Python heap state |
| NOP | (skip) | N/A | N/A | Passthrough |

**Result:** 100% of computation relies on Python fallbacks. Tests **always pass** because Python is correct.

### Pure-Neural Mode (`pure_neural=True`)

**Architecture:** `_dispatch_step` skips all Python overrides. Pseudo-code:

```python
def _dispatch_step(self, context, bytecode, exec_idx, prefix_len, output):
    if self.pure_neural:
        # Extract registers from neural output ONLY
        neural_pc = self._extract_register(context, Token.REG_PC)
        neural_ax = self._extract_register(context, Token.REG_AX)
        # ... (no overrides, no _compute_alu_legacy, no handler dispatch)
        # Only PUTCHAR/GETCHAR bypass logic (tool boundary)
```

**What works (Phase 1+):**
- `IMM N, EXIT` — Phase 1 ✓ (13/13 tests pass per AGENT_CONTEXT.md)
- `IMM, IMM, ..., EXIT` — Phase 1 multi-IMM: 1/13 pass (others have L6 IMM-bake leak, L14 OUTPUT corruption)
- `PSH, ADD` — Phase 2 partial (10/16 pass; bitwise AND/OR/XOR still broken)
- `DIV/MOD` — Phase 7 partial (likely work; under-tested)

**What fails:**
- Bitwise AND/OR/XOR (L6 threshold bug uncovered during Phase 2)
- LEV semantics (OP_LEV relay disabled at L6; all L9/L15/L16 gates never fire)
- JSR semantics (multiple bugs: L6 SP-decrement missing, L14 val routes wrong source)
- JMP/BZ/BNZ when using raw idx instead of `idx * INSTR_WIDTH + PC_OFFSET`
- 6+ sequential IMMs (L4 carry-forward bug, not in same layer as L3 FFN)
- HEAP (MALC/FREE/MSET/MCMP) — No neural implementation yet

**Result:** Tests fail because neural network has incomplete/incorrect behavior.

---

## 3. Blocker Analysis: Which Phases Block Phase 8?

### Dependency Chain

```
Phase 1: PC + AX coherence
  ↓ (required before Phase 2)
Phase 2: SP arithmetic + PSH MEM write + binary ops
  ↓ (required before Phase 3)
Phase 3: Multi-byte AX + ALU completeness (ADD/SUB multi-byte, SHL/SHR cross-byte)
  ↓ (required before Phase 4)
Phase 4: Control flow (JMP/BZ/BNZ)
  ↓ (required before Phase 5)
Phase 5: Function calls (JSR/ENT/LEV)
  ↓ (required before Phase 6)
Phase 6: I/O syscalls (PRTF/READ/GETCHAR/PUTCHAR)
  ↓ (required before Phase 7)
Phase 7: Heap (MALC/FREE/MSET/MCMP) + DIV/MOD parity
  ↓ (required before Phase 8)
Phase 8: Flip headline runner to pure_neural=True
```

### Current Completion Status

From AGENT_CONTEXT.md (2026-05-09):

| Phase | Test File | Status | Key Blocker |
|---|---|---|---|
| **1** | `test_pure_neural_pc.py` | **✓ 13/13** | NONE — Phase 1 DONE |
| 2 | `test_pure_neural_psh_add.py` | 10/16 | Bitwise AND/OR/XOR fail (L6 threshold theory disproved) |
| 2-ext | `test_pure_neural_phase2_ext.py` | new | (same as Phase 2) |
| 3 | `test_pure_neural_multibyte.py` | 9 xfail | Multi-byte routing (OUTPUT_BYTE_1/2/3 dims needed) |
| 4 | `test_pure_neural_jmp_bz.py` | 12 xfail | Test encoding bug (raw idx vs PC) likely; OP_JMP/BZ/BNZ circuits unknown |
| 5 | `test_pure_neural_jsr_ent_lev.py` | 7 xfail | LEV relay disabled; JSR L6 SP-decrement missing |
| 6 | `test_pure_neural_io.py` | 7 xfail | PUTCHAR neural routing ≈80% done; READ/GETCHAR untested |
| 7 | `test_pure_neural_heap_div.py` | 25 xfail | DIV/MOD likely work (8 tests probably pass — under-tested); HEAP not implemented |

### Minimum Viable Phase 8 Scope

**Option A (Full Pure-Neural):**
Flip *all* headline tests to pure_neural mode. Requires **all Phases 1-7 complete** (~16-25 weeks per PURE_NEURAL_GAP_ANALYSIS.md estimate).

**Option B (Phase 1 Only + Gradual Migration):**
- Flip `test_smoke.py` to pure_neural mode immediately (works for Phase 1 tests)
- Leave `test_suite_1096_pytest.py` in handler mode
- Add separate `test_suite_1096_pure_neural_pytest.py` that skips phases 2+ until they land

**Option C (Parametrized Dual-Path):**
- Keep existing fixture, add parameter: `quick_runner(mode="handler")` vs `quick_runner(mode="pure_neural")`
- Run each test twice, report separately
- Useful for regression detection but doubles test time

### Realistic Phase 8 Expectations

Even after Phase 7 lands, the 1096-test suite **will not pass 1096/1096 in pure-neural mode**. The test suite includes:

- Compiled C programs with varargs (PRTF %d, %s, etc.) → requires neural format-string parsing
- Recursion and deep call stacks → requires LEV to work perfectly
- Malloc/free intensive programs → heap not yet neurally implemented
- Floating-point math (if any) → no FPU opcodes defined

**Realistic target:** 200-400/1096 passing in pure-neural mode by end of Phase 7. Residual failures become Phase 8+ backlog.

---

## 4. What Needs to Change for Phase 8

### Step-by-Step Implementation Plan

#### Step 1: Understand Current Fixture Dependency Graph

**Current usage:**

```
conftest.py:
  - quick_runner (function-scoped, AutoregressiveVMRunner())
    → used by test_smoke.py (all 300 tests)
  - runner (class-scoped, AutoregressiveVMRunner())
    → used by test_suite_1096_pytest.py::TestSuite1096.test_program()
    → used by test_suite_1096_pytest.py::TestSuiteCategories (slow tests)
  - pure_neural_runner (function-scoped, AutoregressiveVMRunner(pure_neural=True, ...))
    → used by test_pure_neural_pc.py, test_pure_neural_psh_add.py, ... (7 phase test files)
  - smoke_runner (function-scoped, BatchedSpeculativeRunner(...))
    → NOT used anywhere currently
```

**Task:** Grep for all fixture references:

```bash
grep -r "def test_\|quick_runner\|smoke_runner\|fast_runner\|batch_runner\|neural_runner" tests/*.py \
  | grep -E "(quick_runner|smoke_runner|runner\()" | head -50
```

#### Step 2: Decide: Replace vs Parallel

Three options:

**A. Replace In-Place** (High risk, fast)
```python
# conftest.py: line 109-116
@pytest.fixture
def quick_runner():
    """Quick runner with limited steps for unit tests."""
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)  # ← CHANGE
    return runner
```

**Pros:**
- Minimal code change (1 line per fixture)
- Honest reporting: failures in pure_neural show up immediately

**Cons:**
- All 300 smoke tests + 1096 suite tests fail until all Phases complete
- No regression detection (pure_neural failures mask handler bugs)
- High CI friction during development

---

**B. Parallel (Pure-Neural + Handler)** (Low risk, clear separation)
```python
# conftest.py
@pytest.fixture
def quick_runner():
    """Handler mode (Python overrides)."""
    return AutoregressiveVMRunner()

@pytest.fixture
def quick_runner_pure_neural():
    """Pure-neural mode (no Python overrides)."""
    return _pure_neural_runner_model

# Then create new test files:
# test_smoke_pure_neural.py    (imports quick_runner_pure_neural)
# test_suite_1096_pure_neural_pytest.py (imports runner_pure_neural)
```

**Pros:**
- Handler mode tests continue passing (no friction)
- Pure-neural suite grows as phases complete
- Clear separation: handler vs neural

**Cons:**
- Test duplication (2x code)
- Eventually need to retire handler mode tests

---

**C. Parametrized** (Medium risk, data-driven)
```python
@pytest.fixture(params=["handler", "pure_neural"])
def quick_runner(request):
    if request.param == "handler":
        return AutoregressiveVMRunner()
    else:
        return AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)

# test_smoke.py automatically runs each test twice (once per mode)
```

**Pros:**
- Single test file, both modes
- Regression detection (handler mode stays fast, pure_neural progresses)

**Cons:**
- 2x test runtime
- xfail/skip logic gets complex

---

**Recommendation:** **Option B (Parallel)** for Phase 8.

- Keep handler tests running (zero friction, no regressions)
- Create `tests/test_smoke_pure_neural.py` that mirrors `test_smoke.py` but uses `pure_neural_runner`
- Create `tests/test_suite_1096_pure_neural_pytest.py` that mirrors `test_suite_1096_pytest.py`
- Mark pure_neural tests with `@pytest.mark.xfail` until phases complete
- Phase 8 completion = flip xfail markers to pass as phases land

#### Step 3: Create New Pure-Neural Test Suite

**New file: `tests/test_smoke_pure_neural.py`**

```python
"""
Smoke Tests - Pure Neural Mode

Mirror of test_smoke.py but using pure_neural_runner fixture.
These tests validate neural-only execution without Python overrides.

Phases 1-3: Expected to pass as phases complete.
Phases 4+: Will fail until those phases are implemented.
"""

import pytest
from neural_vm.embedding import Opcode

class TestSmokePureNeuralPhase1:
    """Phase 1: PC + AX coherence (EXPECTED TO PASS)."""

    def test_imm_exit(self, pure_neural_runner, make_bytecode):
        """IMM + EXIT works in pure-neural."""
        bytecode = make_bytecode([(Opcode.IMM, 42), Opcode.EXIT])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=10)
        assert result == 42

    def test_multi_imm(self, pure_neural_runner, make_bytecode):
        """Multiple IMMs work in pure-neural."""
        bytecode = make_bytecode([
            (Opcode.IMM, 10),
            (Opcode.IMM, 32),
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 32  # AX = last IMM value

@pytest.mark.xfail(reason="Phase 2 not yet complete")
class TestSmokePureNeuralPhase2:
    """Phase 2: PSH + ADD (EXPECTED TO FAIL UNTIL PHASE 2 DONE)."""

    def test_psh_add(self, pure_neural_runner, make_bytecode):
        """PSH + ADD works in pure-neural."""
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 32), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 42
```

**New file: `tests/test_suite_1096_pure_neural_pytest.py`**

```python
"""
1096 Tests - Pure Neural Mode

Pure-neural version of test_suite_1096_pytest.py.
Tests run through AutoregressiveVMRunner(pure_neural=True) and report
honest results without Python overrides.

Status tracking:
- Phase 1 complete: ~50 IMM-only programs pass
- Phase 2-7: ~0 programs pass (bitwise, control flow, function calls, I/O, heap all blocked)
- Residual: ~1000 programs will likely require Phase 8+ infrastructure
"""

import pytest
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

ALL_TESTS = generate_test_programs()

class TestSuite1096PureNeural:
    """Run 1096 tests in pure-neural mode."""

    @pytest.fixture(scope="class")
    def runner(self):
        """Pure-neural runner (no Python overrides)."""
        return AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)

    @pytest.mark.parametrize(
        "source,expected,description",
        ALL_TESTS,
        ids=[t[2].replace(" ", "_")[:50] for t in ALL_TESTS]
    )
    def test_program(self, runner, source, expected, description):
        """Run a single test program in pure-neural mode."""
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=2000)
        assert result == expected, \
            f"{description}: expected {expected}, got {result}"
```

#### Step 4: Update conftest.py Documentation

Add a section explaining fixture modes:

```python
# =============================================================================
# Fixture Mode Documentation
# =============================================================================
#
# The C4 Neural VM supports two execution modes for testing:
#
# 1. HANDLER MODE (default: pure_neural=False)
#    - Used by: test_smoke.py, test_suite_1096_pytest.py, etc.
#    - Architecture: Neural network generates candidate outputs;
#                    Python handler layer validates and corrects each step.
#    - Purpose: Fast regression detection, handler correctness, and during
#               development of pure-neural phases.
#    - Always passes (Python is correct by construction).
#
# 2. PURE-NEURAL MODE (pure_neural=True)
#    - Used by: test_pure_neural_pc.py, test_smoke_pure_neural.py, etc.
#    - Architecture: Neural network alone drives all computation.
#                    Python layer is blocked; only external tool boundary
#                    handlers (PRTF, OPEN, READ) are allowed.
#    - Purpose: Validate autoregressive transformer can learn full VM semantics.
#    - Passes only when corresponding Phase is complete.
#
# Per-phase test status:
#   Phase 1 (PC + AX): test_pure_neural_pc.py ✓ (13/13)
#   Phase 2 (PSH/ADD): test_pure_neural_psh_add.py (10/16, bitwise blocked)
#   Phase 3 (multibyte): test_pure_neural_multibyte.py (9 xfail)
#   Phase 4 (control): test_pure_neural_jmp_bz.py (12 xfail)
#   Phase 5 (JSR/ENT/LEV): test_pure_neural_jsr_ent_lev.py (7 xfail)
#   Phase 6 (I/O): test_pure_neural_io.py (7 xfail)
#   Phase 7 (Heap/DIV): test_pure_neural_heap_div.py (25 xfail, 8 likely pass)
#
# Recommendation: Use handler mode for main regression suite; track
# pure-neural mode separately until Phase 7 lands, then flip.
```

#### Step 5: CI/Test Plan

**During Phase 2-7 (Parallel Mode):**

```bash
# Handler mode (always fast, always passes)
pytest tests/test_smoke.py -v --tb=short
pytest tests/test_suite_1096_pytest.py::TestSuiteCategories::test_all_addition --tb=short

# Pure-neural tracking (shows progress per phase)
pytest tests/test_smoke_pure_neural.py::TestSmokePureNeuralPhase1 -v
pytest tests/test_suite_1096_pure_neural_pytest.py -k "add_" --tb=line  # Filter by category
```

**After Phase 7 lands:**

```bash
# Both modes side-by-side
pytest tests/test_smoke.py tests/test_smoke_pure_neural.py -v --tb=short
pytest tests/test_suite_1096_pytest.py tests/test_suite_1096_pure_neural_pytest.py --tb=line
```

**Final (Phase 8 complete):**

```bash
# Option 1: Deprecate handler mode, keep pure-neural
# Option 2: Keep both, mark handler as "legacy"
```

---

## 5. Risk Assessment & Execution Steps

### Step-by-Step Execution Plan

| # | Step | Effort | Risk | Blockers | Notes |
|---|---|---|---|---|---|
| 1 | Create `test_smoke_pure_neural.py` (mirror of `test_smoke.py`, Phase 1 tests only) | 1 hour | **LOW** | None | Define structure; Phase 1 tests should pass (all 13/13 from test_pure_neural_pc.py) |
| 2 | Create `test_suite_1096_pure_neural_pytest.py` (parametrized over ALL_TESTS) | 30 min | **LOW** | None | Mark as `@pytest.mark.xfail` initially; tests will fail until phases complete |
| 3 | Verify pure_neural_runner fixture works across new test files | 30 min | **LOW** | Fixture scoping | Session-scoped model + function-scoped state reset; ensure isolation |
| 4 | Update conftest.py with mode documentation | 30 min | **LOW** | None | Document why two modes exist, when to use each |
| 5 | Run Phase 1 tests, verify 13/13 still pass in new structure | 10 min | **LOW** | CPU/GPU load | Should be ~1 second per test, total ~30 seconds |
| 6 | Set up xfail tracking: create matrix of phase → expected pass count | 30 min | **MEDIUM** | Phase completeness | As Phases 2-7 land, update xfail counts |
| 7 | Add CI job that reports pure-neural pass rate per phase | 1-2 hours | **MEDIUM** | CI/CD knowledge | Useful for monitoring progress; can skip if not needed |
| 8 | (Post-Phase 7) Flip handler mode tests to pure_neural or deprecate | 1 hour | **HIGH** | All phases complete | Final decision point; may want to keep both for regression |

### Risk Breakdown

#### Risk 1: Fixture Scoping Bug (LOW risk)

**Issue:** `pure_neural_runner` is function-scoped; session-scoped model may leak state.

**Mitigation:** Code already resets `_memory`, `_mem_history`, `_mem_access_order` per test. Verify in conftest.py line 310-326.

**Test:** Run `test_pure_neural_pc.py` twice (same session), verify no cross-test pollution.

#### Risk 2: New Tests Don't Execute Correctly (LOW risk)

**Issue:** Copy-paste errors in new test files; fixture injection doesn't work.

**Mitigation:** Keep new test files as close as possible to existing ones. Use exact same `make_bytecode` and `opcodes` helpers.

**Test:** `pytest tests/test_smoke_pure_neural.py::TestSmokePureNeuralPhase1::test_imm_exit -v` should pass.

#### Risk 3: Phases Incomplete When Phase 8 Runs (MEDIUM risk)

**Issue:** If Phase 2-7 aren't done, new pure_neural tests all fail, CI becomes noisy.

**Mitigation:** Mark with `@pytest.mark.xfail(reason="Phase X not complete")`. Use `--runxfail` to see actual status.

**Monitoring:** Create a simple dashboard script that reports xfail → pass rate progression.

#### Risk 4: Handler Mode Regressions (LOW risk)

**Issue:** New fixture code breaks `quick_runner` (handler mode).

**Mitigation:** Don't modify handler fixtures. Create separate `pure_neural_*` fixtures.

**Test:** `pytest tests/test_smoke.py -v` should still pass, unchanged.

#### Risk 5: Model Cache Collision (LOW risk)

**Issue:** `AutoregressiveVMRunner._MODEL_CACHE` may collide between handler and pure_neural builds.

**Mitigation:** Cache key includes `alu_mode` ('lookup' vs 'efficient'). Different modes → different cache entries.

**Verify:** `run_vm.py:176-177` shows cache key construction.

---

## 6. Recommendation: Concrete Implementation Plan

### Phase 8 Scope (Right Now — No Code Changes)

**Goal:** Answer "What does Phase 8 actually require?"

**Deliverables (this document):**
1. ✓ Define "headline runner" (test_smoke.py, test_suite_1096_pytest.py)
2. ✓ Map handler vs neural split (conftest.py, run_vm.py, embedding.py)
3. ✓ Identify blockers (Phases 1-7 must complete)
4. ✓ Recommend strategy (Option B: Parallel)
5. ✓ Risk assessment (mostly LOW)

### Phase 8A (Quick Win — Verify Phase 1 Still Works)

**Effort:** 1-2 hours (no code changes)

**Steps:**
1. Run `pytest tests/test_pure_neural_pc.py -v` to verify all 13/13 still pass
2. Spot-check `test_smoke.py` with `quick_runner` (handler mode) to ensure no regressions
3. Document results in this file

**Risk:** NONE (read-only exploration)

### Phase 8B (Prep — Create New Test Files, Don't Run Yet)

**Effort:** 2-3 hours

**Steps:**
1. Create `tests/test_smoke_pure_neural.py` with Phase 1 tests (copy from test_pure_neural_pc.py)
2. Create `tests/test_suite_1096_pure_neural_pytest.py` with full parametrization
3. Mark all Phase 2+ tests with `@pytest.mark.xfail(reason="Phase X not yet complete")`
4. Commit as draft (branch `f-phase8-prep`)

**Risk:** LOW (new files don't interfere with existing tests)

### Phase 8C (Execution — Flip Fixtures Once Phases Land)

**Effort:** 1 hour (final flip)

**Steps:**
1. Once all Phases 1-7 complete and pure_neural tests reach >90% pass rate:
   - Option 1: Update conftest.py to flip `quick_runner` default to pure_neural
   - Option 2: Deprecate handler-mode tests, promote pure_neural tests to main suite
2. Run full suite: `pytest tests/test_smoke_pure_neural.py tests/test_suite_1096_pure_neural_pytest.py`
3. Report final metrics (X/1096 now passing in pure_neural mode)

**Risk:** HIGH (breaking change, but only after phases are done)

---

## 7. Detailed Blocker List: What Phases 2-7 Must Deliver

### Phase 2 Blockers (PSH/ADD in pure_neural)

**Must-have:**
- PSH correctly decrement SP by 8 (neural circuit in L7/L8/L9)
- PSH correctly write AX to MEM at SP-8 (MEM token sequence at PSH)
- Binary ops (ADD/SUB) retrieve stack operand from most-recent MEM-write at SP
- Bitwise AND/OR/XOR work (currently blocked by L6 threshold issue)

**Test:** `test_pure_neural_psh_add.py` reaches 16/16 (currently 10/16)

**Risk if not done:** ~100-200 tests in 1096 suite fail (most involve arithmetic)

---

### Phase 3 Blockers (Multi-byte AX in pure_neural)

**Must-have:**
- AX bytes 1/2/3 routed through sequence positions (not just byte 0)
- OUTPUT_BYTE_1/2/3 dims or equivalent in BD format
- Multi-byte SUB borrow fixed
- Cross-byte SHL/SHR (shift ≥ 8) fixed

**Test:** `test_pure_neural_multibyte.py` reaches 9/9 (currently 9 xfail)

**Risk if not done:** All tests with values ≥ 256 fail (~300 in 1096 suite)

---

### Phase 4 Blockers (Control flow in pure_neural)

**Must-have:**
- JMP target resolved from immediate (neural circuit in L5/L6)
- BZ/BNZ branch condition read from AX==0 / AX!=0
- L5 hard-coded address 3 issue fixed (MEDIUM priority per notes)

**Test:** `test_pure_neural_jmp_bz.py` reaches 12/12 (currently 12 xfail; likely test encoding bug)

**Risk if not done:** All if/while/for loops fail (~150 in 1096 suite)

---

### Phase 5 Blockers (Function calls in pure_neural)

**Must-have:**
- JSR push PC+8 to MEM at SP-8, jump to target
- ENT frame setup: SP -= 8; MEM[SP] = BP; BP = SP; SP = BP + imm*4
- LEV frame teardown: restore BP/PC from MEM
- OP_LEV relay re-enabled at L6 (currently disabled)

**Test:** `test_pure_neural_jsr_ent_lev.py` reaches 7/7 (currently 7 xfail)

**Risk if not done:** All compiled C functions fail (~900 in 1096 suite — largest blocker)

---

### Phase 6 Blockers (I/O in pure_neural)

**Must-have:**
- PUTCHAR: neural network routes AX byte 0 → OUTPUT, runner reads (≈80% done)
- GETCHAR: neural attention reads from USER_INPUT_START..END in context
- PRTF: format-string parsing + neural routing of format specifiers (%d, %s, %c)
- READ: extract bytes from stdin into MEM/AX

**Test:** `test_pure_neural_io.py` reaches 7/7 (currently 7 xfail)

**Risk if not done:** All printf/scanf programs fail (~50 in 1096 suite)

---

### Phase 7 Blockers (Heap & DIV/MOD in pure_neural)

**Must-have:**
- DIV/MOD multi-byte verified (likely already work; under-tested)
- MALC: bump allocator state in neural (address, size tracking)
- FREE: de-allocation + zero-fill in neural
- MSET/MCMP: memory utilities in neural

**Test:** `test_pure_neural_heap_div.py` reaches 25/25 (currently 25 xfail; 8 DIV/MOD likely pass)

**Risk if not done:** All malloc-using programs fail (~100-200 in 1096 suite)

---

## Appendix A: Current Fixture Reference Map

### Handler Mode (Default)

```
conftest.py:60       quick_runner()
  ↓
test_smoke.py        (300+ tests, all use quick_runner)
  ├─ test_imm_exit
  ├─ test_add_basic
  ├─ test_sub_basic
  └─ ... (all pass in handler mode)

conftest.py:57       runner() [class-scoped]
  ↓
test_suite_1096_pytest.py (1096 tests)
  ├─ TestSuite1096.test_program (parametrized over ALL_TESTS)
  ├─ TestSuite1096Quick.test_quick_program (first 100)
  └─ TestSuiteCategories (add_, sub_, etc. — all pass)
```

### Pure-Neural Mode

```
conftest.py:173      _pure_neural_runner_model() [session-scoped]
  ↓
conftest.py:311      pure_neural_runner() [function-scoped wrapper]
  ↓
test_pure_neural_pc.py (13 tests)
  ├─ TestPureNeuralSingleInstruction (6 tests, all pass)
  ├─ TestPureNeuralTwoInstructions (3 tests, all pass)
  └─ TestPureNeuralThreeOrMoreInstructions (4 tests, all pass)

test_pure_neural_psh_add.py (16 tests, 10 pass, 6 xfail)
test_pure_neural_multibyte.py (9 xfail)
test_pure_neural_jmp_bz.py (12 xfail)
test_pure_neural_jsr_ent_lev.py (7 xfail)
test_pure_neural_io.py (7 xfail)
test_pure_neural_heap_div.py (25 xfail, 8 likely pass)
```

---

## Appendix B: Opcode Dispatch Matrix

| Opcode | Python Handler | Neural Path | Phase Dependency | Status |
|---|---|---|---|---|
| **Arithmetic** |
| ADD | _compute_alu_legacy | L8/L9 ALU | Phase 2 | ✓ Phase 1 (byte 0); Phase 3 (multi-byte) |
| SUB | _compute_alu_legacy | L8/L9 ALU | Phase 2 | ✓ Phase 1 (byte 0); Phase 3 (borrow) |
| MUL | _compute_alu_legacy | L11/L12 ALU | Phase 3 | ? Unknown; likely works |
| DIV | _compute_alu_legacy | L13 ALU | Phase 7 | ✓ Likely works (under-tested) |
| MOD | _compute_alu_legacy | L13 ALU | Phase 7 | ✓ Likely works (under-tested) |
| **Bitwise** |
| OR | _compute_alu_legacy | L10 ALU | Phase 2 | ✗ L6 threshold bug |
| XOR | _compute_alu_legacy | L10 ALU | Phase 2 | ✗ L6 threshold bug |
| AND | _compute_alu_legacy | L10 ALU | Phase 2 | ✗ L6 threshold bug |
| **Shift** |
| SHL | _compute_alu_legacy | L13 ALU | Phase 3 | ✗ Cross-byte (≥8) fails |
| SHR | _compute_alu_legacy | L13 ALU | Phase 3 | ✗ Cross-byte (≥8) fails |
| **Compare** |
| EQ | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| NE | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| LT | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| GT | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| LE | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| GE | _compute_alu_legacy | L4/L5 FFN | Phase 2 | ✓ Likely works (under-tested) |
| **Immediate** |
| IMM | _func_call_IMM | L6 IMM bake | Phase 1 | ✓ Phase 1 (1-5 IMMs); Phase 2+ (6+ IMMs) |
| **Control Flow** |
| JMP | _func_call_JMP | L5/L6 circuits | Phase 4 | ✗ Untested in pure_neural |
| BZ | _func_call_BZ | L5/L6 circuits | Phase 4 | ✗ Untested in pure_neural |
| BNZ | _func_call_BNZ | L5/L6 circuits | Phase 4 | ✗ Untested in pure_neural |
| **Function Calls** |
| JSR | _func_call_JSR | L6/L7/L14 relay | Phase 5 | ✗ L6 SP-decrement missing |
| ENT | _func_call_ENT | L7/L8/L9 state machine | Phase 5 | ✗ Complex multi-step; untested |
| LEV | _func_call_LEV | L9/L15/L16 gates | Phase 5 | ✗ OP_LEV relay disabled at L6 |
| **Memory** |
| PSH | _func_call_PSH | L7/L8/L9 + L15 MEM | Phase 2 | ✓ Byte 0; Phase 3 (multi-byte) |
| LI | (none) | L15 attention | Phase 2 | ✓ Likely works (under-tested) |
| LC | (none) | L15 attention | Phase 2 | ✓ Likely works (under-tested) |
| SI | (none) | L7/L14/L15 MEM | Phase 2 | ✗ Unknown neural path |
| SC | (none) | L7/L14/L15 MEM | Phase 2 | ✗ Unknown neural path |
| **Stack** |
| ADJ | (none) | L7/L8/L9 | Phase 2 | ✓ Now fully neural (migrated) |
| **I/O** |
| PRTF | _syscall_prtf | L6 routing + TOOL_CALL | Phase 6 | ✗ Untested; format parsing incomplete |
| OPEN | _syscall_open | TOOL_CALL | Phase 6 | ✗ TOOL_CALL boundary exists; neural trigger unknown |
| READ | _syscall_read | TOOL_CALL | Phase 6 | ✗ Untested |
| CLOS | _syscall_clos | TOOL_CALL | Phase 6 | ✗ Untested |
| PUTCHAR | (none) | L6 routing + OUTPUT | Phase 6 | ≈80% (OUTPUT routing done; TOOL_CALL emit needs work) |
| GETCHAR | (none) | USER_INPUT attention | Phase 6 | ✗ No neural path yet |
| **Heap** |
| MALC | (none) | stdlib | Phase 7 | ✗ No neural allocator state |
| FREE | (none) | stdlib | Phase 7 | ✗ No neural deallocator |
| MSET | (none) | stdlib | Phase 7 | ✗ No neural memory set |
| MCMP | (none) | stdlib | Phase 7 | ✗ No neural memory compare |
| **Misc** |
| NOP | (skip) | (passthrough) | Phase 1 | ✓ Works |
| EXIT | (none) | (reader) | Phase 1 | ✓ Works |

---

## Conclusion

Phase 8 is a **flip, not a build**. The architecture is already in place:

- `pure_neural_runner` fixture exists and works for Phase 1 (13/13 tests)
- `AutoregressiveVMRunner(pure_neural=True)` already blocks Python overrides
- New test files can mirror existing ones with minimal changes

**Timeline:**
- **Now:** Write this scope document (DONE)
- **Weeks 1-12:** Complete Phases 2-7 (in parallel with Phase 8 prep)
- **Week 13:** Create new pure_neural test suites (test_smoke_pure_neural.py, etc.)
- **Week 14:** Flip fixture defaults or deprecate handler mode
- **Week 15+:** Residual Phase 8+ backlog (currently ~1000/1096 tests)

**Success Metrics:**
- Phase 1: 13/13 ✓
- Phase 2-7: All phases reach ≥90% xfail→pass rate
- Phase 8: Flip completes with 200-400/1096 honest pass rate (residual backlog clearly identified)

