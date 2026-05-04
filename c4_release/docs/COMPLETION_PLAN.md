# Completion Plan: Testing Checklist

**Date:** 2026-05-03
**Current:** 3/10 complete (items 1, 2, 10)
**Target:** 10/10 complete

---

## Dependency Graph

```
Phase 1 (independent, can run in parallel)
├── 1A: KV Cache Eviction Tests ← code exists, just need to run + expand
├── 1B: Fix Tool Use Hardcoded Path ← 1-line fix, 5 tests unblocked
├── 1C: Expand Conversational I/O Tests ← write ~50 new tests
└── 1D: Fix Bundler ← debug 9/100, get to 1096/1096

Phase 2 (depends on Phase 1C)
├── 2A: Pure Autoregressive I/O ← remove _memory shadow dict
└── 2B: Quine ← depends on 2A

Phase 3 (depends on Phases 1D, 1A)
├── 3A: ONNX Export ← replace ALiBi/softmax1 with ONNX-compatible ops
├── 3B: ONNX Runtime 100+ Tests ← depends on 3A
└── 3C: C4 C Runtime 1000+ Tests ← depends on 3A + bundler

Phase 4 (cleanup)
└── 4A: CI Pipeline + Coverage Dashboard
```

---

## Phase 1: Independent Fixes (3-4 days)

### 1A. KV Cache Eviction — VERIFY EXISTING TESTS (0.5 day)

**Status:** 17 tests fully implemented in `test_kv_cache_eviction.py`, kv_cache.py complete.
**Action:**
1. Run the existing 17 tests: `pytest tests/test_kv_cache_eviction.py -v`
2. If all pass → checklist item done (rename from "NOT TESTED" to "PASSING")
3. If failures → debug and fix
4. Optionally add 8-10 stress tests for long programs (500+ steps) from TEST_COVERAGE_PLAN.md Phase 1B

**Delivers:** Checklist item 6 ✅

### 1B. Fix Tool Use Hardcoded Path (0.5 day)

**Status:** 19 tool_use tests pass, 5 tool_use_io tests skipped due to `tooluse_io.py` hardcoded path.
**Action:**
1. Read `tools/tooluse_io.py` — find hardcoded path
2. Replace with `os.path.dirname(__file__)` or relative path
3. Run `pytest tests/test_tool_use_io.py -v` — verify 27/27 pass
4. Combined with existing 19 tool_use tests → 46 tests total

**Delivers:** Checklist item 5 ✅

### 1C. Expand Conversational I/O Tests (1-2 days)

**Status:** Only 2 tests exist. PRTF still uses `_memory` shadow dict for format strings.
**Action:**
1. Write 50 tests in `tests/test_conversational_io_comprehensive.py`:
   - Literal string printf: `printf("hello")` (15 tests)
   - Format specifiers: `%d`, `%x`, `%c`, `%s`, `%%` (15 tests)
   - Multiple printf calls (5 tests)
   - Return value after printf (5 tests)
   - Printf in loops (5 tests)
   - Edge cases: empty string, long string, special chars (5 tests)
2. These tests work with current architecture (shadow dict is fine for now)
3. Later Phase 2A removes the shadow dict dependency

**Delivers:** Checklist item 4 PARTIAL → can verify PRTF output is correct

### 1D. Fix Bundler (1-2 days)

**Status:** 35 unit tests pass, but 1096 end-to-end tests broken (9/100).
**Root Cause:** `BundlerRunner` requires `.c4onnx` model file; `neural_bundler.py` generates C code but the generated runtime may have weight format mismatches or opcode handling issues after handler removal.
**Action:**
1. Generate `.c4onnx` model weights file from current model:
   ```python
   model = AutoregressiveVM()
   set_vm_weights(model)
   model.save("models/transformer_vm.c4onnx")
   ```
2. Run single failing test through bundler with verbose output
3. Debug: likely issues are:
   - Weight serialization format (`.arvm` binary vs `.c4onnx`)
   - Generated C runtime missing handler-removal updates
   - Token prediction loop in C differs from Python
4. Fix root cause, run to 1096/1096
5. Verify `tests/test_bundler_1096.py` passes

**Delivers:** Checklist item 8 ✅

---

## Phase 2: I/O & Quine (3-4 days, depends on Phase 1C)

### 2A. Pure Autoregressive I/O (2-3 days)

**Status:** PRTF reads format strings from `_memory` shadow dict in Python runner.
**Target:** Model reads format strings via L15 memory lookup, outputs characters directly.
**Action:**
1. Verify L15 memory lookup works for data section (format strings at 0x10000+)
2. Remove `_memory` shadow dict dependency from runner's conversational_io path
3. Model should emit THINKING_END → character tokens → THINKING_START autonomously
4. Run all conversational I/O tests from Phase 1C against pure-autoregressive path
5. Fix any failures

**Delivers:** Checklist item 4 ✅ (full)

### 2B. Quine (1 day)

**Status:** 25 quine tests exist. 4 conditionally skip. Requires Phase 2A for PRTF.
**Action:**
1. Ensure Phase 2A works (PRTF produces correct output)
2. Run quine execution tests: `pytest tests/test_quine.py -v -m quine`
3. Verify self-replication: output == source
4. Run through bundler (requires Phase 1D) for standalone executable
5. Fix any failures

**Delivers:** Checklist item 9 ✅

---

## Phase 3: ONNX & C Runtime (5-7 days, depends on Phases 1D, 1A)

### 3A. ONNX Export (2-3 days)

**Blockers:**
- ALiBi attention: dynamic shape broadcasting incompatible with ONNX
- softmax1 (ZFOD): not a standard ONNX operator
- DivModModule post-op: custom PyTorch module

**Action:**
1. Replace ALiBi with learned positional embeddings for export
   - OR: decompose ALiBi into standard ONNX ops (broadcast over sequence dim)
2. Replace softmax1 with `softmax(x, dim=-1)` + small epsilon fix for last position
   - OR: register as custom ONNX op via `onnxscript`
3. Decompose DivModModule into standard ops (integer division + modulo via mul+sub)
4. Create `scripts/export_onnx.py`:
   ```python
   model = AutoregressiveVM()
   set_vm_weights(model)
   torch.onnx.export(model, dummy_input, "models/transformer_vm.onnx")
   ```
5. Verify: identical outputs for identical inputs (PyTorch vs ONNX Runtime)
6. Write 20 export verification tests in `tests/test_onnx_export.py`

**Delivers:** Checklist item 3 PARTIAL (export works)

### 3B. ONNX Runtime 100+ Tests (1-2 days)

**Status:** `ONNXVMRunner` stub returns False. onnxruntime 1.23.2 installed.
**Action:**
1. Implement `ONNXVMRunner.run_program()` using `onnxruntime.InferenceSession`
2. Port autoregressive generation loop to ONNX inference
3. Run 100+ tests from the 1096 suite
4. Verify exit codes match Python implementation
5. Write results to `tests/test_onnx_runtime_1096.py` (convert from CLI stub to pytest)

**Delivers:** Checklist item 3 ✅

### 3C. C4 C ONNX Runtime 1000+ Tests (2 days)

**Status:** Multiple C runtimes exist. `CRuntimeRunner` implemented but depends on C binary CLI.
**Action:**
1. Fix C runtime to accept bytecode + data as CLI args and output "Exit code: N"
2. Wire `CRuntimeRunner.run_program()` end-to-end
3. Compile with: `gcc -O3 -lm bundler/neural_runtime.c -o c4_runtime`
4. Run 1000+ tests through C runtime
5. Verify results match Python/ONNX implementations

**Delivers:** Checklist item 7 ✅

---

## Phase 4: CI & Cleanup (1 day)

### 4A. CI Pipeline + Coverage Dashboard

**Action:**
1. Update `.github/workflows/test.yml`:
   - Fast CI (<5 min): lint + 44 smoke tests
   - Comprehensive CI (30-60 min): 1096 suite + stress tests + tool use + KV cache
   - Nightly: ONNX + C runtime + quine
2. Run `pytest --cov=neural_vm --cov-report=html tests/` — target 80%+ line coverage
3. Add `pytest-xdist` for parallel execution
4. Update `docs/TESTING_CHECKLIST_STATUS.md` to 10/10 ✅
5. Update `docs/OPCODE_TEST_STATUS.md` — all opcodes neural
6. Update `NEURAL_VM_STATUS.md` for handler-free architecture

**Delivers:** Continuous quality infrastructure

---

## Effort Summary

| Phase | Items | Effort | Dependencies |
|-------|-------|--------|-------------|
| **Phase 1** | KV cache ✅, Tool use ✅, Conv I/O partial, Bundler ✅ | **3-4 days** | None |
| **Phase 2** | Pure I/O ✅, Quine ✅ | **3-4 days** | Phase 1C |
| **Phase 3** | ONNX export ✅, ONNX 100+ ✅, C runtime ✅ | **5-7 days** | Phase 1D |
| **Phase 4** | CI + docs | **1 day** | All above |

**Total: 12-16 days**

## Checklist Completion Timeline

| Day | Milestone |
|-----|-----------|
| 1 | Items 5 (tool use) + 6 (KV cache) ✅ |
| 2-3 | Item 8 (bundler) ✅ |
| 3-4 | Item 4 partial (50 conv I/O tests) |
| 5-7 | Item 4 full (pure autoregressive I/O) ✅ |
| 7-8 | Item 9 (quine) ✅ |
| 8-10 | Item 3 partial (ONNX export) |
| 10-11 | Item 3 full (ONNX 100+ tests) ✅ |
| 12-13 | Item 7 (C runtime 1000+) ✅ |
| 14-16 | Phase 4 (CI, docs, coverage) |

## Risk Factors

1. **ONNX export blockers are real** — ALiBi/softmax1 may require significant rework. Mitigation: use `onnxscript` custom ops or decompose into standard ops.
2. **Bundler may have deep issues** — handler removal changed many internals. Mitigation: incremental debugging, compare Python vs C outputs step-by-step.
3. **C runtime accuracy** — fixed-point arithmetic may lose precision vs float32. Mitigation: verify against Python baseline per-opcode.
4. **Quine requires 50k steps** — slow to test iteratively. Mitigation: use speculative execution for speed.
