# C4 Transformer VM Testing Guide

**Complete guide to testing the C4 VM across all backends and features**

---

## Quick Start

### Run Fast Smoke Test (< 10 seconds)
```bash
cd c4_release
python tests/run_1000_tests.py --quick --mode fast
```

Expected: ✅ 100/100 tests passed

### Run Full Test Suite (30-60 minutes)
```bash
python tests/run_1000_tests.py
```

Expected: ✅ 1250/1250 tests passed

---

## Test Suite Overview

**Total Tests**: 1250
**Categories**: 16

| Category | Count | Description |
|----------|-------|-------------|
| Arithmetic | 200 | Basic add, sub, mul, div operations |
| Modulo | 50 | Modulo operations |
| Variables | 100 | Variable assignment and usage |
| Conditionals | 100 | If/else control flow |
| Loops | 100 | While loops |
| Functions | 150 | Function calls and returns |
| Recursion | 100 | Recursive functions |
| Expressions | 100 | Complex expressions |
| GCD | 50 | GCD algorithm |
| Nested Functions | 50 | Nested function definitions |
| Edge Cases | 50 | Boundary conditions |
| Abs Diff | 25 | Absolute difference |
| Boolean Logic | 25 | Boolean operations |
| **Long Context** | **50** | **Deep recursion, large loops** |
| **Conversational I/O** | **50** | **I/O operations** |
| **Tool Calling** | **50** | **Tool/syscall simulation** |

---

## Testing Backends

### 1. Fast Mode (Python Reference)
```bash
python tests/run_1000_tests.py --mode fast
```

- **Speed**: ⚡ Fastest (20,000+ tests/sec)
- **Use**: Quick validation, CI pipelines
- **Dependencies**: None

### 2. Transformer Mode (Neural VM)
```bash
python tests/run_1000_tests.py --mode transformer
```

- **Speed**: 🐌 Slow (model loading ~2 min, execution varies)
- **Use**: Full neural validation, correctness checks
- **Dependencies**: PyTorch, model weights

### 3. ONNX Mode
```bash
python tests/run_1000_tests.py --mode onnx
```

- **Speed**: ⚡ Fast
- **Use**: ONNX export validation
- **Dependencies**: `pip install onnxruntime`, exported .onnx model

### 4. C Runtime Mode
```bash
python tests/run_1000_tests.py --mode c-runtime
```

- **Speed**: ⚡ Fast (after compilation)
- **Use**: C implementation validation
- **Dependencies**: gcc, C runtime source

### 5. Bundler Mode
```bash
python tests/run_1000_tests.py --mode bundler
```

- **Speed**: 🐌 Slow (first run), ⚡ Fast (cached)
- **Use**: Standalone executable validation
- **Dependencies**: gcc, bundler module, model weights

---

## Testing Specific Categories

### Test Long-Context Programs
```bash
python tests/run_1000_tests.py --category long_context
```

Tests programs requiring >1000 VM steps:
- Fibonacci(10-24) - deep recursion
- Sum(1 to 500) - large loops
- GCD, factorial, nested loops

### Test Conversational I/O
```bash
python tests/run_1000_tests.py --category conversational_io
```

Tests I/O operations (currently as computations).

### Test Tool Calling
```bash
python tests/run_1000_tests.py --category tool_calling
```

Tests tool/syscall simulation.

---

## KV Cache Validation

### Correctness Tests
```bash
python tests/test_kv_cache_correctness.py
```

Validates KV cache eviction doesn't break correctness:
- ✅ Determinism (10 runs produce identical results)
- ✅ Cache size sweep (64-2048 all produce same result)
- ✅ Eviction boundary (cache < steps vs cache > steps)
- ✅ Batch consistency
- ✅ Long-context stress test (fib(20) with heavy eviction)

### Performance Benchmarks
```bash
python tests/benchmark_kv_cache.py
```

Measures memory/speed tradeoffs:
- Memory usage reduction
- Speedup vs no-cache
- Optimal cache size determination

---

## Comprehensive Testing

### All Backends (Comparison)
```bash
python tests/run_comprehensive_tests.py --all-modes --count 100
```

Runs 100 tests across all 5 backends and compares results.

Expected output:
```
BACKEND COMPARISON
------------------------------------------------
Backend         Status       Passed     Failed
------------------------------------------------
fast            ✓ OK         100        0
transformer     ✓ OK         100        0
onnx            UNAVAILABLE  -          -
c-runtime       UNAVAILABLE  -          -
bundler         UNAVAILABLE  -          -

✓ All available backends produce consistent results
```

### New Features
```bash
python tests/run_comprehensive_tests.py --features
```

Tests Categories 14-16 (long-context, I/O, tool-calling).

---

## Continuous Integration

### Fast CI (< 5 minutes)
```bash
# Quick validation on every commit
python tests/run_1000_tests.py --quick --mode fast
python tests/run_1000_tests.py --quick --mode transformer
```

### Comprehensive CI (30-60 minutes)
```bash
# Nightly or pre-release validation
python tests/run_1000_tests.py --mode fast
python tests/run_1000_tests.py --mode transformer
python tests/test_kv_cache_correctness.py
```

### Full Regression (2-3 hours)
```bash
# Pre-release comprehensive validation
python tests/run_comprehensive_tests.py --all-modes --count 500
python tests/test_kv_cache_correctness.py
python tests/benchmark_kv_cache.py
```

---

## Troubleshooting

### ONNX Mode Unavailable
```
ERROR: ONNX mode requires onnxruntime
Install: pip install onnxruntime
```

**Fix**: `pip install onnxruntime` and export model

### C Runtime Mode Unavailable
```
ERROR: C runtime mode requires gcc
Install: apt-get install gcc  OR  brew install gcc
```

**Fix**: Install gcc compiler

### Transformer Mode Slow
Model loading takes 1-2 minutes.

**Fix**: Use `--mode fast` for quick tests

### Tests Fail on One Backend
Different backends may have different bugs.

**Fix**: Compare results across backends to identify issues

---

## Test Development

### Adding New Tests

1. Edit `tests/test_suite_1000.py`
2. Add tests to appropriate category
3. Update `CATEGORY_COUNTS` dict
4. Run: `python tests/run_1000_tests.py --quick`

### Creating New Categories

See Categories 14-16 as examples:
- Category 14: Long-context programs (50 tests)
- Category 15: Conversational I/O (50 tests)
- Category 16: Tool calling (50 tests)

---

## Performance Expectations

| Mode | Test Count | Time | Tests/sec |
|------|-----------|------|-----------|
| Fast | 100 | < 1s | 20,000+ |
| Fast | 1250 | ~1min | 20,000+ |
| Transformer | 100 | ~5min | ~0.3 |
| Transformer | 1250 | ~60min | ~0.3 |
| ONNX | 100 | ~1min | ~100 |
| C Runtime | 100 | ~30s | ~200 |
| Bundler | 100 | ~5min | ~20 |

*Note: First-run times may be higher due to compilation/model loading*

---

## Test Quality Metrics

### Code Coverage
- ✅ All opcodes tested
- ✅ All ALU operations tested
- ✅ Control flow tested (JMP, JSR, BZ, BNZ)
- ✅ Memory operations tested
- ✅ Function calls tested
- ✅ Recursion tested

### Feature Coverage
- ✅ Basic arithmetic (100%)
- ✅ Loops (100%)
- ✅ Functions (100%)
- ✅ Recursion (100%)
- ✅ Long-context (NEW)
- ⏸️ Conversational I/O (partial)
- ⏸️ Tool calling (partial)

### Backend Coverage
- ✅ Fast VM (100%)
- ✅ Transformer VM (100%)
- ⏸️ ONNX (infrastructure ready)
- ⏸️ C Runtime (infrastructure ready)
- ⏸️ Bundler (infrastructure ready)

---

## References

- **TESTING_CHECKLIST.md** - Testing requirements
- **TESTING_INFRASTRUCTURE_ANALYSIS.md** - Infrastructure assessment
- **TESTING_IMPLEMENTATION_STATUS.md** - Current status
- **OPCODE_TABLE.md** - Opcode reference
- **README.md** - Project overview

---

**Last Updated**: 2026-04-08
**Test Suite Version**: 1.2 (1250 tests)
