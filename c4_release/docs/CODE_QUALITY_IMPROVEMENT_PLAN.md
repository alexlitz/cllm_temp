# Code Quality and Debugging Improvement Plan

**Date**: 2026-04-17
**Status**: Planning

## Current State Analysis

### Codebase Statistics
- **Total pytest tests**: 643+
- **neural_vm/ files**: 65+ Python modules
- **docs/**: 60+ markdown files
- **Duplicate/legacy code**: debug_archive/, test_archive/

### Current Issues
1. **Code duplication**: Multiple ALU implementations (efficient_alu.py, byte_efficient_alu.py, etc.)
2. **Scattered debugging code**: Many debug_*.py scripts in root
3. **Inconsistent documentation**: Some docs are outdated
4. **No type hints**: Most code lacks type annotations
5. **No linting**: No consistent code style enforcement
6. **Large monolithic files**: vm_step.py is very large

---

## Phase 1: Code Cleanup (Priority: High)

### 1.1 Archive Legacy Debug Scripts
```bash
# Move all debug_*.py and trace_*.py to debug_archive/
mkdir -p debug_archive
mv debug_*.py trace_*.py debug_archive/
mv check_*.py diagnose_*.py debug_archive/
```

### 1.2 Archive Legacy Test Scripts
```bash
# Move test_*.py from root to test_archive/
mkdir -p test_archive
mv test_*.py test_archive/  # Only root-level test scripts
```

### 1.3 Consolidate ALU Implementations
Current files:
- `efficient_alu.py`
- `efficient_alu_8bit.py`
- `efficient_alu_byte.py`
- `efficient_alu_integrated.py`
- `efficient_alu_neural.py`
- `byte_efficient_alu.py`

Action: Consolidate into single `alu/` module with clear interfaces.

### 1.4 Remove Unused Code
- Identify unused modules with coverage analysis
- Remove dead code paths
- Consolidate duplicate functionality

---

## Phase 2: Code Quality Tools (Priority: High)

### 2.1 Add Type Hints
```python
# Before
def run(self, bytecode, data, max_steps=1000):
    ...

# After
def run(
    self,
    bytecode: List[int],
    data: bytes,
    max_steps: int = 1000
) -> Tuple[List[int], int]:
    ...
```

### 2.2 Setup Linting (ruff/black)
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]

[tool.black]
line-length = 100
```

### 2.3 Add Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
```

### 2.4 Add mypy for Type Checking
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

---

## Phase 3: Testing Infrastructure (Priority: Medium)

### 3.1 Add pytest-cov for Coverage
```bash
pytest --cov=neural_vm --cov-report=html tests/
```

### 3.2 Add Benchmark Tests
```python
# tests/test_benchmarks.py
@pytest.mark.benchmark
def test_runner_throughput(benchmark, runner):
    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)
    result = benchmark(runner.run, bytecode, data)
    assert result[1] == 42
```

### 3.3 Add Property-Based Testing
```python
# Using hypothesis
from hypothesis import given, strategies as st

@given(st.integers(0, 255), st.integers(0, 255))
def test_add_commutative(a, b):
    source = f"int main() {{ return {a} + {b}; }}"
    bytecode, data = compile_c(source)
    _, result = runner.run(bytecode, data)
    assert result == (a + b) & 0xFFFF  # 16-bit wrap
```

### 3.4 Add Integration Test Suite
```bash
# Run all tests in order of dependency
pytest tests/test_network_purity.py  # Structure first
pytest tests/test_suite_1096_pytest.py  # Core functionality
pytest tests/test_quine.py -m slow  # End-to-end
```

---

## Phase 4: Debugging Infrastructure (Priority: Medium)

### 4.1 Structured Debugger
Create `neural_vm/debug/` module:
```
neural_vm/debug/
├── __init__.py
├── tracer.py          # Step-by-step execution tracer
├── weight_inspector.py # Inspect layer weights
├── attention_viz.py   # Visualize attention patterns
├── state_dump.py      # Dump VM state
└── comparator.py      # Compare two executions
```

### 4.2 Tracer Implementation
```python
# neural_vm/debug/tracer.py
class ExecutionTracer:
    """Trace VM execution step by step."""

    def __init__(self, verbose=False):
        self.steps = []
        self.verbose = verbose

    def trace_step(self, step_num, pc, opcode, ax, sp, bp):
        """Record a single step."""
        self.steps.append({
            'step': step_num,
            'pc': pc,
            'opcode': opcode,
            'ax': ax,
            'sp': sp,
            'bp': bp,
        })
        if self.verbose:
            print(f"[{step_num}] PC={pc} OP={opcode} AX={ax} SP={sp} BP={bp}")

    def dump(self, path=None):
        """Dump trace to file or return as dict."""
        ...
```

### 4.3 Weight Inspector
```python
# neural_vm/debug/weight_inspector.py
class WeightInspector:
    """Inspect model weights for debugging."""

    def __init__(self, model):
        self.model = model

    def show_embedding(self, token_id):
        """Show embedding vector for token."""
        ...

    def show_ffn_activations(self, layer, input_hidden):
        """Show FFN unit activations."""
        ...

    def show_attention_pattern(self, layer, head, query_pos):
        """Show attention weights for position."""
        ...
```

### 4.4 State Comparator
```python
# neural_vm/debug/comparator.py
class ExecutionComparator:
    """Compare two VM executions."""

    def compare(self, trace1, trace2):
        """Find first divergence point."""
        for i, (s1, s2) in enumerate(zip(trace1, trace2)):
            if s1 != s2:
                return {
                    'step': i,
                    'expected': s1,
                    'actual': s2,
                    'diff': self._diff(s1, s2)
                }
        return None
```

---

## Phase 5: Documentation (Priority: Low)

### 5.1 Update/Consolidate Docs
- Archive outdated session docs to `docs/archive/`
- Create `docs/ARCHITECTURE.md` overview
- Update `docs/README.md` as main entry point

### 5.2 Add API Documentation
```python
# Use Google-style docstrings
def run(self, bytecode: List[int], data: bytes, max_steps: int = 1000) -> Tuple[List[int], int]:
    """Run bytecode in the neural VM.

    Args:
        bytecode: List of 32-bit instructions
        data: Initial data section bytes
        max_steps: Maximum execution steps before timeout

    Returns:
        Tuple of (output_tokens, exit_code)

    Raises:
        TimeoutError: If max_steps exceeded
        ValueError: If bytecode is invalid

    Example:
        >>> runner = AutoregressiveVMRunner()
        >>> bytecode, data = compile_c("int main() { return 42; }")
        >>> output, code = runner.run(bytecode, data)
        >>> assert code == 42
    """
```

### 5.3 Generate API Docs
```bash
# Use pdoc or sphinx
pdoc --html neural_vm -o docs/api/
```

---

## Phase 6: CI/CD (Priority: Low)

### 6.1 GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -m "not slow" --tb=short
```

### 6.2 Add Coverage Badge
```yaml
- run: pytest --cov=neural_vm --cov-report=xml
- uses: codecov/codecov-action@v4
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Code Cleanup | High | Medium | High |
| 2. Quality Tools | High | Low | High |
| 3. Testing Infra | Medium | Medium | Medium |
| 4. Debug Infra | Medium | High | High |
| 5. Documentation | Low | Medium | Medium |
| 6. CI/CD | Low | Low | Medium |

---

## Quick Wins (Do First)

1. **Archive debug scripts** - Immediate cleanup
2. **Add pyproject.toml** - Modern Python config
3. **Run ruff --fix** - Auto-fix simple issues
4. **Add .pre-commit-config.yaml** - Prevent future issues

---

## Success Metrics

- [ ] Zero ruff/mypy errors
- [ ] 80%+ test coverage
- [ ] All 1096 tests passing
- [ ] < 10 files in root directory
- [ ] All public APIs documented
- [ ] CI passing on every commit
