# Archive Plan: BakedC4Transformer and Legacy Code

## Status: BakedC4Transformer is Already Archived

The `BakedC4Transformer` class has already been moved to archive and is a legacy wrapper.

**Current Status** (as of `src/baked_c4.py`):
```python
"""
MOVED TO src/archive/baked_c4.py

BakedC4Transformer is a legacy wrapper around C4TransformerVM.
The current implementation is AutoregressiveVMRunner in neural_vm/run_vm.py.

For backward compatibility, import from archive:
    from src.archive.baked_c4 import BakedC4Transformer
"""

from src.archive.baked_c4 import BakedC4Transformer
```

## Current Architecture

### ✅ Active (Primary Implementation)

**Core VM**:
- `neural_vm/vm_step.py` - `AutoregressiveVM` (transformer model)
- `neural_vm/run_vm.py` - `AutoregressiveVMRunner` (execution loop)
- `neural_vm/batch_runner.py` - `BatchRunner` (batch execution)
- `neural_vm/speculative.py` - `SpeculativeRunner` (draft model + verification)

**Supporting Modules**:
- `neural_vm/embedding.py` - Token/opcode definitions
- `neural_vm/alu/` - ALU operations (layers 8-10)
- `neural_vm/kv_cache.py` - KV cache for efficient generation
- `src/compiler.py` - C to bytecode compiler
- `src/io_support.py` - I/O token handling

### ⚠️ Legacy (Archived or Should Be)

**Already in Archive**:
- `src/archive/baked_c4.py` - `BakedC4Transformer` (old wrapper)
- `src/archive/prompt_baking_v1.py` - Old prompt baking approach
- `neural_vm/archive/baked_program.py` - Old program baking
- `neural_vm/archive/baked_attention.py` - Old attention implementation

**Using Legacy (Should Update or Archive)**:

Test files using `BakedC4Transformer`:
```
test_kv_cache.py                    - KV cache tests (March 27)
test_init_only.py                   - Init tests (March 27)
test_validation_*.py                - Validation tests (March 27)
test_bytecode_check.py              - Bytecode validation
show_validation_config.py           - Config display
test_before_after.py                - Before/after comparison
```

Debug/exploration files (root directory):
```
debug_*.py                          - ~50+ debug scripts
test_*.py                           - ~30+ test scripts
quick_*.py                          - Quick test scripts
run_*.py                            - One-off run scripts
```

## Archival Strategy

### Phase 1: Organize Active Tests ✅

**Keep Active** (move to `tests/` if not already there):
- `tests/test_vm.py` - Main VM tests
- `tests/test_programs.py` - Program execution tests
- `tests/test_suite_1000.py` - 1000 random program tests
- `tests/run_1000_tests.py` - Test runner
- `neural_vm/tests/test_opcodes.py` - Opcode unit tests
- `neural_vm/tests/test_opcodes_fast.py` - Fast opcode tests
- `neural_vm/tests/test_dim_registry.py` - Dimension registry tests

### Phase 2: Archive Legacy Tests

**Create**: `tests/archive/legacy_baked_c4/`

**Move these BakedC4Transformer tests**:
```bash
mkdir -p tests/archive/legacy_baked_c4
mv test_kv_cache.py tests/archive/legacy_baked_c4/
mv test_init_only.py tests/archive/legacy_baked_c4/
mv test_validation_*.py tests/archive/legacy_baked_c4/
mv test_bytecode_check.py tests/archive/legacy_baked_c4/
mv test_before_after.py tests/archive/legacy_baked_c4/
mv show_validation_config.py tests/archive/legacy_baked_c4/
```

**Add README**:
```
tests/archive/legacy_baked_c4/README.md:

# Legacy BakedC4Transformer Tests

These tests use the old BakedC4Transformer wrapper.

**Current Implementation**: AutoregressiveVMRunner in neural_vm/run_vm.py

**Status**: Archived March 2024
- Tests may still run (BakedC4Transformer redirects to archive)
- Not actively maintained
- Use tests/ directory for new tests with AutoregressiveVM

**Migration**: To use current implementation:
```python
# Old:
from src.baked_c4 import BakedC4Transformer
c4 = BakedC4Transformer()

# New:
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
model = AutoregressiveVM()
set_vm_weights(model)
runner = AutoregressiveVMRunner(model=model)
```
```

### Phase 3: Archive Debug Scripts

**Create**: `debug/archive/`

These are one-off exploration scripts from development:
```bash
mkdir -p debug/archive
mv debug_*.py debug/archive/ 2>/dev/null
mv test_psh_*.py debug/archive/ 2>/dev/null
mv test_spec_*.py debug/archive/ 2>/dev/null
mv test_quick_*.py debug/archive/ 2>/dev/null
mv test_step*.py debug/archive/ 2>/dev/null
# ... (50+ debug scripts)
```

**Keep active**:
- `verify_imm_fix.py` - Recent (April 2024)
- `verify_neural_imm_complete.py` - Recent (April 2024)
- Any debug scripts referenced in recent docs

### Phase 4: Clean Root Directory

**Current Root** (too cluttered):
```
c4_release/
├── debug_*.py          (~50 files)
├── test_*.py           (~30 files)
├── run_*.py            (~10 files)
├── quick_*.py          (~5 files)
├── check_*.py          (~3 files)
├── mandelbrot.png
├── mandelbrot_native
└── ... (actual code)
```

**Goal**:
```
c4_release/
├── neural_vm/          # Core VM implementation
├── src/                # Compiler, runtime support
├── tests/              # Active test suite
├── docs/               # Documentation
├── examples/           # Example programs
├── tools/              # Utility scripts
├── debug/              # Active debug scripts
│   └── archive/        # Old debug scripts
└── archive/            # Archived code
    └── legacy_tests/   # Old test files
```

### Phase 5: Update Documentation

**Create**: `docs/MIGRATION_GUIDE.md`

Guide for moving from BakedC4Transformer to AutoregressiveVM:
- API differences
- Feature mapping
- Migration examples
- Breaking changes

## Recommended Commands

### Step 1: Archive Old Tests
```bash
# Create archive directories
mkdir -p tests/archive/legacy_baked_c4
mkdir -p debug/archive

# Move BakedC4Transformer tests
mv test_kv_cache.py test_init_only.py test_validation_*.py \
   test_bytecode_check.py test_before_after.py show_validation_config.py \
   tests/archive/legacy_baked_c4/

# Create README
cat > tests/archive/legacy_baked_c4/README.md << 'EOF'
# Legacy BakedC4Transformer Tests (Archived March 2024)

Current implementation: AutoregressiveVMRunner in neural_vm/run_vm.py
Status: Not actively maintained, use tests/ for new tests
EOF
```

### Step 2: Archive Debug Scripts
```bash
# Move old debug scripts to archive
mv debug_*.py test_psh_*.py test_spec_*.py test_quick_*.py \
   test_step*.py debug/archive/ 2>/dev/null

# Keep recent verification scripts
mv debug/archive/verify_imm_fix.py . 2>/dev/null
mv debug/archive/verify_neural_imm_complete.py . 2>/dev/null

# Create README
cat > debug/archive/README.md << 'EOF'
# Archived Debug Scripts

One-off exploration scripts from Neural VM development.
Historical record - not actively maintained.
EOF
```

### Step 3: Verify Active Code Still Works
```bash
# Run main test suites
python -m pytest tests/test_vm.py -v
python -m pytest neural_vm/tests/test_opcodes.py -v
python tests/run_1000_tests.py

# Run recent verification
python verify_neural_imm_complete.py
```

## What NOT to Archive

### Keep Active:

1. **Core Implementation**:
   - `neural_vm/` - entire directory
   - `src/compiler.py`, `src/io_support.py`, `src/speculator.py`

2. **Active Tests**:
   - `tests/` - entire directory (except archive/)
   - Any test with recent modifications

3. **Documentation**:
   - `docs/` - entire directory
   - `*.md` files in root (README, status docs)

4. **Recent Work**:
   - `verify_*.py` - recent verification scripts
   - `IMM_*.md` - recent implementation docs
   - `HANDLER_STATUS.md`, `REMAINING_HANDLERS_PLAN.md`

5. **Active Tools**:
   - `tools/` - utility scripts
   - `bundler/` - bundling scripts

## Migration Path for Users

If you have code using `BakedC4Transformer`:

### Before:
```python
from src.baked_c4 import BakedC4Transformer

c4 = BakedC4Transformer(use_speculator=True)
result = c4.run(bytecode, data=b'', argv=[])
```

### After:
```python
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

# One-time setup
model = AutoregressiveVM()
set_vm_weights(model)
runner = AutoregressiveVMRunner(model=model)

# Run programs
result = runner.run(bytecode, data=b'', argv=[])
```

### With Speculative Execution:
```python
from neural_vm.speculative import SpeculativeRunner

runner = SpeculativeRunner()  # Auto-creates model + draft
result = runner.run(bytecode, data=b'', argv=[])
```

## Benefits of Archiving

1. **Clearer Repository Structure**:
   - Root directory has only essential files
   - Tests organized by type (active vs legacy vs debug)
   - Easy to find current implementation

2. **Historical Record**:
   - Old code preserved for reference
   - Can still run if needed (backward compatibility)
   - README explains why archived

3. **Development Focus**:
   - New contributors see current architecture
   - Less confusion about which code to use
   - Clear migration path documented

## Timeline

- **April 9, 2024**: BakedC4Transformer already archived (redirects)
- **April 9, 2024**: IMM neural implementation completed
- **Next**: Archive old tests and debug scripts (optional, low priority)
- **Future**: Archive when cleaning up for release

## Conclusion

**Current State**: BakedC4Transformer is already archived ✓
- The class redirects to `src/archive/baked_c4.py`
- Current implementation is `AutoregressiveVMRunner`
- Old tests still work (backward compatibility)

**Recommendation**:
- Leave as-is for now (working backward compatibility)
- Archive old test files when preparing for release
- Focus effort on removing remaining handlers (JSR/ENT/LEV)

**Priority**: Low
- Not blocking any development
- Archive cleanup can wait
- More important: complete neural implementation
