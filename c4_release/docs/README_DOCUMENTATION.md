# Documentation Index

**Purpose**: Central index of all documentation in the C4 Neural VM project

## Quick Navigation

### Core Architecture
- [README.md](README.md) - Main architecture overview
- [OPCODE_TABLE.md](OPCODE_TABLE.md) - Complete opcode reference
- [C4_NATIVE_VM.md](C4_NATIVE_VM.md) - Native C4 VM implementation

### Weight Setting & Compilation
- [WEIGHT_SETTING_APPROACHES.md](WEIGHT_SETTING_APPROACHES.md) - **NEW** Hand-set vs compiled weights comparison
- [WEIGHT_COMPILER_DESIGN.md](WEIGHT_COMPILER_DESIGN.md) - Compiler design philosophy
- [WEIGHT_COMPILER_PRIMITIVES.md](WEIGHT_COMPILER_PRIMITIVES.md) - Primitive operations
- [GRAPH_WEIGHT_COMPILER.md](GRAPH_WEIGHT_COMPILER.md) - Graph-based compilation
- [NEURAL_COMPILER.md](NEURAL_COMPILER.md) - Neural compilation approach

### Testing & Validation
- [MEMORY_TEST_COVERAGE.md](MEMORY_TEST_COVERAGE.md) - **NEW** Memory test analysis
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Testing checklist
- [CHECKLIST_STATUS.md](CHECKLIST_STATUS.md) - Current status
- [C4_OPCODE_STATUS.md](C4_OPCODE_STATUS.md) - Opcode implementation status

### Specialized Features
- [IO_ATTENTION_MECHANISM.md](IO_ATTENTION_MECHANISM.md) - I/O handling
- [KV_CACHE_EVICTION.md](KV_CACHE_EVICTION.md) - Cache eviction strategies
- [EVICTION_ALGORITHM.md](EVICTION_ALGORITHM.md) - Eviction algorithm details
- [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) - Speculative execution
- [NEURAL_QUINE.md](NEURAL_QUINE.md) - Self-replicating programs

### Optimization & Efficiency
- [COMPUTATIONAL_EFFICIENCY.md](COMPUTATIONAL_EFFICIENCY.md) - Efficiency analysis
- [SPARSE_EFFICIENCY.md](SPARSE_EFFICIENCY.md) - Sparse matrix optimization
- [ALU_OPTIMIZATION_ANALYSIS.md](ALU_OPTIMIZATION_ANALYSIS.md) - ALU optimizations
- [ALU_CHUNK_CONFIGS.md](ALU_CHUNK_CONFIGS.md) - ALU configuration
- [PRECISION_ADAPTATION.md](PRECISION_ADAPTATION.md) - Numerical precision

### Export & Deployment
- [ONNX_EXPORT.md](ONNX_EXPORT.md) - ONNX export guide
- [BUNDLER_GUIDE.md](BUNDLER_GUIDE.md) - Creating standalone executables
- [PROGRAM_BAKING.md](PROGRAM_BAKING.md) - Baking programs into weights
- [BAKED_COMPILER.md](BAKED_COMPILER.md) - Compiler baked into model

### Examples & Guides
- [MANDELBROT_EXAMPLE.md](MANDELBROT_EXAMPLE.md) - Mandelbrot set renderer
- [ARGV_SETUP.md](ARGV_SETUP.md) - Command-line argument handling
- [TOOL_CALLING.md](TOOL_CALLING.md) - Tool use integration

### Project Management
- [DOCUMENT_FIXES.md](DOCUMENT_FIXES.md) - Documentation fixes log
- [DOCUMENT_CHECKLISTS.md](DOCUMENT_CHECKLISTS.md) - Documentation checklists
- [POTENTIAL_PROJECTS.md](POTENTIAL_PROJECTS.md) - Future project ideas

### Advanced Topics
- [SELF_HOSTING.md](SELF_HOSTING.md) - Self-hosting C4 compiler
- [QUINE.md](QUINE.md) - Quine implementation
- [VANILLA_GENERATION.md](VANILLA_GENERATION.md) - Standard generation mode
- [SOFTMAX_TOGGLE_PLAN.md](SOFTMAX_TOGGLE_PLAN.md) - Softmax configuration

## Recent Additions (2026-04-07)

### New Documentation

1. **WEIGHT_SETTING_APPROACHES.md** - Comprehensive comparison of weight setting methods
   - Hand-set weights: ~2,000 lines (including helpers)
   - Compiled weights: 3,704 lines
   - Code examples, use cases, migration path

2. **MEMORY_TEST_COVERAGE.md** - Memory test analysis and gaps
   - Current test coverage analysis
   - Identified gaps in memory testing
   - Recommendations for new tests
   - Memory mechanism implementation details

## Documentation by Category

### For New Contributors

**Start here**:
1. [README.md](README.md) - Understand the architecture
2. [OPCODE_TABLE.md](OPCODE_TABLE.md) - Learn the instruction set
3. [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Run tests
4. [WEIGHT_SETTING_APPROACHES.md](WEIGHT_SETTING_APPROACHES.md) - Understand weight configuration

### For Researchers

**Architectural deep dives**:
- [NEURAL_COMPILER.md](NEURAL_COMPILER.md)
- [IO_ATTENTION_MECHANISM.md](IO_ATTENTION_MECHANISM.md)
- [COMPUTATIONAL_EFFICIENCY.md](COMPUTATIONAL_EFFICIENCY.md)
- [WEIGHT_COMPILER_DESIGN.md](WEIGHT_COMPILER_DESIGN.md)

### For Developers

**Implementation guides**:
- [BUNDLER_GUIDE.md](BUNDLER_GUIDE.md)
- [ONNX_EXPORT.md](ONNX_EXPORT.md)
- [PROGRAM_BAKING.md](PROGRAM_BAKING.md)
- [MANDELBROT_EXAMPLE.md](MANDELBROT_EXAMPLE.md)

### For Testing

**Test-related documentation**:
- [MEMORY_TEST_COVERAGE.md](MEMORY_TEST_COVERAGE.md) - Memory tests
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Test checklist
- [C4_OPCODE_STATUS.md](C4_OPCODE_STATUS.md) - Opcode status
- [`../neural_vm/tests/README_MEMORY_TESTS.md`](../neural_vm/tests/README_MEMORY_TESTS.md) - Memory stress tests

## Documentation Standards

### File Naming
- Use UPPERCASE_WITH_UNDERSCORES.md for documentation files
- Use README.md for directory-specific guides
- Use lowercase_with_underscores.md for code-related docs

### Structure
- Start with # Title
- Include "Date" and "Status" if applicable
- Use ## for major sections
- Include code examples with syntax highlighting
- Add cross-references to related docs

### Updates
- Update DOCUMENT_FIXES.md when fixing documentation
- Add entries to this index when creating new docs
- Keep status markers (✅ ⚠️ ❌) consistent

## Test Documentation Location

**Test-specific docs** are in the test directories:
- `neural_vm/tests/README_MEMORY_TESTS.md` - Memory stress test guide
- `tests/STATUS.md` - Test suite status
- `tests/COMPACTION_TEST_PLAN.md` - Compaction tests

## Contributing to Documentation

### Adding New Documentation

1. Create file in appropriate location (`docs/` for architecture, test dirs for tests)
2. Follow naming conventions
3. Add entry to this index
4. Update DOCUMENT_FIXES.md if fixing existing docs
5. Cross-reference related documents

### Updating Existing Documentation

1. Make changes
2. Update "Date" field if present
3. Add entry to DOCUMENT_FIXES.md
4. Check cross-references still valid

## Auto-Generated Documentation

Some documentation is auto-generated:
- ALU configuration tables
- Opcode reference tables
- Weight dimension mappings

See individual files for generation scripts.

## External Resources

- [C4 Compiler Original](https://github.com/rswier/c4) - Original C4 implementation
- [Anthropic Research](https://www.anthropic.com/research) - Transformer research
- [ONNX Documentation](https://onnx.ai/onnx/) - ONNX format reference

## Documentation TODOs

- [ ] Add tutorial for creating custom opcodes
- [ ] Document weight initialization strategies
- [ ] Add performance benchmarking guide
- [ ] Create troubleshooting guide
- [ ] Document KV cache strategies in detail

## Recently Moved Files

Files moved from root to `docs/` (2026-04-07):
- `WEIGHT_SETTING_APPROACHES.md` → `docs/WEIGHT_SETTING_APPROACHES.md`
- `MEMORY_TEST_COVERAGE_ANALYSIS.md` → `docs/MEMORY_TEST_COVERAGE.md`

## Quick Reference Card

### Running Tests
```bash
# All tests
pytest neural_vm/tests/ tests/ -v

# Memory stress tests
pytest neural_vm/tests/test_memory_stress.py -v

# Opcode tests
pytest neural_vm/tests/test_opcodes.py -v
```

### Building Documentation
```bash
# Generate opcode table
python tools/generate_opcode_table.py > docs/OPCODE_TABLE.md

# Update status
python tools/check_opcode_status.py > docs/C4_OPCODE_STATUS.md
```

### Exporting Model
```bash
# Export to ONNX
python tools/export_onnx.py --output model.onnx

# Bundle executable
python bundler/bundle_onnx_standard.py
```

---

**Last Updated**: 2026-04-07
**Maintainer**: See git history
