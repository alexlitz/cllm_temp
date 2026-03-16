# c4_release Folder Structure

This document describes the organization of the Neural VM (C4-compatible transformer VM).

## Core Directories

### `neural_vm/` - Neural VM Module (Primary Implementation)

The main Python module containing all neural network layers and operations.

```
neural_vm/
├── __init__.py           # Module exports
├── embedding.py          # E class: embedding slots, opcodes, dimensions
├── base_layers.py        # PureFFN, PureAttention base classes
├── pure_alu.py           # PureALU: main ALU (nn.Sequential, no Python control)
├── pure_moe.py           # SoftMoEFFN: Mixture-of-Experts routing
│
├── arithmetic_ops.py     # ADD, SUB operations
├── bitwise_ops.py        # AND, OR, XOR operations
├── comparison_ops.py     # EQ, NE, LT, GT, LE, GE operations
├── shift_ops.py          # SHL, SHR operations
├── mul_div_ops.py        # MUL, DIV, MOD operations
├── newton_raphson_div.py # Newton-Raphson division layers
├── schoolbook_mul.py     # Schoolbook multiplication
│
├── carry_ffn.py          # Carry propagation (flattened FFN)
├── reduce_ffn.py         # Cross-nibble reduce/broadcast FFNs
├── variable_shift_ops.py # Variable shift operations
├── zero_detection.py     # Zero detection for BZ/BNZ
│
├── io_ops.py             # GETCHAR, PUTCHAR, EXIT
├── io_handler.py         # I/O streaming and tool-use handlers
├── memory_ops.py         # LI, LC, SI, SC (load/store)
├── memory_subroutines.py # MSET, MCMP, MCPY subroutines
├── bump_allocator.py     # MALLOC, FREE (bump allocation)
├── missing_ops.py        # Additional opcode implementations
│
├── control_flow_ops.py   # JMP, JSR, BZ, BNZ, ENT, ADJ, LEV
├── c4_stack_ops.py       # Stack frame operations
├── position_encoding.py  # ALiBi/RoPE position encoding
├── cascaded_binary_io.py # ALiBi-based input position extraction
│
├── kv_cache_eviction.py  # KV cache with eviction policy
├── neural_state.py       # VM state, memory attention
├── token_sequence_vm.py  # Token-based VM execution
├── baked_program.py      # Program embedding in weights
├── baked_attention.py    # Attention with baked weights
│
├── sparse_layers.py      # Sparse weight representations
├── optimized_ops.py      # Multi-head gather optimizations
├── purity_checker.py     # Verify no Python control flow
├── onnx_export.py        # ONNX export functionality
│
├── tests/                # Test suite
│   ├── test_c4_ops.py    # Main opcode tests (77 tests)
│   ├── test_32bit_ops.py # 32-bit arithmetic (86 tests)
│   └── ...
│
└── alternative_implementations/
    ├── attention_based_ops.py  # Old attention-based cross-nibble ops
    └── rope_binary_io.py       # RoPE-based position encoding
```

### `docs/` - Documentation

Extended documentation and diagrams.

### `test_programs/` - C Test Programs

Sample C programs for testing the VM.

### `models/` - Saved Models

Exported model weights and ONNX files.

## Core Files (Root)

### Documentation
- `README.md` - Project overview
- `DOCUMENT_FIXES.md` - Technical notes and implementation details
- `OPCODE_TABLE.md` - Complete opcode reference with weight counts
- `TOOLUSE.md` - Tool-use I/O protocol
- `FOLDER_STRUCTURE.md` - This file

### Python Scripts
- `run_program.py` - Run C programs through VM
- `run_eliza_tooluse.py` - Eliza demo with tool-use I/O
- `bundle_executable.py` - Bundle C4 programs for deployment
- `llm_io.py` - LLM streaming I/O protocol
- `chat_io.py` - Chat-style I/O handler
- `tooluse_io.py` - Tool-use I/O protocol
- `tooluse_c_handler.py` - C handler for tool calls
- `neural_bundle_fixedpoint.py` - Fixed-point neural bundling

### C Files
- `onnx_runner_c4.c` - ONNX runtime in C (C4 compatible)
- `tooluse_onnx_c4.c` - Tool-use ONNX runner
- `bundle_c4.c` - C4 bundler
- `c4_neural.c` - Neural VM in C
- `neural_quine_v2.c` - Neural quine (transformer forward = source output)
- `neural_quine_transformer.c` - Full transformer quine
- `test_llm_ops.c` - LLM ops test
- `complex_tests_c4.c` - Complex C4 tests
- `mandelbrot_color_png_c4.c` - Mandelbrot demo

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     PureALU (nn.Sequential)                 │
│                                                             │
│  Input: [batch, 8 nibbles, 160 dims]                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Stage 1: Raw Computation (FFN)                      │    │
│  │   ADD: AddRawSumFFN                                 │    │
│  │   SUB: SubRawDiffFFN                                │    │
│  │   MUL: SchoolbookMulLayers                          │    │
│  │   DIV: NewtonRaphsonDivLayers                       │    │
│  │   CMP: CompareDiffFFN                               │    │
│  │   ...                                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Stage 2: Carry Propagation (CarryPropagateFFN)      │    │
│  │   Flattened [batch, 8*160] for cross-nibble routing │    │
│  │   10 iterations for cascading carries               │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Stage 3: Reduction (CompareReduceEqFFNNew, etc.)    │    │
│  │   Sum per-nibble results to position 0              │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  Output: [batch, 8 nibbles, 160 dims]                       │
│  Results in RESULT slot (E.RESULT)                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Pure Neural**: All operations via FFN/Attention forward passes
   - No Python control flow in `forward()` methods
   - Weights are **baked** (hardcoded), NOT trained

2. **Baked Weights**: `_bake_weights()` sets specific matrix entries
   - Example: `W_up[0, E.NIB_A] = SCALE` enables ADD operation
   - No gradient updates, no training loop

3. **ONNX Compatible**: `SoftMoEFFN` uses static indices
   - All experts run, opcode one-hot selects output
   - Exportable via `neural_vm.onnx_export.export_to_onnx()`

4. **C4 Compatible**: Full C4 compiler opcode support
   - 46 opcodes: arithmetic, bitwise, comparison, shift, memory, I/O
   - Run C programs compiled with c4 compiler

## Testing

```bash
# Run all tests
python -m neural_vm.tests.run_all_tests

# Run specific tests
python -m neural_vm.tests.test_c4_ops      # 77 tests
python -m neural_vm.tests.test_32bit_ops   # 86 tests
python -m neural_vm.tests.test_memory_subs # Memory subroutines

# Export to ONNX
python -m neural_vm.onnx_export --verify --compare
```

## Deprecated Files

Moved to `../old/`:
- `pure_gen_vm_v7.py` - Old standalone implementation (now in `neural_vm/`)
- `test_v5_comprehensive.py`, `test_v7_comprehensive.py` - Old test files
- `alibi_io.py`, `binary_position_io.py`, `multihead_position_io.py` - Old I/O
- `sparse_vm.py`, `flop_analysis.py` - Analysis scripts
