"""Migration shims: wrap existing `_set_layerN_*` functions as compiler Operations.

Phase 0 M3 (2026-05-09): rather than rewriting every weight-setting function from
scratch, this module wraps them as `Operation` instances usable by the LayerCompiler.

The wrapped ops:
- Read `dim_positions` from the compiler (which matches `_SetDim` for compat)
- Call the original `_set_layerN_*` function
- Declare their reads/writes so the compiler can do dependency analysis

This lets the LayerCompiler drive layer assignment based on the actual data
dependencies rather than the hardcoded `model.blocks[N]` indexing in
`set_vm_weights`.

End-state vision: every `_set_layerN_*` function in vm_step.py becomes a wrapped
Operation here. The compiler then auto-allocates layers, and `set_vm_weights`
becomes a thin wrapper around the compiler. Today this module migrates a small
subset as proof-of-concept; the rest is iterative work.
"""

from typing import Dict
import torch.nn as nn

from .layer_compiler import Operation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_setdim_proxy(dim_positions: Dict[str, int]):
    """Build an object that mimics _SetDim using compiler dim positions.

    The original `_set_layerN_*` functions reference `BD.MARK_PC`, `BD.OUTPUT_LO`
    etc. We need to give them a BD-like object whose attribute values are the
    integer positions that the compiler chose.

    Returns an object where `proxy.MARK_PC == dim_positions['MARK_PC']`.
    """
    class _Proxy:
        pass
    proxy = _Proxy()
    # Copy ALL _SetDim class attributes that are integers, then override with
    # compiler positions where declared. This way, references to undeclared
    # dims (e.g., NUM_OPCODES) still work, while declared dims use compiler
    # positions.
    from ..vm_step import _SetDim
    for name in dir(_SetDim):
        if name.startswith('_'):
            continue
        val = getattr(_SetDim, name)
        if isinstance(val, int):
            setattr(proxy, name, val)
    # Override with compiler-declared positions
    for name, pos in dim_positions.items():
        setattr(proxy, name, pos)
    return proxy


# ---------------------------------------------------------------------------
# Migrated operations
# ---------------------------------------------------------------------------

def make_phase_a_ffn_op() -> Operation:
    """Step-structure FFN: detect marker transitions and emit NEXT_* flags.

    Originally: `_set_phase_a_ffn` at vm_step.py:2872. Lives at L0 in the
    hand-set layout.

    Reads H0/H1/H2/H3/H4 threshold-head outputs (per marker type).
    Writes NEXT_PC, NEXT_AX, NEXT_SP, NEXT_BP, NEXT_STACK0, NEXT_MEM, NEXT_SE.
    """
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_phase_a_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_phase_a_ffn(ffn, S, proxy)

    # The threshold heads write 7 dims each (one per marker type), so we
    # express reads as the head-base names; the FFN reads any element in the
    # H0..H4 ranges, which are size-7 dims.
    return Operation(
        name="phase_a_ffn",
        reads={"H0", "H1", "H2", "H3", "H4"},
        writes={"NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer1_ffn_op() -> Operation:
    """L1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags from threshold differences.

    Originally: `_set_layer1_ffn` at vm_step.py:2922.

    Reads L1H0/L1H1/L1H2/L1H4/H0/H1 threshold outputs and IS_BYTE.
    Writes STACK0_BYTE0, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3.
    """
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer1_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer1_ffn(ffn, S, proxy)

    return Operation(
        name="layer1_ffn",
        reads={"L1H0", "L1H1", "L1H2", "L1H4", "H0", "H1", "IS_BYTE"},
        writes={"STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
                "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer3_ffn_op() -> Operation:
    """L3 FFN: PC/SP/BP first-step defaults + PC byte-0 increment.

    Originally: `_set_layer3_ffn` at vm_step.py:3351.

    Reads MARK_PC, MARK_SP, MARK_BP, MARK_STACK0, HAS_SE, EMBED_LO/HI,
    H1, H4, BYTE_INDEX_*, OP_LEV.
    Writes OUTPUT_LO/HI, EMBED_LO/HI.
    """
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer3_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer3_ffn(ffn, S, proxy)

    return Operation(
        name="layer3_ffn",
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO", "EMBED_HI", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI"},
        kind="ffn",
        bake_fn=bake,
    )


# ---------------------------------------------------------------------------
# Dim spec compatible with _SetDim
# ---------------------------------------------------------------------------

def declare_setdim_compat_dims(compiler) -> None:
    """Declare to a LayerCompiler all dims that match the existing _SetDim layout.

    Every dim has its size and a pinned position equal to its `_SetDim` value
    (so the compiler outputs the same positions). This is the "backward-compat
    mode" for the migration: existing weight-setting code can run unchanged
    against compiler-driven layer assignment.

    For real auto-allocation, drop this function and declare dims with sizes
    only — the compiler will pack them via bump-pointer or AutoAllocator.
    """
    from ..vm_step import _SetDim
    from ..constants import INSTR_WIDTH  # noqa: F401 (touched for completeness)

    # Single-dim flags
    one_dim = [
        "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
        "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
        "MARK_SE_ONLY", "MARK_STACK0",
        "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "CMP_GROUP",
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
        "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
        "IO_IS_PUTCHAR", "IO_OUTPUT_READY",
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
        "OP_SI", "OP_SC", "OP_PSH",
        "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_NE", "OP_LT",
        "OP_GT", "OP_LE", "OP_GE", "OP_SHL", "OP_SHR",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_EXIT", "OP_NOP", "OP_PUTCHAR", "OP_GETCHAR",
        "MEM_STORE", "MEM_ADDR_SRC",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP", "MEM_EXEC",
    ]
    # 7-dim threshold head outputs (one per marker type)
    seven_dim = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
                 "L1H0", "L1H1", "L1H2", "L1H4", "L2H0"]
    # 16-dim nibble groups
    sixteen_dim = ["EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
                   "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
                   "FETCH_LO", "FETCH_HI", "MUL_ACCUM", "DIV_STAGING",
                   "AX_FULL_LO", "AX_FULL_HI",
                   "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                   "ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                   "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI",
                   "FORMAT_PTR_LO", "FORMAT_PTR_HI",
                   "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"]
    four_dim = ["CARRY"]
    eight_dim = ["CMP"]
    forty_eight_dim = ["ADDR_KEY"]
    thirty_two_dim = ["TEMP"]

    for name in one_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 1)
    for name in seven_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 7)
    for name in sixteen_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 16)
    for name in four_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 4)
    for name in eight_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 8)
    for name in forty_eight_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 48)
    for name in thirty_two_dim:
        if hasattr(_SetDim, name):
            compiler.declare_dim(name, 32)
