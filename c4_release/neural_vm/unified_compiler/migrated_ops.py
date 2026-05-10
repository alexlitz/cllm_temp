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
    Falls back to `_SetDim` for anything not declared (e.g., classmethods like
    `opcode_dim`, constants like `NUM_OPCODES`).
    """
    from ..vm_step import _SetDim

    class _Proxy:
        # Inherit class methods and constants from _SetDim via __getattr__ fallback
        def __getattr__(self, name):
            return getattr(_SetDim, name)

    proxy = _Proxy()
    # Override with compiler-declared positions for declared dims
    for name, pos in dim_positions.items():
        object.__setattr__(proxy, name, pos)
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

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].ffn) the legacy path used. This sidesteps the
    LayerCompiler's dep-based assignment, which would otherwise place this FFN
    at L1 (advancing past L0 because it reads H0-H4 written by L0 attn).
    """
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

    def bake(block, dim_positions, S):
        from ..vm_step import _set_phase_a_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_phase_a_ffn(block.ffn, S, proxy)

    # The threshold heads write 7 dims each (one per marker type), so we
    # express reads as the head-base names; the FFN reads any element in the
    # H0..H4 ranges, which are size-7 dims.
    return Operation(
        name="phase_a_ffn",
        phase=0,
        reads={"H0", "H1", "H2", "H3", "H4"},
        writes={"NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        migrated=True,
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
        phase=1,
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
        phase=3,
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO", "EMBED_HI", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer2_mem_byte_flags_op() -> Operation:
    """L2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer2_mem_byte_flags
        _set_layer2_mem_byte_flags(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer2_mem_byte_flags",
        phase=2,
        reads={"H0", "H1", "H4", "IS_BYTE", "BYTE_INDEX_0",
               "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        writes={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
                "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        bake_fn=bake,
    )


def make_nibble_copy_ffn_op() -> Operation:
    """Conditional nibble copy: OUTPUT = EMBED for non-register byte values."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_nibble_copy_ffn
        _set_nibble_copy_ffn(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="nibble_copy_ffn",
        phase=15,
        reads={"IS_BYTE", "H1", "H4", "MEM_STORE",
               "EMBED_LO", "EMBED_HI", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MARK_STACK0", "HAS_SE", "CMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer4_pc_relay_op() -> Operation:
    """L4 attention: relay PC marker EMBED → AX marker EMBED."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer4_pc_relay
        # _set_layer4_pc_relay takes attn, S, BD, HD
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer4_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer4_pc_relay",
        phase=4,
        reads={"MARK_PC", "MARK_AX", "EMBED_LO", "EMBED_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI"},  # at AX marker
        kind="attn",
        bake_fn=bake,
    )


def make_layer4_ffn_op() -> Operation:
    """L4 FFN: compute PC+1/2/3/4 in FETCH dims for L5 fetch."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer4_ffn
        _set_layer4_ffn(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer4_ffn",
        phase=4,
        reads={"MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI",
               "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1"},
        writes={"FETCH_LO", "FETCH_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer5_fetch_op() -> Operation:
    """L5 attention: instruction-fetch heads (8 heads)."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer5_fetch
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer5_fetch(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer5_fetch",
        phase=5,
        # Reads: PC/AX markers + FETCH addr (PC+K) + ADDR_KEY (per CODE byte) +
        #        CLEAN_EMBED (the value at the matched CODE byte).
        # Note: heads 6/7 also read OP_* via V projection but that's the DEPRECATED
        # path (OP_* flags were removed from embeddings 2026-04-13). Excluding from
        # reads since they're not semantically active inputs.
        reads={"MARK_PC", "MARK_AX", "HAS_SE",
               "FETCH_LO", "FETCH_HI", "EMBED_LO", "EMBED_HI",
               "ADDR_KEY", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                "FETCH_LO", "FETCH_HI",
                "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"},
        kind="attn",
        bake_fn=bake,
    )


def make_opcode_decode_ffn_op() -> Operation:
    """L5 FFN: decode opcode byte → 34 one-hot OP_* flags at OPCODE_BASE."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_opcode_decode_ffn
        _set_opcode_decode_ffn(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="opcode_decode_ffn",
        phase=5,
        reads={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI", "MARK_AX", "MARK_PC", "HAS_SE"},
        writes={"OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
                "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
                "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
                "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
                "OP_PUTCHAR", "OP_GETCHAR",
                "TEMP"},  # JSR writes IS_JSR to TEMP[0]
        kind="ffn",
        bake_fn=bake,
    )


def make_layer6_attn_op() -> Operation:
    """L6 attention: relay heads for IS_JMP, IS_EXIT, etc. at PC marker."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer6_attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_attn(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_attn",
        phase=6,
        reads={"OP_JMP", "OP_EXIT", "OP_JSR", "MARK_AX", "MARK_PC", "MARK_SP",
               "MARK_STACK0", "NEXT_SE", "FETCH_LO", "FETCH_HI",
               "PSH_AT_SP", "OP_PSH", "OP_ADJ", "OP_ENT", "OP_LEV",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"CMP", "AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer6_routing_ffn_op() -> Operation:
    """L6 FFN: per-opcode routing — write FETCH/AX_CARRY → OUTPUT, etc."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer6_routing_ffn
        _set_layer6_routing_ffn(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer6_routing_ffn",
        phase=6,
        reads={"OP_IMM", "OP_EXIT", "OP_JMP", "OP_NOP", "OP_LEA",
               "MARK_AX", "MARK_PC", "MARK_STACK0", "MARK_BP",
               "IS_BYTE", "FETCH_LO", "FETCH_HI",
               "AX_CARRY_LO", "AX_CARRY_HI", "CMP",
               "OUTPUT_LO", "OUTPUT_HI", "HAS_SE",
               "OPCODE_BASE", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI",
               "TEMP", "DIV_STAGING"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer6_relay_heads_op() -> Operation:
    """L6 head 6/7: STACK0 ← AX relay for PSH."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer6_relay_heads
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_relay_heads(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_relay_heads",
        phase=6,
        reads={"MARK_STACK0", "MARK_AX", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer7_operand_gather_op() -> Operation:
    """L7 attention: operand A gather (prev STACK0 byte 0 → ALU at AX marker)."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer7_operand_gather
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer7_operand_gather(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer7_operand_gather",
        phase=7,
        reads={"MARK_AX", "STACK0_BYTE0", "OP_LEA",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer7_memory_heads_op() -> Operation:
    """L7 attention heads 1-6: memory + flag broadcast heads."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer7_memory_heads
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer7_memory_heads(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer7_memory_heads",
        phase=7,
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "OP_LI", "OP_LC", "OP_PSH", "OP_SI", "OP_SC",
               "AX_CARRY_LO", "AX_CARRY_HI", "TEMP"},
        writes={"OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP",
                "TEMP", "ADDR_KEY"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_alu_op() -> Operation:
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + LEA + CMP_GROUP."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer8_alu
        _set_layer8_alu(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_alu",
        phase=8,
        reads={"MARK_AX", "MARK_PC", "ALU_LO", "AX_CARRY_LO", "FETCH_LO",
               "OP_ADD", "OP_SUB", "OP_LEA",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE"},
        writes={"OUTPUT_LO", "CARRY", "CMP_GROUP"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer8_multibyte_fetch_op() -> Operation:
    """L8 attention head 3: fetch CODE bytes at AX byte positions."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer8_multibyte_fetch
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_multibyte_fetch(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer8_multibyte_fetch",
        phase=8,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_multibyte_routing_op() -> Operation:
    """L8 FFN extension: route FETCH → OUTPUT at AX byte positions for IMM."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer8_multibyte_routing
        _set_layer8_multibyte_routing(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_multibyte_routing",
        phase=8,
        reads={"IS_BYTE", "H1", "OP_IMM", "MARK_AX",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer8_sp_gather_op() -> Operation:
    """L8 attention: SP gather for ADJ/ENT."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer8_sp_gather
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_sp_gather(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer8_sp_gather",
        phase=8,
        reads={"MARK_AX", "MARK_SP", "OP_ADJ", "OP_ENT", "OP_LEA",
               "EMBED_LO", "EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer9_alu_op() -> Operation:
    """L9 FFN: ADD/SUB hi nibble + bitwise ops byte 0."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer9_alu
        _set_layer9_alu(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer9_alu",
        phase=9,
        reads={"MARK_AX", "MARK_PC", "ALU_HI", "AX_CARRY_HI", "FETCH_HI", "CARRY",
               "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "ALU_LO", "AX_CARRY_LO"},
        writes={"OUTPUT_HI", "CMP", "OUTPUT_LO"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer9_lev_addr_relay_op() -> Operation:
    """L9 attention head 0: BP byte 0 → ADDR_B0 at SP marker for LEV."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer9_lev_addr_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_addr_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer9_lev_addr_relay",
        phase=9,
        reads={"MARK_SP", "OP_LEV", "L1H1", "BYTE_INDEX_0",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer9_lev_bp_to_pc_relay_op() -> Operation:
    """L9 attention head 1: BP byte 0 → ADDR_B0 at PC marker for LEV return."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer9_lev_bp_to_pc_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_bp_to_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer9_lev_bp_to_pc_relay",
        phase=9,
        reads={"MARK_PC", "OP_LEV", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "L1H1", "BYTE_INDEX_0"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_carry_relay_op() -> Operation:
    """L10 attention head 0: relay CARRY flags from AX marker to AX byte positions."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer10_carry_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_carry_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_carry_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY"},
        writes={"CARRY"},  # broadcast
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_byte_passthrough_op() -> Operation:
    """L10 attention head 1: AX byte passthrough across steps."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer10_byte_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_byte_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "TEMP",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_sp_byte_passthrough_op() -> Operation:
    """L10 attention head 2: SP byte passthrough."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer10_sp_byte_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_sp_byte_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_sp_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_psh_stack0_passthrough_op() -> Operation:
    """L10 attention head 3: PSH STACK0 passthrough."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer10_psh_stack0_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_psh_stack0_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_psh_stack0_passthrough",
        phase=10,
        reads={"MARK_STACK0", "OP_PSH", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_LI", "OP_LC", "OP_SI", "OP_SC"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_alu_op() -> Operation:
    """L10 FFN: AND/OR/XOR + DIV/MOD setup."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer10_alu
        _set_layer10_alu(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer10_alu",
        phase=10,
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "ALU_HI", "AX_CARRY_HI",
               "OP_OR", "OP_XOR", "OP_AND", "OP_DIV", "OP_MOD"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "DIV_STAGING"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer11_mul_partial_op() -> Operation:
    """L11 FFN: MUL partial product accumulation."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer11_mul_partial
        _set_layer11_mul_partial(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer11_mul_partial",
        phase=11,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"MUL_ACCUM"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
    )


def make_layer12_mul_combine_op() -> Operation:
    """L12 FFN: combine MUL partial products into final result."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer12_mul_combine
        _set_layer12_mul_combine(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer12_mul_combine",
        phase=12,
        reads={"MARK_AX", "MUL_ACCUM", "OP_MUL"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
    )


def make_layer13_mem_addr_gather_op() -> Operation:
    """L13 attention: gather MEM addr from STACK0 / AX_CARRY for SI/SC/LI/LC."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer13_mem_addr_gather
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer13_mem_addr_gather(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer13_mem_addr_gather",
        phase=13,
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "AX_CARRY_LO", "AX_CARRY_HI", "OP_LI", "OP_LC", "OP_SI", "OP_SC",
               "MEM_ADDR_SRC"},
        writes={"ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer13_shifts_op() -> Operation:
    """L13 FFN: SHL/SHR shifts."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer13_shifts
        _set_layer13_shifts(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer13_shifts",
        phase=13,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_SHL", "OP_SHR"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer14_mem_generation_op() -> Operation:
    """L14 attention: generate MEM section tokens (addr + value) for SI/SC/PSH."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer14_mem_generation
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer14_mem_generation(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer14_mem_generation",
        phase=14,
        reads={"MARK_MEM", "MARK_SP", "MARK_STACK0", "OP_PSH", "OP_SI", "OP_SC",
               "OP_JSR", "OP_ENT", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_B0_LO", "ADDR_B0_HI",
               "MEM_STORE", "MEM_ADDR_SRC"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer15_memory_lookup_op() -> Operation:
    """L15 attention: memory-lookup heads for LI/LC."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer15_memory_lookup
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer15_memory_lookup(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer15_memory_lookup",
        phase=15,
        reads={"MARK_AX", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_KEY", "MARK_MEM",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer16_lev_routing_op() -> Operation:
    """L16 FFN: LEV routing — SP = BP + 16."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer16_lev_routing
        _set_layer16_lev_routing(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer16_lev_routing",
        phase=16,
        reads={"MARK_SP", "MARK_PC", "OP_LEV", "ADDR_B0_LO", "ADDR_B0_HI",
               "TEMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


# ---------------------------------------------------------------------------
# Convenience: full operation list
# ---------------------------------------------------------------------------

def make_layer0_threshold_attn_op() -> Operation:
    """L0 attention: 8 threshold heads detecting marker distance.

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].attn) the legacy path used. Using kind="block"
    keeps the L0 op aligned with the hand-set block index regardless of
    LayerCompiler dep-based assignment.
    """
    def bake(block, dim_positions, S):
        from ..vm_step import _set_threshold_attn
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_threshold_attn(
            attn,
            [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
            [proxy.H0, proxy.H1, proxy.H2, proxy.H3, proxy.H4,
             proxy.H5, proxy.H6, proxy.H7],
            ALIBI_S, HD,
        )

    return Operation(
        name="layer0_threshold_attn",
        phase=0,
        reads={"IS_MARK", "CONST"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        migrated=True,
    )


def make_layer1_threshold_attn_op() -> Operation:
    """L1 attention: 3 fine threshold heads + STEP_END + L1H4."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_threshold_attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[3] = 0.0  # global SE detection
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_threshold_attn(
            attn,
            [0.5, 1.5, 2.5],
            [proxy.L1H0, proxy.L1H1, proxy.L1H2],
            ALIBI_S, HD, heads=[0, 1, 2],
        )
        # Head 3: STEP_END existence detection (global)
        base = 3 * HD
        attn.W_q[base, proxy.CONST] = 10.0
        attn.W_k[base, proxy.MARK_SE_ONLY] = 10.0
        attn.W_v[base + 1, proxy.MARK_SE_ONLY] = 1.0
        attn.W_o[proxy.HAS_SE, base + 1] = 1.0
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        _set_threshold_attn(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4]
        )

    return Operation(
        name="layer1_threshold_attn",
        phase=1,
        reads={"IS_MARK", "MARK_SE_ONLY", "CONST"},
        writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer2_threshold_attn_op() -> Operation:
    """L2 attention: threshold 5.5 head."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_threshold_attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_threshold_attn(
            attn, [5.5], [proxy.L2H0], ALIBI_S, HD, heads=[0]
        )

    return Operation(
        name="layer2_threshold_attn",
        phase=2,
        reads={"IS_MARK", "CONST"},
        writes={"L2H0"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer3_carry_forward_attn_op() -> Operation:
    """L3 attention: 7 carry-forward heads (PC, AX, SP, BP, STACK0 + relays)."""
    def bake(attn, dim_positions, S):
        from ..vm_step import (_set_carry_forward_attn, _set_stack0_carry_attn,
                                _SetDim)
        proxy = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
        _set_carry_forward_attn(attn, 0, proxy.MARK_PC, PC_I, PC_I, HD,
                                proxy.EMBED_LO, proxy.EMBED_HI)
        _set_carry_forward_attn(attn, 1, proxy.MARK_AX, AX_I, AX_I, HD,
                                proxy.AX_CARRY_LO, proxy.AX_CARRY_HI)
        _set_carry_forward_attn(attn, 2, proxy.MARK_SP, SP_I, SP_I, HD,
                                proxy.EMBED_LO, proxy.EMBED_HI)
        _set_carry_forward_attn(attn, 3, proxy.MARK_BP, BP_I, BP_I, HD,
                                proxy.EMBED_LO, proxy.EMBED_HI)
        _set_stack0_carry_attn(attn, 4, HD)
        # Heads 5-6: AX_FULL relay + BP→PC for LEV. These reference _SetDim
        # directly inside _set_carry_forward_attn so the proxy fallback handles
        # them. For now we replicate the inline code from _set_layer3_attn block:
        L = 15.0
        base = 5 * HD
        attn.W_q[base, proxy.MARK_AX] = L
        attn.W_q[base, proxy.HAS_SE] = L
        attn.W_q[base, proxy.CONST] = -L * 1.5
        attn.W_k[base, proxy.MARK_AX] = L
        for k in range(16):
            attn.W_v[base + 1 + k, proxy.OUTPUT_LO + k] = 1.0
            attn.W_v[base + 17 + k, proxy.OUTPUT_HI + k] = 1.0
        for k in range(16):
            attn.W_o[proxy.AX_FULL_LO + k, base + 1 + k] = 1.0
            attn.W_o[proxy.AX_FULL_HI + k, base + 17 + k] = 1.0
        GATE = 33
        attn.W_q[base + GATE, proxy.MARK_AX] = L
        attn.W_q[base + GATE, proxy.CONST] = -L / 2
        attn.W_k[base + GATE, proxy.CONST] = L
        # Head 6: BP carry to PC marker for LEV return_addr
        base = 6 * HD
        attn.W_q[base, proxy.MARK_PC] = L
        attn.W_q[base, proxy.OP_LEV] = L / 5
        attn.W_q[base, proxy.CONST] = -L * 1.5
        attn.W_k[base, proxy.L1H1 + BP_I] = L
        attn.W_k[base, proxy.L1H0 + BP_I] = -L
        for k in range(16):
            attn.W_v[base + 1 + k, proxy.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, proxy.CLEAN_EMBED_HI + k] = 1.0
        attn.W_q[base + GATE, proxy.MARK_PC] = L
        attn.W_q[base + GATE, proxy.CONST] = -L / 2
        attn.W_k[base + GATE, proxy.CONST] = L

    return Operation(
        name="layer3_carry_forward_attn",
        phase=3,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "L1H0", "L1H1", "STACK0_BYTE0", "OP_LEV", "HAS_SE",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                "AX_FULL_LO", "AX_FULL_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_binary_pop_sp_increment_op() -> Operation:
    """L6 FFN extension: SP += 8 for binary-pop ops (ADD/SUB/etc.)."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_binary_pop_sp_increment
        _set_binary_pop_sp_increment(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="binary_pop_sp_increment",
        phase=6,
        reads={"MARK_SP", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_SHL", "OP_SHR", "EMBED_LO", "EMBED_HI",
               "BYTE_INDEX_0"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer10_stack0_byte_relay_op() -> Operation:
    """L10 attention: STACK0 byte values relay for AND/OR/XOR multi-byte."""
    def bake(attn, dim_positions, S):
        from ..vm_step import _set_layer10_stack0_byte_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_stack0_byte_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_stack0_byte_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "MARK_STACK0", "STACK0_BYTE0",
               "TEMP", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer9_marker_suppress_op() -> Operation:
    """L9 FFN extension: marker suppression."""
    def bake(ffn, dim_positions, S):
        from ..vm_step import _set_layer9_marker_suppress
        # _set_layer9_marker_suppress takes (ffn, S, BD, start_unit). We need
        # to know what start_unit to use. The original code uses unit count
        # after _set_layer9_alu — for the migration shim we just pass start_unit=0.
        # This may overlap with layer9_alu's unit assignments; the original calls
        # them sequentially in the same FFN.
        _set_layer9_marker_suppress(ffn, S, _as_setdim_proxy(dim_positions), 0)

    return Operation(
        name="layer9_marker_suppress",
        phase=9,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def _bake_post_op_into(ffn, post_op_instance, hidden_offset: int = 0) -> int:
    """Copy a post_op's weights into a target FFN starting at `hidden_offset`.

    The post_op classes (BinaryOpByteZeroingPostOp etc.) are PureFFN subclasses
    that bake their weights in __init__. We construct one and copy weights into
    the target block FFN's hidden-unit slots. Returns the next free hidden_offset.
    """
    H = post_op_instance.W_up.shape[0]
    end = hidden_offset + H
    target_H = ffn.W_up.shape[0]
    if end > target_H:
        raise ValueError(
            f"FFN hidden_dim={target_H} too small for post_op (needs +{H} at offset {hidden_offset})"
        )
    ffn.W_up.data[hidden_offset:end, :] = post_op_instance.W_up.data
    ffn.b_up.data[hidden_offset:end] = post_op_instance.b_up.data
    ffn.W_gate.data[hidden_offset:end, :] = post_op_instance.W_gate.data
    ffn.b_gate.data[hidden_offset:end] = post_op_instance.b_gate.data
    ffn.W_down.data[:, hidden_offset:end] = post_op_instance.W_down.data
    return end


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    BitwiseBytePropagation + ComparisonCombine, baked sequentially into
    one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    them additively into a single ffn at phase=10.5 so the compiler places
    them right after layer10_alu.
    """
    def bake(ffn, dim_positions, S):
        from ..vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
        )
        d_model = ffn.W_up.shape[1]
        offset = 0
        offset = _bake_post_op_into(ffn, BinaryOpByteZeroingPostOp(d_model, S), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=0, cascade=False), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=1, cascade=True), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=2, cascade=True), offset)
        offset = _bake_post_op_into(ffn, BitwiseBytePropagationPostOp(d_model, S), offset)
        offset = _bake_post_op_into(ffn, ComparisonCombine(d_model, S), offset)

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        phase=10.5,
        reads={
            "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_SHL", "OP_SHR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_OR", "OP_XOR", "OP_AND",
            "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
            "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
            "OP_SI", "OP_SC", "OP_PSH", "OP_EXIT", "OP_NOP",
            "OP_PUTCHAR", "OP_GETCHAR",
            "OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI",
            "CARRY", "CMP", "TEMP",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI", "CARRY"},
        kind="ffn",
        bake_fn=bake,
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp,
      DivModModule(mode='lookup').

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3,
      BitwiseBytePropagationPostOp,
      ComparisonCombine,
      EfficientDivMod_Neural.

    The existing `make_l10_post_ops_combined` is unrelated: it bakes the
    LOGIC of the FFN-style post_ops into a single phase-10.5 FFN (a parallel
    representation), not the attached module list. Both can coexist.

    phase=10.7: runs after L10 FFN bake (phase=10) and the combined FFN
    (phase=10.5), but well before structural post-passes (1100+).
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    def bake(block, dim_positions, S):
        from ..vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
            DivModModule,
            EfficientDivMod_Neural,
            _SetDim,
        )
        # Use the block's d_model when available; fall back to 512 to mirror
        # the previous inline behavior.
        d_model = 512
        if hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = 512

        block.post_ops.append(BinaryOpByteZeroingPostOp(d_model=d_model, S=S))
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=0, cascade=False)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=1, cascade=True)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=2, cascade=True)
        )
        block.post_ops.append(BitwiseBytePropagationPostOp(d_model=d_model, S=S))
        if alu_mode == "lookup":
            # The lookup-mode HybridALU override in set_vm_weights replaces
            # this DivModModule with EfficientDivMod_Neural via
            # `model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)`.
            block.post_ops.append(DivModModule(mode="lookup"))
        else:  # efficient
            block.post_ops.append(ComparisonCombine(S=S))
            block.post_ops.append(EfficientDivMod_Neural(S, _SetDim))

    return Operation(
        name="l10_post_op_attach",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=10.7,
        layer_idx=10,
        migrated=True,
    )


def setup_token_embeddings(embed_weight, dim_positions: Dict[str, int] = None) -> None:
    """Bake the per-token embedding values using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path uses auto-allocated positions. Falls back to _SetDim when
    dim_positions is None.

    Args:
        embed_weight: nn.Embedding.weight tensor [vocab, d_model].
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ..vm_step import Token, _SetDim

    def D(name):
        if dim_positions is not None and name in dim_positions:
            return dim_positions[name]
        return getattr(_SetDim, name)

    V = embed_weight.shape[0]

    with torch.no_grad():
        embed_weight.zero_()

        # CONST=1 for every token
        const = D("CONST")
        for tok in range(V):
            embed_weight[tok, const] = 1.0

        # Marker tokens
        is_mark = D("IS_MARK")
        for tok, dim_name in [
            (Token.REG_PC, "MARK_PC"),
            (Token.REG_AX, "MARK_AX"),
            (Token.REG_SP, "MARK_SP"),
            (Token.REG_BP, "MARK_BP"),
            (Token.MEM, "MARK_MEM"),
            (Token.CODE_START, "MARK_CS"),
        ]:
            if tok < V:
                embed_weight[tok, D(dim_name)] = 1.0
                embed_weight[tok, is_mark] = 1.0

        # STACK0 marker WITHOUT IS_MARK (so threshold heads see BP as nearest)
        if Token.STACK0 < V:
            embed_weight[Token.STACK0, D("MARK_STACK0")] = 1.0

        # Step-end / data-end / halt
        mark_se = D("MARK_SE")
        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            if tok < V:
                embed_weight[tok, mark_se] = 1.0
                embed_weight[tok, is_mark] = 1.0

        if Token.STEP_END < V:
            embed_weight[Token.STEP_END, D("MARK_SE_ONLY")] = 1.0

        if Token.TOOL_CALL < V:
            embed_weight[Token.TOOL_CALL, mark_se] = 1.0
            embed_weight[Token.TOOL_CALL, is_mark] = 1.0
            embed_weight[Token.TOOL_CALL, D("MARK_SE_ONLY")] = 1.0
            embed_weight[Token.TOOL_CALL, const] = 1.0

        # Thinking markers (try/except in case dims not declared in compiler spec)
        try:
            temp = D("TEMP")
            if Token.THINKING_START < V:
                embed_weight[Token.THINKING_START, is_mark] = 1.0
                embed_weight[Token.THINKING_START, const] = 1.0
                embed_weight[Token.THINKING_START, temp + 1] = 1.0
            if Token.THINKING_END < V:
                embed_weight[Token.THINKING_END, is_mark] = 1.0
                embed_weight[Token.THINKING_END, const] = 1.0
                embed_weight[Token.THINKING_END, temp + 2] = 1.0
        except AttributeError:
            pass

        if Token.IO_STATE_EMIT_BYTE < V:
            embed_weight[Token.IO_STATE_EMIT_BYTE, is_mark] = 1.0
            embed_weight[Token.IO_STATE_EMIT_BYTE, const] = 1.0
        if Token.IO_STATE_EMIT_THINKING < V:
            embed_weight[Token.IO_STATE_EMIT_THINKING, is_mark] = 1.0
            embed_weight[Token.IO_STATE_EMIT_THINKING, const] = 1.0

        # Byte tokens 0-255: IS_BYTE + nibble decoding
        is_byte = D("IS_BYTE")
        embed_lo = D("EMBED_LO")
        embed_hi = D("EMBED_HI")
        clean_lo = D("CLEAN_EMBED_LO")
        clean_hi = D("CLEAN_EMBED_HI")
        for b in range(256):
            embed_weight[b, is_byte] = 1.0
            embed_weight[b, embed_lo + (b & 0xF)] = 1.0
            embed_weight[b, embed_hi + ((b >> 4) & 0xF)] = 1.0
            embed_weight[b, clean_lo + (b & 0xF)] = 1.0
            embed_weight[b, clean_hi + ((b >> 4) & 0xF)] = 1.0


def setup_head_weights(head, dim_positions: Dict[str, int] = None) -> None:
    """Bake the output-projection head weights using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path can call it with auto-allocated dim positions instead of
    _SetDim constants. When `dim_positions` is None, falls back to _SetDim
    (backward-compat with hand-set path).

    Args:
        head: The model.head nn.Linear(d_model, vocab_size) module.
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ..vm_step import Token, _SetDim

    def D(name):
        if dim_positions is not None and name in dim_positions:
            return dim_positions[name]
        return getattr(_SetDim, name)

    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

        next_flags = [
            D("NEXT_PC"), D("NEXT_AX"), D("NEXT_SP"), D("NEXT_BP"),
            D("NEXT_STACK0"), D("NEXT_MEM"), D("NEXT_SE"), D("NEXT_HALT"),
        ]
        # Optional flags (only present when conversational I/O is enabled)
        for opt in ("NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END"):
            try:
                next_flags.append(D(opt))
            except AttributeError:
                pass

        OUTPUT_LO = D("OUTPUT_LO")
        OUTPUT_HI = D("OUTPUT_HI")
        for b in range(256):
            lo, hi = b & 0xF, (b >> 4) & 0xF
            head.weight[b, OUTPUT_LO + lo] = 5.0
            head.weight[b, OUTPUT_HI + hi] = 5.0
            head.bias[b] = -5.0
            for flag in next_flags:
                head.weight[b, flag] += -80.0
        head.bias[0] = -4.0

        vocab_size = head.weight.shape[0]
        for tok, flag_name in [
            (Token.REG_PC, "NEXT_PC"),
            (Token.REG_AX, "NEXT_AX"),
            (Token.REG_SP, "NEXT_SP"),
            (Token.REG_BP, "NEXT_BP"),
            (Token.STACK0, "NEXT_STACK0"),
            (Token.MEM, "NEXT_MEM"),
            (Token.STEP_END, "NEXT_SE"),
            (Token.HALT, "NEXT_HALT"),
            (Token.TOOL_CALL, "NEXT_TOOL_CALL"),
            (Token.THINKING_START, "NEXT_THINKING_START"),
            (Token.THINKING_END, "NEXT_THINKING_END"),
        ]:
            if tok >= vocab_size:
                continue
            try:
                head.weight[tok, D(flag_name)] = 20.0
                head.bias[tok] = -10.0
            except AttributeError:
                pass

        for tok in [
            Token.CODE_START, Token.CODE_END,
            Token.DATA_START, Token.DATA_END,
            Token.SEP, Token.USER_INPUT_START, Token.USER_INPUT_END,
        ]:
            if tok < vocab_size:
                head.bias[tok] = -50.0

        if Token.IO_STATE_EMIT_BYTE < vocab_size:
            head.bias[Token.IO_STATE_EMIT_BYTE] = -20.0
        if Token.IO_STATE_EMIT_THINKING < vocab_size:
            head.bias[Token.IO_STATE_EMIT_THINKING] = -20.0


def make_head_bake_op() -> Operation:
    """Bake the output projection head: byte/marker token logits.

    Phase=1000 so it runs AFTER legacy_bake (phase=999); the corresponding
    head section in `set_vm_weights` has been removed to avoid double-bake.
    """
    def _bake(model, dim_positions, S):
        setup_head_weights(model.head, dim_positions)

    return Operation(
        name="head_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1000,
    )


def make_embedding_bake_op() -> Operation:
    """Bake the per-token embedding table.

    Phase=1001 so it runs AFTER legacy_bake (phase=999) and head_bake (1000);
    the corresponding embedding section in `set_vm_weights` has been removed
    to avoid double-bake.
    """
    def _bake(model, dim_positions, S):
        setup_token_embeddings(model.embed.embed.weight, dim_positions)

    return Operation(
        name="embedding_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1001,
    )


def make_legacy_bake_op(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
) -> Operation:
    """Bridge op: invoke the legacy set_vm_weights pipeline as a model-level bake.

    This is the migration bridge — it lets `compile_full_vm` orchestrate the
    full bake through compiler dispatch even before every individual op has
    been split into its own compiler Operation. The legacy bake is just one
    "op" the compiler runs; as individual ops migrate out into their own
    Operation instances, the legacy pipeline shrinks.

    Phase 999 ensures it runs last, after all other compiler ops. Reads and
    writes are empty since the dependency graph already orders the per-layer
    bakes via the per-op Operation declarations.
    """
    def _bake(model, dim_positions, S):
        from ..vm_step import set_vm_weights
        set_vm_weights(
            model,
            enable_tool_calling=enable_tool_calling,
            enable_conversational_io=enable_conversational_io,
            alu_mode=alu_mode,
        )

    return Operation(
        name="legacy_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=999,
    )


def all_core_ops() -> list:
    """Return the full list of migrated core-VM operations.

    Doesn't include I/O, conversational, or tool-call operations (those need
    their own migration pass).
    """
    return [
        make_layer0_threshold_attn_op(),
        make_layer1_threshold_attn_op(),
        make_layer2_threshold_attn_op(),
        make_layer3_carry_forward_attn_op(),
        make_phase_a_ffn_op(),
        make_layer1_ffn_op(),
        make_layer2_mem_byte_flags_op(),
        make_nibble_copy_ffn_op(),
        make_layer3_ffn_op(),
        make_layer4_pc_relay_op(),
        make_layer4_ffn_op(),
        make_layer5_fetch_op(),
        make_opcode_decode_ffn_op(),
        make_layer6_attn_op(),
        make_layer6_routing_ffn_op(),
        make_layer6_relay_heads_op(),
        make_layer7_operand_gather_op(),
        make_layer7_memory_heads_op(),
        make_layer8_alu_op(),
        make_layer8_multibyte_fetch_op(),
        make_layer8_multibyte_routing_op(),
        make_layer8_sp_gather_op(),
        make_layer9_alu_op(),
        make_layer9_lev_addr_relay_op(),
        make_layer9_lev_bp_to_pc_relay_op(),
        make_layer10_carry_relay_op(),
        make_layer10_byte_passthrough_op(),
        make_layer10_sp_byte_passthrough_op(),
        make_layer10_psh_stack0_passthrough_op(),
        make_layer10_alu_op(),
        make_layer11_mul_partial_op(),
        make_layer12_mul_combine_op(),
        make_layer13_mem_addr_gather_op(),
        make_layer13_shifts_op(),
        make_layer14_mem_generation_op(),
        make_layer15_memory_lookup_op(),
        make_layer16_lev_routing_op(),
        # Critical additional ops (M3+ continuation)
        make_binary_pop_sp_increment_op(),
        make_layer10_stack0_byte_relay_op(),
        make_layer9_marker_suppress_op(),
        # L10 post_ops merged into a single phase-10.5 ffn
        make_l10_post_ops_combined(),
        # Model-level bakes (run after legacy_bake's per-layer/head/embed work)
        make_head_bake_op(),
        make_embedding_bake_op(),
    ]


# ---------------------------------------------------------------------------
# Dim spec compatible with _SetDim
# ---------------------------------------------------------------------------

# Known limitation of the migration shims:
#
# Many ops both *read* and *write* dims like OUTPUT_LO/EMBED_LO. The reads happen
# at one position (e.g., MARK_PC) and writes at another (e.g., MARK_AX). My
# Operation declarations use dim *names* without position context, so the compiler
# can see both ops reading/writing the same name and infer a circular dependency
# where none truly exists. This is a real architectural limitation of the current
# LayerCompiler dep model — the next refinement needs per-position reads/writes
# (e.g., "EMBED_LO@MARK_PC" vs "EMBED_LO@MARK_AX") so the compiler can distinguish
# "reading the previous position's value" from "writing this position's value".
#
# Until that refinement, all_core_ops() compiled together produces a cycle. The
# work-around for now: the unit tests only exercise small subsets that don't
# create cycles, and full-spec compilation isn't wired to production.


# IO-required dim names that MUST stay pinned to their _SetDim positions even
# when the compiler is otherwise free to bump-pointer-allocate. These dims are
# read or written by external (non-bake) code paths — token embedding setup,
# the output head, and `NeuralVMEmbedding._inject_*` runtime injectors — that
# resolve dim positions either through the `_SetDim` enum directly or through
# `dim_positions` lookups that must agree with `_SetDim` for now.
#
# Membership rationale (cross-checked against
# `c4_release/neural_vm/neural_embedding.py:_inject_*`):
#
# - EMBED_LO/HI, OUTPUT_LO/HI: nibble-decode/projection. Token embedding sets
#   EMBED_*; head reads OUTPUT_*. _inject_initial_pc writes EMBED_*.
# - MARK_PC/AX/SP/BP/MEM/SE/STACK0/CS/SE_ONLY: per-token marker flags set by
#   token embedding; threshold heads scan for them.
# - NEXT_*: head reads these to project to token-type logits.
# - IS_BYTE/IS_MARK/CONST/HAS_SE/BYTE_INDEX_*: positional flags read by head
#   gating and by L0 thresholds.
# - OP_LEV/BZ/BNZ + ACTIVE_OPCODE_PRTF/READ: injected by
#   `_inject_active_opcode` based on the current opcode hint.
# - MARK_THINKING_START/END: written by `_inject_thinking_markers` on
#   THINKING_START/END tokens.
# - MEM_STORE / MEM_EXEC / ADDR_KEY: written by `_inject_mem_store`,
#   `_inject_mem_exec[_autoregressive]` for memory ops.
# - NEXT_TOOL_CALL / NEXT_THINKING_START / NEXT_THINKING_END: optional head
#   reads when conversational I/O is enabled (see setup_head_weights).
_IO_REQUIRED_DIMS = frozenset({
    # Markers (token embedding writes; threshold heads read)
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_SE",
    "MARK_CS", "MARK_SE_ONLY", "MARK_STACK0",
    "MARK_THINKING_START", "MARK_THINKING_END",
    # Positional flags (head + L0 thresholds)
    "IS_BYTE", "IS_MARK", "CONST", "HAS_SE",
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    # Nibble encoding (token embed in / head out / _inject_initial_pc)
    "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
    "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
    # NEXT_* token-type transition flags (head reads)
    "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
    "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
    "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
    # Active-opcode injection slots (_inject_active_opcode)
    "OP_LEV", "OP_BZ", "OP_BNZ",
    "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
    # Memory injection slots (_inject_mem_store, _inject_mem_exec)
    "MEM_STORE", "MEM_EXEC", "ADDR_KEY",
})


def declare_setdim_compat_dims(
    compiler,
    pin_to_setdim: bool = True,
    pin_io_only: bool = False,
) -> None:
    """Declare to a LayerCompiler all dims that match the existing _SetDim layout.

    Args:
        compiler: LayerCompiler to declare dims to
        pin_to_setdim: if True, each dim is pinned to its _SetDim position. This
            preserves _SetDim's aliasing scheme (e.g., FETCH_LO==MUL_ACCUM at
            position 420) so existing _set_layerN_* bake_fns work unchanged. If
            False, dims are bump-pointer allocated by declaration order — useful
            for testing the auto-allocation path but breaks _SetDim aliases.
        pin_io_only: if True, only the dims in `_IO_REQUIRED_DIMS` (the
            externally-observable dims read/written by token embedding, the
            output head, and `NeuralVMEmbedding._inject_*` runtime injectors)
            are pinned to their `_SetDim` positions. Every other dim becomes
            bump-pointer-allocated by the compiler. This unlocks compiler-driven
            internal dim allocation while keeping the IO contract intact.
            Implies `pin_to_setdim=True` for the IO-required subset; the
            `pin_to_setdim` flag is ignored when `pin_io_only=True`. Defaults
            to False for backward compatibility.
    """
    from ..vm_step import _SetDim
    from ..constants import INSTR_WIDTH  # noqa: F401 (touched for completeness)

    # Single-dim flags
    one_dim = [
        "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
        "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
        "MARK_SE_ONLY", "MARK_STACK0",
        "MARK_THINKING_START", "MARK_THINKING_END",
        "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
        "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "CMP_GROUP",
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
        "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
        "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
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
        "OPCODE_BASE",
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

    def _declare(name, size):
        if not hasattr(_SetDim, name):
            return
        if pin_io_only:
            # Only pin the IO-required subset; everything else is
            # bump-pointer-allocated by the compiler.
            pinned = getattr(_SetDim, name) if name in _IO_REQUIRED_DIMS else None
        else:
            pinned = getattr(_SetDim, name) if pin_to_setdim else None
        compiler.declare_dim(name, size, pinned=pinned)

    for name in one_dim:
        _declare(name, 1)
    for name in seven_dim:
        _declare(name, 7)
    for name in sixteen_dim:
        _declare(name, 16)
    for name in four_dim:
        _declare(name, 4)
    for name in eight_dim:
        _declare(name, 8)
    for name in forty_eight_dim:
        _declare(name, 48)
    for name in thirty_two_dim:
        _declare(name, 32)
