"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer14_mem_generation_op() -> Operation:
    """L14 attention: generate MEM section tokens (addr + value) for SI/SC/PSH."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer14_mem_generation
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
        migrated=True,
    )


def make_layer14_temp_clear_op() -> Operation:
    """L14 FFN: Clear TEMP[0] at PC marker when OP_LEV is active.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Chains with the other L14
    additive cleanup ops (``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``) via a shared FFN unit counter stored
    on ``block.ffn._l14_unit_counter``. First in the chain (phase=14.1).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_temp_clear
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_temp_clear(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_temp_clear",
        phase=14.1,
        reads={"OP_LEV", "MARK_PC", "CONST"},
        writes={"TEMP"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


def make_layer14_clear_addr_key_pollution_op() -> Operation:
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Second in the chain (phase=14.2).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_addr_key_pollution
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_addr_key_pollution(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_addr_key_pollution",
        phase=14.2,
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "MARK_PC", "MARK_BP", "MARK_AX", "MARK_STACK0", "MARK_SP",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


def make_layer14_clear_output_corruption_op() -> Operation:
    """L14 FFN: Boost OUTPUT[0] at STACK0 byte positions to fix attention bleed.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Third in the chain (phase=14.3).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_output_corruption
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_output_corruption(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_output_corruption",
        phase=14.3,
        reads={"H4", "H1", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


def make_layer14_clear_mem_marker_output_op() -> Operation:
    """L14 FFN: Clear OUTPUT at MEM marker for OP_JSR/OP_ENT.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops
    (``layer14_temp_clear``, ``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``). Last in the chain (phase=14.4).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_mem_marker_output
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_mem_marker_output(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_mem_marker_output",
        phase=14.4,
        reads={"OP_JSR", "OP_ENT", "MARK_MEM", "IS_BYTE",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


