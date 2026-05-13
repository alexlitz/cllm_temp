"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer16_lev_routing_op() -> Operation:
    """L16 FFN: LEV routing — SP = BP + 16."""
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer16_lev_routing
        _set_layer16_lev_routing(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer16_lev_routing",
        phase=16,
        reads={"MARK_SP", "MARK_PC", "OP_LEV", "ADDR_B0_LO", "ADDR_B0_HI",
               "TEMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        layer_idx=16,
        bake_fn=bake,
        migrated=True,
        # ``_set_layer16_lev_routing`` writes 121 units (vm_step.py:6845):
        #   16 (cancel OUTPUT_LO at SP) + 16 (cancel OUTPUT_HI at SP) +
        #   16 (SP=BP+16 lo) + 16 (SP=BP+16 hi) +
        #   16 (cancel OUTPUT_HI at PC) + 16 (TEMP_LO→OUTPUT_LO) +
        #   16 (TEMP_HI→OUTPUT_HI) + 3 (clear OUTPUT_LO[10] byte 1-3) +
        #   3 (set OUTPUT_LO[0] byte 1-3) + 3 (set OUTPUT_HI[0] byte 1-3)
        #   = 121. The bytes 1-3 blocks (if False) are disabled.
        ffn_units_used=121,
        # Staleness invariants: on OP_LEV, this FFN writes SP=BP+16 nibbles
        # to OUTPUT at MARK_SP and the return PC (TEMP) to OUTPUT at MARK_PC.
        # Canonical register: SP — the SP-marker emission is the primary
        # consumer-facing output (the PC-marker return-address emission shares
        # the same OUTPUT_LO/HI write pattern with a parallel set of units).
        produces={
            "OUTPUT_LO": "SP",
            "OUTPUT_HI": "SP",
        },
    )


