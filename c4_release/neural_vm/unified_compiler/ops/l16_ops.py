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
        bake_fn=bake,
        migrated=True,
    )


