"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 15 helpers: convo-IO output routing.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_conversational_io_output_routing(ffn, S, BD):
    """L15 FFN addition: Route OUTPUT_BYTE to OUTPUT when in output mode.

    When IO_IN_OUTPUT_MODE (emitting output bytes):
    - Copy OUTPUT_BYTE_LO → OUTPUT_LO (all 16 nibbles)
    - Copy OUTPUT_BYTE_HI → OUTPUT_HI (all 16 nibbles)

    This routes the fetched format string byte to the output head for emission.

    Note: We don't need to suppress normal OUTPUT routing because IO_IN_OUTPUT_MODE
    only activates after THINKING_END, at which point we're not in the normal
    35-token generation cycle.

    Starts at unit 1200 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1200

    # Copy each OUTPUT_BYTE nibble to corresponding OUTPUT nibble when in output mode
    for k in range(16):
        # Lo nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

        # Hi nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1



