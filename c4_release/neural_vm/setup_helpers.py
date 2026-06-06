"""Setup helpers for VM weight baking.

This module contains weight-setup helpers extracted from ``vm_step.py`` after
Wave 3 deleted ``make_legacy_bake_op``. These functions are now only called by
migrated bake_fns inside ``unified_compiler/migrated_ops.py``.

They are re-exported by ``vm_step`` for backward compatibility, so external
imports like ``from neural_vm.vm_step import _set_X`` continue to work.

All functions are self-contained: they take ``ffn``/``attn`` and ``BD``
(a ``_SetDim`` class) as parameters, and only depend on stdlib ``math`` and
``PC_OFFSET`` from ``.constants``. Internal references to ``unified_compiler``
primitives are lazy-imported inside function bodies.

Phase 6 wave 1.5 (mechanical split, current-base): the per-layer helpers were
moved into sibling modules ``setup_helpers_l<N>.py`` so subsequent Phase 6
declarative migrations can edit each layer independently without conflicting
on this file. This module re-exports them so existing call sites
(``vm_step`` and ``tests/test_v18_convo_io_neural_bakes``) continue to import
from ``neural_vm.setup_helpers`` byte-identically.
"""

import math

from .constants import PC_OFFSET

# Per-layer re-exports (Phase 6 wave 1.5 mechanical split).
from .setup_helpers_l1 import _set_layer1_ffn
from .setup_helpers_l2 import _set_layer2_mem_byte_flags
from .setup_helpers_l3 import (
    _set_convo_io_step_resume,
    _set_stack0_carry_attn,
)
from .setup_helpers_l4 import _set_convo_io_prtf_transport
from .setup_helpers_l5 import (
    _set_conversational_io_opcode_decode,
    _set_layer5_fetch,
    _set_tool_call_opcode_decode,
)
from .setup_helpers_l6 import (
    _set_bz_bnz_relay,
    _set_conversational_io_relay_heads,
    _set_conversational_io_state_machine,
    _set_convo_io_pc_sp_latch,
    _set_tool_call_detection,
    _set_tool_call_relay_head,
)
from .setup_helpers_l7 import _set_convo_io_prtf_capture
from .setup_helpers_l9 import (
    _set_layer9_lev_addr_relay,
    _set_layer9_lev_bp_to_pc_relay,
)
from .setup_helpers_l10 import (
    _set_layer10_bp_byte_passthrough,
    _set_layer10_byte_passthrough,
    _set_layer10_carry_relay,
    _set_layer10_psh_stack0_passthrough,
    _set_layer10_sp_byte_passthrough,
    _set_layer10_stack0_byte_relay,
    _set_null_terminator_detection,
)
from .setup_helpers_l11 import _set_layer11_mul_partial
from .setup_helpers_l12 import _set_layer12_mul_combine
from .setup_helpers_l13 import (
    _set_layer13_mem_addr_gather,
    _set_layer13_shifts,
)
from .setup_helpers_l14 import (
    _set_layer14_add_byte1_high_zero_cleanup,
    _set_layer14_clear_addsub_temp_negative_residue,
    _set_layer14_clear_output_corruption,
    _set_layer14_jsr_mem_default_suppress,
    _set_layer14_mem_addr_src_default_suppress,
)
from .setup_helpers_l15 import _set_conversational_io_output_routing


# Note: ``_set_cs_threshold_attn`` (a single CS-only threshold attention head)
# previously lived here. It had no live callers (only an import-only re-export
# in vm_step.py) and was deleted per BD_SETDIM_HARDCODE_AUDIT M2.


# =============================================================================
# V18 Phase 1: step-resumption + PC/SP latch (V18_CONVO_IO_NEURAL_PLAN.md §3)
# =============================================================================
#
# These two helpers close the gap identified in §3 of the V18 plan: with the
# existing convo-IO pipeline (L2/L3/L5/L6/L7/L8/L9/L10/L15) plus these two
# bakes, the model can autonomously emit
# ``THINKING_END → bytes → THINKING_START → REG_PC → ... → STEP_END`` with no
# Python intervention.
#
# Both are wired into ``set_vm_weights`` only when both
# ``enable_conversational_io=True`` AND the Phase 1 gates
# (``enable_convo_io_step_resume`` / ``enable_convo_io_pc_sp_latch``) are
# True. The Phase 1 gates default to False so this landing is byte-identical
# to the prior compile (the FFN units below are simply left zero).
#
# Unit allocation (kept disjoint from existing convo-IO + V9 PUTCHAR ranges):
#
#   L3 FFN unit 1035     — _set_convo_io_step_resume (3a)
#                          extends _set_conversational_io_state_init at 1034
#   L6 FFN units 1402-1465 — _set_convo_io_pc_sp_latch (3b) replay unit band
#                            (32 units for PC nibbles + 32 for SP nibbles).
#                            Starts at 1402 — directly after the existing
#                            convo-IO state-machine units 1400-1401 and
#                            below the V9 PUTCHAR routing band (1500-1532),
#                            so no overlap with any baked unit today.
#   L7 FFN units 800-863  — _set_convo_io_prtf_capture (3c, Phase 1b)
#                            capture-side band (32 units for PC nibbles +
#                            32 for SP nibbles). Writes the POST_PRTF_PC /
#                            POST_PRTF_SP cache dims that 3b reads.
#   L4 attn head 4        — _set_convo_io_prtf_transport (3d, Phase 1c)
#                            transport-side attention head. At the post-
#                            THINKING_START position attends back to the
#                            most recent PRTF AX marker (gated by
#                            ACTIVE_OPCODE_PRTF + MARK_AX) and copies the
#                            captured POST_PRTF_PC/SP nibbles forward
#                            across the variable-length output-byte
#                            interlude. ALiBi slope=0.1 (shallow) so the
#                            head can reach back ~80 tokens. Closes the
#                            transport gap so 3c→3d→3b round trip works
#                            end-to-end.


__all__ = [
    "PC_OFFSET",
    "math",
    "_set_bz_bnz_relay",
    "_set_conversational_io_opcode_decode",
    "_set_conversational_io_output_routing",
    "_set_conversational_io_relay_heads",
    "_set_conversational_io_state_machine",
    "_set_convo_io_pc_sp_latch",
    "_set_convo_io_prtf_capture",
    "_set_convo_io_prtf_transport",
    "_set_convo_io_step_resume",
    "_set_layer10_bp_byte_passthrough",
    "_set_layer10_byte_passthrough",
    "_set_layer10_carry_relay",
    "_set_layer10_psh_stack0_passthrough",
    "_set_layer10_sp_byte_passthrough",
    "_set_layer10_stack0_byte_relay",
    "_set_layer11_mul_partial",
    "_set_layer12_mul_combine",
    "_set_layer13_mem_addr_gather",
    "_set_layer13_shifts",
    "_set_layer14_add_byte1_high_zero_cleanup",
    "_set_layer14_clear_addsub_temp_negative_residue",
    "_set_layer14_clear_output_corruption",
    "_set_layer14_jsr_mem_default_suppress",
    "_set_layer14_mem_addr_src_default_suppress",
    "_set_layer1_ffn",
    "_set_layer2_mem_byte_flags",
    "_set_layer5_fetch",
    "_set_layer9_lev_addr_relay",
    "_set_layer9_lev_bp_to_pc_relay",
    "_set_null_terminator_detection",
    "_set_stack0_carry_attn",
    "_set_tool_call_detection",
    "_set_tool_call_opcode_decode",
    "_set_tool_call_relay_head",
]
