"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 3 helpers: STACK0 carry attention + V18 convo-IO step resume.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_stack0_carry_attn(attn, head_idx, HD, BD):
    """Set attention head for STACK0 carry-forward.

    At STACK0 marker positions, attend to previous step's STACK0 byte 0
    (identified by STACK0_BYTE0 flag from L1 FFN).
    Copies EMBED_LO/HI to EMBED_LO/HI at STACK0 marker.

    Args:
        BD: Required dim spec (proxy or ``_SetDim``). Callers must pass a
            compiler proxy from a migrated op so pin_io_only=True layouts
            wire to the correct residual lanes. The previous ``BD=None``
            legacy fallback was removed per BD_SETDIM_HARDCODE_AUDIT M3
            (the sole caller in ``unified_compiler/ops/l3_ops.py`` always
            passes ``BD=proxy``).
    """
    if BD is None:
        raise TypeError(
            "_set_stack0_carry_attn requires BD (use _as_setdim_proxy(dim_positions))"
        )
    base = head_idx * HD
    L = 15.0

    # Q: fires at STACK0 markers
    attn.W_q[base, BD.MARK_STACK0] = L
    # K: fires at positions with STACK0_BYTE0 flag
    attn.W_k[base, BD.STACK0_BYTE0] = L

    # V: copies EMBED_LO/HI (the byte value from embedding)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O: writes to EMBED_LO/HI at STACK0 marker
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate (same shape as Primitives.carry_forward_attention)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_STACK0] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L



def _set_convo_io_step_resume(ffn, S, BD):
    """L3 FFN addition (V18 Phase 1, bake 3a): resume normal stepping after
    ``LAST_WAS_THINKING_START``.

    When the previous token was THINKING_START, this fires:
    - Set NEXT_PC = 1 (the head decodes Token.REG_PC for the next token,
      starting a fresh VM step).
    - Clear IO_STATE (state machine returns to "normal execution").
    - Clear IO_IN_OUTPUT_MODE (defensive — L10's null_terminator unit
      already clears it, but re-clearing on the resume edge guarantees the
      flag is off before the next step's bytes flow through L15 routing).

    This replaces the runner-side line
    ``self._inject_synthetic_step(context, new_pc, ...)`` in
    ``run_vm.py:_handle_thinking_end``: instead of Python appending
    ``REG_PC`` after THINKING_START, the model autoregressively emits
    Token.REG_PC because NEXT_PC is now set.

    Uses L3 FFN unit 1035 (one above _set_conversational_io_state_init's
    1034). Pattern mirrors the state-init unit — a single up-projection on
    LAST_WAS_THINKING_START with a gate bias of 1.0 (no positional
    constraint needed: the L2 lookback head only sets the flag at t+1 of
    THINKING_START).
    """
    unit = 1035

    ffn.W_up[unit, BD.LAST_WAS_THINKING_START] = S
    ffn.b_up[unit] = -S * 0.5  # fire when LAST_WAS_THINKING_START ~ 1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_PC, unit] = 2.0 / S
    ffn.W_down[BD.IO_STATE, unit] = -2.0 / S
    ffn.W_down[BD.IO_IN_OUTPUT_MODE, unit] = -2.0 / S


