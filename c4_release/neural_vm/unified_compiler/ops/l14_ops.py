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
        layer_idx=14,
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


def make_layer14_jsr_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_JSR is active.

    FIX 2026-05-12 (fix-jsr-ax-bytes-1-3): Per C4's 8-bit-AX convention, AX
    bytes 1-3 must be 0. For OP_JSR, the L9 ALU JSR-preserve routing writes
    the previous AX byte 0 at MARK_AX (correctly emitting AX byte 0), but
    bytes 1-3 of AX are predicted at AX byte 0/1/2 positions where L14
    attention / L7 head 1 SP gather contaminate OUTPUT with SP/PC bytes.

    This op mirrors ``BinaryOpByteZeroingPostOp``: at AX byte positions
    (IS_BYTE + H1[AX]) when OP_JSR is active, write -3/S to every OUTPUT_LO
    /OUTPUT_HI nibble dim and +5/S to OUTPUT_LO[0]/OUTPUT_HI[0], so the
    byte-value-0 token wins argmax — producing AX bytes 1-3 = 0x00. OP_JSR
    is broadcast to AX byte positions by L7 head 5 (V slot 8, also new in
    this commit).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.6: runs AFTER ``layer14_addr_key_neural_decode`` (14.5).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_jsr_ax_bytes_zero
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_jsr_ax_bytes_zero(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_jsr_ax_bytes_zero",
        phase=14.6,
        reads={"OP_JSR", "IS_BYTE", "H1", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


def _bake_addr_key_neural_decode(ffn, dim_positions, S, start_unit=0):
    """Bake the BLOG_SPEC.md:830 ADDR_KEY nibble decode into ``ffn``.

    Computes the same per-val-byte ADDR_KEY one-hot writes that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, but via
    a baked FFN reading the addr byte values that L13's
    ``_set_layer13_mem_addr_gather`` has already gathered into
    ``ADDR_B0_LO``/``ADDR_B0_HI``/``ADDR_B1_LO`` at MEM val byte
    positions.

    Per val byte position (gated by MEM_VAL_B{0,1,2,3}):
      Let addr_b0 = (hi << 4 | lo) be the byte 0 of the MEM section's
      address.  Let addr_b1_lo be the low nibble of byte 1.  Then
      ``byte_addr = addr_b0 + byte_off`` for byte_off ∈ {0,1,2,3}.

      ADDR_KEY[byte_addr & 0xF]              = 1.0  (lo nibble)
      ADDR_KEY[16 + (byte_addr >> 4) & 0xF]  = 1.0  (hi nibble)
      ADDR_KEY[32 + (addr_b1 + carry) & 0xF] = 1.0  (top nibble)

      where carry = 1 iff (lo + byte_off) >= 16 and the hi nibble's
      addition (hi + carry_from_lo) overflowed past 15.  In the
      common case (byte_off ≤ 3, hi nibble < 15), the top nibble is
      just ``addr_b1_lo``; the high-byte carry case only occurs when
      hi==15 AND lo+byte_off >= 16, which is rare.

    Encoding strategy: enumerate the 16×16×4 combinations of
    (addr_b0_lo, addr_b0_hi, byte_off).  For each, compute byte_addr
    and emit two FFN units:
      - one writing the lo+hi nibbles of byte_addr into ADDR_KEY[0..31]
      - one writing the top nibble (addr_b1_lo+carry) into
        ADDR_KEY[32..47]

    Both units use a 3-way AND in the silu path:
      MEM_VAL_B{byte_off} + ADDR_B0_LO[lo] + ADDR_B0_HI[hi] >= 3
    with threshold ``-S*2.5`` so only all-3-match fires.

    The top-nibble unit additionally reads ADDR_B1_LO (a 4-way AND with
    threshold ``-S*3.5``) to source the high byte.  The carry case
    (rare) is handled by selecting addr_b1_lo+1 instead of addr_b1_lo
    when (lo + byte_off >= 16) AND (hi == 15).

    Returns ``next_unit`` so callers can chain.
    """
    BD = _as_setdim_proxy(dim_positions)
    unit = start_unit
    MEM_VAL_DIMS = [BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]

    # Lo + Hi nibble units (write ADDR_KEY[0..31]): one unit per
    # (byte_off, hi, lo) combination, 4*16*16 = 1024 units.
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for hi in range(16):
            for lo in range(16):
                byte_addr = ((hi << 4) | lo) + byte_off
                new_lo = byte_addr & 0xF
                new_hi = (byte_addr >> 4) & 0xF
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + hi] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.b_gate[unit] = 1.0
                ffn.W_down[BD.ADDR_KEY + new_lo, unit] = 2.0 / S
                ffn.W_down[BD.ADDR_KEY + 16 + new_hi, unit] = 2.0 / S
                unit += 1

    # --- Top nibble units (write ADDR_KEY[32..47]) ---
    # For each (lo, hi, byte_off, b1_lo): top = (b1_lo + carry) & 0xF
    # where carry = 1 iff (byte_addr_full = ((hi<<4|lo) + byte_off) >> 8) > 0.
    # That carry-into-byte-1 happens only when hi==15 AND lo+byte_off >= 16
    # for byte_off ≤ 3. For correctness we still iterate the full lookup.
    # 16 * 16 * 4 * 16 = 16384 units — too many. We exploit the fact that
    # the high carry condition only depends on (hi, lo, byte_off), so we
    # split into:
    #   (a) common case (no carry): always emit ADDR_B1_LO[b1_lo] →
    #       ADDR_KEY+32+b1_lo. This is independent of addr_b0 / byte_off
    #       and just needs MEM_VAL_B{0..3} to gate the val-byte position.
    #   (b) carry case (hi==15, lo+byte_off>=16): emit ADDR_KEY+32+((b1_lo+1)&0xF)
    #       AND subtract the (a) unit's contribution. This is the
    #       (hi==15, lo, byte_off, b1_lo) combo — bounded by 16*4*16 = 1024.

    # (a) Common case: 4 byte_off × 16 b1_lo = 64 units.
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for b1_lo in range(16):
            ffn.W_up[unit, gate_dim] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 1.5
            ffn.b_gate[unit] = 1.0
            ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
            unit += 1

    # (b) Carry-correction: when hi==15 AND (lo + byte_off) >= 16:
    #     - subtract from ADDR_KEY+32+b1_lo (cancel (a))
    #     - add to ADDR_KEY+32+((b1_lo+1)&0xF)
    # The trigger is a 4-way AND: MEM_VAL_B{byte_off} + ADDR_B0_HI[15] +
    # ADDR_B0_LO[lo] + ADDR_B1_LO[b1_lo], for lo such that lo+byte_off>=16.
    # That set of lo values is: byte_off=0 → none; byte_off=1 → {15};
    # byte_off=2 → {14,15}; byte_off=3 → {13,14,15}.
    # 4 byte_off × variable × 16 b1_lo = 0+1+2+3 = 6 × 16 = 96 units.
    carry_los = {
        0: [],
        1: [15],
        2: [14, 15],
        3: [13, 14, 15],
    }
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for lo in carry_los[byte_off]:
            for b1_lo in range(16):
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + 15] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.b_gate[unit] = 1.0
                # Cancel the common-case write at b1_lo, add at (b1_lo+1)&0xF.
                ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = -2.0 / S
                ffn.W_down[BD.ADDR_KEY + 32 + ((b1_lo + 1) & 0xF), unit] = 2.0 / S
                unit += 1

    return unit


def make_layer14_addr_key_neural_decode_op(enable: bool = False) -> Operation:
    """L14 FFN: BLOG_SPEC.md:830 neural ADDR_KEY decode at MEM val byte positions.

    Phase 0 of the V2 ADDR_KEY migration (see
    ``docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md``).  Computes the same
    per-val-byte ADDR_KEY one-hot encoding that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, baked
    into FFN weights instead of a Python loop.

    Per BLOG_SPEC.md:830:

        First the simple case key value retrieval, we want an exact
        match for the key so we take the binary key break it up into
        bytes or nibbles, perform an equality check for each byte or
        nibble, then in a subsequent layer we perform a logical AND
        over those results.

    The "binary key" here is the MEM section's 32-bit address.  L13's
    ``_set_layer13_mem_addr_gather`` already gathers the first 3 addr
    bytes to MEM val byte positions as 4-bit one-hot nibbles
    (``ADDR_B{0,1,2}_LO/HI``).  This op takes those nibbles plus the
    ``MEM_VAL_B{0..3}`` flags (= byte_off) and emits the
    ``ADDR_KEY[lo, 16+hi, 32+top]`` one-hot encoding of
    ``byte_addr = addr + byte_off`` (with carry handling).

    Gating: ``enable=False`` (default) → bake is a no-op.  Existing
    tests stay byte-identical.  When flipped to True (in a follow-up
    PR), the bake must produce identical ADDR_KEY values to
    ``_inject_mem_metadata`` on all (addr_b0_lo, addr_b0_hi,
    addr_b1_lo, byte_off) inputs.

    Pinned to ``layer_idx=14`` via ``kind="block"``.  Shares the FFN
    unit counter ``block.ffn._l14_unit_counter`` with the other L14
    cleanup ops.  Phase 14.5: runs AFTER
    ``layer14_clear_addr_key_pollution`` (phase 14.2) so the decode
    output is authoritative (not pre-cleared away).

    Total FFN units consumed when enabled: ~1184
    (1024 lo+hi + 64 common-top + 96 carry-top).  When disabled, 0.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _bake_addr_key_neural_decode(
            ffn, dim_positions, S, start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_addr_key_neural_decode",
        phase=14.5,
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        layer_idx=14,
        migrated=True,
    )


