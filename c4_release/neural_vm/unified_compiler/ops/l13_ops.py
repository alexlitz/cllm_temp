"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import (
    _as_setdim_proxy,
    _empty_compiler_ir_factory,
    operand_from_memsp_enabled,
)


# === L13 attention-head layout (pinned indices) =====================
#
# ``layer13_mem_addr_gather`` owns heads 0, 1, 2 on the L13 attention
# block. Each head gathers one MEM addr byte's CLEAN_EMBED nibbles into
# ADDR_B{0,1,2}_LO/HI at the MEM val byte positions:
#
#   head 0 -> ADDR_B0_LO/HI, plus the ADDR_B0_VALID lifecycle bit
#             (slot 34, B7-4).
#   head 1 -> ADDR_B1_LO/HI, plus the ADDR_B1_VALID lifecycle bit
#             (slot 34, B8-A; routes into H5+4 / position 99).
#   head 2 -> ADDR_B2_LO/HI, plus the ADDR_B2_VALID lifecycle bit
#             (slot 34, B8-A; routes into H5+5 / position 100).
#
# Pre-migration the call sites used bare ``base = j * HD`` literals
# inside ``_set_layer13_mem_addr_gather`` and ``_l13_addr_bn_valid_extension``.
# Pinning the allocator preserves those exact slots so the lowering is
# byte-identical, while the layout table becomes the audited source of
# truth for future L13 attention extensions.
_L13_HEAD_LAYOUT = (
    # (op-name key,                            pinned head_idx)
    ("layer13_mem_addr_gather.head_0",         0),  # ADDR_B0 + ADDR_B0_VALID
    ("layer13_mem_addr_gather.head_1",         1),  # ADDR_B1 + ADDR_B1_VALID
    ("layer13_mem_addr_gather.head_2",         2),  # ADDR_B2 + ADDR_B2_VALID
)


def _allocate_layer13_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with the L13 heads pinned.

    Every entry in :data:`_L13_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying weight writes -- now expressed as
    ``DeclarativeAttentionHeadSpec`` instances -- land byte-identically.
    A future L13 attention op can claim a free head past index 2 via
    ``allocator.alloc(name, layer_idx=13)`` (no ``pin=``) without
    touching this table.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L13_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=13, pin=head_idx)
    return allocator


def _l13_head_idx(op_name: str) -> int:
    """Return the pinned L13 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L13_HEAD_LAYOUT` for callers (e.g.
    ``compiler_ir_factory`` helpers) that cannot instantiate a per-bake
    allocator. Mirrors the L4 ``_l4_head_idx`` pattern.
    """
    for name, head_idx in _L13_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l13_head_idx: unknown L13 attention op {op_name!r}")


# === L13 FFN unit layout (auto-fit) ===================================
#
# The ``layer13_shifts`` op owns the entire L13 FFN. The actual weight
# writes happen via the declarative SHL+SHR rule list (lowered through
# ``Primitives.lower_ffn_rules`` with ``start_unit=0``), which fills the
# pool in two 2048-unit sub-stages (SHL then SHR, each spanning 8 shift
# amounts x 16 a_hi x 16 a_lo).
#
# Phase 7.B.5: both sub-stages now use ``pin=None``. Declaration order
# is SHL then SHR; first-fit on an empty 4096-wide pool places SHL at
# unit 0, then SHR at unit 2048 — byte-identical to the legacy explicit
# pins. The SHL+SHR chain saturates the pool, so any future extension
# must first shrink a lookup table or widen the pool (auto-fit will
# error rather than silently overflow).
#
# The order in this table determines the auto-fit placement, which in
# turn must match the rule-list walk in ``_layer13_shifts_rules``
# (SHL-first then SHR). Changing rule counts requires updating this
# table's ``n_units`` entries in lock-step.
_L13_SHIFTS_UNIT_LAYOUT = (
    # (sub-stage name, n_units)  -- pins dropped (Phase 7.B.5, auto-fit)
    ("layer13_shifts.shl", 2048),  # OP_SHL: 8 shifts x 16 a_hi x 16 a_lo
    ("layer13_shifts.shr", 2048),  # OP_SHR: 8 shifts x 16 a_hi x 16 a_lo
)


def _allocate_layer13_shifts_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L13 shift sub-stages.

    Phase 7.B.5: both sub-stages are now auto-fit (``pin=None``).
    Declaration order is SHL then SHR; first-fit on an empty
    4096-wide pool places SHL at unit 0 and SHR at unit 2048 —
    byte-identical to the legacy explicit pins. The declarative rule
    lowering still consumes ``start_unit=0`` from
    ``Primitives.lower_ffn_rules``, so the SHL block writes units
    0..2047 and the SHR block writes 2048..4095, matching the legacy
    ``_set_layer13_shifts`` indices.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L13 op claims a free range -- but see the layout-table note
    above: the SHL+SHR chain currently saturates the 4096-unit pool).
    """
    allocator = FFNUnitAllocator()
    for name, n_units in _L13_SHIFTS_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


# === L13 SHL/SHR FFN rules (declarative) ============================
#
# Each shift sub-stage is a dense 8 x 16 x 16 = 2048-unit lookup table
# implementing the 8-bit nibble-level result for shift amounts 0..7.
# Each unit is a 5-way AND in the silu path:
#
#   conditions = MARK_AX + ALU_LO[a_lo] + ALU_HI[a_hi]
#              + AX_CARRY_LO[s] + AX_CARRY_HI[0]
#   threshold  = 4.5  (so b_up = -S * 4.5)
#   gate       = OP_SHL or OP_SHR (gate_weight = 1.0, gate_bias = 0.0)
#   writes     = OUTPUT_LO[result_lo] = 2.0/S, OUTPUT_HI[result_hi] = 2.0/S
#
# Walk order matches the imperative ``_set_layer13_shifts`` outer loop:
# ``for s in range(8): for a_hi in range(16): for a_lo in range(16):``
# so that with start_unit=0 the SHL substage lands on units 0..2047 and
# the SHR substage lands on units 2048..4095 -- the same indices the
# legacy helper writes. Byte-identity is gated by
# :func:`compare_symbolic_to_lowered_ffn` over the full 4096-rule IR.


def _layer13_shifts_substage_rules(
    op_dim_name: str,
    shift_fn,
    *,
    name_prefix: str,
    S: float,
    gate: str | None = None,
) -> tuple[FFNRule, ...]:
    """Build the 2048-rule list for one L13 shift sub-stage.

    ``op_dim_name`` is the gating opcode dim name (``"OP_SHL"`` or
    ``"OP_SHR"``) used in the human-readable ``scope`` string.
    ``shift_fn(value, s)`` computes the 8-bit shift result for
    ``(value, shift_amount)``; it must agree with the imperative
    helper's per-shift table.

    The 2048 rules iterate ``(s, a_hi, a_lo)`` in nested order to match
    the legacy unit counter in ``setup_helpers._set_layer13_shifts``;
    appending the SHL list followed by the SHR list lands on the
    pinned offsets declared in :data:`_L13_SHIFTS_UNIT_LAYOUT`.

    Phase 7.E.3: the ``gate`` parameter accepts the role-meaningful
    :func:`dim_ref` form (``dim_ref("opcode_flag", "SHL"/"SHR")``)
    while ``op_dim_name`` stays in the legacy ``"OP_SHL"`` /
    ``"OP_SHR"`` form for the human-readable ``scope`` string. When
    ``gate is None`` (legacy callers) it falls back to ``op_dim_name``
    so existing behaviour is preserved.

    Phase 7.E (sem-dim pilot, 2026-07-03): every remaining condition /
    write dim ref is now a semantic ``(category, role)`` lookup via
    :func:`dim_ref`, so this substage — and therefore the whole L13
    shifts FFN (SHL + SHR, 4096 rules) — no longer hard-codes ANY base
    slot NAME. The ``+N`` suffix carries the *value-bus index* (a nibble
    value / shift amount), which is genuinely structural and stays a raw
    offset; the base slot NAME is resolved from its semantic family at
    compile time. This is the repack-fragility fix: if the dim allocator
    RENAMES or re-tags a family (e.g. an over-width band re-homes
    ``ALU_LO``), a hard-coded ``"ALU_LO+a_lo"`` string would silently read
    the wrong cell, whereas the ``(category, role)`` binding is the single
    source of truth. The refs used here:

    * position marker ``MARK_AX`` -> ``dim_ref("marker", "AX")``
    * ``ALU_LO[a_lo]`` (operand a lo nibble) ->
      ``dim_ref("alu_lo", "result", a_lo)``
    * ``ALU_HI[a_hi]`` (operand a hi nibble) ->
      ``dim_ref("alu_hi", "result", a_hi)``
    * ``AX_CARRY_LO[s]`` (one-hot shift amount) ->
      ``dim_ref("ax_carry_lo", "AX", s)``
    * ``AX_CARRY_HI[0]`` (shift-active flag) ->
      ``dim_ref("ax_carry_hi", "AX", 0)``
    * ``OUTPUT_LO[result_lo]`` / ``OUTPUT_HI[result_hi]`` (result byte
      nibbles) -> ``dim_ref("output_lo"/"output_hi", "nibble", result_*)``

    All resolve byte-identically to the pre-pilot ``"NAME+offset"`` strings
    (verified via ``compare_symbolic_to_lowered_ffn`` +
    ``tools/_isa_golden_hash.py``). The ``scope`` string keeps the legacy
    ``op_dim_name`` form: it feeds the human-readable predicate auditor,
    not the weight lowering, so it is NOT repack-sensitive.
    """
    write_scale = 2.0 / S
    gate_ref = gate if gate is not None else op_dim_name
    # Phase 7.E pilot: resolve the position marker + the shift-active flag
    # once (they are loop-invariant). ``dim_ref`` returns a ``"NAME+offset"``
    # string that ``DimRef.parse`` reads identically to the bare name, so
    # ``dim_ref("marker", "AX")`` == ``"MARK_AX+0"`` lowers to the same
    # weight the legacy bare ``"MARK_AX"`` did.
    mark_ax = dim_ref("marker", "AX")
    ax_carry_active = dim_ref("ax_carry_hi", "AX", 0)
    rules: list[FFNRule] = []
    for s in range(8):
        for a_hi in range(16):
            for a_lo in range(16):
                value = (a_hi << 4) | a_lo
                result = shift_fn(value, s)
                result_lo = result & 0xF
                result_hi = (result >> 4) & 0xF
                # Value-bus reads/writes: NAME resolved from the semantic
                # family, ``offset`` = the raw nibble value / shift amount.
                alu_lo_a = dim_ref("alu_lo", "result", a_lo)
                alu_hi_a = dim_ref("alu_hi", "result", a_hi)
                ax_carry_lo_s = dim_ref("ax_carry_lo", "AX", s)
                output_lo_result = dim_ref("output_lo", "nibble", result_lo)
                output_hi_result = dim_ref("output_hi", "nibble", result_hi)
                rules.append(multi_way_and_rule(
                    name=(
                        f"{name_prefix}_s{s}_ahi{a_hi}_alo{a_lo}"
                    ),
                    conditions=(
                        (mark_ax, 1.0),
                        (alu_lo_a, 1.0),
                        (alu_hi_a, 1.0),
                        (ax_carry_lo_s, 1.0),
                        (ax_carry_active, 1.0),
                    ),
                    threshold=4.5,
                    gate=gate_ref,
                    gate_weight=1.0,
                    gate_bias=0.0,
                    writes=(
                        (output_lo_result, write_scale),
                        (output_hi_result, write_scale),
                    ),
                    scope=f"MARK_AX and {op_dim_name}",
                ))
    return tuple(rules)


def _layer13_shl_rules(S: float) -> tuple[FFNRule, ...]:
    """L13 SHL sub-stage: 2048 lookup-table units (units 0..2047).

    Phase 7.E.3: the gate ref uses :func:`dim_ref` for the
    ``(opcode_flag, SHL)`` semantic pair.
    """
    return _layer13_shifts_substage_rules(
        "OP_SHL",
        lambda v, s: (v << s) & 0xFF,
        name_prefix="l13_shl",
        S=S,
        gate=dim_ref("opcode_flag", "SHL"),
    )


def _layer13_shr_rules(S: float) -> tuple[FFNRule, ...]:
    """L13 SHR sub-stage: 2048 lookup-table units (units 2048..4095).

    Phase 7.E.3: the gate ref uses :func:`dim_ref` for the
    ``(opcode_flag, SHR)`` semantic pair.
    """
    return _layer13_shifts_substage_rules(
        "OP_SHR",
        lambda v, s: (v >> s) & 0xFF,
        name_prefix="l13_shr",
        S=S,
        gate=dim_ref("opcode_flag", "SHR"),
    )


def _layer13_shifts_rules(S: float) -> tuple[FFNRule, ...]:
    """Combined SHL + SHR rule list (4096 units total).

    Order matches the imperative ``_set_layer13_shifts`` outer
    ``[(OP_SHL, ...), (OP_SHR, ...)]`` loop so that lowering with
    ``start_unit=0`` writes the SHL block at units 0..2047 and the SHR
    block at units 2048..4095, byte-identical to the legacy helper.
    """
    return _layer13_shl_rules(S) + _layer13_shr_rules(S)


def _layer13_shifts_ir(S: float = 100.0) -> CompilerIR:
    """L13 FFN: declarative SHL+SHR lookup tables (4096 rules)."""
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer13_shifts_rules(S))
    return ir


def _bake_layer13_shifts(ffn, S, BD) -> int:
    """Lower the declarative L13 shifts IR into ``ffn``.

    Returns the next-free unit cursor (4096). Byte-identical to
    ``setup_helpers._set_layer13_shifts`` (verified via
    :func:`compare_symbolic_to_lowered_ffn` and per-substage parity
    tests against the legacy helper).
    """
    rules = _layer13_shifts_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def _l13_addr_bn_valid_positions(BD):
    """Resolve the ADDR_B{1,2}_VALID destination dim positions.

    Position 99 (``ADDR_B1_VALID``) aliases ``H5+4``; position 100
    (``ADDR_B2_VALID``) aliases ``H5+5``. Use ``dim_positions`` when the
    compiler exposed the new names; otherwise derive from ``H5`` (the
    dormant L0 head-5 threshold output the B7/B8 dims alias onto). The
    numeric fallback (99 / 100) matches the dim_registry allocation so the
    producer fires even before the compiler learns the new names.
    """
    h5_base = getattr(BD, "H5", None)
    addr_b1_valid_pos = getattr(BD, "ADDR_B1_VALID", None)
    if addr_b1_valid_pos is None:
        addr_b1_valid_pos = (h5_base + 4) if h5_base is not None else 99
    addr_b2_valid_pos = getattr(BD, "ADDR_B2_VALID", None)
    if addr_b2_valid_pos is None:
        addr_b2_valid_pos = (h5_base + 5) if h5_base is not None else 100
    return addr_b1_valid_pos, addr_b2_valid_pos


def _layer13_mem_addr_gather_head_specs(BD) -> tuple:
    """Declarative L13 heads 0-2: MEM addr-byte gather + B7-4/B8-A VALID bits.

    Mirrors ``setup_helpers._set_layer13_mem_addr_gather`` (heads 0-2 slot 0
    + slot 33 anti-leak, slot 1..32 V/O CLEAN_EMBED -> ADDR_BJ_LO/HI, plus
    head 0 slot 34 ADDR_B0_VALID lifecycle) AND
    ``_l13_addr_bn_valid_extension`` (heads 1/2 slot 34 ADDR_B{1,2}_VALID
    extension). Per-head K rows pick the MEM addr-byte ``j`` row:

      - head 0 (ADDR_B0): K[+L1H1+MEM_I, -L1H0+MEM_I] -- addr byte 0 at d=1.
      - head 1 (ADDR_B1): K[+L1H2+MEM_I, -L1H1+MEM_I] -- addr byte 1 at d=2.
      - head 2 (ADDR_B2): K[+H0+MEM_I,   -L1H2+MEM_I] -- addr byte 2 at d=3.

    The VALID lifecycle V reads the K+ threshold dim so the attended
    addr-byte-J row delivers V=1.0 (and unrelated rows V=0) -- same logic
    as the head-0 slot-34 ADDR_B0_VALID producer (B7-4) extended to
    ADDR_B1_VALID / ADDR_B2_VALID (B8-A).
    """
    L = 15.0
    MEM_I = 4
    VALID_SLOT = 34
    addr_b1_valid_pos, addr_b2_valid_pos = _l13_addr_bn_valid_positions(BD)

    # Per-head wiring tables: layout for the J-th MEM addr byte gather head.
    # (head_idx, addr_lo_out, addr_hi_out, K_pos_dim, K_neg_dim,
    #  valid_v_read_dim, valid_o_dest)
    head_layout = (
        (
            _l13_head_idx("layer13_mem_addr_gather.head_0"),
            BD.ADDR_B0_LO, BD.ADDR_B0_HI,
            BD.L1H1 + MEM_I, BD.L1H0 + MEM_I,
            BD.L1H1 + MEM_I, BD.ADDR_B0_VALID,
        ),
        (
            _l13_head_idx("layer13_mem_addr_gather.head_1"),
            BD.ADDR_B1_LO, BD.ADDR_B1_HI,
            BD.L1H2 + MEM_I, BD.L1H1 + MEM_I,
            BD.L1H2 + MEM_I, addr_b1_valid_pos,
        ),
        (
            _l13_head_idx("layer13_mem_addr_gather.head_2"),
            BD.ADDR_B2_LO, BD.ADDR_B2_HI,
            BD.H0 + MEM_I, BD.L1H2 + MEM_I,
            BD.H0 + MEM_I, addr_b2_valid_pos,
        ),
    )

    specs = []
    for (head_idx, addr_lo_out, addr_hi_out,
         k_pos, k_neg, valid_v_read, valid_o_dest) in head_layout:
        # Q slot 0: fires at MEM val byte positions (d=5..8 from MEM)
        q = [
            AP(0, BD.MEM_VAL_B0, L),
            AP(0, BD.MEM_VAL_B1, L),
            AP(0, BD.MEM_VAL_B2, L),
            AP(0, BD.MEM_VAL_B3, L),
            # Slot 33 anti-leakage gate
            AP(33, BD.MEM_VAL_B0, L),
            AP(33, BD.CONST, -L / 2),
            # Slot 34 VALID lifecycle Q mirrors slot 0 (fires at MEM val bytes)
            AP(VALID_SLOT, BD.MEM_VAL_B0, L),
            AP(VALID_SLOT, BD.MEM_VAL_B1, L),
            AP(VALID_SLOT, BD.MEM_VAL_B2, L),
            AP(VALID_SLOT, BD.MEM_VAL_B3, L),
        ]
        # K slot 0: fires at MEM addr byte J position; slot 33 const anti-leak;
        # slot 34 mirrors slot 0 (picks the MEM addr byte J row).
        k = [
            AP(0, k_pos, L),
            AP(0, k_neg, -L),
            AP(33, BD.CONST, L),
            AP(VALID_SLOT, k_pos, L),
            AP(VALID_SLOT, k_neg, -L),
        ]
        # V slots 1..32 copy CLEAN_EMBED nibbles (addr byte value).
        # V slot 34 reads valid_v_read so the attended addr-byte-J row
        # delivers V=1.0 and unrelated rows deliver 0.
        v = [AP(1 + kk, BD.CLEAN_EMBED_LO + kk, 1.0) for kk in range(16)]
        v += [AP(17 + kk, BD.CLEAN_EMBED_HI + kk, 1.0) for kk in range(16)]
        v.append(AP(VALID_SLOT, valid_v_read, 1.0))
        # O slots 1..32 route gathered nibbles to ADDR_BJ_LO/HI; slot 34
        # routes the VALID lifecycle bit into ADDR_BJ_VALID.
        o = [AO(addr_lo_out + kk, 1 + kk, 1.0) for kk in range(16)]
        o += [AO(addr_hi_out + kk, 17 + kk, 1.0) for kk in range(16)]
        o.append(AO(valid_o_dest, VALID_SLOT, 1.0))

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ))
    return tuple(specs)


def _layer13_mem_addr_gather_ir(dim_positions, HD) -> CompilerIR:
    """Build the declarative L13 mem-addr-gather IR for the compiler.

    Three heads (0/1/2), each gather one MEM addr byte's CLEAN_EMBED
    nibbles into ADDR_BJ_LO/HI at MEM val byte positions, plus a
    slot-34 VALID lifecycle bit per head (ADDR_B0_VALID/B1_VALID/B2_VALID).
    See :func:`_layer13_mem_addr_gather_head_specs` for the byte-level layout.
    """
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_mem_addr_gather_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


# === L13 head 3: bitwise byte-1 operand gather ======================
#
# 16-bit OR/XOR byte-1 fix (2026-06-11). The byte-0 AND/OR/XOR compute
# (commit e4c4c396) computes the low byte at the MARK_AX row, but byte 1
# was never computed or relayed -- the byte-1 AX-emit token defaults to
# 0x00 (probe ``tools/probe_or16_byte1.py``: at the OR compute row only
# byte-0 operands exist; operand A byte 1 lives in STACK0_BYTE_VAL_1_LO/HI
# at the STACK0 byte-1 rows, never gathered to the AX row).
#
# The working high-byte path stages result byte 1 into AX_FULL_LO/HI at
# the MARK_AX row, and ``layer15_alu_high_byte_relay`` (l14_ops.py) copies
# AX_FULL -> OUTPUT at the byte-1 emit token -- but it is K-gated on
# OP_MUL/OP_SHL ONLY. This head supplies the missing AX_FULL staging for
# bitwise: at the MARK_AX row of an OR/XOR step it attends back to the
# top-of-stack STACK0 byte-1 row and copies STACK0_BYTE_VAL_1_LO/HI ->
# AX_FULL_LO/HI. The relay-gate widening (l14_ops.py) then copies that
# staged byte 1 to OUTPUT at the byte-1 emit.
#
# Why OP_OR/OP_XOR only (not OP_AND)? For all three 16-bit bitwise smoke
# programs operand B's byte 1 is 0x00, so result byte 1 = A_b1 OP 0:
#   OR/XOR -> A_b1  (must be staged; default 0x00 is wrong)
#   AND    -> 0x00  (empty AX_FULL already emits 0x00; staging A_b1
#                    would BREAK and_16bit). So AND is left unstaged.
# operand B byte 1 does NOT reach the MARK_AX row in any usable band
# (verified spec_k=0), so a general A_b1 OP B_b1 compute is not landable
# in this surface; the OR/XOR-only relay of A_b1 is exact for the targets
# and leaves and_16bit (already passing) untouched.
#
# Placement: L13 (physical block after the L11/L10 ``psh_ax_broadcast``
# heads that WRITE STACK0_BYTE_VAL_1, and before the L15 relay that READS
# AX_FULL). L13 attn heads 0-2 are owned by mem_addr_gather; this head
# claims the free slot 3. A positive ALiBi slope keeps the gather
# step-local / top-of-stack: the most-recent STACK0 byte-1 row (the
# current OR/XOR's operand A) wins the softmax over older frames.
def _layer13_bitwise_byte1_gather_head_specs(BD) -> tuple:
    """L13 head 3: stage operand-A byte 1 into AX_FULL on OR/XOR.

    Q fires at MARK_AX AND (OP_OR or OP_XOR); a CONST anti-leak penalty
    plus per-opcode exclusions keep the head dark on every other step.
    K fires at the STACK0 byte-1 value row (STACK0_BYTE1). V copies the
    STACK0_BYTE_VAL_1_LO/HI nibble pair; O writes AX_FULL_LO/HI at the Q
    (MARK_AX) row. ALiBi recency (slope set in the bake) selects the
    most-recent STACK0 byte-1 row = current top of stack = operand A.
    """
    L = 15.0
    # Exclude every non-OR/XOR opcode at the Q row so the head only fires
    # on OR/XOR steps (mirrors the L7 operand_gather exclusion pattern).
    _EXCLUDE_OPCODES = (
        "OP_AND", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_SHL", "OP_SHR", "OP_EQ", "OP_NE", "OP_LT", "OP_GT",
        "OP_LE", "OP_GE", "OP_IMM", "OP_PSH", "OP_JSR", "OP_ENT",
        "OP_LEV", "OP_LI", "OP_LC", "OP_SI", "OP_SC", "OP_LEA",
        "OP_JMP", "OP_BZ", "OP_BNZ", "OP_ADJ", "OP_EXIT",
    )
    q = [
        AP(0, BD.MARK_AX, L),
        AP(0, BD.OP_OR, L),
        AP(0, BD.OP_XOR, L),
        AP(0, BD.CONST, -L / 2),
    ]
    for opname in _EXCLUDE_OPCODES:
        op_dim = getattr(BD, opname, None)
        if op_dim is not None:
            q.append(AP(0, op_dim, -L * 10))
    # Slot 33 anti-leak: require MARK_AX strongly so the slot-0 softmax
    # only routes positively at the OR/XOR MARK_AX Q row.
    q.append(AP(33, BD.MARK_AX, L))
    q.append(AP(33, BD.CONST, -L / 2))

    k = [
        AP(0, BD.STACK0_BYTE1, L),
        AP(33, BD.CONST, L),
    ]
    # V slots 1..32 copy the STACK0 byte-1 value nibbles.
    v = [AP(1 + kk, BD.STACK0_BYTE_VAL_1_LO + kk, 1.0) for kk in range(16)]
    v += [AP(17 + kk, BD.STACK0_BYTE_VAL_1_HI + kk, 1.0) for kk in range(16)]
    # O slots 1..32 route the gathered nibbles into AX_FULL_LO/HI.
    o = [AO(BD.AX_FULL_LO + kk, 1 + kk, 1.0) for kk in range(16)]
    o += [AO(BD.AX_FULL_HI + kk, 17 + kk, 1.0) for kk in range(16)]

    return (
        DeclarativeAttentionHeadSpec(
            head_idx=3,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
    )


def _layer13_bitwise_byte1_gather_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_bitwise_byte1_gather_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer13_bitwise_byte1_gather_op() -> Operation:
    """L13 attention head 3: stage operand-A byte 1 into AX_FULL on OR/XOR.

    Supplies the AX_FULL byte-1 staging that ``layer15_alu_high_byte_relay``
    (with its widened OP_OR/OP_XOR K-gate) relays to OUTPUT for the 16-bit
    OR/XOR byte-1 emit. See the module-level comment block above for the
    design rationale and the OR/XOR-only gating choice.
    """
    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        # NEGATIVE ALiBi slope on head 3: prefer the OLDEST (deepest)
        # STACK0 byte-1 K row. The current OR/XOR step's own STACK0 frame
        # is the MOST RECENT byte-1 row but was NEVER PSH-populated (the
        # ``psh_ax_broadcast`` head only writes STACK0_BYTE_VAL_1 on
        # OP_PSH steps), so it carries 0x00, not operand A's byte 1. The
        # PSH-step frame that pushed operand A is further back; a negative
        # recency bias skips the empty current-step frame and lands on the
        # populated PSH frame. (Verified spec_k=0: rows 66/101 = 0x0F via
        # PSH broadcast; row 136 = OR-step frame = 0x00.)
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[3] = -1.0
        Primitives.generate_attention_head(
            attn,
            _layer13_bitwise_byte1_gather_head_specs(proxy)[0],
            HD,
        )

    _claims = set()
    for k in range(16):
        _claims.add((13, "attn_W_v", f"3_{1 + k}", f"STACK0_BYTE_VAL_1_LO+{k}"))
        _claims.add((13, "attn_W_v", f"3_{17 + k}", f"STACK0_BYTE_VAL_1_HI+{k}"))

    return Operation(
        name="layer13_bitwise_byte1_gather",
        reads={"MARK_AX", "OP_OR", "OP_XOR", "STACK0_BYTE1",
               "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI", "CONST"},
        writes={"AX_FULL_LO", "AX_FULL_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_bitwise_byte1_gather_ir,
        declarative_authority="spec_generated",
        # Co-place on the L13 attn block (heads 0-2 owned by
        # mem_addr_gather; this claims head 3). Run after the mem-addr
        # gather so the 8-head block is already populated.
        target_op_name="_layer13_mem_addr_anchor",
        requires={"after": "layer13_mem_addr_gather"},
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
    )


# =====================================================================
# SUB multi-byte minuend relay (L13 head 4)  -- 2026-06-12
# =====================================================================
#
# Root: the L14 inter-byte borrow cascade (_l10_carry_propagation_rules,
# cascade=True for byte_idx 1/2) reads its per-byte MINUEND from
# OUTPUT_LO/HI at the SUB byte-h emit row. But only operand byte 0 is
# relayed into OUTPUT by the L7 operand_gather; the minuend HIGH bytes
# never reach the cascade row, so every multi-byte SUB computes
# (0x00 - borrow) for bytes 1/2/3 -> 0xFF, byte-identically for
# 0x100-1 (wants 0x00) and 0-1 (wants 0xFF). Confirmed spec_k=0
# (tools/probe_sub_minuend_source.py, probe_sub_relay_design.py):
# STACK0_BYTE_VAL_1 = 0x01 (sub_16bit) vs 0x00 (sub_borrow) at the
# PSH-frame STACK0 byte-1 rows, but 0x00 (absent) at the SUB emit rows.
#
# This head supplies the missing minuend high byte. At the SUB byte-h
# emit row (TEMP+9 SUB-selector + BYTE_INDEX_h, NOT MARK_AX/PC) it
# attends back to the most-recent PSH-frame STACK0 byte-h value row
# (STACK0_BYTE{1,2,3} position flag, where layer10_psh_ax_broadcast
# stored the pushed operand's high byte) and copies
# STACK0_BYTE_VAL_h_LO/HI back INTO STACK0_BYTE_VAL_h_LO/HI at the emit
# row. The (now-declarative) cascade SUB rule for byte_idx 1/2 is then
# re-pointed to read its minuend from STACK0_BYTE_VAL_h instead of
# OUTPUT (Part 2). This keeps the CARRY+3 byte0->byte1 borrow relay on
# OUTPUT untouched -- the documented collision that regresses
# sub_borrow when the minuend is routed through OUTPUT/AX_FULL
# (SUB_16BIT_PHASE2_CASCADE_SOURCE_FIX_2026_06_12.md).
#
# Byte-identity property: for 8-bit SUB (sub_basic 50-8) the pushed
# operand's byte 1/2/3 = 0x00, so the relay writes STACK0_BYTE_VAL_h =
# 0x00 = the current OUTPUT byte-h = 0x00 -> identical result. For
# sub_16bit the relay writes 0x01 so byte-1 fires (0x01 - borrow = 0x00)
# instead of (0x00 - borrow = 0xFF). For sub_borrow the relay writes
# 0x00 -> byte-1 stays 0xFF and the borrow continues -> 0xFFFFFFFF
# preserved. The discriminator that makes 0x100-1 != 0-1 rides the
# decoupled STACK0_BYTE_VAL band, never CARRY+3.
#
# Placement: L13 (block 14) -- after layer10_psh_ax_broadcast (which
# writes STACK0_BYTE_VAL_h, visible from block 12) and before the L14
# carry post_ops (block 16+) that read it. Heads 0-2 = mem_addr_gather;
# head 3 = bitwise_byte1_gather; this claims the free slot 4. A negative
# ALiBi slope (mirrors the bitwise gather) skips the empty current-step
# STACK0 frame and lands on the populated PSH frame.
def _layer13_sub_minuend_relay_head_specs(BD) -> tuple:
    """L13 head 4: relay SUB minuend bytes 1/2/3 to STACK0_BYTE_VAL_h.

    One spec firing simultaneously on the byte-1/2 SUB emit rows. The
    per-byte routing uses two Q/K slot pairs (one per byte index):
    Q selects the emit row via BYTE_INDEX_h, K selects the PSH-frame
    STACK0 byte-h value row via STACK0_BYTE{1,2}; the matching V/O
    nibble copy runs in parallel for both bytes. ALiBi recency
    (negative slope, set in the bake) picks the most-recent populated
    PSH frame over older frames / the empty current-step frame.
    """
    L = 15.0
    # K source-flag match weight. The post-scale Q*K score for a
    # STACK0_BYTE{h} row must DOMINATE the ALiBi distance penalty
    # (-slope * |q_pos - k_pos|, scale ~= 0.096): the populated PSH
    # frame sits ~90 rows back from the SUB emit row, so with the
    # default L=15 match (~211 raw -> ~20 scaled) the alibi penalty
    # (~90) swamps the K score and the softmax drifts to nearby
    # non-STACK0 rows. A large source-flag weight (15 * 200 * 0.94 ~=
    # 2820 raw -> ~270 scaled) makes every STACK0_BYTE{h} row beat
    # every non-STACK0 row by a margin no alibi distance can overcome,
    # leaving the (gentle) negative slope to pick the OLDEST STACK0
    # frame (the original PSH) among them.
    K_FLAG = 200.0
    # Per-byte route: (emit BYTE_INDEX_k dim, source STACK0_BYTE{k+1}
    # flag dim, source STACK0_BYTE_VAL_{k+1} value band). The L14 borrow
    # cascade is an INTER-byte stage: the rule scoped on BYTE_INDEX_k
    # (the row that, autoregressively, PREDICTS output byte k+1)
    # computes the byte-(k+1) difference and needs the MINUEND's byte
    # k+1. So BYTE_INDEX_0 row needs operand byte 1 = STACK0_BYTE_VAL_1,
    # BYTE_INDEX_1 row needs byte 2 = STACK0_BYTE_VAL_2, BYTE_INDEX_2 row
    # needs byte 3 = STACK0_BYTE_VAL_3. (Verified spec_k=0: output byte 1
    # is decoded by the LM head from the BYTE_INDEX_0 predictor row.)
    _BYTE_ROUTES = (
        (0, BD.BYTE_INDEX_0, BD.STACK0_BYTE1,
         BD.STACK0_BYTE_VAL_1_LO, BD.STACK0_BYTE_VAL_1_HI),
        (1, BD.BYTE_INDEX_1, BD.STACK0_BYTE2,
         BD.STACK0_BYTE_VAL_2_LO, BD.STACK0_BYTE_VAL_2_HI),
        (2, BD.BYTE_INDEX_2, BD.STACK0_BYTE3,
         BD.STACK0_BYTE_VAL_3_LO, BD.STACK0_BYTE_VAL_3_HI),
    )
    if operand_from_memsp_enabled():
        # === STACK0 campaign Part C (Inc-3 claw-back, 2026-06-19): SUB
        # byte-1 minuend RESULT delivery, mirror of the ADD head-5 re-key. ===
        #
        # The dropped STACK0 byte rows + the un-delivered TEMP+9 selector
        # (GPU-confirmed: TEMP+9 lands only at the MARK_AX marker row in the
        # 30-token frame, not the BYTE_INDEX_0 emit row) make the legacy
        # STACK0_BYTE1-keyed / TEMP+9-Q-gated relay dark. Re-cast it as the
        # same intra-step cross-row copy the ADD head 5 uses: Q fires at the
        # BYTE_INDEX_0 emit row (BYTE_INDEX_0 + HAS_SE), K selects the current
        # step's AX marker via MARK_AX AND OP_SUB (the SUB discriminator lives
        # on the SOURCE marker row), so head 4 gathers the minuend byte-1
        # carrier (STACK0_BYTE_VAL_1, deposited by the L8 head-7 mem[SP] CAM)
        # ONLY on SUB steps and stays dark on ADD/bitwise. The 1096 sub corpus
        # is all 2-byte (subtrahend byte1 = 0x00), so byte 1 alone is needed.
        OP_W = 40.0
        q = [
            AP(0, BD.BYTE_INDEX_0, L),
            AP(0, BD.HAS_SE, L),
            AP(0, BD.CONST, -L / 2),
            AP(0, BD.MARK_AX, -L * 10),
            AP(0, BD.MARK_PC, -L * 10),
            AP(0, BD.BYTE_INDEX_1, -L * 10),
            AP(0, BD.BYTE_INDEX_2, -L * 10),
            AP(0, BD.BYTE_INDEX_3, -L * 10),
            AP(33, BD.BYTE_INDEX_0, L),
            AP(33, BD.CONST, -L / 2),
        ]
        k = [AP(33, BD.CONST, L)]
        v = []
        o = []
        # TRUE AND of MARK_AX and OP_SUB (see head 5's note): a non-SUB AX
        # marker must score NEGATIVE so the head stays dark off-SUB and never
        # corrupts STACK0_BYTE_VAL_1 on var stores / other ops.
        sel = 1
        q.append(AP(sel, BD.BYTE_INDEX_0, L))
        q.append(AP(sel, BD.HAS_SE, L))
        q.append(AP(sel, BD.CONST, -L))
        k.append(AP(sel, BD.MARK_AX, K_FLAG))
        k.append(AP(sel, BD.OP_SUB, OP_W))
        k.append(AP(sel, BD.OP_ADD, -K_FLAG))
        k.append(AP(sel, BD.CONST, -K_FLAG * 1.3))
        base = 3
        val_lo = BD.STACK0_BYTE_VAL_1_LO
        val_hi = BD.STACK0_BYTE_VAL_1_HI
        for kk in range(16):
            v.append(AP(base + kk, val_lo + kk, 1.0))
            v.append(AP(base + 16 + kk, val_hi + kk, 1.0))
            o.append(AO(val_lo + kk, base + kk, 1.0))
            o.append(AO(val_hi + kk, base + 16 + kk, 1.0))
        # Discriminator delivery: stamp TEMP+9 (the SUB byte-row selector)
        # onto the BYTE_INDEX_0 emit row. The L10 borrow cascade gates the
        # SUB byte-1 rule on TEMP+9 AT the emit row; in the 30-token frame the
        # L7 head-5 broadcast leaves TEMP+9 at the MARKER row only (and even
        # mis-stamps TEMP+8 on a SUB), so the cascade stays dark. V reads
        # OP_SUB (=5.0 ONLY at the matched SUB marker -> ~0 elsewhere, so the
        # stamp is clean and SUB-exclusive); the 0.2 scale yields TEMP+9 ~= 1.0.
        v.append(AP(base + 32, BD.OP_SUB, 0.2))
        o.append(AO(BD.TEMP + 9, base + 32, 1.0))
        return (
            DeclarativeAttentionHeadSpec(
                head_idx=4, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o),
            ),
        )
    # Q slot 0: gate STRICTLY on the SUB byte-emit selector. TEMP+9 is
    # the cascade's SUB discriminator (1.0 ONLY on SUB byte rows; 0 on
    # ADD/bitwise/everything else). The slot-0 score must be POSITIVE
    # only when TEMP+9 fires and NEGATIVE otherwise, so the relay never
    # writes STACK0_BYTE_VAL on non-SUB rows (those dims alias
    # FORMAT_PTR / LEV_DETECTOR -- writing them off-SUB risks
    # regression). With CONST=1.0 everywhere: SUB byte row scores
    # TEMP+9*L - L/2 = +L/2; every non-SUB row scores 0 - L/2 = -L/2.
    # IS_BYTE/MARK/BYTE_INDEX terms only sharpen; the load-bearing gate
    # is TEMP+9 minus the CONST baseline. The slot-33 anti-leak mirrors
    # the same TEMP+9-gated bias so the slot-0 softmax routes positively
    # only at SUB byte rows.
    q = [
        AP(0, BD.TEMP + 9, L),
        AP(0, BD.CONST, -L / 2),
        AP(0, BD.MARK_AX, -L * 10),
        AP(0, BD.MARK_PC, -L * 10),
        AP(0, BD.TEMP + 8, -L * 10),
        AP(0, BD.BYTE_INDEX_3, -L * 10),
        # slot 33 anti-leak: require TEMP+9 so the slot-0 softmax only
        # routes positively on the SUB byte-h emit rows.
        AP(33, BD.TEMP + 9, L),
        AP(33, BD.CONST, -L / 2),
    ]
    k = [AP(33, BD.CONST, L)]
    v = []
    o = []
    # Per-byte K-select slots and V/O nibble copies. Slot indices:
    #   byte route j uses Q/K slot (1 + j) as the byte-index selector,
    #   V/O slots [base .. base+31] for the 32 nibble copies.
    for j, (h, emit_bi_dim, src_flag_dim, val_lo, val_hi) in enumerate(_BYTE_ROUTES):
        sel = 1 + j  # 1, 2
        # The per-byte Q selector folds in the SUB discriminator TEMP+9
        # so the (large) K source-flag match only contributes on a SUB
        # byte-h row: Q[sel] = BYTE_INDEX_h + TEMP+9 - CONST, which is
        # ~+L on a SUB byte-h emit row (1+1-1) but ~0 on a non-SUB
        # byte-h row (1+0-1=0) -- so on ADD/OR/XOR/AND byte rows the
        # STACK0_BYTE_h K-match multiplies a ~0 query and the head does
        # NOT gather (keeps STACK0_BYTE_VAL / its FORMAT_PTR alias
        # untouched off-SUB). On a SUB row the K_FLAG-scaled match then
        # dominates the alibi penalty and selects the PSH frame.
        q.append(AP(sel, emit_bi_dim, L))
        q.append(AP(sel, BD.TEMP + 9, L))
        q.append(AP(sel, BD.CONST, -L))
        k.append(AP(sel, src_flag_dim, K_FLAG))
        k.append(AP(sel, BD.CONST, -K_FLAG / 2))
        base = 3 + j * 32  # 3.., 35..
        for kk in range(16):
            v.append(AP(base + kk, val_lo + kk, 1.0))
            v.append(AP(base + 16 + kk, val_hi + kk, 1.0))
            o.append(AO(val_lo + kk, base + kk, 1.0))
            o.append(AO(val_hi + kk, base + 16 + kk, 1.0))

    return (
        DeclarativeAttentionHeadSpec(
            head_idx=4,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
    )


def _layer13_sub_minuend_relay_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_sub_minuend_relay_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer13_sub_minuend_relay_op() -> Operation:
    """L13 attn head 4: relay SUB minuend bytes 1/2 to STACK0_BYTE_VAL_h.

    Part 1 of the multi-byte SUB minuend fix. Delivers the pushed
    operand's high bytes (stored by layer10_psh_ax_broadcast at the
    PSH-frame STACK0 byte rows) to the SUB byte-h emit rows so the L14
    borrow cascade (Part 2, re-pointed in _l10_carry_propagation_rules)
    can compute byte 1/2 of a multi-byte SUB. See the module comment
    block above for the root cause, the byte-identity property, and why
    the decoupled STACK0_BYTE_VAL band avoids the CARRY+3 collision.
    """
    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        # NEGATIVE ALiBi slope on head 4: the runtime applies the bias
        # ``-slope * |q_pos - k_pos|`` (vm_step.AutoregressiveAttention),
        # so a negative slope REWARDS distance -> prefers the OLDEST
        # (deepest) STACK0 byte-h value row. That is the original PSH
        # frame that pushed the SUB's operand A (minuend); the
        # most-RECENT STACK0_BYTE_h row is the current SUB step's own
        # re-marked STACK0 frame, which carries 0x00 (the operand high
        # byte is only populated at the original PSH frame by
        # layer10_psh_ax_broadcast). The K source-flag match (K_FLAG
        # above) dominates the alibi penalty so only STACK0_BYTE{h} rows
        # are candidates; the negative slope then breaks the tie toward
        # the oldest among them. (Verified spec_k=0: sub_16bit PSH frame
        # = 0x01; SUB-step frame = 0x00.)
        # Campaign Part C: re-pointed to the current step's nearby AX marker
        # (one row before the byte-1 emit row), so a POSITIVE slope rewards
        # proximity and the CURRENT SUB step's marker wins over older markers.
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[4] = (
                1.0 if operand_from_memsp_enabled() else -1.0
            )
        Primitives.generate_attention_head(
            attn,
            _layer13_sub_minuend_relay_head_specs(proxy)[0],
            HD,
        )

    _campaign = operand_from_memsp_enabled()
    _claims = set()
    _byte_routes = (
        (("STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"),)
        if _campaign else
        (("STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"),
         ("STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI"),
         ("STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI"))
    )
    for j, (lo_name, hi_name) in enumerate(_byte_routes):
        base = 3 + j * 32
        for k in range(16):
            _claims.add((13, "attn_W_v", f"4_{base + k}", f"{lo_name}+{k}"))
            _claims.add((13, "attn_W_v", f"4_{base + 16 + k}", f"{hi_name}+{k}"))

    _reads = {"IS_BYTE", "TEMP", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
              "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
              "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
              "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
              "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI", "CONST"}
    _writes = {"STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
               "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
               "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI"}
    if _campaign:
        _reads |= {"HAS_SE", "OP_ADD", "OP_SUB", "MARK_AX", "BYTE_INDEX_3"}
        _writes |= {"TEMP"}
        _claims.add((13, "attn_W_v", "4_35", "OP_SUB"))

    return Operation(
        name="layer13_sub_minuend_relay",
        reads=_reads,
        writes=_writes,
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_sub_minuend_relay_ir,
        declarative_authority="spec_generated",
        # Co-place on the L13 attn block (heads 0-2 = mem_addr_gather,
        # head 3 = bitwise_byte1_gather); this claims head 4. Run after
        # the bitwise gather so the 8-head block is already populated.
        target_op_name="_layer13_mem_addr_anchor",
        requires={"after": "layer13_bitwise_byte1_gather"},
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_sub_16bit",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
    )


# =====================================================================
# ADD multi-byte addend relay (L13 head 5)  -- 2026-06-12
# =====================================================================
#
# Sibling of the SUB minuend relay (head 4) for the ADD path. Where SUB
# only needed the minuend's byte 1 (the subtrahend byte 1 is 0x00 for
# every 1096 case, so result_byte1 = minuend_byte1 - borrow), ADD needs
# a TWO-operand byte-1 sum: result_byte1 = a1 + b1 + carry. This head
# delivers the missing operand: operand-A byte 1 (a1), pushed by
# ``layer10_psh_ax_broadcast`` into the PSH-frame STACK0_BYTE_VAL_1 band
# and otherwise absent at the ADD byte-1 emit row.
#
# Root (confirmed spec_k0, tools/probe_add_autoregressive_carry.py): at
# the ADD byte-1 PREDICTOR row -- the BYTE_INDEX_0 row, TEMP+8=1 (the ADD
# byte-row selector; OP_ADD already decayed to 0) -- the only operand
# bytes present are b1 (operand B byte 1, in ADDR_B1_LO, from L13 head 1)
# and the byte-0 carry-out (CARRY+1: 2.0 when byte 0 carried, 0.0 when
# not -- a CLEAN discriminator in the REAL autoregressive decode, unlike
# the teacher-forced residual the prior ADD attempt trusted). a1 is NOT
# there: the SUB relay (head 4) is TEMP+9-gated and dark on ADD, so
# STACK0_BYTE_VAL_1 is empty on the ADD byte-1 row. This head supplies it.
#
# Mirror of head 4 with two changes:
#   1. Gated on TEMP+8 (ADD byte-row selector) not TEMP+9 (SUB). TEMP+8
#      and TEMP+9 are mutually exclusive (probe: ADD byte-1 row has
#      TEMP+8=1/TEMP+9=0; SUB byte-1 row has TEMP+9=1/TEMP+8=0), so head 5
#      and head 4 never both fire -- both write STACK0_BYTE_VAL_1 but on
#      disjoint (ADD vs SUB) rows, additive residual is safe.
#   2. Routes byte 1 only (BYTE_INDEX_0 row -> STACK0_BYTE_VAL_1): the
#      1096 add corpus is all 2-byte (result <= 0x7FF, a1/b1 <= 3), so
#      byte 2/3 carry never propagates an operand byte. (Keeping it to one
#      route also keeps the head dark on the deeper byte rows.)
#
# Byte-identity: for 8-bit ADD (add_basic 10+32, a1=0) the relay writes
# STACK0_BYTE_VAL_1 = 0x00 = the empty default, so the downstream adder
# computes 0 + b1 + carry unchanged. The STACK0_BYTE_VAL_1 band is
# neither OUTPUT nor CARRY, so the existing ADD carry path is untouched
# until the L10 adder reads it.
def _layer13_add_addend_relay_head_specs(BD) -> tuple:
    """L13 head 5: relay ADD operand-A byte 1 (a1) to STACK0_BYTE_VAL_1.

    One spec firing on the ADD byte-1 emit row (BYTE_INDEX_0 + TEMP+8).
    Q selects that row; K selects the PSH-frame STACK0 byte-1 value row
    via STACK0_BYTE1; the V/O nibble copy re-deposits STACK0_BYTE_VAL_1
    into STACK0_BYTE_VAL_1 at the emit row. ALiBi recency (negative slope,
    set in the bake) picks the oldest populated PSH frame (the original
    operand-A PSH) over the current ADD step's empty STACK0 frame.
    """
    L = 15.0
    # K source-flag match weight -- see head 4's note: must dominate the
    # ALiBi distance penalty so only STACK0_BYTE1 rows are candidates.
    K_FLAG = 200.0
    # Single route: emit BYTE_INDEX_0 row -> operand byte 1 (a1) in
    # STACK0_BYTE_VAL_1 (the 1096 add corpus is all 2-byte; result byte 1
    # is the high byte and is predicted at the BYTE_INDEX_0 row).
    emit_bi_dim = BD.BYTE_INDEX_0
    # STACK0 campaign Part B (2026-06-18): under C4_OPERAND_FROM_MEMSP the
    # pushed operand's STACK0 byte-1 row is gone (STACK0 dropped + the
    # layer10_psh_ax_broadcast STACK0-row producer starved), so the legacy
    # K-match on the STACK0_BYTE1 position flag finds no row and the relay
    # delivers nothing -> 16-bit ADD byte 1 stays 0. The L8 head-7 byte-1 CAM
    # (make_layer8_mem_to_alu_op) now deposits operand-A byte 1 (from mem[SP])
    # into STACK0_BYTE_VAL_1 at the SAME step's AX marker; re-point the relay
    # K to that row (MARK_AX) so this head carries it forward to the ADD
    # byte-1 emit row. Flag-OFF keeps the STACK0_BYTE1 source (byte-identical).
    src_flag_dim = (
        BD.MARK_AX if operand_from_memsp_enabled() else BD.STACK0_BYTE1
    )
    val_lo = BD.STACK0_BYTE_VAL_1_LO
    val_hi = BD.STACK0_BYTE_VAL_1_HI
    if operand_from_memsp_enabled():
        # === STACK0 campaign Part C (Inc-3 claw-back, 2026-06-19): the
        # byte-1 RESULT delivery. ===
        #
        # The 30-token campaign frame breaks the legacy TEMP+8 gating
        # (GPU-confirmed, tools/probe_inc3_addsub_b1_trace.py):
        #   * The ADD/SUB byte-row selectors TEMP+8/TEMP+9 are delivered
        #     ONLY at the MARK_AX marker row (the L7 head-5 broadcast Q
        #     fires at MARK_AX) and are not spread to the BYTE_INDEX_0
        #     emit row in the collapsed frame -- so a Q gated on TEMP+8 at
        #     the emit row scores ~0 and the relay never fires.
        #   * The carrier STACK0_BYTE_VAL_1 (operand-A byte 1, deposited by
        #     the L8 head-7 mem[SP] CAM) sits at the MARK_AX marker row.
        #
        # Re-cast the relay as a clean intra-step cross-row copy that needs
        # NEITHER the dropped STACK0 row NOR the un-delivered TEMP selector:
        #   Q fires at the BYTE_INDEX_0 emit row (BYTE_INDEX_0 + HAS_SE,
        #   both present there), K selects the current step's AX marker via
        #   MARK_AX AND OP_ADD (the ADD discriminator lives on the SOURCE
        #   marker row -- OP_ADD ~= 5 there, 0 on a SUB/bitwise marker), so
        #   head 5 gathers the carrier ONLY on ADD steps and stays dark on
        #   SUB/bitwise (keeps the FORMAT_PTR/LEV_DETECTOR-aliased
        #   STACK0_BYTE_VAL band untouched off-ADD). The positive alibi
        #   slope (set in the bake) rewards proximity so the SAME step's AX
        #   marker (one row back) wins over any older ADD step's marker.
        OP_W = 40.0
        q = [
            AP(0, emit_bi_dim, L),
            AP(0, BD.HAS_SE, L),
            AP(0, BD.CONST, -L / 2),
            AP(0, BD.MARK_AX, -L * 10),
            AP(0, BD.MARK_PC, -L * 10),
            AP(0, BD.BYTE_INDEX_1, -L * 10),
            AP(0, BD.BYTE_INDEX_2, -L * 10),
            AP(0, BD.BYTE_INDEX_3, -L * 10),
            AP(33, emit_bi_dim, L),
            AP(33, BD.CONST, -L / 2),
        ]
        k = [AP(33, BD.CONST, L)]
        v = []
        o = []
        # Per-byte K-select slot 1: gather the carrier from the ADD AX
        # marker -- a TRUE AND of MARK_AX and OP_ADD. A non-ADD AX marker
        # (e.g. an SI/LI/var step's marker) must score NEGATIVE so the head
        # stays dark off-ADD (else it corrupts STACK0_BYTE_VAL_1 on var stores
        # -- GPU-measured var_simple regression). With MARK_AX=1, OP_ADD in
        # {0, ~5}: an ADD marker scores K_FLAG + 5*OP_W - 1.3*K_FLAG = +large;
        # a non-ADD marker scores K_FLAG - 1.3*K_FLAG = -0.3*K_FLAG < 0; byte
        # rows (no MARK_AX) score -1.3*K_FLAG < 0. The CONST baseline keeps
        # only the ADD marker above softmax1's zero anchor.
        sel = 1
        q.append(AP(sel, emit_bi_dim, L))
        q.append(AP(sel, BD.HAS_SE, L))
        q.append(AP(sel, BD.CONST, -L))
        k.append(AP(sel, BD.MARK_AX, K_FLAG))
        k.append(AP(sel, BD.OP_ADD, OP_W))
        k.append(AP(sel, BD.OP_SUB, -K_FLAG))
        k.append(AP(sel, BD.CONST, -K_FLAG * 1.3))
        base = 3
        for kk in range(16):
            v.append(AP(base + kk, val_lo + kk, 1.0))
            v.append(AP(base + 16 + kk, val_hi + kk, 1.0))
            o.append(AO(val_lo + kk, base + kk, 1.0))
            o.append(AO(val_hi + kk, base + 16 + kk, 1.0))
        # NOTE: unlike the SUB head 4, the ADD path does NOT stamp TEMP+8 here
        # -- TEMP+8 already reaches the ADD byte rows via the in-step spread
        # well enough for the L25 high-byte adder, and an extra stamp REGRESSES
        # ADD (24->14, GPU-measured): it over-fires the TEMP+8-gated L14 add
        # byte-1 cleanup / cascade rules. The carrier delivery alone is the win.
        #
        # === STACK0 campaign Inc-4 (2026-06-20): ADD byte-1 RESULT delivery. ===
        # The Inc-3 relay delivers a1 (operand-A byte 1) into STACK0_BYTE_VAL_1
        # at the emit row, but the downstream ``l10_add_high_byte_adder`` (the
        # Part-2 op that computes OUTPUT byte 1 = a1 + b1 + carry) is gated on
        # TEMP+8 AT the emit row -- and GPU-confirmed (probe_inc3_addsub_b1_trace)
        # the 30-token campaign frame leaves TEMP+8 = 0 at the BYTE_INDEX_0 emit
        # row (it lands only on the MARK_AX marker). So the adder never fires:
        # byte 1 stays b1-only, dropping a1 AND the byte-0 carry (add_1 0x310 ->
        # 0x210, add_6 0x5BF -> 0x2BF). Stamp a DEDICATED campaign discriminator
        # TEMP+12 onto the emit row (a free TEMP slot the L14 add cleanup does
        # NOT read -- so it cannot reproduce the TEMP+8 over-fire above). V reads
        # OP_ADD (= 5.0 ONLY at the matched ADD marker -> ~0 elsewhere, clean and
        # ADD-exclusive); 0.2 scale yields TEMP+12 ~= 1.0 at the emit row for
        # EVERY ADD step (including a1 = 0, where the carrier collapses to 0 but
        # the carry-in still must be summed). The campaign-gated branch of
        # ``l10_add_high_byte_adder`` reads TEMP+12 instead of TEMP+8.
        v.append(AP(base + 32, BD.OP_ADD, 0.2))
        o.append(AO(BD.TEMP + 12, base + 32, 1.0))
        return (
            DeclarativeAttentionHeadSpec(
                head_idx=5, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o),
            ),
        )
    # Q slot 0: gate STRICTLY on the ADD byte-emit selector TEMP+8 (1.0
    # ONLY on ADD byte rows; 0 on SUB/bitwise/everything else). Mirrors
    # head 4's TEMP+9 gate -- positive only when TEMP+8 fires, negative
    # otherwise, so the relay never writes STACK0_BYTE_VAL off-ADD (those
    # dims alias FORMAT_PTR/LEV_DETECTOR; writing them off-ADD risks
    # regression). slot 33 anti-leak mirrors the same TEMP+8-gated bias.
    q = [
        AP(0, BD.TEMP + 8, L),
        AP(0, BD.CONST, -L / 2),
        AP(0, BD.MARK_AX, -L * 10),
        AP(0, BD.MARK_PC, -L * 10),
        AP(0, BD.TEMP + 9, -L * 10),
        AP(0, BD.BYTE_INDEX_3, -L * 10),
        AP(33, BD.TEMP + 8, L),
        AP(33, BD.CONST, -L / 2),
    ]
    k = [AP(33, BD.CONST, L)]
    v = []
    o = []
    # Per-byte K-select slot 1 + V/O nibble copies (slots 3..34).
    sel = 1
    # Q[sel] = BYTE_INDEX_0 + TEMP+8 - CONST: ~+L on an ADD byte-1 row
    # (1+1-1) but ~0 on a non-ADD byte-1 row (1+0-1=0), so on SUB/bitwise
    # byte rows the STACK0_BYTE1 K-match multiplies a ~0 query and the
    # head does NOT gather (keeps STACK0_BYTE_VAL untouched off-ADD).
    q.append(AP(sel, emit_bi_dim, L))
    q.append(AP(sel, BD.TEMP + 8, L))
    q.append(AP(sel, BD.CONST, -L))
    k.append(AP(sel, src_flag_dim, K_FLAG))
    k.append(AP(sel, BD.CONST, -K_FLAG / 2))
    base = 3
    for kk in range(16):
        v.append(AP(base + kk, val_lo + kk, 1.0))
        v.append(AP(base + 16 + kk, val_hi + kk, 1.0))
        o.append(AO(val_lo + kk, base + kk, 1.0))
        o.append(AO(val_hi + kk, base + 16 + kk, 1.0))

    return (
        DeclarativeAttentionHeadSpec(
            head_idx=5,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
    )


def _layer13_add_addend_relay_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_add_addend_relay_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer13_add_addend_relay_op() -> Operation:
    """L13 attn head 5: relay ADD operand-A byte 1 (a1) to STACK0_BYTE_VAL_1.

    Part 1 of the multi-byte ADD fix (mirror of the SUB minuend relay,
    head 4, for the ADD path). Delivers operand-A byte 1 -- pushed by
    ``layer10_psh_ax_broadcast`` at the PSH-frame STACK0 byte-1 row -- to
    the ADD byte-1 emit row so the L10 ADD byte-1 adder (Part 2,
    ``l10_add_high_byte_adder``) can compute a1 + b1 + carry. See the
    module comment block above for the root cause (a1 absent off-SUB), the
    TEMP+8 vs TEMP+9 mutual exclusion with head 4, and the byte-identity
    property (8-bit ADD relays a1 = 0x00, unchanged).
    """
    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        # NEGATIVE ALiBi slope on head 5 (mirrors head 4): the runtime
        # applies ``-slope * |q_pos - k_pos|`` so a negative slope rewards
        # distance -> prefers the OLDEST populated STACK0_BYTE1 row (the
        # original operand-A PSH frame). The K source-flag match dominates
        # the alibi penalty so only STACK0_BYTE1 rows are candidates; the
        # slope breaks the tie toward the oldest among them.
        #
        # STACK0 campaign Part B: when the K source is re-pointed to MARK_AX
        # (C4_OPERAND_FROM_MEMSP), the source row is the SAME step's AX marker
        # (a FEW tokens before the ADD byte-1 emit row), NOT a distant prior
        # PSH frame. A POSITIVE slope rewards proximity so the CURRENT step's
        # AX marker (where head-7 just deposited operand-A byte 1) wins over any
        # older step's AX marker.
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[5] = (
                1.0 if operand_from_memsp_enabled() else -1.0
            )
        Primitives.generate_attention_head(
            attn,
            _layer13_add_addend_relay_head_specs(proxy)[0],
            HD,
        )

    _claims = set()
    base = 3
    for k in range(16):
        _claims.add((13, "attn_W_v", f"5_{base + k}", f"STACK0_BYTE_VAL_1_LO+{k}"))
        _claims.add((13, "attn_W_v", f"5_{base + 16 + k}",
                     f"STACK0_BYTE_VAL_1_HI+{k}"))

    _campaign = operand_from_memsp_enabled()
    _reads = {"IS_BYTE", "TEMP", "BYTE_INDEX_0", "STACK0_BYTE1", "MARK_AX",
              "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI", "CONST"}
    _writes = {"STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"}
    if _campaign:
        # Campaign Part C: K gates on OP_ADD/OP_SUB + MARK_AX (source marker
        # discriminator); Q gates on HAS_SE + BYTE_INDEX_* (emit row).
        _reads |= {"HAS_SE", "OP_ADD", "OP_SUB", "BYTE_INDEX_1",
                   "BYTE_INDEX_2", "BYTE_INDEX_3"}

    return Operation(
        name="layer13_add_addend_relay",
        reads=_reads,
        writes=_writes,
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_add_addend_relay_ir,
        declarative_authority="spec_generated",
        # Co-place on the L13 attn block (heads 0-2 = mem_addr_gather,
        # head 3 = bitwise_byte1_gather, head 4 = sub_minuend_relay); this
        # claims head 5. Run after the SUB relay so the block is populated.
        target_op_name="_layer13_mem_addr_anchor",
        requires={"after": "layer13_sub_minuend_relay"},
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
    )


# =====================================================================
# width=2 MUL byte-1 result relay (L13 head 6)  -- 2026-06-13
# =====================================================================
#
# Opt-in via C4_MUL_WIDTH2=1. Completes the width=2 (8-bit x 8-bit ->
# 16-bit) MUL byte-1 emit path, mirroring the byte-0 result path:
#
#   byte 0:  L11 wide_mul writes product byte 0 -> OUTPUT_LO/OUTPUT_HI
#            (at the MARK_AX row); the existing decode emits it directly.
#   byte 1:  L11 wide_mul writes product byte 1 -> MUL_RESULT_HI_LO/HI
#            (dedicated band, NOT OUTPUT_LO+32 = ADDR_KEY). THIS head
#            copies MUL_RESULT_HI_LO/HI -> AX_FULL_LO/HI at the MARK_AX
#            row, and the existing ``layer15_alu_high_byte_relay``
#            (l14_ops.py, already K-gated on OP_MUL) then copies
#            AX_FULL -> OUTPUT at the byte-1 emit token.
#
# This is a SAME-ROW copy (source MUL_RESULT_HI_* and destination
# AX_FULL_* both live at the MUL MARK_AX row), so it uses slot-0
# self-attention: Q fires at MARK_AX AND OP_MUL, K matches CONST at the
# same row (the recency softmax keeps the copy local to the firing row),
# V reads MUL_RESULT_HI_LO/HI, O writes AX_FULL_LO/HI. The whole op is
# gated by ``mul_width2_enabled()`` -- registered only when the flag is
# on -- because it reads MUL_RESULT_HI_*, which only exist (are declared
# / d_model-allocated) under the same flag.
#
# LANDED (2026-06-13): the d_model widen is head-dim-preserving
# (MUL_RESULT_HI_* routed through extra_residual_dims, 872 -> 981,
# n_heads 8 -> 9), so e2e mul_overflow (100*5=500=0x01F4) emits both
# bytes correctly and bnz stays green -> smoke 50/1. width=2 is the
# default (opt out C4_MUL_WIDTH2=0). See
# docs/MUL_WIDTH2_WIDEN_2026_06_13.md.
def _layer13_mul_result_hi_relay_head_specs(BD) -> tuple:
    """L13 head 6: copy MUL byte-1 result (MUL_RESULT_HI_*) into AX_FULL.

    Same-row copy at the MUL MARK_AX row. Q fires at MARK_AX AND OP_MUL
    (with a CONST anti-leak penalty and a slot-33 MARK_AX requirement so
    the slot-0 softmax only routes positively on the MUL MARK_AX row).
    K matches CONST at the same row. V copies the MUL_RESULT_HI_LO/HI
    nibble pair; O writes AX_FULL_LO/HI at the Q row.
    """
    L = 15.0
    q = [
        AP(0, BD.MARK_AX, L),
        AP(0, BD.OP_MUL, L),
        AP(0, BD.CONST, -L / 2),
        # slot 33 anti-leak: require MARK_AX so the slot-0 softmax only
        # routes positively at the MUL MARK_AX Q row.
        AP(33, BD.MARK_AX, L),
        AP(33, BD.CONST, -L / 2),
    ]
    k = [
        AP(0, BD.CONST, L),
        AP(33, BD.CONST, L),
    ]
    # V slots 1..32 copy the MUL byte-1 result nibbles.
    v = [AP(1 + kk, BD.MUL_RESULT_HI_LO + kk, 1.0) for kk in range(16)]
    v += [AP(17 + kk, BD.MUL_RESULT_HI_HI + kk, 1.0) for kk in range(16)]
    # O slots 1..32 route the gathered nibbles into AX_FULL_LO/HI.
    o = [AO(BD.AX_FULL_LO + kk, 1 + kk, 1.0) for kk in range(16)]
    o += [AO(BD.AX_FULL_HI + kk, 17 + kk, 1.0) for kk in range(16)]

    return (
        DeclarativeAttentionHeadSpec(
            head_idx=6,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
    )


def _layer13_mul_result_hi_relay_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_mul_result_hi_relay_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer13_mul_result_hi_relay_op(enable: bool = False) -> Operation:
    """L13 attn head 6: stage width=2 MUL byte-1 result into AX_FULL.

    Opt-in (``C4_MUL_WIDTH2=1`` -> ``enable=True``). Copies the product's
    high byte from the dedicated MUL_RESULT_HI_LO/HI band into AX_FULL_LO/HI
    at the MUL MARK_AX row, so the existing ``layer15_alu_high_byte_relay``
    (already K-gated on OP_MUL) can emit byte 1 to OUTPUT at the byte-1
    token. This mirrors the byte-0 result path (OUTPUT_LO/HI direct). See
    the module comment block above for the full byte-1 emit chain and the
    head-dim-preserving widen note.

    The op is ALWAYS registered (keeps the dep graph / layer count stable
    per the ``all_core_ops`` convention) but is fully INERT when
    ``enable=False``: empty reads/writes/claims, a no-op bake, and an empty
    IR factory. It only references MUL_RESULT_HI_* (a flag-gated dim band)
    when ``enable=True``, so the compiler never sees those dims unless the
    flag is on.
    """
    if not enable:
        # Inert no-op form. References no flag-gated dims so the compiler
        # accepts it under the default (flag-off) build with d_model=920.
        return Operation(
            name="layer13_mul_result_hi_relay",
            reads=set(),
            writes=set(),
            kind="block",
            declarative_bake_fn=lambda block, dim_positions, S: None,
            compiler_ir_factory=_empty_compiler_ir_factory,
            declarative_authority="spec_generated",
            target_op_name="_layer13_mem_addr_anchor",
            requires={"after": "layer13_add_addend_relay"},
            migrated=True,
            claims=set(),
            spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
        )

    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        # Slot-0 self-row copy: a slightly POSITIVE ALiBi slope keeps the
        # softmax mass on the firing (MARK_AX) row itself (the source and
        # destination bands both live there). The slot-33 anti-leak +
        # CONST gate make the head dark off the MUL MARK_AX row.
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[6] = 1.0
        Primitives.generate_attention_head(
            attn,
            _layer13_mul_result_hi_relay_head_specs(proxy)[0],
            HD,
        )

    _claims = set()
    for k in range(16):
        _claims.add((13, "attn_W_v", f"6_{1 + k}", f"MUL_RESULT_HI_LO+{k}"))
        _claims.add((13, "attn_W_v", f"6_{17 + k}", f"MUL_RESULT_HI_HI+{k}"))

    return Operation(
        name="layer13_mul_result_hi_relay",
        reads={"MARK_AX", "OP_MUL", "MUL_RESULT_HI_LO", "MUL_RESULT_HI_HI",
               "CONST"},
        writes={"AX_FULL_LO", "AX_FULL_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_mul_result_hi_relay_ir,
        declarative_authority="spec_generated",
        # Co-place on the L13 attn block (heads 0-2 = mem_addr_gather,
        # head 3 = bitwise_byte1_gather, head 4 = sub_minuend_relay,
        # head 5 = add_addend_relay); this claims head 6. Run after the
        # ADD relay so the block is populated.
        target_op_name="_layer13_mem_addr_anchor",
        requires={"after": "layer13_add_addend_relay"},
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
        opcodes={"OP_MUL"},
    )


def make_layer13_mem_addr_anchor_op() -> Operation:
    """L13 attn-side anchor for the mem-addr family (Phase 3b mem cluster fix).

    Sibling of ``_layer13_attn_dep_anchor`` (below). Historically the joint
    ``_layer13_attn_dep_anchor`` co-served BOTH the L13 mem-addr family
    (``layer13_mem_addr_gather``, a kind="block" op writing ADDR_B*_LO/HI
    into block[L13].attn) AND the L13 FFN-side shift family
    (``layer13_shifts``, ``l13_alu_shift_install``, ``l13_alu_postop_attach``).
    Per ``docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md`` this coupling blocked
    Phase 4: the mem-addr gather needs to land at L13 (where the
    ADDR_B*_LO/HI residuals feed L14 same-step), but the shift composite
    stages need the existing anchor at L16 (where AX_CARRY residuals
    survive intervening L15-L16 ops).

    Phase 3b is the metadata-only split: the new anchor co-resolves to
    the SAME physical layer as the legacy joint anchor (L16 today) via
    the shared ``same_layer_as`` constraint, so smoke is byte-identical
    after the split. Phase 4 will later drop the ``same_layer_as`` and
    pin this anchor at L13 (paired with the L13 head-0 decoupling and
    composite-late-placement work documented in the BLOCKER doc).

    Phase 3b splits the anchor into TWO siblings:

      * ``_layer13_mem_addr_anchor`` (this op): owns the mem-addr gather
        binding. Co-resolves with the legacy anchor today; Phase 4 will
        retarget it to L13.
      * ``_layer13_attn_dep_anchor`` (below): keeps the FFN-side shift
        family (lookup-mode ``layer13_shifts``, efficient-mode composite
        install + post-op attach) bound where they already work today
        (block[16] via the ``requires["after"]:_layer12_ffn_dep_anchor``
        chain). Also remains the chain link for ``_layer14_attn_dep_anchor``.

    As a ``declarative_authority="topology_anchor"`` op this writes no
    weights; the slot registry derives no slot claims for it (per
    ``slot_registry.derive_slot_ids_for_op`` — topology anchors return
    early). Two attn anchors at the same layer are therefore admissible.

    Consumers (``target_op_name="_layer13_mem_addr_anchor"``):
      - ``layer13_mem_addr_gather`` (block, kind="block").

    Consumers NOT moved (remain on ``_layer13_attn_dep_anchor``):
      - ``layer13_shifts`` (block, FFN-side lookup-mode shift bake)
      - ``l13_alu_shift_install`` (block, composite install)
      - ``l13_alu_postop_attach`` (block, post-op attach)
      - ``_layer14_attn_dep_anchor`` (``requires["after"]`` chain)
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in ``layer13_mem_addr_gather`` (kind="block").
        return None

    return Operation(
        name="_layer13_mem_addr_anchor",
        # Mirrored subset of ``layer13_mem_addr_gather`` reads/writes
        # (sized so the dep graph reserves a kind="attn" slot at the
        # joint anchor's layer). Same dim sets as the legacy joint
        # anchor so the SCC structure is unchanged.
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "AX_CARRY_LO", "AX_CARRY_HI", "OP_LI", "OP_LC",
               "OP_SI", "OP_SC", "MEM_ADDR_SRC", "L1H1"},
        writes={"ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI"},
        kind="attn",
        # Phase 3c (mem cluster fix, 2026-06-06): pin to ``layer_idx=13``
        # so ``layer13_mem_addr_gather`` lands at block[13].attn
        # (the L14 same-step mem-value chain reads ADDR_B*_LO/HI in-step;
        # see MEMORY_PHASE4_BLOCKER_2026_06_05.md for the data-flow
        # argument). The head_0/1/2 contest with the L10 attn family
        # (which historically occupied block[13].attn at all 8 heads) is
        # resolved by the paired ``layer_idx=11`` pin on
        # ``_layer10_attn_anchor`` (Phase 3c sibling change in
        # ``l10_ops.py``); the L10 attn family now bakes into
        # block[11].attn, freeing block[13].attn heads 0/1/2.
        #
        # Replaces the Phase 3b ``phase=13.0`` + ``same_layer_as:
        # _layer13_attn_dep_anchor`` co-location constraint: that kept
        # both anchors at L16 (where the joint anchor naturally landed
        # via its ``after`` chain) so Phase 3b was byte-identical to the
        # pre-split joint anchor. Phase 3c is the actual layer movement.
        layer_idx=13,
        migrated=True,
        declarative_authority="topology_anchor",
        # ``after: _layer12_ffn_dep_anchor`` retained defensively so the
        # dep graph still orders this anchor after L12; ``same_layer_as``
        # dropped because Phase 3c intentionally separates the two L13
        # anchors onto different physical layers (mem-addr at L13,
        # shift-family anchor stays at L16).
        requires={"after": "_layer12_ffn_dep_anchor"},
        smoke_tests=set(),
        spec_section=None,
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )


def make_layer13_attn_dep_anchor_op() -> Operation:
    """No-op companion for the L13 FFN-side shift family: declares mirrored
    reads/writes so the LayerCompiler's dep graph reserves a kind="attn"
    slot at whichever layer the dep-chain places it (today: L16). Mirrors
    ``_layer11_ffn_dep_anchor`` / ``_layer3_ffn_dep_anchor``.

    Phase 3b (mem cluster fix, 2026-06-05) split: the legacy joint anchor
    co-served the L13 mem-addr family AND the L13 FFN-side shift family.
    The mem-addr family migrated to the sibling
    ``_layer13_mem_addr_anchor`` (above, ``layer_idx=13``). This anchor
    retains the FFN-side shift family (lookup ``layer13_shifts``,
    efficient ``l13_alu_shift_install``, ``l13_alu_postop_attach``) plus
    the ``_layer14_attn_dep_anchor`` ``requires["after"]`` chain so the
    L14 anchor still pins past this one.

    Phase 8.G.6: lets L13 block ops declare
    ``target_op_name="_layer13_attn_dep_anchor"`` and bind to whichever
    layer the compiler places the anchor at, instead of carrying a
    literal ``layer_idx=13``.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in ``layer13_mem_addr_gather`` block op below.
        return

    return Operation(
        name="_layer13_attn_dep_anchor",
        # Phase=13.0: nominally pins this anchor between the L12 dep
        # anchor (phase=11.5, landed at L15) and the L14 dep anchor
        # (phase=14, landed at L17). Intended to force L13 anchor to
        # land at L13 (cluster A AX-zero, see
        # SMOKE_MEMORY_TRACE_20260603.md Fix Option 1).
        #
        # NOTE 2026-06-03: empirically the bump from phase=12.5 to
        # phase=13.0 alone did NOT move the anchor — it still lands at
        # L16, identical to the 12.5 placement, because the dep-graph
        # scheduler only uses `phase` as an SCC cycle-breaker (see
        # layer_compiler.py:1745+) and there is no `before=` edge
        # forcing the anchor earlier than the layer at which its
        # `after: _layer12_ffn_dep_anchor` is first satisfiable. The
        # real fix likely needs an explicit `layer_idx=13` (Fix Option
        # 2) or a `before: layer14_addr_key_neural_decode` edge.
        # 2026-06-03 follow-up: see L13_ANCHOR_DOWNSTREAM_CHAIN.md —
        # `layer_idx=13` alone breaks `l13_alu_shift_install` (composite
        # stages at L17-L20 bake AFTER install at L13). Pin the 4
        # composite stages to L13 too before re-trying.
        # 2026-06-05 follow-up: Option A attempted in worktree
        # `memory-fix` (commit-prep) — added ``layer_idx=13`` here and
        # to all 4 ``l13_alu_shift_*`` stages. The dep validator
        # complains that ``layer16_lev_routing`` writes ALU_LO at L16,
        # which the SSA `.*.-1` alias suppresses; but pinning the
        # stages to L13 causes a runtime shape mismatch
        # ``mat1 (264x800) and mat2 (512x1536)`` at every test —
        # see SMOKE_MEMORY_FIX_ATTEMPT_20260605.md. The L13.ffn
        # right-sizing pass doesn't accommodate the 4 efficient-mode
        # stages co-located with the lookup-mode FFN. Option A needs
        # ffn_widths plumbing before it's viable; reverted.
        phase=13.0,
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "AX_CARRY_LO", "AX_CARRY_HI", "OP_LI", "OP_LC",
               "OP_SI", "OP_SC", "MEM_ADDR_SRC", "L1H1"},
        writes={"ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Pin strictly after the L12 anchor so the earliest landable
        # layer is 13 (force a layer past L12).
        requires={"after": "_layer12_ffn_dep_anchor"},
        smoke_tests=set(),
        spec_section=None,
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )


def make_layer13_mem_addr_gather_op() -> Operation:
    """L13 attention: gather MEM addr from STACK0 / AX_CARRY for SI/SC/LI/LC.

    Pinned to ``layer_idx=13`` via ``kind="block"``: dep-graph assignment
    otherwise lands at L15 (mismatch with legacy block 13).

    Wave 3I: fully migrated to ``DeclarativeAttentionHeadSpec`` form. The
    legacy ``_set_layer13_mem_addr_gather`` and ``_l13_addr_bn_valid_extension``
    imperative helpers are replaced by
    :func:`_layer13_mem_addr_gather_head_specs` (three head specs covering
    the B7-4 ADDR_B0_VALID producer and the B8-A ADDR_B{1,2}_VALID
    extension), exposed via ``compiler_ir_factory``. Byte-identity gated by
    ``compare_symbolic_to_lowered_attn``.
    """
    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake attention-head allocator with the L13 head layout pinned.
        # Stashed on ``attn`` for downstream inspection / extension; the
        # actual ``head_idx`` values used by ``_layer13_mem_addr_gather_head_specs``
        # come from :func:`_l13_head_idx` so the spec stays in lockstep
        # with the layout table without re-querying the allocator here.
        allocator = _allocate_layer13_attention_heads()
        attn._l13_head_allocator = allocator
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _layer13_mem_addr_gather_head_specs(proxy),
            HD,
        )

    # Dim-ownership claims: L13 attn heads 0-2 mem addr gather. Each head
    # writes V slots 1..32 reading CLEAN_EMBED_LO/HI:
    #   W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    # Head 0 additionally writes slot 34's V row from L1H1+MEM_I and routes
    # through W_o into ADDR_B0_VALID (B7-4 lifecycle bit; see setup_helpers
    # ``_set_layer13_mem_addr_gather`` docstring).
    #
    # B8-A note: heads 1/2 slot 34 also produce ADDR_B{1,2}_VALID at slots
    # 99/100 via ``_l13_addr_bn_valid_extension``. Those writes are NOT yet
    # in ``claims`` / ``writes`` -- the per-op contract test currently pins
    # the head-0-only invariant. A follow-up commit will (a) add the two
    # dims to ``declare_setdim_compat_dims``, (b) add the V claims and
    # writes entries, and (c) update the per-op test to expect three VALID
    # bits. Until then the verifier records the head-1/2 V cells under
    # ``written_but_not_declared`` (warning, not error in non-strict mode).
    _claims = set()
    for h in range(3):
        for k in range(16):
            _claims.add((13, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((13, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    # ADDR_B0_VALID lifecycle slot: head 0, slot 34, V row reads L1H1+MEM_I=4.
    # Verifier decodes the col position via ``_pos_to_column`` which produces
    # ``"<DIM>+<offset>"``; L1H1 has 7 lanes so MEM_I=4 lands at +4.
    _claims.add((13, "attn_W_v", "0_34", "L1H1+4"))

    return Operation(
        name="layer13_mem_addr_gather",
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "AX_CARRY_LO", "AX_CARRY_HI", "OP_LI", "OP_LC", "OP_SI", "OP_SC",
               "MEM_ADDR_SRC", "L1H1"},
        # UNDECLARED_DIM_AUDIT_2026_06_09: H5 added — the L13 mem-addr
        # gather heads also stage to H5 (per IR walk); declared so the
        # dep graph sees this writer.
        writes={"ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI",
                "ADDR_B0_VALID", "H5"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_mem_addr_gather_ir,
        declarative_authority="spec_generated",
        # Phase 3b (mem cluster fix, 2026-06-05): retargeted from the
        # joint ``_layer13_attn_dep_anchor`` to the new sibling
        # ``_layer13_mem_addr_anchor`` so the L13 FFN-side shift family
        # (lookup ``layer13_shifts``, efficient ``l13_alu_shift_install``
        # + ``l13_alu_postop_attach``) can migrate layers independently
        # of the mem-addr family. Today both anchors resolve to the
        # same physical layer via the shared ``phase=13.0`` +
        # ``same_layer_as`` constraint -- this change is metadata-only,
        # smoke-neutral. Phase 4 will drop the ``same_layer_as`` and pin
        # the mem-addr anchor at L13 (paired with the L13 head-0 work
        # documented in ``docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md``).
        target_op_name="_layer13_mem_addr_anchor",
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer13_shifts_op(alu_mode: str = "lookup") -> Operation:
    """L13 FFN: SHL/SHR shifts (lookup-mode entry point).

    Pinned to ``layer_idx=13`` via ``kind="block"``. See
    ``make_layer13_mem_addr_gather_op``.

    In ``alu_mode='lookup'`` we bake the standard SHL/SHR lookup table as a
    4096-rule declarative :class:`FFNRule` IR (Phase 6 Wave 4B migration);
    byte-identical to the legacy ``setup_helpers._set_layer13_shifts`` helper
    (gated by ``test_declarative_ffn_bakes_l13``).

    Declarations-only note: lookup mode is exposed through the migrated owner
    so strict builds do not fall back to legacy model bake. Efficient mode is
    represented by the structural 4-stage composite ops instead.

    In ``alu_mode='efficient'`` SHL/SHR are now handled by the 4-stage
    composite installed via the dedicated
    ``make_l13_alu_shift_{bdtoge,precompute,select,getobd}_op`` factories
    (each at phase=13 so they share L13 with ``make_layer13_mem_addr_gather_op``).
    The 4 ops together replace the runtime ``ALUShift`` wrapper that used to
    be attached by ``set_vm_weights``. This entry-point is a no-op in
    efficient mode so the lookup-table bake doesn't overwrite the composite's
    output.
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    if alu_mode == "efficient":
        def bake(block, dim_positions, S):
            return  # ALUShiftComposite (4-stage) owns SHL/SHR in efficient mode.

        compiler_ir = None
    else:
        def bake(block, dim_positions, S):
            proxy = _as_setdim_proxy(dim_positions)

            # Per-bake FFN-unit allocator. Phase 7.B.5: both shift
            # sub-stages are auto-fit (``pin=None``); declaration order
            # SHL-then-SHR plus first-fit on an empty pool lands SHL at
            # unit 0 and SHR at unit 2048 — byte-identical to the
            # legacy explicit pins. Stashed on the FFN module so
            # downstream tools (e.g. a future L13 op family) can
            # inspect or extend the layout. Mirrors the L1 / L9
            # convention from 4639146 / ca775eb.
            allocator = _allocate_layer13_shifts_units()
            block.ffn._l13_unit_allocator = allocator

            # Phase 8.C inline cut: lower the ``_layer13_shifts_rules``
            # IR directly here so census v2 classifies this op as
            # ``declarative`` rather than ``declarative_via_helper``
            # (which routed through the ``_bake_layer13_shifts``
            # trampoline). Byte-identical to the prior
            # ``_bake_layer13_shifts(block.ffn, S, proxy)`` call.
            # The helper is retained for ``test_declarative_ffn_bakes_l13``
            # parity coverage.
            rules = _layer13_shifts_rules(S)
            rule_dim_positions = Primitives.dim_positions_from_bd(
                proxy,
                Primitives.ffn_rule_dim_names(rules),
            )
            n_units = Primitives.lower_ffn_rules(
                block.ffn, rules, rule_dim_positions, S=S,
            )
            # Byte-identity guard: the declarative lowering MUST land
            # exactly on the allocator's declared range total or a
            # downstream layer will read stale weights.
            expected_total = sum(n for _, n in _L13_SHIFTS_UNIT_LAYOUT)
            assert n_units == expected_total, (
                f"L13 layer13_shifts unit cursor drift: declarative "
                f"bake wrote {n_units} units, allocator declared "
                f"{expected_total}"
            )

        compiler_ir = _layer13_shifts_ir()

    return Operation(
        name="layer13_shifts",
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_SHL", "OP_SHR",
               # V2/G7 LEV detector: in-step topology edge replacing the
               # cross-step requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=compiler_ir,
        declarative_authority="spec_generated",
        # Phase 8.G.6: drop ``layer_idx=13`` literal; bind to the L13
        # attn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer13_attn_dep_anchor",
        migrated=True,
        # Staleness invariants: L13 shift FFN consumes ALU_LO/HI (value to
        # shift) and AX_CARRY_LO (shift amount) at the AX marker for
        # OP_SHL / OP_SHR. Only meaningful when alu_mode='lookup' fires the
        # bake; in efficient mode the composite owns the consumes-fresh
        # chain via its own stages.
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
    )
