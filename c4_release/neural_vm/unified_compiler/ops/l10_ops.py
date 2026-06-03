"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from dataclasses import replace
from typing import Mapping, Optional

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, ConditionTerm, DimRef, FFNRule
from ..layer_compiler import Operation
from ..band_guarantees import expected_byte_guarantee_rules
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L10 attention-head layout (auto-fit; legacy head_idx as docs) ===
#
# L10 attention hosts 8 primary heads, each owned by one ``kind="block"``
# bake op below. Pre-migration the spec helpers used bare ``head_idx=N``
# literals; resolving every literal through :func:`_l10_head_idx`
# preserves byte-identity while declaring the L10 head axis as audited
# data. Order mirrors the bake-op factory order below so the table
# reads top-to-bottom alongside the bakes that own each row.
#
# Head 1 (AX byte passthrough) and head 7 (BP byte passthrough) both reuse
# the byte-passthrough chain template :func:`_byte_passthrough_chain_spec`;
# heads 4/5/6 are co-owned by ``layer10_stack0_byte_relay_bake`` (one
# allocator row per spec for inspection clarity).
#
# Phase 7.B.6: the allocator runs without ``pin=`` -- first-fit picks
# 0..7 in declaration order, which matches the legacy layout bit-for-bit
# because :data:`_L10_HEAD_LAYOUT` is contiguous and ordered. The
# ``legacy_head_idx`` column is documentation only; the load-bearing
# copy is the :func:`_l10_head_idx` lookup, consumed by the head-spec
# factories that write Q/K/V/O weights at the resolved index.
_L10_HEAD_LAYOUT = (
    # (op-name key,                                       legacy_head_idx (docs only))
    ("layer10_carry_relay_bake.head_0",                  0),  # ADD/SUB byte carry
    ("layer10_byte_passthrough_bake.head_1",             1),  # AX byte passthrough
    ("layer10_sp_byte_passthrough_bake.head_2",          2),  # SP byte passthrough
    ("layer10_psh_stack0_passthrough_bake.head_3",       3),  # PSH STACK0 passthrough
    ("layer10_stack0_byte_relay_bake.head_4",            4),  # bitwise stack-byte relay
    ("layer10_stack0_byte_relay_bake.head_5",            5),  # non-bitwise stack-byte relay
    ("layer10_stack0_byte_relay_bake.head_6",            6),  # STACK0 byte persistence
    ("layer10_bp_byte_passthrough_bake.head_7",          7),  # BP byte passthrough
)


def _allocate_layer10_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with the L10 heads.

    Phase 7.B.6: ``pin=`` is dropped from every entry. The allocator's
    first-fit picks the lowest free head index in declaration order;
    because :data:`_L10_HEAD_LAYOUT` is contiguous (0..7) and ordered,
    first-fit reproduces the legacy ``head_idx`` values bit-for-bit.
    The actual weight-write head indices are still looked up via
    :func:`_l10_head_idx` inside the head-spec factories below, so
    byte-identity with the legacy bake is preserved regardless of
    allocator order. ``layer_max_heads=8`` so the layer is full today.
    """
    allocator = AttentionHeadAllocator()
    for name, _legacy_head_idx in _L10_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=10)
    return allocator


def _l10_head_idx(op_name: str) -> int:
    """Return the pinned L10 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L10_HEAD_LAYOUT` for callers (e.g.
    ``compiler_ir_factory`` helpers, spec functions) that cannot
    instantiate a per-bake allocator. Mirrors the L4/L13 pattern.
    """
    for name, head_idx in _L10_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l10_head_idx: unknown L10 attention op {op_name!r}")


# === L10 FFN unit layouts (auto-fit; legacy offsets retained as docs) ==
#
# L10 hosts three distinct FFNs in the compiled model:
#
#   1. ``model.blocks[10].ffn`` -- baked by ``make_layer10_alu_op`` via
#      ``vm_step._set_layer10_alu``. 1846 units = comparison combine (18)
#      + bitwise OR/XOR/AND lo+hi (3 * 512 = 1536) + MUL lo (256) +
#      SHL/SHR zero shortcut (4) + AX passthrough (32).
#
#   2. A dependency-assigned FFN block carrying the *combined* L10 post-op
#      logic (``l10_post_ops_combined``, ``kind="ffn"``). 1562 units =
#      ``BinaryOpByteZeroingPostOp`` (8) + 3x ``CarryPropagationPostOp``
#      (512 each, slice later zeroed) + ``ComparisonCombine`` (18). The
#      carry slice keeps its unit range so OUTPUT-row offset accounting
#      stays byte-identical even though the weights are wiped.
#
#   3. A late ``tail_bit32_result_correction`` block on layer 17 -- its
#      own freshly-allocated ``PureFFN`` sized to ``len(rules)`` (2059).
#      Single-owner layout, declared here so a future second tenant in
#      that bank goes through ``allocator.alloc(...)``.
#
# Phase 7.B.6: every layout below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because each layout is fully
# contiguous in declaration order (every entry starts exactly where
# the previous one ended), first-fit reproduces the legacy pinned
# offsets bit-for-bit -- so byte-identity with the legacy bakes and
# their monotonic ``unit = 0`` / ``offset = 0`` cursors survives the
# pin drop. The ``legacy_start`` columns are kept purely as
# documentation; downstream MUL/LEA tail rules in the wide-MUL fix
# chain are unaffected because they consume layout-by-name, not by
# pin offset.
#
# The per-block ``make_l10_post_op_attach_op`` bake appends six (lookup)
# or seven (efficient) *independent* ``PureFFN`` modules onto
# ``block.post_ops`` -- each is a standalone bank, not a shared hidden
# axis, so it deliberately does NOT appear in these tables. Migrating
# those into an allocator would require flattening them into one FFN,
# which is the next refactor stage, not this commit.
#
# Migration is byte-identical bookkeeping: each underlying helper still
# writes via its own monotonic ``unit = 0`` / ``offset = 0`` cursor.
# The allocator declares ranges by name, the helpers write the weights,
# and an ``assert`` after each helper verifies the cursor lands exactly
# where the layout table says it should. Changing any helper's unit
# count requires updating the matching table in lock-step.

# Main L10 FFN (model.blocks[10].ffn). Walk mirrors the order of writes
# in ``vm_step._set_layer10_alu``.
_L10_FFN_UNIT_LAYOUT_MAIN = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer10_alu.cmp_combine",       0,   18),  # 6 default + 12 override
    ("layer10_alu.bitwise_or",       18,  512),  # 256 lo + 256 hi
    ("layer10_alu.bitwise_xor",     530,  512),  # 256 lo + 256 hi
    ("layer10_alu.bitwise_and",    1042,  512),  # 256 lo + 256 hi
    ("layer10_alu.mul_lo",         1554,  256),  # (a*b)%16 lookup
    ("layer10_alu.shl_shr_zero",   1810,    4),  # 2 per opcode (SHL, SHR)
    ("layer10_alu.ax_passthrough", 1814,   32),  # 16 lo + 16 hi
)
_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL = 1846

# Combined post-op FFN baked by ``make_l10_post_ops_combined`` (kind="ffn",
# dependency-assigned). Each range maps 1:1 to a post-op class's
# ``hidden_dim`` and lands at the offset the inline ``offset`` counter
# walks to in the original bake.
_L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("l10_post_ops_combined.binary_op_byte_zeroing",     0,    8),  # PureFFN H=8
    ("l10_post_ops_combined.carry_propagation_byte0",    8,  512),  # PureFFN H=512
    ("l10_post_ops_combined.carry_propagation_byte1",  520,  512),  # PureFFN H=512
    ("l10_post_ops_combined.carry_propagation_byte2", 1032,  512),  # PureFFN H=512
    ("l10_post_ops_combined.comparison_combine",      1544,   18),  # PureFFN H=18
)
_L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED_TOTAL = 1562

# Tail bit32 result correction (lives on L17 block.post_ops as its own
# fresh PureFFN). Single tenant today; the layout makes the bank
# explicit so a future tenant claims through the allocator.
_L10_FFN_UNIT_LAYOUT_TAIL_BIT32 = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("tail_bit32_result_correction.rules", 0, 2059),
)
_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL = 2059


def _allocate_l10_main_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for ``model.blocks[10].ffn``.

    Phase 7.B.6: ``pin=`` is dropped. First-fit walks
    :data:`_L10_FFN_UNIT_LAYOUT_MAIN` in declaration order and lands
    each sub-stage at the lowest free gap; because the layout is fully
    contiguous (every entry starts where the previous one ended)
    first-fit reproduces the legacy offsets bit-for-bit, so
    ``_set_layer10_alu``'s monotonic ``unit = 0`` counter still lands
    on the same indices. The allocator is stashed on
    ``block.ffn._l10_unit_allocator`` for downstream auditing.
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_MAIN:
        allocator.alloc(name, n_units)
    return allocator


def _allocate_l10_post_ops_combined_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the combined post-op FFN.

    Mirrors the inline ``offset`` walk in ``make_l10_post_ops_combined``:
    one ``BinaryOpByteZeroingPostOp``, three ``CarryPropagationPostOp``,
    one ``ComparisonCombine`` -- each occupying the slot its
    predecessor's ``hidden_dim`` advances to. The carry slice is zeroed
    by the existing post-bake step but still occupies its declared
    range so the comparison-combine offset remains stable.

    Phase 7.B.6: ``pin=`` is dropped. First-fit reproduces the legacy
    offsets bit-for-bit because the layout is fully contiguous in
    declaration order.
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED:
        allocator.alloc(name, n_units)
    return allocator


_L10_BINARY_OP_BYTE_ZEROING_OP_DIMS = (
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR", "OP_MUL", "OP_DIV", "OP_MOD",
)


def _l10_binary_op_byte_zeroing_rules(S: float) -> tuple[FFNRule, ...]:
    """Declarative rules for ``BinaryOpByteZeroingPostOp`` (8 units).

    Mirrors ``vm_step.BinaryOpByteZeroingPostOp._bake_weights``: four
    opcode-gated detectors (units 0..3, gate by the binary-op set) and
    four bitwise-gated detectors (units 4..7, gate by ``TEMP+3``).

    Per the legacy ``wire_zeroing_writes(unit_offset)`` helper, within
    each group of four:

      * unit_offset+0 wipes the OUTPUT_LO band (16 cells, each -3.0/S),
      * unit_offset+1 wipes the OUTPUT_HI band (16 cells, each -3.0/S),
      * unit_offset+2 adds OUTPUT_LO+0 += 5.0/S,
      * unit_offset+3 adds OUTPUT_HI+0 += 5.0/S.

    The opcode-gated detectors use ``gate=None`` with multi-term
    ``gate_terms`` summing all 11 opcode flags (each weight 1.0). The
    bitwise detectors use ``gate="TEMP+3"`` directly.
    """

    def conds_opcode_gated():
        return (
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("TEMP+8", -1000.0),
            ("TEMP+9", -1000.0),
        )

    def conds_bitwise_gated():
        return (
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("TEMP+3", 1.0),
            ("TEMP+8", -1000.0),
            ("TEMP+9", -1000.0),
        )

    opcode_gate_terms = tuple(
        (op_dim, 1.0) for op_dim in _L10_BINARY_OP_BYTE_ZEROING_OP_DIMS
    )

    output_lo_wipe = tuple(
        (f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)
    )
    output_hi_wipe = tuple(
        (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
    )

    rules: list[FFNRule] = []

    # Units 0..3: opcode-gated detectors.
    for unit_idx, writes in enumerate((
        output_lo_wipe,
        output_hi_wipe,
        (("OUTPUT_LO+0", 5.0 / S),),
        (("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
    )):
        rules.append(FFNRule.gated_write(
            conditions=conds_opcode_gated(),
            threshold=1.5,
            gate=None,
            gate_terms=opcode_gate_terms,
            gate_bias=0.0,
            writes=writes,
            name=f"l10_binary_op_byte_zeroing_opcode_unit{unit_idx}",
            scope=(
                "IS_BYTE and ("
                + " or ".join(_L10_BINARY_OP_BYTE_ZEROING_OP_DIMS)
                + ")"
            ),
        ))

    # Units 4..7: bitwise-gated detectors (gate = TEMP+3).
    for unit_idx, writes in enumerate((
        output_lo_wipe,
        output_hi_wipe,
        (("OUTPUT_LO+0", 5.0 / S),),
        (("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
    )):
        rules.append(FFNRule.gated_write(
            conditions=conds_bitwise_gated(),
            threshold=2.5,
            gate="TEMP+3",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=writes,
            name=f"l10_binary_op_byte_zeroing_bitwise_unit{unit_idx}",
            scope="IS_BYTE and TEMP+3",
        ))

    return tuple(rules)


_L10_CARRY_NON_ARITH_OPS = (
    "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
    "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND", "OP_EQ",
    "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_SHL",
    "OP_SHR", "OP_MUL", "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
    "OP_PUTCHAR", "OP_GETCHAR",
)

_L10_CARRY_BYTE_DIM_BY_IDX = (
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
)


def _l10_carry_propagation_rules(
    S: float, *, byte_idx: int, cascade: bool,
) -> tuple[FFNRule, ...]:
    """Declarative rules for one ``CarryPropagationPostOp`` instance (512 units).

    Mirrors ``vm_step.CarryPropagationPostOp._bake_weights``: 256 ADD
    carry units followed by 256 SUB borrow units, indexed by
    ``(lo, hi)`` over ``range(16) x range(16)`` for the active byte.

    Differences captured per (byte_idx, cascade):

      * ``byte_dim`` flag: BYTE_INDEX_0/1/2 (no byte_idx=3 here).
      * ``add_carry_in`` / ``sub_carry_in`` choose between CARRY+1/+2
        (non-cascade byte_idx=0) and CARRY+3/+3 (cascade=True for
        byte_idx=1/2).
      * ``cascade=True`` routes mutual exclusion through TEMP+8/9
        instead of the carry dim pair.
      * The threshold drops to ``cascade_carry_threshold=40`` when
        cascade=True; non-cascade uses ``carry_threshold=56``.

    Output cells: the imperative bake writes
    ``W_down[OUTPUT_LO+lo] = -2/S`` then later
    ``W_down[OUTPUT_LO+new_lo] = +2/S`` with ``=`` semantics. When
    ``new_lo == lo`` (or ``new_hi == hi``) the second assignment
    overwrites the first, giving a net +2/S on that cell. The
    ``CompilerIR.lower_ffn`` lowerer uses ``+=`` so the rule below
    emits only the surviving write per cell (``+2/S`` when they
    collide, both writes when they don't). The CARRY+3 high-overflow
    write only fires at ``(15,15)`` for ADD and ``(0,0)`` for SUB,
    and only when ``byte_idx < 2``.
    """

    if byte_idx not in (0, 1, 2):
        raise ValueError(f"byte_idx must be 0, 1, or 2; got {byte_idx}")

    byte_dim_name = _L10_CARRY_BYTE_DIM_BY_IDX[byte_idx]
    wrong_byte_dim_names = tuple(
        name for i, name in enumerate(_L10_CARRY_BYTE_DIM_BY_IDX)
        if i != byte_idx
    )
    # Phase 8.D: carry-in / carry-out byte-cell refs name the
    # (carry, alu, byte_index) semantic family lookup; byte-identical
    # to the legacy "CARRY+<k>" strings via DimRef.parse.
    add_carry_in_name = dim_ref("carry", "alu", 3 if cascade else 1)
    sub_carry_in_name = dim_ref("carry", "alu", 3 if cascade else 2)
    carry_byte3 = dim_ref("carry", "alu", 3)
    threshold = 40.0 if cascade else 56.0

    # carry_weight=1.0, output_weight=20.0, mismatch_weight=0.0 in the
    # legacy bake; mismatch writes drop out of the rule because they
    # multiply to zero (``-S * 0 = 0`` produces no W_up cell).
    carry_weight = 1.0
    output_weight = 20.0

    def base_conds(carry_in_name: str) -> list[tuple[str, float]]:
        conds: list[tuple[str, float]] = [
            (carry_in_name, carry_weight),
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("MARK_AX", -5000.0),
            ("MARK_PC", -5000.0),
            (byte_dim_name, 1.0),
        ]
        # Suppress non-arithmetic opcodes.
        for op_name in _L10_CARRY_NON_ARITH_OPS:
            conds.append((op_name, -20.0))
        # TEMP+3 suppression (BITWISE_OP indicator).
        conds.append(("TEMP+3", -10.0))
        # Wrong byte position suppression.
        for wrong_name in wrong_byte_dim_names:
            conds.append((wrong_name, -10.0))
        return conds

    def add_rule_for(lo: int, hi: int) -> FFNRule:
        new_val = lo + hi * 16 + 1
        new_lo = new_val & 0xF
        new_hi = (new_val >> 4) & 0xF
        conds = base_conds(add_carry_in_name)
        # ADD-specific mutual exclusion vs SUB.
        conds.append(("OP_SUB", -20.0))
        if cascade:
            conds.append(("TEMP+8", 1.0))
            conds.append(("TEMP+9", -10.0))
        else:
            # sub_carry_in (CARRY+2) suppresses ADD firing under SUB.
            conds.append((sub_carry_in_name, -10.0))
        # OUTPUT_LO/HI match boosts (only matching nibble; mismatches
        # are 0 because mismatch_weight=0 in the legacy bake).
        conds.append((f"OUTPUT_LO+{lo}", output_weight))
        conds.append((f"OUTPUT_HI_THIS_STEP+{hi}", output_weight))

        writes: list[tuple[str, float]] = []
        # Collapse the legacy ``= -2/S`` + ``= +2/S`` pair using the
        # rule that lower_ffn uses ``+=``: when new == old, emit only
        # the surviving +2/S; otherwise emit both writes as distinct
        # cells.
        if new_lo == lo:
            writes.append((f"OUTPUT_LO+{lo}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_LO+{lo}", -2.0 / S))
            writes.append((f"OUTPUT_LO+{new_lo}", 2.0 / S))
        if new_hi == hi:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", -2.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{new_hi}", 2.0 / S))
        if lo == 15 and hi == 15 and byte_idx < 2:
            writes.append((carry_byte3, 2.0 / S))

        return FFNRule.gated_write(
            conditions=tuple(conds),
            threshold=threshold,
            gate=add_carry_in_name,
            gate_weight=0.5,
            gate_bias=0.0,
            writes=tuple(writes),
            name=f"l10_carry_byte{byte_idx}_add_lo{lo}_hi{hi}",
            scope=(
                f"IS_BYTE and {byte_dim_name} and not MARK_AX "
                f"and not MARK_PC"
            ),
        )

    def sub_rule_for(lo: int, hi: int) -> FFNRule:
        new_val = (lo + hi * 16 - 1) & 0xFF
        new_lo = new_val & 0xF
        new_hi = (new_val >> 4) & 0xF
        conds = base_conds(sub_carry_in_name)
        conds.append(("OP_ADD", -20.0))
        if cascade:
            conds.append(("TEMP+9", 1.0))
            conds.append(("TEMP+8", -10.0))
        else:
            conds.append((add_carry_in_name, -10.0))
        conds.append((f"OUTPUT_LO+{lo}", output_weight))
        conds.append((f"OUTPUT_HI_THIS_STEP+{hi}", output_weight))

        writes: list[tuple[str, float]] = []
        if new_lo == lo:
            writes.append((f"OUTPUT_LO+{lo}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_LO+{lo}", -2.0 / S))
            writes.append((f"OUTPUT_LO+{new_lo}", 2.0 / S))
        if new_hi == hi:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", -2.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{new_hi}", 2.0 / S))
        if lo == 0 and hi == 0 and byte_idx < 2:
            writes.append((carry_byte3, 2.0 / S))

        return FFNRule.gated_write(
            conditions=tuple(conds),
            threshold=threshold,
            gate=sub_carry_in_name,
            gate_weight=0.5,
            gate_bias=0.0,
            writes=tuple(writes),
            name=f"l10_carry_byte{byte_idx}_sub_lo{lo}_hi{hi}",
            scope=(
                f"IS_BYTE and {byte_dim_name} and not MARK_AX "
                f"and not MARK_PC"
            ),
        )

    rules: list[FFNRule] = []
    for lo in range(16):
        for hi in range(16):
            rules.append(add_rule_for(lo, hi))
    for lo in range(16):
        for hi in range(16):
            rules.append(sub_rule_for(lo, hi))
    return tuple(rules)


def _l10_comparison_combine_rules(S: float) -> tuple[FFNRule, ...]:
    """Declarative rules for ``ComparisonCombine`` (18 units).

    Mirrors ``vm_step.ComparisonCombine._bake_weights``. For each of
    EQ/NE/LT/GT/LE/GE the post-op emits one *default* unit that writes
    the initial result and one or two *override* units that flip the
    result when CMP[0..3] flags indicate the opposite outcome.

      * Default unit: constant_write style (``W_gate[unit, CONST]``
        was the legacy gate, but ``ComparisonCombine`` actually sets
        ``b_gate = 1.0`` with no W_gate cell, matching
        ``constant_write``'s ``gate=None``/``gate_bias=1.0`` form).
        Writes ``OUTPUT_LO+default_result`` and ``OUTPUT_HI+0`` at
        +2/S.
      * Override 2-way: gated by the opcode dim, conditions sum
        MARK_AX + one CMP flag with threshold 1.5, writes a +4/S/-4/S
        pair on OUTPUT_LO.
      * Override 3-way: gated by the opcode dim, conditions sum
        MARK_AX + two CMP flags with threshold 2.5, writes a +4/S/
        -4/S pair on OUTPUT_LO.

    All units include a strong MARK_PC blocker (``-50``) to prevent
    leaked OP_NE/OP_GT/OP_GE or CMP residue from corrupting PC
    predictions; see the 2026-05-09 fix comment in vm_step.py.
    """

    MARK_PC_BLOCK = -50.0

    def cmp_default(op_name: str, default_result: int, *, idx: int) -> FFNRule:
        return FFNRule.constant_write(
            conditions=(
                ("MARK_AX", 1.0),
                (op_name, 1.0),
                ("MARK_PC", MARK_PC_BLOCK),
            ),
            threshold=1.5,
            writes=(
                (f"OUTPUT_LO+{default_result}", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
            name=f"l10_cmp_default_{op_name.lower()}_{idx}",
            scope=f"MARK_AX and {op_name} and not MARK_PC",
        )

    def cmp_override_2way(
        op_name: str, cmp_name: str, to_result: int, from_result: int,
        *, idx: int,
    ) -> FFNRule:
        return FFNRule.gated_write(
            conditions=(
                ("MARK_AX", 1.0),
                (cmp_name, 1.0),
                ("MARK_PC", MARK_PC_BLOCK),
            ),
            threshold=1.5,
            gate=op_name,
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                (f"OUTPUT_LO+{to_result}", 4.0 / S),
                (f"OUTPUT_LO+{from_result}", -4.0 / S),
            ),
            name=f"l10_cmp_override2_{op_name.lower()}_{idx}",
            scope=f"MARK_AX and {op_name} and {cmp_name} and not MARK_PC",
        )

    def cmp_override_3way(
        op_name: str, cmp_name1: str, cmp_name2: str,
        to_result: int, from_result: int, *, idx: int,
    ) -> FFNRule:
        return FFNRule.gated_write(
            conditions=(
                ("MARK_AX", 1.0),
                (cmp_name1, 1.0),
                (cmp_name2, 1.0),
                ("MARK_PC", MARK_PC_BLOCK),
            ),
            threshold=2.5,
            gate=op_name,
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                (f"OUTPUT_LO+{to_result}", 4.0 / S),
                (f"OUTPUT_LO+{from_result}", -4.0 / S),
            ),
            name=f"l10_cmp_override3_{op_name.lower()}_{idx}",
            scope=(
                f"MARK_AX and {op_name} and {cmp_name1} and "
                f"{cmp_name2} and not MARK_PC"
            ),
        )

    rules: list[FFNRule] = []

    # EQ: default 0, override (CMP+1,CMP+2 -> 1)
    rules.append(cmp_default("OP_EQ", 0, idx=0))
    rules.append(cmp_override_3way("OP_EQ", "CMP+1", "CMP+2", 1, 0, idx=1))

    # NE: default 1, override (CMP+1,CMP+2 -> 0)
    rules.append(cmp_default("OP_NE", 1, idx=2))
    rules.append(cmp_override_3way("OP_NE", "CMP+1", "CMP+2", 0, 1, idx=3))

    # LT: default 0, override CMP+0 -> 1, override (CMP+1,CMP+3 -> 1)
    rules.append(cmp_default("OP_LT", 0, idx=4))
    rules.append(cmp_override_2way("OP_LT", "CMP+0", 1, 0, idx=5))
    rules.append(cmp_override_3way("OP_LT", "CMP+1", "CMP+3", 1, 0, idx=6))

    # GT: default 1, override CMP+0 -> 0, two 3-way overrides -> 0
    rules.append(cmp_default("OP_GT", 1, idx=7))
    rules.append(cmp_override_2way("OP_GT", "CMP+0", 0, 1, idx=8))
    rules.append(cmp_override_3way("OP_GT", "CMP+1", "CMP+3", 0, 1, idx=9))
    rules.append(cmp_override_3way("OP_GT", "CMP+1", "CMP+2", 0, 1, idx=10))

    # LE: default 0, override CMP+0 -> 1, two 3-way overrides -> 1
    rules.append(cmp_default("OP_LE", 0, idx=11))
    rules.append(cmp_override_2way("OP_LE", "CMP+0", 1, 0, idx=12))
    rules.append(cmp_override_3way("OP_LE", "CMP+1", "CMP+3", 1, 0, idx=13))
    rules.append(cmp_override_3way("OP_LE", "CMP+1", "CMP+2", 1, 0, idx=14))

    # GE: default 1, override CMP+0 -> 0, override (CMP+1,CMP+3 -> 0)
    rules.append(cmp_default("OP_GE", 1, idx=15))
    rules.append(cmp_override_2way("OP_GE", "CMP+0", 0, 1, idx=16))
    rules.append(cmp_override_3way("OP_GE", "CMP+1", "CMP+3", 0, 1, idx=17))

    return tuple(rules)


def _allocate_l10_tail_bit32_units(n_rules: int) -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for ``tail_bit32_result_correction``.

    The tail PureFFN's ``hidden_dim`` equals ``len(rules)`` so this
    layout is parameterised: ``n_rules`` must match the
    ``_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL`` constant. A mismatch
    means someone changed the tail rule set without updating the
    layout table; fail loudly rather than silently miss a declared range.

    Phase 7.B.6: ``pin=`` is dropped. With an empty L17 tail FFN pool
    first-fit lands the single 2059-unit range at start=0, matching
    the legacy bake bit-for-bit. The wide-MUL high-byte fix chain
    (MUL/LEA tail rules) is unaffected because those rules consume
    layout-by-name, not by pin offset.
    """
    if n_rules != _L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL:
        raise ValueError(
            f"tail_bit32_result_correction rule count drift: helper "
            f"produced {n_rules} rules, allocator expects "
            f"{_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL}"
        )
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_TAIL_BIT32:
        allocator.alloc(name, n_units)
    return allocator


# === L10 ALU FFNRule generators (Phase 6 Wave 4I migration) ==========
#
# Per-sub-stage declarative rules that reproduce ``_set_layer10_alu``
# (vm_step.py) byte-for-byte. Each generator mirrors one contiguous
# range of the ``_L10_FFN_UNIT_LAYOUT_MAIN`` table so a downstream
# lower via ``Primitives.lower_ffn_rules`` lands on the same pinned
# offsets the imperative helper writes today.

def _layer10_alu_cmp_combine_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 cmp_combine rules: 18 units for EQ/NE/LT/GT/LE/GE.

    Mirrors the ``_cmp_default`` / ``_cmp_override_2way`` /
    ``_cmp_override_3way`` helpers in ``vm_step._set_layer10_alu``.

    Each comparison opcode has one default unit (writes a baseline 0 or
    1 result + an OUTPUT_HI[0]=1 marker, ungated via ``b_gate=1.0``)
    followed by 1-3 override units (each gated on the OP_* dim,
    flipping the result via a +4.0/S / -4.0/S pair on OUTPUT_LO) that
    fire on CMP-flag combinations from L9.
    """

    def _cmp_default(op_name: str, default_result: int) -> FFNRule:
        # Helper: W_up[MARK_AX]=S, W_up[op]=S, b_up=-S*1.5, b_gate=1.0
        # (gate=None), W_down[OUTPUT_LO+default_result]=2/S,
        # W_down[OUTPUT_HI+0]=2/S.
        return FFNRule.constant_write(
            name=f"l10_cmp_{op_name.lower()}_default",
            conditions=(
                ("MARK_AX", 1.0),
                (f"OP_{op_name}", 1.0),
            ),
            threshold=1.5,
            writes=(
                (f"OUTPUT_LO+{default_result}", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
        )

    def _cmp_override_2way(
        op_name: str, cmp_idx: int, to_result: int, from_result: int,
        suffix: str,
    ) -> FFNRule:
        # Helper: W_up[MARK_AX]=S, W_up[CMP+i]=S, b_up=-S*1.5,
        # W_gate[op]=1.0 (b_gate=0), W_down[OUTPUT_LO+to]=4/S,
        # W_down[OUTPUT_LO+from]=-4/S.
        # Phase 8.D: the OP_<NAME> gate names the (opcode_flag, NAME)
        # semantic family member.
        return FFNRule.gated_write(
            name=f"l10_cmp_{op_name.lower()}_override2_{suffix}",
            conditions=(
                ("MARK_AX", 1.0),
                (f"CMP+{cmp_idx}", 1.0),
            ),
            threshold=1.5,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                (f"OUTPUT_LO+{to_result}", 4.0 / S),
                (f"OUTPUT_LO+{from_result}", -4.0 / S),
            ),
        )

    def _cmp_override_3way(
        op_name: str, cmp_idx1: int, cmp_idx2: int,
        to_result: int, from_result: int, suffix: str,
    ) -> FFNRule:
        # Helper: W_up[MARK_AX]=S, W_up[CMP+i]=S, W_up[CMP+j]=S,
        # b_up=-S*4.0, W_gate[op]=1.0 (b_gate=0),
        # W_down[OUTPUT_LO+to]=4/S, W_down[OUTPUT_LO+from]=-4/S.
        # Phase 8.D: the OP_<NAME> gate names the (opcode_flag, NAME)
        # semantic family member.
        return FFNRule.gated_write(
            name=f"l10_cmp_{op_name.lower()}_override3_{suffix}",
            conditions=(
                ("MARK_AX", 1.0),
                (f"CMP+{cmp_idx1}", 1.0),
                (f"CMP+{cmp_idx2}", 1.0),
            ),
            threshold=4.0,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                (f"OUTPUT_LO+{to_result}", 4.0 / S),
                (f"OUTPUT_LO+{from_result}", -4.0 / S),
            ),
        )

    return (
        # EQ: default=0, override to 1 when hi_eq AND lo_eq.
        _cmp_default("EQ", 0),
        _cmp_override_3way("EQ", 1, 2, 1, 0, "hi_eq_lo_eq"),
        # NE: default=1, override to 0 when hi_eq AND lo_eq.
        _cmp_default("NE", 1),
        _cmp_override_3way("NE", 1, 2, 0, 1, "hi_eq_lo_eq"),
        # LT: default=0, override to 1 on hi_lt OR (hi_eq AND lo_lt).
        _cmp_default("LT", 0),
        _cmp_override_2way("LT", 0, 1, 0, "hi_lt"),
        _cmp_override_3way("LT", 1, 3, 1, 0, "hi_eq_lo_lt"),
        # GT: default=1, override to 0 on hi_lt / (hi_eq AND lo_lt) /
        # (hi_eq AND lo_eq).
        _cmp_default("GT", 1),
        _cmp_override_2way("GT", 0, 0, 1, "hi_lt"),
        _cmp_override_3way("GT", 1, 3, 0, 1, "hi_eq_lo_lt"),
        _cmp_override_3way("GT", 1, 2, 0, 1, "hi_eq_lo_eq"),
        # LE: default=0, override to 1 on hi_lt / (hi_eq AND lo_lt) /
        # (hi_eq AND lo_eq).
        _cmp_default("LE", 0),
        _cmp_override_2way("LE", 0, 1, 0, "hi_lt"),
        _cmp_override_3way("LE", 1, 3, 1, 0, "hi_eq_lo_lt"),
        _cmp_override_3way("LE", 1, 2, 1, 0, "hi_eq_lo_eq"),
        # GE: default=1, override to 0 on hi_lt / (hi_eq AND lo_lt).
        _cmp_default("GE", 1),
        _cmp_override_2way("GE", 0, 0, 1, "hi_lt"),
        _cmp_override_3way("GE", 1, 3, 0, 1, "hi_eq_lo_lt"),
    )


def _layer10_alu_bitwise_rules(
    S: float, *, op_name: str, op_fn,
) -> tuple[FFNRule, ...]:
    """L10 bitwise OR/XOR/AND rules: 256 lo + 256 hi units per opcode.

    Each unit is a 3-way AND across (MARK_AX, ALU_*[a], AX_CARRY_*[b])
    that fires only when the OP_* gate is hot. Weights (40, 30, 30) and
    threshold 80 implement the balanced 3-way AND from
    ``vm_step._set_layer10_alu`` (see the BUG FIX 2026-04-16 comment):

      * all three present: 40 + 30 + 30 = 100 > 80 -> fires
      * any two present:   max(40 + 30) = 70 < 80  -> blocked

    ``op_fn`` is the bitwise function (``operator.or_`` / ``xor`` /
    ``and_``) used to compute the result nibble.
    """

    # Phase 8.D: pre-bind the (opcode_flag, op_name) gate ref so
    # the inner loop reuses one dim_ref call.
    gate_op = dim_ref("opcode_flag", op_name)
    rules: list[FFNRule] = []
    for nibble_label, alu_dim, carry_dim, out_dim in (
        ("lo", "ALU_LO", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "ALU_HI", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                rules.append(FFNRule.gated_write(
                    name=(
                        f"l10_bitwise_{op_name.lower()}_{nibble_label}_"
                        f"a{a:x}_b{b:x}"
                    ),
                    conditions=(
                        ("MARK_AX", 40.0),
                        (f"{alu_dim}+{a}", 30.0),
                        (f"{carry_dim}+{b}", 30.0),
                    ),
                    threshold=80.0,
                    gate=gate_op,
                    gate_weight=1.0,
                    gate_bias=0.0,
                    writes=((f"{out_dim}+{result}", 2.0 / S),),
                ))
    return tuple(rules)


def _layer10_alu_bitwise_or_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise OR: 512 units (256 lo + 256 hi) gated on OP_OR."""

    import operator

    return _layer10_alu_bitwise_rules(S, op_name="OR", op_fn=operator.or_)


def _layer10_alu_bitwise_xor_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise XOR: 512 units (256 lo + 256 hi) gated on OP_XOR."""

    import operator

    return _layer10_alu_bitwise_rules(S, op_name="XOR", op_fn=operator.xor)


def _layer10_alu_bitwise_and_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise AND: 512 units (256 lo + 256 hi) gated on OP_AND."""

    import operator

    return _layer10_alu_bitwise_rules(S, op_name="AND", op_fn=operator.and_)


def _layer10_alu_shl_shr_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 SHL/SHR shift-out-zero shortcut: 4 units (2 per opcode).

    For shifts >= 8 the result byte is 0x00. Two cases per opcode:

      * Case A (shift >= 16): high nibble of the shift count is non-zero
        so ``AX_CARRY_HI[0]`` is NOT hot. The unit fires on
        ``MARK_AX * 60`` after subtracting ``-S * AX_CARRY_HI[0]`` to
        suppress shifts 0-15; ``b_up = -S * 59`` requires MARK_AX to be
        present to clear the threshold.
      * Case B (shift 8-15): high nibble = 0 (so ``AX_CARRY_HI[0] = 1``)
        and the low nibble is in 8..15. The unit fires when MARK_AX +
        ``AX_CARRY_HI[0]`` plus any one of ``AX_CARRY_LO[8..15]`` are
        present (threshold 80 vs 60 + 1 + 1 = 62 forces all three terms).

    Both cases write ``OUTPUT_LO[0]`` and ``OUTPUT_HI[0]`` to 1 so the
    next position emits a 0x00 byte. The OP_* gate selects which opcode
    the shortcut fires for.
    """

    def _case_a(op_name: str) -> FFNRule:
        # Phase 8.D: OP_<NAME> gate -> (opcode_flag, NAME).
        return FFNRule.gated_write(
            name=f"l10_{op_name.lower()}_shift_ge16_zero",
            conditions=(
                ("MARK_AX", 60.0),
                ("AX_CARRY_HI+0", -1.0),
            ),
            threshold=59.0,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                ("OUTPUT_LO+0", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
        )

    def _case_b(op_name: str) -> FFNRule:
        conditions = [
            ("MARK_AX", 60.0),
            ("AX_CARRY_HI+0", 1.0),
        ]
        for lo_bit in range(8, 16):
            conditions.append((f"AX_CARRY_LO+{lo_bit}", 1.0))
        # Phase 8.D: OP_<NAME> gate -> (opcode_flag, NAME).
        return FFNRule.gated_write(
            name=f"l10_{op_name.lower()}_shift_8_15_zero",
            conditions=tuple(conditions),
            threshold=80.0,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                ("OUTPUT_LO+0", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
        )

    return (
        _case_a("SHL"),
        _case_b("SHL"),
        _case_a("SHR"),
        _case_b("SHR"),
    )


# Suppressed opcodes for L10 ALU AX passthrough -- mirrors the
# ``suppressed_ops`` list in ``vm_step._set_layer10_alu``. Each opcode
# already owns the AX-byte-0 emission for its specific lane (L6 routing,
# L8 ALU, L15 memory lookup, DivModModule, etc.), so the L10 passthrough
# must NOT fire for them.
_L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS = (
    "OP_IMM",
    "OP_ADD",
    "OP_SUB",
    "OP_OR",
    "OP_XOR",
    "OP_AND",
    "OP_EQ",
    "OP_NE",
    "OP_LT",
    "OP_GT",
    "OP_LE",
    "OP_GE",
    "OP_MUL",
    "OP_DIV",
    "OP_MOD",
    "OP_SHL",
    "OP_SHR",
    "OP_LEA",
    "OP_LI",
    "OP_LC",
    "OP_JMP",
    "OP_EXIT",
    "OP_NOP",
    "OP_PUTCHAR",
)


def _layer10_alu_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 AX passthrough: 32 units (16 lo + 16 hi).

    Each unit fires at the AX marker only when none of the
    ``_L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS`` opcodes are active --
    every suppressed opcode contributes a ``-S`` term that pushes the
    pre-activation below the ``-S * 0.5`` threshold whenever the
    opcode flag is hot. Units 0..15 gate on ``AX_CARRY_LO[k]`` and route
    that one-hot into ``OUTPUT_LO[k]``; units 16..31 do the same on the
    hi nibble via ``AX_CARRY_HI[k]`` and ``OUTPUT_HI[k]``.
    """

    rules: list[FFNRule] = []
    for nibble_label, carry_dim, out_dim in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            conditions = [("MARK_AX", 1.0)]
            for op_dim in _L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS:
                conditions.append((op_dim, -1.0))
            rules.append(FFNRule.gated_write(
                name=f"l10_ax_passthrough_{nibble_label}_{k}",
                conditions=tuple(conditions),
                threshold=0.5,
                gate=f"{carry_dim}+{k}",
                gate_weight=1.0,
                gate_bias=0.0,
                writes=((f"{out_dim}+{k}", 2.0 / S),),
            ))
    return tuple(rules)


def _layer10_alu_mul_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 MUL lo-nibble lookup: 256 units gated on OP_MUL.

    For each (a, b) in 0..15 x 0..15, one 3-way AND unit fires only on
    the matching ALU_LO[a] / AX_CARRY_LO[b] one-hot pair at the AX
    marker, writing the lo nibble of (a * b) (i.e. ``(a * b) % 16``) to
    ``OUTPUT_LO``. Weights and threshold reuse the (40, 30, 30) / 80
    balanced 3-way AND from the bitwise sub-stages above so a single
    spurious one-hot in either operand band cannot fire the unit.
    """

    # Phase 8.D: OP_MUL gate -> (opcode_flag, MUL).
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []
    for a in range(16):
        for b in range(16):
            result = (a * b) % 16
            rules.append(FFNRule.gated_write(
                name=f"l10_mul_lo_a{a:x}_b{b:x}",
                conditions=(
                    ("MARK_AX", 40.0),
                    (f"ALU_LO+{a}", 30.0),
                    (f"AX_CARRY_LO+{b}", 30.0),
                ),
                threshold=80.0,
                gate=gate_mul,
                gate_weight=1.0,
                gate_bias=0.0,
                writes=((f"OUTPUT_LO+{result}", 2.0 / S),),
            ))
    return tuple(rules)


def _layer10_alu_rules(S: float) -> tuple[FFNRule, ...]:
    """Composite ordered ``FFNRule`` sequence for ``layer10_alu``.

    Concatenates all seven sub-stage rule lists in the exact order
    declared by ``_L10_FFN_UNIT_LAYOUT_MAIN`` so a single
    ``Primitives.lower_ffn_rules`` call lowers the entire 1846-unit FFN
    in cursor order (matching the legacy ``_set_layer10_alu`` walk
    byte-for-byte).
    """

    return (
        _layer10_alu_cmp_combine_rules(S)
        + _layer10_alu_bitwise_or_rules(S)
        + _layer10_alu_bitwise_xor_rules(S)
        + _layer10_alu_bitwise_and_rules(S)
        + _layer10_alu_mul_lo_rules(S)
        + _layer10_alu_shl_shr_zero_rules(S)
        + _layer10_alu_ax_passthrough_rules(S)
    )


def _layer10_alu_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` exposed by ``layer10_alu``."""

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer10_alu_rules(S))
    return ir


def _bake_layer10_alu_rules(ffn, S: float, BD) -> int:
    """Lower the composite ``layer10_alu`` rule list into ``ffn``.

    Returns the post-bake unit cursor (must equal
    :data:`_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL` for byte-identity with the
    historical 1846-unit footprint of ``vm_step._set_layer10_alu``).
    """

    rules = _layer10_alu_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD, Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )


def _bake_layer10_carry_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 0 carry relay spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_carry_relay_head_spec(BD, S),
        HD,
    )


def _layer10_carry_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_carry_relay_bake.head_0"),
        q=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.CONST, -L / 2),
            AP(33, BD.H1 + AX_IDX, L),
            AP(33, BD.CONST, -L / 2),
        ),
        k=(AP(0, BD.MARK_AX, L), AP(33, BD.CONST, L)),
        v=(AP(1, BD.CARRY + 1, 1.0), AP(2, BD.CARRY + 2, 1.0)),
        o=(AO(BD.CARRY + 1, 1, 1.0), AO(BD.CARRY + 2, 2, 1.0)),
    )


def _byte_passthrough_chain_spec(
    BD,
    *,
    head_idx: int,
    source_marker_dim: int,
    target_marker_dim: int,
    value_lo_dim: int,
    value_hi_dim: int,
    suppress_op_dims,
    S: float,
    is_byte_strength: float = 3.0,
    has_se_strength: float = 1.0,
    suppress_strength: float = 3.0,
    q0_threshold: float = 3.5,
    gate_const: float = -20000.0,
    gate_target_marker: float = 10000.0,
    gate_has_se: float = 10000.0,
    gate_extras=None,
) -> DeclarativeAttentionHeadSpec:
    L = S
    q = [
        AP(0, BD.IS_BYTE, L * is_byte_strength),
        AP(0, BD.HAS_SE, L * has_se_strength),
        AP(0, BD.CONST, -L * q0_threshold),
        AP(1, target_marker_dim, L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_0, L),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, gate_const),
        AP(33, target_marker_dim, gate_target_marker),
        AP(33, BD.HAS_SE, gate_has_se),
    ]
    for dim in suppress_op_dims:
        q.append(AP(0, dim, -L * suppress_strength))
    if gate_extras:
        for dim, weight in gate_extras:
            q.append(AP(33, dim, weight))

    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, source_marker_dim, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_1, L),
        AP(4, BD.BYTE_INDEX_2, L),
        AP(5, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for idx in range(16):
        v.append(AP(idx, value_lo_dim + idx, 1.0))
        v.append(AP(16 + idx, value_hi_dim + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, idx, 2.0))
        o.append(AO(BD.OUTPUT_HI + idx, 16 + idx, 2.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _bake_layer10_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 1 AX byte passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_ax_byte_passthrough_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[1] = 1.0


def _layer10_ax_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry AX bytes and reload LI bytes from the prior MEM value rows."""

    PC_IDX = 0
    AX_IDX = 1
    SP_IDX = 2
    BP_IDX = 3
    MEM_IDX = 4
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_byte_passthrough_bake.head_1"),
        source_marker_dim=BD.H1 + AX_IDX,
        target_marker_dim=BD.H1 + AX_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[
            BD.OP_IMM,
            BD.OP_LI_RELAY,
            BD.OP_LC_RELAY,
            BD.TEMP + 3,
            BD.CMP + 3,
        ],
        S=S,
    )
    M = 50.0 * S

    # LI reloads overwrite AX from memory. The ordinary AX byte chain is
    # intentionally blocked by OP_LI_RELAY so it does not carry the stale
    # address AX value. These side slots instead select the latest matching
    # MEM value byte and reuse the existing CLEAN_EMBED -> OUTPUT value path.
    STORE_SELECT = 5.0 * S
    VALUE_SELECT = 20.0
    ROW_SELECT = 50.0 * S
    ROW_SELECT_BIAS = -75.0 * S
    STORE_ROW_SELECT = 50.0 * S
    STORE_ROW_SELECT_BIAS = -50.0 * S
    STORE_AX_BYTE1_SELECT = 20.0 * S
    li_value_query = (
        AP(39, BD.OP_LI_RELAY, M),
    )
    marker_query = (
        AP(40, BD.OP_LI_RELAY, M),
        AP(40, BD.MARK_AX, ROW_SELECT),
        AP(40, BD.CONST, ROW_SELECT_BIAS),
    )
    byte0_query = (
        AP(41, BD.OP_LI_RELAY, M),
        AP(41, BD.BYTE_INDEX_0, ROW_SELECT),
        AP(41, BD.CONST, ROW_SELECT_BIAS),
    )
    byte1_query = (
        AP(42, BD.OP_LI_RELAY, M),
        AP(42, BD.BYTE_INDEX_1, ROW_SELECT),
        AP(42, BD.CONST, ROW_SELECT_BIAS),
    )
    byte2_query = (
        AP(43, BD.OP_LI_RELAY, M),
        AP(43, BD.BYTE_INDEX_2, ROW_SELECT),
        AP(43, BD.CONST, ROW_SELECT_BIAS),
    )
    marker_store_query = (
        AP(44, BD.OP_LI_RELAY, M),
        AP(44, BD.MARK_AX, STORE_ROW_SELECT),
        AP(44, BD.CONST, STORE_ROW_SELECT_BIAS),
    )
    marker_addr_source_query = (
        AP(48, BD.OP_LI_RELAY, M),
        AP(48, BD.MARK_AX, STORE_ROW_SELECT),
        AP(48, BD.CONST, STORE_ROW_SELECT_BIAS),
    )
    byte0_store_query = (
        AP(45, BD.OP_LI_RELAY, M),
        AP(45, BD.BYTE_INDEX_0, STORE_ROW_SELECT),
        AP(45, BD.CONST, STORE_ROW_SELECT_BIAS),
    )
    byte1_store_query = (
        AP(46, BD.OP_LI_RELAY, M),
        AP(46, BD.BYTE_INDEX_1, STORE_ROW_SELECT),
        AP(46, BD.CONST, STORE_ROW_SELECT_BIAS),
    )
    byte2_store_query = (
        AP(47, BD.OP_LI_RELAY, M),
        AP(47, BD.BYTE_INDEX_2, STORE_ROW_SELECT),
        AP(47, BD.CONST, STORE_ROW_SELECT_BIAS),
    )
    store_ax_byte1_query = (
        AP(81, BD.OP_SI, STORE_AX_BYTE1_SELECT),
        AP(81, BD.OP_SC, STORE_AX_BYTE1_SELECT),
        AP(81, BD.MARK_AX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + PC_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + SP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + BP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H4 + BP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H3 + MEM_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_1, -10.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_2, -10.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_3, -10.0 * STORE_AX_BYTE1_SELECT),
    )
    store_ax_byte1_key = (
        AP(81, BD.IS_BYTE, STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + AX_IDX, STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_1, STORE_AX_BYTE1_SELECT),
        AP(81, BD.OP_IMM, STORE_AX_BYTE1_SELECT),
    )
    return replace(
        spec,
        q=(
            spec.q
            + li_value_query
            + marker_query
            + byte0_query
            + byte1_query
            + byte2_query
            + marker_store_query
            + byte0_store_query
            + byte1_store_query
            + byte2_store_query
            + marker_addr_source_query
            + store_ax_byte1_query
        ),
        k=spec.k + (
            AP(39, BD.MEM_VAL_B0, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B1, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B2, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B3, VALUE_SELECT),
            AP(40, BD.MEM_VAL_B1, M),
            AP(41, BD.MEM_VAL_B2, M),
            AP(42, BD.MEM_VAL_B3, M),
            AP(43, BD.MEM_VAL_B3, M),
            AP(44, BD.MEM_STORE, STORE_SELECT),
            AP(45, BD.MEM_STORE, STORE_SELECT),
            AP(45, BD.MEM_ADDR_SRC, 1.0),
            AP(46, BD.MEM_STORE, STORE_SELECT),
            AP(46, BD.MEM_ADDR_SRC, 1.0),
            AP(47, BD.MEM_STORE, STORE_SELECT),
            AP(47, BD.MEM_ADDR_SRC, 1.0),
            AP(48, BD.MEM_ADDR_SRC, 1.0),
        ) + store_ax_byte1_key,
    )


def _bake_layer10_sp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 2 SP byte passthrough spec."""
    SP_IDX = 2
    Primitives.generate_attention_head(
        attn,
        _layer10_sp_byte_passthrough_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[2] = 1.0


def _bake_layer10_bp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 7 BP byte passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_bp_byte_passthrough_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[7] = 1.0


def _layer10_sp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    SP_IDX = 2
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_sp_byte_passthrough_bake.head_2"),
        source_marker_dim=BD.H1 + SP_IDX,
        target_marker_dim=BD.H1 + SP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.PSH_AT_SP, BD.MARK_SP],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.MARK_SP, 10000.0),
            (BD.PSH_AT_SP, -10000.0),
        ],
    )
    L = S
    marker_s = 300.0
    # Marker carry-forward for SP byte 0 on step 1+. The byte-chain above
    # handles SP byte positions; the marker itself needs to copy the previous
    # step's SP byte 0 unless the current op is actively rewriting SP (PSH,
    # JSR, ENT, or POP/LEV/ADJ). L6 relays these as CMP[0]/CMP[2]/CMP[3]/CMP[4]
    # to the SP marker; without these blockers the marker carry-forward
    # re-copies the stale pre-op SP byte after L6 has applied the SP delta.
    # CMP[2]=ENT was previously missing, causing func_identity_* and other
    # post-ENT programs to emit step1:SP_byte0 = stale 0xf8 (pre-JSR target)
    # or 0xff (pre-bootstrap) instead of the ENT-adjusted 0xf0/0xe0.
    return DeclarativeAttentionHeadSpec(
        head_idx=spec.head_idx,
        q=spec.q + (
            AP(34, BD.MARK_SP, marker_s),
            AP(34, BD.HAS_SE, marker_s),
            AP(34, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(34, BD.CMP + 4, -2.0 * marker_s),
            AP(34, BD.OP_JSR, -2.0 * marker_s),
            AP(34, BD.CMP + 3, -2.0 * marker_s),
            AP(34, BD.CMP + 2, -2.0 * marker_s),
            AP(34, BD.OP_ENT, -2.0 * marker_s),
            AP(34, BD.CONST, -marker_s),
            AP(35, BD.MARK_SP, marker_s),
            AP(35, BD.HAS_SE, marker_s),
            AP(35, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(35, BD.CMP + 4, -2.0 * marker_s),
            AP(35, BD.OP_JSR, -2.0 * marker_s),
            AP(35, BD.CMP + 3, -2.0 * marker_s),
            AP(35, BD.CMP + 2, -2.0 * marker_s),
            AP(35, BD.OP_ENT, -2.0 * marker_s),
            AP(35, BD.CONST, -marker_s),
        ),
        k=spec.k + (
            AP(34, BD.H1 + SP_IDX, marker_s),
            AP(35, BD.BYTE_INDEX_0, marker_s),
        ),
        v=spec.v,
        o=spec.o,
    )


def _layer10_bp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    BP_IDX = 3
    AX_IDX = 1
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_bp_byte_passthrough_bake.head_7"),
        source_marker_dim=BD.H1 + BP_IDX,
        target_marker_dim=BD.H1 + BP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.OP_ENT, BD.OP_LEV],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.OP_ENT, -10000.0),
            (BD.OP_LEV, -10000.0),
        ],
    )
    # Stack-source top stores update STACK0 from the current AX byte, but the
    # ordinary stack persistence head shares an H1+AX key with a negative store
    # route.  Keep this top-store byte-0 route isolated in head 7 and select
    # the AX byte-0 row with both H1+AX and BYTE_INDEX_0, not H1+AX alone.
    M = 50.0 * S
    TOP_STORE_BIAS = -95.0 * S
    TOP_STORE_ADDR = 10.0 * S
    TOP_STORE_ADDR_BLOCK = -20.0 * TOP_STORE_ADDR
    TOP_STORE_CMP = 5.0 * S
    TOP_STORE_HAS_SE = 5.0 * S

    def top_store_query(slot: int, target_dim: int) -> tuple:
        return (
            AP(slot, BD.CONST, TOP_STORE_BIAS),
            AP(slot, BD.MEM_STORE, M),
            AP(slot, BD.MEM_ADDR_SRC, M),
            AP(slot, BD.CMP + 3, TOP_STORE_CMP),
            AP(slot, BD.HAS_SE, TOP_STORE_HAS_SE),
            AP(slot, target_dim, M),
            AP(slot, BD.ADDR_B0_LO + 0, TOP_STORE_ADDR),
            AP(slot, BD.ADDR_B0_HI + 14, TOP_STORE_ADDR),
            AP(slot, BD.ADDR_B0_LO + 8, TOP_STORE_ADDR_BLOCK),
            AP(slot, BD.ADDR_B0_HI + 15, TOP_STORE_ADDR_BLOCK),
            AP(slot, BD.H1 + 0, -3.0 * M),
            AP(slot, BD.H1 + 1, -3.0 * M),
            AP(slot, BD.H1 + 2, -3.0 * M),
            AP(slot, BD.H1 + 3, -3.0 * M),
        )

    return replace(
        spec,
        q=spec.q + (
            *top_store_query(40, BD.MARK_STACK0),
            *top_store_query(41, BD.MARK_STACK0),
            *top_store_query(42, BD.STACK0_BYTE0),
            *top_store_query(43, BD.STACK0_BYTE0),
            *top_store_query(44, BD.STACK0_BYTE1),
            *top_store_query(45, BD.STACK0_BYTE1),
            *top_store_query(46, BD.STACK0_BYTE2),
            *top_store_query(47, BD.STACK0_BYTE2),
        ),
        k=spec.k + (
            AP(40, BD.H1 + AX_IDX, M),
            AP(41, BD.BYTE_INDEX_0, M),
            AP(42, BD.H1 + AX_IDX, M),
            AP(43, BD.BYTE_INDEX_1, M),
            AP(44, BD.H1 + AX_IDX, M),
            AP(45, BD.BYTE_INDEX_2, M),
            AP(46, BD.H1 + AX_IDX, M),
            AP(47, BD.BYTE_INDEX_3, M),
        ),
    )


def _bake_layer10_psh_stack0_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 3 PSH STACK0 passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_psh_stack0_passthrough_head_spec(BD, S),
        HD,
    )


def _layer10_psh_stack0_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    BP_IDX = 3
    L = S
    q = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H4 + BP_IDX, L),
        AP(1, BD.H1 + BP_IDX, -L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.PSH_AT_SP, L),
        AP(3, BD.CONST, -L / 2),
        AP(4, BD.BYTE_INDEX_0, L),
        AP(5, BD.BYTE_INDEX_1, L),
        AP(6, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H4 + BP_IDX, 10000.0),
        AP(33, BD.H1 + BP_IDX, -10000.0),
        AP(33, BD.PSH_AT_SP, 10000.0),
        AP(33, BD.MARK_STACK0, -10000.0),
    ]
    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H1 + AX_IDX, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(6, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    # ---- LEA-local AX byte 0 differential routing (bug #33) ----
    #
    # The existing slot 0..31 V/O routes CLEAN_EMBED_LO/HI from the attended
    # AX byte 0 source row into STACK0 byte 0's OUTPUT_LO/HI at the PSH step.
    # For most opcodes (IMM, ADD, SHR, ...) this is correct because the
    # source's CLEAN_EMBED nibbles match its byte-0 value -- L8 wrote
    # OUTPUT[k] = CLEAN_EMBED[k] for the byte value at the AX byte 0 row.
    #
    # For LEA-local (BP + signed offset, e.g. var_simple / var_update / if_var)
    # the AX-byte-0 source row's nibbles diverge: CLEAN_EMBED holds the raw
    # immediate operand (e.g. 0x18) while OUTPUT holds the L8-computed
    # effective-address low byte (e.g. 0xE8). The PSH-passthrough head needs
    # the OUTPUT-band value, not the immediate. The L10 post-op rule
    # ``tail_lea_local_ax_marker_byte0_e8`` writes 0xE8 into OUTPUT_LO[8]/
    # OUTPUT_HI[14] at the AX marker row but only after L10 post_ops execute,
    # whereas head 3 fires earlier at phase 10.3 -- so head 3 reads the
    # L8-produced OUTPUT directly. That L8 value is exactly the desired
    # LEA-local byte. 2026-06-01 triage attributes 78 ``step4:STACK0_byte0``
    # corruption rows (var_simple +25, var_update +25, if_var +25, ~3 loop)
    # to this missing producer.
    #
    # Differential sub-pattern (slots 32..63, free per HD=64; main routing
    # uses 0..31 in V/O and slot 33 in Q/K only):
    #   slot 32 + k (k=0..15): V reads (OUTPUT_LO+k - CLEAN_EMBED_LO+k),
    #                          O writes the diff into OUTPUT_LO+k at Q row.
    #   slot 48 + k (k=0..15): V reads (OUTPUT_HI+k - CLEAN_EMBED_HI+k),
    #                          O writes the diff into OUTPUT_HI+k at Q row.
    #
    # Semantic-neutrality on non-LEA paths: at AX byte 0 source rows for
    # IMM / ADD / SHR / etc. the L8 FFN writes OUTPUT_LO/HI[k] = the same
    # one-hot nibble pattern that CLEAN_EMBED_LO/HI[k] carries (the byte
    # value matches the immediate input). The diff is ~0, so the added
    # routing contributes nothing -- existing behavior is preserved
    # (including var_three_* whose STACK0_byte0 came out correct under
    # the CLEAN_EMBED-only routing).
    #
    # On LEA-local AX byte 0 rows the diff is (LEA_computed - immediate);
    # adding it on top of the existing CLEAN_EMBED routing yields the
    # LEA-computed byte at STACK0_byte0 -- the fix.
    #
    # Q/K attention scoring is unchanged: head 3 already attends from PSH
    # STACK0 byte 0 to the most recent AX byte 0 row. For programs with a
    # PSH following a LEA-local (var_*, if_var, loop_*), that AX byte 0 is
    # the LEA's, and the differential V routes the (0xE8-0x18) nibble
    # deltas into the STACK0_byte0 OUTPUT band.
    for k_idx in range(16):
        v.append(AP(32 + k_idx, BD.OUTPUT_LO + k_idx, 1.0))
        v.append(AP(32 + k_idx, BD.CLEAN_EMBED_LO + k_idx, -1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, 32 + k_idx, 3.0))
    for k_idx in range(16):
        v.append(AP(48 + k_idx, BD.OUTPUT_HI + k_idx, 1.0))
        v.append(AP(48 + k_idx, BD.CLEAN_EMBED_HI + k_idx, -1.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 48 + k_idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_psh_stack0_passthrough_bake.head_3"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _bake_layer10_stack0_byte_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 STACK0 byte relay specs."""
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_persistence_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[6] = 1.0


def _layer10_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.TEMP + 3, 1000.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.TEMP + 3, 50.0),
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.TEMP + 3, 10000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 11.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 11.0))
    v.append(AP(0, BD.CONST, 1.0))
    for k_idx in range(16):
        o.append(AO(BD.ALU_LO + k_idx, 0, -8.0))
        o.append(AO(BD.ALU_HI + k_idx, 0, -8.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_4"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Relay stored STACK0 bytes for non-bitwise pop ALU byte post-ops.

    Head 4 is shared with the existing bitwise byte propagation path. Its
    scoring is intentionally touchy, so ADD/SUB use this separate head to
    recover higher STACK0 bytes from the stored MEM row without perturbing
    AND/OR/XOR. TEMP[3] is the bitwise relay and suppresses this head.
    """
    AX_IDX = 1
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.CMP + 3, 150000.0),
        AP(0, BD.TEMP + 3, -500.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.CMP + 3, 1000.0),
        # Do not put a negative TEMP[3] term in this match slot: the MEM
        # marker key uses negative MARK_MEM/CONST terms there, so weak
        # TEMP[3] residue turns into positive evidence for the wrong row.
        # The high-magnitude TEMP[3] blockers in slots 0 and 33 still suppress
        # true bitwise rows without polluting the stack-byte selection score.
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.CMP + 3, 1500000.0),
        AP(33, BD.TEMP + 3, -5000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 6.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 6.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_5"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_stack0_persistence_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry and update STACK0 bytes.

    L3 carries STACK0 byte 0 at the marker. This head handles bytes 1-3 by
    querying at the preceding byte position and reading the latest previous
    STACK0 byte through ALiBi preference.

    STORE rows are mutating: after SI/SC, the popped SP points at the just
    written local, so STACK0 must come from the current AX bytes. The store
    subroute below uses extra source-match slots so it dominates the ordinary
    persistence source and reads same-step AX byte 0..3 for STACK0
    marker/byte0/byte1/byte2 respectively.
    """
    AX_IDX = 1
    # The persistence route competes with inactive store-source slots. Those
    # store slots carry negative bias in Q but still see byte-index keys on
    # ordinary STACK0 byte rows; keep the direct STACK0-byte match dominant so
    # byte K reliably predicts byte K+1 across non-mutating steps.
    M = 50.0 * S
    STORE_TARGET = 50.0 * S
    STORE_GATE = 50.0 * S
    STORE_CMP = 5.0 * S
    STORE_HAS_SE = 5.0 * S
    STORE_BIAS = -85.0 * S
    q = [
        AP(0, BD.PSH_AT_SP, -300.0),
        AP(0, BD.OP_PSH, -300.0),
        AP(0, BD.CMP + 0, -300.0),
        AP(0, BD.CMP + 1, -300.0),
        AP(0, BD.CMP + 2, -300.0),
        AP(0, BD.CMP + 4, -300.0),
        AP(0, BD.OP_LEV, -300.0),
        AP(4, BD.STACK0_BYTE0, M),
        AP(4, BD.CMP + 3, -M),
        AP(5, BD.STACK0_BYTE1, M),
        AP(5, BD.CMP + 3, -M),
        AP(6, BD.STACK0_BYTE2, M),
        AP(6, BD.CMP + 3, -M),
        AP(7, BD.CONST, STORE_BIAS),
        AP(7, BD.MEM_STORE, STORE_GATE),
        AP(7, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(7, BD.CMP + 3, STORE_CMP),
        AP(7, BD.HAS_SE, STORE_HAS_SE),
        AP(7, BD.MARK_STACK0, STORE_TARGET),
        AP(8, BD.CONST, STORE_BIAS),
        AP(8, BD.MEM_STORE, STORE_GATE),
        AP(8, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(8, BD.CMP + 3, STORE_CMP),
        AP(8, BD.HAS_SE, STORE_HAS_SE),
        AP(8, BD.STACK0_BYTE0, STORE_TARGET),
        AP(9, BD.CONST, STORE_BIAS),
        AP(9, BD.MEM_STORE, STORE_GATE),
        AP(9, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(9, BD.CMP + 3, STORE_CMP),
        AP(9, BD.HAS_SE, STORE_HAS_SE),
        AP(9, BD.STACK0_BYTE1, STORE_TARGET),
        AP(10, BD.CONST, STORE_BIAS),
        AP(10, BD.MEM_STORE, STORE_GATE),
        AP(10, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(10, BD.CMP + 3, STORE_CMP),
        AP(10, BD.HAS_SE, STORE_HAS_SE),
        AP(10, BD.STACK0_BYTE2, STORE_TARGET),
        AP(11, BD.CONST, STORE_BIAS),
        AP(11, BD.MEM_STORE, STORE_GATE),
        AP(11, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(11, BD.CMP + 3, STORE_CMP),
        AP(11, BD.HAS_SE, STORE_HAS_SE),
        AP(11, BD.MARK_STACK0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE1, STORE_TARGET),
        AP(11, BD.STACK0_BYTE2, STORE_TARGET),
        AP(33, BD.CONST, -15000.0),
        AP(33, BD.HAS_SE, 10000.0),
        AP(33, BD.PSH_AT_SP, -30000.0),
        AP(33, BD.OP_PSH, -30000.0),
        AP(33, BD.CMP + 0, -30000.0),
        AP(33, BD.CMP + 1, -30000.0),
        AP(33, BD.CMP + 2, -30000.0),
        AP(33, BD.CMP + 4, -30000.0),
        AP(33, BD.OP_LEV, -30000.0),
        AP(33, BD.STACK0_BYTE0, 10000.0),
        AP(33, BD.STACK0_BYTE1, 10000.0),
        AP(33, BD.STACK0_BYTE2, 10000.0),
        AP(33, BD.STACK0_BYTE3, -30000.0),
    ]
    k = [
        AP(4, BD.STACK0_BYTE1, M),
        AP(5, BD.STACK0_BYTE2, M),
        AP(6, BD.STACK0_BYTE3, M),
        AP(7, BD.BYTE_INDEX_0, M),
        AP(8, BD.BYTE_INDEX_1, M),
        AP(9, BD.BYTE_INDEX_2, M),
        AP(10, BD.BYTE_INDEX_3, M),
        AP(11, BD.H1 + AX_IDX, M),
        AP(33, BD.CONST, 100.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_6"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_carry_relay_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_carry_relay_head_spec(proxy, 100.0),
        name="layer10_carry_relay_bake.head_0",
    )
    return ir


def _layer10_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_ax_byte_passthrough_head_spec(proxy, 100.0),
        name="layer10_byte_passthrough_bake.head_1",
    )
    return ir


def _layer10_sp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_sp_byte_passthrough_head_spec(proxy, 100.0),
        name="layer10_sp_byte_passthrough_bake.head_2",
    )
    return ir


def _layer10_bp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_bp_byte_passthrough_head_spec(proxy, 100.0),
        name="layer10_bp_byte_passthrough_bake.head_7",
    )
    return ir


def _layer10_psh_stack0_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_psh_stack0_passthrough_head_spec(proxy, 100.0),
        name="layer10_psh_stack0_passthrough_bake.head_3",
    )
    return ir


def _layer10_stack0_byte_relay_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_stack0_byte_relay_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_4",
    )
    ir.layer(0).attention.append(
        _layer10_nonbitwise_stack0_byte_relay_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_5",
    )
    ir.layer(0).attention.append(
        _layer10_stack0_persistence_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_6",
    )
    return ir


def make_layer10_carry_relay_op() -> Operation:
    """Topology anchor for L10 head 0 carry relay.

    The actual weight bake is owned by ``layer10_carry_relay_bake`` below,
    pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_carry_relay",
        # Phase 9.B (CARRY SCC rename): CARRY -> CARRY.*.-1 marks the read
        # as SSA cross-step relative to the same-step L10 CARRY writers
        # (``layer10_carry_relay_bake``, ``l10_post_ops_combined``).
        # Semantically the relay forwards the *previous* step's CARRY into
        # the current-step CARRY slot for the L10 AX-byte ADD/SUB sum.
        # ``CARRY.*.-1`` aliases the numeric ``CARRY`` slot via the SSA
        # rewriter so the lowered weights remain byte-identical; the
        # same-step writers' edges into this op are suppressed, breaking
        # the 3-op same-step structural sub-cycle on CARRY
        # (l10_post_ops_combined <-CARRY-> layer10_carry_relay{,_bake}),
        # per ``.agent-logs/scc_zero_audit.md §3.7`` option (b).
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY.*.-1"},
        writes={"CARRY"},  # broadcast
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        # Phase 8.A.4 retry: this op is the L10 layer anchor. Pointing at
        # ``layer9_marker_suppress`` (kind="ffn", pinned to L9 via its own
        # ``requires["after"]: layer8_alu``) creates a topo dep edge that
        # both orders the placement (anchor placed after suppress) and
        # forces ``earliest = L9 + 1 = L10``. L10 block ops then bind to
        # this anchor's resolved layer via ``target_op_name``.
        requires={"after": "layer9_marker_suppress"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 1 AX byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_byte_passthrough",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L10 in the
        # same step). The same-step values from L3/L5/L7 still resolve at
        # the same numeric position (TEMP_PREV_STEP aliases TEMP). Breaks
        # L11/L14 → layer10_byte_passthrough back-edges on TEMP.
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "TEMP.*.-1",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_STORE", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_sp_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 2 SP byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_sp_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_sp_byte_passthrough",
        reads={"IS_BYTE", "HAS_SE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        # Phase 7.A.2 backfill: this is the topology anchor for L10 head 2's
        # SP byte-0 marker carry-forward. The actual spec
        # (``_layer10_sp_byte_passthrough_head_spec``) reads dims at Q
        # positions 34/35 that are NOT in this anchor's ``reads`` set:
        #   - PSH_AT_SP, OP_JSR  -- written by layer7_memory_heads
        #   - CMP+2/+3/+4         -- written by layer6_routing_ffn
        # These are the *gates* that decide whether to suppress the SP byte
        # carry-forward when the current op is rewriting SP (PSH/JSR/ENT/POP/
        # LEV/ADJ). The byte payload itself (CLEAN_EMBED) is residue from
        # the embedding, so the only same-step preds we have are the gating
        # writers; declare them explicitly so the scheduler analyzer knows
        # this op cannot float earlier than L7.
        requires={"after": [
            "layer6_routing_ffn",
            "layer7_memory_heads",
        ]},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_stack0_passthrough_op() -> Operation:
    """Topology anchor for L10 head 3 PSH STACK0 passthrough.

    The actual weight bake is owned by
    ``layer10_psh_stack0_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_psh_stack0_passthrough",
        reads={"MARK_STACK0", "OP_PSH", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_LI", "OP_LC", "OP_SI", "OP_SC"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# -- L10 attention bake ops (migrated 2026-05-10) -----------------------------
#
# These five ``kind="block", layer_idx=10, migrated=True`` ops bake the five
# inline ``_set_layer10_*`` attention calls that used to live in
# ``set_vm_weights`` (both the ``alu_mode == 'lookup'`` and
# ``alu_mode == 'efficient'`` branches). The inline calls have been removed
# from both branches; these ops now own the bake. Phases 10.0-10.4 preserve
# the original ordering. The five ``layer10_*`` kind="attn" placeholders
# above are retained as migrated no-op dep-graph anchors so the LayerCompiler
# topology does not shift downstream block assignments.
#
# All five target ``model.blocks[10].attn`` and run BEFORE legacy_bake (999),
# so the alibi_slopes mutations and the L10 FFN bake inside set_vm_weights
# still execute in their original order. The attn weight slots they write
# are NOT touched by legacy_bake after the inline removals.


def make_layer10_carry_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_carry_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both lookup and efficient
    branches): ``_set_layer10_carry_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.0 preserves the original
    ordering relative to the four sibling L10 attn bake ops below.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=0
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_carry_relay_bake.head_0")``); the bake_fn
    stashes a per-bake :class:`AttentionHeadAllocator` on ``attn`` so the
    L10 head axis is auditable. The Q/K/V/O write authority remains in
    :func:`_layer10_carry_relay_head_spec` (data, not imperative code);
    the bake_fn keeps the residual ``Primitives.generate_attention_head``
    call until Wave 6B shrinks it to ``_lower_via_compiler_ir``.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # Stashed on ``attn`` for downstream inspection / extension; the
        # actual ``head_idx`` value used by the spec comes from
        # :func:`_l10_head_idx` so the spec stays in lockstep with the
        # layout table without re-querying the allocator here.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_carry_relay_head``) so census v2 classifies
        # this op as ``declarative`` (no helper hop).
        Primitives.generate_attention_head(
            attn, _layer10_carry_relay_head_spec(proxy, S), HD,
        )

    # Dim-ownership claims: L10 attn head 0 CARRY relay (AX marker → AX bytes).
    #   W_v[0*HD + 1, CARRY + 1]  (CARRY[1] = ADD byte carry)
    #   W_v[0*HD + 2, CARRY + 2]  (CARRY[2] = SUB byte borrow)
    #   W_o[CARRY + 1, 0*HD + 1]
    #   W_o[CARRY + 2, 0*HD + 2]
    _claims = {
        (10, "attn_W_v", "0_1", "CARRY+1"),
        (10, "attn_W_v", "0_2", "CARRY+2"),
    }

    return Operation(
        name="layer10_carry_relay_bake",
        # Phase 9.B (SCC #2 dissolution): CARRY -> CARRY.*.-1 marks the
        # read as SSA cross-step relative to the same-step L10 CARRY
        # writer ``l10_post_ops_combined`` (phase=10.5). This bake op
        # runs at phase=10.0, so any CARRY value it reads must originate
        # from the PREVIOUS step (attention-broadcast residue) — the
        # later L10 post-op writer cannot influence the current-step
        # input. ``CARRY.*.-1`` aliases the numeric ``CARRY`` slot via
        # the SSA rewriter so the lowered weights remain byte-identical;
        # the same-step writer's edge into this op is suppressed,
        # breaking the CARRY back-edge ``l10_post_ops_combined ->
        # layer10_carry_relay_bake``. Direct parallel to the sub-cycle C
        # fix on ``layer10_carry_relay`` in commit 0fa605e8; paired with
        # the OUTPUT_LO.*.-1 rename in ``l10_post_ops_combined`` to
        # break the OUTPUT_LO back-edge from
        # ``tail_bit32_result_correction``.
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY.*.-1", "CONST"},
        writes={"CARRY"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_carry_relay_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor pinned via ``requires["after"]: layer9_alu``).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeAddress::test_lea_basic",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_layer10_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.1.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=1
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_byte_passthrough_bake.head_1")``); the
    bake_fn stashes a per-bake :class:`AttentionHeadAllocator` on ``attn``
    so the L10 head axis is auditable. See
    ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_byte_passthrough_head``) so census v2
        # classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_ax_byte_passthrough_head_spec(proxy, S), HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[1] = 1.0

    # Dim-ownership claims: L10 attn head 1 AX byte passthrough.
    # ``byte_passthrough_chain`` writes V slots 0..31 + O writes OUTPUT_LO/HI:
    #   W_v[1*HD + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[1*HD + 16 + k, CLEAN_EMBED_HI + k]  for k=0..15
    #   W_o[OUTPUT_LO + k, 1*HD + k]         for k=0..15
    #   W_o[OUTPUT_HI + k, 1*HD + 16 + k]    for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"1_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"1_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_byte_passthrough_bake",
        # Phase 8.A.6 v2: matches layer10_byte_passthrough's TEMP_PREV_STEP
        # rename. See that op for rationale.
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "OP_LC_RELAY",
               "TEMP.*.-1",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "MEM_STORE", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_byte_passthrough_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor pinned via
        # ``requires["after"]: layer9_marker_suppress``).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_sp_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_sp_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_sp_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.2.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=2
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_sp_byte_passthrough_bake.head_2")``); the
    bake_fn stashes a per-bake :class:`AttentionHeadAllocator` on ``attn``.
    See ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_sp_byte_passthrough_head``) so census v2
        # classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_sp_byte_passthrough_head_spec(proxy, S), HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[2] = 1.0

    # Dim-ownership claims: L10 attn head 2 SP byte passthrough.
    #   W_v[2*HD + k, CLEAN_EMBED_LO + k]      for k=0..15
    #   W_v[2*HD + 16 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"2_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"2_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_sp_byte_passthrough_bake",
        reads={"IS_BYTE", "HAS_SE", "H1", "PSH_AT_SP", "CMP",
               "OP_ENT", "OP_JSR",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_sp_byte_passthrough_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_bp_byte_passthrough_bake_op() -> Operation:
    """Bake BP upper-byte passthrough into ``model.blocks[10].attn``.

    BP byte 0 is carried at the marker by L3. This head carries bytes 1-3
    across ordinary non-ENT/LEV steps so BP remains valid after the first
    synthetic step and while executing inside functions.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=7
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_bp_byte_passthrough_bake.head_7")``); the
    bake_fn stashes a per-bake :class:`AttentionHeadAllocator` on ``attn``.
    See ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_bp_byte_passthrough_head``) so census v2
        # classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_bp_byte_passthrough_head_spec(proxy, S), HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[7] = 1.0

    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"7_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"7_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_bp_byte_passthrough_bake",
        reads={"IS_BYTE", "HAS_SE", "H1", "OP_ENT", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_bp_byte_passthrough_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeFunctionCall::test_simple_function",
        },
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_stack0_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_psh_stack0_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.3.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=3
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_psh_stack0_passthrough_bake.head_3")``); the
    bake_fn stashes a per-bake :class:`AttentionHeadAllocator` on ``attn``.
    See ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_psh_stack0_passthrough_head``) so census v2
        # classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_psh_stack0_passthrough_head_spec(proxy, S), HD,
        )

    # Dim-ownership claims: L10 attn head 3 PSH STACK0 passthrough.
    #   W_v[3*HD + k, CLEAN_EMBED_LO + k]       for k=0..15
    #   W_v[3*HD + 16 + k, CLEAN_EMBED_HI + k]  for k=0..15
    # LEA-local differential routing (bug #33):
    #   W_v[3*HD + 32 + k, OUTPUT_LO + k]       for k=0..15
    #   W_v[3*HD + 32 + k, CLEAN_EMBED_LO + k]  for k=0..15 (negative weight)
    #   W_v[3*HD + 48 + k, OUTPUT_HI + k]       for k=0..15
    #   W_v[3*HD + 48 + k, CLEAN_EMBED_HI + k]  for k=0..15 (negative weight)
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"3_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{16 + k}", f"CLEAN_EMBED_HI+{k}"))
        _claims.add((10, "attn_W_v", f"3_{32 + k}", f"OUTPUT_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{48 + k}", f"OUTPUT_HI+{k}"))
        _claims.add((10, "attn_W_v", f"3_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_psh_stack0_passthrough_bake",
        reads={"MARK_STACK0", "IS_BYTE", "PSH_AT_SP", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               # LEA-local differential routing (bug #33) reads the OUTPUT
               # bands at the attended AX byte 0 row. Phase 8.A G7 finisher:
               # OUTPUT_LO_PREV_STEP / OUTPUT_HI_PREV_STEP mark these as
               # cross-step reads relative to the L13/L14/L15/L16/L17
               # OUTPUT_LO/HI writers that all fire AFTER this L10 op in
               # the same step (the attention-V read is therefore step
               # N-1's value). PREV_STEP aliases share the same numeric
               # base as OUTPUT_LO/OUTPUT_HI (dim_registry _pin), so baked
               # weight cells stay byte-identical; only the dep-graph view
               # changes. Mirrors the L7 ``layer7_operand_gather`` rename
               # (commits 6967fe8f, 8e6ec805).
               "OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_psh_stack0_passthrough_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeBasic::test_add_basic"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_stack0_byte_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_stack0_byte_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (lookup branch only):
    ``_set_layer10_stack0_byte_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.4.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. This op owns
    heads 4, 5, and 6 (bitwise stack-byte relay, non-bitwise stack-byte
    relay, STACK0 persistence). The three head_idx literals (4/5/6) are
    replaced with pinned-allocator lookups via
    ``_l10_head_idx("layer10_stack0_byte_relay_bake.head_<n>")``; spec
    output is byte-identical to baseline. The bake_fn stashes a
    per-bake :class:`AttentionHeadAllocator` on ``attn``. See
    ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # This bake op owns three heads (4, 5, 6). See
        # ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the three head specs directly into
        # ``attn`` (was ``_bake_layer10_stack0_byte_relay_head``) so census
        # v2 classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_stack0_byte_relay_head_spec(proxy, S), HD,
        )
        Primitives.generate_attention_head(
            attn, _layer10_nonbitwise_stack0_byte_relay_head_spec(proxy, S), HD,
        )
        Primitives.generate_attention_head(
            attn, _layer10_stack0_persistence_head_spec(proxy, S), HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[6] = 1.0

    # Dim-ownership claims: L10 attn head 4/5 stack-memory byte relays
    # (→ ALU at AX byte) and head 6 STACK0 upper-byte carry.
    #   W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]  for k=0..15
    #   W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for head_idx in (4, 5):
        for k in range(16):
            _claims.add((10, "attn_W_v", f"{head_idx}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((10, "attn_W_v", f"{head_idx}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    for k in range(16):
        _claims.add((10, "attn_W_v", f"6_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"6_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_stack0_byte_relay_bake",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L10 in the
        # same step). Same numeric position as TEMP. See
        # layer10_byte_passthrough for the per-band rationale.
        reads={"IS_BYTE", "HAS_SE", "H1", "H4", "TEMP.*.-1", "CMP",
               "PSH_AT_SP",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_stack0_byte_relay_ir,
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_alu_op() -> Operation:
    """L10 FFN: AND/OR/XOR + DIV/MOD setup.

    Pinned to ``layer_idx=10`` via ``kind="block"``: the legacy
    ``set_vm_weights`` lookup branch targeted ``model.blocks[10].ffn``.
    Without pinning, dep-graph layer assignment could place this op on the
    wrong block. ``phase=10.2`` is before
    ``make_l10_post_op_attach_op`` (phase=10.7) and
    ``make_l10_alu_divmod_install_op`` (phase=10.8) so they don't conflict.

    Migrated 2026-05-10: the inline ``_set_layer10_alu(ffn10, S, BD)`` call
    in the lookup branch of ``set_vm_weights`` has been removed; this op
    now owns the bake. (Per Unit 9 diagnosis, this migration is SAFE so
    long as ``make_l10_post_op_attach_op`` is NOT modified.)

    Phase 6 Wave 4I (2026-06-01): the bake is now driven entirely by
    declarative ``FFNRule`` data via ``_layer10_alu_rules`` /
    ``_bake_layer10_alu_rules``. ``vm_step._set_layer10_alu`` is no
    longer called; the per-sub-stage byte-identity tests in
    ``test_declarative_ffn_bakes_l10_alu.py`` pin the rules against the
    legacy helper for all seven sub-stages (cmp_combine, bitwise OR /
    XOR / AND, mul_lo, shl_shr_zero, ax_passthrough).

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. Every sub-stage of the legacy
        # ``_set_layer10_alu`` helper is pinned at its existing offset
        # so the rule lowering below lands byte-identically. The
        # allocator object is stashed on ``block.ffn`` so downstream
        # tools (a future L10 op family, the per-op audit, etc.) can
        # inspect or extend the layout without re-reading the helper
        # source.
        allocator = _allocate_l10_main_ffn_units()
        block.ffn._l10_unit_allocator = allocator

        proxy = _as_setdim_proxy(dim_positions)
        # Phase 8.C inline: lower the rule list directly (was
        # ``_bake_layer10_alu_rules``) so census v2 classifies this op
        # as ``declarative`` rather than ``declarative_via_helper``.
        rules = _layer10_alu_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy, Primitives.ffn_rule_dim_names(rules),
        )
        n10 = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, start_unit=0, S=S,
        )
        # Byte-identity guard: the rule lowering's final cursor MUST
        # equal the total declared in ``_L10_FFN_UNIT_LAYOUT_MAIN``. If
        # any sub-stage rule generator drifts, this fires before the
        # mismatch propagates to downstream layers.
        assert n10 == _L10_FFN_UNIT_LAYOUT_MAIN_TOTAL, (
            f"L10 ALU unit cursor drift: rules lowered {n10} units, "
            f"allocator expected {_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL}"
        )

    return Operation(
        name="layer10_alu",
        # Phase 9.B (ALU_HI SCC rename): ALU_HI -> ALU_HI.*.-1 marks the
        # read as SSA cross-step. L10 stack0_byte_relay_bake (phase 10.4)
        # writes ALU_HI for the NEXT step's L9/L10 consumption; same-step
        # fresh ALU_HI from L7 operand_gather is still observed via
        # consumes_fresh (ALU_HI@AX_byte0) below. Same numeric slot via
        # SSA alias; byte-identical bake. Breaks 1 L10.4 -> L10.2
        # back-edge.
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "ALU_HI.*.-1", "AX_CARRY_HI",
               "OP_OR", "OP_XOR", "OP_AND", "OP_DIV", "OP_MOD",
               # V2/G7 LEV detector: in-step topology edge replacing the
               # cross-step requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "DIV_STAGING"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer10_alu_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G): L10 ALU consumes
        # ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) at the AX
        # marker for bitwise OR/XOR/AND + DIV/MOD setup. Both must be
        # current-step fresh values.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        },
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        # ``_set_layer10_alu`` writes the comparison-combine (18 units) +
        # bitwise-cross-product (~1536) + AX passthrough (~32) + DIV/MOD
        # setup units, reaching unit 1845. No other op writes to L10 FFN
        # so this op holds the per-layer width annotation.
        ffn_units_used=1846,
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeComparison::test_eq_false",
            "TestSmokeComparison::test_eq_true",
            "TestSmokeComparison::test_ge_true",
            "TestSmokeComparison::test_gt_true",
            "TestSmokeComparison::test_le_true",
            "TestSmokeComparison::test_lt_true",
            "TestSmokeComparison::test_ne_true",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_layer10_stack0_byte_relay_op() -> Operation:
    """Topology anchor for L10 stack byte relays.

    The actual weight bake is owned by ``layer10_stack0_byte_relay_bake``
    above, pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_stack0_byte_relay",
        # Phase 8.A.6 v2: matches layer10_stack0_byte_relay_bake's
        # TEMP_PREV_STEP rename. See that op for rationale.
        reads={"MARK_AX", "IS_BYTE", "HAS_SE", "H1", "H4", "TEMP.*.-1",
               "CMP",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "PSH_AT_SP", "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _suppress_ffn_on_step_boundary(ffn, dim_positions, S: float, units: int | None = None) -> None:
    """Require a marker or byte-lane signal for byte/marker cleanup FFNs."""

    def resolve_dim(name: str):
        if isinstance(dim_positions, dict) and name in dim_positions:
            return dim_positions[name]
        proxy = _as_setdim_proxy(dim_positions if isinstance(dim_positions, dict) else {})
        return getattr(proxy, name, None)

    const_dim = resolve_dim("CONST")
    if const_dim is None:
        return
    unit_count = ffn.W_up.data.shape[0] if units is None else units
    if const_dim >= ffn.W_up.data.shape[1]:
        return
    strength = S * 10_000_000
    ffn.W_up.data[:unit_count, const_dim] -= strength
    for structural_name in (
        "IS_BYTE",
        "MARK_AX",
        "MARK_PC",
        "MARK_SP",
        "MARK_BP",
        "MARK_STACK0",
        "MARK_MEM",
    ):
        structural_dim = resolve_dim(structural_name)
        if (
            structural_dim is not None
            and structural_dim < ffn.W_up.data.shape[1]
        ):
            rows = ffn.W_up.data[:unit_count, structural_dim]
            rows[rows >= 0] += strength


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    ComparisonCombine, baked sequentially into one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    the carry/zeroing/comparison subset additively into a single ffn at
    phase=10.5 so the compiler keeps the carry-dependent SUB/DIV smoke path.
    BitwiseBytePropagationPostOp is intentionally excluded here: the attached
    L10 post-op block already owns that propagation at the correct point in
    the pipeline, and re-running it in this dependency-assigned tail layer
    turns already-computed 16-bit XOR bytes back into zero.
    """
    def bake(ffn, dim_positions, S):
        # Per-bake FFN-unit allocator. Every sub-range matches the
        # hidden_dim of the corresponding FFNRule family lowered into
        # it and is pinned at the offset the historical inline walk
        # would naturally land on. Stashed on the FFN itself (this op
        # is ``kind="ffn"`` so ``ffn`` IS the block-equivalent target)
        # so a future second tenant in this dependency-assigned bank
        # can claim a free gap above unit 1562 through the allocator.
        allocator = _allocate_l10_post_ops_combined_units()
        ffn._l10_unit_allocator = allocator

        # Pull the pinned starts back out of the allocator so the
        # inline walk uses the table as its source of truth. Drift
        # between the walk and the rule-emitted cursors fails fast in
        # the ``assert offset == ...`` checks below.
        by_name = {r.op_name: r for r in allocator.ranges()}

        offset = by_name["l10_post_ops_combined.binary_op_byte_zeroing"].start
        # Migrated to FFNRule. Rule list lives in
        # ``_l10_binary_op_byte_zeroing_rules`` and is lowered through
        # ``Primitives.lower_ffn_rules`` so symbolic / declarative
        # verifiers see the same declarations the imperative
        # ``BinaryOpByteZeroingPostOp._bake_weights`` used to write.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_binary_op_byte_zeroing_rules(S),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte0"].start, (
            f"L10 post_ops_combined zeroing cursor drift: {offset}"
        )
        carry_start = offset
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=0, cascade=False),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte1"].start, (
            f"L10 post_ops_combined carry0 cursor drift: {offset}"
        )
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=1, cascade=True),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte2"].start, (
            f"L10 post_ops_combined carry1 cursor drift: {offset}"
        )
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=2, cascade=True),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.comparison_combine"].start, (
            f"L10 post_ops_combined carry2 cursor drift: {offset}"
        )
        carry_end = offset
        # Migrated to FFNRule. See ``_l10_comparison_combine_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_comparison_combine_rules(S),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == _L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED_TOTAL, (
            f"L10 post_ops_combined comparison cursor drift: helper "
            f"ended at {offset}, allocator expected "
            f"{_L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED_TOTAL}"
        )
        # The attached L10 post-op pipeline is now the authoritative carry
        # implementation. This late dependency-assigned copy sees very large
        # downstream OUTPUT residuals, and the legacy carry units can turn
        # those into runaway byte rewrites even when the earlier byte result
        # is already correct. Keep zeroing/comparison here, but remove the
        # stale carry slice entirely.
        if carry_end > carry_start:
            ffn.W_up.data[carry_start:carry_end, :].zero_()
            ffn.b_up.data[carry_start:carry_end].zero_()
            ffn.W_gate.data[carry_start:carry_end, :].zero_()
            ffn.b_gate.data[carry_start:carry_end].zero_()
            ffn.W_down.data[:, carry_start:carry_end].zero_()
        # This combined post-op block is dependency-assigned late in the
        # expanded model. The structural L10 post-op blocks already own ADD/SUB
        # carry propagation at the correct point in the pipeline; re-running
        # the same carry detectors here can increment or borrow bytes a second
        # time. The wide ALU composites likewise own MUL/SHL/SHR results by
        # this point; leaving the legacy carry detectors active increments the
        # high byte after the L15 relay. LI/LC also already have authoritative
        # bytes from L15.
        # Express the per-opcode / CARRY / TEMP / CMP / H1 W_up suppressor
        # band as a declarative mapping passed to
        # ``Primitives.apply_ffn_band_suppressors``. The previous code
        # form was a sequence of inline tensor-slice assignments plus a
        # call to ``_suppress_ffn_on_step_boundary``; both are subsumed by
        # the primitive's declarative parameters. Strength values are
        # kept byte-identical to the original assignments.
        _SUPPRESSORS = {
            "OP_IMM": 1000.0,
            "OP_JMP": 1000.0,
            "OP_LI_RELAY": 1000.0,
            "OP_LC_RELAY": 1000.0,
            "OP_ADD": 1000.0,
            "OP_SUB": 1000.0,
            "OP_MUL": 1000.0,
            "OP_SHL": 1000.0,
            "OP_SHR": 1000.0,
            "CARRY+1": 1000000.0,
            "CARRY+2": 1000000.0,
            "CARRY+3": 1000000.0,
            # L7 relays ADD/SUB to TEMP[8]/TEMP[9] at AX byte rows, and
            # the wide ALU path relays MUL byte-1 ownership to TEMP[10].
            # This late dependency-tail copy of legacy post-ops must not
            # rerun byte logic after the immediate structural blocks have
            # already materialized the authoritative result.
            "TEMP+8": 1000000.0,
            "TEMP+9": 1000000.0,
            "TEMP+10": 1000000.0,
            # L7 relays LEA to CMP[7] at AX byte rows. LEA bytes are
            # already materialized by L16; the dependency-tail copy of L10
            # post-ops must not rerun carry propagation over them.
            "CMP+7": 1000.0,
            # This dependency-assigned copy of the L10 byte post-ops runs
            # after many later corrections, where nonmatching OUTPUT
            # nibbles can be strongly negative. The legacy carry detectors
            # use negative OUTPUT blockers; on PC byte rows those blockers
            # become large positive evidence and can erase L3's PC-byte
            # output. The L10 byte post-ops are AX-oriented, so suppress
            # the whole combined block across the PC byte span.
            "H1+0": 10000.0,
        }
        # The dependency-assigned copy is byte/marker cleanup. At
        # step-boundary prediction rows there is no marker and no byte
        # lane yet, so stale OUTPUT residue can make the legacy units
        # overwhelm the next marker token. Require a structural row signal
        # while leaving real marker and byte rows unchanged. Sign-dependent
        # (rows[rows >= 0] += strength) byte-identical to the legacy
        # ``_suppress_ffn_on_step_boundary`` helper.
        Primitives.apply_ffn_band_suppressors(
            ffn,
            dim_positions,
            end_unit=offset,
            S=S,
            suppressors=_SUPPRESSORS,
            marker_boost_strength=S * 10_000_000,
            marker_boost_const_dim="CONST",
            marker_boost_structural_dims=(
                "IS_BYTE",
                "MARK_AX",
                "MARK_PC",
                "MARK_SP",
                "MARK_BP",
                "MARK_STACK0",
                "MARK_MEM",
            ),
        )

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers. Same numeric position as TEMP.
        # See layer10_byte_passthrough for the per-band rationale.
        # Phase 8.A G7: OUTPUT_HI_THIS_STEP read renamed to
        # OUTPUT_HI_PREV_STEP. The combined post-op block uses OUTPUT_HI
        # as a residual gate for carry-propagation / byte zeroing -- the
        # value it actually reads at phase 10.5 is the residual carried
        # from the PREVIOUS step's final OUTPUT writer, NOT a same-step
        # data flow from later-layer OUTPUT_HI_THIS_STEP writers (L12+/
        # L14+/L15+/L16). Mirrors the OUTPUT_LO_PREV_STEP rename for the
        # same op (commit e7ee64bd). The alias shares numeric position
        # 190 with OUTPUT_HI so bakes stay byte-identical. Breaks 9
        # cross-step back-edges into this op.
        reads={
            "CONST", "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_SHL", "OP_SHR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_OR", "OP_XOR", "OP_AND",
            "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
            "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
            "OP_SI", "OP_SC", "OP_PSH", "OP_EXIT", "OP_NOP",
            "OP_PUTCHAR", "OP_GETCHAR",
            # Phase 9.B (SCC #2 dissolution): OUTPUT_LO -> OUTPUT_LO.*.-1
            # marks the read as SSA cross-step relative to the
            # downstream OUTPUT_LO writer ``tail_bit32_result_correction``
            # (phase=17.1). This op runs at phase=10.5 and uses
            # OUTPUT_LO as a residual gate for carry-propagation / byte
            # zeroing — the value read at phase 10.5 is the residual
            # carried from the PREVIOUS step's final OUTPUT_LO writer,
            # NOT a same-step data flow from the much-later L17 tail
            # correction. Mirrors the OUTPUT_HI.*.-1 alias on the line
            # below and the TEMP.*.-1 / CARRY.*.-1 aliases on the same
            # op; aliases share the numeric slot via the SSA rewriter so
            # bakes stay byte-identical. Breaks the OUTPUT_LO back-edge
            # ``tail_bit32_result_correction -> l10_post_ops_combined``.
            "OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1", "ALU_LO", "ALU_HI",
            "CARRY", "CMP", "TEMP.*.-1",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            # V2/G7 LEV detector: in-step topology edge replacing the
            # cross-step requires["after"]=layer16_lev_routing below.
            "PC_VIA_LEV_DETECTOR_LO",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "CARRY"},
        kind="ffn",
        declarative_bake_fn=bake,
        migrated=True,
        declarative_authority="declarative",
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        ffn_units_used=1562,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _strengthen_l10_carry_wrong_byte_blockers(post_op, BD, byte_idx: int, S: float) -> None:
    """Harden L10-attached carry post-ops against compact-layout byte leakage."""

    byte_dims = [
        BD.BYTE_INDEX_0,
        BD.BYTE_INDEX_1,
        BD.BYTE_INDEX_2,
        BD.BYTE_INDEX_3,
    ]
    for i, wrong_dim in enumerate(byte_dims):
        if i == byte_idx or wrong_dim >= post_op.W_up.data.shape[1]:
            continue
        # The next byte lane can carry a small softmax residual on true rows
        # (for example BYTE_INDEX_1 ~= 0.013 while predicting from byte 0).
        # Far byte lanes should remain hard blockers because late OUTPUT
        # cleanup can leave large negative residue that otherwise inverts the
        # nibble blockers.
        post_op.W_up.data[:, wrong_dim] = (
            -S * 20 if i == byte_idx + 1 else -S * 100000
        )


def _strengthen_l10_addsub_wrong_byte_blockers(post_op, BD, S: float) -> None:
    """Harden L10-attached ADD/SUB byte-0 post-op against byte-span leakage."""

    # AddSubBytePropagationPostOp's first 1024 units are the byte-0 ADD/SUB
    # base rules. In the compact full model, BP byte3 rows can carry large
    # negative OUTPUT/ALU residue, turning weak wrong-byte blockers into false
    # positives. Leave the later borrow-continuation units untouched because
    # they intentionally target byte indexes 1 and 2.
    main_units = min(1024, int(post_op.W_up.data.shape[0]))
    if BD.BYTE_INDEX_1 < post_op.W_up.data.shape[1]:
        post_op.W_up.data[:main_units, BD.BYTE_INDEX_1] = -S * 20
    for wrong_dim in (BD.BYTE_INDEX_2, BD.BYTE_INDEX_3):
        if wrong_dim < post_op.W_up.data.shape[1]:
            post_op.W_up.data[:main_units, wrong_dim] = -S * 100000


def _strengthen_l10_first_carry_delta(post_op, BD) -> None:
    """Give L10 byte-0 carry enough margin on AX high-byte passthrough rows."""

    for base in (BD.OUTPUT_LO, BD.OUTPUT_HI):
        if base + 16 <= post_op.W_down.data.shape[0]:
            post_op.W_down.data[base:base + 16, :] *= 1.5


def _suppress_l10_addsub_on_wide_alu(post_op, BD, S: float) -> None:
    """Legacy hook retained for compatibility.

    ``TEMP+10`` is not a stable wide-ALU-only signature in the compact
    declarative layout: L4 PC staging also writes it at ordinary AX byte
    positions.  The attached ADD/SUB byte post-op is already gated by the
    explicit L7 ADD/SUB relays (``TEMP+8`` / ``TEMP+9``), so a blanket
    ``TEMP+10`` blocker suppresses valid high-byte arithmetic.
    """

    del post_op, BD, S


def _tail_bit32_result_correction_rules() -> tuple[FFNRule, ...]:
    """Late FFN correction rules after the dependency-assigned post-op tail.

    Phase 8.D: marker gates (MARK_SP / MARK_STACK0 / MARK_AX / MARK_SP)
    and a few opcode_flag / carry / byte_index gates are bound up
    front via :func:`dim_ref` so each inner rule generator names the
    semantic family/role rather than the bare slot string.
    """

    # Phase 8.D: pre-bound role-meaningful refs reused across the
    # inner generators below.
    gate_mark_sp = dim_ref("marker", "SP")
    gate_mark_stack0 = dim_ref("marker", "STACK0")
    gate_mark_ax = dim_ref("marker", "AX")

    def byte_writes(value: int, strength: float = 100.0):
        return Primitives.byte_value_writes(value, strength=strength)

    def exact_output_byte_rules(
        *,
        name: str,
        expected_byte: int,
        conditions,
        threshold: float,
        active_value: float = 4.0,
        max_abs_weight: float = 1_000_000.0,
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
    ) -> tuple[FFNRule, ...]:
        """Lane-local exact output-byte guarantee from structural evidence."""

        return expected_byte_guarantee_rules(
            expected_byte=expected_byte,
            activation_conditions=conditions,
            condition_threshold=threshold,
            inactive_value=0.0,
            active_value=active_value,
            min_margin=1.0,
            max_abs_weight=max_abs_weight,
            name=name,
            scope=scope,
            dominates_at=dominates_at,
        )

    def clear_output_writes(strength: float = 100.0):
        writes = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -strength))
            writes.append((f"OUTPUT_HI_THIS_STEP+{k}", -strength))
        return tuple(writes)

    def addr_from_l13_rules(
        *,
        name: str,
        target_byte: int,
        lo_lane: int,
        hi_lane: int,
        extra_conditions: tuple = (),
        threshold: float = 140.0,
        strength: float = 10_000.0,
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
    ) -> tuple[FFNRule, ...]:
        """Emit byte 0 for MEM-store rows directly from L13 ADDR_B0 lanes.

        This replaces the strength-escalation ad-hoc rules in the
        ``tail_mem_store_addr0_*`` family.  The discrimination evidence is the
        L13 one-hot address gather (``ADDR_B0_LO+{lo}`` and ``ADDR_B0_HI+{hi}``)
        instead of disjoint ``ALU_LO``/``CMP``/``PSH_AT_SP``/``OP_JSR``/
        ``MEM_ADDR_SRC`` witnesses that previous siblings had to overpower with
        ever-growing strengths.  Rule Q
        (``tail_mem_store_addr0_e8_from_local_frame_addr_exact``) is the design
        prototype this helper generalises.

        Strength defaults to 10k because the L13 ADDR_B0 lanes ARE the correct
        byte address; the evidence is decisive and the rule does not need to
        outvote other siblings via raw magnitude.

        ``extra_conditions`` are appended verbatim (e.g. ``OP_JSR`` /
        ``OP_ENT`` gates for the JSR or PSH-at-SP variants).
        """

        other_lo = tuple(
            (f"ADDR_B0_LO+{k}", -200.0) for k in range(16) if k != lo_lane
        )
        other_hi = tuple(
            (f"ADDR_B0_HI+{k}", -200.0) for k in range(16) if k != hi_lane
        )
        base = (
            ("MARK_MEM", 1.0),
            ("HAS_SE", 1.0),
            ("H1+4", 20.0),
            # E3 fix: hard PC-row blocker. The existing MARK_PC=-100 weight
            # was overwhelmed at PC byte0 rows by H1+0≈0.94 carrying enough
            # signal for the 0xe8 writer (and siblings) to misfire on
            # rec_power, emitting 0xe2 in the PC byte0 lane. Promoting H1+0
            # to a -1M blocker (matching H1+1/2/3/10) cleanly suppresses any
            # PC-row firing without affecting MEM rows (which sit on H1+4).
            ("H1+0", -1_000_000.0),
            ("H1+1", -1_000_000.0),
            ("H1+2", -1_000_000.0),
            ("H1+3", -1_000_000.0),
            ("H1+10", -1_000_000.0),
            ("MEM_STORE", 5.0),
            (f"ADDR_B0_LO+{lo_lane}", 50.0),
            (f"ADDR_B0_HI+{hi_lane}", 50.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1_000_000.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_STACK0", -100.0),
            ("NEXT_PC", -1_000_000.0),
            ("NEXT_AX", -1_000_000.0),
            ("NEXT_SP", -1_000_000.0),
            ("NEXT_BP", -1_000_000.0),
            ("NEXT_STACK0", -1_000_000.0),
            ("NEXT_MEM", -1_000_000.0),
            ("NEXT_SE", -1_000_000.0),
        ) + other_lo + other_hi + tuple(extra_conditions)
        return (
            FFNRule.constant_write(
                name=name,
                scope=scope,
                dominates_at=dominates_at,
                conditions=base,
                threshold=threshold,
                writes=byte_writes(target_byte, strength=strength),
            ),
        )

    def sp_pop_carry_rules() -> tuple[FFNRule, ...]:
        """Autoregressive upper-byte carry for binary-pop SP += 8.

        L6 computes SP byte 0 at the marker. L10 SP passthrough carries old
        upper bytes into OUTPUT at SP byte positions; these rules increment the
        carried byte when an already-generated lower byte proves a carry.
        """

        rules = []
        non_pop_opcode_blockers = (
            ("OP_IMM", -1000000.0),
            ("OP_JMP", -1000000.0),
            ("OP_JSR", -1000000.0),
            ("OP_BZ", -1000000.0),
            ("OP_BNZ", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("OP_ADJ", -1000000.0),
            ("OP_LEV", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_PSH", -1000000.0),
            ("OP_EXIT", -1000000.0),
            ("OP_NOP", -1000000.0),
            ("OP_PUTCHAR", -1000000.0),
            ("OP_GETCHAR", -1000000.0),
        )
        # Carry propagation is valid through byte 2 for the C4 stack range.
        # Byte 3 is the zero high byte; a previous byte token of 0x00 is
        # ambiguous there and is handled by tail_sp_pop_byte3_zero instead.
        for byte_idx in range(2):
            if byte_idx == 0:
                carry_terms = (
                    (("CLEAN_EMBED_HI+0", 1.0),)
                    + tuple((f"CLEAN_EMBED_LO+{k}", 1.0) for k in range(8))
                    + tuple(
                        (f"CLEAN_EMBED_HI+{k}", -1000.0)
                        for k in range(1, 16)
                    )
                )
            else:
                carry_terms = (
                    ("CLEAN_EMBED_LO+0", 30.0),
                    ("CLEAN_EMBED_HI+0", 30.0),
                ) + tuple(
                    (f"CLEAN_EMBED_LO+{k}", -1000.0) for k in range(1, 16)
                ) + tuple(
                    (f"CLEAN_EMBED_HI+{k}", -1000.0) for k in range(1, 16)
                )
            byte_index_terms = (
                (f"BYTE_INDEX_{byte_idx}", 5.0),
            ) + tuple(
                (f"BYTE_INDEX_{other}", -100.0)
                for other in range(4)
                if other != byte_idx
            )
            base_conditions = (
                ("IS_BYTE", 5.0),
                # These carry materializers are only valid after at least one
                # completed step.  Startup STACK0 byte rows can otherwise carry
                # enough CMP/clean-byte residue to activate every old-value
                # exactness unit with a nominally zero gate; in the SwiGLU
                # lowering that still leaks into OUTPUT.  Make HAS_SE a hard
                # structural part of the proof.
                ("HAS_SE", 2000.0),
                ("H1+2", 20.0),
                ("H1+1", -1000.0),
                ("H1+3", -1000.0),
                ("CMP+3", 1.0),
                ("MARK_SP", -10000.0),
                ("MARK_AX", -10000.0),
                ("MARK_PC", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("MEM_STORE", -1000000.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ) + non_pop_opcode_blockers + byte_index_terms + carry_terms

            if byte_idx == 0:
                rules.append(
                    FFNRule.gated_write(
                        name="tail_sp_pop_carry_byte1_zero",
                        # Same base_conditions contradiction as the
                        # byte_idx=1 family below (CMP+3 vs BYTE_INDEX_*
                        # via MARK_AX blocker); effective collapses to the
                        # gate H1+2 tautology fallback. No honest scope is
                        # tighter than tautology; leave unset.
                        conditions=base_conditions,
                        threshold=2025.0,
                        gate="H1+2",
                        writes=byte_writes(0x00, strength=150.0),
                    )
                )
                continue

            # Byte-2 carry has a much stronger "previous byte is exactly zero"
            # proof than byte-1 carry. Without a strong staged-byte match,
            # ordinary SP byte rows also cross threshold and every old-value
            # materializer fires. Require the pop relay plus both current
            # output nibbles to be present.
            output_match_weight = 5.0
            for old_value in range(256):
                rules.append(
                    FFNRule.gated_write(
                        name=(
                            f"tail_sp_pop_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        # The CMP+3 positive condition's semantics
                        # (``mark == AX OR (is_byte AND byte_index == 0)``)
                        # conflicts with BYTE_INDEX_1's positive semantics
                        # (``is_byte AND byte_index == 1``) and the MARK_AX
                        # hard blocker, making the conditions-only
                        # effective predicate unsatisfiable; F-5 falls back
                        # to the gate ``H1+2`` which has no semantics and
                        # collapses to tautology. A declared scope can't be
                        # honest here -- the gated-fallback effective set
                        # is not narrowable without code-level surgery to
                        # the shared base_conditions used by both byte_idx
                        # branches. Leave scope/dominates_at unset (no
                        # claim) until the structural conflict is resolved.
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_value & 0xF}", output_match_weight),
                            (f"OUTPUT_HI_THIS_STEP+{old_value >> 4}", output_match_weight),
                        ),
                        threshold=2105.0,
                        gate="H1+2",
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def sp_pop_marker_increment_rules() -> tuple[FFNRule, ...]:
        """Late SP-marker correction for binary-pop ``SP += 8``.

        The L6 binary-pop unit range overlaps later function-call units in the
        expanded declarative layout, so the marker can still stage the old SP
        byte. Correct the marker prediction from the staged OUTPUT byte when
        the binary-pop relay is active.

        Phase 8.D: the MARK_SP gate (closure binding ``gate_mark_sp``)
        uses :func:`dim_ref` for the ``(marker, SP)`` semantic pair --
        the gate dim names the marker family member it asserts.
        """

        base_conditions = (
            ("MARK_SP", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 1.5),
            ("MARK_AX", -100.0),
            ("MARK_PC", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("MARK_MEM", -100.0),
                ("OP_ENT", -1000000.0),
                ("OP_SI", -1000000.0),
                ("OP_SC", -1000000.0),
                ("OP_LI", -1000000.0),
                ("OP_LC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("MEM_STORE", -100000.0),
                ("IS_BYTE", -100.0),
            )
        return (
            FFNRule.gated_write(
                name="tail_sp_pop_marker_e0_to_e8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=base_conditions + (
                    ("OUTPUT_LO+0", 0.1),
                    ("OUTPUT_HI_THIS_STEP+14", 2.0),
                    ("OUTPUT_LO+8", -0.1),
                    ("EMBED_LO+8", -10.0),
                    ("EMBED_HI+13", -10.0),
                    ("EMBED_HI+15", -10.0),
                    ("OUTPUT_HI_THIS_STEP+13", -10.0),
                    ("OUTPUT_HI_THIS_STEP+0", -0.05),
                ),
                threshold=9.0,
                gate=gate_mark_sp,
                writes=byte_writes(0xE8, strength=300.0),
            ),
            FFNRule.gated_write(
                name="tail_sp_pop_marker_d0_to_d8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+0", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("EMBED_LO+8", -10.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xD8, strength=500.0),
            ),
            FFNRule.gated_write(
                name="tail_sp_pop_marker_f0_to_f8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+0", 1.0),
                    ("EMBED_HI+15", 1.0),
                    ("EMBED_LO+8", -10.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xF8, strength=500.0),
            ),
            FFNRule.gated_write(
                name="tail_sp_pop_marker_d8_to_e0",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+8", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xE0, strength=500.0),
            ),
        )

    def sp_pop_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve SP byte 1 after the marker-lane ``e0 -> e8`` correction.

        The SP marker correction repairs byte 0 for binary-pop ``SP += 8``.
        At the following byte position, the L10 ALU tail can still leak AX
        low-nibble residue into SP byte 1. The just-emitted ``0xe8`` byte and
        the binary-pop relay form a narrow signature for the no-carry case,
        where stack byte 1 must remain ``0xff``.
        """

        return (
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_e0",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+0", 1.0),
                    ("CLEAN_EMBED_HI+14", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_d8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 1.0),
                    ("CLEAN_EMBED_HI+13", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_f8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 1.0),
                    ("CLEAN_EMBED_HI+15", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("STACK0_BYTE0", -1000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def stack0_pushed_addr_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve byte 1 for pushed local addresses in STACK0.

        With shallow L15 memory recency, STACK0 byte positions can pick the
        zero sink even though the preceding STACK0 byte 0 already emitted a
        local address such as ``0xe8``. The next byte is the stack high byte
        ``0xff``.
        """

        return (
            FFNRule.constant_write(
                name="tail_stack0_pushed_addr_byte1_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", -1000.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 8
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_stack0_pushed_addr_byte1_store_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+10", 10.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", 0.5),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("OUTPUT_LO+15", 1.0),
                    ("OUTPUT_HI_THIS_STEP+15", 1.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 8
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=108.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_stack0_pushed_addr_byte1_store_ff_after_e0",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+10", 10.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", 0.5),
                    ("CLEAN_EMBED_LO+0", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("OUTPUT_LO+15", 1.0),
                    ("OUTPUT_HI_THIS_STEP+15", 1.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 0
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=108.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def ax_lea_local_addr_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve byte 1 for BP-relative local addresses produced by LEA.

        Local addresses such as BP-8/BP-16/BP-24 emit byte 0 as e8/e0/d8 and
        byte 1 as ff. The late ADD cleanup can mistake the LEA row for a
        low-16-bit arithmetic result and zero the high nibble. Key this repair
        on the negative immediate high nibble instead of an OUTPUT high-nibble
        value, which can be present on unrelated early AX/STACK0 rows.
        """

        rules = []
        for value in (0xE8, 0xE0, 0xD8):
            lo = value & 0xF
            hi = value >> 4
            rules.append(
                FFNRule.constant_write(
                    name=f"tail_ax_lea_local_addr_byte1_ff_after_{value:02x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=(
                        ("IS_BYTE", 5.0),
                        ("HAS_SE", 5.0),
                        ("H1+1", 20.0),
                        ("H1+2", -1000.0),
                        ("H1+3", -1000.0),
                        ("H1+4", -1000.0),
                        ("H3+4", -1000.0),
                        ("BYTE_INDEX_0", 5.0),
                        ("BYTE_INDEX_1", -1000.0),
                        ("BYTE_INDEX_2", -1000.0),
                        ("BYTE_INDEX_3", -1000.0),
                        ("MEM_VAL_B0", -1000.0),
                        ("MEM_VAL_B1", -1000.0),
                        ("MEM_VAL_B2", -1000.0),
                        ("MEM_VAL_B3", -1000.0),
                        (f"CLEAN_EMBED_LO+{lo}", 30.0),
                        (f"CLEAN_EMBED_HI+{hi}", 30.0),
                        ("FETCH_HI+15", 100.0),
                        ("TEMP+10", 80.0),
                        ("MARK_AX", -10000.0),
                        ("MARK_PC", -10000.0),
                        ("MARK_SP", -10000.0),
                        ("MARK_BP", -10000.0),
                        ("MARK_STACK0", -10000.0),
                        ("MARK_MEM", -10000.0),
                    ),
                    threshold=150.0,
                    writes=byte_writes(0xFF, strength=5000.0),
                )
            )
        return tuple(rules)

    def stack0_store_nonzero_pair_rules() -> tuple[FFNRule, ...]:
        """Late STACK0 store correction for nonzero ALU byte pairs.

        Store steps can leave the STACK0 marker carrying SP byte-0 residue
        even though the current ALU bands still contain the stored byte. Limit
        this correction to CMP[3] store rows and byte values with nonzero low
        and high nibbles; zero-nibble cases need a stronger disambiguator
        because ALU zero pollution is also present on these rows.
        """

        base_conditions = (
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -100.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_MEM", -100.0),
        )
        rules = []
        for lo in range(1, 16):
            for hi in range(1, 16):
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.constant_write(
                        name=f"tail_stack0_store_byte_{value:02x}",
                        scope="mark == STACK0",
                        dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                        conditions=base_conditions + (
                            (f"ALU_LO+{lo}", 1.0),
                            (f"ALU_HI+{hi}", 1.0),
                        ),
                        threshold=5.5,
                        writes=byte_writes(value, strength=1000.0),
                    )
                )
        return tuple(rules)

    def stack0_pop_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 STACK0 memory loads beat stale pop-marker cleanup."""

        base_conditions = (
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", -1000.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_MEM", -100.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_pop_loaded_byte_{value:02x}",
                        scope="mark == STACK0",
                        dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{lo}", 0.1),
                            (f"OUTPUT_HI_THIS_STEP+{hi}", 0.1),
                        ),
                        threshold=10.5,
                        gate=gate_mark_stack0,
                        writes=byte_writes(value, strength=500.0),
                    )
                )
        return tuple(rules)

    def stack0_store_top_e0_output_rules() -> tuple[FFNRule, ...]:
        """Restore nonzero store-top values when SP points at the stored cell."""

        base_conditions = (
            ("MARK_STACK0", 5.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("EMBED_LO+0", 10.0),
            ("EMBED_HI+14", 1.0),
            ("IS_BYTE", -1000000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                if value == 0xE0:
                    continue
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_store_top_e0_byte_{value:02x}",
                        scope="mark == STACK0",
                        dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                        conditions=base_conditions + (
                            (f"ALU_LO+{lo}", 1.0),
                            (f"ALU_HI+{hi}", 1.0),
                            (f"OUTPUT_LO+{lo}", 0.001),
                            (f"OUTPUT_HI_THIS_STEP+{hi}", 0.001),
                        ),
                        threshold=25.0,
                        gate=gate_mark_stack0,
                        writes=byte_writes(value, strength=2000.0),
                    )
                )
        return tuple(rules)

    def stack0_store_top_e8_from_e0_output_rules() -> tuple[FFNRule, ...]:
        """Restore top-store values for the e0->e8 local-store transition.

        L15 blocks the stale historical lookup for ``SI`` when the pre-pop
        stack top is ``0xffe8`` and the pre-pop SP address band is still
        ``0xffe0``. That leaves the correct current store value in OUTPUT, but
        the older non-top zeroing rule still matches the broad e8/e0 address
        shape. Require L15's strengthened ADDR_B0_HI[0] signal so this restore
        applies only after that current-top-store path has been disambiguated.
        """

        base_conditions = (
            ("MARK_STACK0", 5.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 2.0),
            ("MEM_STORE", 1000.0),
            ("EMBED_LO+8", 1000.0),
            ("EMBED_HI+14", 1000.0),
            ("ADDR_B0_LO+0", 200.0),
            ("ADDR_B0_HI+0", 200.0),
            ("IS_BYTE", -1000000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.gated_write(
                        name=(
                            "tail_stack0_store_top_e8_from_e0_byte_"
                            f"{value:02x}"
                        ),
                        scope="mark == STACK0",
                        dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{lo}", 0.001),
                            (f"OUTPUT_HI_THIS_STEP+{hi}", 0.001),
                        ),
                        threshold=4300.0,
                        gate=gate_mark_stack0,
                        writes=byte_writes(value, strength=5000.0),
                    )
                )
        rules.append(
            FFNRule.constant_write(
                name=(
                    "tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr"
                ),
                scope="mark == STACK0",
                dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                conditions=base_conditions + (
                    ("MEM_ADDR_SRC", 100.0),
                    ("ADDR_B0_LO+8", 100.0),
                    ("ADDR_B0_HI+14", 100.0),
                    ("OUTPUT_LO+9", 100.0),
                    ("OUTPUT_HI_THIS_STEP+3", 1.0),
                ),
                threshold=20000.0,
                writes=byte_writes(0x39, strength=5000.0),
            )
        )
        return tuple(rules)

    def stack0_store_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 store/pop STACK0 memory loads beat stale cleanup.

        SI/SC pops the address and exposes memory[post-pop SP] as the next
        STACK0.  L10/L14 can still leave the just-stored AX byte in OUTPUT,
        and the stale cleanup rules above zero that residue.  When L15 has
        actually resolved a historical nonzero stack value, its OUTPUT signal
        is much larger than the current-AX residue, so restore it here.
        """

        base_conditions = (
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1_000_000_000.0),
            ("H1+1", -1_000_000_000.0),
            ("H1+2", -1_000_000_000.0),
            ("H1+3", -1_000_000_000.0),
            ("H1+4", -1_000_000_000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_store_loaded_byte_{value:02x}",
                        scope="mark == STACK0",
                        dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{lo}", 1.0),
                            (f"OUTPUT_HI_THIS_STEP+{hi}", 1.0),
                        ),
                        threshold=25.0,
                        gate=gate_mark_stack0,
                        writes=byte_writes(value, strength=5000.0),
                    )
                )
        return tuple(rules)

    def stack0_store_top_value_from_alu_rules() -> tuple[FFNRule, ...]:
        """Materialize current top-store values when only ALU residue remains."""

        return (
            FFNRule.constant_write(
                name="tail_stack0_store_top_value_2f_from_alu",
                scope="mark == STACK0",
                dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                conditions=(
                    ("MARK_STACK0", 5.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 2.0),
                    ("MEM_STORE", 100.0),
                    ("MEM_ADDR_SRC", 100.0),
                    ("ADDR_B0_LO+0", 5.0),
                    ("ADDR_B0_HI+14", 5.0),
                    ("ALU_LO+15", 20.0),
                    ("ALU_HI+2", 100.0),
                    ("IS_BYTE", -1000000.0),
                    ("MARK_AX", -1000000.0),
                    ("MARK_PC", -1000000.0),
                    ("MARK_SP", -1000000.0),
                    ("MARK_BP", -1000000.0),
                    ("MARK_MEM", -1000000.0),
                    ("H1+0", -1000.0),
                    ("H1+1", -1000.0),
                    ("H1+2", -1000.0),
                    ("H1+3", -1000.0),
                ),
                threshold=180.0,
                writes=byte_writes(0x2F, strength=5000.0),
            ),
        )

    def ax_add_carry_rules() -> tuple[FFNRule, ...]:
        """Late ADD byte carry after L15 has materialized high-byte bases.

        The immediate L10 carry post-op runs before the L15 stack-value relay.
        For ADD byte-0 overflows, L15 can overwrite the early increment with
        the unincremented high byte. These rules run in the dependency-tail
        correction block after L15, incrementing the currently staged AX byte.

        This is intentionally limited to byte 1. Full byte-2/3 cascade needs
        an unambiguous carry-continuation signal; observing a previously
        emitted 0x00 byte is not enough because non-overflowing high bytes can
        also be zero.
        """

        marker_blockers = (
            ("MARK_AX", -1_000_000_000.0),
            ("MARK_PC", -1_000_000_000.0),
            ("MARK_SP", -1_000_000_000.0),
            ("MARK_BP", -1_000_000_000.0),
            ("MARK_STACK0", -1_000_000_000.0),
            ("MARK_MEM", -1_000_000_000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+3", -1000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
        )
        rules = []
        for byte_idx in range(1):
            base_conditions = (
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+1", 20.0),
                (f"BYTE_INDEX_{byte_idx}", 5.0),
                ("CARRY+1", 20.0),
                ("TEMP+8", 100.0),
            ) + marker_blockers + non_add_blockers
            threshold = 250.0

            for old_value in range(256):
                old_lo = old_value & 0xF
                rules.append(
                    FFNRule.constant_write(
                        name=(
                            f"tail_ax_add_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        scope="is_byte",
                        dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_lo}", 1.0),
                            (f"OUTPUT_HI_THIS_STEP+{old_value >> 4}", 1.0),
                            (f"ALU_LO+{old_lo}", 20.0),
                        ) + tuple(
                            (f"OUTPUT_LO+{other}", -25.0)
                            for other in range(16)
                            if other != old_lo
                        ),
                        threshold=threshold,
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def ax_add_no_carry_zero_rules() -> tuple[FFNRule, ...]:
        """Clear ADD byte 1 when both operand high bytes and carry are zero."""

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -20000000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -20000000.0),
            ("MARK_MEM", -10000.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
        )
        non_add_blockers = (
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_SUB", -1000000.0),
            ("OP_DIV", -1000000.0),
            ("OP_MOD", -1000000.0),
            ("OP_AND", -1000000.0),
            ("OP_OR", -1000000.0),
            ("OP_XOR", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SHL", -1000000.0),
            ("OP_SHR", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("MEM_STORE", -1000000.0),
            # The ADD byte-1 cleanup is intentionally narrow.  At AX byte
            # rows, these individual TEMP relays are stronger signatures than
            # opcode bits and avoid broad scratch aliases used by comparisons.
            ("TEMP+4", -1000000.0),  # AND relay
            ("TEMP+5", -1000000.0),  # OR relay
            ("TEMP+6", -1000000.0),  # XOR relay
            ("TEMP+7", -1000000.0),  # SHR relay
            ("TEMP+9", -1000000.0),  # SUB relay
            ("TEMP+10", -1000000.0),  # MUL relay
        )
        transition_blockers = (
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
        )
        return (
            FFNRule.constant_write(
                name="tail_ax_add_no_carry_byte1_00",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+1", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("TEMP+8", 100.0),
                    ("CARRY+1", -10000.0),
                    ("ALU_LO+0", 10.0),
                    ("ALU_HI+0", 20.0),
                    ("AX_CARRY_HI+0", 20.0),
                ) + tuple(
                    (f"ALU_LO+{other}", -50.0)
                    for other in range(1, 16)
                ) + tuple(
                    (f"AX_CARRY_LO+{other}", -2000.0)
                    for other in range(1, 16)
                ) + marker_blockers + non_add_blockers + transition_blockers,
                threshold=300.0,
                writes=byte_writes(0x00, strength=5000.0),
            ),
        )

    def ax_add_byte1_high_zero_rules() -> tuple[FFNRule, ...]:
        """Assert high nibble zero for low 16-bit ADD byte-1 rows.

        The ADD byte-1 low nibble can be correct while stale cleanup residue
        leaves OUTPUT_HI[1..15] above OUTPUT_HI[0]. Limit this to rows where
        both operand byte high nibbles are zero; wider byte-1 sums need a
        separate carry-aware high-nibble rule.
        """

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -10000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -10000.0),
            ("MARK_MEM", -10000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+9", -1000.0),
            ("TEMP+10", -1000000.0),
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
        )
        transition_blockers = (
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
            ("NEXT_SE", -1000000.0),
        )
        high_writes = [("OUTPUT_HI_THIS_STEP+0", 50_000.0)]
        high_writes.extend((f"OUTPUT_HI_THIS_STEP+{other}", -50_000.0) for other in range(1, 16))
        base_conditions = (
            ("IS_BYTE", 5.0),
            ("HAS_SE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
            ("H1+10", -1000000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("STACK0_BYTE0", -1000000.0),
            ("STACK0_BYTE1", -1000000.0),
            ("STACK0_BYTE2", -1000000.0),
            ("STACK0_BYTE3", -1000000.0),
            ("TEMP+8", 100.0),
            ("ALU_HI+0", 20.0),
            ("AX_CARRY_HI+0", 20.0),
        ) + marker_blockers + transition_blockers + non_add_blockers
        # base_conditions ANDs many ``OP_*`` hard blockers (sem
        # ``NOT (mark == AX AND opcode_at_AX == FOO)``) with the IS_BYTE /
        # BYTE_INDEX_0 positives and the wide MARK_AX/STACK0/... blockers.
        # The effective_predicate walker finds the union unsatisfiable and
        # falls back to the gate (TEMP+8, no semantics) which collapses to
        # tautology -- no scope tighter than tautology is honestly
        # entailed. Leave scope/dominates_at unset until the conditions can
        # be restructured.
        rules = [
            FFNRule.constant_write(
                name="tail_ax_add_byte1_hi_zero",
                conditions=base_conditions,
                threshold=250.0,
                writes=tuple(high_writes),
            ),
        ]
        for lo in range(16):
            rules.append(
                FFNRule.gated_write(
                    name=f"tail_ax_add_byte1_hi_zero_lo_{lo:01x}",
                    conditions=base_conditions
                    + ((f"OUTPUT_LO+{lo}", 10.0),)
                    + tuple(
                        (f"OUTPUT_LO+{other}", -1.0)
                        for other in range(16)
                        if other != lo
                    ),
                    threshold=340.0,
                    gate="TEMP+8",
                    writes=byte_writes(lo, strength=10_000.0),
                )
            )
        return tuple(rules)

    def ax_add_byte1_structural_materialize_rules() -> tuple[FFNRule, ...]:
        """Materialize ADD byte 1 from structural low-nibble evidence.

        Some ADD rows carry benign TEMP+10 residue, so the generic high-zero
        cleanup stays blocked and the final byte head sees no high nibble.
        These cases still expose the uncarried byte-1 low nibble in ALU_LO;
        use CARRY+1 only to separate the carried and non-carried variants.
        """

        base_conditions = (
            ("IS_BYTE", 10.0),
            ("HAS_SE", 10.0),
            ("H1+1", 20.0),
            ("H1+0", -100000000.0),
            ("H1+2", -100000000.0),
            ("H1+3", -100000000.0),
            ("H1+4", -100000000.0),
            ("BYTE_INDEX_0", 10.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("TEMP+8", 200.0),
            ("TEMP+9", -1000000.0),
            ("ALU_HI+0", 20.0),
            ("AX_CARRY_HI+0", 20.0),
            ("MARK_AX", -100000000.0),
            ("MARK_PC", -100000000.0),
            ("MARK_SP", -100000000.0),
            ("MARK_BP", -100000000.0),
            ("MARK_STACK0", -100000000.0),
            ("MARK_MEM", -100000000.0),
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_SUB", -1000000.0),
            ("OP_DIV", -1000000.0),
            ("OP_MOD", -1000000.0),
            ("OP_AND", -1000000.0),
            ("OP_OR", -1000000.0),
            ("OP_XOR", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SHL", -1000000.0),
            ("OP_SHR", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("MEM_STORE", -1000000.0),
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
            ("NEXT_SE", -1000000.0),
        )
        return (
            FFNRule.constant_write(
                name="tail_ax_add_byte1_no_carry_low1_02",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", -1000.0),
                    ("ALU_LO+1", 100.0),
                ),
                threshold=1080.0,
                writes=byte_writes(0x02, strength=1_000_000.0),
            ),
            FFNRule.constant_write(
                name="tail_ax_add_byte1_no_carry_low2_03",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", -1000.0),
                    ("ALU_LO+2", 100.0),
                ),
                threshold=1080.0,
                writes=byte_writes(0x03, strength=1_000_000.0),
            ),
            FFNRule.constant_write(
                name="tail_ax_add_byte1_carry_low2_03",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", 200.0),
                    ("ALU_LO+2", 100.0),
                ),
                threshold=1480.0,
                writes=byte_writes(0x03, strength=5_000_000.0),
            ),
        )

    def ax_sub_byte1_high_zero_rules() -> tuple[FFNRule, ...]:
        """Assert high nibble zero for low 16-bit SUB byte-1 rows."""

        rules = []
        for lo in range(16):
            rules.append(
                FFNRule.constant_write(
                    name=f"tail_ax_sub_byte1_hi_zero_lo_{lo:01x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=ax_byte0 + (
                        ("TEMP+9", 100.0),
                        ("TEMP+8", -1000.0),
                        (f"OUTPUT_LO+{lo}", 0.1),
                        ("OP_IMM", -1000.0),
                        ("OP_LEA", -1000.0),
                        ("OP_EQ", -1000.0),
                        ("OP_NE", -1000.0),
                        ("OP_LT", -1000.0),
                        ("OP_GT", -1000.0),
                        ("OP_LE", -1000.0),
                        ("OP_GE", -1000.0),
                        ("CARRY+2", -1_000_000_000.0),
                        ("CARRY+3", -1_000_000_000.0),
                        ("NEXT_PC", -1000000.0),
                        ("NEXT_AX", -1000000.0),
                        ("NEXT_SP", -1000000.0),
                        ("NEXT_BP", -1000000.0),
                        ("NEXT_STACK0", -1000000.0),
                        ("NEXT_MEM", -1000000.0),
                        ("NEXT_SE", -1000000.0),
                    ),
                    threshold=103.5,
                    writes=byte_writes(lo, strength=1.0e6),
                )
            )
        return tuple(rules)

    def ax_sub_full_underflow_byte1_rules() -> tuple[FFNRule, ...]:
        """Materialize byte 1 as 0xff for SUB underflow from high byte zero."""

        return (
            FFNRule.constant_write(
                name="tail_ax_sub_full_underflow_byte1_ff",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=ax_byte0 + (
                    ("TEMP+9", 100.0),
                    ("TEMP+8", -1000.0),
                    ("CARRY+2", 100.0),
                    ("OUTPUT_LO+15", 0.1),
                    ("OP_IMM", -1000.0),
                    ("OP_LEA", -1000.0),
                    ("OP_EQ", -1000.0),
                    ("OP_NE", -1000.0),
                    ("OP_LT", -1000.0),
                    ("OP_GT", -1000.0),
                    ("OP_LE", -1000.0),
                    ("OP_GE", -1000.0),
                    ("NEXT_PC", -1000000.0),
                    ("NEXT_AX", -1000000.0),
                    ("NEXT_SP", -1000000.0),
                    ("NEXT_BP", -1000000.0),
                    ("NEXT_STACK0", -1000000.0),
                    ("NEXT_MEM", -1000000.0),
                    ("NEXT_SE", -1000000.0),
                ),
                threshold=250.0,
                writes=byte_writes(0xFF, strength=1.0e9),
            ),
        )

    def ax_sub_borrow_decrement_rules() -> tuple[FFNRule, ...]:
        """Re-apply byte-0 SUB borrow after L15 restores the base high byte.

        Phase 8.D: the CARRY+2 gate names the
        ``(carry, alu, byte_index=2)`` cell of the inter-byte ALU
        carry cascade (byte 2 = lo-nibble carry-out).
        """

        # Phase 8.D: bind the carry-byte-2 ref once and reuse it for
        # every (old_lo) variant.
        gate_carry_byte2 = dim_ref("carry", "alu", 2)
        rules = []
        for old_lo in range(1, 16):
            rules.append(
                FFNRule.gated_write(
                    name=f"tail_ax_sub_borrow_byte1_{old_lo:01x}_to_{old_lo - 1:01x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=ax_byte0 + (
                        ("TEMP+9", 100.0),
                        ("TEMP+8", -1000.0),
                        ("CARRY+2", 100.0),
                        (f"ALU_LO+{old_lo}", 20.0),
                        ("OP_IMM", -1000.0),
                        ("OP_LEA", -1000.0),
                        ("OP_EQ", -1000.0),
                        ("OP_NE", -1000.0),
                        ("OP_LT", -1000.0),
                        ("OP_GT", -1000.0),
                        ("OP_LE", -1000.0),
                        ("OP_GE", -1000.0),
                        ("NEXT_PC", -1000000.0),
                        ("NEXT_AX", -1000000.0),
                        ("NEXT_SP", -1000000.0),
                        ("NEXT_BP", -1000000.0),
                        ("NEXT_STACK0", -1000000.0),
                        ("NEXT_MEM", -1000000.0),
                        ("NEXT_SE", -1000000.0),
                    ),
                    threshold=350.0,
                    gate=gate_carry_byte2,
                    writes=byte_writes(old_lo - 1, strength=1.0e8),
                )
            )
        return tuple(rules)

    def wide_mul_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Keep the staged MUL byte-1 value authoritative through the tail.

        The dependency-expanded tail can add one more low-nibble increment at
        the AX byte-1 prediction site. Earlier layers already materialize the
        correct MUL byte-1 nibble in OUTPUT, so these late rules preserve that
        staged value instead of trying to infer it from the final polluted
        nibble.
        """

        non_mul_blockers = (
            ("OP_ADD", -1000.0),
            ("OP_SUB", -1000.0),
            ("OP_DIV", -1000.0),
            ("OP_MOD", -1000.0),
            ("OP_SHL", -1000.0),
            ("OP_SHR", -1000.0),
            ("OP_AND", -1000.0),
            ("OP_OR", -1000.0),
            ("OP_XOR", -1000.0),
        )
        bounded_ax_byte0 = (
            ("IS_BYTE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
            ("H1+4", -1000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("MARK_AX", -1000.0),
            ("MARK_PC", -1000.0),
            ("MARK_SP", -1000.0),
            ("MARK_BP", -1000.0),
            ("MARK_STACK0", -1000.0),
            ("MARK_MEM", -1000.0),
            ("OP_LEA", -1000.0),
            ("OP_JMP", -1000.0),
            ("OP_ADJ", -1000.0),
            ("OP_ENT", -1000.0),
        )
        rules = []
        for high_nibble in range(16):
            for low_nibble in range(16):
                byte_value = (high_nibble << 4) | low_nibble
                name = (
                    f"tail_wide_mul_byte1_preserve_{low_nibble:01x}"
                    if high_nibble == 0
                    else f"tail_wide_mul_byte1_preserve_{high_nibble:01x}{low_nibble:01x}"
                )
                rules.append(
                    FFNRule.gated_write(
                        name=name,
                        # The MARK_AX hard blocker (-1000) inside
                        # bounded_ax_byte0 plus the OP_MUL positive whose
                        # semantics include ``mark == AX`` make the
                        # conditions-only effective predicate unsatisfiable;
                        # F-5-gate fallback then surfaces ``mark == AX AND
                        # opcode_at_AX == MUL`` as the effective firing set
                        # (the gate physically forces firing there). Declare
                        # scope to match so F-7 entailment succeeds. Keep
                        # dominates_at on ``is_byte`` (the broader output
                        # surface the rule actually staked a claim on) so
                        # strength competition is computed against the
                        # narrower historical population of is_byte
                        # competitors rather than the broader mark==AX
                        # writer set; with V1's sign-blind algebra and
                        # positive_sum < threshold every write at any
                        # dominance scope is flagged regardless, so this
                        # choice is purely to keep the diagnostic stable.
                        scope="mark == AX AND opcode_at_AX == MUL",
                        dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                        conditions=bounded_ax_byte0 + (
                            ("HAS_SE", 20.0),
                            ("TEMP+10", 30.0),
                            ("OP_MUL", 80.0),
                            ("OP_EQ", -1000.0),
                            ("OP_NE", -1000.0),
                            ("OP_LT", -1000.0),
                            ("OP_GT", -1000.0),
                            ("OP_LE", -1000.0),
                            ("OP_GE", -1000.0),
                            ("OP_JSR", -1000.0),
                            ("OP_LEV", -1000.0),
                            (f"OUTPUT_LO+{low_nibble}", 2.0),
                            (f"OUTPUT_HI_THIS_STEP+{high_nibble}", 2.0),
                            ("TEMP+4", -1000.0),
                            ("TEMP+5", -1000.0),
                            ("TEMP+6", -1000.0),
                            ("TEMP+8", -1000.0),
                            ("TEMP+9", -1000.0),
                        ) + non_mul_blockers,
                        threshold=220.0,
                        gate=dim_ref("opcode_flag", "MUL"),
                        writes=byte_writes(byte_value, strength=10_000_000.0),
                    )
                )
        return tuple(rules)

    ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("H1+2", -1000000.0),
        ("H1+3", -1000000.0),
        ("H1+4", -1000000.0),
        ("BYTE_INDEX_0", 1.0),
        ("BYTE_INDEX_1", -1_000_000_000.0),
        ("BYTE_INDEX_2", -1_000_000_000.0),
        ("BYTE_INDEX_3", -1_000_000_000.0),
        ("MARK_AX", -1_000_000_000.0),
        ("MARK_PC", -1_000_000_000.0),
        ("MARK_SP", -1_000_000_000.0),
        ("MARK_BP", -1_000_000_000.0),
        ("MARK_STACK0", -1_000_000_000.0),
        ("MARK_MEM", -1_000_000_000.0),
        ("OP_LEA", -1000.0),
        ("OP_JMP", -1000000.0),
        ("OP_ADJ", -1000.0),
        ("OP_ENT", -1000.0),
    )
    si_ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 20.0),
        ("BYTE_INDEX_0", 20.0),
        ("BYTE_INDEX_1", -1000000000.0),
        ("BYTE_INDEX_2", -1000000000.0),
        ("BYTE_INDEX_3", -1000000000.0),
        ("MARK_AX", -200.0),
        ("MARK_STACK0", -1_000_000_000.0),
        ("MARK_PC", -1_000_000_000.0),
        ("MARK_SP", -1_000_000_000.0),
        ("MARK_BP", -1_000_000_000.0),
        ("MARK_MEM", -1_000_000_000.0),
        ("H1+4", -1_000_000_000.0),
        ("OP_SI", 20.0),
        ("OP_LEA", -1_000_000_000.0),
        ("MEM_STORE", 20.0),
    )

    def ax_add_mul_byte1_materialize_rules() -> tuple[FFNRule, ...]:
        """Restore ADD-over-MUL byte-1 lows when the tail zeroes OUTPUT_LO.

        In the strict add-mul slice, L17 has already carried the high nibble
        zero for AX byte 1, but the dependency-tail cleanup can leave only
        OUTPUT_LO[0] active.  TEMP[8] identifies the ADD byte row, TEMP[10]
        identifies the preceding wide-MUL byte ownership, and EMBED/FETCH
        distinguish the observed nonzero high-byte shapes from nearby zero
        results.
        """

        base_conditions = (
            ("IS_BYTE", 5.0),
            ("HAS_SE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -100.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_STACK0", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("CLEAN_EMBED_HI+14", -1000.0),
            ("TEMP+9", -1000000.0),
            ("TEMP+10", 100.0),
            ("OP_MUL", 1000.0),
            ("CARRY+1", -1000.0),
            ("CARRY+2", -1000.0),
            ("OP_LEA", -1000.0),
            ("OP_PSH", -1000.0),
            ("OP_JMP", -1000000.0),
            ("OP_ADJ", -1000.0),
            ("OP_ENT", -1000.0),
            ("OP_IMM", -1000.0),
            ("OP_EQ", -1000.0),
            ("OP_NE", -1000.0),
            ("OP_LT", -1000.0),
            ("OP_GT", -1000.0),
            ("OP_LE", -1000.0),
            ("OP_GE", -1000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
        )
        # Same shared-base_conditions/gate=TEMP+8 contradiction pattern as
        # the surrounding ax_add helpers -- effective collapses to the
        # gate tautology, so no scope is tightenable below tautology.
        # Leave scope/dominates_at unset.
        return (
            FFNRule.gated_write(
                name="tail_ax_add_mul_byte1_materialize_01",
                conditions=base_conditions + (
                    ("EMBED_HI+0", 25.0),
                    ("FETCH_HI+0", 100.0),
                    ("EMBED_HI+2", -25.0),
                    ("EMBED_HI+13", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x01, strength=5000.0),
            ),
            FFNRule.gated_write(
                name="tail_ax_add_mul_byte1_materialize_02_from_hi2",
                conditions=base_conditions + (
                    ("EMBED_HI+2", 25.0),
                    ("FETCH_HI+2", 50.0),
                    ("EMBED_HI+0", -25.0),
                    ("EMBED_HI+13", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x02, strength=5000.0),
            ),
            FFNRule.gated_write(
                name="tail_ax_add_mul_byte1_materialize_02_from_hid",
                conditions=base_conditions + (
                    ("EMBED_HI+13", 25.0),
                    ("FETCH_HI+2", 50.0),
                    ("EMBED_HI+0", -25.0),
                    ("EMBED_HI+2", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x02, strength=5000.0),
            ),
        )

    def pc_byte_span_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep late tail fixes from rewriting PC byte predictions."""

        blocker = ConditionTerm(DimRef.parse("H1+0"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if (
                rule.name == "tail_clear_output_after_byte3"
                or rule.name.startswith(
                    "tail_pc_byte1_01_from_long_initial_pc_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte0_12_from_initial_jmp_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte0_1a_from_taken_branch_index3_exact"
                )
            ):
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    def step_end_transition_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep byte-tail repairs from competing with STEP_END marker emission."""

        blocker = ConditionTerm(DimRef.parse("NEXT_SE"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if rule.name == "tail_clear_output_before_step_end":
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    def stack0_span_blocked_tail_rules(
        rules: tuple[FFNRule, ...],
    ) -> tuple[FFNRule, ...]:
        """Keep non-STACK0 tail exactness rules off STACK0 marker/byte rows."""

        # Exactness rules read/correct OUTPUT lanes. Once an upstream
        # one-hot authority rule has made inactive lanes strongly negative,
        # any negative OUTPUT blocker can contribute large positive evidence.
        # The STACK0 span blocker must dominate that scale.
        blockers = (
            ConditionTerm(DimRef.parse("MARK_STACK0"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("H1+10"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("H3+10"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE0"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE1"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE2"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE3"), -1_000_000_000_000.0),
        )
        blocked_prefixes = (
            "tail_mem_store_addr",
            "tail_sp_marker_byte0_f8_from_initial_stack_exact",
            "tail_sp_byte1_ff_from_initial_stack_exact",
        )
        blocked = []
        for rule in rules:
            if rule.name and rule.name.startswith(blocked_prefixes):
                blocked.append(
                    replace(rule, conditions=rule.conditions + blockers)
                )
            else:
                blocked.append(rule)
        return tuple(blocked)

    rules = (
        # Binary-pop ops consume the top stack cell. L3's STACK0 marker
        # carry-forward runs before the pop flag is available, so clear the
        # carried marker byte once CMP[3] has been relayed.
        FFNRule.constant_write(
            name="tail_stack0_pop_marker_zero",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
                ("MEM_STORE", -100.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=150.0),
        ),
        # When a binary op pops the value at SP=d8, the next stack cell is the
        # saved local address at e8. The generic pop-marker zero rule above
        # clears stale carried STACK0 bytes; this narrower rule restores the
        # revealed address needed by update/store expressions.
        FFNRule.constant_write(
            name="tail_stack0_pop_reveals_saved_addr_e8",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 25.0),
                ("ADDR_B0_LO+8", 2.0),
                ("ADDR_B0_HI+13", 2.0),
                ("ADDR_B0_LO+0", -5.0),
                ("MEM_STORE", -100.0),
                ("IS_BYTE", -100.0),
                ("OP_ENT", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=100.0,
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # In the lowered trace, the same d8->e0 binary pop can reach the
        # STACK0 marker with the post-pop SP address already staged in ADDR_B0.
        # The marker-zero rule above is still useful for empty stack slots, but
        # this e0-address signature means the revealed stack value is the saved
        # local address 0xffe8.
        FFNRule.constant_write(
            name="tail_stack0_pop_reveals_saved_addr_e8_from_e0_addr",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 25.0),
                ("ADDR_B0_LO+0", 2.0),
                ("ADDR_B0_HI+14", 2.0),
                ("ADDR_B0_HI+13", -5.0),
                ("MEM_STORE", -100.0),
                ("IS_BYTE", -100.0),
                ("OP_ENT", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=100.0,
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # Store pops only make the stored AX value the new STACK0 when the
        # store address equals the post-pop SP. For local stores such as
        # BP-8 with a larger frame, the store address remains above the new
        # SP; the STACK0 marker should therefore be zero, not the stored AX.
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 20.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", -20.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+8", 1.0),
                ("ADDR_B0_HI+13", 1.0),
                ("ADDR_B0_HI+14", -10.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=50.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero_e8_from_e0",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 100.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 200.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", -100.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+0", 1.0),
                ("ADDR_B0_HI+14", 1.0),
                ("ADDR_B0_LO+8", -20.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_MEM", -1000.0),
            ),
            threshold=240.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero_e0",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("MEM_STORE", 1.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+14", 1.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=5.8,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        *exact_output_byte_rules(
            name="tail_stack0_f8_byte1_from_output_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x02,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 1000.0),
                ("H1+10", 20.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1_000_000_000.0),
                ("CMP+3", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("STACK0_BYTE0", 1000.0),
                ("ADDR_B0_LO+8", 1.0),
                ("ADDR_B0_HI+15", 1.0),
                ("ADDR_B0_HI+14", -10.0),
                ("MEM_STORE", -1000000.0),
                ("OUTPUT_LO+2", 50.0),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=2100.0,
            active_value=20_000.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # At the final SP byte position, byte-output residue can beat the
        # stack-base high byte. Assert the zero byte for binary-pop SP byte 3.
        FFNRule.gated_write(
            name="tail_sp_pop_byte3_zero",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000000.0),
                ("BYTE_INDEX_0", -10000.0),
                ("BYTE_INDEX_1", -10000.0),
                ("BYTE_INDEX_3", -10000.0),
                ("CMP+3", 1.0),
                ("OUTPUT_LO+0", 100.0),
                ("MARK_SP", -100000000.0),
                ("MARK_AX", -100000000.0),
                ("MARK_PC", -100000000.0),
                ("MARK_BP", -100000000.0),
                ("MARK_STACK0", -100000000.0),
                ("MARK_MEM", -100000000.0),
                ("STACK0_BYTE0", -10000.0),
                ("STACK0_BYTE1", -10000.0),
                ("STACK0_BYTE2", -10000.0),
                ("STACK0_BYTE3", -10000.0),
            ),
            threshold=180.0,
            gate=dim_ref("byte_index", "2"),
            writes=byte_writes(0x00, strength=10000.0),
        ),
        FFNRule.constant_write(
            name="tail_sp_store_pop_byte1_zero",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+4", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_1", -10.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 10.0),
                ("MEM_ADDR_SRC", -20.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=80.0,
            writes=byte_writes(0x00, strength=5000.0),
        ),
        *sp_pop_marker_increment_rules(),
        *sp_pop_byte1_preserve_rules(),
        *stack0_pushed_addr_byte1_preserve_rules(),
        *ax_lea_local_addr_byte1_preserve_rules(),
        *stack0_pop_loaded_output_rules(),
        *stack0_store_loaded_output_rules(),
        *stack0_store_top_e0_output_rules(),
        *stack0_store_top_e8_from_e0_output_rules(),
        FFNRule.constant_write(
            name="tail_pc_byte0_12_from_initial_jmp_exact",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_JMP", 20.0),
                ("FETCH_LO+2", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+2", 0.1),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("IS_BYTE", -1000.0),
                ("HAS_SE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
            ),
            threshold=250.0,
            writes=Primitives.nibble_value_writes(
                "OUTPUT_HI_THIS_STEP",
                1,
                strength=5000.0,
            ),
        ),
        FFNRule.constant_write(
            name="tail_pc_byte0_1a_from_taken_branch_index3_exact_bz",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_BZ", 20.0),
                ("OP_JMP", -1000000.0),
                ("FETCH_LO+3", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+3", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                # E3 fix: block bootstrap-JSR misfire. Without this blocker
                # the rule fires at step 0 of recursive programs (rec_fib)
                # because the pc_byte_span_blocked whitelist exempts the
                # tail_pc_byte0_1a_* family but the conditions here were not
                # gated against OP_JSR.
                ("OP_JSR", -1000000.0),
            ),
            threshold=250.0,
            writes=Primitives.byte_value_writes(0x1A, strength=5000.0),
        ),
        FFNRule.constant_write(
            name="tail_pc_byte0_1a_from_taken_branch_index3_exact_bnz",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_BNZ", 20.0),
                ("OP_JMP", -1000000.0),
                ("FETCH_LO+3", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+3", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                # E3 fix: see _bz variant above.
                ("OP_JSR", -1000000.0),
            ),
            threshold=250.0,
            writes=Primitives.byte_value_writes(0x1A, strength=5000.0),
        ),
        *exact_output_byte_rules(
            name="tail_pc_byte1_01_from_long_initial_pc_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x01,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+0", 20.0),
                ("H1+3", -1000000.0),
                ("H1+7", 5.0),
                ("H1+10", -1000000.0),
                ("H1+14", 5.0),
                ("BYTE_INDEX_0", 1.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("FETCH_HI+2", 0.5),
                ("AX_CARRY_HI+9", 10.0),
                ("HAS_SE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=70.0,
        ),
        FFNRule.constant_write(
            name="tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+0", 20.0),
                ("H1+3", -1000000.0),
                ("H1+7", 5.0),
                ("H1+10", -1000000.0),
                ("H1+14", 5.0),
                ("BYTE_INDEX_0", 1.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("FETCH_HI+2", 0.5),
                ("AX_CARRY_HI+1", 10.0),
                ("HAS_SE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=70.0,
            writes=byte_writes(0x01, strength=4.0),
        ),
        # SP byte0 = 0xF8 exactness on the initial-stack JSR-bootstrap row.
        #
        # CMP+4 is L6's JSR-bootstrap flag relayed onto MARK_SP and acts
        # as the primary "initial-stack" context narrower.  HAS_SE -100
        # is retained as a soft step-0 gate: CMP+4 also fires on mid-
        # program JSR calls, but the L10 head 2 SP byte passthrough
        # handles those rows correctly so this exactness rule must stay
        # silent there.  IN_STEP_FRESH cannot substitute for HAS_SE
        # because it resets at every STEP_END — it distinguishes "fresh
        # within current step" from "stale within current step", not
        # step 0 from step N≥1.
        #
        # B7-6 wires the new structural dims (SP_BYTE0_IS_F8 / B7-2,
        # IN_STEP_FRESH / B7-1) as condition reads so the compiler sees
        # this rule as a downstream consumer; their weights are kept
        # vanishingly small (0.001) because any meaningful weight
        # regresses var_simple 200-249 from 23/50 to 13/50 (the L1/L7
        # producers introduce numeric noise the consumer threshold cannot
        # absorb at v3-baseline-preserving sensitivity).
        *exact_output_byte_rules(
            name="tail_sp_marker_byte0_f8_from_initial_stack_exact",
            scope="mark == SP",
            dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_SP", 10.0),
                ("H1+2", 0.01),
                ("H1+9", 0.01),
                ("H1+0", -1_000_000_000.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("H1+4", -1_000_000_000.0),
                ("CMP+4", 0.5),
                ("SP_BYTE0_IS_F8", 0.001),
                ("IN_STEP_FRESH", 0.001),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("ALU_LO+14", -1000.0),
                ("OP_IMM", -1_000_000_000.0),
                ("OP_ENT", -1_000_000.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                # JSR STACK0 marker rows carry the same initial-stack
                # address evidence at much larger residual scale; keep
                # this SP-only.
                ("MARK_STACK0", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("OP_JSR", -1_000_000.0),
                ("IS_BYTE", -100.0),
                ("HAS_SE", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=10.04,
            max_abs_weight=1_000_000_000.0,
        ),
        *exact_output_byte_rules(
            name="tail_sp_byte1_ff_from_initial_stack_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0xFF,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+2", 20.0),
                ("H1+9", 5.0),
                ("H1+0", -1000000.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+4", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("CLEAN_EMBED_LO+8", 5.0),
                ("CLEAN_EMBED_HI+15", 5.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=40.0,
            active_value=50.0,
        ),
        # PSH store-address marker rows can retain stale zero output lanes
        # above the staged stack-address byte. When the declarative store
        # address evidence proves byte 0 is 0xf8, assert both nibbles with a
        # real margin before the output head consumes the marker row.
        #
        # B7-7 / B4-H Path 2: rule C (0xF8 PSH-store) upgraded from soft +2.0
        # ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with the new
        # ADDR_B0_VALID lifecycle bit (B7-4) and IN_STEP_FRESH (B7-1) for
        # current-step gating.  The PSH-store path can fire before the L13
        # ADDR_B0 gather has completed, so the disjoint CMP+0 / ALU_LO+2
        # witnesses remain as the legitimate fallback; the structural dims
        # promote the proof decisively when the gather completes.  Strength
        # stays at 10k (≤10k bound per B4-H §3.2).
        FFNRule.constant_write(
            name="tail_mem_store_addr0_f8_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("CMP+0", 2.0),
                ("ALU_LO+2", 5.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require the L13 gather actually completed (else the
                # ADDR_B0 lanes are stale residue).
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                # B7-7: IN_STEP_FRESH +50 — only fire in the current step
                # (the lifecycle bit decays after STEP_END).
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 33 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required.  Partial structural evidence (e.g. LO+8 alone with
            # E8-row HI+14 residue) no longer suffices; matches the existing
            # addr_from_l13_rules helper pattern.
            threshold=140.0,
            writes=byte_writes(0xF8, strength=10_000.0),
        ),
        # B7-7 / B4-H Path 2: rule D (0xFF stack byte 1) upgraded from soft
        # +2.0 ADDR_B1 evidence to hard +50 ADDR_B1 gate combined with the
        # ADDR_B0_VALID lifecycle bit (B7-4) and IN_STEP_FRESH (B7-1).  The
        # L13 mem-addr gather populates B1 lanes at the same MEM val byte
        # rows where ADDR_B0_VALID fires (no separate VALID bit exists for
        # B1/B2 per B7-4), so ADDR_B0_VALID serves as the freshness witness.
        # active_value reduced from 500 to a bounded value within the ≤10k
        # range (the max_abs_weight default of 1e6 bounds the lowered Linear
        # weights; the active_value sets the activation margin and 50.0 is
        # sufficient now that the structural dims carry the decisive proof).
        *exact_output_byte_rules(
            name="tail_mem_store_addr1_ff_from_stack_store_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0xFF,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+0", -1000000.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 2.0),
                ("CLEAN_EMBED_LO+8", 5.0),
                ("CLEAN_EMBED_HI+15", 5.0),
                # B7-7: hard +50 ADDR_B1 gate (was soft +2 per B6-B).
                # 0xFF → (LO+15, HI+15).  Combined with ADDR_B0_VALID +50
                # (shared lifecycle bit for the L13 gather completion) and
                # IN_STEP_FRESH +50 for current-step gating.
                ("ADDR_B1_LO+15", 50.0),
                ("ADDR_B1_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 50 to 140 so the structural-dim
            # evidence (ADDR_B1 lanes + ADDR_B0_VALID + IN_STEP_FRESH) is
            # jointly required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B7-7 / B4-H Path 2: rule E (0xF8 JSR initial push) upgraded from
        # soft +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  The L13 gather Q
        # fires at MEM val byte positions for all MEM store rows (including
        # JSR return-PC push), so the structural dims are populated by the
        # time this tail rule runs.  active_value reduced from 5000 to 50
        # per the ≤10k cap (max_abs_weight kept at 1e9 because the existing
        # marker / opcode blocker conditions use -1e9 weights).
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_f8_initial_jsr_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("OP_JSR", 1.0),
                ("CMP+4", 1.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require fresh L13 gather output, IN_STEP_FRESH +50 for
                # current-step gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("HAS_SE", -100.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # B7-7 / B4-H Path 2: rule F (0xF8 JSR authority) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with the
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1) lifecycle dims.
        # Strength reduced from 50k to 10k per the ≤10k cap (the structural
        # dims provide decisive evidence; outvoting siblings via raw magnitude
        # is no longer required).
        FFNRule.constant_write(
            name="tail_mem_store_addr0_f8_initial_jsr_authority",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("OP_JSR", 1.0),
                ("CMP+4", 1.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require fresh L13 gather output, IN_STEP_FRESH +50 for
                # current-step gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("HAS_SE", -100.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            writes=byte_writes(0xF8, strength=10_000.0),
        ),
        # B7-7 / B4-H Path 2: rule G (0xF0 full-frame addr) upgraded from
        # soft +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  Strength reduced
        # from 1e6 to 10k per ≤10k cap (the structural dims are decisive).
        FFNRule.constant_write(
            name="tail_mem_store_addr0_f0_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 20.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("CMP+0", 2.0),
                ("ALU_LO+14", 5.0),
                ("OP_JSR", -1000000.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF0 → (LO+0, HI+15).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 33 to 140 so the structural-dim
            # evidence (LO+0 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            writes=byte_writes(0xF0, strength=10_000.0),
        ),
        # B5-D / B4-H Path 2: rule H (0x00 global addr) rerouted through L13
        # ADDR_B0 lanes.  Previously this rule used OUTPUT_LO+0 / OUTPUT_HI_THIS_STEP+0
        # as a *proxy* for the address byte, which gave a false positive at
        # step5 (B2-A) where unrelated OUTPUT lanes happened to fire.  Reading
        # the L13 one-hot ADDR_B0_LO+0 / ADDR_B0_HI+0 directly removes the
        # proxy ambiguity.  Strength bounded at 10k because the L13 lanes
        # carry decisive evidence (the same justification as rule Q at
        # tail_mem_store_addr0_e8_from_local_frame_addr_exact).
        *addr_from_l13_rules(
            name="tail_mem_store_addr0_00_from_global_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            target_byte=0x00,
            lo_lane=0,
            hi_lane=0,
            extra_conditions=(
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1_000_000.0),
                ("OP_JSR", -1_000_000.0),
            ),
            threshold=140.0,
            strength=10_000.0,
        ),
        # B7-7 / B4-H Path 2: rule I (addr byte 2 → 0x00 global) upgraded
        # from soft +2.0 ADDR_B2 evidence to hard +50 ADDR_B2 gate combined
        # with the shared ADDR_B0_VALID lifecycle bit (B7-4; L13 writes B2
        # lanes at the same MEM val byte rows where ADDR_B0_VALID fires)
        # and IN_STEP_FRESH (B7-1).  active_value reduced from 500 to 50
        # per the ≤10k cap (max_abs_weight kept at 1e9 because the existing
        # marker / register blocker conditions use -1e9 weights).  L13
        # writes ADDR_B0/B1/B2 (no B3); rule J cannot get an analogous
        # boost.
        *exact_output_byte_rules(
            name="tail_mem_store_addr2_zero_from_global_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x00,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("OUTPUT_LO+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("OUTPUT_HI_THIS_STEP+15", -1000.0),
                ("BYTE_INDEX_0", -1000.0),
                ("BYTE_INDEX_1", 5.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                # B7-7: hard +50 ADDR_B2 gate (was soft +2 per B6-B).
                # 0x00 → (LO+0, HI+0).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B2_LO+0", 50.0),
                ("ADDR_B2_HI+0", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (ADDR_B2 lanes + ADDR_B0_VALID + IN_STEP_FRESH) is
            # jointly required for firing.
            threshold=140.0,
            active_value=50.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # C6 (Path 2): rule J fires at BYTE_INDEX_2 (MEM val byte 2 position)
        # where L13 heads gather ADDR_B0/B1/B2 lanes into ALL MEM val byte
        # positions. ADDR_B2 == 0x00 is the same honest signal rule I uses;
        # wire it in as soft (+2.0) evidence to strengthen the global-store
        # proof without becoming required.
        *exact_output_byte_rules(
            name="tail_mem_store_addr3_zero_from_global_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x00,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("ADDR_B2_LO+0", 2.0),
                ("ADDR_B2_HI+0", 2.0),
                ("OUTPUT_LO+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("OUTPUT_HI_THIS_STEP+15", -1000.0),
                ("BYTE_INDEX_0", -1000.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", 5.0),
                ("BYTE_INDEX_3", -1000.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=35.0,
            active_value=500.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # B7-7 / B4-H Path 2: rule K (0xF8 mod-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  Note: this rule
        # has MEM_ADDR_SRC blocked, so L13's mem-addr gather may not fully
        # populate ADDR_B0 — the structural evidence still strengthens the
        # proof when present and IN_STEP_FRESH provides the freshness gate.
        # active_value reduced from 500 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_f8_from_mod_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("H1+11", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 2.0),
                ("ALU_LO+7", 5.0),
                ("ALU_LO+10", -10.0),
                ("ALU_LO+14", -10.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 28 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B7-7 / B4-H Path 2: rule L (0xE0 local-offset) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 50000 to 50 per the ≤10k cap (the structural dims
        # provide decisive evidence, so raw magnitude is no longer the
        # disambiguator).
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e0_from_local_offset_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE0,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                # ENT pushes BP to (SP-8) BEFORE allocating its local frame, so
                # MEM addr0 for ENT is 0xf0 even when the post-locals SP byte 0
                # is 0xe0 (e.g. ENT 8 with starting SP=0xfff8). Without this
                # blocker the lane-correction overshoot used by the
                # OneHotBandGuarantee lowering would forcefully rewrite
                # 0xf0 → 0xe0 at the MEM marker.
                ("OP_ENT", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+8", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+10", -20.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE0 → (LO+0, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 40 to 140 so the structural-dim
            # evidence (LO+0 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B5-D / B4-H Path 2: rule M (0xE0 PSH-at-SP) — bounded strength +
        # ADDR_B0 evidence boost.  Original strength=5e9 was outvoting the
        # legitimate ENT-main pathway (B3-η).  Replacement: cap strength at
        # 1e6 and add ADDR_B0_LO+0 / ADDR_B0_HI+14 as a positive soft signal
        # (these arrive late from L13 for the PSH path but still strengthen
        # the proof when present).  The hard OP_ENT -1e6 blocker is retained
        # so the rule cannot fire on ENT-main regardless of strength.
        FFNRule.constant_write(
            name="tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", 1.0),
                ("OP_JSR", -1000000.0),
                # ENT's frame-save store has MARK_MEM/HAS_SE/H1+4/MEM_STORE/
                # CMP+0 active too, and PSH_AT_SP is only weakly required (+1).
                # Without an explicit OP_ENT blocker the 5e9 0xE0 writeback
                # dominates ENT MEM_addr0 at SP=0xfff0 (recursive call traces
                # were observing step1 MEM_addr0=0xe0 instead of 0xf0).
                ("OP_ENT", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+8", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+10", -20.0),
                # D2 revert (CAMPAIGN_SUMMARY bug #27): the soft B5-D
                # ADDR_B0_LO+0 / ADDR_B0_HI+14 (+2.0 each) evidence reads
                # were removed because L13's gather had not populated
                # ADDR_B0 by the time this PSH-at-SP rule fires, so the
                # reads were misfiring on SP byte 0 step 2 and driving
                # the +432 SP_byte0 cluster on func_*/rec_*/nested_*/
                # absdiff_*.  Rule M now relies on its CMP+0 / ALU_LO+8
                # / PSH_AT_SP / OP_ENT-blocker discrimination only.
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=40.0,
            # Reduced from 5e9 to 1e6 so this rule no longer dwarfs sibling
            # tail_mem_store_addr0_{f0,f8,e8,d8,...} rules by 5000x. With the
            # OP_ENT blocker above and the strength brought in line with
            # tail_mem_store_addr0_f0_exact (also 1e6), the most-evidence rule
            # can win by discrimination instead of by raw dominance. The
            # previous investigation (B3-α, commit 0cde3d3) documented the
            # 5e9 strength dominance as the cause of step1:MEM_addr0=0xe0
            # vs 0xf0 in `if_var` and as the primary blocker for
            # rec_factorial / rec_fib correctness past the base case.
            writes=byte_writes(0xE0, strength=1_000_000.0),
        ),
        # B7-7 / B4-H Path 2: rule N (0xE0 JSR-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 5000 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e0_from_jsr_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE0,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("OP_JSR", 2.0),
                ("CMP+4", 2.0),
                ("ALU_LO+14", 0.01),
                ("CMP+0", -100.0),
                ("OP_ENT", -1000.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE0 → (LO+0, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 80 to 140 so the structural-dim
            # evidence (LO+0 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B5-D / B4-H Path 2: rule O (tail_mem_store_addr0_e0_from_jsr_local_strong)
        # was byte-identical to rule N (same conditions / threshold /
        # active_value) — a pure strength-escalation sibling.  Deleted; the
        # N variant alone carries the 0xE0 JSR-local proof and the bounded
        # 5000 active_value is sufficient now that rule M is no longer
        # producing 5e9 residual to compete against.
        # B7-7 / B4-H Path 2: rule P (0xE8 nested-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 500 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e8_from_nested_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+10", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+14", -10.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE8 → (LO+8, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 40 to 140 so the structural-dim
            # evidence (LO+8 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e8_from_local_frame_addr_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 2.0),
                ("ADDR_B0_LO+8", 2.0),
                ("ADDR_B0_HI+14", 2.0),
                # When upstream L16/L17 store amplifiers have already pushed
                # OUTPUT_LO+8 / OUTPUT_HI_THIS_STEP+14 to ~1e8 (model is confidently
                # emitting 0xe8), the OneHotBandGuarantee lane corrections
                # multiply ``(target - current)`` by ``silu(up) ≈ S * (score -
                # threshold)`` and overshoot into ~1e10 at the inactive lanes,
                # flipping the prediction. These tiny weights suppress the
                # rule in that high-magnitude regime without weakening the
                # uncertain-OUTPUT exactness recovery path.
                ("OUTPUT_LO+8", -0.001),
                ("OUTPUT_HI_THIS_STEP+14", -0.001),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=35.0,
            active_value=5000.0,
        ),
        # MEM_STORE (sem ``mark == MEM AND opcode_in_step in {SI, SC, PSH}``)
        # positive combined with CMP+0 (sem ``mark == AX OR (is_byte AND
        # byte_index == 0)``) positive plus the IS_BYTE/MARK_AX/MARK_STACK0
        # hard blockers makes the conditions-only effective predicate
        # unsatisfiable; F-5 falls back to the gate ``OUTPUT_HI_THIS_STEP+14`` whose
        # semantics is tautological. No scope tighter than tautology is
        # entailable. Leave scope/dominates_at unset for now.
        FFNRule.gated_write(
            name="tail_mem_store_addr0_e8_from_local_frame_output_exact",
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("CMP+0", 2.0),
                ("ALU_LO+8", -1000.0),
                ("OP_JSR", -1000000.0),
                ("IS_BYTE", -1000000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=25.0,
            gate="OUTPUT_HI_THIS_STEP+14",
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # Non-memory binary pops should emit a zero MEM row. Store/load ops
        # have dedicated memory paths and block this cleanup.
        FFNRule.constant_write(
            name="tail_pop_mem_marker_zero",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
                ("OP_SI", -100.0),
                ("OP_SC", -100.0),
                ("OP_LI", -100.0),
                ("OP_LC", -100.0),
                ("MEM_STORE", -100.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=500.0),
        ),
        # BP is stable across ordinary binary ops. The L10 BP passthrough
        # carries byte 2 as a weak 0x01 signal; reinforce it when no frame op
        # is rewriting BP.
        # The 30 negative side-effect writes (byte_writes 0x01 lays out
        # -strength at every non-target nibble in the 16-wide OUTPUT_LO/HI
        # bands) get flagged against the +1e8 magnitude of the global
        # tail_clear_output_after_byte3 suppressor under the V1 contribution
        # algebra used by verify_rule_strength. The conditions' effective
        # predicate also collapses to a gate-fallback tautology (the
        # IS_BYTE/HAS_SE/H1+3 positives combined with the wide MARK_* and
        # OP_* hard blockers turn out unsatisfiable for the
        # effective_predicate walker), so an honest scope claim isn't
        # possible without code-level restructuring of the conditions.
        # Leave scope/dominates_at unset until the verifier supports
        # sign-aware competition and the condition shape can be tightened.
        FFNRule.gated_write(
            name="tail_bp_byte2_preserve_01",
            conditions=(
                ("IS_BYTE", 1.0),
                ("HAS_SE", 1.0),
                ("H1+3", 20.0),
                ("H1+1", -100.0),
                ("H1+2", -100.0),
                ("BYTE_INDEX_1", 1_000_000.0),
                ("BYTE_INDEX_0", -1_000_000.0),
                ("BYTE_INDEX_2", -100.0),
                ("BYTE_INDEX_3", -100.0),
                ("OUTPUT_LO+1", 0.1),
                ("OUTPUT_LO+0", -0.2),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("MARK_AX", -1_000_000.0),
                ("MARK_PC", -1_000_000.0),
                ("MARK_SP", -1_000_000.0),
                ("MARK_BP", -1_000_000.0),
                ("MARK_STACK0", -1_000_000.0),
                ("MARK_MEM", -1_000_000.0),
                ("OP_ENT", -100.0),
                ("OP_LEA", -1000.0),
                ("OP_LEV", -100.0),
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
            ),
            threshold=900_021.5,
            gate="H1+3",
            writes=byte_writes(0x01, strength=500.0),
        ),
        *ax_add_no_carry_zero_rules(),
        *ax_add_byte1_high_zero_rules(),
        *ax_add_byte1_structural_materialize_rules(),
        *ax_sub_byte1_high_zero_rules(),
        *ax_sub_full_underflow_byte1_rules(),
        *ax_sub_borrow_decrement_rules(),
        *wide_mul_byte1_preserve_rules(),
        *ax_add_mul_byte1_materialize_rules(),
        FFNRule.gated_write(
            name="tail_ax_add_byte1_carry_high2_03",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("TEMP+8", 10.0),
                ("TEMP+9", -1000.0),
                ("CARRY+1", 1.0),
                ("CARRY+2", -1000.0),
                ("FETCH_HI+1", 0.1),
                ("OUTPUT_LO+3", 0.001),
                ("OUTPUT_LO+6", -10.0),
                ("OUTPUT_LO+7", -10.0),
                ("OP_IMM", -1000.0),
            ),
            threshold=1000.0,
            gate=dim_ref("carry", "alu", 1),
            writes=byte_writes(0x03, strength=500_000.0),
        ),
        # SHL-by-8 loses byte 1 to the same tail, but its signature is a huge
        # OUTPUT_LO[1] plus carry residue rather than MUL's OUTPUT_LO[2].
        FFNRule.gated_write(
            name="tail_wide_shl_byte1_01",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("OP_SHL", 100.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+0", 1.0),
            ),
            threshold=90.0,
            gate=dim_ref("carry", "alu", 3),
            writes=byte_writes(0x01),
        ),
        # SI preserves AX while writing memory. The dependency-expanded tail
        # can clobber the AX byte-1 prediction after the earlier layers have
        # prepared the right 16-bit value. OP_SI is relayed to AX byte
        # positions by L7; the HI-nibble comparison distinguishes a real
        # nonzero stored high byte from the common zero-high-byte store cases.
        # The si_ax_byte0 conditions combine MARK_AX-blocker, OP_SI positive
        # (sem ``mark == AX AND opcode_at_AX == SI``) and MEM_STORE positive
        # (sem ``mark == MEM AND opcode_in_step in {SI, SC, PSH}``); these
        # contradict via the MARK_AX blocker, and F-5 falls back to the
        # MEM_STORE gate semantics. Match scope/dominates_at to that
        # effective gate firing set so F-7 entailment succeeds and
        # cross-op strength competition is limited to MEM-store rows.
        FFNRule.gated_write(
            name="tail_si_ax_byte1_12",
            scope="mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            dominates_at={
                "OUTPUT_LO": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
                "OUTPUT_HI_THIS_STEP": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            },
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI_THIS_STEP+1", 0.5),
                ("OUTPUT_HI_THIS_STEP+0", -0.5),
            ),
            threshold=161.0,
            gate="MEM_STORE",
            writes=byte_writes(0x12, strength=300.0),
        ),
        FFNRule.gated_write(
            name="tail_si_ax_byte1_00",
            scope="mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            dominates_at={
                "OUTPUT_LO": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
                "OUTPUT_HI_THIS_STEP": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            },
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI_THIS_STEP+0", 0.5),
                ("OUTPUT_HI_THIS_STEP+1", -0.5),
            ),
            threshold=161.5,
            gate="MEM_STORE",
            writes=byte_writes(0x00, strength=300.0),
        ),
        # SUB 0x0100-1 carries borrow residue in CARRY[2]/[3] and must clear
        # byte 1 to zero; the old tail currently leaves 0x01 there. Use
        # CARRY[2] instead of CARRY[3] so wide MUL/SHL carry residue does not
        # accidentally trigger the zeroing rule.
        FFNRule.gated_write(
            name="tail_sub_borrow_byte1_00",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("MARK_AX", -1000.0),
                ("OP_IMM", -1000.0),
                ("TEMP+8", -20000.0),
                ("TEMP+9", 100.0),
                ("CARRY+2", 100.0),
                ("ALU_LO+1", 1.0),
                ("OUTPUT_LO+1", -5.0),
                ("OUTPUT_LO+3", -5.0),
                ("OUTPUT_LO+5", -5.0),
                ("OUTPUT_LO+6", -5.0),
            ),
            threshold=250.0,
            gate=dim_ref("carry", "alu", 2),
            writes=byte_writes(0x00),
        ),
        # 16-bit AND's high byte must zero; CMP/TEMP distinguish AND from
        # OR/XOR, whose high bytes intentionally remain 0x0f.
        FFNRule.constant_write(
            name="tail_and_byte1_00",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", 1.0),
                ("CMP+11", 1.0),
                ("CMP+12", 1.0),
            ),
            # In the expanded full-VM layout CMP[11]/[12] intentionally alias
            # TEMP[3]/[4], so this is effectively ax_byte0 + TEMP[3] + TEMP[4].
            threshold=4.5,
            writes=byte_writes(0x00),
        ),
        # OR/XOR byte 1 should remain 0x0f. The late tail inflates it to
        # 0x1e; TEMP[4] distinguishes AND and is used here as a blocker so
        # the AND-zeroing rule above remains authoritative for AND.
        FFNRule.gated_write(
            name="tail_or_xor_byte1_0f",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", -2.0),
                ("CARRY+3", 0.01),
                ("OUTPUT_LO+14", 0.001),
            ),
            threshold=5.5,
            gate=dim_ref("byte_index", "0"),
            writes=byte_writes(0x0F),
        ),
        # SHR by 8 also needs byte 1 cleared after the marker correction emits
        # byte 0 as 0x01; TEMP[7] is the reliable non-carry/SHR signature at
        # byte positions.
        # The shared ``ax_byte0`` prefix combines MARK_AX hard blocker (-1e9)
        # with positives whose semantics include ``mark == AX``, so the
        # conditions-only effective collapses to a contradiction and F-5
        # falls back to the TEMP+7 gate (no semantics) producing a
        # tautology. Leave scope/dominates_at unset until the shared
        # ax_byte0 conditions are restructured.
        FFNRule.gated_write(
            name="tail_shr_byte1_00",
            conditions=ax_byte0 + (
                ("TEMP+7", 1.0),
                ("OUTPUT_LO+1", 0.001),
            ),
            threshold=3.9,
            gate="TEMP+7",
            writes=byte_writes(0x00),
        ),
        # SHR by 8 currently computes byte 0 as 0x06 at the AX marker. OP_SHR
        # is still visible at the marker, so correct the marker prediction
        # before byte generation proceeds.
        FFNRule.gated_write(
            name="tail_shr_marker_byte0_01",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("IS_BYTE", -100.0),
                ("H1+1", 1.0),
                ("TEMP+7", 1.0),
                ("OP_SHR", 1000.0),
                ("OP_IMM", -100.0),
                ("OP_LEA", -1000000.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+2", -1.0),
                ("OUTPUT_LO+10", -1.0),
            ),
            threshold=5005.0,
            gate=gate_mark_ax,
            writes=byte_writes(0x01),
        ),
        FFNRule.constant_write(
            name="tail_lea_local_ax_marker_byte0_e8",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                ("CMP+7", 1.0),
                ("FETCH_LO+8", 2.0),
                ("FETCH_HI+15", 0.2),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ),
            threshold=9.0,
            writes=byte_writes(0xE8, strength=1_000_000.0),
        ),
        FFNRule.constant_write(
            name="tail_ax_add_byte1_missing_stack_high_02",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 10.0),
                ("HAS_SE", 10.0),
                ("H1+1", 20.0),
                ("BYTE_INDEX_0", 10.0),
                ("BYTE_INDEX_1", -1000.0),
                ("TEMP+8", 50.0),
                ("TEMP+9", -1000000.0),
                ("CARRY+1", 10000.0),
                ("FETCH_HI+1", 100000.0),
                ("OUTPUT_LO+3", -3.0),
                ("OUTPUT_LO+4", -3.0),
                ("OUTPUT_LO+5", -3.0),
                ("OUTPUT_LO+6", -3.0),
                ("OUTPUT_LO+7", -3.0),
                ("MARK_AX", -10000000000.0),
                ("MARK_PC", -10000000000.0),
                ("MARK_SP", -10000000000.0),
                ("MARK_BP", -10000000000.0),
                ("MARK_STACK0", -10000000000.0),
                ("MARK_MEM", -10000000000.0),
                ("STACK0_BYTE0", -100000000.0),
                ("STACK0_BYTE1", -100000000.0),
                ("STACK0_BYTE2", -100000000.0),
                ("STACK0_BYTE3", -100000000.0),
                ("H1+0", -100000000.0),
                ("H1+2", -100000000.0),
                ("H1+3", -100000000.0),
                ("H1+4", -100000000.0),
            ),
            threshold=130000.0,
            writes=byte_writes(0x02, strength=5000.0),
        ),
        # Comparison combine still sees amplified CMP residuals in the
        # expanded strict path. These two marker-only corrections restore the
        # truthy NE and LE cases without touching byte-lane arithmetic.
        FFNRule.constant_write(
            name="tail_cmp_ne_true_01",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_NE", 1.0),
                ("CMP+1", -0.5),
            ),
            threshold=4.5,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_eq_false_00",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_EQ", 1.0),
                ("CMP+1", -1.0),
                ("CMP+2", -1.0),
            ),
            threshold=4.5,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_le_lt_true_01",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LE", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=7.0,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_le_eq_prefix_false_00",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LE", 1.0),
                ("CMP+1", 0.1),
                ("CMP+0", -1.0),
                ("CMP+2", -1.0),
                ("CMP+3", -1.0),
            ),
            threshold=6.1,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_lt_false_00",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LT", 1.0),
                ("CMP+0", 0.01),
            ),
            threshold=7.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_gt_false_00",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_GT", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=8.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        # Keep this appended after the legacy tail rules so existing generated
        # unit indexes remain stable. It repairs SP marker d8->e0 when the
        # staged value exists only in OUTPUT, not in EMBED.
        # The rule writes 0xE0 (one-hot: +5000 at LO+0/HI+14, -5000 at the
        # other 30 nibbles). V1's sign-blind comparison flags the 30
        # negative side writes against a stronger negative competitor
        # (tail_clear_output_after_byte3 at -1e8); both rules cooperatively
        # push those lanes down, so the rivalry is spurious. The +5000
        # positive writes at the target nibbles dominate correctly. Narrow
        # dominates_at down to the actual firing site so positive-write
        # competition is limited to SP marker rows; the negative side
        # writes remain flagged until the verifier becomes sign-aware.
        FFNRule.gated_write(
            name="tail_sp_pop_marker_output_d8_to_e0",
            scope="mark == SP",
            dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
            conditions=(
                ("CONST", -100000000.0),
                ("MARK_SP", 100000000.0),
                ("HAS_SE", 1000.0),
                ("CMP+3", 100.0),
                ("OUTPUT_LO+8", 10.0),
                ("OUTPUT_HI_THIS_STEP+13", 100.0),
                ("OUTPUT_HI_THIS_STEP+15", -100.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("OP_ENT", -1000000.0),
                ("OP_LEV", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("IS_BYTE", -100.0),
            ),
            threshold=1505.0,
            gate=gate_mark_sp,
            writes=byte_writes(0xE0, strength=5000.0),
        ),
        # Pure suppressor (clear_output_writes lays out -1e8 on every
        # OUTPUT_LO/HI nibble lane) with no positive write. Under V1's
        # cross-sign contribution algebra used by verify_rule_strength,
        # the -1e8 magnitude is compared against unrelated positive
        # override contributions (tail_sp_pop_byte3_zero at +5e11) and
        # falsely flagged at all 32 nibble lanes. In practice this is the
        # designated dominator that drives every nibble lane to 0 at the
        # byte3 step boundary; competition with positive-write rules is
        # spurious. Leave scope/dominates_at unset; the verifier cannot
        # prove a useful claim with the current sign-blind algebra.
        FFNRule.gated_write(
            name="tail_clear_output_after_byte3",
            conditions=(
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_3", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("STACK0_BYTE0", -1000.0),
                ("STACK0_BYTE1", -1000.0),
                ("STACK0_BYTE2", -1000.0),
                ("STACK0_BYTE3", -1000.0),
                ("MEM_VAL_B0", -1000.0),
            ),
            threshold=1.5,
            gate=dim_ref("byte_index", "3"),
            writes=clear_output_writes(strength=100_000_000.0),
        ),
        # Pure suppressor (clear_output_writes lays out -1e8 on every
        # OUTPUT_LO/HI nibble lane) with no positive write. Under V1's
        # cross-sign contribution algebra used by verify_rule_strength,
        # this rule's -1e8 magnitude is compared against the very large
        # POSITIVE override contributions of unrelated tail materializers
        # (tail_sp_pop_byte3_zero etc. at +5e11) and falsely flagged. In
        # practice the rule cooperatively drives every nibble lane to 0
        # in tandem with tail_clear_output_after_byte3 (the cross-sign
        # rivalry isn't real). The effective predicate is also collapsed
        # to the NEXT_SE gate fallback (semantics ``NOT is_byte``) because
        # the IS_BYTE+NEXT_SE conditions contradict in the registry
        # semantics. Leave scope/dominates_at unset; the verifier cannot
        # prove a useful claim with the current algebra.
        FFNRule.gated_write(
            name="tail_clear_output_before_step_end",
            conditions=(
                ("IS_BYTE", 1.0),
                ("NEXT_SE", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("STACK0_BYTE0", -1000.0),
                ("STACK0_BYTE1", -1000.0),
                ("STACK0_BYTE2", -1000.0),
                ("STACK0_BYTE3", -1000.0),
            ),
            threshold=1.5,
            gate="NEXT_SE",
            writes=clear_output_writes(strength=100_000_000.0),
        ),
    ) + sp_pop_carry_rules()
    return step_end_transition_blocked(
        pc_byte_span_blocked(stack0_span_blocked_tail_rules(rules))
    )


def make_tail_bit32_result_correction_op() -> Operation:
    """Append a generated FFN block after the dependency-assigned L10 tail."""

    rules = _tail_bit32_result_correction_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = block.ffn.W_up.shape[1] if hasattr(block.ffn, "W_up") else 512
        # Per-bake FFN-unit allocator. The tail FFN is a standalone bank
        # whose ``hidden_dim`` equals ``len(rules)``; pinning the full
        # range under a single op name makes the bank's tenancy
        # explicit so a future second tenant goes through
        # ``allocator.alloc(...)`` instead of silently aliasing rule
        # rows. The factory also fails fast on rule-count drift between
        # the layout table and the materialised rule set.
        allocator = _allocate_l10_tail_bit32_units(len(rules))
        ffn = PureFFN(d_model, len(rules))
        ffn._l10_unit_allocator = allocator
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        _suppress_ffn_on_step_boundary(ffn, dim_map, S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="tail_bit32_result_correction",
        reads={
            "CONST", "IS_BYTE", "HAS_SE", "H1", "H3",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
            "NEXT_STACK0", "NEXT_MEM", "NEXT_SE", "OP_SHL", "OP_SHR", "OP_IMM", "OP_JSR",
            "PSH_AT_SP",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_ENT", "OP_LEV", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
            "OP_ADD", "OP_SUB", "OP_DIV", "OP_MOD", "OP_AND", "OP_OR",
            "OP_XOR",
            "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            "TEMP", "CMP", "CARRY", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "FETCH_HI",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "MEM_STORE",
            "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
            "ADDR_B0_LO", "ADDR_B0_HI",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        # Phase 8.A.4 retry: layer_idx=17 literal dropped (was redundant
        # alongside ``target_op_name`` since ``target_op_name`` takes
        # precedence in ``resolve_block_op_layer``). The L17 placement is
        # dep-derived from ``l10_post_ops_combined``'s position.
        target_op_name="l10_post_ops_combined",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeMemory::test_si_li_16bit_value",
        },
        spec_section="BLOG_SPEC.md#wide-alu-tail-correction",
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
      CarryPropagationPostOp x3,
      BitwiseBytePropagationPostOp,
      ComparisonCombine.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

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
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            AddSubBytePropagationPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
            _SetDim,
        )
        # Derive d_model from the block's residual-stream width. Preferred
        # source is ``block.attn.dim`` (always set to the model's d_model on
        # the AutoregressiveAttention used by every TransformerBlock); the
        # legacy ``block.ffn.W_up.shape[1]`` fallback is incorrect for
        # efficient-mode L10 where ``block.ffn`` is a ``PureNeuralALU``
        # subclass (e.g. ``ALUAndOrXor``) that has no ``W_up`` attribute,
        # silently bottoming out at the hard-coded ``512`` constant. With
        # the dynamic dim allocator (e.g. ``alu_mode='efficient'`` lifting
        # d_model to 800 to accommodate the V2/G7 LEV detector residual
        # band) this caused a stale ``BD.TEMP + 8 = 706`` to overflow the
        # 512-wide post-op ``W_up`` row and raise ``IndexError``.
        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512

        # Pass dim_positions so each post-op bakes against the compact layout
        # rather than legacy `_SetDim` positions. Without this, the post-ops
        # write to / read from `_SetDim.OUTPUT_LO/HI/CARRY/H1/OP_*` etc.,
        # which alias unrelated compact dims (e.g. `_SetDim.H1+1=68` aliases
        # compact `EMBED_HI[15]`; `_SetDim.CARRY=392` aliases a different
        # compact slot, etc.), corrupting OUTPUT/CARRY/CMP flags and silently
        # zeroing or scrambling the binary-op result. Threading dim_positions
        # to all 4 post-op classes is the L10 counterpart of the L1 fix in
        # commit 5fc519d (BinaryOpByteZeroingPostOp).
        zeroing = BinaryOpByteZeroingPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(zeroing, dim_positions, S)
        block.post_ops.append(zeroing)
        BD = _as_setdim_proxy(dim_positions) if isinstance(dim_positions, dict) else _SetDim
        addsub = AddSubBytePropagationPostOp(
            d_model=d_model,
            S=S,
            dim_positions=dim_positions,
        )
        _strengthen_l10_addsub_wrong_byte_blockers(addsub, BD, S)
        _suppress_l10_addsub_on_wide_alu(addsub, BD, S)
        _suppress_ffn_on_step_boundary(addsub, dim_positions, S)
        block.post_ops.append(addsub)
        carry0 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=0, cascade=False,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry0, BD, byte_idx=0, S=S)
        _strengthen_l10_first_carry_delta(carry0, BD)
        _suppress_ffn_on_step_boundary(carry0, dim_positions, S)
        block.post_ops.append(carry0)
        carry1 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=1, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry1, BD, byte_idx=1, S=S)
        _suppress_ffn_on_step_boundary(carry1, dim_positions, S)
        block.post_ops.append(carry1)
        carry2 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=2, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry2, BD, byte_idx=2, S=S)
        _suppress_ffn_on_step_boundary(carry2, dim_positions, S)
        block.post_ops.append(carry2)
        bitwise = BitwiseBytePropagationPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(bitwise, dim_positions, S)
        block.post_ops.append(bitwise)
        if alu_mode == "efficient":
            # Pass the model's actual d_model so the underlying PureFFN's
            # Linear input dim matches the residual stream width. Without
            # this, ComparisonCombine builds a Linear(512, 18) which fails
            # forward when d_model != 512 (e.g., pin_io_only=True paths).
            compare = ComparisonCombine(
                d_model=d_model, S=S, dim_positions=dim_positions
            )
            _suppress_ffn_on_step_boundary(compare, dim_positions, S)
            block.post_ops.append(compare)
        # DIV/MOD post_op (FlattenedDivMod) appended by
        # ``make_l10_alu_divmod_install_op`` (phase=10.8). Both modes use the
        # same flattened composite — its forward is byte-identical to the
        # previous EfficientDivMod_Neural.

    return Operation(
        name="l10_post_op_attach",
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        # Phase 11.A r3: dropped phase=10.7 — target_op_name and
        # requires['after'] already pin ordering at layer10_carry_relay.
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        requires={"after": "layer10_carry_relay"},
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
