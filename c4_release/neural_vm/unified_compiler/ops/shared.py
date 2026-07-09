"""Shared helpers and constants used by the per-layer op modules.

Extracted from the legacy ``migrated_ops.py`` (2026-05-11) so the per-layer
factory modules can import these without circular dependencies. The public
API is unchanged: ``from c4_release.neural_vm.unified_compiler.migrated_ops
import _as_setdim_proxy, declare_setdim_compat_dims, ...`` continues to work
via the re-export shim in ``migrated_ops.py``.
"""

import os
from typing import Dict
import torch.nn as nn

from ..layer_compiler import Operation


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

def campaign_enabled() -> bool:
    """Return True iff the single CAMPAIGN-CONFIG entry point is set
    (``C4_CAMPAIGN=1``). DEFAULT OFF.

    This is the one flag that turns on the coherent 30-token *campaign* set
    without the caller having to juggle the growing pile of individual campaign
    flags (``C4_NO_STACK0_EMIT`` ``C4_OPERAND_FROM_MEMSP`` ``C4_SI_STORE_ADDR``
    ``C4_OPERAND_CAM_FIX`` ...). It is wired in by having each campaign-gating
    predicate OR-in ``campaign_enabled()``:

      * ``no_stack0_emit_enabled()``     -- STACK0 register block dropped (30-tok)
      * ``operand_from_memsp_enabled()`` -- operand-A read from ``mem[SP]``
      * ``si_store_addr_enabled()``      -- SI/SC store address-provenance CAM
      * ``operand_cam_fix_enabled()``    -- widened operand-CAM address-leak clear

    So ``C4_CAMPAIGN=1`` ALONE reproduces the full campaign config, and it is
    exactly equivalent to setting each of those explicit flags to its campaign
    value. The two predicates that ALREADY default ON in a bare env
    (``no_stack0_emit`` / ``operand_from_memsp``, both ``!= "0"``) are unchanged
    by the OR-in; the two that default OFF (``si_store_addr`` / ``operand_cam_fix``)
    are the ones ``C4_CAMPAIGN`` flips on.

    An explicit per-flag override still wins for the two default-ON predicates
    (setting ``C4_NO_STACK0_EMIT=0`` opts that one op out even under the
    campaign): the OR-in only supplies a default-ON floor for the normally-OFF
    campaign fixes; the default-ON predicates keep reading their own env first.
    DEFAULT OFF -> the golden (35-tok) build is byte-identical (``C4_CAMPAIGN``
    unset -> every predicate keeps its exact pre-existing value). Registered in
    BOTH cache-key snapshots in ``full_vm_compiler_dynamic.py`` so a campaign
    build never shares a memo / disk cache entry with a golden build.
    """
    return os.environ.get("C4_CAMPAIGN", "0") != "0"


def derive_imm_enabled() -> bool:
    """Flag for the FULLY-DERIVED IMM opcode (task #392). Default OFF.

    When ``C4_DERIVE_IMM=1`` the ENTIRE IMM opcode is derived from its
    declarative spec with ZERO hand-authored IMM rules, closing the last
    LOWERING gap the DERIVE_DECODE pilot named (docs/DERIVE_DECODE_PILOT
    §G-IMM-RELAY):

      * decode -> ``OP_IMM``: routed through the generic :func:`decode_band`
        engine (the same path ``C4_DERIVE_DECODE`` enables) — implied ON by
        ``C4_DERIVE_IMM`` (see ``l5_ops._derive_decode_enabled``).
      * relay ``OP_IMM`` -> AX byte positions: the L8 head-4 relay is
        re-expressed by the new generic :func:`marker_broadcast` head
        generator (``isa_semantics_dsl.py``) instead of the hand-built
        ``_layer8_op_imm_relay_head_spec``.
      * value route imm -> OUTPUT: the 32-rule per-nibble AX_CARRY->OUTPUT
        copy is re-expressed by the generic value-route lowering
        (``isa_semantics_dsl.value_route``) instead of the hand-built
        ``_layer8_multibyte_routing_rules``.

    All three derivations reproduce the hand-authored weights byte-for-byte
    (proof: ``tools/_isa_golden_hash.py`` unchanged under ``C4_DERIVE_IMM=1``),
    so IMM becomes the FIRST 100%-derived opcode. Default OFF => the
    hand-authored path stays the golden build.
    """
    return os.environ.get("C4_DERIVE_IMM", "0") != "0"


def derive_bitwise_enabled() -> bool:
    """Flag for the SPEC-DERIVED BITWISE family (OR/XOR/AND). Default OFF.

    BLOG_SPEC §568: *"Bitwise: 10 weights — one formula (a+b-ab) for all"*.
    All three bitwise ops are ONE per-bit polynomial
    ``r_bit = c_a*a_bit + c_b*b_bit + c_ab*(a_bit*b_bit)`` applied across the
    4 nibble bits, with a single per-op coefficient triple read straight from
    the spec text:

      * OR  = ``a + b - a*b``   -> ``(c_a, c_b, c_ab) = ( 1,  1, -1)``
      * AND = ``        a*b``   -> ``( 0,  0,  1)``
      * XOR = ``a + b - 2*a*b`` -> ``( 1,  1, -2)``

    When ``C4_DERIVE_BITWISE=1`` the L10 bitwise result nibbles (lookup post-op
    AND both L10-main-FFN callers) are produced by this ONE shared formula
    (``wide_alu_dsl._bitwise_result_from_spec_formula``) instead of the three
    enumerated ``operator.and_/or_/xor`` bit functions
    (``wide_alu_dsl._BITWISE_OP_FN``). The result nibbles are provably
    identical per ``(a, b)`` pair (each per-bit result is in ``{0, 1}`` by
    construction), so ``tools/_isa_golden_hash.py`` is UNCHANGED under the flag
    — the derivation is a SOURCE collapse (3 distinct bit operators -> 1 spec
    formula + 3 coefficient triples), not a weight change.

    Default OFF => the enumerated-operator path stays the golden build. Both
    ON/OFF are registered in the ``full_vm_compiler_dynamic.py`` cache-key
    snapshots so a derived build never shares a memo / disk entry with the
    hand path. See ``docs/DERIVE_BITWISE_2026_07_09.md``.
    """
    return os.environ.get("C4_DERIVE_BITWISE", "0") != "0"


def mul_width2_enabled() -> bool:
    """Return True iff the width=2 (16-bit) MUL path is active (DEFAULT ON).

    The width=2 (8-bit x 8-bit -> 16-bit) MUL is now the PRODUCTION default
    (2026-06-13). When enabled:
      * ``compile_full_vm_dynamic`` injects the dedicated
        ``MUL_RESULT_HI_LO/HI`` byte-1 result band (16+16 dims) into
        ``extra_residual_dims`` so the d_model auto-widen (872 -> 981,
        n_heads 8 -> 9) is HEAD-DIM-PRESERVING (adds heads, keeps every
        existing head's span) -- this is what makes the widen bnz-safe.
      * ``make_efficient_l11_alumul_wrap_op`` bakes the
        ``wide_mul_rules(width_bytes=2)`` 65,536-rule lookup with the
        operand-magnitude-matched 5-way AND + cell-0 artifact blocker,
        routing the product's BYTE 1 into ``MUL_RESULT_HI_*`` (NOT
        ``OUTPUT_LO+32`` = ADDR_KEY) and keeping BYTE 0 in OUTPUT_LO/HI.
      * ``make_layer13_mul_result_hi_relay_op`` stages byte 1 into AX_FULL
        for the existing ``layer15_alu_high_byte_relay`` byte-1 emit.

    Result: ``mul_overflow`` (100*5=500=0x01F4) emits both bytes correctly
    (smoke 49/2 -> 50/1) while ``mul_basic`` (6*7=42), bnz, and the 32-bit
    ALU / bitwise / shift / cmp suite stay green.

    Opt-OUT with ``C4_MUL_WIDTH2=0`` to restore the pre-width2 build
    (d_model 920, width=1 lo-byte MUL, ``mul_overflow`` decodes 20). Any
    other value (or unset) keeps the width=2 default ON.

    ``C4_MUL_MULTIPASS=1`` IMPLIES width=2: the multipass cascade routes the
    product's byte 1 into the ``MUL_RESULT_HI_LO/HI`` band, so that band (and
    the L13 relay that stages it into AX_FULL) MUST be present even if
    ``C4_MUL_WIDTH2=0`` was passed. OR-in the multipass flag so the band is
    always collected when the cascade is installed.
    """
    if os.environ.get("C4_MUL_MULTIPASS", "0") == "1":
        return True
    return os.environ.get("C4_MUL_WIDTH2", "1") != "0"


def mul_multipass_enabled() -> bool:
    """Return True iff the multi_pass (schoolbook cascade) MUL compute replaces
    the L11 mul-partial / L12 mul-combine lookup chain (DEFAULT OFF — opt in via
    ``C4_MUL_MULTIPASS=1``).

    GAP-PRIMITIVE #2 install. The live default (``alu_mode='lookup'``) MUL
    computes only the product's BYTE 0 across three lookup blocks:

      * L10 ``_layer10_alu_mul_lo`` (256 units) -> ``OUTPUT_LO`` nib0 = (a0*b0)%16
      * L11 ``mul_partial`` (4096 units)         -> ``TEMP[partial]``
      * L12 ``mul_combine`` (4096 units)         -> ``OUTPUT_HI`` nib1

    Byte 1 (bits 8..15) is never computed on the lookup path, so wide_mul
    programs cross-contaminate byte-0/byte-1 (#334). When this flag is on,
    ``make_mul_partial_op`` replaces ``block.ffn`` with a
    :class:`~neural_vm.efficient_alu_neural.MultiPassMulBlock` — the 7 lowered
    ``PureFFN`` passes of ``multi_pass_mul_rules`` packed into ONE physical
    block (like ``FlattenedALUMul``, so the absolute-position lea contract
    holds). The cascade computes the FULL 16-bit product from a compact
    schoolbook spec (2848 units vs the 4096+4096 lookup), routing byte 0 to
    ``OUTPUT_LO/OUTPUT_HI`` and byte 1 to the ``MUL_RESULT_HI_LO/HI`` band that
    the L13 relay stages into AX_FULL for the byte-1 emit.

    The redundant L10 mul_lo + L12 mul_combine byte-0 writers are gated OFF
    when this flag is on (the cascade owns every result nibble); the L11
    ``mul_partial`` TEMP band is no longer produced. Needs
    :func:`mul_width2_enabled` (the ``MUL_RESULT_HI_*`` byte-1 band + the L13
    relay) — implies it.

    DEFAULT OFF: flag-off is byte-identical to golden ``91f55411`` (the lookup
    chain is untouched). Opt in with ``C4_MUL_MULTIPASS=1``.
    """
    return os.environ.get("C4_MUL_MULTIPASS", "0") == "1"


def div_multipass_enabled() -> bool:
    """Return True iff the multi_pass (binary long-division cascade) DIV/MOD
    compute replaces the live L10 ``FlattenedDivMod`` composite (DEFAULT OFF —
    opt in via ``C4_DIV_MULTIPASS=1``).

    GAP-PRIMITIVE #2 DIV install (the DIV analogue of ``mul_multipass_enabled``).
    The live default (``alu_mode='lookup'``) computes DIV/MOD via the
    hand-authored ``FlattenedDivMod`` — an 8-outer x 3-inner GE-workspace long
    division (BD->GE, DIV pipeline, MOD pipeline, GE->BD). When this flag is on,
    the L10 divmod install (``make_alu_divmod_composite_ops`` -> install op)
    appends a :class:`~neural_vm.efficient_alu_neural.MultiPassDivBlock` instead
    — the lowered ``PureFFN`` passes of ``multi_pass_div_rules`` (bit-serial
    shift-subtract with a cross-pass running-remainder carry) packed into ONE
    physical block (like ``MultiPassMulBlock``, so the absolute-position lea
    contract holds). The cascade computes ``a // b`` (quotient) + ``a % b``
    (remainder) from a compact spec on dedicated result lanes, then routes
    q -> OUTPUT for OP_DIV and r -> OUTPUT for OP_MOD at MARK_AX. It reads the
    dividend from ``ALU_LO/HI`` (byte 0) and divisor from ``AX_CARRY_LO/HI``,
    the SAME operand bands the GE-format lookup / ``FlattenedDivMod`` consume.

    DEFAULT OFF: flag-off is byte-identical to golden ``91f55411`` (the
    ``FlattenedDivMod`` composite is untouched). Opt in with
    ``C4_DIV_MULTIPASS=1``.
    """
    return os.environ.get("C4_DIV_MULTIPASS", "0") == "1"


def mul_w2_thresh_fix_enabled() -> bool:
    """Return True iff the width=2 MUL 5-way-AND threshold is lowered to fire
    the clean-operand wide-product cases the default 19.5 just barely blocks
    (DEFAULT OFF — opt-in via ``C4_MUL_W2_THRESH_FIX=1``).

    The bug this lifts (verified spec_k=0 ground-truth full_trace + the offline
    SwiGLU sim ``tools/_mul16_retune.py`` over 32 probed operand vectors):
    the wide-product 16-bit MUL path is ALREADY ~26/30 correct on the pure-MUL
    band, but a handful of CLEAN-operand cases (mul_11 100*68 -> 0x1A90,
    mul_31 52*86 -> 0x1178) miss because their true-quad 5-way-AND condition
    lands at ~19.45 — a razor-thin 0.05 below the default ``threshold=19.5``
    (their A high-nibble one-hot is ~0.13 weaker than the smoke cases'). The
    rule never fires, so ALL four result nibble bands are empty and the product
    truncates to byte 0 (got 0x90 / 0x78, high byte lost).

    The fix lowers the wide_mul width=2 5-way-AND threshold from 19.5 to 19.0.
    Verified over the 32-case probed grid: every CLEAN-operand fixable case now
    fires (30/30) with a STRICTLY BETTER worst-case result-band margin (0.044 vs
    0.000) — the threshold does NOT admit any spurious quad, the smoke cases
    (6*7=0x2A, 100*5=0x1F4) stay byte-correct, and the band stays a clean
    one-hot for the L13 byte-1 relay's raw V@O copy. 19.0 (not the more
    aggressive 18.5) is the chosen value: at 18.5 an INTERMEDIATE MUL feeding a
    downstream DIV (expr_mul_div_19 3*16/8) spuriously emits a byte-1 that leaks
    into the divisor; the 0.44-margin 19.0 reliably fires the true quad while
    staying above that leak boundary (ground-truth full_trace verified — no expr
    regression). The 2 residual pure-MUL misses (9*98, 89*26) are an UPSTREAM
    operand-gather defect (ALU_HI reads nibble 3 instead of the true high
    nibble) — NOT a threshold issue, NOT fixable here.

    DEFAULT ON (flipped 2026-06-17 after smoke 51/0 flag-ON + +2 16-bit MUL
    verified): lowers the wide_mul width=2 firing threshold 19.5->19.0. Opt OUT
    via ``C4_MUL_W2_THRESH_FIX=0``. Only meaningful when ``mul_width2_enabled()``.
    """
    return os.environ.get("C4_MUL_W2_THRESH_FIX", "1") != "0"


def mul_stack0_byte39_guard_enabled() -> bool:
    """Return True iff the L10 tail ``byte_39_from_e8_addr`` STACK0 restore rule
    is hardened to require its full store-pop context (DEFAULT OFF — opt-in via
    ``C4_MUL_STACK0_BYTE39_GUARD=1``).

    The bug this lifts (root-caused spec_k=0 GPU full_trace, 2026-06-18):
    the L10 tail-correction rule
    ``tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr``
    (``ops/l10_ops.py`` ``stack0_store_top_e8_from_e0_output_rules``) restores
    the stored byte value ``0x39`` (= 57) at the e8/e0 local-store address
    transition (load-bearing for genuine ``0x39`` store-pops:
    ``func_identity(57)``, ``func_add(.., 57)``, ...). Its activation is driven
    almost entirely by the UNBOUNDED ``("OUTPUT_LO+9", 100.0)`` term: at the
    binary-op STACK0 byte-0 emission row OUTPUT_LO+9 carries the operand's low
    nibble at magnitude ~3654, so ``100 * 3654 = 365428`` alone blows past the
    ``threshold=20000`` even though EVERY store-pop witness (MEM_STORE,
    EMBED_LO+8, EMBED_HI+14, MEM_ADDR_SRC) is COLD on that row. The rule then
    forces the byte to ``0x39``, corrupting the STACK0 byte-0 token to 57
    (low nibble 9 — coincidentally matching the operand — high nibble 3) for
    EVERY binary op whose operand-A low nibble is 9.

    For MUL specifically the downstream width=2 lookup reads that corrupted
    STACK0 byte-0 as operand A: ALU_LO=9 (right by accident), ALU_HI=3 (the
    phantom 0x39 high nibble), so the 16-bit product is computed against the
    wrong high nibble. That is the SOLE remaining root behind the 4 GPU
    full_trace ``mul_*`` fails — exactly the a_lo=9 cases ``mul_20 (9*98)``,
    ``mul_29 (89*26)``, ``mul_36 (9*44)``, ``mul_43 (9*5)`` (``9*9`` is a_lo=9
    too but its byte-0 is right by accident so it passes). ADD/SUB read the
    same corrupted ALU_HI but tolerate it (the byte-0 add only needs ALU_LO),
    which is why only MUL surfaces it. Verified: sweeping operand-A 1..255
    with B=5, step-3 STACK0 byte-0 is byte-correct for EVERY low nibble except
    9, where it is universally 0x39.

    The genuine ``0x39`` store-pop the rule was built for has the value's HIGH
    nibble ALSO present in OUTPUT (``OUTPUT_HI_THIS_STEP+3`` hot) — it is
    restoring 0x39, so both nibbles evidence 0x39. The binary-op false-fire has
    ONLY the low nibble (``OUTPUT_HI_THIS_STEP+3 == 0``). When enabled this
    guard rebalances the rule so the byte's HIGH nibble (``OUTPUT_HI_THIS_STEP+3``)
    is a load-bearing co-requirement at the same weight as the low nibble, and
    a hard negative blocker suppresses the rule when ``OUTPUT_LO+9`` is hot but
    ``OUTPUT_HI_THIS_STEP+3`` is cold (the binary-op-result signature). The
    genuine 0x39 store-pop (both nibbles hot) is unaffected.

    DEFAULT OFF so the flag-off build stays byte-identical to golden
    ``b9d8861f``. Opt-in via ``C4_MUL_STACK0_BYTE39_GUARD=1``.
    """
    return os.environ.get("C4_MUL_STACK0_BYTE39_GUARD", "0") == "1"


def div_multibyte_enabled() -> bool:
    """Return True iff the multi-byte-dividend DIV/MOD relay is active
    (DEFAULT ON — opt-out via ``C4_DIV_MULTIBYTE=0``; +51: div 21->48/50,
    mod 24->48/50, add/sub guard 84/100 unchanged, smoke 51/0).

    Background — the wall this lifts (verified spec_k=0, built dims):
    the lookup-mode ``FlattenedDivMod`` long-division pipeline
    (``alu/ops/divmod_longdiv.py::LongDivisionModule``) is ALREADY
    multi-byte-capable: it reads the dividend as a full 8-nibble vector
    from GE positions 0..7 and does a real MSB->LSB long division. The
    ``BDToGEConverter`` (``efficient_alu_neural.py``) maps BD operand
    bands into those GE positions: byte 0 from ALU_LO/HI (positions 0/1),
    byte 1 into positions 2/3. For DIV/MOD it RECOVERS byte 1 from the
    autoregressive prefix via a cummax over ``STACK0_BYTE1`` rows.

    The bug (NOT "STACK0_BYTE_VAL_1 written nowhere" — that was a
    static-dim / wrong-row artifact): the converter's DIV/MOD fallback
    gathered ``CLEAN_EMBED_LO/HI`` at the picked STACK0-frame row, where
    that band is 0x00 — the pushed value's high byte is deposited by
    ``layer10_psh_ax_broadcast`` into ``STACK0_BYTE_VAL_1_LO/HI`` at that
    SAME row (e.g. 1162/37: STACK0_BYTE_VAL_1 = 0x04 at the PSH frame).
    Compounded by TIMING: the broadcast runs at L11 (physical block 15)
    but the FlattenedDivMod compute ran at L10 (physical block 14), ONE
    block too early to see the high byte.

    When enabled this op:
      * routes the FlattenedDivMod install to ``layer_idx=11`` so the
        divmod compute runs AFTER the L11 ``layer10_psh_ax_broadcast``
        head populates ``STACK0_BYTE_VAL_1`` (operands ALU_LO/HI +
        AX_CARRY are byte-stable across blocks 14..20, verified
        ``tools/probe_div_operand_survival.py``);
      * has ``BDToGEConverter`` read ``STACK0_BYTE_VAL_1_LO/HI`` (the
        designated high-byte carrier) at the cummax-picked STACK0_BYTE1
        row instead of ``CLEAN_EMBED_LO/HI``.

    DEFAULT ON (verified +51 with no add/sub/smoke regression). Opt-out
    via ``C4_DIV_MULTIBYTE=0`` restores the byte-identical-to-HEAD path
    (divmod stays at block 14, converter reads CLEAN_EMBED). See
    ``docs/DIV_MOD_MULTIBYTE_DIVIDEND_BLOCKER_2026_06_12.md``.
    """
    return os.environ.get("C4_DIV_MULTIBYTE", "1") != "0"


def divmod_byte0_se_recover_enabled() -> bool:
    """Return True iff the divmod DIVIDEND byte-0 SE_ALU recovery is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_DIVMOD_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``):

    operand-A byte 0 (the dividend low byte) is delivered from ``mem[SP]``
    into ALU_LO/HI by the L8 ``make_layer8_mem_to_alu_op`` head 5 at block
    11. For a MULTI-BYTE dividend the L10 ALU-clear (block 14) then crushes
    ALU_LO/HI all-negative (probe ``tools/probe_div_ge_positions.py``,
    1162/37: ALU_LO == -39 at the divmod-input block 27), so the
    ``FlattenedDivMod`` (block 28) reconstructs dividend byte 0 as 0x00 and
    divides ``high_byte(dividend)*256 / divisor`` -> wrong quotient. The
    SINGLE-byte case is unaffected (ALU_LO survives at +6.0). The byte-1
    (positions 2/3) path is already correct here via AX_FULL /
    STACK0_BYTE_VAL_1 (L8 head 7, Part B) — this fix is the byte-0 half.

    The L9 ``step_end_operand_relay`` head mirrors ALU_LO/HI into
    SE_ALU_LO/HI at block 13 — BEFORE the L10 clear — and that mirror
    SURVIVES to the divmod block (probe: SE_ALU == 0xA/0x8 at block 27 for
    1162/37, == the same nibbles as ALU_LO for single-byte 100/7). When
    enabled, ``BDToGEConverter`` OR-recovers the dividend byte-0 one-hot
    from SE_ALU_LO/HI onto the (possibly crushed) ALU band, gated on the
    divmod opcode + the AX marker so it touches no other op/config.

    DEFAULT ON. Opt-out via ``C4_DIVMOD_BYTE0_SE_RECOVER=0`` restores the
    raw ALU_LO/HI read (the byte-identical-OFF path: flag-OFF or
    ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden
    ``4958b35b``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_DIVMOD_BYTE0_SE_RECOVER`` can
    A/B it inside the campaign config.
    """
    return os.environ.get("C4_DIVMOD_BYTE0_SE_RECOVER", "1") != "0"


def mul_byte0_se_recover_enabled() -> bool:
    """Return True iff the MUL operand-A byte-0 SE_ALU recovery + the L11/L15
    MUL OUTPUT-flood cap are active (DEFAULT ON in the campaign config — opt-out
    via ``C4_MUL_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, ~16 of 23 mul fails whose
    product decodes to 0, e.g. mul_2 93*34, mul_3 65*98, mul_28 97*94):

    This is the MUL analog of :func:`divmod_byte0_se_recover_enabled` —
    operand-A byte 0 is delivered from ``mem[SP]`` into ALU_LO/HI by the L8
    ``make_layer8_mem_to_alu_op`` head 5 at block 11 (CLEAN one-hot, +6.0 at
    the true nibble; spec_k=0 ALU-band block trace). The L10
    ALU-clear (block 14) then crushes ALU_LO/HI all-negative (~-39 at the true
    nibble, ~-45 elsewhere) — the SAME crush the divmod path hits. The
    ``BDToGEConverter._clean_onehot`` thresholds those negative cells to 0, so
    operand A reconstructs as 0x00 and the ``FlattenedALUMul`` (block 29 /
    logical L15) computes ``0 * b == 0``. (Operand B byte 0 is read CLEAN from
    AX_CARRY_LO/HI at +0.996 — only operand A is crushed.)

    Two coupled effects in the campaign config:

      (1) BYTE-0 RECOVER. The L9 ``step_end_operand_relay`` head mirrors
          ALU_LO/HI into SE_ALU_LO/HI at block 13 — BEFORE the L10 clear — and
          that mirror SURVIVES to the FlattenedALUMul block (spec_k=0:
          SE_ALU_LO==0xD/SE_ALU_HI==5 at +0.78/
          +0.69 across blocks 14..29 for 93*34, == the operand-A nibbles).
          ``BDToGEConverter`` OR-recovers the operand-A byte-0 one-hot from
          SE_ALU_LO/HI onto the (crushed) ALU band on OP_MUL+MARK_AX rows, so
          FlattenedALUMul multiplies the REAL operand A.

      (2) OUTPUT-FLOOD CAP. The L11 ``efficient_l11_alumul_wrap`` (block 16 /
          logical L11) wide_mul FFN reads the RAW crushed ALU_LO (its own
          threshold-AND rules, NOT through ``_clean_onehot``) and — driven by
          the ~-39 uniform-negative band — FLOODS OUTPUT_LO/HI + MUL_RESULT_HI
          to ~2.4e9 (spec_k=0 OUTPUT-band block trace: OUTPUT_HI sum
          0 -> 2.42e9 at block 16, self-amplified to 6.9e9 by L14, 1.2e20 by
          L20, +inf by L25). This flood (a) trips the ``_MulCombineStage``
          ``already_fired`` guard (OUTPUT_HI band > 1.5) so the CORRECT
          FlattenedALUMul product is NEVER written, and (b) out-votes the +2.0
          product one-hot at the LM-head argmax. When enabled, the
          ``_MulCombineStage`` CLEARS the OUTPUT_LO/HI band on the OP_MUL+
          MARK_AX row before the clean product write and IGNORES the
          ``already_fired`` veto on those rows, so the SE-recovered product
          survives. (The golden 35-token config keeps clean POSITIVE operand
          one-hots through L11, so the wide_mul never floods and this cap is a
          no-op there — but the flag is gated on ``C4_NO_STACK0_EMIT`` so it
          can never touch the golden path.)

    DEFAULT ON. Opt-out via ``C4_MUL_BYTE0_SE_RECOVER=0`` restores the raw
    ALU_LO/HI read + the raw OUTPUT-flood path (the byte-identical-OFF path:
    flag-OFF or ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_MUL_BYTE0_SE_RECOVER`` can A/B it
    inside the campaign config.
    """
    return os.environ.get("C4_MUL_BYTE0_SE_RECOVER", "1") != "0"


def mul_l11_se_recover_enabled() -> bool:
    """Return True iff the L11 wide_mul operand-A SE_ALU recovery is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_MUL_L11_SE_RECOVER=0``;
    only takes effect when the STACK0 emission is dropped, i.e.
    ``C4_NO_STACK0_EMIT=1`` AND the MUL byte-0 SE recovery is on).

    The wall this lifts (#321, root aa52e1c9, verified spec_k=0 / BUILT dims,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the
    campaign mul byte-1 DROP fails (104/106/124/138, plus the a_lo=7 family) —
    the decode is correct for the LOW byte only, the product's high byte (byte
    1) is dropped at the emit token.

    Distinct from :func:`mul_byte0_se_recover_enabled` effect (2). That flag's
    OUTPUT-flood cap (``_MulCombineStage``) treats the L11 wide_mul flood
    AFTER it has happened — it CLEARS the flooded OUTPUT band so the
    SE-recovered ``FlattenedALUMul`` (block 29 / logical L15) product survives
    at the LM-head argmax. But the wide_mul ALSO computes the product's BYTE 1
    into the dedicated ``MUL_RESULT_HI`` band (block 16 / logical L11), and the
    ``_layer14_alu_high_byte_relay`` (l14_ops.py:1330, OP_MUL-gated) EMITs that
    byte 1 at the emit token. Because the L11 wide_mul reads the L10-CRUSHED
    operand-A (``ALU_LO/HI`` all-negative for the failing rows), its byte-1
    write is the FLOOD, NOT the true product byte 1 — and the flood drowns the
    +20-scaled byte-1 emit relay write, so the emit token decodes only the low
    byte. The L15 ``FlattenedALUMul`` already gets a clean byte-0 recovery (via
    ``BDToGEConverter``), but the L11 wide_mul one block EARLIER does NOT.

    FIX. When enabled, the efficient-mode L11 wrap
    (``make_efficient_l11_alumul_wrap_op``) installs ``MulOperandSeRecoverFFN``
    as ``block.ffn`` — a drop-in wrapper holding the rule-lowered wide_mul
    ``PureFFN`` (``inner``) that, on the OP_MUL + MARK_AX row ONLY, restores the
    crushed operand-A ``ALU_LO/HI`` band from the surviving ``SE_ALU_LO/HI``
    mirror BEFORE the wide_mul rules read it. It reproduces the SAME golden
    hybrid operand band (true-nibble one-hot + the index-0 / cell-8 / cell-15
    magnitude artifacts) the width=2 wide_mul AND-threshold + artifact-blocker
    were tuned against, so the wide_mul computes the CORRECT ``MUL_RESULT_HI``
    (no flood) and the byte-1 emit relay propagates. It runs in ONE block (it
    IS ``block.ffn``) so the physical block count is unchanged (the lea
    absolute-position contract holds). Exactly the byte-0 SE-recovery precedent
    (``CmpOperandSeRecoverFFN`` at L10), applied one block earlier at L11.

    DEFAULT ON. Opt-out via ``C4_MUL_L11_SE_RECOVER=0`` restores the raw crushed
    ``ALU_LO/HI`` read at the L11 wide_mul (the byte-identical-OFF path: flag-OFF,
    or ``C4_NO_STACK0_EMIT=0``, or ``C4_MUL_BYTE0_SE_RECOVER=0`` are all
    byte-identical to golden ``7f6f2e5d``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_MUL_L11_SE_RECOVER`` can A/B it
    inside the campaign config.
    """
    return os.environ.get("C4_MUL_L11_SE_RECOVER", "1") != "0"


def sub_full_borrow_enabled() -> bool:
    """Return True iff the SUB full-borrow multi-byte 0xFF completion is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_SUB_FULL_BORROW=0``;
    only takes effect when the operand is sourced from ``mem[SP]``, i.e. the
    campaign ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`` config).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config):
    ``sub_borrow_cascade`` (``0 - 1`` should = ``0xFFFFFFFF``) emits only
    ``0x000000FF`` — byte 0 (0xFF) is correct, but bytes 1-3 stay 0x00. It is a
    three-layer wall the per-byte cascade rules cannot reach in the 30-token
    campaign frame:

      1. The minuend byte 1 == 0x00 has NO ``STACK0_BYTE_VAL_1`` one-hot in the
         campaign (the L8 mem[SP] CAM delivers only the NON-zero nibble: ``+6``
         at cell ``v``, nothing for ``v == 0``). So the band is EMPTY and the
         ``_layer14_sub_borrow_high_byte_passthrough`` ``v==0`` rule — keyed on
         ``STACK0_BYTE_VAL_1_LO+0`` being lit — never fires.
      2. The byte-1 result is 0xFF = lo nibble 0xF AND hi nibble 0xF; the v>=1
         cascade rules only set OUTPUT_LO (assuming the hi nibble is 0).
      3. The block-32 (L18) OUTPUT_HI slam adds ``+664`` to OUTPUT_HI cell 0
         (the 0x00 hi default) and ``-434`` elsewhere — additively crushing any
         pre-slam OUTPUT_HI cell-15 write, so the high nibble can never win
         pre-slam.

    The fix is a two-part build, both campaign-gated:
      * a PRECURSOR FFN at an EARLY L14 block (where the STACK0_BYTE_VAL_1 band
        is still fresh) writes the bounded ``SUB_FULL_BORROW`` flag on the SUB
        byte-1 emit row (``TEMP+9`` + ``IS_BYTE`` + ``H1+1`` + ``BYTE_INDEX_0``
        + ``CARRY+2`` borrow) ONLY when the STACK0_BYTE_VAL_1 band is EMPTY (the
        minuend byte 1 == 0 full-underflow case);
      * a POST-SLAM writer on the L25 tail block (after
        ``tail_bit32_result_correction``, the last OUTPUT writer before the LM
        head) reads the persisted flag and overwrites OUTPUT byte 1 = 0xFF
        (cancel OUTPUT_HI cell-0, boost OUTPUT_LO/HI cell 15), which DOMINATES
        the slam because it runs after it.

    DEFAULT ON. Opt-out via ``C4_SUB_FULL_BORROW=0`` restores the byte-identical
    pre-fix path (flag-OFF or ``C4_OPERAND_FROM_MEMSP=0`` are both byte-identical
    to golden ``7f6f2e5d``: the band is flag-gated so a flag-off build omits it
    entirely → smaller d_model). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_SUB_FULL_BORROW`` can A/B it inside
    the campaign config.
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_SUB_FULL_BORROW", "1") != "0"


def l8_operand_sp_disc_enabled() -> bool:
    """Return True iff the L8 head-5 operand-A SP-frame discriminator is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_L8_OPERAND_SP_DISC=0``;
    only takes effect when the operand-A read is routed to ``mem[SP]``, i.e. the
    campaign ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`` config).

    The wall this lifts (validated GPU spec_k=0, BUILT dims, campaign config —
    tools/probe_addmul_head5_cam.py id816 4+5*2): ``expr_add_mul`` (``a+b*c``) is
    the only depth-2 expr cluster with TWO live stack stores at once (PSH a,
    PSH b, IMM c, MUL pops b, ADD pops a). The ADD's operand-A = mem[SP] = a (the
    FIRST push; top-of-stack after the MUL popped b). The recency-only L8 head-5
    ``mem[SP]`` CAM (``make_layer8_mem_to_alu_op``) picks the more-recent matching
    STORE row (the popped b push) over the live a store — the two store value
    rows are byte-identical (MEM_STORE_AT_VAL=1, MEM_VAL_B1=1 on BOTH; ADDR_B0/1/2
    = 0; emitted mem address byte = 0xE0 for BOTH). So AX = b + b*c (id816:
    5+10=15 vs 14).

    The ONLY real discriminator is the SP at push time: store a was pushed at
    SP=0xF8, store b at SP=0xF0; after the MUL pop SP returns to 0xF8, so the
    ADD's operand address is 0xF8 == a's frame. The SP low-byte is a clean
    one-hot in OUTPUT_LO at the MARK_SP marker row (id816: SP rows 108->8 (0xF8),
    168->0 (0xF0), 228 (post-MUL)->8 (0xF8)).

    When this flag is on:
      * ``make_layer7_sp_addr_relay_op`` (L7, block 9 — BEFORE head-5 reads at
        block 11) relays the push-time SP low byte into the fresh per-row band
        ``SP_ADDR_LO``: Q fires at MEM store value rows + binary-op AX query
        rows; K matches MARK_SP; steep ALiBi recency (0.5) picks the NEAREST
        PRIOR MARK_SP; V copies its OUTPUT_LO one-hot. add-step query@253
        SP_ADDR_LO=8; store@123 (a=4) =8 (MATCH); store@183 (b=5) =0 (MISMATCH).
      * ``make_layer8_mem_to_alu_op`` head-5 adds an OP_*-gated SP-mismatch
        penalty (anti-complement form): K[base+30+i]=SP_ADDR_LO+i (store side),
        Q[base+30+i]=G*SP_ADDR_LO+i (query side) + one const dim K=1/Q=-G, so the
        head-5 score gains ``G*(match - 1)`` = 0 on a frame MATCH and ``-G`` on a
        MISMATCH. The popped (mismatched) store is demoted by G; recency then
        picks the live (matched) store. match==0 keeps single-store ops
        (mul_div/mod/paren/standalone add/sub) byte-identical (their live store
        SP-matches its query so the penalty is 0 there).

    DEFAULT ON. Opt-out via ``C4_L8_OPERAND_SP_DISC=0`` restores the byte-
    identical pre-fix path (flag-OFF or ``C4_OPERAND_FROM_MEMSP=0`` are both
    byte-identical to golden ``f2b040aa``: the SP_ADDR_LO band is flag-gated so a
    flag-off build omits it entirely → smaller d_model, and the head-5 penalty
    dims only bake under the flag). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_L8_OPERAND_SP_DISC`` can A/B it
    inside the campaign config.

    *** DEFAULT ON (2026-06-26) — BLOCKER-1 RESOLVED, +9 expr_add_mul. ***
    The DISCRIMINATOR CORE was GPU-validated 2026-06-25 (at the depth-2 expr_add_
    mul ADD step it demotes the popped-``b`` store and head-5 delivers operand-A =
    the live ``a``: probe_addmul_head5_cam.py id816 ALU_LO 5->4, AX 15->14) but
    shipped OFF pending BLOCKER-1. BLOCKER-1 is now FIXED by the block-10
    WINNER-TAKE-ALL sharpener (``make_layer7_sp_addr_sharpen_op``): it saturating-
    clamps each relayed SP_ADDR_LO cell to [0,1] (-> SP_ADDR_LO_SHARP /
    SP_ADDR_PRESENT_SHARP, which head-5's penalty now reads), so the bilinear
    G*(<q,k> - PRESENT_q*PRESENT_k) penalty stays bounded (~few hundred, was ~+80k)
    and cancels EXACTLY at the matched winner for ALL ops. var_simple no longer
    regresses. GPU full_trace 250-274,800-899 (campaign, spec_k=0): flag-ON
    85/125 vs flag-OFF 76/125 = +9 -- expr_add_mul 0/25 -> 9/25 with NO regression
    (var_simple 25/25, expr_mod 25/25, expr_mul_div 10/10, expr_paren 16/16 all
    HOLD). lint_cross_op_attention PASS (no shared-head softmax hazard). Flag-OFF
    and non-campaign builds stay byte-identical (the SP_ADDR* bands are flag-gated,
    omitted -> smaller d_model -> golden ``2d227d48`` unchanged), so this default
    flip only takes effect INSIDE the campaign config where it was validated.

    BLOCKER-2 (downstream expr_add_mul roots) is INDEPENDENT and remains: the 16
    still-failing expr_add_mul diverge later at step-5 (MUL large-value corruption)
    / step-7 (framing on small-value progs; id816 4+5*2 reaches neural=14=expected
    but mis-frames an intermediate step) -- the discriminator+sharpener is now net
    +9 on its own and BLOCKER-2 is the remaining headroom (0/25 -> 9/25 done).

    Opt-out: ``C4_L8_OPERAND_SP_DISC=0`` restores the byte-identical pre-fix path
    (kept as a kill-switch so ``tools/flag_regression_gate.py --flag
    C4_L8_OPERAND_SP_DISC`` can still A/B it inside the campaign).
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_L8_OPERAND_SP_DISC", "1") != "0"


def cmp_byte0_se_recover_enabled() -> bool:
    """Return True iff the COMPARISON operand-A byte-0 SE_ALU recovery is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_CMP_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0 emission
    is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the cmp low-nibble /
    bool_and (nested-GT) fails — if_eq {412 7==45, 418 7==28, 420 28==12},
    if_lt 96<70 family, and bool_and 1071-1095 (6 fail) — whose comparison
    result decodes WRONG because operand A is crushed at the cmp-engine read.

    This is the comparison analog of :func:`mul_byte0_se_recover_enabled`. For
    a SUBSET of operands (value-dependent: e.g. 7, 57 crushed; 5, 96 NOT
    crushed — the same pipeline-path dependence the MUL/divmod crush shows),
    operand-A byte 0 — delivered from ``mem[SP]`` into ALU_LO/HI by the L8
    ``make_layer8_mem_to_alu_op`` head 5 (CLEAN +6.0 one-hot at the true
    nibble at blocks 11..13) — is CRUSHED ALL-NEGATIVE by the L10 ALU-clear
    (block 14): every cell ~-45, the true nibble ~-39 (spec_k=0
    ``tools/probe_cmp_se_recover.py``). The crush PERSISTS to the cmp-engine
    read (block ~19). The ``_layer10_alu_ordering_engine_rules`` /
    ``_layer10_alu_eq_engine_rules`` recompute the CMP cascade from the raw
    ``ALU_HI/LO`` (operand A) + ``AX_CARRY_HI/LO`` (operand B) at MARK_AX via
    per-nibble AND units whose ``-0.5``/``-0.8`` index BLOCKERS reject the
    operand-gather index-0 magnitude artifact. With ALU crushed to ~-45 every
    blocker term flips strongly POSITIVE (``-0.5 * -45 = +22.5``), so EVERY
    nibble unit clears threshold and the four CMP flags saturate to garbage
    (spec_k=0: CMP=[25947,17146,16997,32593] for 7==45). The
    ``ComparisonCombine`` then mis-decodes (eq_false -> 1, etc.). The clean
    operands the engines need ARE present at the AX row in
    ``SE_ALU_LO/HI`` (~+0.8 one-hot at the true nibble; the L9
    ``step_end_operand_relay`` mirror, written BEFORE the crush and surviving
    through block 19 — spec_k=0: SE_ALU_LO==0x7/SE_ALU_HI==0x0 for 7).

    FIX. When enabled, a forward-pass recover prepended to the efficient-mode
    L10 wrap (``make_efficient_l10_andorxor_wrap_op``) — running on the SAME
    block as the cmp engines, BEFORE they read, so NO physical block is added
    (the absolute-position lea contract holds) — does, on the cmp opcodes +
    MARK_AX row ONLY: (1) multiplicatively CLEAR the ``ALU_LO/HI`` band (so a
    crushed -45 floor AND an already-clean +6 one-hot both go to 0 —
    IDEMPOTENT, the non-crushed passing rows are not perturbed because their
    own clean operand is re-materialized identically), then (2) WRITE the
    clean operand-A one-hot from ``SE_ALU_LO/HI`` at the golden +6.0 magnitude
    the engines were tuned against. The engines then recompute the cascade
    from the SAME clean positive one-hot they see in the golden 35-token
    config — so NO threshold re-tuning is needed and the blockers reject the
    index-0 artifact exactly as designed. Operand B (``AX_CARRY``, clean) is
    untouched; ``CMP`` is never written here.

    DEFAULT ON. Opt-out via ``C4_CMP_BYTE0_SE_RECOVER=0`` restores the raw
    crushed ALU_LO/HI read (the byte-identical-OFF path: flag-OFF, or
    ``C4_NO_STACK0_EMIT=0``, are both byte-identical to golden ``7f6f2e5d``).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_CMP_BYTE0_SE_RECOVER`` can A/B it
    inside the campaign config.
    """
    return os.environ.get("C4_CMP_BYTE0_SE_RECOVER", "1") != "0"


def bitwise_byte0_se_recover_enabled() -> bool:
    """Return True iff the BITWISE (OR/XOR/AND) operand-A byte-0 SE_ALU recovery
    is active (DEFAULT ON in the campaign config — opt-out via
    ``C4_BITWISE_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0 emission
    is dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is
    byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (GPU-confirmed spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the 16-bit ``or_16bit`` /
    ``xor_16bit`` (and the per-nibble ``or``/``xor`` cases) byte-0 result is LOST
    EXACTLY when operand A's nibble == 0. ``and_16bit`` passes only because its
    nibbles are 0xF (the result then == operand A's surviving cell). The chain:
    operand A is delivered from ``mem[SP]`` into ``ALU_LO/HI`` by the L8
    ``make_layer8_mem_to_alu_op`` head 5 at the AX marker, but the L14 ALU-clear
    (block 19 of the campaign 55-block layout) CRUSHES ``ALU_LO/HI+0`` to ~0 —
    destroying the LEGIT A==0 one-hot. (Golden survives because its A==0
    magnitude ~11 >> the ~5.5 cell-0 artifact; the campaign mem-CAM delivers the
    legit A==0 at only ~5.5 ≈ the artifact, so the cleanup + L14 clear net it to
    zero.) The bitwise lookup post_op (``bitwise_rules``, gated on MARK_AX + the
    op flag, reading ``ALU`` × ``AX_CARRY``) then reads the crushed ALU at its
    own downstream block: the A==0 nibble is absent, the rule for that nibble
    never fires, and OUTPUT_LO/HI stays empty → the byte-0 token decodes 0x00.

    The clean operand A IS present at the bitwise compute row in
    ``SE_ALU_LO/HI`` (the L9 ``step_end_operand_relay`` mirror, written BEFORE
    the crush and surviving to the lookup block — spec_k=0: for ``or_16bit``
    ``SE_ALU_LO+0 == SE_ALU_HI+0 == ~0.85`` = operand A byte0 0x00; for
    ``xor_16bit`` ``SE_ALU_LO+15`` / ``SE_ALU_HI+0`` = 0x0F).

    FIX (the MUL/CMP byte-0 SE-recover precedent applied to the bitwise lookup):
    a forward-pass recover WRAPS the bitwise lookup post_op so it runs in the
    SAME block, BEFORE the lookup reads ``ALU`` (NO physical block is added — the
    absolute-position lea contract holds). On the OR/XOR/AND opcode + MARK_AX +
    crushed row ONLY it (1) multiplicatively CLEARS the ``ALU_LO/HI`` band (a
    crushed floor AND an already-clean one-hot both go to 0 — idempotent, the
    non-crushed passing rows are re-materialized identically), then (2) WRITES
    the clean operand-A one-hot from ``SE_ALU_LO/HI`` at the cleaned ~5.82
    magnitude the rescaled lookup cond weight (``30/5.82``) was tuned against.
    Operand B (``AX_CARRY``, clean) and the result band are untouched.

    DEFAULT ON. Opt-out via ``C4_BITWISE_BYTE0_SE_RECOVER=0`` restores the raw
    crushed ALU read. Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_BITWISE_BYTE0_SE_RECOVER`` can A/B
    it inside the campaign config.
    """
    return os.environ.get("C4_BITWISE_BYTE0_SE_RECOVER", "1") != "0"


def shift_output_byte0_clear_enabled() -> bool:
    """Return True iff the SHIFT (SHL/SHR) consumer OUTPUT byte-0 zero-default
    clear is active (DEFAULT ON in the campaign config — opt-out via
    ``C4_SHIFT_OUTPUT_B0_CLEAR=0``; only takes effect when the STACK0 emission
    is dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is
    byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (GPU-confirmed spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the ``test_shr`` smoke
    program (``84 >> 1 == 42``) decodes 0x00 instead of 0x2A. ``test_shl``
    (``21 << 1 == 42``) passes. Block-trace of the SHR MARK_AX compute row:
    operand A (0x54) survives intact in ``ALU_LO/HI`` and the L13 shift lookup
    (block 31 / logical L17 ``ALUShiftComposite``) CORRECTLY computes the result
    — ``OUTPUT_LO+10 += 2.0`` / ``OUTPUT_HI+2 += 2.0`` = 0x2A. BUT a SPURIOUS
    ``OUTPUT_LO+0 = 2.0`` / ``OUTPUT_HI+0 = 2.0`` byte-0 zero-default was written
    UPSTREAM at block 16 (logical L11) by the L11 attention's OUTPUT-byte-0
    broadcast on the SHR row (campaign cross-step OUTPUT-band leak; golden's
    block-16 OUTPUT band is empty so it never appears there). Because
    ``GEToBDConverter`` ADDS its result one-hot (``OUTPUT += indicator*2.0``)
    rather than overwriting, the stale 0x00 SURVIVES and TIES the true 0x2A at
    magnitude 2.0 — and the LM-head argmax breaks the tie toward the LOWER nibble
    index (cell-0), so both bytes decode 0x00 → result 0x00. ``shl`` is clean
    because its block-16 OUTPUT band is empty (no stale 0x00 to tie against).

    FIX (consumer-side, the ``GEToBDConverter`` output_amplitude precedent
    applied to the SHIFT writeback): a forward-pass clear WRAPS the
    ``ALUShiftComposite`` so it runs in the SAME block (NO physical block is
    added — the absolute-position lea contract holds). On the OP_SHL/OP_SHR +
    MARK_AX row ONLY it ZEROES the ``OUTPUT_LO``/``OUTPUT_HI`` band BEFORE the
    composite writes its result, so the stale upstream 0x00 default is removed
    and the shift's own ``+2.0`` one-hot stands unopposed. Idempotent for the
    passing ``shl`` (its OUTPUT band is already empty at the shift block — the
    clear is a no-op, then the composite re-writes the SAME 0x2A). No operand /
    ALU / carry / result band is touched besides OUTPUT; the composite's own
    fully SHL/SHR-gated writeback is unchanged.

    DEFAULT ON. Opt-out via ``C4_SHIFT_OUTPUT_B0_CLEAR=0`` restores the raw
    additive writeback (the byte-identical-OFF path: flag-OFF, or
    ``C4_NO_STACK0_EMIT=0``, are both byte-identical to golden ``7f6f2e5d``).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_SHIFT_OUTPUT_B0_CLEAR`` can A/B it
    inside the campaign config.
    """
    return os.environ.get("C4_SHIFT_OUTPUT_B0_CLEAR", "1") != "0"


def mul_multibyte_l19_boost_enabled() -> bool:
    """Return True iff the NARROWED MULTI-byte MUL byte-0 product is BOOSTED on
    the LITERAL-mul ``MUL+MARK_AX`` product row so it survives the block-34
    (logical L19) OUTPUT-byte-0 overwrite (DEFAULT ON in the campaign config —
    opt-out via ``C4_MUL_MULTIBYTE_L19_BOOST=0``; only takes effect under
    ``C4_NO_STACK0_EMIT=1`` + ``C4_MUL_BYTE0_SE_RECOVER=1``, so flag-OFF /
    non-campaign is byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the 6 multi-byte
    LITERAL-mul fails ``{127,130,134,139,141,144}`` = ``28*70, 92*40, 58*16,
    76*29, 91*81, 43*10`` — the byte-0 of a product whose byte 1 != 0. The
    ``mul_l19_product_boost`` cap handles the SINGLE-byte products (byte 1 ==
    0); these multi-byte ones keep a CLEAN byte-0 one-hot (mag ~14.3 through
    block 32) but at block 34 (logical L19) a PureFFN ADDS the BYTE-1 value into
    ``OUTPUT_LO`` at a DOMINANT magnitude (~180.8 at the byte-1 cell) on the
    MARK_AX row, so the LM-head argmax flips the byte-0 LOW nibble to the byte-1
    value (``1960 -> 1799 = 0x0707``).

    THE NARROWING (vs the dropped 2026-06-24 version, which gated only on
    ``res_b1 >= 0.5`` and so ALSO fired on ``var_mul``'s multi-byte MUL row —
    where the L11 wide_mul has ALREADY mis-fired on the crushed multi-local
    operand band so the boosted OUTPUT is garbage, REGRESSING ``var_mul``): the
    boost is gated to fire ONLY on the LITERAL-mul product row. The
    DISCRIMINATOR (probed spec_k=0 READING the EXACT ``state.x_bd_in`` the
    FlattenedALUMul composite receives, BUILT dims, on the MUL+MARK_AX
    first-fire row): the ``STACK0_B0_H1_PREV`` + ``STACK0_B0_H3_PREV`` cross-step
    carry band SUM (the ``C4_STACK0_B0_DUMP`` re-supply that only runs in a
    MULTI-step / multi-local frame) is

      * LITERAL mul (``return N*M;`` -> IMM/PSH/IMM/MUL, no ENT frame): 55..667
      * ``var_mul`` (``int a;int b;a=N;b=M;return a*b;`` -> ENT frame +
        LI-loaded operands -> cross-step STACK0 carry active): 7379 (rock-solid
        UNIFORM across var_mul_0..23).

    The band is NON-ZERO on the literal row at the COMPOSITE INPUT (block-30
    attention adds it) even though it is ~0 at the block-29 OUTPUT — so the gate
    reads ``x_bd_in`` and splits on MAGNITUDE: ``var_frame_carry < 2000`` (a >3x
    margin on BOTH sides: 667 << 2000 << 7379) marks the literal frame. So the
    narrowed gate adds that test to the multi-byte test -> the boost fires on
    the 6 literal fails ONLY and leaves every ``var_mul`` row untouched.

    FIX. ``_MulCombineStage`` marks the LITERAL multi-byte MUL+MARK_AX rows
    (``res_b1 >= 0.5`` AND ``var_frame_carry < 2000`` AND NOT the single-byte
    cap row); ``_GEToBDStage`` SCALES (25x, NO clear — the band is a clean
    one-hot, not a flood) the byte-0 ``OUTPUT_LO/HI`` band there so the true
    product LOW nibble out-votes the L19 add. byte-0 (``OUTPUT_LO/HI``) ONLY ->
    the byte-1 ``AX_FULL`` relay (a different emit row) is untouched; a uniform
    scale of a single-cell one-hot is argmax-invariant so the PASSING multi-byte
    literal muls (21*59, 65*98, ...) are byte-identical.

    DEFAULT ON. Opt-out via ``C4_MUL_MULTIBYTE_L19_BOOST=0`` restores the
    unboosted byte-0 (flag-OFF, or ``C4_NO_STACK0_EMIT=0``, or
    ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden — the boost
    path is campaign-only). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py``.
    """
    return os.environ.get("C4_MUL_MULTIBYTE_L19_BOOST", "1") != "0"


def cmp_eq_hinib_veto_enabled() -> bool:
    """Return True iff the EQ engine's HIGH-nibble artifact-veto is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_CMP_EQ_HINIB_VETO=0``; only takes effect when the STACK0 emission is
    dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is
    byte-identical to golden ``7f6f2e5d``).

    The residual cmp wall this lifts (the NON-crushed operand-gather index-0
    artifact, ``project_operand_gather_hybrid_encoding_is_cmp_alu_root``): the
    per-nibble EQ engine (:func:`_layer10_alu_eq_engine_rules`) writes the
    decisive ``eq_one`` 0x01 OUTPUT decode-margin push at the AX row for EQUAL
    operands. Its 4-way AND weighted ``ALU_HI`` at only ``0.1`` so the
    high-nibble term cannot VETO a mismatch: for ``if_eq_20: 28 == 12`` (A=0x1C
    / B=0x0C, sharing low nibble 0xC) the unit ``(h=0, l=0xC)`` fires on B's
    high nibble 0 even though A's high nibble is 1 — the index-0 magnitude
    artifact (``ALU_HI+0 ≈ +5.3``) HELPS the wrong unit clear threshold — so
    the 0x01 push lands and ``28 == 12`` mis-decodes to 1 (and the same for
    ``30 == 28``, ``40 == 35``, the documented residual EQ-false band).

    FIX (CONSUMER-SIDE, the lower-risk lever): copy the proven
    ``_layer10_alu_ordering_engine_rules`` ``hi_eq``/``lo_eq`` index-blocker
    pattern into the EQ engine's ``(h, l)`` units — a per-cell negative weight
    on every OTHER non-zero ``ALU_HI``/``ALU_LO`` index. When operand A's true
    high (or low) nibble is genuinely non-zero its strong ``+6.0`` one-hot is
    subtracted by the blocker (``-BLK * 6.0``) on every unit whose ``h``/``l``
    does NOT match A's nibble, so only the unit matching BOTH A's AND B's
    nibbles (i.e. A == B) survives. The weights/threshold were locked offline
    against the SAME golden HYBRID operand band the
    :class:`CmpOperandSeRecoverFFN` reconstructs in campaign (true nibble
    ``+6.0`` + index-0 artifact ``+5.3`` + cell-8/15 residues ``+0.45/+0.47``),
    exhaustively verified over the full ``0..99 × 0..99`` operand space: EVERY
    equal pair fires (margin ``+0.40``) and EVERY unequal pair is vetoed
    (worst-false margin on the if_eq corpus ``-0.87``). No ``CMP`` flag is
    touched (the ordering engine remains the sole CMP-flag writer), so
    lt/le/gt/ge/ne are structurally untouched; only the EQ ``eq_one`` OUTPUT
    push gains the high-nibble discrimination it lacked.

    DEFAULT ON. Opt-out via ``C4_CMP_EQ_HINIB_VETO=0`` restores the
    low-nibble-only EQ engine (byte-identical-OFF). Kept as a dedicated
    kill-switch so ``tools/flag_regression_gate.py --flag C4_CMP_EQ_HINIB_VETO``
    can A/B it inside the campaign config.
    """
    return os.environ.get("C4_CMP_EQ_HINIB_VETO", "1") != "0"


def cmp_gt_lo_margin_enabled() -> bool:
    """Return True iff the CMP equal-high-nibble GT lo-margin knock-down is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_CMP_GT_LO_MARGIN=0``;
    only takes effect when the STACK0 emission is dropped, i.e.
    ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is byte-identical to
    golden ``7f6f2e5d``).

    The residual cmp wall this lifts (#322, isolated CPU-autoregressive, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, BUILT dims, spec_k=0,
    ``tools/probe_gt_lo_margin.py``): the equal-high-nibble GT-TRUE comparisons
    ``if_gt 54>53 / 60>54 / 54>50`` (A.hi == B.hi, A.lo > B.lo, so ``lo_lt = 0``)
    decoded GT=0 because the live ComparisonCombine's ``(hi_eq AND lo_lt) -> GT=0``
    3-way override (fires iff ``MARK_AX + hi_eq + lo_lt > 2.5``) SPURIOUSLY tripped
    on ``hi_eq`` ALONE. In the campaign config the ``CmpOperandSeRecoverFFN``
    re-materializes a STRONGER operand-A one-hot than golden, so the shared
    ``C4_CMP_FLAG_MARGIN_FIX`` ``FLAG_EQ=0.45`` overshoots and lands ``hi_eq`` at
    ~1.86 at the decode row -- ABOVE its own (0.75, 1.5) single-flag ceiling --
    making ``1 + 1.86 + 0 = 2.86 > 2.5`` flip GT-true to GT=0
    (probed OUTPUT_LO@blk26 = [25.5@0, -15.9@1]).

    FIX: campaign-only, lower the ordering-engine ``FLAG_EQ`` to 0.30 so ``hi_eq``
    lands at ~1.24 (comfortably in-window). The spurious single-flag trip is gone
    (``1 + 1.24 + 0 = 2.24 < 2.5`` -> override OFF -> GT stays default=1, probed
    [0.6@0, 8.9@1] -> GT=1) while every INTENDED ``(hi_eq AND lo_lt)`` override
    still fires decisively (53>54 / 86<87: ``1 + 1.24 + 1.45 = 3.69 > 2.5``) and
    ``lo_lt``-alone still does NOT trip (50<44: ``1 + 0 + 1.45 = 2.45 < 2.5``).
    The ``lo_lt`` (CMP+3) write strength is UNTOUCHED so lt/le/ge margins are
    unchanged -- DISCRIMINATING, no zero-sum trade. ``lint_cross_op_ffn`` PASS
    (band-local to CMP; no shared OUTPUT/ALU read perturbed).

    DEFAULT ON. Opt-out via ``C4_CMP_GT_LO_MARGIN=0`` restores the 0.45 campaign
    value (byte-identical-OFF). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_CMP_GT_LO_MARGIN`` can A/B it inside
    the campaign config.
    """
    return os.environ.get("C4_CMP_GT_LO_MARGIN", "1") != "0"


def cmp_hi_lt_alu15_leak_guard_enabled() -> bool:
    """Return True iff the L10 ordering-engine ``hi_lt`` (CMP+0) blocker DROPS its
    ``ALU_HI+15`` (0xF address-high-nibble) veto term in the campaign config
    (DEFAULT ON — opt-out via ``C4_CMP_HI_LT_ALU15_GUARD=0``; only takes effect
    when the STACK0 emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF
    / non-campaign is byte-identical to golden).

    The residual cmp wall this lifts (#339, GPU full_trace + isolated
    intervention, campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``,
    BUILT dims, spec_k=0, ``tools/probe_ifvar_result_step.py`` +
    ``tools/probe_ifvar_alu15_intervene.py``): the if_var GT-FALSE comparisons of
    a LOADED variable -- ``if_var 23>62`` (id430), ``35>76`` (id433), both
    ``A.hi < B.hi`` so the genuine ``hi_lt`` GT-FALSE override must fire -- decoded
    GT=1 (returns 1 instead of 0).

    ROOT. On the LOADED-variable path the result step's operand-A ``ALU_HI``
    carries a SPURIOUS ``+6.5`` at cell 15 (the 0xF address-high-nibble leak from
    the ``LI``-load relay; IMPOSSIBLE for the corpus operands, which are 0..99 so
    ``A.hi <= 6``). The ``hi_lt`` unit is a balanced AND
    ``MARK_AX(6.0) + ALU_HI[a](0.5) + AX_CARRY_HI[b](6.0)`` with a ``-0.5`` blocker
    on every OTHER ``ALU_HI`` cell. The ``-0.5 * 6.5 = -3.25`` cell-15 blocker term
    drops the unit's pre-activation to ``6 + 3 + 6 - 3.25 - 0.2 = 11.55`` -- BELOW
    its ``13.22`` threshold -- so ``hi_lt`` (CMP+0) does NOT fire and the GT result
    defaults to 1. The PASSING LITERAL path (``ifGT 23>62``) has a clean
    ``ALU_HI[15] ~= 0.5`` (blocker ``-0.25``), so its ``hi_lt`` sum ``14.55`` clears
    threshold and GT decodes 0. (Probed: FAIL result-step AX-row
    ``OUTPUT_LO=[0.3@0, 9.4@1]`` -> byte 1 WRONG; literal ``[15.1@0, -5.3@1]`` ->
    byte 0 correct. CMP cascade at that row: FAIL ``[1.2@3]`` (lo_lt only, no
    hi_lt); literal ``[1.0@0, 1.2@3]`` (hi_lt AND lo_lt).)

    FIX. Campaign-only: DROP the ``ALU_HI+15`` term from the ``hi_lt`` blocker.
    Cell 15 (``A.hi == 0xF``, i.e. operand-A high byte >= 0xF0 == value >= 240) is
    UNREACHABLE for every single-byte comparison in the corpus, so the cell-15
    veto only ever fires on the spurious 0xF leak -- never on a legitimate
    operand. With cell 15 excluded the leak no longer penalizes ``hi_lt``: the
    FAIL sum recovers to ``6 + 3 + 6 - 0.2 = 14.8 > 13.22`` -> ``hi_lt`` fires ->
    GT-FALSE override lands -> result 0. INTERVENTION-VERIFIED to be DISCRIMINATING
    (``probe_ifvar_alu15_intervene.py``): zeroing ``ALU_HI[15]`` FLIPS id430/433 to
    0 (FIXED) while the GT-TRUE ``85>48`` (id427) HOLDS at 1 and the GT-FALSE
    literal HOLDS at 0 -- no zero-sum trade. Only the ``hi_lt`` (CMP+0) blocker is
    touched; ``lo_lt`` / ``hi_eq`` / ``lo_eq`` (CMP+3/+1/+2) write strengths and
    blockers are UNTOUCHED, so lt/le/ge/eq/ne margins are unchanged. Band-local to
    the CMP engine (no shared OUTPUT/ALU read perturbed).

    DEFAULT ON. Opt-out via ``C4_CMP_HI_LT_ALU15_GUARD=0`` restores the full
    cell-15 blocker (the byte-identical-OFF path: flag-OFF, or
    ``C4_NO_STACK0_EMIT=0``, are both byte-identical to golden). Kept as a
    dedicated kill-switch so ``tools/flag_regression_gate.py --flag
    C4_CMP_HI_LT_ALU15_GUARD`` can A/B it inside the campaign config.
    """
    if not no_stack0_emit_enabled():
        return False
    return os.environ.get("C4_CMP_HI_LT_ALU15_GUARD", "1") != "0"


def cmp_gt_lo_lt_hieq_guard_enabled() -> bool:
    """Return True iff the ``GT``/``GE`` ``(hi_eq AND lo_lt) -> 0`` override
    RAISES its firing threshold from 2.5 to 2.75 in the campaign config so a
    spurious ``lo_lt`` (CMP+3) alone can no longer trip the GT-result flip
    (DEFAULT ON; opt-out via ``C4_CMP_GT_LO_LT_HIEQ_GUARD=0``; only takes
    effect when the STACK0 emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``,
    so flag-OFF / non-campaign is byte-identical to golden).

    The residual cmp wall this lifts (the last 6 if_var fails 425/436/440/441/
    445/448 -- the SYMMETRIC GT-TRUE companion to the GT-FALSE
    ``cmp_hi_lt_alu15_leak_guard``; GPU full_trace + isolated CMP+3 intervention,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, BUILT dims,
    spec_k=0, ``tools/probe_ifvar_gt_true.py`` /
    ``probe_ifvar_gt_true_intervene.py`` / ``probe_gt_cmp_values.py``): the
    if_var GT-TRUE of a LOADED variable whose operands satisfy ``A.hi > B.hi``
    but ``A.lo < B.lo`` (e.g. ``66 > 24`` id436: ``A=0x42`` so ``A.lo=2 <
    B.lo=8`` while ``A.hi=4 > B.hi=1``) decoded GT=0 instead of 1.

    ROOT. GT is true (high nibble already orders ``A > B``) so the result must
    default to 1. But on the LOADED-variable path the campaign
    ``CmpOperandSeRecoverFFN`` re-materializes the operand-A one-hot strongly
    enough that the genuine ``lo_lt`` (``A.lo < B.lo``) flag lands at the
    GT-result-step AX row as ``CMP+3 ~= 1.67`` (clean GT-TRUE id427 ``85>48``
    has ``A.lo=5 > B.lo=0`` so NO lo_lt -> CMP+3 ~= 0). The live GT decoder is
    the ``ComparisonCombine`` ``(hi_eq AND lo_lt) -> 0`` 3-way override (block
    28 unit 9, OP_GT-gated, reading raw CMP at the result row): it fires iff
    ``marker(1) + CMP+1 + CMP+3 - 0.1*CMP+0 - 2.5 > 0``. With ``CMP+1`` (hi_eq)
    == 0 (A.hi != B.hi) and ``CMP+3`` == 1.67 alone the sum ``1 + 0 + 1.67 -
    2.5 = 0.17 > 0`` SPURIOUSLY trips the override -> GT flips to 0 (probed
    OUTPUT_LO@AX-row = ``[9.1@0, 0.6@1]`` -> result byte low-nibble 0 -> GT=0
    WRONG; clean GT-TRUE / GT-FALSE refs ``[0.1@0, 9.6@1]`` / ``[18.7@0,
    -9.0@1]``). The GENUINE ``(hi_eq AND lo_lt)`` GT-FALSE -- e.g. literal eq-hi
    ``35>43`` (id350), ``A.hi==B.hi`` so ``CMP+1 ~= 1.24`` AND ``CMP+3 ~=
    1.46`` -- sums ``1 + 1.24 + 1.46 = 3.70``, well above threshold, and MUST
    still fire (GT=0 correct).

    FIX. Campaign-only: RAISE the GT (and the symmetric GE) ``(CMP+1, CMP+3)``
    override threshold from 2.5 to 2.75 -- the SMALLEST raise that rejects the
    spurious lo_lt-alone trip (the loaded-var GT-TRUE result-step firing sum is
    just above 2.5; 2.75 drops it below) while the genuine hi_eq+lo_lt
    GT-FALSE override (firing sum ~3.7) STILL fires. A larger raise (3.0) ALSO
    works on the result-step decode but its bigger ``b_up`` shift (-0.5*S vs
    -0.25*S) perturbs the fp-accumulation at the saturated-tie step-0 operand
    leak of two ALREADY-MARGINAL GT-FALSE programs (``if_gt 8>27`` id360,
    ``bool_and`` id1074, both decided by hi_lt) and FLIPS their exit-code;
    2.75 leaves the hi_lt-decided cases' 3-way override firing UNCHANGED (no
    residual perturbation) so if_gt/if_lt/if_eq/bool_and all HOLD 25/25.
    INTERVENTION + REBUILD-VERIFIED discriminating
    (``probe_gt_combine_threshold_patch.py`` + ``run_1096_canonical
    --criterion exit_code``): 2.75 flips the 6 loaded-var GT-TRUE
    425/436/440/441/445/448 to GT=1 (FIXED, isolated 6/6) while the GT-TRUE
    pass id427, the GT-FALSE pass id430, the genuine eq-hi GT-FALSE
    id350/353/365/368, the eq-hi GT-TRUE id357/359/361, the hi_lt GT-FALSE
    id360, and bool_and all HOLD -- no zero-sum trade.
    Only the GT/GE ``(CMP+1, CMP+3)`` overrides are touched; the GT/GE 2-way
    (hi_lt) override and EQ/NE/LT/LE overrides are UNTOUCHED, so lt/le/eq/ne and
    the hi_lt-driven GT/GE-false margins are unchanged.

    DEFAULT ON. Opt-out via ``C4_CMP_GT_LO_LT_HIEQ_GUARD=0`` restores the 2.5
    threshold (the byte-identical-OFF path: flag-OFF, or
    ``C4_NO_STACK0_EMIT=0``, are both byte-identical to golden). Kept as a
    dedicated kill-switch so ``tools/flag_regression_gate.py --flag
    C4_CMP_GT_LO_LT_HIEQ_GUARD`` can A/B it inside the campaign config.
    """
    if not no_stack0_emit_enabled():
        return False
    return os.environ.get("C4_CMP_GT_LO_LT_HIEQ_GUARD", "1") != "0"


def cmp_combine_margin_enabled() -> bool:
    """Return True iff the ComparisonCombine result byte gets an explicit
    OUTPUT-HIGH-nibble CLAMP so the boolean result cannot leak an operand
    ``(hi<<4)`` high nibble (DEFAULT OFF; opt-in via ``C4_CMP_COMBINE_MARGIN=1``;
    only takes effect when the STACK0 emission is dropped, i.e.
    ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is byte-identical to
    golden ``91f55411``).

    The residual cmp wall this lifts (survey R5, ~150 programs — if_gt/if_lt/
    if_eq step-3, bool_and, func_max/min tail step-13). ROOT: the six
    comparison opcodes (EQ/NE/LT/GT/LE/GE) write their boolean RESULT to
    ``OUTPUT_LO`` (0 or 1) via the ComparisonCombine default+override banks, but
    the accompanying ``OUTPUT_HI_THIS_STEP+0`` write is only ``+2.0/S`` (default
    unit) and the OVERRIDE units write ``OUTPUT_LO`` ONLY — they never re-assert
    ``OUTPUT_HI``. On the both-nibbles-nonzero / loaded-var / re-materialized-
    operand paths a leaked operand HIGH nibble can survive in the OUTPUT_HI band
    at the compare decode row with a comparable (~1-point) amplitude, so the
    emitted result byte decodes as ``(hi<<4) | result`` instead of the clean
    ``result`` in {0, 1}.

    FIX (CLEAN MARGIN AT THE SOURCE, not a tail darken). A comparison result is
    PROVABLY in {0, 1}, so its byte's HIGH nibble is ALWAYS 0 — there is no
    operand-dependent case where a comparison opcode legitimately writes a
    non-zero ``OUTPUT_HI``. So per comparison opcode we add ONE campaign-only
    clamp unit that DARKENS every non-zero ``OUTPUT_HI_THIS_STEP+1..15`` nibble
    (strong negative) AND reinforces ``OUTPUT_HI_THIS_STEP+0`` (positive),
    out-voting any leaked operand high nibble at the decode row.

    SURGICAL GATE (2026-07). The clamp fires on
    ``MARK_SE_ONLY`` + ``SE_OP_<cmp>`` + **``SE_CMP_GROUP+0``** + the
    ``MARK_PC`` blocker (threshold 2.5 so ALL THREE positive markers are
    required, not just two-of-three). The added ``SE_CMP_GROUP+0`` condition is
    the FRESH-comparison discriminator: probing the SE step-end rows
    (``tools/_probe_cmp_se_row_disc.py``) shows ``SE_CMP_GROUP`` is ~0.94 on the
    genuine comparison-result decode row and ~0.00 on EVERY other SE row, while
    ``SE_OP_<cmp>`` ALONE can survive as a stale/cross-frame leak on a
    value-carrying row (a compare feeding a value — e.g. ``func_max``'s RETURN
    value ``0x63`` in a deeper 30-token frame). Without the group flag the clamp
    could clobber a legitimately-nonzero OUTPUT_HI high nibble on such a
    non-comparison / value row; with it the clamp fires ONLY when THIS step's
    opcode is a genuinely-relayed comparison whose result is provably in
    {0, 1}. This CANNOT weaken the intended fix — every genuine comparison
    decode carries ``SE_CMP_GROUP`` (the SAME L9 ``step_end_operand_relay`` that
    mirrors ``OP_<cmp> -> SE_OP_<cmp>``). Because the high nibble of a boolean
    result is invariantly 0, the clamp still cannot change any already-correct
    comparison; it only pulls a leaked ``(hi<<4)`` back to a clean low-nibble
    result. ``OUTPUT_LO`` (the actual 0/1 result) and every non-cmp opcode's
    OUTPUT_HI (which DO carry value high-nibbles) stay UNTOUCHED.

    DEFAULT OFF. Opt-in via ``C4_CMP_COMBINE_MARGIN=1`` (campaign only). Kept as
    a dedicated kill-switch so ``tools/flag_regression_gate.py --flag
    C4_CMP_COMBINE_MARGIN`` can A/B it inside the campaign config. Flag-OFF (or
    ``C4_NO_STACK0_EMIT=0``) is byte-identical to golden.
    """
    if not no_stack0_emit_enabled():
        return False
    return os.environ.get("C4_CMP_COMBINE_MARGIN", "0") == "1"


def func_lea_reread_bp_resharpen_enabled() -> bool:
    """Return True iff the L7 head-1 re-read-LEA BP-frame re-sharpen is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_FUNC_LEA_REREAD_BP_RESHARPEN=0``; only takes effect when the operand
    read is re-routed to ``mem[SP]`` i.e. ``C4_OPERAND_FROM_MEMSP=1``, so
    flag-OFF / non-campaign builds are byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (GPU-confirmed, agent adc27e branch ``worktree-agent-
    adc27e4fcbefaa6fd`` commit ``57363ac4``; re-verified spec_k=0, BUILT dims,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``;
    ``tools/_probe_lea_head1_attn.py``): the func_add/mul/square/max/min
    clusters (575-699) fail at the **re-read LEA** (the 2nd ``LEA n`` for a
    callee's later parameter — e.g. ``func_add`` step 11 ``LEA 16`` for &b,
    want ax=0xFFE0; value-INDEPENDENT).

    L7 head-1 (``layer7_operand_gather.head_1``) gathers the live frame BP/SP
    OUTPUT into the ALU at the LEA AX marker (operand-A address relay). Its K
    attends to BP/SP MARKER rows recency-ordered by the head's ALiBi slope
    (0.5). The BP marker row that actually CARRIES the live frame value is the
    one emitted on the frame-establishing ``ENT`` step (probed: ``OP_ENT=5.0``
    and ``OUTPUT_LO[0]~=14.9`` on that row ONLY; every other BP marker row is
    EMPTY — ``OP_ENT~=0``, ``OUTPUT_LO[0]~=1.0``). The FIRST LEA after the ENT
    works because the ENT-frame BP row is still the most-recent BP marker (it
    wins ALiBi recency). By the RE-READ LEA, three+ EMPTY BP marker rows
    (emitted by the intervening LI/PSH/LEA steps) sit between the query and the
    ENT-frame row, so ALiBi recency picks an EMPTY recent BP marker over the
    value-carrying ENT-frame row (probed: re-read row 471 raw 205 EMPTY beats
    value row 381 raw 160). The L8 LEA ALU is then EMPTY on the re-read → the
    address relay collapses to background residue (~15x attenuated).

    FIX (campaign-only, this flag): add ONE Q/K scoring slot to head-1, gated
    on the QUERY side by ``OP_LEA`` and on the KEY side by ``OP_ENT``. The slot
    contributes ``qw*kw * resid[q,OP_LEA] * resid[k,OP_ENT] * scale`` to the
    attention score, so it fires ONLY when the query is a LEA step AND the key
    is an ENT-frame BP row — re-pinning the re-read LEA's gather onto the LIVE
    ENT-frame BP row (it out-scores the empty recent markers). ALiBi recency
    breaks the tie between the two ENT-frame rows (caller `main` vs callee)
    toward the more-recent INNER frame, which is exactly the frame the callee's
    re-read LEA wants. On ADJ/ENT/non-LEA queries ``resid[q,OP_LEA]~=0`` so the
    slot is inert (ADJ's SP gather is untouched); on EMPTY BP / SP / non-ENT
    rows ``resid[k,OP_ENT]~=0`` so the slot adds nothing. This MIRRORS the
    first-LEA-after-ENT behaviour for the re-read LEA without disturbing any
    other gather.

    SHARED-HEAD DISCIPLINE: ``layer7_operand_gather.head_1`` is a golden head
    shared by ALL LEA/ADJ/ENT operand-A relays. The new Q/K writes land in a
    FRESH scoring slot (head_dim is 111; slots 0/1 are the only ones used by
    the head's Q/K scoring) and are ONLY emitted when this flag is on, so a
    flag-OFF build never touches ``W_q``/``W_k`` → byte-identical golden.
    ``tools/lint_cross_op_attention.py`` (MANDATORY for shared-head edits)
    gates the post-softmax head OUTPUT at OTHER-op / OTHER-context probe rows.

    DEFAULT ON. Opt-out via ``C4_FUNC_LEA_REREAD_BP_RESHARPEN=0`` (the
    byte-identical-OFF path: flag-OFF, or ``C4_OPERAND_FROM_MEMSP=0``, are both
    byte-identical to golden ``7f6f2e5d``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_FUNC_LEA_REREAD_BP_RESHARPEN`` can
    A/B it inside the campaign config.
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_FUNC_LEA_REREAD_BP_RESHARPEN", "1") != "0"


def mul_l19_flood_cap_enabled() -> bool:
    """Return True iff the MUL L19-EXPLODE flood cap fires on MODERATE-magnitude
    (not just >100) wide_mul OUTPUT floods (DEFAULT ON in the campaign config —
    opt-out via ``C4_MUL_L19_FLOOD_CAP=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1`` AND the MUL byte-0
    SE-recover is on, since it reuses that path's clear+rewrite machinery).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the 4 "L19-EXPLODE" mul
    fails 3*15, 11*11, 1*10, 8*30 whose product decodes to a HUGE garbage AX
    (e.g. 11*11 -> 2752768)):

    For a SMALL product the L11 ``efficient_l11_alumul_wrap`` wide_mul writes the
    CORRECT byte-0 product into OUTPUT_LO/HI but at a MODERATE flood magnitude
    (spec_k=0 block trace, 11*11: OUTPUT band sum ~41 at block 16 — the correct
    0x79 one-hot, but inflated). That ~41 band is BELOW the existing
    ``mul_byte0_se_recover`` flood-cap threshold (``output_hi_band > 100.0``), so
    the cap does NOT fire and the inflated raw OUTPUT survives. The downstream
    block-33 (logical L19) attention then AMPLIFIES that ~41 band to ~555
    (spec_k=0: ATTN in_LO=40.7 -> out_LO=555.6 at the MUL-AX row), spreading the
    band so the LM-head argmax flips to OUTPUT cell 0 == byte 0x00 and the AX
    high bytes pick up the flood -> the huge garbage AX. (Passing muls whose
    product writes a low ~14 band, e.g. 21*59, stay below the L19 amplification
    onset and decode correctly.)

    FIX. When this flag is on, the ``_MulCombineStage`` flood-cap threshold is
    LOWERED from ``> 100.0`` to ``> 4.0`` (well above the legitimate single-fire
    ``+2.0`` product write, below the ~41 small-product wide_mul flood) so the
    cap ALSO fires for the moderate-magnitude floods: it clears the inflated raw
    OUTPUT band on the OP_MUL+MARK_AX row and lets the clean ``+2.0`` byte-0
    product (the FlattenedALUMul schoolbook result, which is byte-0-correct for
    every mul-cluster operand — verified spec_k=0) survive. The normalized
    ``+2.0`` byte-0 product is below the L19 amplification onset, so the L19
    attention no longer explodes it. OUTPUT is byte-0 ONLY (byte 1 rides
    AX_FULL), so clearing+rewriting it never disturbs the byte-1 relay.

    DEFAULT ON. Opt-out via ``C4_MUL_L19_FLOOD_CAP=0`` restores the ``> 100.0``
    threshold (the byte-identical-OFF path: flag-OFF, or ``C4_NO_STACK0_EMIT=0``,
    or ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden). Kept as a
    dedicated kill-switch for ``tools/flag_regression_gate.py``.
    """
    return os.environ.get("C4_MUL_L19_FLOOD_CAP", "1") != "0"


def mul_l19_product_boost_enabled() -> bool:
    """Return True iff the capped-MUL byte-0 product is re-written at a DOMINANT
    OUTPUT amplitude that out-votes the block-33 (logical L19) zero-default add
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_MUL_L19_PRODUCT_BOOST=0``; only takes effect under
    ``C4_NO_STACK0_EMIT=1`` + ``C4_MUL_BYTE0_SE_RECOVER=1``, since it rides the
    same ``_MulCombineStage`` cap path).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the expr_add_mul cluster
    0/25 + the single-byte standalone-mul fails ``11*11`` etc.):

    ``mul_l19_flood_cap`` clears the L11 wide_mul OUTPUT flood on the
    OP_MUL+MARK_AX row and lets ``GEToBDConverter`` re-write the byte-0 product
    one-hot at the default ``+2.0`` amplitude. That fixed the explosion (band ->
    555) but NOT the SECOND L19 mechanism: the block-33 (logical L19) attention
    UNCONDITIONALLY ADDS ``+40`` into ``OUTPUT_LO[0]`` / ``OUTPUT_HI[0]`` (a
    broad "OUTPUT zero-byte default" copy, present on the MUL emit row in the
    depth>=1 stack contexts — expr_add_mul / expr_paren / expr_mul_div AND some
    standalone single-byte muls like ``11*11``). With the product at only ``+2.0``
    that ``+40`` cell-0 add OUT-VOTES the true product cell at the LM-head argmax,
    so the byte decodes to ``0x00`` (expr_add_mul ``5*2`` -> 0) or a stale value.
    Multi-byte products are NOT capped (band ~41 survives), so they already beat
    the ``+40`` add — only the capped single-byte products are starved.

    FIX. When this flag is on, the ``_GEToBDStage`` re-writes the freshly-cleared
    capped-row OUTPUT band (which by construction now holds ONLY the clean
    GEToBD product one-hot, the flood having been cleared) scaled to a DOMINANT
    magnitude (``MUL_L19_PRODUCT_BOOST`` = 50.0, > the L19 ``+40`` zero-default)
    so the true product cell beats the cell-0 add. Scoped to the cap rows ONLY
    (``output_clear_mask``), byte-0 ONLY (byte 1 rides AX_FULL, untouched), so it
    cannot disturb the byte-1 relay or any non-capped (multi-byte / non-MUL) row.

    DEFAULT ON. Opt-out via ``C4_MUL_L19_PRODUCT_BOOST=0`` restores the ``+2.0``
    re-write (flag-OFF, or ``C4_NO_STACK0_EMIT=0``, or
    ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden — the cap path
    is campaign-only). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py``.
    """
    return os.environ.get("C4_MUL_L19_PRODUCT_BOOST", "1") != "0"


def l15_lookup_cmp_veto_enabled() -> bool:
    """Return True iff the L15 ``li_lc_stack0_h0`` lookup head's slot-0
    discriminator VETOES on the comparison opcodes (OP_GT/OP_LT/OP_GE/OP_LE/
    OP_EQ/OP_NE), suppressing its spurious +40 CLEAN_EMBED->OUTPUT_LO copy on
    a COMPARISON step (DEFAULT ON; opt-out via ``C4_L15_LOOKUP_CMP_VETO=0``).

    The wall this lifts (verified spec_k=0, BUILT dims, GPU full_trace;
    ``bool_and`` 24/25, the sole fail id=1087 ``97>20 && 20>34``):

    L15 memory-lookup head 0 (``li_lc_stack0_h{0}``, value_scale=40) is the
    LI/LC + STACK0-POP load head. Its slot-0 discriminator fires the lookup
    on ``CMP+3`` (the "POP group" flag) -- but ``CMP+3`` is OVERLOADED: the L9
    cmp cascade ALSO drives it as the low-nibble-less-than flag, so on a
    GT-true comparison step (probed id=1087 step-3 AX row: ``CMP+3=1.45``,
    ``OP_GT+0=5.23``) the head MIS-FIRES, attends a cross-step ``CLEAN_EMBED``
    row, and copies ``+40.0`` into ``OUTPUT_LO+0``. That ``+40`` buries the
    clean GT result one-hot (``OUTPUT_LO+1=5.11`` for GT=1) at the LM-head
    argmax, so the comparison byte decodes ``0`` instead of ``1`` -> the
    full_trace step-3 AX diverges (``expected ax=1 got ax=0``). It fires only
    for diff-hi-nibble GT-true operands (the ones whose ``CMP+3`` lo_lt flag
    rides high enough to clear the head's threshold).

    FIX. The slot-0 discriminator already vetoes the non-load opcodes
    (``OP_JSR/OP_ENT/OP_LEA/OP_IMM`` at ``-1e6``) but NOT the comparison
    opcodes. A genuine LI/LC/POP load NEVER has a comparison opcode hot at its
    own marker (the opcode is OP_LI/OP_LC/OP_POP), whereas a comparison step
    has exactly one of OP_GT/OP_LT/OP_GE/OP_LE/OP_EQ/OP_NE one-hot at MARK_AX.
    Adding those six opcodes to the SAME ``non_load_suppression`` veto keeps
    the head silent on comparison rows (``OP_GT*-1e6 << CMP+3*50000``) and
    byte-identical on every real load row. Scoped to head 0's slot-0 Q only;
    no V/O / scale change, so the LI/LC/POP delivery is untouched.

    DEFAULT ON. Opt-out via ``C4_L15_LOOKUP_CMP_VETO=0`` restores the
    no-veto discriminator (the byte-identical-OFF path). Kept as a dedicated
    kill-switch for ``tools/flag_regression_gate.py`` and the flag-OFF
    golden byte-identity gate.
    """
    return os.environ.get("C4_L15_LOOKUP_CMP_VETO", "1") != "0"


def mul_se_recover_strict_onehot_enabled() -> bool:
    """Return True iff the ``MulOperandSeRecoverFFN`` rebuilds the recovered
    operand-A one-hot as a STRICT SINGLE-ARGMAX (one cell only) rather than the
    ``clamp(0,1) > 0.5`` multi-cell threshold (DEFAULT ON in the campaign config
    — opt-out via ``C4_MUL_SE_RECOVER_STRICT_ONEHOT=0``; only takes effect under
    ``C4_NO_STACK0_EMIT=1`` + ``C4_MUL_BYTE0_SE_RECOVER=1``, since the recover
    branch is campaign-only and crush-gated).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; var_mul 275-299 = 0/25, the
    MULTI-LOCAL ``a*b`` frame — DECISIVE control: the literal mul of the same
    values PASSES, var FAILS):

    In the deep multi-local var frame the L9 ``step_end_operand_relay`` SE_ALU
    mirror accumulates a SPURIOUS extra operand-A HIGH-nibble cell. Probed at the
    block-16 (logical L11) MUL row for var_mul 275 (23*47, a=23=0x17 so the true
    A high nibble is cell 1): ``SE_ALU_HI`` reads ``hot=[(1,0.7),(15,0.6)]`` — the
    true cell 1 (0.7) AND a stale cell 15 (0.6) — whereas the PASSING literal mul
    reads a clean ``hot=[(1,0.7)]``. The ``MulOperandSeRecoverFFN`` rebuilds the
    recovered one-hot with ``clamp(SE_ALU_HI,0,1) > 0.5``, which keeps BOTH cells
    (0.7>0.5 AND 0.6>0.5) -> the recovered ``ALU_HI`` becomes ``[(1,6.0),(15,6.0)]``
    (TWO non-zero A high-nibble cells). The width=2 wide_mul's per-rule operand-A
    artifact BLOCKER (``operand_a_artifact_blocker_weight=3.0`` on every OTHER
    non-zero A cell) then TRIPS on the true rule: the 5-way AND drops below its
    19.0 threshold and the wide_mul fires on NO rule -> ``MUL_RESULT_HI``/OUTPUT
    stay 0 (probed: var block-16 OUTPUT band == 0 vs lit == 41.6). The only thing
    left is the downstream ``+2.0`` GEToBD/L16 default at the WRONG nibble, so the
    product decodes to garbage (id275 23*47=1081 -> neural 73).

    FIX. When this flag is on, the recover rebuilds the operand-A one-hot from the
    SE mirror as a STRICT single-argmax (the single largest cell only). cell 1
    (0.7) beats the stale cell 15 (0.6) so the recovered ``ALU_HI`` is a clean
    ``[(1,6.0)]`` one-hot -> the wide_mul fires on the true rule -> the correct
    product band (41.6) is computed and survives. This is byte-IDENTICAL to the
    current ``clamp>0.5`` behaviour for any case whose SE mirror is already a
    single cell (the literal mul + every PASSING crushed mul), and only changes
    the multi-cell var-frame case the blocker was silently eating.

    DEFAULT ON. Opt-out via ``C4_MUL_SE_RECOVER_STRICT_ONEHOT=0`` restores the
    ``clamp>0.5`` rebuild (flag-OFF, or ``C4_NO_STACK0_EMIT=0``, or
    ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden — the recover
    path is campaign-only). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py``.
    """
    return os.environ.get("C4_MUL_SE_RECOVER_STRICT_ONEHOT", "1") != "0"


def divmod_axcarry_clear_enabled() -> bool:
    """Return True iff the divmod writeback CLEARS the AX_CARRY (divisor) band
    at the divmod AX row (DEFAULT ON in the campaign config — opt-out via
    ``C4_DIVMOD_AXCARRY_CLEAR=0``; only takes effect when the STACK0 emission
    is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; ~10 div/mod fails of the
    "got_ax == divisor" pattern, e.g. 1162/37 -> 37, 2009/43 -> 43, 106/4 ->
    4, 794/49 -> 49):

    The ``FlattenedDivMod`` (block 28 / logical L14) computes the CORRECT
    quotient/remainder into OUTPUT_LO/HI at +2.0 (the long-division compute is
    bit-exact for every residual operand — there is NO divider arch-wall). But
    for a MULTI-BYTE dividend the L10 ALU-clear crushes ALU_LO/HI all-negative
    at the divmod row (probe ``tools/probe_divmod_l20_src.py``: 1162/37 ->
    ALU == -45 uniform). The downstream L20 ``layer16_lev_routing`` frame-relay
    then MIS-FIRES on that crushed-ALU divmod row and MATERIALIZES the
    AX_CARRY band (the DIVISOR, e.g. 0x25 == 37) into OUTPUT at +4.4, out-
    voting the +2.0 quotient — so the emitted AX byte is the divisor, not the
    quotient. (The PASSING divmod rows keep a clean positive ALU one-hot and
    the L20 relay stays silent — verified pass-vs-fail discriminator.)

    The divmod has already CONSUMED the divisor (operand-B byte 0 read from
    AX_CARRY_LO/HI into GE NIB_B by ``BDToGEConverter`` at the divmod block
    input) by the time the writeback runs, so the AX_CARRY band is dead at the
    divmod AX row from L14 onward. Clearing it removes the divisor source the
    L20 relay leaks, so the +2.0 quotient survives to the AX emit. Verified
    (hook ``tools/probe_divmod_alurestore_test.py MODE=clearaxc``): 1162/37,
    106/4, 2009/43, 794/49 -> CORRECT quotient; PASSING rows (843/31, 176/4,
    54/7) UNCHANGED. The 0xE8 (744) and L18 slam patterns are SEPARATE
    downstream-tail corruptors (out of the divmod writeback's reach).

    AX_CARRY is only read downstream by opcode-gated ops (OP_MUL at L12,
    OP_LI/LC/SI/SC at L13/L15) whose gates are inactive on a divmod row, and
    by the L14 ``AX_CARRY_HI+15`` NOT-blocker (cleared band == "not 15" ==
    safe), so the divmod-row clear touches no other op/config.

    DEFAULT ON. Opt-out via ``C4_DIVMOD_AXCARRY_CLEAR=0`` restores the raw
    AX_CARRY passthrough (the byte-identical-OFF path: flag-OFF or
    ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden ``4958b35b``).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_DIVMOD_AXCARRY_CLEAR`` can A/B it
    inside the campaign config.
    """
    return os.environ.get("C4_DIVMOD_AXCARRY_CLEAR", "1") != "0"


def divmod_stack0_byte1_clear_enabled() -> bool:
    """Return True iff the divmod writeback CLEARS the STACK0_BYTE_VAL_1
    (dividend byte-1 carrier) band at the divmod AX row (DEFAULT ON in the
    campaign config — opt-out via ``C4_DIVMOD_STACK0_BYTE1_CLEAR=0``; only
    takes effect when the STACK0 emission is dropped, i.e.
    ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the residual ~4 div/mod
    fails of the "L18 slam" pattern the ``divmod_axcarry_clear_enabled`` note
    explicitly flags as a SEPARATE downstream-tail corruptor, e.g. 364/14 ->
    1, 1132/33 -> 4, 268%17 -> 1, 428%16 -> 1):

    The ``FlattenedDivMod`` (block 28 / logical L14) computes the CORRECT
    quotient/remainder into OUTPUT_LO/HI at +2.0 (the long-division compute is
    bit-exact — verified blk28 OUTPUT == the right answer for every fail). But
    for a MULTI-BYTE dividend the divmod AX row still carries the dividend's
    byte-1 in the ``STACK0_BYTE_VAL_1_LO/HI`` carrier (the band the
    ``BDToGEConverter`` cummax-gathers operand-A byte 1 from). The L18 (block
    32) ``layer14_mem_generation`` ADDRESS head 1 — whose V/O slots 32+/48+
    read ``STACK0_BYTE_VAL_1`` into OUTPUT_LO/HI to generate the SI/SC store
    address byte 1 — MIS-FIRES on that crushed divmod AX row and MATERIALIZES
    the dividend byte 1 into OUTPUT at +13.9, overwriting the +2.0 quotient ->
    the emitted AX byte is ``hi(dividend)`` (e.g. 0x04 for 1132, 0x01 for 364),
    not the quotient. (PASSING divmod rows have ``STACK0_BYTE_VAL_1 == 0`` —
    single-byte dividend, e.g. 89%10 hi=0 — so head 1 reads zeros and stays
    silent: the clean pass-vs-fail discriminator, verified
    ``tools/probe_divmod_fast.py``.)

    The divmod has already CONSUMED the dividend byte 1 (the
    ``BDToGEConverter`` cummax-gathered it into GE operand-A positions 2/3 at
    the divmod block INPUT, BEFORE this writeback runs) by the time the
    writeback executes, so the ``STACK0_BYTE_VAL_1`` band is dead at the divmod
    AX row from this block onward. Clearing it here — gated on the SAME
    divmod-AX ``opcode_mask`` the OUTPUT write + the AX_CARRY clear use —
    removes the byte-1 source the L18 mem-gen head 1 leaks, so the +2.0
    quotient survives to the emit. (The L18 head reads ``STACK0_BYTE_VAL_1``
    only for SI/SC store-address byte-1 generation, whose opcode rows are not
    DIV/MOD, so the divmod-row clear touches no other op/config — exactly the
    ``divmod_axcarry_clear`` precedent applied to the byte-1 carrier instead of
    AX_CARRY.)

    DEFAULT ON. Opt-out via ``C4_DIVMOD_STACK0_BYTE1_CLEAR=0`` restores the raw
    ``STACK0_BYTE_VAL_1`` passthrough (the byte-identical-OFF path: flag-OFF or
    ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden ``7f6f2e5d``).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_DIVMOD_STACK0_BYTE1_CLEAR`` can
    A/B it inside the campaign config.
    """
    return os.environ.get("C4_DIVMOD_STACK0_BYTE1_CLEAR", "1") != "0"


def addsub_output_boost_enabled() -> bool:
    """Return True iff the imperative AddSub5StageBlock writes its byte-0
    OUTPUT_LO/HI at a DOMINANT amplitude (DEFAULT ON — opt-out via
    ``C4_ADDSUB_DUMP_BOOST=0``).

    The bug this lifts (verified spec_k=0, BUILT dims, full_trace 0..99):
    the imperative ``_AddSubGEToBD`` stage (block 10 / logical L8) computes
    byte-0 add/sub CORRECTLY from the ``_clean_onehot``-thresholded operands
    and writes the result one-hot into ``OUTPUT_LO/HI`` at amplitude 2.0.
    The DOWNSTREAM block 11 (logical L9) then floods ``OUTPUT_LO`` with a
    near-uniform ~83.5 pedestal PLUS the documented L9 ``ALU_LO -> OUTPUT_LO``
    operand leak (a peak at operand-A's low-nibble lane). With the block-10
    write only at 2.0 that spurious leaked lane out-votes the correct result
    lane by a tiny margin (~1.2 out of ~96), flipping the argmax. This is the
    SAME "downstream L9 ALU_LO->OUTPUT_LO leak" the ``DeclarativeAddSubBlock``
    out-votes with its dominant amplitude (see ``addsub_declarative_enabled``);
    we apply the identical remedy on the imperative path, which (unlike the
    declarative wrap) already reads CLEAN operands so it does not regress the
    multi-byte cascade.

    When enabled, ``_AddSubGEToBD`` passes ``output_amplitude`` to its
    ``GEToBDConverter`` so the byte-0 OUTPUT one-hot is written at the boosted
    magnitude; the carry/borrow CARRY writes and byte-1 AX_FULL staging are
    UNCHANGED. Flag-OFF restores the amplitude-2.0 write (byte-identical to
    HEAD). Measured: add 40->XX, sub 44->XX on full_trace ids 0-99, smoke
    51/0, mul/div/mod guards held.
    """
    return os.environ.get("C4_ADDSUB_DUMP_BOOST", "1") != "0"


def addsub_declarative_enabled() -> bool:
    """Return True iff efficient-mode L8 ADD/SUB uses the DECLARATIVE wrap
    (DEFAULT OFF — opt-in via ``C4_ADDSUB_DECLARATIVE=1``).

    When enabled, ``make_efficient_l8_addsub_wrap_op`` installs the
    ``DeclarativeAddSubBlock`` composite (two ``PureFFN`` passes, both lowered
    purely from ``wide_alu_dsl.wide_add_rules`` + ``wide_sub_rules``) instead
    of the imperative ``AddSub5StageBlock`` (BDToGEConverter + GE add/sub
    layers + GEToBDConverter). The declarative band writes OUTPUT at a
    DOMINANT amplitude (the folded-in +5 fix) so the correct result cell
    out-votes the downstream L9 ALU_LO->OUTPUT_LO leak. The two passes run
    sequentially inside ONE block (carry CARRY+0 cascades lo->hi internally),
    keeping the physical block count identical to the imperative.

    DEFAULT OFF (2026-06-14): the declarative byte-0 add/sub + carry/borrow
    flags are byte-identical to the imperative on CLEAN one-hot operands
    (tests/test_addsub_decl_wrap.py: full 0..255 value grid) AND in isolation
    on the real model. BUT the live MARK_AX operand bands are DIRTY (operand A
    in ALU_LO/HI arrives ~6.0 at the true nibble plus a ~5.4 index-0
    magnitude artifact — the documented operand-gather hybrid encoding). On
    those dirty bands the lo pass's CARRY+0 over-accumulates (multiple partial
    rule firings) and the hi pass over-fires, so multi-byte add/sub (the
    add_16bit / sub_16bit / *_cascade smoke + 32-bit negative sub) regress.
    The imperative ``BDToGEConverter._clean_onehot`` thresholds operands to
    0/1 BEFORE computing, which is what makes it dirty-operand robust; the
    declarative wrap needs an analogous operand-cleanup pre-pass (the same
    technique ``make_efficient_l10_andorxor_wrap_op`` uses for bitwise) as a
    FOLLOW-UP wave before it can be the default. See
    docs/ADDSUB_DSL_MIGRATION_2026_06_14.md.
    """
    return os.environ.get("C4_ADDSUB_DECLARATIVE", "0") == "1"


def operand_from_memsp_enabled() -> bool:
    """Return True iff the binary-op operand-A read is routed to ``mem[SP]``
    via the L4 SP-to-ADDR_KEY + L8 mem-to-ALU memory-attention CAM
    (DEFAULT OFF — opt-in via ``C4_OPERAND_FROM_MEMSP=1``).

    This is the STACK0-emission-drop *prerequisite*. The binary-op operand-A
    (the top-of-stack value that ADD/SUB/AND/OR/XOR/EQ/NE/SI/SC read) is
    sourced today from the EMITTED STACK0 byte-0 token via L7 head 0's
    ``STACK0_BYTE0``-keyed gather. When ``C4_NO_STACK0_EMIT=1`` drops the
    STACK0 register block from the step, that emitted token is gone and the
    operand read is starved (-130 full_trace, see
    ``docs/NO_STACK0_EMIT_MEASURED_2026_06_16.md``).

    When this flag is on:
      * ``make_layer4_sp_to_addr_key_op`` (L4 attn heads 2-3) stages the live
        SP value into the ADDR_KEY band at the AX marker (Q-side address).
      * ``make_layer8_mem_to_alu_op`` (L8 attn head 5) reads ``mem[SP]`` byte 0
        via the ADDR_KEY CAM (most-recent matching ``MEM_STORE`` wins via
        ALiBi recency) and writes ``ALU_LO/HI`` at the AX marker — the exact
        source/dest L7 head 0 uses today, one block later, sourced from
        memory instead of the emitted token.
      * L7 head 0's ``STACK0_BYTE0`` -> ALU write is suppressed (the L8 head
        is the sole operand-A source; head 1's LEA/ADJ/ENT frame relay is
        untouched).

    This implements Phase 1 of ``docs/STACK0_VIA_MEM_ATTENTION_PLAN.md``
    (the L4 + L8 ops were built but left ``enable=False`` / unvalidated).
    DEFAULT OFF = byte-identical to HEAD. Flip on TOGETHER with
    ``C4_NO_STACK0_EMIT`` to eliminate the Root #2 framing-drift class without
    starving the operand read. With ``C4_NO_STACK0_EMIT=0`` (STACK0 still
    emitted) it is a no-regression equivalence check (operand now read from
    ``mem[SP]`` instead of the still-emitted token).

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_OPERAND_FROM_MEMSP=0`` still opts out.
    """
    if os.environ.get("C4_OPERAND_FROM_MEMSP") is None and campaign_enabled():
        return True
    return os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"


def no_stack0_emit_enabled() -> bool:
    """Return True iff the STACK0 register block is dropped from the emitted
    step (DEFAULT OFF — opt-in via ``C4_NO_STACK0_EMIT=1``).

    Mirror of ``l0_ops._no_stack0_emit`` (kept here so the L1 positional-gate
    fix and other ops can consult the same env switch without importing l0).
    When on, the 35-token step collapses to 30 tokens (STACK0 marker + 4 value
    bytes dropped), so the MEM register block shifts 5 positions earlier:
    the d=6-from-BP slot that used to be STACK0 byte 0 is now MEM addr byte 0.
    The ``STACK0_BYTE0`` positional flag (L1 FFN unit 0, fired at d=6 from BP)
    must therefore be neutralized so it does not misfire onto the MEM addr
    byte — see ``l1_ops._threshold_ffn_rules``.

    This flag does NOT itself drop the emission (that machinery — the L0
    marker-transition chain, ``Token.STEP_TOKENS``, the DraftVM oracle, the
    decode offsets — lives on the ``proto/drop-stack0-emit-measure`` branch).
    It is the per-op consultation point so flag-off is byte-identical.

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_NO_STACK0_EMIT=0`` still opts out.
    """
    if os.environ.get("C4_NO_STACK0_EMIT") is None and campaign_enabled():
        return True
    return os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"


def si_store_addr_enabled() -> bool:
    """Return True iff the SI/SC store address-provenance CAM head is active
    (DEFAULT OFF — opt in via ``C4_SI_STORE_ADDR=1``; only meaningful in the
    30-token campaign config).

    ROOT (the ``var_mul``/multilocal-LI store-provenance TWO-ROOT wall, verified
    2026-06-18 by built-layout probe + hook-inject; ``var_mul`` id275
    ``int a;int b;a=23;b=47;return a*b;`` fails: ``LI a`` returns b's 47 not a's
    23). The existing L15 head-0 LI value-load CAM (``C4_L15_LI_ADDR_CAM``)
    discriminates two same-frame locals by matching ``ADDR_B0_LO/HI`` on each
    store's VALUE row and copies that row's ``CLEAN_EMBED`` into OUTPUT. But
    (Root 1) the SI/SC store VALUE rows carry ``ADDR_B0==0x00`` (address-blind)
    so the candidates tie and ALiBi recency picks the last store (b); AND (Root
    2) even with the correct ``ADDR_B0`` injected the value rows carry an
    address-like ``CLEAN_EMBED`` pattern, NOT the clean value 23. Provenance is
    missing on BOTH axes on the exact rows the CAM reads.

    THE LEVER (built-layout probe, ``tools/_probe_si_store_addr.py``, campaign
    config, id275): the clean address AND clean value BOTH live on the store's
    AX-MARKER row (a-marker: ``ADDR_B0=0xE8`` + ``AX_CARRY=0x17``=23; b-marker:
    ``ADDR_B0=0xE0`` + ``AX_CARRY=0x2F``=47), and the ``LI a`` query row carries
    its target address in ``AX_CARRY=0xE8`` (NOT in ``ADDR_B0``, which is 0x00
    there). So a DIRECT address-keyed CAM bypasses BOTH roots: LI-query row Q
    keyed on its ``AX_CARRY`` (target addr) -> SI/SC store AX-marker K keyed on
    the marker's ``ADDR_B0`` (store addr) -> V copies the marker's ``AX_CARRY``
    (the clean stored value) -> O writes the LI result into OUTPUT.

    Implemented as a dedicated flag-gated L15 attention head
    (``layer15_si_store_addr_cam``) appended to the L15 head layout only when the
    flag is on, so flag-OFF keeps the L15 head count + weights byte-identical to
    golden (the CAM never installs). Campaign-only: the ``AX_CARRY``/``ADDR_B0``
    marker signals it keys on are produced by the 30-token MEM-from-SP path;
    golden (35-token, flag-OFF) is byte-identical.

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_SI_STORE_ADDR`` value wins (so
    ``C4_SI_STORE_ADDR=0`` opts out even under the campaign).
    """
    explicit = os.environ.get("C4_SI_STORE_ADDR")
    if explicit is None and campaign_enabled():
        return True
    return explicit == "1"


def var_three_li_enabled() -> bool:
    """Return True iff the L15 head-0 OP_SI/OP_SC store-row VETO is active
    (DEFAULT OFF — opt in via ``C4_VAR_THREE_LI=1``; only meaningful in the
    30-token campaign config where the stray relay flag appears).

    ROOT (verified spec_k=0, BUILT dims, campaign config, teacher-forced oracle
    tape, ``tools/_probe_vt_head0_sistore.py``; var_three id300
    ``int a;int b;int c;a=29;b=6;c=20;return a+b+c;`` diverges at the ``SI b``
    store step): the ``SI b`` store's AX byte-0 emit row carries a STRAY
    ``OP_LI_RELAY==1.0`` alongside ``OP_SI==5.23`` (unlike the ``SI a`` / ``SI c``
    store rows, which carry ``OP_SI`` alone). The L15 memory-lookup head 0
    (``li_lc_stack0_h0``, value_scale=40) slot-0 discriminator fires the lookup
    on ``OP_LI_RELAY`` (Q weight ``lookup_bias``=2e5) and already VETOES the
    non-load opcodes ``OP_JSR/OP_ENT/OP_LEA/OP_IMM`` at ``-1e6`` — but NOT the
    STORE opcodes ``OP_SI``/``OP_SC``. So on the ``SI b`` row the stray relay
    clears the veto, head 0 fires OFF-self (attends a cross-step ``CLEAN_EMBED``
    row whose value is 0), and copies ``+0`` into ``OUTPUT``: the store value
    byte 6 decodes ``0`` instead of ``6`` -> the frame desyncs at ``SI b`` and
    the whole var_three program drifts (probe: ``SI b`` row got=0x0000 want
    0x0006; the ``SI a``/``SI c`` rows head0 self_p=1.000, correct).

    FIX. A genuine LI/LC LOAD never has ``OP_SI``/``OP_SC`` hot at its own
    marker (the opcode is OP_LI/OP_LC), whereas an SI/SC store step has exactly
    one of ``OP_SI``/``OP_SC`` one-hot at MARK_AX. Extend the SAME slot-0
    ``non_load_suppression`` veto (already covering OP_JSR/ENT/LEA/IMM and — via
    ``C4_L15_LOOKUP_CMP_VETO`` — the six comparison opcodes) to ``OP_SI`` and
    ``OP_SC``: ``OP_SI*-1e6`` dominates the stray ``OP_LI_RELAY*2e5``, keeping
    head 0 silent (self-firing) on every store row so the store value survives.
    Scoped to head 0's slot-0 Q only; no V/O / scale change, so the LI/LC/POP
    delivery on real load rows is untouched (byte-identical there —
    OP_SI==OP_SC==0 on every load row). Additive slot-0 Q writes, gated by the
    flag, so flag-OFF omits them entirely -> byte-identical to golden.

    DEFAULT OFF. Reads its OWN env var (NOT floored ON by ``C4_CAMPAIGN``) so
    the campaign default set stays untouched until this fix is proven
    net-positive + HOLD-clean; opt in via ``C4_VAR_THREE_LI=1`` (only takes
    effect under the campaign / MEM-from-SP path, since the stray relay flag is
    produced only by the 30-token frame — flag-ON at golden 35-tok is
    byte-identical). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py`` and the flag-OFF golden byte-identity gate.
    """
    return os.environ.get("C4_VAR_THREE_LI", "0") == "1"


def absdiff_fix_enabled() -> bool:
    """Return True iff the absdiff arg-b LI value byte-0 LO-nibble
    de-contamination is active (DEFAULT OFF — opt in via ``C4_ABSDIFF_FIX=1``;
    only meaningful in the 30-token campaign config).

    ROOT (verified spec_k=0 AUTHORITATIVE AR verdict via ``tools/cpu_full_trace``
    AND teacher-forced ``tools/_probe_vt_head0_sistore.py`` + per-block
    OUTPUT_LO attribution, campaign config, BUILT dims; absdiff_0 id1046
    ``int abs_diff(int a,int b){if(a>b)return a-b;return b-a;} main(){return
    abs_diff(16,85);}`` diverges at step 12, the 2nd LI = the arg-``b`` deref
    inside the ``if(a>b)`` comparison): ``exp=(pc=74,ax=85) got=(pc=74,ax=80)``.
    PC is correct; only AX byte-0's LO nibble is wrong (0x55 -> 0x50).

    Arg ``a`` lives at BP-relative 0xFFE8 (addr byte-0 = 0xE8, lo-nibble 8) and
    arg ``b`` at 0xFFE0 (addr byte-0 = 0xE0, lo-nibble 0). The L15 head-0
    memory-lookup value delivery (value_scale=40) correctly content-addresses
    ``b``'s pushed value row (CLEAN_EMBED=0x55) and delivers OUTPUT_LO+5 = +40,
    but at this ZERO-lo-nibble arg address the same value row's CLEAN_EMBED_LO
    ALSO carries the ADDRESS lo-nibble 0, so head-0 delivers OUTPUT_LO+0 = +67.5
    as well. Attribution proves BOTH writes originate in block 35 (= logical
    L15). At the LM head the byte-0 token 0x50 (hi=5 correct, lo=0 contaminated)
    wins at logit 542 over the correct 0x55 at logit 403 (margin ~139), so byte-0
    decodes ``value & 0xF0``. This affects ALL 25 absdiff cases (the arg-``b``
    deref precedes the branch, so both taken and not-taken paths corrupt).

    This is the documented L15 head-0 LI value-CAM zero-lo-nibble aliasing WALL:
    the value nibble and the address nibble ALIAS in CLEAN_EMBED at a
    zero-lo-nibble arg address, and var_simple's ``x`` at BP+0 (addr 0x00) shares
    the exact separating signature, so every head-0 slot discriminator that
    flips absdiff also regresses var_simple (blueprint, 3 attempts). The clean
    fix needs an FFN-materialized (committed-zero-address) indicator dim so the
    AND is done outside the bilinear head — a multi-block build, NOT a single
    additive slot. DEFAULT OFF; flag-OFF registers NO rules -> byte-identical to
    golden ``b1dcae63``. Kept as a dedicated kill-switch for the flag-regression
    gate and the byte-identity gate.

    STATUS (measured, NOT net-positive — kept DEFAULT-OFF): the L10 tail-block
    ``l10_absdiff_argb_li_lo`` corrector (a per-nibble OUTPUT_LO winner-take-all
    gated on the head-0-delivered value LO nibble ``OUTPUT_LO+k`` + the 0xE0
    address discriminator) was BUILT and measured on the authoritative AR
    verdict (``cpu_full_trace --spec-k 0``). It REGRESSES its own target:
    absdiff_0 goes from ``div_step=12`` (byte-0 lo nibble) OFF to ``div_step=1``
    (``exp=(pc=210,ax=0) got=(pc=210,ax=2)``) ON — the ``OUTPUT_LO+k`` gate is a
    RESIDUAL value that is nonzero on non-value-delivery rows too, so the
    corrector over-fires at an earlier step. This CONFIRMS the wall: without an
    FFN-materialized indicator dim, the value LO nibble cannot be re-stamped
    cleanly (it aliases with residual OUTPUT_LO mass on other rows AND with the
    address-0 contaminant on the value row). The corrector stays flag-OFF; do
    NOT flip ON. Next session: build the (committed-zero-address) indicator dim
    at L13/L14 and gate a MARK_AX-only head-0 slot on it (blueprint).
    """
    return os.environ.get("C4_ABSDIFF_FIX", "0") == "1"


def absdiff_ret_byte1_enabled() -> bool:
    """Return True iff the absdiff / func-return AX byte-1 OUTPUT_LO stale-marker
    de-contamination is active (DEFAULT OFF — opt in via
    ``C4_ABSDIFF_RET_BYTE1=1``; only meaningful in the 30-token campaign config).

    NOTE (2026-07 round-1 land): held DEFAULT-OFF — flipping it ON regressed the
    ``test_lea_basic`` smoke (``ENT; IMM 0; LEA 2; EXIT`` -> address decodes 0):
    the ``ABSDIFF_RET_LEAK`` crush-band / OP_LEV-return selector over-fires on
    the ENT-frame LEA row and zeroes the real address OUTPUT_LO nibble. Needs a
    surgical re-gate (exclude non-LEV LEA rows) before it can go default-ON.

    ROOT (block-input attribution via ``tools/interp_oracle_gate`` +
    ``tools/_probe_absdiff_ret_byte1.py``, spec_k=0 campaign, BUILT dims,
    golden ``f725c06e``; absdiff_0 id1046, absdiff_7 id1053, absdiff_8 id1054):
    on the ADJ step that IMMEDIATELY FOLLOWS the ``abs_diff`` LEV (function
    return) — the failure-map "step-22" — the AX byte-1 predictor row emits
    token ``0x01`` instead of ``0`` (``got_ax_bytes=[69,1,0,0]`` for
    ``|16-85|=69``). The AX byte VALUE is decoded from the ``OUTPUT_LO`` nibble
    one-hots (``W[1]-W[0]`` peaks at ``OUTPUT_LO+1`` / ``OUTPUT_LO+0``), and at
    the leak row ``OUTPUT_LO+0`` is crushed to ``-199.5`` while ``OUTPUT_LO+1``
    rises to ``+9.1`` -> the argmax nibble is 1 -> byte 0x01.

    The runtime-dominant writer of ``OUTPUT_LO+1`` there is
    ``layer6_routing_ffn :: l6_psh_stack0_marker_final_lo_1`` (contrib +9.94 vs
    the clean_emitter default +0.96). That L6 rule is a 3-way AND
    ``PSH_AT_SP AND MARK_STACK0 AND ALU_LO+1`` (threshold 2.5, unit weights) that
    is only meant to fire at a genuine PSH-STACK0 marker step. On the post-LEV
    ADJ step BOTH ``PSH_AT_SP`` and ``MARK_STACK0`` are 0, but ``ALU_LO+1`` reads
    a NON-BINARY ``5.81`` (stale ALU low-nibble after the SUB) which alone clears
    the AND threshold — so the AND mis-fires and stamps the value-1 nibble into
    ``OUTPUT_LO``. This is NOT the ``H*_DUMP_OUT`` byte-1 dump path (already
    killed at LEV by ``C4_LEV_AX_BYTE1_KILL``); it is the parallel OUTPUT-band
    decode leak that still wins.

    FIX (this flag, correct-by-construction, CLEAN DISCRIMINATOR): on the AX
    value-byte rows of the LEV-return step — ``OP_LEV`` (the return-context
    opcode marker, > 0.8 ONLY on the ADJ/return step, PATH-INDEPENDENT across
    both the ``a>b`` true and false absdiff paths, and decaying AX~1.1 > SP~0.9 >
    BP~0.6 across the step) AND ``IS_BYTE`` (a value byte, excludes the
    byte-0/MARK_AX row so byte-0 keeps its real nibble) AND the ELEVATED
    ``OUTPUT_LO+k`` leak itself (additive threshold, so the ~1.0 clean nibble
    baseline stays dark) AND ``NOT PSH_AT_SP`` AND ``NOT MARK_STACK0`` (the L6
    marker rule's own gates are provably OFF here, so its OUTPUT_LO write is
    spurious) — for each nibble k in 1..15, write ``-DOM`` to ``OUTPUT_LO+k`` and
    ``+DOM`` to ``OUTPUT_LO+0`` so the AX high bytes decode nibble-0 -> byte
    value 0. (An earlier ``H2_PREV_STEP+0`` discriminator was PATH-DEPENDENT — 0
    on the ``a>b`` true path — and over-fired on the clean nibble baseline via a
    multiplicative gate; the current OP_LEV + additive-threshold form is the
    robust one.)

    NOT-a-multi-byte GUARD (the bounded ``ABSDIFF_RET_LEAK`` l6-crush flag):
    ``func_mul`` returns a PRODUCT that can EXCEED 255 (e.g. mul(49,36)=1764,
    byte-1=6; mul(17,21)=357, byte-1=1), so its byte-1 is REAL and must NOT be
    zeroed. func_mul shares the LEV-return-step OP_LEV signature, so OP_LEV alone
    wrongly fires there (regressing e.g. func_mul_3 PASS->FAIL). The separator is
    the l6-CRUSH: on the absdiff single-byte leak the mis-firing
    ``l6_psh_stack0_marker_cancel_output`` sub-loop CRUSHES ``OUTPUT_LO+0``
    NEGATIVE (measured -46..-199 on ALL 25 absdiff at the ADJ AX byte-1 row)
    while a genuine func_mul return leaves it CLEAN POSITIVE (+3.91) OR SATURATES
    it hugely-negative (~-6e9). A PRECURSOR FFN writes a BANDED indicator
    ``ABSDIFF_RET_LEAK = step(-(OUTPUT_LO+0) >= 10) - 2*step(-(OUTPUT_LO+0) >=
    1000)``, and the corrector MULTIPLICATIVELY gates on it: clean-POSITIVE
    OUTPUT_LO+0 (func_mul +3.91) -> flag 0 -> gate 0 -> NO write (legit byte-1
    preserved); MODERATE crush (absdiff -46..-199) -> flag > 0 -> the correction
    fires (byte-1 -> 0); SATURATED crush (genuine multi-byte func_mul ~-6e9, e.g.
    mul(26,33)=858) -> the 2x hi-cancel DRIVES the flag NEGATIVE -> the corrector's
    sign-flipped writes stay proportional to the ~6e9 delivered nibble so the
    genuine byte-1 is PRESERVED (verified: passing func_mul id603/605/608 all stay
    OK). The 2x cancel is load-bearing: silu is linear (not a true step) so an
    equal-weight cancel would leave a positive residual at saturation that zeroes
    the genuine byte-1. The raw (ungated) crush value was rejected: it let the
    huge SP/BP/LEA/func_mul crushes amplify the multiplicative gate.

    SAFETY: every absdiff / func_identity / nested single-byte return is <= 255,
    so AX bytes 1..3 are 0 at the return step — forcing OUTPUT_LO to nibble-0 on
    the l6-crushed rows is exactly correct (same invariant ``C4_LEV_AX_BYTE1_KILL``
    relies on), and the crush guard excludes the genuine multi-byte func_mul
    case. The OP_LEV>=~0.95 selector + crush term keep this off every non-AX byte
    row (SP OP_LEV<=0.909 -> score<threshold; BP<=0.708) and off the pre-LEV
    steps (OP_LEV=0); ``IS_BYTE`` keeps byte-0's real nibble; the marker NOT-gates
    keep it off a genuine PSH-STACK0 rewrite. DEFAULT OFF; flag-OFF registers NO
    rules -> byte-identical to golden ``f725c06e``. Kept as a dedicated
    kill-switch for the flag-regression gate and the byte-identity gate.
    """
    return os.environ.get("C4_ABSDIFF_RET_BYTE1", "0") == "1"


def loaded_operand_add_hi15_clear_enabled() -> bool:
    """Return True iff the loaded-operand ADD high-nibble cell-15 address-leak
    clear is active. DEFAULT campaign-ON (``C4_LOADED_OPERAND_ADD_HI15_CLEAR=1``),
    opt out with ``C4_LOADED_OPERAND_ADD_HI15_CLEAR=0``; gated behind the
    ``no_stack0_emit`` campaign flag so the flag-OFF golden (35-tok) build is
    byte-identical (the wrap is never installed off the campaign).

    ROOT (GPU full_trace + oracle-tape probe, campaign default, spec_k=0, BUILT
    dims, ``tools/probe_varupd_add_survival.py``): ``var_update`` (ids 325-349,
    0/25) diverges at step 12 = the ``x = x + k`` ADD, neural ``got = expected +
    240 (0xF0)`` for ALL 25. Operand A is the LOADED variable ``x`` (``LI`` ->
    ``PSH`` -> ``mem[SP]``); the L8 head-5 mem-to-ALU value copy leaks the ``0xF``
    high nibble of the SP-relative store address (``0xFFE8``/``0xFFF8``) into
    ``ALU_HI+15`` (~+5.5) on top of the clean byte-0 one-hot, so the block-12
    AddSub high-nibble add reads a TWO-hot and the result gains ``0xF0``.
    IMMEDIATE operands have no such leak (so the immediate ``add`` cluster
    PASSES).

    THE FIX (``LoadedOperandAddHi15ClearFFN`` wrapping the L8 main FFN): on the
    ``OP_ADD`` MARK_AX rows ONLY, zero ``ALU_HI+15`` when it is in the
    contaminant window (``(0.5, 5.85)``) — a true ``0xF`` operand one-hot
    (~+6.0) is preserved, an immediate operand's ``@15`` (~0) is untouched.

    DELIBERATELY ADD-ONLY (narrower than the dropped broad
    ``C4_LOADED_OPERAND_HI15_CLEAR``, which also gated on ``OP_SUB`` + the six
    cmp opcodes and cleared ``ALU_LO+15`` — that broad form was DROPPED for
    regressing ``var_mul``). ``var_mul`` (``a*b``) has NO ADD step (its
    operand-delivery rows carry only ``OP_MUL``/``OP_LI``/``OP_PSH``; the MUL
    row's true operand value 0xF legitimately lands ``ALU+15 ~6.0``), so gating
    on ``OP_ADD`` makes the wrap PROVABLY INERT on ``var_mul``. Verified
    ``tools/probe_varmul_alu15.py``.
    """
    return os.environ.get("C4_LOADED_OPERAND_ADD_HI15_CLEAR", "1") != "0"


def funcadd_alu_hi13_clear_enabled() -> bool:
    """Return True iff the loaded-operand ADD high-nibble cell-13 address-leak
    clear is active. DEFAULT campaign-ON (``C4_FUNCADD_ALU_HI13_CLEAR=1``), opt
    out with ``C4_FUNCADD_ALU_HI13_CLEAR=0``; gated behind the same
    ``no_stack0_emit`` + ``loaded_operand_add_hi15_clear_enabled`` campaign
    chain so the flag-OFF golden (35-tok) build is byte-identical (the wider
    contaminant cell set is never installed off the campaign).

    ROOT (GPU full_trace id 575 + teacher-forced argmax probe, campaign default,
    spec_k=0, BUILT dims, ``tools/probe_funcadd_leak.py``): ``func_add``
    (``int add(int a,int b){return a+b;}``, id 575 = add(57,11), 0/25) diverges
    at step 13 = the ``a + b`` ADD. With the LEA ``&b`` fix (cc2ec2a8) the func
    args are delivered (step-9 LI) and addresses correct (step-11 LEA), so step
    13 IS the arithmetic. The operand-A high nibble (``a`` loaded from
    ``mem[BP+off]``) arrives in ALU_HI as a TWO-hot: the true nibble cell
    (``a//16`` <= 6, ~6-7) PLUS a CONSTANT ``~+5.49`` leak at **cell 13** — the
    ``0xD`` high nibble of the single-level call-frame load address (cf.
    var_update's ``0xF``/cell-15 leak). The block-12 AddSub high-nibble add then
    reads the two-hot and writes OUT_HI at the WRONG cell (``add(57,11)``:
    OUT_HI@1 instead of @4 -> AX byte0 = ``0x1B`` not ``0x44``). Sweep across 10
    operand pairs (``probe_funcadd_leak.py``) confirms the leak is ALWAYS cell
    13, ~5.49, and NEVER the true operand cell (func/var operands <= 100 ->
    hi nibble <= 6 << 13).

    THE FIX: add cell 13 to ``LoadedOperandAddHi15ClearFFN``'s contaminant cell
    set (alongside the existing cell 15). The same contaminant window
    (``(0.5, CLEAN_MAX=5.85)``) discriminates the ~5.49 leak from a true
    one-hot (~6-7), so it is value-safe. Flips ``func_add`` and rides to
    ``func_mul`` / ``func_max`` / ``func_min`` (same single-level frame, same
    cell-13 leak on their loaded operand-A ADD/compare steps).
    """
    return os.environ.get("C4_FUNCADD_ALU_HI13_CLEAR", "1") != "0"


def func_add_b0_hinib_enabled() -> bool:
    """Return True iff the func-return ADD byte-0 HIGH-nibble over-count fix is
    active — the ALL-CELL magnitude-windowed ALU_HI operand-B-bleed clear
    (DEFAULT-ON ``C4_FUNC_ADD_B0_HINIB``; opt out with =0). Gated behind the same
    ``no_stack0_emit`` + ``loaded_operand_add_hi15_clear_enabled`` campaign chain
    as the cell-13/15 clear, so the flag-OFF golden (35-tok) build is
    byte-identical (the widened cell set is never installed off the campaign).

    ROOT (BUILT-layout residual probe, campaign default, spec_k=0,
    ``tools/_probe_funcadd_operand.py``): ``func_add`` (id 578 add(42,78)=0x78
    got 0xB8, +0x40) diverges at step 13 = the ``a + b`` return ADD. The
    operand-A high nibble (``a`` loaded from ``mem[BP+off]``) arrives in
    ``ALU_HI`` as a TWO-hot: the true ``a//16`` cell (~+6.0) PLUS a spurious
    ``~+1.0`` one-hot at the cell equal to **operand-B's high nibble**
    (``b//16``). This is operand-B's high nibble bleeding through the L8 head-5
    mem-to-ALU operand-A read into ALU_HI (measured cell-by-cell: id578 b_hi=4 ->
    leak@4; id588 b_hi=3 -> leak@3, ALWAYS == b//16, ALWAYS ~1.0). The block-13
    AddSub high-nibble add then reads the two-hot k-weighted sum
    ``a_hi + b_hi`` for operand A, so the ADD result high nibble becomes
    ``(a_hi + b_hi) + b_hi + carry`` = the true ``a_hi + b_hi + carry`` plus an
    EXTRA ``b_hi`` -> ``+0x{b_hi}0`` over-count (the observed +0x30/+0x40/+0x50).
    The prior cell-13 (0xD frame-address) clear does NOT catch this leak because
    the leak cell is operand-B's high nibble (0x3/0x4/0x5), not the frame nibble.

    THE FIX: the leak (~1.0) and the true operand one-hot (~6.0, >= ``CLEAN_MAX``)
    are cleanly magnitude-separated by the SAME contaminant window the cell-13/15
    clear uses. So extend ``LoadedOperandAddHi15ClearFFN``'s contaminant cell set
    to ALL 16 ALU_HI cells: the window ``(0.5, CLEAN_MAX=5.85)`` clears the ~1.0
    operand-B bleed while PRESERVING the ~6.0 true operand-A high-nibble one-hot.
    Value-safe by construction (a true loaded operand-A high nibble is always
    delivered at the SCALE_O ~6.0 magnitude; no legitimate operand cell sits in
    the (0.5, 5.85) window). ADD-only opcode gate (unchanged) so it is inert on
    every non-ADD row. Flips the 12/25 func_add whose operand-B high nibble is
    non-zero (b >= 48). Scope is the loaded-operand ADD row (func_add step-13,
    also any expr/var frame ADD reading a loaded operand-A); func_max/min return
    via GT/LT+BZ+LEV with NO ADD step, so they are OUT of this fix's scope (their
    loaded-operand compare rows are the C4_OPERAND_CAM_FIX territory). This fix
    SUBSUMES the cell-13/15 clears when on (all 16 cells >= (13,15)).
    """
    return os.environ.get("C4_FUNC_ADD_B0_HINIB", "1") != "0"


def operand_cam_fix_enabled() -> bool:
    """Return True iff the operand-CAM address-leak clear is WIDENED past OP_ADD
    to the loaded-operand SUB / MUL / MOD / DIV + six-CMP operand-delivery rows
    (DEFAULT OFF — opt in via ``C4_OPERAND_CAM_FIX=1``). Gated behind the
    ``no_stack0_emit`` campaign flag so the flag-OFF golden (35-tok) build is
    byte-identical (the wider opcode gate is never installed off the campaign).

    ROOT (BUILT-layout survey, campaign default, spec_k=0,
    ``tools/probe_operand_cam_leak_survey.py``): the L8 head-5 mem-to-ALU
    operand-A read (``LI`` -> ``PSH`` -> ``mem[SP]``) delivers the LOADED value
    into ``ALU_HI`` as a TWO-HOT on EVERY consumer op, not just ADD: the true
    value hi-nibble one-hot (~+6.0, cell = value//16 <= 6 for corpus operands)
    PLUS a spurious ``~+5.49`` frame-address high-nibble leak at cell 13 (0xD,
    single-level call frame) or cell 15 (0xF, direct ``0xFFF8`` frame). The
    ``LoadedOperandAddHi15ClearFFN`` already discriminates + clears this leak on
    the ``OP_ADD`` rows (window ``(0.5, CLEAN_MAX=5.85)``); the leak is IDENTICAL
    in shape on the loaded-operand SUB (``absdiff``), MUL (``var_mul``), and the
    comparison ops (``if_var`` GT/LT, ``absdiff`` GT), but those rows are NOT
    cleared (the wrap is deliberately ADD-only). The block-12 ALU / L10 cmp
    engine then reads the two-hot and gains the leaked hi-nibble.

    THE FIX: widen the wrap's opcode gate to the loaded-operand binary + cmp
    consumer set, keeping the SAME ALU_HI-only, same-cell (13/15), same-window
    discriminator. This is provably narrower than the DROPPED broad clear
    (``C4_LOADED_OPERAND_HI15_CLEAR``) that regressed ``var_mul``: that form also
    cleared ``ALU_LO+15`` (which IS a true value cell for a low-nibble-0xF
    operand). The measured leak is ``ALU_HI``-ONLY on every op (survey:
    ``ALU_LO`` is a clean single one-hot at every operand-delivery row), and the
    ``(0.5, 5.85)`` window spares any genuine 0xD/0xF hi-nibble operand
    (immediate operands land ~+6.0 > CLEAN_MAX; the leak is ~5.49). So the
    widen is value-safe by the same construction that makes the ADD case safe.

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``, which also turns on the ``no_stack0_emit`` prerequisite);
    an explicit ``C4_OPERAND_CAM_FIX=0`` still opts out even under the campaign.
    """
    explicit = os.environ.get("C4_OPERAND_CAM_FIX")
    if explicit is None and campaign_enabled():
        return no_stack0_emit_enabled()
    return no_stack0_emit_enabled() and (explicit is not None and explicit != "0")


def sili_cam_b1_enabled() -> bool:
    """Return True iff the si/li LOAD byte-1 address-leak discriminator (Inc-2)
    is active. DEFAULT campaign-ON (``C4_SILI_CAM_B1=1``), opt out with
    ``C4_SILI_CAM_B1=0``; gated behind the two campaign flags so the flag-OFF
    golden (35-tok) build is byte-identical (the slot is never written off the
    campaign).

    ROOT (measured, faithful real-runner probes tools/probe_sili_scores.py +
    probe_var_scores.py, GPU bit-exact, campaign config):
    the LI-reload AX byte-1 is delivered by the shared L10 head-1 slot 82
    (``memax_byte1``), which selects the most-recent OP_IMM AX byte-1 REGISTER
    row by ALiBi recency. The slot-82 K-side scores the value-IMM step and the
    address-IMM step IDENTICALLY (both carry IS_BYTE + H1+AX + BYTE_INDEX_1 +
    OP_IMM with the same CLEAN signature); the ONLY thing separating them is
    ALiBi recency (the addr-IMM step is later -> wins by ~60). For
    ``IMM addr; LI`` (si/li/sc/lc) that later OP_IMM row is the LOAD-ADDRESS
    IMM (0x200 byte-1 = 0x02) so the reload LEAKS the address byte-1
    (roundtrip 42->0x22A, 16bit 0x1234->0x0234). For ``return x`` (var_simple,
    LEA-addressed) there is NO competing IMM-address row, so slot 82 picks the
    value-IMM (load-bearing; var_simple 25/25).

    THE DISCRIMINATOR (slot 83, Q-side gated -> softmax-safe): the address-IMM
    register row is the gathered LOAD ADDRESS, so it carries a STRONG
    ``ADDR_B1`` one-hot (HI/LO cell sums ~3.0) staged by the L13 address-gather
    chain; the genuine value-IMM register row carries only the weak residual
    ``ADDR_B1`` (sums ~1.0). Slot 83 fires its Q on the SAME byte-1 predictor
    row as slot 82 (so it contributes to NO other query -> the byte-0 reload
    softmax is untouched, NOT the K-side ADDR-veto blind spot), and its K
    DOWN-weights candidates by their ADDR_B1 magnitude. The penalty is ~3x on
    the leaky addr-IMM row and ~1x on the value-IMM row, a net margin that
    out-votes the ~60-point ALiBi recency gap and re-points the byte-1 reload
    at the value-IMM register (var_simple's winner, sum ~1.0, keeps its huge
    OP_IMM margin so it is unaffected).

    STATUS: flips 4/6 TestSmokeMemory (roundtrip / zero / multiple / overwrite);
    those have a value byte-1 == 0x00 so once the slot-83 selection picks the
    value-IMM the reload is correct. The remaining 2 are part-c (the SI-store
    byte-1 / downstream OUTPUT_HI corruptor), still open:
      * test_si_li_16bit_value (0x1234): slot 83 DOES re-point the L10 head-1
        byte-1 to the value-IMM (0x12 lands in OUTPUT at block 16, GPU-traced);
        but a DOWNSTREAM L18 (runner block 32) op SLAMS OUTPUT_HI on the
        LI-reload AX byte-1 row -- cell 0 -> +18, cells 1..15 -> -1700 -> hi
        nibble forced to 0 -> 0x12 -> 0x02. The L14 *_ax_bytes_zero FFN rules
        (jsr/lc/alu_nocarry/ent) are NOT the cause (their OP_LC_RELAY / TEMP+7
        gates read 0 on this row); the slam is the layer14_mem_generation
        ADDRESS head (heads 1..3, MEM addr byte) whose addr_b1 position-distance
        ALIASES the LI-reload AX byte-1 row in the 30-tok frame and copies the
        store ADDRESS byte-1 into OUTPUT. FIX (part-c): sharpen its slot-34
        MEM_STORE gate in campaign (move CONST down + MEM_STORE up by the SAME
        delta so STORE rows score identically and non-store rows fall to the
        softmax1 sink) so the head stays silent on the non-store LI row -- but
        wire it through ``_layer14_mem_generation_head_specs_with_overrides``
        (the override q_map is the BAKED path; a base-spec-only edit is inert),
        then re-run the flag_regression_gate (the store-address head is
        load-bearing for ALL SI/SC/PSH stores).
      * test_sc_lc_roundtrip: byte-1 IS now fixed (0x02 -> 0x00); the residual
        fail is a SEPARATE pre-existing LC byte-0 reload bug (got 0, want 42)
        outside this byte-1 scope.
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_SILI_CAM_B1", "1") != "0"
    )


def sili_b1_restore_enabled() -> bool:
    """Return True iff the si/li 16-bit LOAD byte-1 value RESTORE (Inc-2 part-c)
    is active. DEFAULT campaign-ON (``C4_SILI_B1_RESTORE=1``), opt out with
    ``C4_SILI_B1_RESTORE=0``; gated behind the campaign config so the flag-OFF
    golden (35-tok) build is byte-identical (the band + both ops are omitted off
    the campaign).

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU
    block-trace ``tools/probe_sili_slam32.py`` / ``probe_sili_real.py``):
    the slot-83 discriminator (``sili_cam_b1_enabled``) correctly re-points the
    L10 head-1 byte-1 to the value-IMM register, so the loaded byte-1 (0x12 for
    ``si 0x1234; li``) lands in OUTPUT at runner block 16. A DOWNSTREAM op then
    SLAMS OUTPUT_HI on the LI-reload AX byte-1 predictor row at runner BLOCK 32
    (logical L18): the FFN unit that forces the AX byte-1 HIGH nibble to 0 (the
    "AX byte-1 == 0 default" — ``OUTPUT_HI+1..15 = -50`` gated on
    ``H1+1``/``BYTE_INDEX_0``). That default is INTENTIONALLY suppressed on
    register rows whose AX legitimately has a high byte via the ``AX_CARRY``
    cross-step band, but on the LI-reload step that suppression band is empty, so
    the default fires and crushes the loaded 0x12 -> 0x02 (final reload 0x0234,
    smoke want 0x1234). The attn at block 32 is NOT the source (zeroing its
    ``W_o`` leaves the slam); the slam is FFN unit 3 of block 32 (ablating it
    restores 0x12) — so a head-spec edit cannot reach it.

    THE FIX (two PureFFN ops mirroring the ``C4_SUB_FULL_BORROW`` precedent: an
    early CAPTURE + an L25-tail RESTORE that DOMINATES the slam). On the
    LI-reload byte-1 predictor row (the AX byte-0 row of an LI step — discriminated
    by ``IS_BYTE + H1[AX]+1 + BYTE_INDEX_0 + ADDR_B1`` and NOT
    ``OP_IMM``/``MARK_AX``/``MEM_STORE``/``BYTE_INDEX_1..3``; ``ADDR_B1`` is the
    gathered LOAD-address one-hot, strong on LI-reload rows and ~0 on
    PSH/SI/IMM register rows, so the row is LOAD-specific):
      1. CAPTURE (``make_layer14_sili_b1_capture_op``, a standalone PureFFN at the
         EARLY L14 mem-gen block, BEFORE the block-32 slam, where OUTPUT byte-1 is
         still the loaded value): gate-copy the 16 ``OUTPUT_LO`` + 16 ``OUTPUT_HI``
         nibble cells into the private ``LI_RELOAD_B1_{LO,HI}`` band. The silu
         factor is uniform across the 16 cells so the byte-1 nibble argmax is
         preserved.
      2. RESTORE (``make_sili_b1_restore_op``, a standalone PureFFN on the L25 tail
         block AFTER ``tail_bit32_result_correction`` — the LAST OUTPUT writer
         before the LM head): gate-write ``OUTPUT_{LO,HI}`` back from
         ``LI_RELOAD_B1_{LO,HI}`` at a DOMINANT magnitude, so the restored 0x12
         out-votes the block-32 slam (~+18/-1700) additively.
    On the 4 already-passing si/li cases (value byte-1 == 0x00) the captured band
    is the 0x00 one-hot, so the restore re-asserts 0x00 (no-op). var_simple
    (LEA-addressed, no ADDR_B1 on its byte-1 row) and PSH/SI register rows are not
    matched by the discriminator, so they are untouched. Output-affecting only
    inside the campaign; flag OFF (or off-campaign) omits the band + both ops
    (golden ``7f6f2e5d`` byte-identical).
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_SILI_B1_RESTORE", "1") != "0"
    )


def store_ax_b0_override_enabled() -> bool:
    """Return True iff the SI/SC store-AX byte-0 OUTPUT materializer uses the
    zero-default OVERRIDE write form. DEFAULT campaign-ON
    (``C4_STORE_AX_B0_OVERRIDE=1``), opt out with ``C4_STORE_AX_B0_OVERRIDE=0``;
    gated behind the two campaign flags so the flag-OFF golden (35-tok) build is
    byte-identical (the ``l16_store_ax_carry_lo`` rules keep their bare additive
    ``2.0/S`` write off the campaign).

    ROOT (var_mul step-9 store, GPU/CPU full_trace; the ``var_mul`` 2nd-local
    SI store of ``b``): on the SI step's AX marker the store value is carried in
    ``AX_CARRY_LO/HI`` (correct), but the OUTPUT byte-0 LO band carries a STRONG
    zero-byte default (the L19 / block-33 OUTPUT-zero-default, ~+26/+40 on
    ``OUTPUT_LO+0``) that out-votes the weak additive ``+2.0/S`` write of
    ``l16_store_ax_carry_lo``. When the stored value's HIGH nibble of byte-0 is
    zero (``b <= 15``: var_mul ids 279/280/287/290/296) the carried LOW nibble
    is the only nonzero cell, and the bare additive write loses to the
    zero-default -> OUTPUT byte-0 = 0 -> ``SI`` stores 0 -> ``LI b`` reads 0 ->
    ``a * 0``.

    THE FIX (mirrors the campaign ``l16_psh_ax_carry_lo`` materializer added for
    the PSH func-arg path, which solved the IDENTICAL zero-default-out-votes
    problem): switch the ``l16_store_ax_carry_lo`` write to the OVERRIDE form —
    write ``+W`` to the carried LO nibble cell ``k`` AND ``-W`` to ``OUTPUT_LO+0``
    for ``k != 0`` so a NONZERO carried low nibble overrides the zero default.
    ``k == 0`` self-cancels (a genuinely-zero low byte stays at its zero
    default). The HI band keeps the plain additive form (no competing default on
    ``OUTPUT_HI``). The rule COUNT is unchanged (the 16 LO rules' write tuples
    are rewritten in place), so the flag-OFF golden bake is byte-identical and
    the L16 unit allocator's fixed-range assertion still holds.

    Output-affecting only inside the campaign; flag OFF (or off-campaign) keeps
    the bare additive ``2.0/S`` write (golden ``7f6f2e5d`` byte-identical).
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_STORE_AX_B0_OVERRIDE", "0") != "0"
    )


def store_ax_b0_override_v2_enabled() -> bool:
    """Return True iff the SI/SC store-AX byte-0 OUTPUT materializer uses the
    zero-default OVERRIDE write form *with an ALU/cmp opcode ANTI-CONDITION gate*
    (the "clean discriminator" V2). DEFAULT **ON** (``C4_STORE_AX_B0_OVERRIDE_V2``,
    opt out ``=0``); gated behind the two campaign flags so the flag-OFF golden
    (35-tok) build is byte-identical.

    WHY V2 (root: var_three id300 / var_mul step-9 SI-store; task this session):
    the plain ``C4_STORE_AX_B0_OVERRIDE`` (default OFF since 7869e5c3) FIXES the
    store step (var_three id300 SI-of-b step-9: AX byte-0 token 0 -> 6, GPU/CPU
    teacher-forced verified) but was reverted OFF because "ON amplified the
    l16_store_ax_carry misfire onto ADD/SUB/cmp/mul rows (-24)". The plain
    ``store_ax_conditions`` gate (``OP_SI + OP_SC + MARK_AX - 8*MARK_PC -
    10*IS_BYTE - 20*OP_EXIT - 20*OP_JMP``, threshold 4.0) does NOT explicitly
    forbid the ALU/cmp opcodes, so any residual OP_SI/OP_SC energy at an
    ADD/SUB/MUL/DIV/MOD/cmp AX-marker row can lift the AND-gate ``up`` above zero
    and let the strong ``-W`` OUTPUT_LO+0 write bleed onto the arithmetic result
    byte. V2 adds a large-negative anti-condition on every ALU/cmp opcode flag
    (``-20`` each) so the AND-gate ``up`` is driven deeply negative (silu -> 0) on
    ANY ADD/SUB/MUL/DIV/MOD/EQ/NE/LT/GT/LE/GE row — the override CANNOT fire there
    by construction, while a genuine SI/SC store row (all ALU/cmp flags ~0) is
    unaffected. This is the clean store-only discriminator the plain override
    lacked. Rule COUNT is unchanged (only the SI/SC store rules' condition tuple +
    write tuples change, and only when V2 is ON), so the flag-OFF golden bake is
    byte-identical.

    Mutually exclusive with the plain override in practice: enable EITHER
    ``C4_STORE_AX_B0_OVERRIDE=1`` (broad, un-discriminated) OR
    ``C4_STORE_AX_B0_OVERRIDE_V2=1`` (discriminated). If both are set, V2 wins (the
    anti-condition gate is the strict superset guard). Output-affecting only inside
    the campaign; flag OFF (or off-campaign) keeps the bare additive ``2.0/S``
    write (golden ``f725c06e`` byte-identical).
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_STORE_AX_B0_OVERRIDE_V2", "1") != "0"
    )


def loop_lea_b0_e8_restore_enabled() -> bool:
    """Return True iff the loop_sum in-loop ``LEA &i`` byte-0 0xE8 RESTORE
    (``C4_LOOP_LEA_B0_E8``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LEA_B0_E8=0``. Flag-off OR
    a non-campaign / golden build registers NO rules and appends NO post_op, so
    the model is bit-for-bit identical to golden ``7f6f2e5d``.

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_lea_b0.py`` 450 2 + ``_probe_loopsum_blk43_attrib.py``
    + ``_probe_loopsum_entax_gate.py``): on the ``loop_sum`` (and loop_mul /
    loop_pow2 / loop_fact family) IN-LOOP comparison step the ``LEA &i`` (the
    1st-local address, want byte-0 = 0xE8) emits 0x00 because of a TWO-part
    failure:

      1. The ``tail_lea_local_ax_marker_byte0_e8`` keystone (block 42) does NOT
         fire on this row: it REQUIRES ``CMP+7`` AND ``MEM_ADDR_SRC`` (each 1.0 on
         a genuine address-eval LEA), but the in-loop LEA row carries
         ``CMP+7 == 0`` and ``MEM_ADDR_SRC == 0`` (its in-loop base is lower than
         the keystone was tuned for). So OUTPUT_LO byte-0 stays DEAD (~0.0)
         through block 42 — the 0xE8 stamp never lands.
      2. The ``_l10_ent_axcarry`` override op (block 43) THEN WRONGLY fires on
         this GENUINE LEA row and slams byte-0 to 0x00. That op's discriminator
         is ``MARK_AX + OP_LEA(>=4) + no-owning-opcode + MEM_ADDR_SRC cold``; it
         was designed for the multilocal MAIN-ENT step (where OP_LEA LEAKS ~0.81
         and ``MEM_ADDR_SRC == 0``) and EXCLUDES genuine address-eval LEAs via
         the ``MEM_ADDR_SRC`` NOT-block. But this in-loop ``LEA &i`` is a genuine
         LEA (``OP_LEA == 5.24``) that ALSO carries ``MEM_ADDR_SRC == 0``, so the
         exclusion fails — ent_axcarry routes ``AX_CARRY`` (byte-0 = 0x00) into
         OUTPUT, winner-take-all to cell 0. Final emitted AX byte-0 = 0x00.

    FETCH on this row carries the imm=-8 signature (``FETCH_LO+8`` argmax,
    ``FETCH_HI+15`` argmax, ``FETCH_LO+0`` / ``FETCH_HI+14`` cold) — the EXACT
    pattern the e8 keystone keys on for a 0xE8 byte-0, and DISTINCT from the
    multi-local 2nd local (``FETCH_LO+0``, want 0xE0) and 3rd local
    (``FETCH_HI+14``, want 0xD8).

    FIX (mirrors the ``C4_SILI_B1_RESTORE`` / ``_l10_ent_axcarry`` precedent):
    a flag-gated ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``l10_ent_axcarry`` (so it is the LAST OUTPUT writer at this row and DOMINATES
    the slam), that — gated on the in-loop ``LEA &i`` discriminator (genuine
    ``OP_LEA`` + imm=-8 FETCH signature + ``MEM_ADDR_SRC`` cold + no owning opcode
    + the multi-local FETCH NOT-blocks) — writes byte-0 = 0xE8 (``OUTPUT_LO+8``
    HIGH, ``OUTPUT_HI_THIS_STEP+14`` HIGH, all other nibble cells driven
    ``-DOM``) via a per-cell winner-take-all. The ``MEM_ADDR_SRC`` NOT-block keeps
    it OFF every genuine address-eval LEA (those carry ``MEM_ADDR_SRC == 1`` and
    are already correct via the keystone), and the imm=-8 FETCH AND-gate keeps it
    off the 2nd/3rd-local LEAs (``var_mul`` / ``var_three``) and every non-LEA AX
    row, so the whole op is a no-op everywhere except the loop in-loop ``LEA &i``
    row it targets.
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_LOOP_LEA_B0_E8", "1") != "0"
    )


def loop_lea_b0_e8_oplea_req_enabled() -> bool:
    """Return True iff the ``C4_LOOP_LEA_B0_E8`` discriminator's ``OP_LEA``
    HARD-REQUIREMENT narrowing is active (POST-FLIP func_identity step-9 LI
    regression fix).

    DEFAULT **ON** wherever the e8-restore op itself is active; opt-out via
    ``C4_LOOP_LEA_B0_E8_OPLEA_REQ=0`` (which reverts to the pre-fix
    ``OP_LEA`` weight 60 / no CONST bias, reproducing the regression — kept as a
    dedicated kill-switch so ``tools/flag_regression_gate.py`` can A/B JUST the
    narrowing). Inert (and therefore byte-identical) whenever the e8-restore op
    is OFF, since the discriminator is only emitted then.

    ROOT (GPU spec_k=0, BUILT dims, campaign config; ``tools/_probe_funcid_loope8.py``
    on ``func_identity`` id550 ``identity(70)``): the e8-restore op's discriminator
    (``MARK_AX*100 + OP_LEA*60 + FETCH_LO+8*1000 - FETCH_LO+0*1000
    - FETCH_HI+14*1000`` > threshold 500) was MEANT to fire only on a GENUINE
    in-loop ``LEA &i`` (``OP_LEA == 5.24``), but ``OP_LEA`` is NOT load-bearing:
    the ``FETCH_LO+8 * 1000`` term ALONE clears threshold 500. ``func_identity``'s
    ``return x`` body issues ``LEA &x`` (step 8, fires correctly) THEN ``LI``
    (step 9, loads value 70 = 0x46). The LI's FETCHED-instruction byte carries
    ``FETCH_LO+8 == 1.00`` (the LI opcode encodes to FETCH low-nibble 8) while
    ``OP_LEA == 0`` and ``OP_LI == 0.01`` (cold) — so the op scores
    100 + 0 + 1000 = 1100 > 500 and FALSE-FIRES on the LI row, stamping 0xE8 over
    the loaded 0x46 (got_ax 0xFFE8 = sign-extended 0xE8, oracle 70). The bug was
    introduced by ``C4_LOOP_LEA_B0_E8`` (commit ``fd60f5f4``, the first post-flip
    bad commit for id550; bisect-confirmed) and is INVISIBLE to ``OP_LI`` /
    ``OP_LI_RELAY`` NOT-blocks (both ~0 at the LI byte-0 lookup row).

    THE FIX: make ``OP_LEA`` a HARD requirement. The genuine in-loop ``LEA &i``
    carries ``OP_LEA == 5.24``; the LI carries ``OP_LEA == 0`` EXACTLY. Bump the
    ``OP_LEA`` condition weight 60 -> 200 and add a ``CONST -650`` bias so the
    score needs the genuine LEA's OP_LEA term to cross threshold 500:

      * genuine ``&i`` LEA: 100 + 5.24*200 + 160(FETCH) - 650 = 658 > 500 -> FIRES
        (158-pt margin, up from the pre-fix 74).
      * ``func_identity`` LI: 100 + 0 + 1000(FETCH) - 650 = 450 < 500 -> VETOED.
      * 2nd/3rd-local LEAs (``var_mul`` / ``var_three``, FETCH net -1000):
        100 + 1048 - 1000 - 650 = -502 -> still SILENT.
      * clean non-LEA AX row (no FETCH, OP_LEA 0): 100 - 650 = -550 -> SILENT.

    OFF leaves the discriminator at the pre-fix weights (byte-identical to the
    regressed build).
    """
    return os.environ.get("C4_LOOP_LEA_B0_E8_OPLEA_REQ", "1") != "0"


def loop_lea_b0_e0_restore_enabled() -> bool:
    """Return True iff the loop_sum in-loop 2nd-local ``LEA &sum`` byte-0 0xE0
    RESTORE (``C4_LOOP_LEA_B0_E0``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LEA_B0_E0=0``. Flag-off OR
    a non-campaign / golden build registers NO rules and appends NO post_op, so
    the model is bit-for-bit identical to golden ``fd60f5f4``.

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_lea_b0.py`` 450 6 + ``_probe_loopsum_blk43_attrib.py``
    450 6 44): after the merged step-2 fix (``C4_LOOP_LEA_B0_E8``) advances
    ``loop_sum`` to step 6, the in-loop ``sum = sum + i`` body issues a 2nd-local
    ``LEA &sum`` whose AX-marker wants byte-0 = 0xE0 (the full AX = 0xFFE0 =
    65504 sign-extended local address). It instead emits 0x01 because:

      1. The 0xE8 keystone / e8-restore op DEFERS here (its ``FETCH_LO+0``
         NOT-block) -- correctly, since this is the 2nd local, not ``&i``.
         Through the L25 tail (block 43, the post-block-43 residual) OUTPUT_LO
         byte-0 carries cell-0 ~49.5 (the correct LO nibble for 0xE0) but no
         keystone has stamped the 0xE high nibble.
      2. The post-tail block-44 LEA effective-address materializer (a 32-unit
         standalone PureFFN, gated MARK_AX + OP_LEA*60 + IS_BYTE) then WTA-slams
         OUTPUT_LO -> cell **1** and OUTPUT_HI_THIS_STEP -> cell 0 on this row,
         deriving byte-0 = 0x01 (its in-loop 2nd-local effective-address compute
         is wrong). The e8-restore op (which DOMINATES this slam for ``&i`` at a
         LATER post-op block) does not fire here, so the 0x01 ships.

    The 2nd-local is distinguished from the 1st (``&i`` -> 0xE8) and 3rd
    (``&c`` -> 0xD8) locals by WHICH FETCH cell dominates: the 2nd local carries
    ``FETCH_LO+0`` argmax (=1.0; ``FETCH_LO+8`` / ``FETCH_HI+14`` cold) -- the
    EXACT inverse of the e8 op's ``FETCH_LO+8``-dominant signature.

    FIX (mirrors ``C4_LOOP_LEA_B0_E8`` / ``_l10_loop_lea_b0_e8_rules``): a
    flag-gated ``PureFFN`` post_op appended AFTER ``l10_loop_lea_b0_e8`` (so it
    is the LAST OUTPUT writer at the 2nd-local LEA row and DOMINATES the block-44
    slam) that -- gated on the in-loop ``LEA &sum`` discriminator (genuine
    ``OP_LEA`` + imm=-16 ``FETCH_LO+0``-dominant signature + ``MEM_ADDR_SRC``
    cold + no owning opcode + the multi-local FETCH NOT-blocks) -- writes byte-0
    = 0xE0 (``OUTPUT_LO+0`` HIGH, ``OUTPUT_HI_THIS_STEP+14`` HIGH, all other
    nibble cells driven ``-DOM``) via a per-cell winner-take-all. The
    ``FETCH_LO+8`` / ``FETCH_HI+14`` NOT-blocks keep it OFF the 1st/3rd-local
    LEAs, the ``MEM_ADDR_SRC`` NOT-block keeps it OFF every genuine
    address-eval LEA, and the ``IS_BYTE`` + non-AX marker NOT-blocks keep it OFF
    every value-byte / non-AX row.
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_LOOP_LEA_B0_E0", "1") != "0"
    )


def loop_lea_oplea_gate_enabled() -> bool:
    """Return True iff the ``C4_LOOP_LEA_OPLEA_GATE`` MULTIPLICATIVE ``OP_LEA``
    gate on the ``C4_LOOP_LEA_B0_E8`` / ``C4_LOOP_LEA_B0_E0`` restore ops is
    active (PROJECT_0XE8_SLAM Phase-2 fix).

    DEFAULT **OFF** (opt in ``C4_LOOP_LEA_OPLEA_GATE=1``). Kept as a dedicated
    kill-switch so ``tools/flag_regression_gate.py`` / ``tools/_isa_golden_hash.py``
    can A/B JUST the gate: OFF leaves the loop_lea ops byte-identical to golden
    ``b1dcae63`` (the gate is the only structural change), ON adds the gate to
    every unit in both rule families. Inert whenever the loop_lea ops themselves
    are OFF (non-campaign / golden), since the gate is only emitted then.

    ROOT (docs/PROJECT_0XE8_SLAM_2026_07_07.md §5; measured spec_k=0, BUILT dims,
    campaign): the loop_lea restore ops' discriminator scores an ``imm=-8`` /
    ``imm=-16`` FETCH signature (``FETCH_LO+8`` / ``FETCH_LO+0`` at weight 1000)
    that was calibrated against a SOFT FETCH one-hot (0.4-1.0 on genuine in-loop
    ``LEA &i`` rows). On an ``if_gt`` / ``if_eq`` step-0 IMM comparison AX row the
    FETCH band carries ``+40`` (a 40x-amplified broadcast, NOT a 0/1 one-hot), so
    ``FETCH_LO+8 * 1000 = +4.0e6`` (S=100) OVERWHELMS every calibrated additive
    margin -- the additive ``OP_LEA`` HARD-req (weight 200 + CONST -650) and the
    ``OP_IMM * -500`` opcode block are dwarfed. Net ``up > 0`` -> the AND
    spuriously fires and the per-cell winner-take-all slams ``0xE8`` / ``0xE0``
    over the correct operand byte (id360 ``8>27`` wants 0x08, id402 ``16==9``
    wants 0x10).

    THE FIX: make ``OP_LEA`` a MULTIPLICATIVE gate that no FETCH amplitude can
    cross. The FFN math is ``hidden = silu(up) * gate`` with
    ``gate = gate_bias + Sum(gate_terms . x)`` (the gate weights are NOT scaled by
    S). With ``gate_bias=0.0`` + ``gate_terms=(("OP_LEA", 1/5.23),)``:

      * genuine in-loop ``LEA &i`` (``OP_LEA == 5.23``): gate = 1/5.23 * 5.23
        = ~1.0 -> ``hidden = silu(up) * ~1.0`` -> the unit fires at the SAME
        magnitude as today (loop function BYTE-IDENTICAL: the per-cell +/-DOM
        winner-take-all is preserved).
      * leak IMM / comparison row (``OP_LEA == 0``): gate = 0.0 EXACTLY ->
        ``hidden = silu(up) * 0 = 0`` -> the whole unit ZEROES, regardless of the
        ``+40`` FETCH amplitude, handing the OUTPUT cell back to the correct
        comparison-decode writer.

    A gate_bias of 0.0 (rather than the doc's illustrative -1.0) is deliberate:
    with ``silu(up) * gate`` a NEGATIVE gate would INVERT the +/-DOM writes on a
    false-fire row (silu(up) stays large-positive since ``up`` is only gated at
    the DOWN projection), producing a DIFFERENT wrong byte rather than a clean
    no-op. gate_bias=0.0 + a linear ``OP_LEA`` term is the unique gate that both
    (a) is EXACTLY 0 on the ``OP_LEA==0`` leak (true zero-out) and (b) is ~1.0 on
    the genuine ``OP_LEA==5.23`` LEA (write magnitude preserved).
    """
    return os.environ.get("C4_LOOP_LEA_OPLEA_GATE", "1") != "0"


def jsr_bp_byte3_clear_enabled() -> bool:
    """Return True iff the func step-0 JSR-step BP byte-3 high-byte CLEAR
    (``C4_JSR_BP_BYTE3_CLEAR``) is active.

    DEFAULT **ON** (opt out ``C4_JSR_BP_BYTE3_CLEAR=0``). Requires the campaign
    config (``C4_NO_STACK0_EMIT=1`` + ``C4_OPERAND_FROM_MEMSP=1``). Flag-off OR a
    non-campaign / golden build registers NO rules and appends NO post_op, so the
    model is bit-for-bit identical to golden ``f725c06e``.

    ROOT — THE func-cluster STEP-0 (JSR) POISONING BYTE (func_min id675 /
    func_max id650 / func_identity id550 / func_add id575; measured spec_k=0,
    BUILT dim_positions, campaign; ``tools/_probe_funcmin_vcorr.py`` +
    ``_probe_bp_b3_logit_attr.py`` + ``_probe_funcmin_inject.py``):

    Every func-cluster program's FIRST value-byte correction (the cross-step
    autoregressive poisoning point ``interp_oracle_gate._value_correction_step``)
    is at **step 0 (the JSR that calls the callee)**, on **BP byte 3**. The
    caller's BP is ``0x00010000`` (byte-2 = 0x01, byte-3 = 0x00), but the model
    emits **BP byte-3 = 0x01** -> BP decodes to ``0x01010000`` and every
    downstream frame-relative (``mem[BP-8]`` / ENT / LEV / LI) read is poisoned,
    flat-diverging the whole cluster. Hook-inject clearing this one byte advances
    the vcorr from step 0 to step 2 on ALL four programs (root confirmed).

    ATTRIBUTION (LM-head logit, BUILT dims): at the BP byte-3 PREDICTOR row (the
    BP byte-2 value row: ``IS_BYTE`` + ``BYTE_INDEX_2`` + ``H1+3`` + ``OP_JSR``,
    ``HAS_SE==0``) the OUTPUT band is a near-TIE the WRONG way -- ``OUTPUT_LO+0 =
    +6.94`` (nibble 0, correct 0x?0) vs ``OUTPUT_LO+1 = +8.00`` (nibble 1) -- so
    the byte-3 low nibble decodes to 1 (``logit[1]-logit[0] = +4.3``, driven
    100% by ``OUTPUT_LO+1`` via ``W_head[1,OUTPUT_LO+1]-W_head[0,·] = +5``). A
    weak ``+2.0`` enters ``OUTPUT_LO+1`` at physical block ~18 and is amplified
    ~3.4x at block ~58, and unlike the ENT step (which
    ``l6_ent_after_jsr_bp_byte3_00`` clears) the JSR step has NO byte-3 clear.
    The clean, always-present discriminator that isolates this exact row is
    ``OP_JSR`` (the ENT-step clear is gated on ``OP_ENT``/``HAS_SE`` instead) +
    the BP-register-byte ``H1+3`` staging signal + ``BYTE_INDEX_2``.

    FIX (mirrors ``l6_ent_after_jsr_bp_byte3_00`` but JSR-gated + placed in the
    L25 tail so it DOMINATES the block-58 amplifier): a flag-gated ``PureFFN``
    post_op appended after the loop_lea ops that -- on the JSR-step BP byte-3
    predictor row -- WTA-forces ``OUTPUT_LO`` and ``OUTPUT_HI_THIS_STEP`` to
    nibble 0 (byte-3 = 0x00). ``OP_JSR`` + ``H1+3`` + ``BYTE_INDEX_2`` hard-req,
    with ``HAS_SE`` (ENT rows) + every non-BP marker + wrong BYTE_INDEX
    NOT-blocked, so it is a no-op on every non-JSR-BP-byte3 row.
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_JSR_BP_BYTE3_CLEAR", "1") != "0"
    )


def loop_si_byterow_marker_clear_enabled() -> bool:
    """Return True iff the loop back-edge SI-step value-byte-row MARKER-residue
    clear (``C4_LOOP_SI_BYTEROW_CLEAR``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_SI_BYTEROW_CLEAR=0``.
    Flag-off OR a non-campaign / golden build registers NO rules and appends NO
    post_op, so the model is bit-for-bit identical to golden ``5acb3d23``.

    ROOT — THE FIRST LOOP BACK-EDGE CONTROL-FLOW DESYNC (loop_sum id450,
    measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_backedge.py`` + ``_probe_loopsum_spdrift.py`` +
    ``_probe_loopsum_explosion.py``):

    After the step-2 / step-6 in-loop LEA byte-0 fixes (``C4_LOOP_LEA_B0_E8`` /
    ``C4_LOOP_LEA_B0_E0``) advance ``loop_sum`` to step 9, the ``sum = 0`` store
    (step 9 = ``SI``) DESYNCS the 30-token frame: it emits **41 tokens not 30**.
    The over-run begins at the SP value-byte-0 row (the row whose input token is
    the just-emitted SP byte-0 = 0xE8). At that row the LM-head BYTE logits
    collapse to ~ -5.8e11 so a register-MARKER token (``REG_PC`` = 257) wins by
    DEFAULT, injecting a spurious PC/AX/SP block (+11 tokens). The fixed-30-token
    slicer then mis-reads the NEXT step: step 10 (the loop-condition ``LEA &i``,
    oracle ``pc_after`` = 106) is decoded as ``pc_after`` = 114 — the PC appears
    to advance +16 not +8, the brief's reported symptom.

    WHY the byte logits collapse (per-block + per-unit attribution): at the SI
    step's value-byte rows ``MEM_STORE`` (born ~block 11) and ``OP_SI`` (born
    ~block 31) carry a small NEGATIVE residue (~ -2.4e-3 / -4.2e-4) that SHOULD be
    exactly 0 on a value-byte row (markers/opcodes belong on the marker row). The
    L25 tail bank (``tail_bit32_result_correction``, block 43) reads these dims at
    ``-1e8`` as NOT-blockers ASSUMING they are 0. ``MEM_STORE * -1e8 = +2.4e5``
    (plus ``OP_SI * -1e8 = +4.2e4``) is enough to flip the bank's silu gate from
    ``up`` ~ -1.4e4 (OFF, the clean-row value) to ``up`` ~ +6.8e4 (ON): a whole
    band of OUTPUT-decode units fires asymmetrically -> the OUTPUT_LO/HI decode
    band explodes to ~ -7.4e11 -> the LM byte head (each band cell drives a byte
    token at +5.0) is crushed all-negative -> marker token. On a clean (PSH/IMM)
    step those residues are ~0 so the bank stays OFF and the band decodes a real
    byte (+14.3, the SP value).

    FIX: a flag-gated ``PureFFN`` block scheduled IMMEDIATELY BEFORE the L25 tail
    bank that, on EVERY value-byte row (``IS_BYTE`` gate), ADDS a small POSITIVE
    bias (effective +0.02) to ``MEM_STORE`` and ``OP_SI`` so their residue can no
    longer flip the tail bank's ``-1e8`` silu gate positive (``+0.02 * -1e8 =
    -2e6`` keeps ``up`` deeply negative -> bank OFF -> OUTPUT band decodes the
    real byte). The +0.02 magnitude is CRITICAL: it must be SMALL enough that it
    does not also flip the AX HIGH-BYTE sign-extension materializer (at +0.5 the
    AX byte 2/3 over-sign-extend to 0xFF) NOR the multi-byte ADD high-byte adder
    (at the unscaled FFN-saturated +100 the ADD high byte is lost, 768 -> 256) --
    both read the same MEM_STORE / OP_SI dims; the AX-safe window is ~0.005..0.1
    so 0.02 is the robust middle (calibrated via the deterministic ~5000 silu
    saturation, W_down = 0.02/5000 = 4e-6). It fires ONLY on value-byte rows
    (``IS_BYTE`` AND every marker NOT-blocked) so it never perturbs the marker
    rows the tail's MEM-store address materializers legitimately use. Verified
    (``_probe_loopsum_backedge.py`` + canonical id450 + a 72-id cross-cluster
    campaign sample, ZERO regressions): the SI step re-emits 30 tokens, step-10
    PC is 106 (correct), and steps 0-10 are PC+AX byte-correct. The remaining
    loop_sum residual (step-11 in-loop ``LI &i`` returning AX=0 not 1) is a
    DISTINCT downstream value-load / operand-CAM root (task #342 family), out of
    scope for the back-edge desync.
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_LOOP_SI_BYTEROW_CLEAR", "1") != "0"
    )


def loop_li_opcode_fetch_addrkey_clamp_enabled() -> bool:
    """Return True iff the in-loop opcode-fetch ADDR_KEY top-byte over-count
    clamp (``C4_LOOP_LI_FETCH_ADDRKEY_CLAMP``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LI_FETCH_ADDRKEY_CLAMP=0``.
    Flag-off OR a non-campaign / golden build registers NO rules and appends NO
    post_op, so the model is bit-for-bit identical to golden.

    ROOT — THE IN-LOOP ``LI`` VALUE-LOAD RETURNS 0 IS AN UPSTREAM OPCODE-FETCH
    MISS ON THE STORE (loop_sum id450 step 11, measured spec_k=0, BUILT
    dim_positions, campaign config; GPU teacher-forced trace
    ``tools/_probe_loopsum_li_cam.py`` + ``_probe_loopsum_memstore_chain.py`` +
    ``_probe_loopsum_pcfetch.py`` + ``_probe_loopsum_fetchattn.py``):

    After the back-edge desync fix (``C4_LOOP_SI_BYTEROW_CLEAR``) advances
    ``loop_sum`` to step 11, the in-loop ``LI &i`` (loop-condition variable load,
    pc 106->114) returns AX=0 not 1. The L15 ``memory_lookup`` CAM (head 0) that
    content-addresses the load only attends to MEM rows carrying ``MEM_STORE=1``.
    But the two genuine in-frame ``SI`` stores -- step 5 (``i = 1`` -> 0xFFE8) and
    step 9 (``sum = 0`` -> 0xFFE0) -- carry ``MEM_STORE = -0.0`` on their MARK_MEM
    marker (only the prologue JSR + an in-loop PSH show ``MEM_STORE = 1``). So the
    LI CAM's candidate set is EMPTY of the real store and the load reads garbage.

    WHY ``MEM_STORE`` is missing on the SI stores: ``MEM_STORE`` is the relayed OR
    of ``OP_SI/OP_SC/OP_PSH/OP_JSR/OP_ENT`` (L6 head 6 broadcasts it from the AX
    marker to the MARK_MEM marker). At the SI steps the opcode dim ``OP_SI`` is
    DEAD (~0) because the L5 opcode-fetch (block 6) reads the WRONG opcode byte:
    OPCODE_BYTE decodes to 0x00 (LEA) instead of 0x0B (SI). The fetch is a content
    -addressed match of the relayed PC (``EMBED_LO/HI`` nibbles, head 1 Q) against
    each code byte's positional ``ADDR_KEY`` (head 1 K). For the SI PCs (58, 90 --
    low nibble 0xa) the correct code row (the SI opcode byte) and a STRAY zero
    high-byte code row sit in a RAZOR-THIN tie (score 112.71 vs 112.94); the wrong
    zero byte wins by 0.23 and its V copies 0x00 into OPCODE_BYTE.

    The 0.23 margin is created by a DOUBLED ``ADDR_KEY+32`` (3rd address nibble,
    top byte) on the wrong row: it carries 2.0 not 1.0, so its head-1 slot-35
    contribution is ~803 (vs the correct row's ~402), exactly enough to overcome
    its missing mid-nibble match. The doubling is an aliasing collision produced
    at block 3 (L3 ``_pc_byte1_prev_head_spec`` stages ``CLEAN_EMBED`` ->
    ``ADDR_KEY+32`` for the 12-bit code-address match) landing ON TOP of the
    positional ``ADDR_KEY+32`` one-hot (both index 0 for PC < 256) at certain
    zero code bytes (e.g. id450 pos 233). Short programs (var_simple id250, same
    SI PC=58) have NO such colliding zero byte, so their SI fetch wins cleanly and
    OP_SI/MEM_STORE are correct -- this is a LONG-PROGRAM / loop-body root.

    FIX: a flag-gated ``PureFFN`` block scheduled inside L4 (block 4, AFTER the L3
    doubling at block 3, BEFORE the L5 opcode fetch at block 6) that CLAMPS the
    over-counted ``ADDR_KEY+32`` band back to 1.0 on prompt code-byte rows. Per
    cell k a step-function unit fires iff ``ADDR_KEY+32+k >= 1.5`` (i.e. the cell
    was doubled to ~2.0) on an ``IS_BYTE`` row with every register MARKER hard
    NOT-blocked (so only PROMPT code bytes, never an emitted PC/AX/SP/.. value
    byte where L3 head 7 legitimately single-stages the nibble), and subtracts
    1.0 from that cell. A clean single-staged cell (1.0 < 1.5) never fires, so the
    legitimate 12-bit code-address match and every short program are byte-
    identical. With the doubling removed the correct SI code row wins the fetch
    p=1.000 -> OP_SI fires -> MEM_STORE=1 on the in-loop SI stores again
    (verified GPU AR: loop_sum id450 steps 0-10 PC+AX byte-correct; the L15 CAM
    candidate set now contains the genuine stores; teacher-forced LI returns the
    correct value).

    SCOPE / BLUEPRINT (NOT a full flip of id450 yet): this op is the NECESSARY
    upstream half. The in-loop LI still returns 0 in the AUTOREGRESSIVE path
    because the SI store's ADDRESS is never materialized into ``ADDR_KEY`` at the
    MEM store row -- in the AR run EVERY store row has ``ADDR_KEY_set=[]`` (the
    store's emitted MEM address bytes are 0x00000000, identical for the PASSING
    var_simple), so the L15 value-load lookup falls back to ALiBi recency. A
    single-store program (var_simple) is fine on recency; a LOOP has multiple
    intervening stores between the i-store (step 5) and the in-loop LI (step 11)
    so recency picks the wrong (later) store and the load returns 0. The
    downstream lever is to MATERIALIZE the store address into ``ADDR_KEY`` at the
    in-loop MEM store row (the #342 operand-CAM / value-load-CAM family) so the
    L15 CAM can content-address the correct store -- then this op's restored
    MEM_STORE flag (which gates the candidate set) lets the load land. Byte-
    identical flag-OFF (golden ``928e49ec`` / campaign ``53118713`` both UNCHANGED
    vs base bafc16b6). var_simple HOLDS (PASS); func_identity id550 fails
    IDENTICALLY flag-ON and flag-OFF (a pre-existing base failure, not a
    regression).
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_LOOP_LI_FETCH_ADDRKEY_CLAMP", "1") != "0"
    )


def ffn_lint_mull14_demo_enabled() -> bool:
    """Return True iff the cross-op FFN-lint MUL-L14-ENTANGLEMENT demo op is on.

    DEFAULT OFF — opt-in via ``C4_FFN_LINT_MULL14_DEMO=1``. TOOLING-ONLY: a
    faithful, reaches-the-build reproduction of the −60 mul-l14 entanglement
    class for ``tools/lint_cross_op_ffn.py --demo``. It registers ONE PureFFN
    post-op unit on the l14 ALU block, AUTHORED as "MUL-only" (its W_up reads
    ``OP_MUL`` strongly) but with a positive ``b_up`` bias so ``silu(up)`` is
    NON-ZERO even when ``OP_MUL==0`` — the smooth-nonlinearity leak that writes
    the SHARED ``OUTPUT_LO`` band on ADD/SUB/DIV rows too (the exact −60
    mechanism the lint exists to catch). The whole op is registered ONLY when
    the flag is on (lookahead-chain pattern), so a flag-off / production build
    is byte-identical to golden ``4958b35b``.
    """
    return os.environ.get("C4_FFN_LINT_MULL14_DEMO", "0") == "1"


def ffn_lint_clean_demo_enabled() -> bool:
    """Return True iff the cross-op FFN-lint CLEAN-CONTROL branch is on.

    DEFAULT OFF — opt-in via ``C4_FFN_LINT_CLEAN_DEMO=1``. This flag is
    TOOLING-ONLY: it exists so ``tools/lint_cross_op_ffn.py --demo`` has a
    same-layout, in-place modification of a SHARED l14 ALU attention head that
    nonetheless writes ONLY a PRIVATE scratch dim (``TEMP``), touching no
    OUTPUT/ALU residual band any downstream op reads. The lint must PASS it
    (the clean side of the discrimination contract) exactly as it FLAGS the
    band-perturbing mul-l14 change (``C4_NO_STACK0_EMIT``). DEFAULT OFF keeps
    every production build byte-identical to golden ``4958b35b``.
    """
    return os.environ.get("C4_FFN_LINT_CLEAN_DEMO", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Phase 11.A IR exposure: shared empty-IR factory for flag-gated ops whose
# bake bodies are no-ops when their gating flag is off.
def _empty_compiler_ir_factory(dim_positions, HD):
    """Return an empty ``CompilerIR``; used when a flag-gated bake is a no-op."""
    from ..ir import CompilerIR
    return CompilerIR()

# Reverse map (lazily populated): Opcode int value -> "OP_<NAME>" string
# for dim_positions lookup. Module-level so _SetDimProxy can be pickled.
_OP_NAME_CACHE: Dict[int, str] = {}


def _setdim_to_positions(BD) -> Dict[str, int]:
    """Build a ``dim_positions`` dict from a ``_SetDim``-like class.

    Phase 7.D.3 helper. Used by the legacy ``setup_token_embeddings`` /
    ``setup_head_weights`` fallback path when no ``dim_positions`` arg was
    supplied. Walks every public class attribute that resolves to an
    ``int`` so the returned mapping is closed.
    """
    positions: Dict[str, int] = {}
    for name in dir(BD):
        if name.startswith("_"):
            continue
        val = getattr(BD, name, None)
        if isinstance(val, int) and not isinstance(val, bool):
            positions[name] = val
    return positions


def _opcode_name_map() -> Dict[int, str]:
    """Return (and lazily build) the Opcode -> "OP_<NAME>" lookup map."""
    if _OP_NAME_CACHE:
        return _OP_NAME_CACHE
    from ...embedding import Opcode
    _OP_NAME_CACHE.update({
        Opcode.LEA: "OP_LEA", Opcode.IMM: "OP_IMM", Opcode.JMP: "OP_JMP",
        Opcode.JSR: "OP_JSR", Opcode.BZ: "OP_BZ", Opcode.BNZ: "OP_BNZ",
        Opcode.ENT: "OP_ENT", Opcode.ADJ: "OP_ADJ", Opcode.LEV: "OP_LEV",
        Opcode.LI: "OP_LI", Opcode.LC: "OP_LC", Opcode.SI: "OP_SI",
        Opcode.SC: "OP_SC", Opcode.PSH: "OP_PSH",
        Opcode.OR: "OP_OR", Opcode.XOR: "OP_XOR", Opcode.AND: "OP_AND",
        Opcode.EQ: "OP_EQ", Opcode.NE: "OP_NE", Opcode.LT: "OP_LT",
        Opcode.GT: "OP_GT", Opcode.LE: "OP_LE", Opcode.GE: "OP_GE",
        Opcode.SHL: "OP_SHL", Opcode.SHR: "OP_SHR",
        Opcode.ADD: "OP_ADD", Opcode.SUB: "OP_SUB", Opcode.MUL: "OP_MUL",
        Opcode.DIV: "OP_DIV", Opcode.MOD: "OP_MOD",
        Opcode.EXIT: "OP_EXIT", Opcode.NOP: "OP_NOP",
        Opcode.PUTCHAR: "OP_PUTCHAR", Opcode.GETCHAR: "OP_GETCHAR",
    })
    return _OP_NAME_CACHE


class _SetDimProxy:
    """BD-like object that resolves dim names via compiler-allocated positions.

    Module-level (not a closure) so instances of this class can be pickled —
    several runtime modules (``efficient_byte_alu``, ``efficient_wrappers``)
    hold a proxy as ``self.BD``, which means the model object must be
    picklable for the ``compile_full_vm_dynamic`` on-disk cache to work.

    Falls back to ``_SetDim`` for any attribute not in ``dim_positions``
    (e.g. constants like ``NUM_OPCODES``). ``opcode_dim`` resolves via
    ``dim_positions`` so OP_* writes land at compiler-allocated positions
    even under ``pin_io_only=True`` layouts.
    """

    def __init__(self, dim_positions: Dict[str, int]):
        # Store the dim_positions dict so __getattr__ / opcode_dim can use it
        # and so pickling round-trips correctly.
        object.__setattr__(self, "_dim_positions", dict(dim_positions))
        # Mirror declared dim positions as instance attributes for fast access
        # (matches the closure-based proxy's behavior).
        for name, pos in dim_positions.items():
            object.__setattr__(self, name, pos)

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup failed (so the mirrored
        # attrs above shadow this path for declared dims).
        from ...vm_step import _SetDim
        return getattr(_SetDim, name)

    def __getstate__(self):
        # Only persist the dim_positions; on load __setstate__ re-mirrors them.
        return {"_dim_positions": self._dim_positions}

    def __setstate__(self, state):
        object.__setattr__(self, "_dim_positions", dict(state["_dim_positions"]))
        for name, pos in self._dim_positions.items():
            object.__setattr__(self, name, pos)

    def opcode_dim(self, op_value):
        """Resolve op_value -> dim position via dim_positions (override).

        Falls back to _SetDim.opcode_dim if the OP_<NAME> entry isn't in
        dim_positions (e.g. opcodes that aren't declared by the compiler).
        """
        from ...vm_step import _SetDim
        name = _opcode_name_map().get(op_value)
        if name is not None and name in self._dim_positions:
            return self._dim_positions[name]
        return _SetDim.opcode_dim(op_value)


def _as_setdim_proxy(dim_positions: Dict[str, int]):
    """Build an object that mimics _SetDim using compiler dim positions.

    The original `_set_layerN_*` functions reference `BD.MARK_PC`, `BD.OUTPUT_LO`
    etc. We need to give them a BD-like object whose attribute values are the
    integer positions that the compiler chose.

    Returns an object where `proxy.MARK_PC == dim_positions['MARK_PC']`.
    Falls back to `_SetDim` for anything not declared (e.g., constants like
    ``NUM_OPCODES``).

    Note: ``proxy.opcode_dim(op_val)`` is overridden to resolve via
    ``dim_positions`` (looking up ``OP_<name>``) rather than ``_SetDim.OP_*``.
    Without this override, callers like ``_set_opcode_decode_ffn`` would
    write OP_* flags at the LEGACY ``_SetDim`` positions instead of the
    compiler-allocated ones, breaking pin_io_only=True layouts.

    If ``dim_positions`` is already a ``_SetDim``-like object (the legacy
    class itself or a previously-built proxy), return it unchanged. This
    lets legacy umbrella entry points (``vm_step._set_layerN_*``) that
    route through declarative IR factories pass ``BD = _SetDim`` directly
    instead of materializing a mirror dict.
    """
    if not isinstance(dim_positions, dict):
        # Already a class / proxy that supports ``.NAME`` attribute lookup
        # for the dim positions. Skip the dict→proxy wrap.
        return dim_positions
    return _SetDimProxy(dim_positions)


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


# Per-ALU-class opcode gates. Each ALU module's forward applies an
# opcode_mask at the GE→BD writeback stage that zeros all residual writes
# except for the listed BD-format OP_* dims. Sources:
#   - AddSub5StageBlock (``efficient_alu_addsub_split.py``): op_add /
#     op_sub merge at stage 3 → OP_ADD, OP_SUB.
#   - FlattenedALUMul (``efficient_alu_neural.py:_MulCombineStage``):
#     op_mul gate at the combine stage → OP_MUL.
#   - ALUShiftComposite (``efficient_alu_neural.py:ALUShiftComposite``):
#     op_shl + op_shr merge → OP_SHL, OP_SHR.
#   - ALUAndOrXor (``efficient_alu_neural.py:PureNeuralALU(operations=
#     'bitwise')``): per-opcode OR/XOR/AND extract+combine + AX-marker
#     gate → OP_OR, OP_XOR, OP_AND.
# The attach op installs the module as a ``post_op`` whose forward gates
# every residual write on those opcodes. Declaring ``opcodes={...}`` on
# the attach op tells the per-opcode block-skip analyser the attached
# post-op layer is unreachable for any other opcode, unlocking the
# layer skip for the bulk of the opcode table.
_ALU_CLASS_OPCODES = {
    "ALUAddSub": {"OP_ADD", "OP_SUB"},
    "ALUMul": {"OP_MUL"},
    "ALUShift": {"OP_SHL", "OP_SHR"},
    "ALUAndOrXor": {"OP_AND", "OP_OR", "OP_XOR"},
}


def _make_alu_postop_attach_op(name: str, layer_idx: int, alu_cls_name: str,
                               alu_mode: str = 'lookup',
                               same_layer_as: str = None,
                               target_op_name: str = None) -> Operation:
    """Construct an ALU postop-attach Operation.

    Args:
        name: op name (e.g. ``l8_alu_postop_attach``).
        layer_idx: home layer for the postop attach.
        alu_cls_name: ALU class to instantiate (e.g. ``ALUAddSub``).
        alu_mode: ``'lookup'`` (only supported mode for now).
        same_layer_as: B12 backfill (plan §B12 / B10 schema). When provided,
            the returned ``Operation`` declares
            ``requires={"same_layer_as": same_layer_as}`` so the dynamic
            scheduler pins the postop-attach to the same layer as the
            wrapped ALU op (``layerN_alu`` / ``layer11_mul_partial`` etc.).
            Moves the op from ``phase_required_but_undeclared`` to
            ``phase_pinned_by_deps`` in ``analyze_scheduler``. None preserves
            pre-B12 behaviour (no ``requires`` declaration; analyzer flags
            it as ``phase_required_but_undeclared``).
    """
    if alu_mode != 'lookup':
        # TODO(efficient-mode): efficient alu_mode REPLACES ffn rather than
        # wrapping it (see vm_step.py:2385-2434), so the bake_fn semantics
        # differ. Migrate that branch in a follow-up.
        raise NotImplementedError(
            f"alu_mode={alu_mode!r} not yet supported for alu postop attach ops"
        )

    def bake(block, dim_positions, S):
        from ...vm_step import _SetDim
        from ... import efficient_alu_neural as eau
        # ALUAddSub has been replaced by the 5-stage flattened AddSub5StageBlock
        # (see efficient_alu_addsub_split.py). ALUMul / ALUShift are likewise
        # replaced by the already-flattened ``FlattenedALUMul`` /
        # ``ALUShiftComposite`` composites (nn.Sequential of PureFFNs, byte-
        # identical forward). Other ALU classes still come from
        # ``efficient_alu_neural``.
        proxy = _as_setdim_proxy(dim_positions)
        if alu_cls_name == "ALUAddSub":
            from ...efficient_alu_addsub_split import AddSub5StageBlock as alu_cls
            instance = alu_cls(S, proxy)
        elif alu_cls_name == "ALUMul":
            instance = eau.FlattenedALUMul.build_fully_baked(S, proxy)
        elif alu_cls_name == "ALUShift":
            instance = eau.ALUShiftComposite(S, proxy)
        else:
            alu_cls = getattr(eau, alu_cls_name)
            instance = alu_cls(S, proxy)
        # Attach as a post_op (rather than wrapping block.ffn with HybridALUBlock).
        # ``_expand_wrapper_blocks`` then splits each post_op into a passthrough
        # transformer block, preserving the original execution order.
        # Use compiler-allocated dim_positions (via proxy) so the structural
        # ALU wires inputs to layout-correct residual lanes; bare _SetDim
        # breaks pin_io_only=True (IO dims sit at different positions there).
        block.post_ops.insert(0, instance)

    # Phase=1180 + layer_idx*0.01: hybrid wraps must fire AFTER all FFN
    # bakes (including L14 cleanup and convo-IO ops at phases 8.5/10.6/15.1)
    # AND AFTER the dead-unit zero passes (l6_dead_unit_zero=1160,
    # l7_dead_unit_zero=1170 which require the original PureFFN), but BEFORE
    # right_size_ffns (1200) which prunes dead units after wrapping.
    #
    # B12 backfill: ``requires["same_layer_as"]`` pins the postop attach to
    # the same layer as the wrapped ALU op under the dynamic scheduler.
    # Without it, the analyzer flags this op as
    # ``phase_required_but_undeclared`` (the magic 1180+ phase carries the
    # ordering constraint but the dep DAG has no way to see it).
    requires: Dict[str, str] = {}
    if same_layer_as is not None:
        requires["same_layer_as"] = same_layer_as
    # Phase 8.G.6: drop the ``layer_idx=`` literal pin in favour of
    # ``target_op_name=`` pointing at an attn/ffn anchor that resolves to
    # the wrapped ALU op's layer. ``requires["same_layer_as"]`` is the
    # strict-mode dep-edge signal that this co-placement is intentional.
    # ``phase=1180+`` keeps the >= 100 post-pass short-circuit in the
    # strict admission gate (``_current_layer_for_strict`` returns
    # ``floor(phase) = 1180`` which the categoriser treats as ``ok``).
    #
    # The factories pass ``target_op_name=<wrapped op's anchor>`` because
    # the wrapped ``layerN_alu`` is itself a kind="block" op and
    # ``Operation.target_op_name`` can only reference attn/ffn ops (see
    # ``ModelLayout.resolve_block_op_layer``). When no ``target_op_name``
    # is provided we fall back to the legacy ``layer_idx`` pin (which
    # still triggers the strict-mode flag).
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind="block",
        target_op_name=target_op_name,
        layer_idx=layer_idx if target_op_name is None else None,
        bake_fn=bake,
        phase=1180 + layer_idx * 0.01,
        migrated=True,
        requires=requires,
        # Tier A opcode gating: the installed post_op module gates every
        # OUTPUT/CARRY write on the listed opcodes (see ``_ALU_CLASS_OPCODES``
        # above for the per-module derivations).
        opcodes=set(_ALU_CLASS_OPCODES.get(alu_cls_name, set())),
    )


def _ensure_l11_mul_module(block, S, dim_positions=None):
    """Get or install the FlattenedALUMul module on ``block.ffn``.

    The 9 phase-ordered installer ops each call this helper; the first one
    (lowest phase) installs the module, the rest re-use it. Idempotent.

    Args:
        block: target transformer block (FlattenedALUMul is installed on
            ``block.ffn``).
        S: number of sequence positions (passed through to FlattenedALUMul).
        dim_positions: optional dict mapping dim name -> start position. When
            provided, FlattenedALUMul receives a `_SetDim`-shaped proxy whose
            attribute values are the compiler-allocated positions; without
            this, the sub-stages (BDToGEConverter, _BDToGEStage,
            _MulCombineStage, _GEToBDStage) bake weights at the legacy
            ``_SetDim`` positions and silently mis-address ALU_LO/HI,
            AX_CARRY_LO/HI, OP_MUL, OUTPUT_LO/HI etc. under
            ``pin_io_only=True``. Falls back to raw ``_SetDim`` when None
            (legacy hand-set callers).
    """
    from ...efficient_alu_neural import FlattenedALUMul
    from ...vm_step import _SetDim
    existing = getattr(block, "ffn", None)
    if isinstance(existing, FlattenedALUMul):
        return existing
    BD = _as_setdim_proxy(dim_positions) if dim_positions is not None else _SetDim
    module = FlattenedALUMul(S, BD)
    block.ffn = module
    return module


# ---------------------------------------------------------------------------
# 4-stage SHL/SHR ops (efficient-mode replacement for ALUShift wrapper).
#
# Each op is kind="ffn" at phase=13 so the 4 ops + ``layer13_shifts`` all
# share L13's FFN slot (phase-equality => shared (layer, kind) slot per
# ``LayerCompiler._assign_layers``). The bake_fns cooperate:
#
#   1. bdtoge  : install the ``ALUShiftComposite`` on ``block.ffn`` and assign
#                the bdtoge stage. Subsequent bakes look up the existing
#                composite via ``block.ffn``.
#   2. precompute : assign the precompute stage onto the composite. (No-op if
#                   the composite was already fully built by another path.)
#   3. select  : same for select.
#   4. getobd  : same for getobd.
#
# Conceptually these are 4 distinct compiler ops carrying ownership of the
# 4 sub-FFN stages. Mechanically they share one layer because the rest of
# ``set_vm_weights`` (legacy_bake) still hardcodes ``model.blocks[14..16]``
# for downstream layers; spreading the stages across 4 layers would shift
# those indices and break that legacy bake until it migrates too. Once the
# downstream legacy bakes follow the layout, the phases can be split into
# 13.0/13.1/13.2/13.3 and the stages will land in their own layers.
# ---------------------------------------------------------------------------


class _ALUShiftCompositeBuilder:
    """Mutable holder shared across the 4 stage bake_fns + the install op.

    The compiler may assign the 4 ffn stage ops to whichever block its dep
    analyser picks (often a block far from the legacy ``model.blocks[13]``).
    The install op (kind="block", layer_idx=13) is what actually swaps the
    L13 ``block.ffn`` for the composite. The shared builder lets stage bakes
    populate the composite from any FFN module they happen to receive.
    """

    def __init__(self):
        self.composite = None

    def ensure(self, S, BD_proxy):
        from ...efficient_alu_neural import ALUShiftComposite
        if self.composite is None:
            self.composite = ALUShiftComposite(S, BD_proxy)
        return self.composite


# ---------------------------------------------------------------------------
# L10 DIV/MOD ALU flattening (2026-05-10)
#
# The previous lookup-mode override
#   model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)
# in ``set_vm_weights`` and the efficient-mode append
#   block.post_ops.append(EfficientDivMod_Neural(S, _SetDim))
# in ``make_l10_post_op_attach_op`` both wrapped 3 logical sub-stages
# (BD→GE convert, long-division pipeline, GE→BD convert) inside a single
# ``PureNeuralALU(operations='div_mod')`` runtime class (alias
# ``ALUDivMod`` / ``EfficientDivMod_Neural``). The 4 ops below split that
# wrapper into discrete compiler operations:
#
#   phase=10.0  install BD → GE converter         (FlattenedDivMod.bd_to_ge)
#   phase=10.1  install long-division pipeline    (FlattenedDivMod.div_layers + mod_layers)
#                                                  = ClearDivSlotsFFN +
#                                                    LongDivisionModule +
#                                                    EmitDivResultModule per opcode
#   phase=10.2  install GE → BD converter         (FlattenedDivMod.ge_to_bd)
#   phase=10.8  install composite onto post_ops   (model.blocks[10].post_ops.append)
#
# The first 3 stage ops are kind="block", layer_idx=10. They run after
# `make_l10_post_op_attach_op` (phase=10.7) since 10.0/10.1/10.2 are < 10.7
# only in numeric-phase comparison — but since BLOCK ops sort by
# (layer_idx, phase), the smaller phases run FIRST. That's fine: the
# first 3 ops only construct sub-stages on a builder; nothing depends on
# `block.post_ops` until the install op (phase=10.8) actually inserts
# the composite.
#
# The install op (phase=10.8, kind="block", layer_idx=10) appends the
# fully-constructed FlattenedDivMod composite to ``block.post_ops``.
# It runs AFTER `make_l10_post_op_attach_op` (phase=10.7) which appends
# the standard L10 post_ops (BinaryOpByteZeroingPostOp etc.) but no longer
# appends EfficientDivMod_Neural / DivModModule.
#
# The legacy lookup-mode override in set_vm_weights
# (`model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)`) is
# also removed so the composite isn't clobbered.
#
# Forward is byte-identical to the previous EfficientDivMod_Neural — see
# ``FlattenedDivMod.forward`` in efficient_alu_divmod_split.py.
# ---------------------------------------------------------------------------


class _FlattenedDivModBuilder:
    """Mutable holder shared across the 4 cooperating ops.

    Each of the 4 ops accesses the same ``FlattenedDivMod`` instance via
    this builder. Stage ops (phase=10.0/10.1/10.2) install one sub-stage
    each; the install op (phase=10.8) appends the fully-assembled composite
    to ``model.blocks[10].post_ops``.

    Idempotent: ``ensure`` returns the existing composite if any.
    """

    def __init__(self):
        self.composite = None

    def ensure(self, S, BD_proxy):
        from ...efficient_alu_divmod_split import FlattenedDivMod
        if self.composite is None:
            self.composite = FlattenedDivMod(S, BD_proxy)
        return self.composite


def setup_token_embeddings(embed_weight, dim_positions: Dict[str, int] = None) -> None:
    """Bake the per-token embedding values using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path uses auto-allocated positions. Falls back to _SetDim when
    dim_positions is None.

    Phase 7.D.3 migration: replaced the per-token imperative writes with a
    call to ``CompilerIR.lower_token_embeddings`` using the same
    ``_embedding_bake_rules`` list that the active production op
    ``make_embedding_bake_op`` uses -- single source of truth.

    Args:
        embed_weight: nn.Embedding.weight tensor [vocab, d_model].
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import _SetDim
    from ..ir import CompilerIR
    # Local import to avoid module-load cycle (model_ops imports from shared).
    from .model_ops import _embedding_bake_rules

    if dim_positions is None:
        dim_positions = _setdim_to_positions(_SetDim)

    V = embed_weight.shape[0]

    with torch.no_grad():
        embed_weight.zero_()

    # Model-like shim so ``lower_token_embeddings`` can resolve
    # ``model.embed.embed.weight``. ``.head`` is stubbed so attribute
    # resolution succeeds even though the embedding bake never writes there.
    class _InnerEmbed:
        def __init__(self, w):
            self.weight = w

    class _OuterEmbed:
        def __init__(self, w):
            self.embed = _InnerEmbed(w)

    class _ModelShim:
        def __init__(self, w):
            self.embed = _OuterEmbed(w)
            self.head = None

    ir = CompilerIR()
    ir.embeddings.extend(_embedding_bake_rules(V))
    ir.lower_token_embeddings(_ModelShim(embed_weight), dim_positions)


def setup_head_weights(head, dim_positions: Dict[str, int] = None) -> None:
    """Bake the output-projection head weights using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path can call it with auto-allocated dim positions instead of
    _SetDim constants. When `dim_positions` is None, falls back to _SetDim
    (backward-compat with hand-set path).

    Phase 7.D.3 migration: replaced the per-byte / per-marker imperative
    writes with a call to ``CompilerIR.lower_token_embeddings`` using the
    same ``_head_bake_rules`` list that the active production op
    ``make_head_bake_op`` uses -- single source of truth.

    Args:
        head: The model.head nn.Linear(d_model, vocab_size) module.
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import _SetDim
    from ..ir import CompilerIR
    # Local import to avoid module-load cycle (model_ops imports from shared).
    from .model_ops import _head_bake_rules

    if dim_positions is None:
        dim_positions = _setdim_to_positions(_SetDim)

    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

    # Model-like shim so ``lower_token_embeddings`` can resolve
    # ``model.head.weight`` / ``model.head.bias``. ``lower_token_embeddings``
    # also touches ``model.embed.embed.weight`` to set ``embed_weight`` when
    # ``chosen`` is truthy, even when no embed-target rules exist; stub
    # ``.embed`` to a no-op object so attribute resolution succeeds.
    class _NullEmbed:
        weight = None

    class _NullOuter:
        embed = _NullEmbed()

    class _ModelShim:
        def __init__(self, h):
            self.head = h
            self.embed = _NullOuter()

    vocab_size = head.weight.shape[0]
    ir = CompilerIR()
    ir.embeddings.extend(_head_bake_rules(vocab_size, dim_positions))
    ir.lower_token_embeddings(_ModelShim(head), dim_positions)


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
# - OP_LEV/BZ/BNZ: decoded at MARK_PC by the L5 FFN all-step PC-marker
#   opcode decode (see vm_step.py); set at MARK_AX by the standard L5
#   opcode decoder. ACTIVE_OPCODE_PRTF/READ: legacy conversational-I/O
#   layout placeholders (no longer written from Python).
# - MARK_THINKING_START/END: baked into the embedding table on
#   THINKING_START/END tokens (see ``setup_token_embeddings``).
# - MEM_STORE / ADDR_KEY: written by `_inject_mem_store` /
#   `_inject_mem_metadata` for memory ops. (MEM_EXEC@468 is retained in the
#   IO set as a layout placeholder — Phase A 2026-05-11 removed the writes
#   and external-hints API but kept the dim slot so the compact-IO layout
#   stays stable. The slot is aliased by IO_FORMAT_POS.)
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
    # Active-opcode dims (decoded at PC by L5 FFN; layout retained for
    # ACTIVE_OPCODE_PRTF/READ as conversational-I/O placeholders).
    "OP_LEV", "OP_BZ", "OP_BNZ",
    "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
    # Memory injection slots (_inject_mem_store, _inject_mem_metadata).
    # MEM_EXEC is a retained placeholder — see header comment above.
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
        pin_io_only: if True, the dims in `_IO_REQUIRED_DIMS` (the
            externally-observable dims read/written by token embedding, the
            output head, and `NeuralVMEmbedding._inject_*` runtime injectors)
            are pinned to a *compact, contiguous block starting at position
            0*, in declaration order. Every non-IO dim is bump-pointer
            allocated by the compiler above the IO block. This unlocks
            compiler-driven internal dim allocation AND shrinks d_model:
            instead of pinning IO dims at their scattered `_SetDim` positions
            (which span up to ~507 with large gaps, forcing unpinned dims to
            stack on top for d_model ~1038), they are laid out densely so
            d_model collapses to roughly (IO total size) + (non-IO total
            size). Code that still reads `_SetDim.X` *directly* will get the
            wrong position — all baked weights must resolve dim positions
            through `dim_positions` (e.g., via `_as_setdim_proxy`). The
            `pin_to_setdim` flag is ignored when `pin_io_only=True`. Defaults
            to False for backward compatibility.
    """
    from ...vm_step import _SetDim
    from ...constants import INSTR_WIDTH  # noqa: F401 (touched for completeness)

    # Single-dim flags
    one_dim = [
        "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
        "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
        "MARK_SE_ONLY", "MARK_STACK0",
        "MARK_THINKING_START", "MARK_THINKING_END",
        "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
        "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "CMP_GROUP",
        # B7-1: in-step freshness lifecycle bit (L1 attn head 5; see
        # _SetDim.IN_STEP_FRESH docstring for semantics).
        "IN_STEP_FRESH",
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
        "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
        "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
        "IO_IS_PUTCHAR", "IO_OUTPUT_READY",
        "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV",
        # Phase 9.C: OP_LEV_PREV_STEP alias retired - corpus reads now
        # use SSA spellings (OP_LEV.<writer>.-1) instead of the
        # numeric alias.
        "OP_LI", "OP_LC",
        "OP_SI", "OP_SC", "OP_PSH",
        "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_NE", "OP_LT",
        "OP_GT", "OP_LE", "OP_GE", "OP_SHL", "OP_SHR",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_EXIT", "OP_NOP", "OP_PUTCHAR", "OP_GETCHAR",
        "MEM_STORE", "MEM_ADDR_SRC",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        # MEM_EXEC is a layout placeholder — its writes were removed in
        # Phase A (2026-05-11) but the slot is retained so the compact-IO
        # layout is stable. IO_FORMAT_POS@468 aliases MEM_EXEC.
        "OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP", "MEM_EXEC",
        "OPCODE_BASE",
        # B7-4 / B6-K slot 97: ADDR_B0 lifecycle "VALID" bit. Written 1.0 by
        # L13 mem-addr-gather at MEM val byte positions whenever the ADDR_B0
        # one-hot lanes carry freshly-computed nibbles. L10 tail addr0 family
        # gates on this to distinguish fresh ADDR_B0 evidence from stale
        # residue (see B4-H tail-correction-family PLAN §3.2).
        "ADDR_B0_VALID",
        # Conversational I/O state (aliases noted in _SetDim):
        # IO_FORMAT_POS@468 aliases MEM_EXEC, IO_IN_OUTPUT_MODE@469 and
        # IO_OUTPUT_COMPLETE@470 are dedicated, LAST_WAS_BYTE@503 is
        # dedicated. Declared unconditionally so the compiler accepts the
        # convo-io migrated ops' reads/writes even when the flag is False.
        "IO_FORMAT_POS", "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "LAST_WAS_BYTE",
        # Conversational I/O state dims that were previously left out of the
        # declaration list and fell back to bare `_SetDim` positions via the
        # proxy. With `pin_io_only=True` those legacy positions collide with
        # compiler-allocated dims (ALU_LO, AX_FULL_*, OPCODE_BYTE_HI, ...);
        # declare them so the compiler hands out unique positions. Bake
        # functions that reference them (L2 lookback head, L3 state init,
        # null-terminator detection) only fire when conversational I/O is
        # enabled — they remain unused under the default smoke config but
        # must have collision-free positions in either layout.
        "LAST_WAS_THINKING_START", "LAST_WAS_THINKING_END",
        "LAST_WAS_IO_STATE_EMIT_BYTE", "LAST_WAS_IO_STATE_EMIT_THINKING",
        "IO_IS_PRTF", "IO_IS_READ", "IO_STATE", "IO_OUTPUT_COUNT",
        "IO_IS_TOOL_CALL",
        "NEXT_IO_STATE_EMIT_BYTE", "NEXT_IO_STATE_EMIT_THINKING",
        # B7-2 SP_BYTE0_IS_F8: L7 head-6 producer, 1.0 only when carry-forward
        # proves SP byte 0 is 0xF8 (slot 95 — dead L0 H5+0).
        "SP_BYTE0_IS_F8",
        # B7-1 IN_STEP_FRESH: L1 head-5 producer with ALiBi slope 0.5,
        # decays from 1.0 immediately after STEP_BOUNDARY toward 0.0;
        # resets at next STEP_BOUNDARY. Replaces HAS_SE -1e9 hammer
        # for L10 tail_* rules (slot 96 — dead L0 H5+1).
        "IN_STEP_FRESH",
        # B7-5 SP_GATHERED_THIS_STEP: 1.0 at MARK_SP positions after L8
        # SP gather has fired in the current step. Produced by
        # ``make_layer8_sp_gathered_sentinel_op`` (L8 FFN, phase 8.6);
        # consumed by L10 tail_sp_marker_* rules. See
        # ``investigation/bd-dim-usage-map`` REPORT Section 5.
        "SP_GATHERED_THIS_STEP",
        # C5 BZ branch-target re-fire fix (2026-06-09): 1.0 at MARK_PC
        # positions on BZ-taken steps. Written by
        # ``post_l9_bz_bnz_pc_override``; consumed by the SAME op on the
        # NEXT step via the ``BZ_TARGET_FRESH.*.-1`` cross-step alias as
        # a gate term on the OUTPUT_LO cancel band. See
        # ``docs/BZ_TARGET_FRESH_CROSS_STEP_2026_06_09.md``.
        "BZ_TARGET_FRESH",
    ]
    # 7-dim threshold head outputs (one per marker type)
    # H1_DUMP is a same-position alias of H1 (slots 67..73) declared AFTER
    # H1 so the alias machinery picks up H1's resolved position. The AX
    # byte-1 DUMP carry head writes the re-supplied byte-1 H1 one-hot here
    # on carried steps; distinct name keeps it off the 54 same-step H1
    # readers' dep-graph edges. See _ALIAS_OF below.
    seven_dim = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
                 "L1H0", "L1H1", "L1H2", "L1H4", "L2H0", "H1_DUMP"]
    # 16-dim nibble groups
    sixteen_dim = ["EMBED_LO",
                   "EMBED_HI",
                   "OUTPUT_LO", "OUTPUT_HI",
                   # B9 OUTPUT_HI split: OUTPUT_HI_THIS_STEP is the
                   # canonical name for the same-step write band. Same
                   # numeric base as OUTPUT_HI in _SetDim (190) so baked
                   # weight indices are byte-identical; the alias keeps
                   # ``BD.OUTPUT_HI`` lookups in legacy bake bodies
                   # working unchanged. Phase 9.C retired the sibling
                   # ``OUTPUT_HI_PREV_STEP`` alias - cross-step readers
                   # (layer3_carry_forward_attn head 5,
                   # layer8_head6_ax_carry_refresh) now use SSA reads
                   # plus ``requires["after"]`` for the cross-step
                   # boundary. See docs/B9_OUTPUT_HI_SPLIT_SPEC.md.
                   "OUTPUT_HI_THIS_STEP",
                   "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
                   "FETCH_LO", "FETCH_HI", "MUL_ACCUM", "DIV_STAGING",
                   "AX_FULL_LO", "AX_FULL_HI",
                   "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                   "ADDR_B0_LO",
                   "ADDR_B1_LO",
                   "ADDR_B2_LO",
                   "ADDR_B0_HI",
                   "ADDR_B1_HI",
                   "ADDR_B2_HI",
                   "FORMAT_PTR_LO", "FORMAT_PTR_HI",
                   "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"]
    # Phase 9.C: CARRY/CMP/ADDR_KEY/TEMP PREV_STEP aliases retired -
    # corpus reads now use SSA spellings (``<DIM>.<writer>.-1``).
    four_dim = ["CARRY"]
    eight_dim = ["CMP"]
    forty_eight_dim = ["ADDR_KEY"]
    thirty_two_dim = ["TEMP"]

    # Cursor for the compact IO block when pin_io_only=True. IO dims are
    # pinned at consecutive positions starting at 0, in declaration order
    # (the order of the `one_dim` / `seven_dim` / ... lists below). Non-IO
    # dims are left unpinned and bump-pointer-allocated above the IO block
    # by `_allocate_dims`.
    io_cursor = [0]

    # B9 OUTPUT_HI split: alias map. Dim names on the LHS share the same
    # numeric position as the dim on the RHS. The alias must be declared
    # AFTER the base in the relevant size-bucket list so the base's
    # pinned position is already in ``compiler._pinned`` before the alias
    # is declared. See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §6.4.
    _ALIAS_OF = {
        "OUTPUT_HI_THIS_STEP": "OUTPUT_HI",
        # AX byte-1 DUMP carry: H1_DUMP shares H1's 7 slots (67..73). The
        # carry head writes the re-supplied byte-1 one-hot here on carried
        # steps; the LM head reads the physical slots so emission is
        # byte-identical, while the distinct name avoids the H1-write
        # 2-cycle. See vm_step.py:_SetDim.H1_DUMP.
        "H1_DUMP": "H1",
        # Phase 9.C: all ``*_PREV_STEP`` aliases (OUTPUT_HI/LO, TEMP,
        # ADDR_KEY, ALU_LO, AX_CARRY_{LO,HI}, EMBED_{LO,HI}, ADDR_B*_*,
        # CARRY, CMP, OP_LEV, OPCODE_BYTE_LO) were retired now that
        # Phase 9.B migrated every cross-step reader to its SSA spelling
        # (``<DIM>.<writer>.-1``). The numeric-position aliasing was
        # purely cosmetic since each alias shared its base's position.
    }

    def _declare(name, size):
        if not hasattr(_SetDim, name):
            return
        # Aliases inherit the base dim's position (in BOTH pinning modes)
        # so byte-identical residual cells are guaranteed regardless of
        # compaction layout. We declare via ``alias_of=`` so the compiler
        # resolves the position at _allocate_dims time, even when the base
        # is bump-pointer-allocated (which happens in pin_io_only=True
        # mode for non-IO-required dims like AX_CARRY_LO/HI, ALU_LO,
        # ADDR_KEY, TEMP, OUTPUT_LO).
        base = _ALIAS_OF.get(name)
        if base is not None:
            existing = getattr(compiler, "_pinned", {}) or {}
            if base in existing:
                pinned = existing[base]
            else:
                # Base is unpinned (will be bump-allocated). Fall back to
                # _SetDim if pin_to_setdim is set; otherwise leave
                # pinned=None — the compiler's alias machinery resolves the
                # position post-allocation via the ``alias_of=`` link.
                pinned = getattr(_SetDim, base, None) if pin_to_setdim else None
            compiler.declare_dim(name, size, pinned=pinned, alias_of=base)
            return
        if pin_io_only:
            if name in _IO_REQUIRED_DIMS:
                # Compact: assign consecutive positions starting at 0,
                # ignoring _SetDim's scattered legacy positions. Without
                # this compaction, IO dims pinned at their _SetDim positions
                # leave huge gaps (max IO position ~507) and force unpinned
                # dims to stack on top, producing d_model ~1038.
                pinned = io_cursor[0]
                io_cursor[0] += size
            else:
                pinned = None
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
    # Internal-only STACK0 byte flags. Declare these last so adding them does
    # not renumber any pre-existing compiler-allocated non-IO dims.
    for name in ("STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"):
        _declare(name, 1)

    # V2/G7 LEV detector head output dims (Phase 9 spike from
    # docs/CONTROL_FLOW_DETECTOR_HEADS.md §2.3). These are fresh residual
    # bands written by ``make_lev_detector_head_op`` (L8 attn head, default
    # ``enable=False``). The head's V/O projection materialises the saved
    # PC / BP / SP from the prev-step LEV row at the current-step PC/BP/SP
    # marker rows; downstream readers (L9 alu, L8 sp_gather_bake) can prefer
    # the detector dim when present, falling back to the existing
    # OUTPUT_LO/HI writes from L16 ``layer16_lev_routing`` on non-LEV-
    # following steps.
    #
    # The dims are declared unconditionally so the residual layout / dim
    # registry is stable regardless of the head's ``enable`` flag. The
    # explicit ``compiler.declare_dim`` call bypasses the
    # ``hasattr(_SetDim, name)`` guard inside ``_declare`` -- these slots
    # have no ``_SetDim`` legacy position (they are V2-native and emerge
    # from the bump-pointer allocator above the STACK0_BYTE3 high-water
    # mark; ``pinned=None`` lets the compiler pick the lowest free slot).
    #
    # PC has lo/hi nibble pair (mirrors OUTPUT_LO/HI structure); BP and SP
    # are single 16-wide bands. See docs/CONTROL_FLOW_DETECTOR_HEADS.md
    # §2.3 for the residual-stream rationale and §2.2 for the V/O write
    # table.
    for name in (
        "PC_VIA_LEV_DETECTOR_LO",
        "PC_VIA_LEV_DETECTOR_HI",
        "BP_VIA_LEV_DETECTOR",
        "SP_VIA_LEV_DETECTOR",
    ):
        compiler.declare_dim(name, 16, pinned=None)

    # Wave 1 A3: STACK0_BYTE_VAL_h_LO/HI family. 16-wide nibble bands
    # holding the AX byte h value broadcast to the matching STACK0 byte
    # row during PSH. Producer is ``layer10_psh_ax_broadcast`` (3 new
    # heads at L10 slots 8/9/10); consumer is the L14 ``mem_generation``
    # read migration (heads 5/6/7). The dim family was scaffolded in
    # Wave 1 A1 (commit c31897aa); _SetDim has no legacy positions for
    # these slots, so ``pinned=None`` lets the bump-pointer allocator
    # place them above the high-water mark in the compat path.
    # See docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md.
    for name in (
        "STACK0_BYTE_VAL_1_LO",
        "STACK0_BYTE_VAL_1_HI",
        "STACK0_BYTE_VAL_2_LO",
        "STACK0_BYTE_VAL_2_HI",
        "STACK0_BYTE_VAL_3_LO",
        "STACK0_BYTE_VAL_3_HI",
    ):
        compiler.declare_dim(name, 16, pinned=None)

    # 2026-06-10: STEP_END register-presence broadcast family. Each
    # ``SE_REG_<MARK>_PRESENT`` is a 1-wide flag written at MARK_SE_ONLY
    # rows by the L1 ``layer1_threshold_attn`` head 6 within-step relay
    # (only SE_REG_AX_PRESENT is wired in this commit; the other 5 slots
    # are declared scaffolding for follow-on broadcast heads). All slots
    # are unpinned so the bump-pointer allocator places them above the
    # wave-aligned high-water mark in the compat path. See
    # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md.
    for name in (
        "SE_REG_AX_PRESENT",
        "SE_REG_PC_PRESENT",
        "SE_REG_SP_PRESENT",
        "SE_REG_BP_PRESENT",
        "SE_REG_STACK0_PRESENT",
        "SE_REG_MEM_PRESENT",
    ):
        compiler.declare_dim(name, 1, pinned=None)

    # 2026-06-10 Wave A v2: register-tagged STEP_END operand relay.
    # Written at MARK_SE_ONLY by ``layer9_step_end_operand_relay`` (two
    # attn heads in L9 attn), consumed by the migrated L9 CMP rules
    # (``_layer9_cmp_rules``, gated on MARK_SE_ONLY). The SE_ prefix
    # keeps the operand bands scoped to the SE row so they do not
    # collide with downstream readers of the raw ALU_LO/HI/CARRY/CMP
    # bands. Pinned to the ``_SetDim.SE_*`` positions (837..911) so
    # they sit above the existing dim layout and don't share slots
    # via the liveness allocator with unrelated live dims (the L9 CMP
    # rules' writes to SE_<NAME> at MARK_SE_ONLY would clobber any
    # collided dim's writes at non-SE rows otherwise). See
    # ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`` and memory
    # note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
    # The SE_* relay dims MUST NOT share slots with other live dims:
    # the relay's attention head produces a small leak at non-SE rows
    # (softmax over -ALiBi penalty distributes some mass to nearby K
    # rows, producing residual contribution at non-SE Q rows even with
    # the score-only slot-0 design). Sharing a slot with a dim that
    # fires at MARK_AX (e.g. IN_STEP_FRESH, SP_BYTE0_IS_F8) would
    # corrupt the partner's value. Leave the dims unpinned so the
    # liveness allocator places them above the existing layout BUT
    # the auto-share is disabled for SE_* by the bumped d_model: with
    # explicit size declarations the bump-pointer puts them above
    # the IO/scratch high-water mark unless the liveness pass coalesces
    # them. The L1 head 6 SE_REG_* family takes the same approach.
    for name, size in (
        ("SE_ALU_LO", 16),
        ("SE_ALU_HI", 16),
        ("SE_AX_CARRY_LO", 16),
        ("SE_AX_CARRY_HI", 16),
        ("SE_CMP", 4),
        ("SE_OP_EQ", 1),
        ("SE_OP_NE", 1),
        ("SE_OP_LT", 1),
        ("SE_OP_GT", 1),
        ("SE_OP_LE", 1),
        ("SE_OP_GE", 1),
        ("SE_CMP_GROUP", 1),
    ):
        compiler.declare_dim(name, size, pinned=None)

    # ------------------------------------------------------------------
    # Qwen R1 — opt-in NORM_COMPENSATOR slot
    # ------------------------------------------------------------------
    # When ``C4_QWEN_EXPORT_COMPAT=1`` is set, declare a width-1 dim
    # that the ``norm_compensator_seed`` model bake populates with the
    # known constant ``K`` (1000.0) for every token id. Declared as
    # unpinned so the bump-pointer allocator places it at the next
    # available position above the wave-aligned IO/scratch high-water
    # mark. With the flag OFF the dim is not declared, so d_model and
    # the residual layout stay byte-identical to pre-R1 main.
    # See ``make_norm_compensator_seed_op`` and
    # docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md §R1.
    import os as _os
    if _os.environ.get("C4_QWEN_EXPORT_COMPAT") == "1":
        compiler.declare_dim("NORM_COMPENSATOR", 1, pinned=None)

    # ------------------------------------------------------------------
    # width=2 MUL — MUL_RESULT_HI byte-1 result band (2026-06-13)
    # ------------------------------------------------------------------
    # The dedicated high-byte result band for the width=2 (8-bit x 8-bit
    # -> 16-bit) MUL (MUL_RESULT_HI_LO/HI; nib2 -> _LO, nib3 -> _HI) is NO
    # LONGER declared here. It is now declared OP-LOCALLY (next to the wide_mul
    # op in ``ops/alu_ops.py``) via ``register_residual_band(..., flag=
    # mul_width2_enabled)`` and AUTO-COLLECTED into ``extra_residual_dims`` at
    # the top of ``compile_full_vm_dynamic`` (see
    # ``ops/residual_band_registry.py``) so the d_model widen is
    # HEAD-DIM-PRESERVING (the auto-widen captures the base head_dim BEFORE the
    # extra bands are declared and rounds up to a multiple of it, ADDING heads
    # instead of repartitioning existing ones).
    # Declaring it here ran BEFORE the base_head_dim capture and re-derived
    # head_dim from the widened width, scrambling attention -> regressed
    # test_bnz_branch. See ``compile_full_vm_dynamic`` and
    # docs/MUL_WIDTH2_WIDEN_2026_06_13.md. With the flag OFF nothing is
    # declared, so d_model / the residual layout stay byte-identical to
    # pre-width2 main (920).
