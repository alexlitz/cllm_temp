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
    """
    return os.environ.get("C4_MUL_WIDTH2", "1") != "0"


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
    """
    return os.environ.get("C4_OPERAND_FROM_MEMSP", "0") == "1"


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
    byte — see ``l1_ops._layer1_ffn_rules``.

    This flag does NOT itself drop the emission (that machinery — the L0
    marker-transition chain, ``Token.STEP_TOKENS``, the DraftVM oracle, the
    decode offsets — lives on the ``proto/drop-stack0-emit-measure`` branch).
    It is the per-op consultation point so flag-off is byte-identical.
    """
    return os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"


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
