"""Shared helpers and constants used by the per-layer op modules.

Extracted from the legacy ``migrated_ops.py`` (2026-05-11) so the per-layer
factory modules can import these without circular dependencies. The public
API is unchanged: ``from c4_release.neural_vm.unified_compiler.migrated_ops
import _as_setdim_proxy, declare_setdim_compat_dims, ...`` continues to work
via the re-export shim in ``migrated_ops.py``.
"""

from typing import Dict
import torch.nn as nn

from ..layer_compiler import Operation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """
    from ...vm_step import _SetDim
    from ...embedding import Opcode

    # Reverse map: Opcode int value -> "OP_<NAME>" string for dim_positions lookup
    _OP_NAME = {
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
    }

    class _Proxy:
        # Inherit class methods and constants from _SetDim via __getattr__ fallback
        def __getattr__(self, name):
            return getattr(_SetDim, name)

        def opcode_dim(self, op_value):
            """Resolve op_value -> dim position via dim_positions (override).

            Falls back to _SetDim.opcode_dim if the OP_<NAME> entry isn't in
            dim_positions (e.g. opcodes that aren't declared by the compiler).
            """
            name = _OP_NAME.get(op_value)
            if name is not None and name in dim_positions:
                return dim_positions[name]
            return _SetDim.opcode_dim(op_value)

    proxy = _Proxy()
    # Override with compiler-declared positions for declared dims
    for name, pos in dim_positions.items():
        object.__setattr__(proxy, name, pos)
    return proxy


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


def _make_alu_postop_attach_op(name: str, layer_idx: int, alu_cls_name: str,
                               alu_mode: str = 'lookup') -> Operation:
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
        # (see efficient_alu_addsub_split.py). Other ALU classes still come
        # from efficient_alu_neural.
        if alu_cls_name == "ALUAddSub":
            from ...efficient_alu_addsub_split import AddSub5StageBlock as alu_cls
        else:
            alu_cls = getattr(eau, alu_cls_name)
        # Attach as a post_op (rather than wrapping block.ffn with HybridALUBlock).
        # ``_expand_wrapper_blocks`` then splits each post_op into a passthrough
        # transformer block, preserving the original execution order.
        # Use compiler-allocated dim_positions (via proxy) so the structural
        # ALU wires inputs to layout-correct residual lanes; bare _SetDim
        # breaks pin_io_only=True (IO dims sit at different positions there).
        proxy = _as_setdim_proxy(dim_positions)
        block.post_ops.insert(0, alu_cls(S, proxy))

    # Phase=1180 + layer_idx*0.01: hybrid wraps must fire AFTER all FFN
    # bakes (including L14 cleanup and convo-IO ops at phases 8.5/10.6/15.1)
    # AND AFTER the dead-unit zero passes (l6_dead_unit_zero=1160,
    # l7_dead_unit_zero=1170 which require the original PureFFN), but BEFORE
    # right_size_ffns (1200) which prunes dead units after wrapping.
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind="block",
        layer_idx=layer_idx,
        bake_fn=bake,
        phase=1180 + layer_idx * 0.01,
        migrated=True,
    )


def _ensure_l11_mul_module(block, S):
    """Get or install the FlattenedALUMul module on ``block.ffn``.

    The 9 phase-ordered installer ops each call this helper; the first one
    (lowest phase) installs the module, the rest re-use it. Idempotent.
    """
    from ...efficient_alu_neural import FlattenedALUMul
    from ...vm_step import _SetDim
    existing = getattr(block, "ffn", None)
    if isinstance(existing, FlattenedALUMul):
        return existing
    module = FlattenedALUMul(S, _SetDim)
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

    Args:
        embed_weight: nn.Embedding.weight tensor [vocab, d_model].
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import Token, _SetDim

    def D(name):
        if dim_positions is not None and name in dim_positions:
            return dim_positions[name]
        return getattr(_SetDim, name)

    V = embed_weight.shape[0]

    with torch.no_grad():
        embed_weight.zero_()

        # CONST=1 for every token
        const = D("CONST")
        for tok in range(V):
            embed_weight[tok, const] = 1.0

        # Marker tokens
        is_mark = D("IS_MARK")
        for tok, dim_name in [
            (Token.REG_PC, "MARK_PC"),
            (Token.REG_AX, "MARK_AX"),
            (Token.REG_SP, "MARK_SP"),
            (Token.REG_BP, "MARK_BP"),
            (Token.MEM, "MARK_MEM"),
            (Token.CODE_START, "MARK_CS"),
        ]:
            if tok < V:
                embed_weight[tok, D(dim_name)] = 1.0
                embed_weight[tok, is_mark] = 1.0

        # STACK0 marker WITHOUT IS_MARK (so threshold heads see BP as nearest)
        if Token.STACK0 < V:
            embed_weight[Token.STACK0, D("MARK_STACK0")] = 1.0

        # Step-end / data-end / halt
        mark_se = D("MARK_SE")
        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            if tok < V:
                embed_weight[tok, mark_se] = 1.0
                embed_weight[tok, is_mark] = 1.0

        if Token.STEP_END < V:
            embed_weight[Token.STEP_END, D("MARK_SE_ONLY")] = 1.0

        if Token.TOOL_CALL < V:
            embed_weight[Token.TOOL_CALL, mark_se] = 1.0
            embed_weight[Token.TOOL_CALL, is_mark] = 1.0
            embed_weight[Token.TOOL_CALL, D("MARK_SE_ONLY")] = 1.0
            embed_weight[Token.TOOL_CALL, const] = 1.0

        # Thinking markers (try/except in case dims not declared in compiler spec)
        try:
            temp = D("TEMP")
            if Token.THINKING_START < V:
                embed_weight[Token.THINKING_START, is_mark] = 1.0
                embed_weight[Token.THINKING_START, const] = 1.0
                embed_weight[Token.THINKING_START, temp + 1] = 1.0
            if Token.THINKING_END < V:
                embed_weight[Token.THINKING_END, is_mark] = 1.0
                embed_weight[Token.THINKING_END, const] = 1.0
                embed_weight[Token.THINKING_END, temp + 2] = 1.0
        except AttributeError:
            pass

        if Token.IO_STATE_EMIT_BYTE < V:
            embed_weight[Token.IO_STATE_EMIT_BYTE, is_mark] = 1.0
            embed_weight[Token.IO_STATE_EMIT_BYTE, const] = 1.0
        if Token.IO_STATE_EMIT_THINKING < V:
            embed_weight[Token.IO_STATE_EMIT_THINKING, is_mark] = 1.0
            embed_weight[Token.IO_STATE_EMIT_THINKING, const] = 1.0

        # Byte tokens 0-255: IS_BYTE + nibble decoding
        is_byte = D("IS_BYTE")
        embed_lo = D("EMBED_LO")
        embed_hi = D("EMBED_HI")
        clean_lo = D("CLEAN_EMBED_LO")
        clean_hi = D("CLEAN_EMBED_HI")
        for b in range(256):
            embed_weight[b, is_byte] = 1.0
            embed_weight[b, embed_lo + (b & 0xF)] = 1.0
            embed_weight[b, embed_hi + ((b >> 4) & 0xF)] = 1.0
            embed_weight[b, clean_lo + (b & 0xF)] = 1.0
            embed_weight[b, clean_hi + ((b >> 4) & 0xF)] = 1.0


def setup_head_weights(head, dim_positions: Dict[str, int] = None) -> None:
    """Bake the output-projection head weights using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path can call it with auto-allocated dim positions instead of
    _SetDim constants. When `dim_positions` is None, falls back to _SetDim
    (backward-compat with hand-set path).

    Args:
        head: The model.head nn.Linear(d_model, vocab_size) module.
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import Token, _SetDim

    def D(name):
        if dim_positions is not None and name in dim_positions:
            return dim_positions[name]
        return getattr(_SetDim, name)

    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

        next_flags = [
            D("NEXT_PC"), D("NEXT_AX"), D("NEXT_SP"), D("NEXT_BP"),
            D("NEXT_STACK0"), D("NEXT_MEM"), D("NEXT_SE"), D("NEXT_HALT"),
        ]
        # Optional flags (only present when conversational I/O is enabled)
        for opt in ("NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END"):
            try:
                next_flags.append(D(opt))
            except AttributeError:
                pass

        OUTPUT_LO = D("OUTPUT_LO")
        OUTPUT_HI = D("OUTPUT_HI")
        for b in range(256):
            lo, hi = b & 0xF, (b >> 4) & 0xF
            head.weight[b, OUTPUT_LO + lo] = 5.0
            head.weight[b, OUTPUT_HI + hi] = 5.0
            head.bias[b] = -5.0
            for flag in next_flags:
                head.weight[b, flag] += -80.0
        head.bias[0] = -4.0

        vocab_size = head.weight.shape[0]
        for tok, flag_name in [
            (Token.REG_PC, "NEXT_PC"),
            (Token.REG_AX, "NEXT_AX"),
            (Token.REG_SP, "NEXT_SP"),
            (Token.REG_BP, "NEXT_BP"),
            (Token.STACK0, "NEXT_STACK0"),
            (Token.MEM, "NEXT_MEM"),
            (Token.STEP_END, "NEXT_SE"),
            (Token.HALT, "NEXT_HALT"),
            (Token.TOOL_CALL, "NEXT_TOOL_CALL"),
            (Token.THINKING_START, "NEXT_THINKING_START"),
            (Token.THINKING_END, "NEXT_THINKING_END"),
        ]:
            if tok >= vocab_size:
                continue
            try:
                head.weight[tok, D(flag_name)] = 20.0
                head.bias[tok] = -10.0
            except AttributeError:
                pass

        for tok in [
            Token.CODE_START, Token.CODE_END,
            Token.DATA_START, Token.DATA_END,
            Token.SEP, Token.USER_INPUT_START, Token.USER_INPUT_END,
        ]:
            if tok < vocab_size:
                head.bias[tok] = -50.0

        if Token.IO_STATE_EMIT_BYTE < vocab_size:
            head.bias[Token.IO_STATE_EMIT_BYTE] = -20.0
        if Token.IO_STATE_EMIT_THINKING < vocab_size:
            head.bias[Token.IO_STATE_EMIT_THINKING] = -20.0


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
# - OP_LEV/BZ/BNZ + ACTIVE_OPCODE_PRTF/READ: injected by
#   `_inject_active_opcode` based on the current opcode hint.
# - MARK_THINKING_START/END: written by `_inject_thinking_markers` on
#   THINKING_START/END tokens.
# - MEM_STORE / MEM_EXEC / ADDR_KEY: written by `_inject_mem_store`,
#   `_inject_mem_exec[_autoregressive]` for memory ops.
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
    # Active-opcode injection slots (_inject_active_opcode)
    "OP_LEV", "OP_BZ", "OP_BNZ",
    "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
    # Memory injection slots (_inject_mem_store, _inject_mem_exec)
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
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
        "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
        "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
        "IO_IS_PUTCHAR", "IO_OUTPUT_READY",
        "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
        "OP_SI", "OP_SC", "OP_PSH",
        "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_NE", "OP_LT",
        "OP_GT", "OP_LE", "OP_GE", "OP_SHL", "OP_SHR",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_EXIT", "OP_NOP", "OP_PUTCHAR", "OP_GETCHAR",
        "MEM_STORE", "MEM_ADDR_SRC",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP", "MEM_EXEC",
        "OPCODE_BASE",
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
    ]
    # 7-dim threshold head outputs (one per marker type)
    seven_dim = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
                 "L1H0", "L1H1", "L1H2", "L1H4", "L2H0"]
    # 16-dim nibble groups
    sixteen_dim = ["EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
                   "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
                   "FETCH_LO", "FETCH_HI", "MUL_ACCUM", "DIV_STAGING",
                   "AX_FULL_LO", "AX_FULL_HI",
                   "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                   "ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                   "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI",
                   "FORMAT_PTR_LO", "FORMAT_PTR_HI",
                   "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"]
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

    def _declare(name, size):
        if not hasattr(_SetDim, name):
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
