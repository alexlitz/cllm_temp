"""V2/G7 control-flow detector attention heads.

This module implements the "form 2" architectural alternative described in
``docs/CONTROL_FLOW_DETECTOR_HEADS.md`` -- a family of declarative
attention heads whose Q matches the **current** instruction-step row,
whose K matches the **prior** instruction-step's opcode-marker row in
the KV cache, and whose V/O projects the saved control-flow payload
(return PC bytes / saved BP / saved SP / branch target) into a fresh
set of residual slots that the downstream L8-L17 ops can read instead
of the legacy ``PREV_STEP`` dim aliases.

The first concrete head is ``make_lev_detector_head_op``: it detects
that the prior instruction-step's opcode was LEV and pulls the saved
return-PC bytes (from the prior step's ``TEMP+0..31`` staging) and the
saved BP nibbles (from the prior step's ``ADDR_B0_LO/HI`` post-gather
residue) into the current step's
``PC_VIA_LEV_DETECTOR_LO/HI`` / ``BP_VIA_LEV_DETECTOR`` /
``SP_VIA_LEV_DETECTOR`` bands. Structural template:
``layer8_head6_ax_carry_refresh`` at ``l8_ops.py:1872-2001`` -- the
existing attention-back-to-prev-step pattern.

Default ``enable=False`` keeps the op registered (so the dep graph sees
its ``produces`` annotations and the residual layout stays stable) but
the bake body is a no-op -- the production compile is byte-identical to
the pre-spike baseline. Flipping ``enable=True`` lowers the head into a
real attention slot allocated via ``AttentionHeadAllocator`` in
``dynamic_first_fit`` mode; the per-VM-block ``num_heads`` must
accommodate the extra head slot (the post-Phase-8.O.2 dynamic head
count path) for the ``enable=True`` bake to fit.
"""

from __future__ import annotations

from ...attention_head_allocator import AttentionHeadAllocator
from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# Head-local Q/K scaling. Matches the ``AX_CARRY_L = 50.0`` constant used
# by ``layer8_head6_ax_carry_refresh`` so the softmax-1 anchor / ALiBi
# decay behave on the same scale. Kept module-local rather than imported
# from ``l8_ops`` because the detector head is a self-contained spec and
# changing the scale should not silently follow upstream tweaks to the
# AX-carry refresh head.
_LEV_DETECTOR_Q_SCALE = 50.0


# Anti-current-opcode gate slots. Mirrors the 27-element anti_ops list at
# ``l8_ops.py:1934-1942``: every OP_* flag except OP_LEV gets an anti-
# slot so the K side of the head matches **only** the most-recent OP_LEV
# row in history, not the current step's opcode marker (which would
# trivially match at distance 0 and dominate the softmax).
#
# OP_LEV is intentionally **omitted** from the anti list -- the detector
# WANTS to fire on OP_LEV marker rows. The corresponding anti-current
# gate that prevents the **current** step's OP_LEV marker from being
# matched (when the current step is itself a LEV) is handled by the Q
# gate slot on ``HAS_SE`` (the prev-step's STEP_END marker is required,
# so the current-step's own marker row gets a -inf score).
_ANTI_OPCODE_DIMS = (
    "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR",
    "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_AND", "OP_OR", "OP_XOR",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR",
    "OP_LI", "OP_LC", "OP_LEA",
)


def _lev_detector_head_spec(
    BD,
    *,
    head_idx: int,
    head_dim: int,
) -> DeclarativeAttentionHeadSpec:
    """Return the :class:`DeclarativeAttentionHeadSpec` for the LEV detector.

    Q fires at the current step's PC/BP/SP marker rows on non-first steps
    (``HAS_SE = 1``). K matches any past row carrying ``OP_LEV + MARK_PC``
    (the prior step's LEV opcode marker), with anti-current-opcode gates
    on the remaining OP_* flags so the softmax cannot land on the current
    step's own marker row.

    V (slots ``base + 1 .. base + 64``):
      * slots  1..16: TEMP+0..15      -> saved return-PC LO nibbles
      * slots 17..32: TEMP+16..31     -> saved return-PC HI nibbles
      * slots 33..48: ADDR_B0_LO+0..15 -> saved BP LO nibbles
      * slots 49..64: ADDR_B0_HI+0..15 -> saved SP LO nibbles

    O routes each V slot into the corresponding ``_VIA_LEV_DETECTOR_*``
    output dim at the current AX/PC/BP/SP marker query position.

    ``head_dim`` is asserted to be at least 65 so all V slots fit within
    the head's row block. The legacy ``HD = dim // num_heads = 64`` budget
    is enough only if we restructure (BP/SP share the same 16 V slots --
    we keep them split as the docs specify because the L16 LEV writes
    them separately too). When the production head_dim is 64 the spike
    is forced into the dynamic-head-dim path (V1/V2 vision).
    """
    if head_dim < 65:
        raise ValueError(
            "_lev_detector_head_spec requires head_dim>=65 to fit 1 Q "
            "slot + 64 V slots; got head_dim={}. Bump num_heads down or "
            "head_dim up (V1/V2 dynamic head_dim path).".format(head_dim)
        )

    q_writes = [
        # Q[base]: fire at current-step PC marker on non-first steps.
        # HAS_SE = 1 only after the first STEP_END (matches the
        # ax_carry_refresh head's first-step gate).
        AP(0, BD.MARK_PC, _LEV_DETECTOR_Q_SCALE),
        AP(0, BD.HAS_SE, _LEV_DETECTOR_Q_SCALE),
        AP(0, BD.CONST, -_LEV_DETECTOR_Q_SCALE * 1.5),
    ]

    # K[base]: match past row with OP_LEV marker. ALiBi recency
    # bias (inherited from the layer's default slope) picks the **most
    # recent** OP_LEV row when the program has executed multiple LEVs.
    k_writes = [
        AP(0, BD.OP_LEV, _LEV_DETECTOR_Q_SCALE),
        AP(0, BD.MARK_PC, _LEV_DETECTOR_Q_SCALE),
    ]

    # V slots: project saved PC / BP / SP nibbles from prev-step row.
    v_writes = []
    for k in range(16):
        # slots 1..16  -> PC LO from TEMP+0..15 (saved return-PC LO)
        v_writes.append(AP(1 + k, BD.TEMP + k, 1.0))
        # slots 17..32 -> PC HI from TEMP+16..31
        v_writes.append(AP(17 + k, BD.TEMP + 16 + k, 1.0))
        # slots 33..48 -> BP LO from ADDR_B0_LO+0..15
        v_writes.append(AP(33 + k, BD.ADDR_B0_LO + k, 1.0))
        # slots 49..64 -> SP LO from ADDR_B0_HI+0..15 (saved BP HI is
        # the BP+16 add input for L16's SP=BP+16 path; the detector
        # exposes the raw nibble so a downstream consumer can apply
        # the +16 increment inline)
        v_writes.append(AP(49 + k, BD.ADDR_B0_HI + k, 1.0))

    # O writes: route each V slot into the corresponding _VIA_LEV_DETECTOR_*
    # output dim at the current query position. The attention output
    # therefore writes the prev-step LEV payload into the current row's
    # detector residual band.
    o_writes = []
    for k in range(16):
        o_writes.append(AO(BD.PC_VIA_LEV_DETECTOR_LO + k, 1 + k, 1.0))
        o_writes.append(AO(BD.PC_VIA_LEV_DETECTOR_HI + k, 17 + k, 1.0))
        o_writes.append(AO(BD.BP_VIA_LEV_DETECTOR + k, 33 + k, 1.0))
        o_writes.append(AO(BD.SP_VIA_LEV_DETECTOR + k, 49 + k, 1.0))

    # Anti-current-opcode gates (slot indices base+34..base+62) mirror
    # the ``ANTI_OP_SLOT_START = 34`` family in
    # ``layer8_head6_ax_carry_refresh``; each anti slot has a Q write
    # of -L and a K write of +L on the SAME OP_* dim so a current-row
    # opcode marker generates a deeply negative pre-softmax score,
    # preventing it from being matched. Index walking matches the
    # ax_carry_refresh pattern so the head_dim accounting stays stable.
    anti_slot_start = 1 + 64  # past Q slot + 64 V slots
    for j, name in enumerate(_ANTI_OPCODE_DIMS):
        slot = anti_slot_start + j
        if slot >= head_dim:
            # Skip remaining anti slots if head_dim is tight; the
            # detector's primary gates (HAS_SE, MARK_PC) already exclude
            # the current row in most cases.
            break
        op_dim = getattr(BD, name)
        q_writes.append(AP(slot, op_dim, -_LEV_DETECTOR_Q_SCALE))
        k_writes.append(AP(slot, op_dim, _LEV_DETECTOR_Q_SCALE))

    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q_writes),
        k=tuple(k_writes),
        v=tuple(v_writes),
        o=tuple(o_writes),
    )


def make_lev_detector_head_op(enable: bool = False) -> Operation:
    """L8 attn LEV detector head -- "form 2" V2/G7 control-flow head.

    Detects that the **prior** instruction-step's opcode was LEV and
    materialises the saved return-PC / BP / SP nibbles into the current
    step's ``PC_VIA_LEV_DETECTOR_LO/HI``, ``BP_VIA_LEV_DETECTOR`` and
    ``SP_VIA_LEV_DETECTOR`` residual bands. Downstream readers (L9 alu,
    L8 sp_gather_bake) can prefer the detector output on LEV-following
    steps and fall back to the L16 ``layer16_lev_routing`` OUTPUT_LO/HI
    writes everywhere else.

    Status: ``enable=False`` (default) keeps the op registered for dep-
    graph stability but the bake body is a no-op. Flipping to
    ``enable=True`` lowers the head via a dynamic-first-fit allocator at
    L8 -- the per-VM-block ``num_heads`` must already accommodate the
    extra head (typically via ``compile_full_vm_dynamic(n_heads=9, ...)``
    or the unbounded head-count path).

    The ``produces`` annotation enumerates the four ``_VIA_LEV_DETECTOR_*``
    dims so the staleness analyser sees a producer for any future
    consumer that declares ``consumes_fresh = {"PC_VIA_LEV_DETECTOR_LO":
    "PC_byte0", ...}``.

    See ``docs/CONTROL_FLOW_DETECTOR_HEADS.md`` for the design.
    """
    def _bake(block, dim_positions, S):
        if not enable:
            return
        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.O.2 dynamic head count: allocate the LEV detector head
        # via ``dynamic_first_fit`` (the spike's V2/G7 form 2 design
        # requires a fresh head slot, NOT bumping any of the legacy
        # ``DEFAULT_LAYER_MAX_HEADS=8`` constants). The unbounded
        # constructor grows the per-layer pool on demand so the head
        # lands at the lowest free index, typically 8 in production.
        allocator = AttentionHeadAllocator(
            layer_max_heads=None,
            strategy="dynamic_first_fit",
        )
        # Replay the existing L8 head claims so the detector first-fits
        # the lowest unclaimed slot. The legacy layout pins 0..7; the
        # detector picks 8 in the unbounded pool.
        for legacy_name, legacy_idx in (
            ("layer8_sp_gather_bake.head_0", 0),
            ("layer8_sp_gather_bake.head_1", 1),
            ("layer8_sp_gather_bake.head_2", 2),
            ("layer8_multibyte_fetch_bake.head_3", 3),
            ("layer8_op_imm_relay.head_4", 4),
            ("layer8_mem_to_alu.head_5", 5),
            ("layer8_sp_gather_bake.head_6_mark_sp_mirror", 6),
            ("layer8_sp_gather_bake.head_7_mark_sp_mirror", 7),
        ):
            allocator.alloc(legacy_name, layer_idx=8, pin=legacy_idx)
        head_idx = allocator.alloc(
            "lev_detector_head", layer_idx=8,
        )
        attn._lev_detector_head_allocator = allocator

        spec = _lev_detector_head_spec(BD, head_idx=head_idx, head_dim=HD)
        Primitives.generate_attention_head(attn, spec, HD)

    return Operation(
        name="lev_detector_head",
        # phase=8.06 places it AFTER ``layer8_head6_ax_carry_refresh``
        # (phase=8.05) and BEFORE ``layer8_op_imm_relay`` (phase=8.4)
        # so the detector output band is materialised early enough for
        # the L8 multibyte / OP_IMM relay heads -- and any future L9
        # alu consumer reading ``PC_VIA_LEV_DETECTOR_LO`` -- to see it
        # in the residual.
        # Cross-step reads via the attention back-edge: TEMP_PREV_STEP /
        # ADDR_B0_LO_PREV_STEP / ADDR_B0_HI_PREV_STEP carry the
        # prev-step row's saved-PC / saved-BP staging through the KV
        # cache. OP_LEV is the K-side marker the detector matches on
        # the prev-step row; MARK_PC anchors the K position. MARK_PC /
        # HAS_SE / CONST drive the Q-side current-step gate.
        reads={
            "MARK_PC", "HAS_SE", "CONST",
            "TEMP.*.-1",
            "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1",
            "OP_LEV",
            "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR",
            "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_AND", "OP_OR", "OP_XOR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_SHL", "OP_SHR",
            "OP_LI", "OP_LC", "OP_LEA",
        },
        writes={
            "PC_VIA_LEV_DETECTOR_LO", "PC_VIA_LEV_DETECTOR_HI",
            "BP_VIA_LEV_DETECTOR", "SP_VIA_LEV_DETECTOR",
        },
        kind="block",
        bake_fn=_bake,
        declarative_bake_fn=_bake if not enable else None,
        # Bind to the L8 attn anchor (the same target ``layer8_head6_
        # ax_carry_refresh`` uses) so this block op resolves to whichever
        # layer the dynamic compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        produces={
            "PC_VIA_LEV_DETECTOR_LO": "PC_byte0",
            "PC_VIA_LEV_DETECTOR_HI": "PC_byte0",
            "BP_VIA_LEV_DETECTOR": "BP_byte0",
            "SP_VIA_LEV_DETECTOR": "SP_byte0",
        },
        # Phase 11.A IR exposure: at the default ``enable=False`` config the
        # bake body is a no-op (``if not enable: return``), so an empty IR
        # is byte-identical. When ``enable=True``, the head is fully
        # declarative (single :class:`DeclarativeAttentionHeadSpec` from
        # :func:`_lev_detector_head_spec`); migrating to a true factory
        # requires the ``head_idx`` allocation, which is bake-time
        # (dynamic_first_fit) -- Phase 11.A follow-up.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="docs/CONTROL_FLOW_DETECTOR_HEADS.md#2-lev-detector-head",
    )


__all__ = [
    "make_lev_detector_head_op",
    "_lev_detector_head_spec",
]
