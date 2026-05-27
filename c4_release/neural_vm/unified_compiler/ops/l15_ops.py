"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import torch.nn as nn
from collections.abc import Mapping

from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


def make_l15_psh_stack_ir() -> CompilerIR:
    """Declarative L15 PSH SP/BP byte producer rules.

    These rules cover the PSH-specific SP byte outputs and BP byte-2
    preservation formerly hand-written in ``_set_nibble_copy_ffn``. Writes are
    expressed in semantic output-delta units; neural lowering applies ``1 / S``
    scaling to recover the legacy W_down coefficients.
    """
    ir = CompilerIR()
    rules = ir.layer(0).ffn
    sp_i = 2
    bp_i = 3
    threshold = 3.5

    def psh_byte_conditions(marker_index, byte_index_name):
        return (
            ("PSH_AT_SP", 1.0),
            (f"H1+{marker_index}", 1.0),
            ("IS_BYTE", 1.0),
            (byte_index_name, 1.0),
        )

    # SP byte 0 position predicts SP byte 1 = 0xff after SP -= 8.
    rules.append(FFNRule.constant_write(
        name="psh_sp_byte1_lo_ff",
        conditions=psh_byte_conditions(sp_i, "BYTE_INDEX_0"),
        threshold=threshold,
        writes=(("OUTPUT_LO+15", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(FFNRule.constant_write(
        name="psh_sp_byte1_hi_ff",
        conditions=psh_byte_conditions(sp_i, "BYTE_INDEX_0"),
        threshold=threshold,
        writes=(("OUTPUT_HI+15", 4.0), ("OUTPUT_HI+0", -4.0)),
    ))

    # SP byte 1 and byte 2 positions predict zero for the following bytes.
    for byte_index_name, predicted_byte in (
        ("BYTE_INDEX_1", "byte2"),
        ("BYTE_INDEX_2", "byte3"),
    ):
        rules.append(FFNRule.constant_write(
            name=f"psh_sp_{predicted_byte}_lo_00",
            conditions=psh_byte_conditions(sp_i, byte_index_name),
            threshold=threshold,
            writes=(("OUTPUT_LO+0", 2.0),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"psh_sp_{predicted_byte}_hi_00",
            conditions=psh_byte_conditions(sp_i, byte_index_name),
            threshold=threshold,
            writes=(("OUTPUT_HI+0", 2.0),),
        ))

    # PSH leaves BP unchanged; preserve STACK_INIT byte 2 = 0x01.
    rules.append(FFNRule.constant_write(
        name="psh_bp_byte2_lo_01",
        conditions=psh_byte_conditions(bp_i, "BYTE_INDEX_1"),
        threshold=threshold,
        writes=(("OUTPUT_LO+1", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(FFNRule.constant_write(
        name="psh_bp_byte2_hi_00",
        conditions=psh_byte_conditions(bp_i, "BYTE_INDEX_1"),
        threshold=threshold,
        writes=(("OUTPUT_HI+0", 4.0),),
    ))
    return ir


def make_l15_nibble_copy_ir() -> CompilerIR:
    """Declarative L15 nibble-copy FFN program.

    Covers the full legacy ``_set_nibble_copy_ffn`` unit range:
    32 generic copy units, 8 PSH stack-byte units, and 2 first-step LEA units.
    Writes are semantic deltas; lowering applies ``1 / S``.
    """

    ir = CompilerIR()
    rules = ir.layer(0).ffn
    pc_i = 0
    ax_i = 1
    sp_i = 2
    bp_i = 3

    copy_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{pc_i}", -1.0),
        (f"H1+{ax_i}", -1.0),
        (f"H1+{sp_i}", -1.0),
        (f"H1+{bp_i}", -1.0),
        (f"H4+{bp_i}", -1.0),
        ("MEM_STORE", -1.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"nibble_copy_lo_{k}",
            conditions=copy_conditions,
            threshold=0.5,
            gate=f"EMBED_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"nibble_copy_hi_{k}",
            conditions=copy_conditions,
            threshold=0.5,
            gate=f"EMBED_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", 2.0),),
        ))

    rules.rules.extend(make_l15_psh_stack_ir().layer(0).ffn.rules)

    lea_conditions = (
        ("CMP+7", 1.0),
        (f"H1+{ax_i}", 1.0),
        ("IS_BYTE", 1.0),
        ("BYTE_INDEX_1", 1.0),
        ("HAS_SE", -1.0),
    )
    rules.append(FFNRule.constant_write(
        name="lea_first_step_ax_byte2_lo_01",
        conditions=lea_conditions,
        threshold=4.5,
        writes=(("OUTPUT_LO+1", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(FFNRule.constant_write(
        name="lea_first_step_ax_byte2_hi_00",
        conditions=lea_conditions,
        threshold=4.5,
        writes=(("OUTPUT_HI+0", 2.0),),
    ))
    return ir


def lower_l15_psh_stack_ir(ffn, dim_positions, *, start_unit, S):
    """Lower L15 PSH stack IR into ``ffn`` and return the next free unit."""
    return make_l15_psh_stack_ir().lower_ffn(
        ffn,
        dim_positions,
        start_unit=start_unit,
        S=S,
        write_scale=1.0 / S,
    )


def lower_l15_nibble_copy_ir(ffn, dim_positions, *, start_unit=0, S=100.0):
    """Lower the full L15 nibble-copy IR and return the next free unit."""
    ir = make_l15_nibble_copy_ir()
    rules = ir.layer(0).ffn.rules
    if not isinstance(dim_positions, Mapping):
        dim_positions = Primitives.dim_positions_from_bd(
            dim_positions,
            Primitives.ffn_rule_dim_names(rules),
        )
    return ir.lower_ffn(
        ffn,
        dim_positions,
        start_unit=start_unit,
        S=S,
        write_scale=1.0 / S,
    )


def make_nibble_copy_ffn_op() -> Operation:
    """Topology anchor for L15 nibble-copy FFN.

    The actual weight bake is owned by ``layer15_nibble_copy`` below, pinned
    to ``model.blocks[15].ffn``.
    """
    def bake(ffn, dim_positions, S):
        return None

    return Operation(
        name="nibble_copy_ffn",
        phase=15,
        reads={"IS_BYTE", "H1", "H4", "MEM_STORE",
               "EMBED_LO", "EMBED_HI", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MARK_STACK0", "HAS_SE", "CMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer15_memory_lookup_op() -> Operation:
    """L15 attention: memory-lookup heads for LI/LC."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer15_memory_lookup
        HD = attn.W_q.shape[0] // attn.num_heads
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer15_memory_lookup(attn, S, proxy, HD)
        _suppress_l15_lookup_during_current_store_generation(attn, proxy, HD)
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            # Memory reads are last-write-wins. Strict neural traces can leave
            # residual address features on older MEM value bytes; once source
            # gating has limited candidates to stored value bytes, use ALiBi
            # only as a same-address tie-breaker. A large slope makes a newer
            # write to a different local slot beat the correct older address.
            attn.alibi_slopes[:4] = 0.05

    # Dim-ownership claims: L15 attn heads 0-3 (memory lookup).
    # Each head writes V slots 32..47 + 48..62 reading CLEAN_EMBED_LO/HI:
    #   W_v[h*HD + 32 + k, CLEAN_EMBED_LO + k]   for k=0..15
    #   W_v[h*HD + 48 + k, CLEAN_EMBED_HI + k]   for k=0..14
    # Slot 63 is repurposed by the non-pop STACK0 marker blocker below.
    # When num_heads >= 12 (LEV-aware build), heads 4-11 are also active;
    # we restrict claims to heads 0-3 which are the universal load heads
    # to keep the claim set stable across head-count configurations.
    _claims = set()
    for h in range(4):
        for k in range(16):
            _claims.add((15, "attn_W_v", f"{h}_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
            if k < 15:
                _claims.add((15, "attn_W_v", f"{h}_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer15_memory_lookup",
        phase=15,
        reads={"MARK_AX", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_KEY", "MARK_MEM", "MEM_STORE",
               "MEM_ADDR_SRC",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "MARK_STACK0", "IS_BYTE",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1", "H2", "H3", "L2H0", "TEMP",
               "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3", "CMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        layer_idx=15,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer15_store_stack0_sp_byte0_addr_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Copy current post-pop SP byte0 into ADDR_B0 at store STACK0 markers."""

    q = (
        AP(0, BD.MARK_STACK0, 300.0),
        AP(0, BD.HAS_SE, 300.0),
        AP(33, BD.MEM_STORE, 10000.0),
        AP(33, BD.CONST, -50000.0),
    )
    k = (
        AP(0, BD.MARK_SP, 100.0),
        AP(33, BD.CONST, 1.0),
    )
    v = [AP(0, BD.CONST, 1.0)]
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.OUTPUT_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.OUTPUT_HI + idx, 1.0))
        o.append(AO(BD.ADDR_B0_LO + idx, 0, -2.0))
        o.append(AO(BD.ADDR_B0_HI + idx, 0, -2.0))
        o.append(AO(BD.ADDR_B0_LO + idx, 1 + idx, 3.0))
        o.append(AO(BD.ADDR_B0_HI + idx, 17 + idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=12,
        q=q,
        k=k,
        v=tuple(v),
        o=tuple(o),
    )


def _layer15_store_stack0_sp_byte0_addr_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer15_store_stack0_sp_byte0_addr_spec(proxy)
    )
    return ir


def make_layer15_store_stack0_sp_byte0_addr_op() -> Operation:
    """L15 attention: expose post-pop SP byte0 for store-top discrimination."""

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer15_store_stack0_sp_byte0_addr_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[12] = 1.0

    return Operation(
        name="layer15_store_stack0_sp_byte0_addr",
        phase=15.2,
        reads={
            "MARK_STACK0", "MARK_SP", "HAS_SE", "MEM_STORE",
            "OUTPUT_LO", "OUTPUT_HI", "CONST",
        },
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        layer_idx=15,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="declarative",
        compiler_ir_factory=_layer15_store_stack0_sp_byte0_addr_ir,
        migrated=True,
        alibi_slopes={12: 1.0},
        smoke_tests={
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#stack-store",
    )


def _layer15_si_mem_addr0_from_stack0_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Override SI/SC MEM addr byte 0 from pre-store STACK0 byte 0.

    L14's generic MEM-address head reads ``CLEAN_EMBED + OUTPUT``. That is
    correct for PSH, where the SP marker carries the freshly computed address
    in OUTPUT, but for SI/SC the source is a STACK0 byte token whose OUTPUT is
    the following stack byte. This head is SI/SC-only and reads only
    CLEAN_EMBED from the pre-store STACK0 byte 0.
    """

    q = (
        AP(0, BD.MARK_MEM, 100.0),
        AP(0, BD.MEM_STORE, 20.0),
        AP(0, BD.MEM_ADDR_SRC, 50.0),
        AP(0, BD.CONST, -140.0),
        AP(33, BD.MARK_MEM, 40000.0),
        AP(33, BD.MEM_STORE, 5000.0),
        AP(33, BD.MEM_ADDR_SRC, 10000.0),
        AP(33, BD.CONST, -55000.0),
    )
    k = (
        AP(0, BD.STACK0_BYTE0, 100.0),
        AP(0, BD.MEM_STORE, -400.0),
        AP(33, BD.CONST, 5.0),
    )
    v = [AP(0, BD.CONST, 1.0)]
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.CLEAN_EMBED_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.CLEAN_EMBED_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, 0, -10.0))
        o.append(AO(BD.OUTPUT_HI + idx, 0, -10.0))
        o.append(AO(BD.OUTPUT_LO + idx, 1 + idx, 20.0))
        o.append(AO(BD.OUTPUT_HI + idx, 17 + idx, 20.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=13,
        q=q,
        k=k,
        v=tuple(v),
        o=tuple(o),
    )


def _layer15_si_mem_addr0_from_stack0_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer15_si_mem_addr0_from_stack0_spec(proxy)
    )
    return ir


def make_layer15_si_mem_addr0_from_stack0_op() -> Operation:
    """L15 attention: clean SI/SC MEM addr0 from STACK0 byte0."""

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer15_si_mem_addr0_from_stack0_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[13] = 1.0

    return Operation(
        name="layer15_si_mem_addr0_from_stack0",
        phase=15.25,
        reads={
            "MARK_MEM", "MEM_STORE", "MEM_ADDR_SRC", "STACK0_BYTE0",
            "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        layer_idx=15,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="declarative",
        compiler_ir_factory=_layer15_si_mem_addr0_from_stack0_ir,
        migrated=True,
        alibi_slopes={13: 1.0},
        smoke_tests={
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _suppress_l15_lookup_during_current_store_generation(attn, BD, HD) -> None:
    """Keep L15 memory lookup from overwriting L14's current store tokens.

    L15 is a load-side attention op: it reads historical MEM stores for LI/LC
    and pop-group STACK0 loads. During PSH/SI/SC/JSR/ENT, L14 is still
    generating the current MEM section. Those in-flight MEM byte positions have
    MEM_STORE set and can also carry byte-index/ADDR_KEY features, making L15
    look target-like and add a zero-valued load result over L14's freshly
    emitted store byte. Suppress only query positions with MEM_STORE set; the
    historical store tokens remain available as K-side memory entries.
    """

    pc_i = 0
    ax_i = 1
    mem_i = 4
    sp_i = 2
    bp_i = 3
    for head in range(4):
        base = head * HD
        # Local stack slots commonly differ only in byte 0 (BP-8, BP-16,
        # BP-24, ...). The generic 24-bit binary address rows are
        # intentionally modest, but after shallow ALiBi recency the byte-0
        # mismatch must dominate newer writes to adjacent locals. Strengthen
        # both byte-0 nibbles: three-local frames can share the same low
        # nibble (e.g. BP-8 and BP-24) and only differ in the high nibble.
        local_slot_scale = 100.0
        for nibble_offset, nibble_base in ((0, BD.ADDR_B0_LO), (4, BD.ADDR_B0_HI)):
            for bit in range(4):
                row = base + 4 + nibble_offset + bit
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    attn.W_q.data[row, nibble_base + k] = local_slot_scale * bit_val
                    attn.W_k.data[row, nibble_base + k] = local_slot_scale * bit_val
        # The bit rows above can be cancelled by negative adjacent-nibble
        # residue from address arithmetic. Add direct one-hot rows so an exact
        # low-byte slot match remains authoritative.
        local_slot_onehot_scale = 100.0
        for k in range(16):
            row = base + 43 + k
            attn.W_q.data[row, BD.CONST] = -local_slot_onehot_scale
            attn.W_q.data[row, BD.ADDR_B0_LO + k] = local_slot_onehot_scale
            attn.W_q.data[row, BD.OP_LI_RELAY] = local_slot_onehot_scale
            if head == 0:
                attn.W_q.data[row, BD.OP_LC_RELAY] = local_slot_onehot_scale
            attn.W_k.data[row, BD.ADDR_B0_LO + k] = local_slot_onehot_scale

        if head == 0:
            # L6/L8 relays can deliver the pop-group flag at ~4.0 by the time
            # L15 runs.  A small non-target bias is not enough once local
            # address residue reaches the STACK0 marker, so make row 0 a
            # strong default blocker and explicitly restore real LI/LC and
            # pop-stack queries.
            lookup_bias = 200000.0
            attn.W_q.data[base + 0, BD.CONST] = -lookup_bias
            attn.W_q.data[base + 0, BD.OP_LI_RELAY] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LC_RELAY] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LI] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LC] = lookup_bias
            attn.W_q.data[base + 0, BD.CMP + 3] = lookup_bias / 4.0
            attn.W_q.data[base + 1, BD.CMP + 3] = 12.5
            # JSR/ENT synthesize STACK0 values through the function-call path,
            # LEA computes AX from BP+imm through the arithmetic path, and IMM
            # preserves STACK0 while loading AX from the immediate. None of
            # these are memory loads; residual address/store metadata can
            # otherwise make the load-only L15 head read an old zero/SP value
            # over the value already produced upstream.
            non_load_suppression = -1000000.0
            attn.W_q.data[base + 0, BD.OP_JSR] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_ENT] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_LEA] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_IMM] = non_load_suppression
            # Preserve lookups at the exact 0xffe8 stack-top slot. This keeps
            # ordinary IMM steps in function-call setup from going through the
            # non-load zero sink while still excluding the adjacent 0xfff8
            # return-address slot.
            attn.W_q.data[base + 0, BD.MARK_STACK0] = 75000.0
            attn.W_q.data[base + 0, BD.HAS_SE] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_LO + 8] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_HI + 14] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_HI + 15] = -100000.0
            attn.W_q.data[base + 0, BD.IS_BYTE] = -2000.0
            attn.W_q.data[base + 1, BD.IS_BYTE] = -50.0
            attn.W_q.data[base + 1, BD.MARK_STACK0] = 50.0
            attn.W_q.data[base + 1, BD.HAS_SE] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_LO + 8] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_HI + 14] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_HI + 15] = -150.0
            attn.W_q.data[base + 28, BD.IS_BYTE] = -500.0
            attn.W_q.data[base + 28, BD.CONST] = -20000.0
            attn.W_q.data[base + 28, BD.MARK_AX] = 20000.0
            attn.W_q.data[base + 28, BD.MARK_STACK0] = 20000.0
            # Store/pop STACK0 markers are usually load queries too: after
            # SI/SC pops the address, the visible stack top is memory[post-pop
            # SP].  The exception is a top-store, where the popped address is
            # exactly the post-pop SP.  In that case the current store value is
            # authoritative, but the current MEM row does not exist yet at the
            # STACK0 marker, so an unblocked L15 lookup would read the old
            # historical value.  Block only those equality signatures and keep
            # non-top stores available for historical memory lookup.
            for row, low, high in (
                (42, 0, 14),   # e0
            ):
                attn.W_q.data[base + row, BD.CONST] = -60000.0
                attn.W_q.data[base + row, BD.MARK_STACK0] = 10000.0
                attn.W_q.data[base + row, BD.HAS_SE] = 10000.0
                attn.W_q.data[base + row, BD.MEM_STORE] = 10000.0
                attn.W_q.data[base + row, BD.EMBED_LO + low] = 10000.0
                attn.W_q.data[base + row, BD.EMBED_HI + high] = 10000.0
                attn.W_q.data[base + row, BD.ADDR_B0_LO + low] = 10000.0
                attn.W_q.data[base + row, BD.ADDR_B0_HI + high] = 10000.0
                attn.W_q.data[base + row, BD.OP_LI_RELAY] = 50000.0
                attn.W_q.data[base + row, BD.OP_LC_RELAY] = 50000.0
                # A partial miss on this signature is negative; keep that
                # negative query from becoming positive stale-key evidence.
                attn.W_k.data[base + row, BD.CONST] = 20.0

            # Rows 59-61 are value lanes for OUTPUT_HI[11:13], not address
            # discriminators.  If they retain any Q/K miss terms, the
            # negative miss can multiply stale negative key residue into a
            # large positive score and undo the top-store blocker above.
            for row in (59, 60, 61):
                if row < HD:
                    attn.W_q.data[base + row, :] = 0.0
                    attn.W_k.data[base + row, :] = 0.0

            # Same top-store equality as the e0 row above, but for the common
            # one-local slot 0xffe8. At the SI/SC STACK0 marker the address is
            # present in ADDR_B0, while EMBED still carries marker/value
            # residue; block the historical lookup so L14's current store
            # value remains authoritative.
            top_store_e8_row = 60
            if top_store_e8_row < HD:
                row = base + top_store_e8_row
                attn.W_q.data[row, BD.CONST] = -30000.0
                attn.W_q.data[row, BD.MARK_STACK0] = 10000.0
                attn.W_q.data[row, BD.MARK_SP] = -100000.0
                attn.W_q.data[row, BD.HAS_SE] = 10000.0
                attn.W_q.data[row, BD.MEM_STORE] = 150000.0
                attn.W_q.data[row, BD.ADDR_B0_LO + 8] = 10000.0
                attn.W_q.data[row, BD.ADDR_B0_HI + 14] = 10000.0
                attn.W_q.data[row, BD.ADDR_B0_HI + 15] = -20000.0
                attn.W_k.data[row, BD.CONST] = -20.0

            # When preserving STACK0 at the one-argument call slot (0xffe8),
            # the adjacent return-address slot (0xfff8) shares byte-0 low
            # nibble 8 and can win by recency. Add an exact high-nibble source
            # discriminator for this marker lookup so the e8 store is selected.
            preserve_e8_row = 59
            preserve_e8_s = 5000.0
            attn.W_q.data[base + preserve_e8_row, BD.CONST] = -3.5 * preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.MARK_STACK0] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.HAS_SE] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.ADDR_B0_LO + 8] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.ADDR_B0_HI + 14] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.IS_BYTE] = -4.0 * preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.MEM_STORE] = -5.0 * preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.ADDR_B0_LO + 8] = preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.ADDR_B0_HI + 14] = preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.STACK0_BYTE0] = 5.0 * preserve_e8_s

            # Pop-group STACK0 marker lookups need the post-pop SP address.
            # The marker can still carry the pre-pop byte-0 address (for
            # example f0 while the value to reveal lives at f8), and the
            # same L15 attention block cannot consume the head-12 correction
            # it emits later in the block. Add a score-only row that shifts
            # low-nibble 0 queries toward historical low-nibble 8 MEM value
            # keys; the ordinary address rows still discriminate the high
            # address bytes.
            pop_low8_row = 34
            attn.W_q.data[base + pop_low8_row, BD.CONST] = -5000.0
            attn.W_q.data[base + pop_low8_row, BD.MARK_STACK0] = 2000.0
            attn.W_q.data[base + pop_low8_row, BD.HAS_SE] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.CMP + 3] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.ADDR_B0_LO + 0] = 2000.0
            attn.W_q.data[base + pop_low8_row, BD.ADDR_B0_LO + 8] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.IS_BYTE] = -10000.0
            attn.W_q.data[base + pop_low8_row, BD.MARK_SP] = -10000.0
            attn.W_q.data[base + pop_low8_row, BD.MEM_STORE] = -20000.0
            attn.W_k.data[base + pop_low8_row, BD.ADDR_B0_LO + 8] = 1000.0
        else:
            # The legacy byte-select rows for heads 1-3 target the source
            # byte behind the autoregressive query. At query byte N, logits
            # predict byte N+1, so the source must also be byte N+1.
            byte_q_flags = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]
            attn.W_q.data[base + 0, BD.MARK_STACK0] = -100000.0
            attn.W_q.data[base + 0, BD.MARK_SP] = -100000.0
            attn.W_q.data[base + 28, BD.CONST] = -20000.0
            attn.W_q.data[base + 28, byte_q_flags[head]] = 20000.0
            for dim in (BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3,
                        BD.H2 + mem_i, BD.H3 + mem_i):
                attn.W_k.data[base + 3, dim] = 0.0
            if head == 1:
                attn.W_k.data[base + 3, BD.MEM_VAL_B2] = 60.0
            elif head == 2:
                attn.W_k.data[base + 3, BD.MEM_VAL_B3] = 60.0
            elif head == 3:
                attn.W_k.data[base + 3, BD.H3 + mem_i] = 60.0
                attn.W_k.data[base + 3, BD.H2 + mem_i] = -60.0

        # Strict neural traces can leave large ADDR_KEY-like residue on code
        # and previous-step byte tokens. The lookup address rows are supposed
        # to choose among stored MEM value bytes only, so add a source-side
        # row that suppresses non-store/non-value keys while preserving the
        # same per-byte source positions selected by row 3.
        source_gate = 37
        source_gate_s = 3000.0
        attn.W_q.data[base + source_gate, BD.CONST] = 0.0
        if head == 0:
            attn.W_q.data[base + source_gate, BD.MARK_STACK0] = source_gate_s
        else:
            attn.W_q.data[base + source_gate, byte_q_flags[head]] = source_gate_s
        source_key_s = 20.0
        attn.W_k.data[base + source_gate, :] = 0.0
        source_key_s = 10.0
        attn.W_k.data[base + source_gate, BD.CONST] = -source_key_s
        attn.W_k.data[base + source_gate, BD.MEM_STORE] = 0.5 * source_key_s
        if head == 0:
            attn.W_k.data[base + source_gate, BD.L2H0 + mem_i] = source_key_s
            attn.W_k.data[base + source_gate, BD.H1 + mem_i] = -70.0
        else:
            attn.W_k.data[base + source_gate, BD.H1 + mem_i] = -70.0
            if head == 1:
                attn.W_k.data[base + source_gate, BD.MEM_VAL_B2] = source_key_s
            elif head == 2:
                attn.W_k.data[base + source_gate, BD.MEM_VAL_B3] = source_key_s
            elif head == 3:
                attn.W_k.data[base + source_gate, BD.H3 + mem_i] = source_key_s
                attn.W_k.data[base + source_gate, BD.H2 + mem_i] = -source_key_s

        # L15 lookup must source historical MEM value rows (or the prior
        # STACK0 row), not SP/BP register bytes.  Frame register bytes can
        # carry address-like residue and beat the intended MEM value source.
        for marker_i in (sp_i, bp_i):
            for dim in (
                BD.H1 + marker_i,
                BD.H2 + marker_i,
                BD.H3 + marker_i,
                BD.L2H0 + marker_i,
            ):
                attn.W_k.data[base + source_gate, dim] = -80.0

        # Load-only reinforcement: the row above also stabilizes pop/STACK0
        # lookup during store ops, so keep it conservative. LI/LC need a
        # stronger source preference to beat ADDR_KEY residue on non-MEM
        # register bytes. Keep the boost head-specific: head 0 owns the AX
        # marker byte, while heads 1-3 own AX byte positions 0-2 which predict
        # loaded bytes 1-3. If every head receives the same load-only boost at
        # every byte position, zero-valued higher-byte heads drown the correct
        # nonzero byte.
        load_source_gate = 39
        load_source_gate_s = 5000.0
        load_source_key_s = 30.0
        attn.W_q.data[base + load_source_gate, :] = 0.0
        attn.W_k.data[base + load_source_gate, :] = 0.0
        if head == 0:
            attn.W_q.data[base + load_source_gate, BD.OP_LI_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.OP_LC_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.MARK_AX] = load_source_gate_s
            # Pop-group STACK0 marker loads are real memory reads too.  Keep
            # this LI/LC reinforcement row neutral-to-positive there instead
            # of letting its negative bias force the softmax1 zero sink.
            attn.W_q.data[base + load_source_gate, BD.CMP + 3] = 2000.0
            attn.W_q.data[base + load_source_gate, BD.CONST] = -1.5 * load_source_gate_s
            attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B1] = 2.0 * load_source_key_s
            attn.W_k.data[base + load_source_gate, BD.MEM_ADDR_SRC] = 40.0
        else:
            attn.W_q.data[base + load_source_gate, BD.OP_LI_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, byte_q_flags[head]] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.CONST] = -1.5 * load_source_gate_s
            if head == 1:
                attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B2] = 2.0 * load_source_key_s
            elif head == 2:
                attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B3] = 2.0 * load_source_key_s
            elif head == 3:
                attn.W_k.data[base + load_source_gate, BD.H3 + mem_i] = 2.0 * load_source_key_s
                attn.W_k.data[base + load_source_gate, BD.H2 + mem_i] = -2.0 * load_source_key_s
            attn.W_k.data[base + load_source_gate, BD.MEM_ADDR_SRC] = 40.0

        # AX-marker byte-0 loads are especially sensitive to stale store-like
        # residue on non-value tokens. Add a marker-only source discriminator
        # that prefers SI/SC value byte 0, mildly allows PSH byte 0, and pushes
        # non-value store positions below the softmax1 anchor.
        if head == 0:
            marker_value_gate = 40
            marker_value_gate_s = 1000.0
            attn.W_q.data[base + marker_value_gate, :] = 0.0
            attn.W_k.data[base + marker_value_gate, :] = 0.0
            attn.W_q.data[base + marker_value_gate, BD.OP_LI] = marker_value_gate_s
            attn.W_q.data[base + marker_value_gate, BD.OP_LC] = marker_value_gate_s
            attn.W_k.data[base + marker_value_gate, BD.MEM_VAL_B1] = 80.0
            attn.W_k.data[base + marker_value_gate, BD.MEM_ADDR_SRC] = 40.0
            attn.W_k.data[base + marker_value_gate, BD.CONST] = -60.0

        # Loaded bytes must beat the default zero-emission path that has
        # accumulated in OUTPUT by this late layer. Keep the attention scores
        # unchanged, but make the selected memory byte authoritative in the
        # output projection.
        value_scale = 40.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + 32 + k] = value_scale
            attn.W_o.data[BD.OUTPUT_HI + k, base + 48 + k] = value_scale

        # L15 lookup is load-only. ADD/SUB rows can still carry stale ADDR_KEY
        # residue, and L10 marks arithmetic byte propagation through TEMP+8/9.
        # Force those queries to the softmax1 zero sink so L15 cannot overwrite
        # the arithmetic output that L10 just produced.
        addsub_blocker = 41
        attn.W_q.data[base + addsub_blocker, BD.TEMP + 8] = 10000.0
        attn.W_q.data[base + addsub_blocker, BD.TEMP + 9] = 10000.0
        attn.W_k.data[base + addsub_blocker, BD.CONST] = -20.0

        # SP value bytes are not L15 lookup targets.  Head 0's pop-group gate
        # can otherwise make SP byte rows attend to historical stack stores
        # when address residue is large, overwriting L3/L10's SP bytes.
        sp_byte_blocker = 62
        attn.W_q.data[base + sp_byte_blocker, BD.H1 + 2] = 100000.0
        attn.W_q.data[base + sp_byte_blocker, BD.IS_BYTE] = 0.0
        attn.W_k.data[base + sp_byte_blocker, BD.CONST] = -20.0

        # PC value bytes are produced by the control-flow path, not by memory
        # lookup. Branch targets can carry address-like residue that otherwise
        # makes L15 copy PC byte0 into the upper PC bytes.
        pc_byte_blocker = 35
        attn.W_q.data[base + pc_byte_blocker, BD.H1 + pc_i] = 100000.0
        attn.W_q.data[base + pc_byte_blocker, BD.IS_BYTE] = 0.0
        attn.W_k.data[base + pc_byte_blocker, BD.CONST] = -20.0
        if head == 0:
            # Binary-pop steps whose pre-pop SP is f8 reveal the empty stack
            # slot at 0x10000.  The STACK0 marker still carries the pre-pop
            # f8 address at L15, so do not let it read the just-popped value.
            attn.W_q.data[base + pc_byte_blocker, BD.MARK_STACK0] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.HAS_SE] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.CMP + 3] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.ADDR_B0_LO + 8] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.ADDR_B0_HI + 15] = 10000.0

        # MEM address bytes are being generated by L14/L16, not loaded from
        # historical memory.  The autoregressive query for MEM_addr{N+1} sits
        # on MEM_addrN, which carries H1[MEM] + IS_BYTE + BYTE_INDEX_N but
        # not MARK_MEM/MEM_STORE.  Block the corresponding byte heads before
        # stale MEM values can project into OUTPUT_LO/HI.
        mem_addr_byte_blocker = 36
        if head in (1, 2, 3):
            mem_addr_block_s = 100000.0
            attn.W_q.data[base + mem_addr_byte_blocker, BD.H1 + mem_i] = (
                mem_addr_block_s
            )
            attn.W_k.data[base + mem_addr_byte_blocker, BD.CONST] = -20.0

        # STACK0 marker queries are memory lookups only for pop-group ops.
        # Non-pop steps should keep the upstream STACK0 passthrough value; L15
        # can otherwise read a stale historical zero through partial top-store
        # matches. Keep this broad blocker modest: CMP can leak onto SP bytes,
        # where an overlarge negative query term creates false-positive scores.
        nonpop_stack0_marker_blocker = 63
        if nonpop_stack0_marker_blocker < HD:
            if hasattr(attn, "W_v"):
                attn.W_v.data[base + nonpop_stack0_marker_blocker, :] = 0.0
            attn.W_o.data[:, base + nonpop_stack0_marker_blocker] = 0.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.MARK_STACK0
            ] = 60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.CMP + 3
            ] = -15000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.IS_BYTE
            ] = 60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.OP_LI_RELAY
            ] = -60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.OP_LC_RELAY
            ] = -60000.0
            # IMM and other non-pop preservers can still need to reveal the
            # already-stored stack top. At the common one-argument call slot
            # 0xffe8, let the ordinary address-matched lookup rows compete
            # instead of forcing the softmax1 zero sink.
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.ADDR_B0_LO + 8
            ] = -40000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.ADDR_B0_HI + 14
            ] = -30000.0
            attn.W_k.data[
                base + nonpop_stack0_marker_blocker, BD.CONST
            ] = -20.0
            top_store_e8_from_e0_s = 10000.0
            row = base + nonpop_stack0_marker_blocker
            attn.W_q.data[row, BD.MEM_STORE] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.EMBED_LO + 8] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.EMBED_HI + 14] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.ADDR_B0_LO + 0] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.ADDR_B0_HI + 14] = -2.0 * top_store_e8_from_e0_s

        # A load-only source gate must not be the thing that keeps L15 quiet
        # during current store generation. Add an explicit current-MEM query
        # blocker with only a negative constant key, so softmax1 chooses the
        # zero sink instead of overwriting L14's in-flight store bytes.
        current_store_blocker = 38
        current_store_block_s = 10000.0
        attn.W_q.data[base + current_store_blocker, BD.MARK_MEM] = current_store_block_s
        attn.W_q.data[base + current_store_blocker, BD.H3 + mem_i] = current_store_block_s
        attn.W_k.data[base + current_store_blocker, BD.CONST] = -20.0

        # Current-store generation (PSH/JSR/ENT/SI/SC) can carry enough
        # address-like residue in the in-flight MEM section to look like a
        # valid lookup query. L15 lookup heads only read historical MEM
        # entries, so query-side gates with nonnegative K semantics get a
        # strong current-MEM-section blocker. Do not key this on MEM_STORE
        # alone: SI/SC legitimately load STACK0 while also broadcasting a
        # leaked MEM_STORE relay at the STACK0 marker.
        #
        # Do not apply this to rows 1 or 3: their K side can be negative at
        # non-target sources, and a huge negative query there would become a
        # false-positive score.
        for dim in (0, 28, 29, 30, 31, 32, 33):
            attn.W_q.data[base + dim, BD.MARK_MEM] = -100000.0
            attn.W_q.data[base + dim, BD.H3 + mem_i] = -100000.0
        attn.W_q.data[base + 29, BD.MARK_MEM] = -100000.0
        attn.W_q.data[base + 29, BD.H3 + mem_i] = -100000.0
        attn.W_q.data[base + 29, BD.H1 + pc_i] = -20000.0
        attn.W_k.data[base + 29, BD.CONST] = 5.0
        attn.W_q.data[base + 30, BD.H1 + ax_i] = -20000.0
        attn.W_k.data[base + 30, BD.CONST] = 5.0
        attn.W_q.data[base + 31, BD.OP_LI_RELAY] = 20000.0
        if head == 0:
            attn.W_q.data[base + 31, BD.OP_LC_RELAY] = 20000.0
        else:
            attn.W_q.data[base + 31, BD.MARK_AX] = -20000.0
        attn.W_q.data[base + 31, BD.OP_SI] = -20000.0
        attn.W_q.data[base + 31, BD.OP_SC] = -20000.0
        attn.W_k.data[base + 31, BD.MEM_STORE] = 5.0

        # The AX marker is the query position for byte 0. H1[AX] only covers
        # AX value-byte positions, so guard MARK_AX separately and restore the
        # marker path only for head 0 on real LI/LC loads.
        attn.W_q.data[base + 32, BD.MARK_AX] = -20000.0
        attn.W_k.data[base + 32, BD.CONST] = 5.0
        if head == 0:
            attn.W_q.data[base + 33, BD.OP_LI_RELAY] = 20000.0
            attn.W_q.data[base + 33, BD.OP_LC_RELAY] = 20000.0
            attn.W_k.data[base + 33, BD.MEM_STORE] = 5.0

    # The LEV-aware 12-head build adds heads 4-11 for saved-BP and return-PC
    # memory reads. They are load-side heads too, so they must also stay silent
    # while the current step is generating a store MEM section. Use only rows
    # whose K side is positive for all sources; adding MEM_STORE blockers to
    # address or negative-constant rows can create negative-query × negative-key
    # false positives.
    for head in range(4, getattr(attn, "num_heads", 4)):
        base = head * HD
        for row in (0, 36, 37):
            if base + row < attn.W_q.data.shape[0]:
                attn.W_q.data[base + row, BD.MARK_MEM] = -100000.0
                attn.W_q.data[base + row, BD.H3 + mem_i] = -100000.0


def make_layer15_nibble_copy_op() -> Operation:
    """L15 FFN: Conditional nibble copy OUTPUT = EMBED for non-register byte values.

    Pinned to ``layer_idx=15`` via ``kind="block"`` so the bake hits block 15's
    FFN regardless of dep-graph placement. ``migrated=True`` claims this bake
    from the legacy ``set_vm_weights`` pipeline (the inline call has been
    removed at the original site).
    """
    def bake(block, dim_positions, S):
        lower_l15_nibble_copy_ir(
            block.ffn,
            _as_setdim_proxy(dim_positions),
            S=S,
        )

    return Operation(
        name="layer15_nibble_copy",
        phase=15,
        reads={"IS_BYTE", "H1", "H4", "MEM_STORE",
               "EMBED_LO", "EMBED_HI", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MARK_STACK0", "HAS_SE", "CMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_l15_nibble_copy_ir(),
        layer_idx=15,
        migrated=True,
        # ``_set_nibble_copy_ffn`` writes 40 units (see vm_step.py:2411):
        #   16 LO copy + 16 HI copy + 2 PSH SP byte0 + 2 PSH SP byte1 +
        #   2 PSH SP byte2 + 2 PSH BP byte2 + 2 LEA first-step AX byte2 = 42.
        ffn_units_used=42,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_l15_attention_resize_op() -> Operation:
    """Resize L15 attention for late memory/ALU relay heads.

    Migrates the inline resize that previously lived in `set_vm_weights`. The
    extra heads (8-11) hold saved_bp and return_addr reads alongside the
    existing LI/LC/STACK0 reads (heads 0-3) and val heads (4-7). Required
    when the model has L16 (>=17 layers) — `_set_layer15_memory_lookup`
    keys off `attn.num_heads >= 12` to populate the LEV-specific heads.

    LEV still needs 12 heads in 17-layer builds. The strict 16-layer neural
    smoke path needs one additional head for the staged wide-ALU byte relay,
    while keeping ``attn.num_heads < 12`` so the LEV-specific memory lookup
    bake remains disabled.

    phase=14.9 places this after L14 (phase=14) and before
    `_set_layer15_memory_lookup` inside legacy_bake at phase=999.
    """
    def bake(block, dim_positions, S):
        import torch

        n_layers_hint = getattr(block, "_n_layers_hint", None)
        num_heads_new = 14
        if n_layers_hint is not None and n_layers_hint <= 16:
            num_heads_new = 9

        attn = block.attn
        if getattr(attn, "num_heads", 8) >= num_heads_new:
            if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
                attn.alibi_slopes[:4] = 0.05
            return

        d = attn.W_q.shape[1]
        head_dim_old = d // attn.num_heads
        new_q_rows = num_heads_new * head_dim_old

        attn.num_heads = num_heads_new
        attn.head_dim = head_dim_old

        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            new_slopes = torch.tensor(
                [2.0 ** (-8.0 / num_heads_new * (i + 1))
                 for i in range(num_heads_new)]
            )
            # Keep load heads aligned with the memory-lookup bake when this
            # resize block runs after the per-layer L15 attention bake.
            new_slopes[:4] = 0.05
            attn.register_buffer('alibi_slopes', new_slopes)

        old_W_q = attn.W_q.data
        old_W_k = attn.W_k.data
        old_W_v = attn.W_v.data
        attn.W_q = nn.Parameter(torch.zeros(new_q_rows, d))
        attn.W_k = nn.Parameter(torch.zeros(new_q_rows, d))
        attn.W_v = nn.Parameter(torch.zeros(new_q_rows, d))
        attn.W_q.data[:d, :] = old_W_q
        attn.W_k.data[:d, :] = old_W_k
        attn.W_v.data[:d, :] = old_W_v

        old_W_o = attn.W_o.data
        attn.W_o = nn.Parameter(torch.zeros(d, new_q_rows))
        attn.W_o.data[:, :d] = old_W_o

    return Operation(
        name="l15_attention_resize",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        phase=14.9,
        layer_idx=15,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )
