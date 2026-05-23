"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import torch.nn as nn
from collections.abc import Mapping

from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
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
            # gating has limited candidates to stored value bytes, favor the
            # most recent valid store strongly enough for the later SI/SC write
            # to beat an older PSH store with stale address residue.
            attn.alibi_slopes[:4] = 30.0

    # Dim-ownership claims: L15 attn heads 0-3 (memory lookup).
    # Each head writes V slots 32..47 + 48..63 reading CLEAN_EMBED_LO/HI:
    #   W_v[h*HD + 32 + k, CLEAN_EMBED_LO + k]   for k=0..15
    #   W_v[h*HD + 48 + k, CLEAN_EMBED_HI + k]   for k=0..15
    # When num_heads >= 12 (LEV-aware build), heads 4-11 are also active;
    # we restrict claims to heads 0-3 which are the universal load heads
    # to keep the claim set stable across head-count configurations.
    _claims = set()
    for h in range(4):
        for k in range(16):
            _claims.add((15, "attn_W_v", f"{h}_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((15, "attn_W_v", f"{h}_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer15_memory_lookup",
        phase=15,
        reads={"MARK_AX", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_KEY", "MARK_MEM", "MEM_STORE",
               "MEM_ADDR_SRC",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "MARK_STACK0", "IS_BYTE",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1", "H2", "H3", "L2H0",
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
    for head in range(4):
        base = head * HD
        if head == 0:
            # L6/L8 relays can deliver the pop-group flag at ~4.0 by the time
            # L15 runs, while the legacy lookup gate was calibrated for ~1.0.
            # Normalize the two head-0 STACK0 query rows that read CMP[3] so
            # the bias row lands near zero and the store-anchor row lands near
            # its intended +50 when a pop-stack read is active.
            attn.W_q.data[base + 0, BD.CMP + 3] = 500.0
            attn.W_q.data[base + 1, BD.CMP + 3] = 12.5
            attn.W_q.data[base + 0, BD.IS_BYTE] = -2000.0
            attn.W_q.data[base + 1, BD.IS_BYTE] = -50.0
            attn.W_q.data[base + 28, BD.IS_BYTE] = -500.0
            attn.W_q.data[base + 28, BD.CONST] = -20000.0
            attn.W_q.data[base + 28, BD.MARK_AX] = 20000.0
            attn.W_q.data[base + 28, BD.MARK_STACK0] = 20000.0
        else:
            # The legacy byte-select rows for heads 1-3 target the source
            # byte behind the autoregressive query. At query byte N, logits
            # predict byte N+1, so the source must also be byte N+1.
            byte_q_flags = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]
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
            attn.W_q.data[base + source_gate, BD.MARK_AX] = source_gate_s
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
        num_heads_new = 12
        if n_layers_hint is not None and n_layers_hint <= 16:
            num_heads_new = 9

        attn = block.attn
        if getattr(attn, "num_heads", 8) >= num_heads_new:
            if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
                attn.alibi_slopes[:4] = 30.0
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
            new_slopes[:4] = 30.0
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
