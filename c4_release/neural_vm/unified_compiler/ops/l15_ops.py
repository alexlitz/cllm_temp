"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy
import torch.nn as nn


def make_nibble_copy_ffn_op() -> Operation:
    """Conditional nibble copy: OUTPUT = EMBED for non-register byte values."""
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_nibble_copy_ffn
        _set_nibble_copy_ffn(ffn, S, _as_setdim_proxy(dim_positions))

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
    )


def make_layer15_memory_lookup_op() -> Operation:
    """L15 attention: memory-lookup heads for LI/LC."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer15_memory_lookup
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer15_memory_lookup(attn, S, _as_setdim_proxy(dim_positions), HD)

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
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_KEY", "MARK_MEM",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        layer_idx=15,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        # Produces the loaded memory byte values in OUTPUT_LO/HI at AX byte 0
        # (LI/LC load destination) and at STACK0 byte positions (POP group
        # *SP read for binary ops). The L15 attn heads attend to the matching
        # MEM val byte position via ADDR_KEY and copy CLEAN_EMBED into
        # OUTPUT. Marker "AX_byte0" is the canonical AX-side anchor.
        produces={
            "OUTPUT_LO": "AX_byte0",
            "OUTPUT_HI": "AX_byte0",
        },
    )


def make_layer15_nibble_copy_op() -> Operation:
    """L15 FFN: Conditional nibble copy OUTPUT = EMBED for non-register byte values.

    Pinned to ``layer_idx=15`` via ``kind="block"`` so the bake hits block 15's
    FFN regardless of dep-graph placement. ``migrated=True`` claims this bake
    from the legacy ``set_vm_weights`` pipeline (the inline call has been
    removed at the original site).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_nibble_copy_ffn
        _set_nibble_copy_ffn(block.ffn, S, _as_setdim_proxy(dim_positions))

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
        layer_idx=15,
        migrated=True,
        # ``_set_nibble_copy_ffn`` writes 40 units (see vm_step.py:2411):
        #   16 LO copy + 16 HI copy + 2 PSH SP byte0 + 2 PSH SP byte1 +
        #   2 PSH SP byte2 + 2 LEA first-step AX byte2 = 40.
        ffn_units_used=40,
    )


def make_l15_attention_resize_op() -> Operation:
    """Resize L15 attention from 8 heads to 12 heads (LEV Phase 2).

    Migrates the inline resize that previously lived in `set_vm_weights`. The
    extra heads (8-11) hold saved_bp and return_addr reads alongside the
    existing LI/LC/STACK0 reads (heads 0-3) and val heads (4-7). Required
    when the model has L16 (>=17 layers) — `_set_layer15_memory_lookup`
    keys off `attn.num_heads >= 12` to populate the LEV-specific heads.

    Resizes (assuming d=512, head_dim=64): W_q/W_k/W_v (512,512)->(768,512),
    W_o (512,512)->(512,768), alibi_slopes 8->12. Preserves heads 0-7 weights
    in the first d rows/cols.

    phase=14.9 places this after L14 (phase=14) and before
    `_set_layer15_memory_lookup` inside legacy_bake at phase=999.
    """
    def bake(block, dim_positions, S):
        import torch

        # Match the original `len(model.blocks) > 16` guard. The dispatcher
        # stashes _n_layers_hint just before invoking this bake_fn.
        n_layers_hint = getattr(block, "_n_layers_hint", None)
        if n_layers_hint is not None and n_layers_hint <= 16:
            return

        attn = block.attn
        if getattr(attn, "num_heads", 8) >= 12:
            return

        d = attn.W_q.shape[1]
        head_dim_old = d // attn.num_heads
        num_heads_new = 12
        new_q_rows = num_heads_new * head_dim_old

        attn.num_heads = num_heads_new
        attn.head_dim = head_dim_old

        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            new_slopes = torch.tensor(
                [2.0 ** (-8.0 / num_heads_new * (i + 1))
                 for i in range(num_heads_new)]
            )
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
        phase=14.9,
        layer_idx=15,
        migrated=True,
    )


