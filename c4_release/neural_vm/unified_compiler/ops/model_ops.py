"""Model-level and post-pass op factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
import torch.nn as nn
from .shared import _as_setdim_proxy, setup_token_embeddings, setup_head_weights


def make_io_putchar_routing_op() -> Operation:
    """Bake L6 FFN PUTCHAR routing units (IO_IS_PUTCHAR + AX_CARRY -> OUTPUT).

    Originally an inline call in `set_vm_weights`:
        `_set_io_putchar_routing(ffn6, S, BD)`

    Operates on `model.blocks[6].ffn` (L6 FFN). Modeled as kind="model" so we
    can resolve `ffn6` from the model handle inside the bake_fn.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units we
    program (starting at unit 1500) are present when `_right_size_ffns`
    (called at the end of legacy_bake) prunes dead units. Running at phase
    > 999 would write into already-rightsized FFN slots that no longer exist.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_io_putchar_routing
        proxy = _as_setdim_proxy(dim_positions)
        _set_io_putchar_routing(model.blocks[6].ffn, S, proxy)

    return Operation(
        name="io_putchar_routing",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=998,
        migrated=True,
    )


def make_function_call_weights_op() -> Operation:
    """Bake function-call opcode weights (JSR, ENT, LEV, LEA).

    Originally an inline call in `set_vm_weights`:
        `_set_function_call_weights(model, S, BD, HD)`

    Operates across multiple blocks (L5 attn, L6 attn, L6 ffn) so this is a
    model-level op.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units
    we program (1700-2158) are present when `_right_size_ffns` (called at the
    end of legacy_bake) prunes dead units. Running at phase > 999 would write
    into already-rightsized FFN slots that no longer exist (IndexError).
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_function_call_weights
        proxy = _as_setdim_proxy(dim_positions)
        attn5 = model.blocks[5].attn
        HD = attn5.W_q.shape[0] // attn5.num_heads
        _set_function_call_weights(model, S, proxy, HD)

    # Dim-ownership claims (see c4_release/docs/DIM_OWNERSHIP_REGISTRY.md).
    # `_set_function_call_weights` programs three ENT/JSR relay attention
    # heads. Each head writes V slots [base+1..base+16] (low-nibble path)
    # and [base+17..base+32] (high-nibble path):
    #
    #   L5 attn5 head 5: BP→TEMP at STACK0 (ENT: STACK0 = old_BP)
    #     W_v[5*HD + 1 + k, EMBED_LO + k]    for k=0..15 (slot 1..16)
    #     W_v[5*HD + 17 + k, EMBED_HI + k]   for k=0..15 (slot 17..32)
    #   L5 attn5 head 6: SP→TEMP at BP        (ENT: BP = old_SP - 8)
    #     W_v[6*HD + 1 + k, EMBED_LO + k]    for k=0..15
    #     W_v[6*HD + 17 + k, EMBED_HI + k]   for k=0..15
    #   L6 attn6 head 7: PC OUTPUT→AX_CARRY at STACK0 (JSR return addr)
    #     W_v[7*HD + 1 + k, OUTPUT_LO + k]   for k=0..15
    #     W_v[7*HD + 17 + k, OUTPUT_HI + k]  for k=0..15
    #
    # The 4-tuple claim's ``column`` records the input dim + offset so the
    # registry can distinguish co-tenants of the same row that touch
    # disjoint input columns. In particular, this op's L5 head 5 slot 32
    # claim is (5, "attn_W_v", "5_32", "EMBED_HI+15") — column-disjoint
    # from layer5_fetch's (5, "attn_W_v", "5_32", "CLEAN_EMBED_LO+0"). The
    # row-only registry treated this as a benign collision via
    # ``KNOWN_BENIGN_COLLISIONS`` in tests; the column-aware registry
    # detects the disjointness automatically.
    #
    # Historical context: the deprecated `_set_layer5_fetch` head 6 (deleted
    # 2026-05-11, commit c1a5398) wrote V slots 1..16 on the same matrix
    # using `attn.W_v[6*HD + 1 + k, EMBED_LO + k]`. That collided with
    # this op's head 6 ENT relay at *the same column*:
    # ``(5, "attn_W_v", "6_<k>", "EMBED_LO+<k-1>")`` for k=1..16 — the
    # kind of true collision the column-aware registry still catches.
    _claims = set()
    # L5 attn5 head 5 ENT relay
    for k in range(16):
        _claims.add(
            (5, "attn_W_v", f"5_{1 + k}", f"EMBED_LO+{k}")
        )
        _claims.add(
            (5, "attn_W_v", f"5_{17 + k}", f"EMBED_HI+{k}")
        )
    # L5 attn5 head 6 ENT relay
    for k in range(16):
        _claims.add(
            (5, "attn_W_v", f"6_{1 + k}", f"EMBED_LO+{k}")
        )
        _claims.add(
            (5, "attn_W_v", f"6_{17 + k}", f"EMBED_HI+{k}")
        )
    # L6 attn6 head 7 JSR PC OUTPUT relay
    for k in range(16):
        _claims.add(
            (6, "attn_W_v", f"7_{1 + k}", f"OUTPUT_LO+{k}")
        )
        _claims.add(
            (6, "attn_W_v", f"7_{17 + k}", f"OUTPUT_HI+{k}")
        )

    return Operation(
        name="function_call_weights",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=998,
        migrated=True,
        claims=_claims,
    )


def make_opcode_relay_head_op() -> Operation:
    """L6 attention head 6: relay opcode flags from AX -> SP/STACK0/PC markers.

    Originally an inline call in ``set_vm_weights`` (with two preceding
    ``attn6.alibi_slopes`` mutations folded in)::

        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes[6] = 5.0
            attn6.alibi_slopes[7] = 5.0  # JSR PC+5 relay: steep for head 7
        _set_opcode_relay_head(attn6, S, BD, HD)

    The L5 FFN decodes opcodes only at the AX marker, but L6 FFN consumes
    OP_PSH/OP_ADJ/OP_LEV/OP_JSR/OP_ENT and a pop-group flag at SP, STACK0,
    BP, PC, and MEM markers. Head 6 copies those flags across markers.
    ALiBi slope=5.0 is also set on head 7 (used by JSR PC+5 relay configured
    by ``_set_function_call_weights``).

    Phase=1002 (AFTER ``legacy_bake`` at 999) is REQUIRED because the
    inline section that this op replaces is preceded inside
    ``set_vm_weights`` by::

        attn6.alibi_slopes.fill_(0.0)
        attn6.alibi_slopes[0] = 5.0
        attn6.alibi_slopes[1] = 5.0
        ...

    which still fires from inside legacy_bake. Running this op at phase
    <999 would have its alibi_slopes[6]/[7]=5.0 writes wiped by that
    ``fill_(0.0)`` call. Phase=1002 also slots cleanly after
    ``head_bake`` (1000) and ``embedding_bake`` (1001), and before the
    defensive post-pass ``branch_override_patch`` (1100).

    Head 6 attn weights (W_q/W_k/W_v/W_o) don't conflict with any other
    L6 attn writes in ``set_vm_weights``: head 0-5 are programmed by
    ``_set_layer6_attn`` and ``_set_bz_bnz_relay`` (now migrated to
    phases 998.5/.7), and head 7 Q/K is programmed by
    ``_set_function_call_weights`` (phase=998) plus
    ``_set_layer6_relay_heads`` (phase=998.6) — none of which touch
    head 6 slots, so phase ordering against those is irrelevant.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_opcode_relay_head
        proxy = _as_setdim_proxy(dim_positions)
        attn = model.blocks[6].attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[6] = 5.0
            attn.alibi_slopes[7] = 5.0  # JSR PC+5 relay: steep for head 7
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_opcode_relay_head(attn, S, proxy, HD)

    return Operation(
        name="opcode_relay_head",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1002,
        migrated=True,
    )


def make_residual_alibi_slopes_op() -> Operation:
    """Bake the residual ALiBi-slope mutations previously inline in set_vm_weights.

    Migrated 2026-05-11 (architectural milestone — final piece of the
    set_vm_weights → compiler-ops migration). Replaces the per-layer
    ``attn.alibi_slopes.fill_(...)`` and ``alibi_slopes[i] = ...`` writes
    that used to live in the inline body of ``set_vm_weights``:

      - L6 (head 0..4): fill_(0.0), then [0]=5.0, [1]=5.0, [2]=0.5,
        [3]=0.5, [4]=5.0 — must run BEFORE the legacy_bake retirement
        because ``make_opcode_relay_head_op`` (phase=1002) writes
        [6]=5.0 / [7]=5.0 and relies on the head-6/7 slots already being
        zero (per the docstring's "fill_(0.0) wipes" argument).
      - L8: fill_(0.5) — head-3/4 multibyte fetch / OP_IMM relay use
        the L8-wide gentle recency.
      - L10 head 0..4 (lookup) / 0..3 (efficient): steep carry relay +
        gentle byte passthrough slopes. Mode-conditional.
      - L14: fill_(0.1) — slight recency bias for MEM generation.
      - L15: fill_(0.01) — gentle latest-write-wins bias for memory lookup.

    Phase=999 places this exactly where ``legacy_bake`` used to run,
    preserving the previous override contract relative to phase-1002
    ``opcode_relay_head`` (which still needs to write [6]/[7]=5.0
    AFTER the L6 fill_(0.0)).
    """
    def _bake(model, dim_positions, S):
        # L6: head 0..4 slopes
        attn6 = model.blocks[6].attn
        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes.fill_(0.0)
            attn6.alibi_slopes[0] = 5.0
            attn6.alibi_slopes[1] = 5.0
            attn6.alibi_slopes[2] = 0.5  # STACK0←AX relay: prefer nearest AX marker
            attn6.alibi_slopes[3] = 0.5  # SP←AX relay: prefer nearest AX marker
            attn6.alibi_slopes[4] = 5.0  # BZ/BNZ relay: attend to nearest AX marker
            # Softmax-sharpness fix (head 5 — first-step OP flag / FETCH
            # relay programmed by _set_layer6_attn). The audit (87442ad)
            # flagged head 5 with slope=0 leading to mass=0.10 at the
            # synthetic K target — the AX-marker-to-PC-marker hop is at
            # distance 4 in the audit context, and with slope=0 the head
            # has no positional preference between target and runner-up.
            # Raising slope to 1.0 closes the gap (1.0 * 4 = 4 nats per
            # the audit's gap recommendation). Paired with the K-scale
            # 10x bump in make_layer6_attn_bake_op (phase 998.5) so the
            # head clears the 99% sharpness threshold in strict mode.
            attn6.alibi_slopes[5] = 1.0

        # L8: per-layer recency
        attn8 = model.blocks[8].attn
        if hasattr(attn8, 'alibi_slopes') and attn8.alibi_slopes is not None:
            attn8.alibi_slopes.fill_(0.5)

        # L14: slight recency bias for same-step preference
        if len(model.blocks) > 14:
            attn14 = model.blocks[14].attn
            if hasattr(attn14, 'alibi_slopes') and attn14.alibi_slopes is not None:
                attn14.alibi_slopes.fill_(0.1)

        # L15: gentle recency bias for latest-write-wins
        if len(model.blocks) > 15:
            attn15 = model.blocks[15].attn
            if hasattr(attn15, 'alibi_slopes') and attn15.alibi_slopes is not None:
                attn15.alibi_slopes.fill_(0.01)

    return Operation(
        name="residual_alibi_slopes",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=999,
        migrated=True,
    )


# ---------------------------------------------------------------------------
# Model-level post-passes
# ---------------------------------------------------------------------------


def make_branch_override_patch_op() -> Operation:  # noqa: E302
    """Defensive gate that suppresses spurious branch/LEV-override FFN units.

    Any FFN unit firing with positive MARK_PC AND a positive
    OP_JMP/OP_LEV/OP_BZ/OP_BNZ/CMP[0] trigger AND writing OUTPUT_LO/HI gets
    strong negative weights for non-target opcodes — only the legit branch/LEV
    opcode lets the unit fire. Without this, L5 attention leaks opcode flags
    into MARK_PC of subsequent steps and FFN units fire spuriously.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        OPCODE_BLOCK_MAP = {
            BD.OP_JMP: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
            BD.OP_LEV: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
            BD.OP_BZ:  [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BNZ, BD.OP_JSR],
            BD.OP_BNZ: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_JSR],
            BD.CMP + 0: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_LEV,
                         BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                         BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                         BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                         BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                         BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                         BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
        }
        branch_override_patches = 0
        for block_idx in range(len(model.blocks)):
            ffn = model.blocks[block_idx].ffn
            if not (hasattr(ffn, 'W_up') and isinstance(getattr(ffn, 'W_up', None), nn.Parameter)):
                continue
            hidden_dim = ffn.W_up.shape[0]
            for u in range(hidden_dim):
                mark_pc_w = ffn.W_up[u, BD.MARK_PC].item()
                if mark_pc_w <= 50:
                    continue
                writes_output = (
                    ffn.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, u].abs().max().item() > 0.01
                    or ffn.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, u].abs().max().item() > 0.01
                )
                if not writes_output:
                    continue
                for trigger_dim, blockers in OPCODE_BLOCK_MAP.items():
                    trigger_w = ffn.W_up[u, trigger_dim].item()
                    if trigger_w <= 5:
                        continue
                    for opcode_dim in blockers:
                        cur = ffn.W_up[u, opcode_dim].item()
                        ffn.W_up.data[u, opcode_dim] = min(cur, -S)
                    branch_override_patches += 1
                    break
        print(f"  BRANCH OVERRIDE PATCH: {branch_override_patches} units gated")

    return Operation(
        name="branch_override_patch",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1100,
        migrated=True,
    )


def make_l6_dead_unit_zero_op() -> Operation:
    """Defensive gate that zeros L6 FFN units misreading OUTPUT_BYTE residuals.

    Originally an inline post-pass in `set_vm_weights` (BUG FIX 2026-04-09 part
    8c/8g). Unprogrammed L6 FFN units can fire spuriously when:

    - they write to OUTPUT_LO/HI (W_down rows in those slices are non-zero), AND
    - they have strong W_up reads on OUTPUT_BYTE_LO/HI (which carry residual
      values from the prior IO step), AND
    - they don't have positive marker weights (MARK_PC, MARK_STACK0, MARK_BP)
      indicating intentional TEMP usage by JSR PC override / ENT.

    The patch zeros W_up/W_gate rows, W_down columns, and biases for every
    matching unit so they cannot fire at any position.

    Operates on `model.blocks[6].ffn` (the L6 routing FFN). Phase=1160 so it
    runs after branch_override_patch (phase=1100) and the head/embedding bakes
    (1000/1001), and before right_size_ffns (1200) — preserving the original
    in-set_vm_weights ordering where this pass ran before the right-size pass.

    Migrated=True: the corresponding inline block in `set_vm_weights` has been
    removed to avoid double-bake.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        ffn6 = model.blocks[6].ffn
        if not (hasattr(ffn6, 'W_up') and isinstance(getattr(ffn6, 'W_up', None), nn.Parameter)):
            return
        hidden_dim = ffn6.W_up.shape[0]
        patched_count = 0
        for u in range(hidden_dim):
            writes_output_lo = (
                ffn6.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO + 16, u].abs().max().item() > 0.01
            )
            writes_output_hi = (
                ffn6.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, u].abs().max().item() > 0.01
            )
            if not (writes_output_lo or writes_output_hi):
                continue

            # Skip units that legitimately use TEMP at marker positions.
            if ffn6.W_up[u, BD.MARK_PC].item() > 10:  # JSR PC override uses S=100
                continue
            if ffn6.W_up[u, BD.MARK_STACK0].item() > 10:  # ENT uses TEMP at STACK0
                continue
            if ffn6.W_up[u, BD.MARK_BP].item() > 10:  # ENT uses TEMP at BP
                continue

            output_byte_lo_weight = (
                ffn6.W_up[u, BD.OUTPUT_BYTE_LO:BD.OUTPUT_BYTE_LO + 16].abs().max().item()
            )
            output_byte_hi_weight = (
                ffn6.W_up[u, BD.OUTPUT_BYTE_HI:BD.OUTPUT_BYTE_HI + 16].abs().max().item()
            )

            if output_byte_lo_weight > 50 or output_byte_hi_weight > 50:
                ffn6.W_up.data[u, :] = 0
                ffn6.W_gate.data[u, :] = 0
                ffn6.W_down.data[:, u] = 0
                ffn6.b_up.data[u] = 0
                ffn6.b_gate.data[u] = 0
                patched_count += 1
        if patched_count:
            print(f"  L6 DEAD-UNIT ZERO: {patched_count} units zeroed")

    return Operation(
        name="l6_dead_unit_zero",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1160,
        migrated=True,
    )


def make_l7_dead_unit_zero_op() -> Operation:
    """Defensive gate that suppresses L7 FFN units firing at PC marker.

    Originally an inline post-pass in `set_vm_weights` (BUG FIX 2026-04-17).
    Some FFN units (e.g. units 746/754) carry W_up[OP_ENT]=100, W_up[MARK_SP]=100,
    b_up=-150 — they're meant for ENT SP byte 0 generation but `OP_ENT * 5 = 500`
    overcomes the bias even when MARK_SP=0, so they spuriously fire at PC marker
    positions. The patch adds strong negative MARK_PC and IS_BYTE weights to the
    matching units so they cannot fire outside their intended marker position.

    Matching criteria:
      - writes to OUTPUT_LO/HI, AND
      - has strong (>50) W_up reads on OP_ENT/OP_LEV/OP_JSR/OP_LEA, AND
      - lacks existing MARK_PC suppression (W_up[MARK_PC] >= -100), AND
      - is not a byte-position unit (no positive IS_BYTE / BYTE_INDEX_0..3
        weights >5), AND
      - does not legitimately fire at PC marker (W_up[MARK_PC] <= 10).

    Operates on `model.blocks[6].ffn` (preserving the original code's choice;
    the legacy comment claims `blocks[6] = L7` but the same FFN is also the L6
    routing FFN in the current block layout — the pass is purely defensive and
    only modifies units matching the criteria above, so co-location is safe).

    Phase=1170 so it runs after the L6 dead-unit zero pass (phase=1160) and
    before right_size_ffns (1200) — preserving original ordering.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        ffn7 = model.blocks[6].ffn  # blocks[6] = L7 (per original code's comment)
        if not (hasattr(ffn7, 'W_up') and isinstance(getattr(ffn7, 'W_up', None), nn.Parameter)):
            return
        hidden_dim = ffn7.W_up.shape[0]
        patched_l7_count = 0
        for u in range(hidden_dim):
            writes_output_lo = (
                ffn7.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO + 16, u].abs().max().item() > 0.01
            )
            writes_output_hi = (
                ffn7.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, u].abs().max().item() > 0.01
            )
            if not (writes_output_lo or writes_output_hi):
                continue

            has_strong_opcode = (
                ffn7.W_up[u, BD.OP_ENT].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_LEV].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_JSR].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_LEA].abs().item() > 50
            )

            has_mark_pc_suppression = ffn7.W_up[u, BD.MARK_PC].item() < -100

            is_byte_unit = (
                ffn7.W_up[u, BD.IS_BYTE].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_0].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_1].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_2].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_3].item() > 5
            )

            # Skip units that legitimately fire at PC marker.
            if ffn7.W_up[u, BD.MARK_PC].item() > 10:
                continue

            if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:
                ffn7.W_up.data[u, BD.MARK_PC] = -S * 100  # -1000 when MARK_PC = 1
                ffn7.W_up.data[u, BD.IS_BYTE] = -S * 100  # -1000 when IS_BYTE = 1
                patched_l7_count += 1
        if patched_l7_count:
            print(f"  L7 DEAD-UNIT ZERO: {patched_l7_count} units suppressed")

    return Operation(
        name="l7_dead_unit_zero",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1170,
        migrated=True,
    )


def make_right_size_ffns_op() -> Operation:
    """Trim each block's FFN hidden dim to actually-programmed unit count."""
    def bake(model, dim_positions, S):
        from ...vm_step import _right_size_ffns
        _right_size_ffns(model)

    return Operation(
        name="right_size_ffns",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1200,
        migrated=True,
    )


def make_expand_wrapper_blocks_op() -> Operation:
    """Split HybridALUBlock + post_ops into separate transformer blocks."""
    def bake(model, dim_positions, S):
        from ...vm_step import _expand_wrapper_blocks
        _expand_wrapper_blocks(model)

    return Operation(
        name="expand_wrapper_blocks",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1300,
        migrated=True,
    )


def make_head_bake_op() -> Operation:
    """Bake the output projection head: byte/marker token logits.

    Phase=1000 so it runs AFTER legacy_bake (phase=999); the corresponding
    head section in `set_vm_weights` has been removed to avoid double-bake.
    """
    def _bake(model, dim_positions, S):
        setup_head_weights(model.head, dim_positions)

    return Operation(
        name="head_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1000,
    )


def make_embedding_bake_op() -> Operation:
    """Bake the per-token embedding table.

    Phase=1001 so it runs AFTER legacy_bake (phase=999) and head_bake (1000);
    the corresponding embedding section in `set_vm_weights` has been removed
    to avoid double-bake.
    """
    def _bake(model, dim_positions, S):
        setup_token_embeddings(model.embed.embed.weight, dim_positions)

    return Operation(
        name="embedding_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1001,
    )


def make_initial_pc_bake_op() -> Operation:
    """Bake the initial PC value (PC_OFFSET) into the REG_PC token embedding.

    Phase=1001.5 so it runs AFTER `embedding_bake` (1001), which zeros and
    rewrites the embedding table. By running just after, we add (rather than
    have-overwritten) the initial-PC pattern on top of the standard REG_PC
    marker pattern.

    Migration 2026-05-11 (i1): replaces the runtime `_inject_initial_pc`
    method on NeuralVMEmbedding. The runtime injection wrote +1.0 to
    EMBED_LO+(PC_OFFSET & 0xF) and +1.0 to EMBED_HI+((PC_OFFSET>>4) & 0xF)
    only at the FIRST REG_PC marker (one with no preceding STEP_END). The
    same residual contribution can be produced by baking those +1.0 values
    into the REG_PC token-embedding row. Because the bake fires at every
    REG_PC position (steps 1+ too), a pair of L3-FFN cancel units (added
    inline at `_set_layer3_ffn`) subtracts -1.0 from those same EMBED_LO/HI
    dims at MARK_PC AND HAS_SE positions — leaving step-1+ residuals
    bit-identical to the pre-migration behavior.
    """
    def _bake(model, dim_positions, S):
        import torch
        from ...vm_step import Token
        from ...constants import PC_OFFSET

        def D(name):
            if dim_positions is not None and name in dim_positions:
                return dim_positions[name]
            from ...vm_step import _SetDim
            return getattr(_SetDim, name)

        embed_weight = model.embed.embed.weight
        if Token.REG_PC >= embed_weight.shape[0]:
            return  # No REG_PC token in vocab (shouldn't happen).

        embed_lo = D("EMBED_LO")
        embed_hi = D("EMBED_HI")
        init_pc_lo = PC_OFFSET & 0xF
        init_pc_hi = (PC_OFFSET >> 4) & 0xF

        with torch.no_grad():
            embed_weight[Token.REG_PC, embed_lo + init_pc_lo] = 1.0
            embed_weight[Token.REG_PC, embed_hi + init_pc_hi] = 1.0

    return Operation(
        name="initial_pc_bake",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1001.5,
    )


def make_contract_validation_op() -> Operation:
    """Run the contract validator. Previously inline in set_vm_weights.

    The validator prints any contract errors but does not raise — it's
    diagnostic only. Kept as a compiler op so the diagnostic still fires
    after every full bake. Phase=1199 (just before ``right_size_ffns``
    at 1200) so all layer/block/model bakes have completed.
    """
    def _bake(model, dim_positions, S):
        from ...dim_registry import (
            build_default_registry,
            build_default_contracts,
            ContractValidator,
        )
        reg = build_default_registry()
        contracts = build_default_contracts(reg)
        errors = ContractValidator(reg, contracts).validate()
        if errors:
            for e in errors:
                print(f"  CONTRACT: {e}")

    return Operation(
        name="contract_validation",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        phase=1199,
        migrated=True,
    )

