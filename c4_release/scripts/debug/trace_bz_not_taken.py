#!/usr/bin/env python3
"""Trace OP_BZ activation and L6 FFN unit firing at step-1 PC marker for
test_bz_not_taken[1] in pure_neural mode.

Phase 4 BZ/BNZ diagnosis: The "not_taken" tests fail because PC is incorrectly
redirected. The leading hypothesis is that OP_BZ activation leaks above 5 at
the step-1 PC marker, triggering the L6 BZ AX-passthrough units (vm_step.py
4917-4933) to write garbage AX_CARRY into OUTPUT at the PC marker.

This script:
  1. Builds the same `pure_neural` runner used by the test fixture.
  2. Programs the BZ_not_taken[1] bytecode into the runner.
  3. Generates 2 autoregressive steps' worth of tokens (so the model has
     produced the step-1 token sequence).
  4. Walks layer-by-layer over the resulting context, tracking the
     residual stream at the step-1 PC marker position. After each block
     prints OP_BZ, OP_BNZ, OP_JMP, MARK_PC, MARK_AX, CMP[2..5] and the
     OUTPUT_LO/HI byte. Then identifies which L6 FFN units fire at the
     PC marker with what hidden activations and what they write to
     OUTPUT_LO/HI.
"""

import sys
import torch

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def main():
    print("Building pure_neural runner...", flush=True)
    runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    model = runner.model
    model.eval()
    device = next(model.parameters()).device

    # BZ not_taken[1] program: AX=1, BZ skip, IMM 7, EXIT.
    prog = [
        (Opcode.IMM, 1),
        (Opcode.BZ, 4),
        (Opcode.IMM, 7),
        Opcode.EXIT,
    ]
    bytecode = _make_bc(prog)
    print(f"Bytecode: {[hex(b) for b in bytecode]}", flush=True)

    # Build context exactly the way runner.run does. We re-use the runner's
    # _build_context for prefix construction.
    context = runner._build_context(bytecode, b"", [], b"")
    prefix_len = len(context)

    # Pure-neural generation is degenerate (model emits all-15s after step 0),
    # so autoregressive generation cannot produce a real step-1 PC marker. We
    # teacher-force step 0 (the IMM 1 step) with the "correct" oracle state,
    # then put a clean PC marker (token 257) at the start of step 1 so the
    # model's residual stream at that position represents what the network
    # would see if step 0 had executed correctly. This isolates "the BZ
    # opcode-fetch and PC-update path at step 1" from "step 0 execution
    # correctness".
    #
    # Step 0 oracle (executes IMM 1):
    #   PC=0 (current), AX_initial=0, SP_initial=0x10000, BP=0, STACK0=0,
    #   after step: PC=4, AX=1
    # Step 1 oracle PC marker at this point: just emit token REG_PC. The model
    # should ATTEND TO the bytecode prefix (BZ at byte 4) and produce
    # OP_BZ activation at the PC marker.

    def append_step(ctx, pc, ax, sp, bp, stack0, mem_addr=0, mem_val=0):
        """Append a full 35-token step to ctx in oracle form."""
        ctx.append(Token.REG_PC)
        for i in range(4):
            ctx.append((pc >> (i * 8)) & 0xFF)
        ctx.append(Token.REG_AX)
        for i in range(4):
            ctx.append((ax >> (i * 8)) & 0xFF)
        ctx.append(Token.REG_SP)
        for i in range(4):
            ctx.append((sp >> (i * 8)) & 0xFF)
        ctx.append(Token.REG_BP)
        for i in range(4):
            ctx.append((bp >> (i * 8)) & 0xFF)
        ctx.append(Token.STACK0)
        for i in range(4):
            ctx.append((stack0 >> (i * 8)) & 0xFF)
        ctx.append(Token.MEM)
        for i in range(4):
            ctx.append((mem_addr >> (i * 8)) & 0xFF)
        for i in range(4):
            ctx.append((mem_val >> (i * 8)) & 0xFF)
        ctx.append(Token.STEP_END)

    # Teacher-force step 0 output (the "after IMM 1 executed" state):
    #   PC_OFFSET=2; step 0 executes IMM at PC=2, emits PC=10 (next instr); AX=1
    sp0 = 0x10000
    append_step(context, pc=10, ax=1, sp=sp0, bp=0, stack0=0)
    print(f"Teacher-forced step 0 (IMM 1 result): PC=10, AX=1", flush=True)

    # Now append step-1 PC marker only — the model would emit subsequent
    # tokens autoregressively, but we just need the residual at the PC marker
    # position.
    step1_pc_marker_pos = len(context)
    context.append(Token.REG_PC)
    print(f"Appended step-1 PC marker at position {step1_pc_marker_pos}", flush=True)

    # Also build a small generated section so we see what the model thinks PC
    # should be at step 1 (without overrides). This is informational.
    print(f"Asking model to generate step-1 PC bytes...", flush=True)
    with torch.no_grad():
        for k in range(5):
            tok = model.generate_next(context, use_incremental=False)
            context.append(tok)
        print(f"  Step-1 emitted after PC marker: {context[step1_pc_marker_pos:step1_pc_marker_pos+5]}", flush=True)
        # Pop those so we have just the PC marker as the last position when
        # inspecting layer activations.
    # We'll do the layer trace on the context that includes those bytes too,
    # to see the trajectory:
    print(f"\nStep-1 PC marker absolute position: {step1_pc_marker_pos}", flush=True)

    # Build a tensor over the full context (matching the model's autoregressive
    # invariant: at the moment of generating token at index N, the model saw
    # tokens [0..N-1]). To replay the activations seen at step-1 PC marker, we
    # need to invoke forward on the context ending at step1_pc_marker_pos.
    # But: the position step1_pc_marker_pos itself is also informative — the
    # PC-marker token was already in context when the model emitted PC byte 0.
    # We do a forward over the full generated context and inspect position
    # step1_pc_marker_pos in the resulting residual stream.
    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    print(f"Token tensor shape: {token_ids.shape}", flush=True)

    BZ_DIM = BD.OP_BZ        # 266
    BNZ_DIM = BD.OP_BNZ      # 267
    JMP_DIM = BD.OP_JMP      # 264
    PC_DIM = BD.MARK_PC      # 0
    AX_DIM = BD.MARK_AX      # 1
    CMP_BASE = BD.CMP        # 396
    OUT_LO = BD.OUTPUT_LO    # 174
    OUT_HI = BD.OUTPUT_HI    # 190
    HAS_SE = BD.HAS_SE       # 137

    pos = step1_pc_marker_pos

    def dump_at(label, x):
        v = x[0, pos]
        op_bz = v[BZ_DIM].item()
        op_bnz = v[BNZ_DIM].item()
        op_jmp = v[JMP_DIM].item()
        mark_pc = v[PC_DIM].item()
        mark_ax = v[AX_DIM].item()
        cmp2 = v[CMP_BASE + 2].item()
        cmp3 = v[CMP_BASE + 3].item()
        cmp4 = v[CMP_BASE + 4].item()
        cmp5 = v[CMP_BASE + 5].item()
        has_se = v[HAS_SE].item()
        out_lo_max = v[OUT_LO:OUT_LO+16].argmax().item()
        out_lo_val = v[OUT_LO:OUT_LO+16].max().item()
        out_hi_max = v[OUT_HI:OUT_HI+16].argmax().item()
        out_hi_val = v[OUT_HI:OUT_HI+16].max().item()
        print(f"  [{label}] OP_BZ={op_bz:+8.3f} OP_BNZ={op_bnz:+8.3f} OP_JMP={op_jmp:+8.3f} "
              f"MARK_PC={mark_pc:+5.2f} MARK_AX={mark_ax:+5.2f} HAS_SE={has_se:+5.2f}", flush=True)
        print(f"        CMP[2,3,4,5]={cmp2:+5.2f},{cmp3:+5.2f},{cmp4:+5.2f},{cmp5:+5.2f} "
              f"OUTPUT_LO[argmax={out_lo_max}, val={out_lo_val:.2f}] OUTPUT_HI[argmax={out_hi_max}, val={out_hi_val:.2f}]", flush=True)

    print("\n=== Layer-by-layer trace at step-1 PC marker (pos {}) ===\n".format(pos), flush=True)
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        dump_at("after embed", x)
        for i, block in enumerate(model.blocks):
            x_in = x
            # Attention
            x_attn = block.attn(x_in)
            dump_at(f"L{i} after attn", x_attn)
            # FFN
            x_ffn = block.ffn(x_attn)
            dump_at(f"L{i} after ffn ", x_ffn)
            # Print delta per layer
            delta = x_ffn[0, pos] - x_in[0, pos]
            dlo_max = delta[OUT_LO:OUT_LO+16].abs().argmax().item()
            dlo_val = delta[OUT_LO:OUT_LO+16][dlo_max].item()
            dhi_max = delta[OUT_HI:OUT_HI+16].abs().argmax().item()
            dhi_val = delta[OUT_HI:OUT_HI+16][dhi_max].item()
            d_bz = delta[BZ_DIM].item()
            d_jmp = delta[JMP_DIM].item()
            print(f"        L{i} delta: dOP_BZ={d_bz:+.2f} dOP_JMP={d_jmp:+.2f} "
                  f"dOUT_LO[{dlo_max}]={dlo_val:+.3f} dOUT_HI[{dhi_max}]={dhi_val:+.3f}", flush=True)
            x = x_ffn

    # === Focused L3 FFN unit-level trace ===
    print("\n=== L3 FFN unit-level trace at step-1 PC marker ===\n", flush=True)
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for i in range(3):
            x = model.blocks[i](x)
        # x is now post-L2. Run L3 attention.
        x_l3_attn = model.blocks[3].attn(x)
        dump_at("post-L2+L3-attn", x_l3_attn)

        # Also dump EMBED_LO at PC marker after L3 attn (carry-forward result)
        EMBED_LO_DIM = BD.EMBED_LO
        EMBED_HI_DIM = BD.EMBED_HI
        elo = x_l3_attn[0, pos, EMBED_LO_DIM:EMBED_LO_DIM+16]
        ehi = x_l3_attn[0, pos, EMBED_HI_DIM:EMBED_HI_DIM+16]
        elo_argmax = elo.argmax().item()
        ehi_argmax = ehi.argmax().item()
        print(f"  After L3 attn at PC marker: EMBED_LO[{elo_argmax}]={elo[elo_argmax].item():.3f}, EMBED_HI[{ehi_argmax}]={ehi[ehi_argmax].item():.3f}", flush=True)
        # Print non-zero EMBED_LO
        nonzero_elo = (elo.abs() > 0.01).nonzero(as_tuple=True)[0].tolist()
        print(f"    Non-zero EMBED_LO indices: {[(i, elo[i].item()) for i in nonzero_elo]}", flush=True)
        nonzero_ehi = (ehi.abs() > 0.01).nonzero(as_tuple=True)[0].tolist()
        print(f"    Non-zero EMBED_HI indices: {[(i, ehi[i].item()) for i in nonzero_ehi]}", flush=True)

        # Also dump EMBED at step 0 PC byte 0 (which is the source of carry-forward)
        # Step 0 PC byte 0 is at position prefix_len + 1
        step0_pc_byte0 = prefix_len + 1
        elo_src = x_l3_attn[0, step0_pc_byte0, EMBED_LO_DIM:EMBED_LO_DIM+16]
        ehi_src = x_l3_attn[0, step0_pc_byte0, EMBED_HI_DIM:EMBED_HI_DIM+16]
        print(f"  Step 0 PC byte 0 (pos {step0_pc_byte0}, token={context[step0_pc_byte0]}): EMBED_LO[{elo_src.argmax().item()}]={elo_src.max().item():.3f}, EMBED_HI[{ehi_src.argmax().item()}]={ehi_src.max().item():.3f}", flush=True)
        # Check L1H1/L1H0 flags at step 0 PC byte 0 (this is the KEY for L3 attn carry-forward)
        L1H1_PC = 123  # BD.L1H1 + PC_I=0
        L1H0_PC = 116
        l1h1_val = x_l3_attn[0, step0_pc_byte0, L1H1_PC].item()
        l1h0_val = x_l3_attn[0, step0_pc_byte0, L1H0_PC].item()
        is_mark = x_l3_attn[0, step0_pc_byte0, 7].item()  # IS_MARK
        is_byte = x_l3_attn[0, step0_pc_byte0, 6].item()  # IS_BYTE
        print(f"    L1H1[PC]={l1h1_val:.3f}, L1H0[PC]={l1h0_val:.3f}, IS_MARK={is_mark:.3f}, IS_BYTE={is_byte:.3f}", flush=True)

        # Manually compute L3 attn head 0 (PC carry-forward) attention scores
        attn3 = model.blocks[3].attn
        # Re-run the attention at this layer manually: need input to L3 attn
        x_pre_l3 = x  # already post-L2
        # Compute Q, K for head 0
        W_q = attn3.W_q if not attn3.W_q.is_sparse else attn3.W_q.to_dense()
        W_k = attn3.W_k if not attn3.W_k.is_sparse else attn3.W_k.to_dense()
        Q_full = torch.nn.functional.linear(x_pre_l3, W_q)  # [B, S, D]
        K_full = torch.nn.functional.linear(x_pre_l3, W_k)
        HD = W_q.shape[0] // attn3.num_heads
        head_idx = 0  # PC carry-forward
        base = head_idx * HD
        # Q for head 0 at PC marker position
        Q_h0 = Q_full[0, pos, base:base+HD]
        K_h0 = K_full[0, :, base:base+HD]
        scores = (Q_h0.unsqueeze(0) * K_h0).sum(-1) / (HD ** 0.5)
        # Apply causal mask: positions > pos get -inf (well, by mask of zeros for valid, -inf for invalid)
        # AutoregressiveAttention has causal mask by default
        # Just print top-5 attended positions
        # Apply ALiBi if present
        if hasattr(attn3, 'alibi_slopes') and attn3.alibi_slopes is not None:
            slope = attn3.alibi_slopes[head_idx].item()
            distances = torch.arange(scores.shape[0], device=scores.device) - pos
            scores = scores - slope * distances.abs().float()
        # Mask future positions
        mask = torch.arange(scores.shape[0], device=scores.device) > pos
        scores[mask] = float('-inf')
        attn_w = torch.softmax(scores, dim=-1)
        # Print top-5 attended positions
        top5 = attn_w.topk(5)
        print(f"  L3 attn head 0 (PC carry-fwd) at pos {pos} - top 5 attended positions:", flush=True)
        for k in range(5):
            p = top5.indices[k].item()
            w = top5.values[k].item()
            tok = context[p] if p < len(context) else "?"
            elo_p = x_pre_l3[0, p, EMBED_LO_DIM:EMBED_LO_DIM+16].argmax().item()
            elo_v = x_pre_l3[0, p, EMBED_LO_DIM:EMBED_LO_DIM+16].max().item()
            l1h1_p = x_pre_l3[0, p, L1H1_PC].item()
            l1h0_p = x_pre_l3[0, p, L1H0_PC].item()
            print(f"    pos={p} tok={tok} weight={w:.3f} EMBED_LO[{elo_p}]={elo_v:.2f} L1H1[PC]={l1h1_p:.2f} L1H0[PC]={l1h0_p:.2f}", flush=True)

        ffn3 = model.blocks[3].ffn
        W_up = ffn3.W_up
        if W_up.is_sparse:
            W_up = W_up.to_dense()
        W_gate = ffn3.W_gate
        if W_gate.is_sparse:
            W_gate = W_gate.to_dense()
        W_down = ffn3.W_down
        if W_down.is_sparse:
            W_down = W_down.to_dense()
        b_up = ffn3.b_up
        b_gate = ffn3.b_gate

        x_pos = x_l3_attn[0, pos]
        up = (W_up @ x_pos) + b_up
        gate = (W_gate @ x_pos) + b_gate
        silu_up = torch.nn.functional.silu(up)
        hidden = silu_up * gate

        # For each unit, print contribution to OUTPUT_LO[k] and OUTPUT_HI[k]
        H = W_up.shape[0]
        out_lo_W = W_down[OUT_LO:OUT_LO + 16]
        out_hi_W = W_down[OUT_HI:OUT_HI + 16]
        out_lo_contrib = out_lo_W * hidden.unsqueeze(0)
        out_hi_contrib = out_hi_W * hidden.unsqueeze(0)
        max_unit_contrib = torch.maximum(
            out_lo_contrib.abs().max(dim=0).values,
            out_hi_contrib.abs().max(dim=0).values,
        )
        topk = max_unit_contrib.topk(min(20, H))
        print(f"\n  Top L3 units by |OUTPUT contribution| at PC marker (HAS_SE={x_l3_attn[0, pos, HAS_SE].item():.2f}):", flush=True)
        print(f"  {'unit':>6} {'hidden':>10} {'up':>10} {'gate':>10}   target OUT dim/value", flush=True)
        for rank in range(topk.values.shape[0]):
            u = topk.indices[rank].item()
            val = topk.values[rank].item()
            if val < 1e-3:
                continue
            lo_argmax = out_lo_contrib[:, u].abs().argmax().item()
            lo_val = out_lo_contrib[lo_argmax, u].item()
            hi_argmax = out_hi_contrib[:, u].abs().argmax().item()
            hi_val = out_hi_contrib[hi_argmax, u].item()
            # Show top 3 input weights for this unit (to identify what triggered it)
            row = W_up[u]
            top_in = row.abs().topk(3)
            in_str = ",".join([f"d{int(top_in.indices[k])}:{row[top_in.indices[k]].item():+.0f}" for k in range(3)])
            row_g = W_gate[u]
            top_g = row_g.abs().topk(2)
            g_str = ",".join([f"d{int(top_g.indices[k])}:{row_g[top_g.indices[k]].item():+.1f}" for k in range(2)])
            print(f"  {u:>6d} {hidden[u].item():>+10.3f} {up[u].item():>+10.3f} {gate[u].item():>+10.3f}   "
                  f"OUT_LO[{lo_argmax}]={lo_val:+.2f} OUT_HI[{hi_argmax}]={hi_val:+.2f}", flush=True)
            print(f"          W_up: {in_str}  W_gate: {g_str}", flush=True)

    # === Focused L6 FFN unit-level trace ===
    # Re-run up to end of L5 then inspect L6 FFN unit activations at PC marker.
    print("\n=== L6 FFN unit-level trace at step-1 PC marker ===\n", flush=True)
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for i in range(6):
            x = model.blocks[i](x)
        # x is now post-L5. Run L6 attention.
        x_l6_attn = model.blocks[6].attn(x)
        dump_at("post-L5+L6-attn", x_l6_attn)

        # Now compute L6 FFN hidden activations at PC marker
        ffn = model.blocks[6].ffn
        # PureFFN: hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
        # Use the same forward formula
        # Some FFN may be compacted/MoE. Try to access W_up/b_up via the
        # standard interface, but be robust.
        W_up = ffn.W_up
        b_up = ffn.b_up
        W_gate = ffn.W_gate
        b_gate = ffn.b_gate
        W_down = ffn.W_down

        if W_up.is_sparse:
            W_up = W_up.to_dense()
        if W_gate.is_sparse:
            W_gate = W_gate.to_dense()
        if W_down.is_sparse:
            W_down = W_down.to_dense()

        # Check if MoE-compacted: we need the *active* expert weights.
        moe_combined = getattr(ffn, "_moe_combined", None)
        if moe_combined is not None:
            print("  (NOTE: L6 FFN is MoE-compacted; using current active expert weights)")
        if getattr(ffn, "_compact_size", None) is not None:
            print(f"  (NOTE: L6 FFN compacted to {ffn._compact_size} hidden units)")

        x_pos = x_l6_attn[0, pos]  # [D]
        up = (W_up @ x_pos) + b_up
        gate = (W_gate @ x_pos) + b_gate
        silu_up = torch.nn.functional.silu(up)
        hidden = silu_up * gate
        # Contribution to OUTPUT_LO/HI per unit:
        # delta[OUTPUT_LO+k] = sum_unit W_down[OUTPUT_LO+k, unit] * hidden[unit]
        # The corrupting units are those with non-zero hidden activation AND
        # non-zero W_down[OUTPUT_LO..OUTPUT_HI+16, unit].
        H = W_up.shape[0]
        # Sum |contribution| over OUTPUT_LO/HI per unit
        out_lo_W = W_down[OUT_LO:OUT_LO + 16]      # [16, H]
        out_hi_W = W_down[OUT_HI:OUT_HI + 16]      # [16, H]
        out_lo_contrib = out_lo_W * hidden.unsqueeze(0)  # [16, H]
        out_hi_contrib = out_hi_W * hidden.unsqueeze(0)
        # For each unit, max |contribution| to OUTPUT
        max_unit_contrib = torch.maximum(
            out_lo_contrib.abs().max(dim=0).values,
            out_hi_contrib.abs().max(dim=0).values,
        )  # [H]

        # Top-30 units by max OUTPUT contribution
        topk = max_unit_contrib.topk(min(30, H))
        print(f"\n  Top units by |OUTPUT contribution| at PC marker:", flush=True)
        print(f"  {'unit':>6} {'hidden':>10} {'up':>10} {'gate':>10} {'maxOUT':>10}   target OUT dim/value", flush=True)
        for rank in range(topk.values.shape[0]):
            u = topk.indices[rank].item()
            val = topk.values[rank].item()
            if val < 1e-3:
                continue
            # Find which OUTPUT_LO[k] or OUTPUT_HI[k] this unit writes to most
            lo_argmax = out_lo_contrib[:, u].abs().argmax().item()
            lo_val = out_lo_contrib[lo_argmax, u].item()
            hi_argmax = out_hi_contrib[:, u].abs().argmax().item()
            hi_val = out_hi_contrib[hi_argmax, u].item()
            tgt = (f"OUT_LO[{lo_argmax}]={lo_val:+.2f}" if abs(lo_val) > abs(hi_val)
                   else f"OUT_HI[{hi_argmax}]={hi_val:+.2f}")
            print(f"  {u:>6d} {hidden[u].item():>+10.3f} {up[u].item():>+10.3f} {gate[u].item():>+10.3f} {val:>+10.3f}   {tgt}", flush=True)

        # Final delta from L6 FFN at PC marker
        delta_final = W_down @ hidden + ffn.b_down  # [D]
        dlo_max_idx = delta_final[OUT_LO:OUT_LO+16].abs().argmax().item()
        dhi_max_idx = delta_final[OUT_HI:OUT_HI+16].abs().argmax().item()
        print(f"\n  L6 FFN delta to OUTPUT_LO at PC marker: argmax={dlo_max_idx} val={delta_final[OUT_LO+dlo_max_idx].item():+.3f}", flush=True)
        print(f"  L6 FFN delta to OUTPUT_HI at PC marker: argmax={dhi_max_idx} val={delta_final[OUT_HI+dhi_max_idx].item():+.3f}", flush=True)

    # Also dump emitted step-1 PC bytes (positions 1..4 of step 1)
    print("\n=== Emitted step-1 register values ===", flush=True)
    step1_start = prefix_len + 35
    pc_bytes = context[step1_start + 1: step1_start + 5]
    ax_bytes = context[step1_start + 6: step1_start + 10]
    print(f"  step 1 emitted PC bytes: {pc_bytes}", flush=True)
    print(f"  step 1 emitted AX bytes: {ax_bytes}", flush=True)
    pc_val = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
    print(f"  step 1 PC value: 0x{pc_val:x} (expected: 8 = next instr after BZ)", flush=True)


if __name__ == "__main__":
    main()
