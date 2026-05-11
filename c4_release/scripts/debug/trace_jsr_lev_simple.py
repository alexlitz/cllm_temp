#!/usr/bin/env python3
"""Trace JSR/LEV roundtrip neural execution for test_jsr_then_lev_simple.

Phase 5 instrumentation:
Program: [IMM 7, JSR 3, EXIT, ENT 0, LEV] — expected AX=7

C4 ABI roundtrip:
  step 0: PC=0  IMM 7   -> PC=8,  AX=7,  SP=0x10000, BP=0,        STACK0=0
  step 1: PC=8  JSR 3   -> PC=24, AX=7,  SP=0x0FFF8, BP=0,        STACK0=16  (return addr)
  step 2: PC=24 ENT 0   -> PC=32, AX=7,  SP=0x0FFF0, BP=0x0FFF8,  STACK0=0
  step 3: PC=32 LEV     -> PC=16, AX=7,  SP=0x10000, BP=0,        (STACK0=*)
  step 4: PC=16 EXIT    -> returns AX=7

This script:
  1. Builds a pure_neural runner.
  2. Teacher-forces the first N steps (we drive the model with the oracle
     state) and then asks the model what bytes follow.
  3. Inspects the residual stream at the marker positions in each step to
     identify where the JSR target/return addr write goes wrong, and where
     LEV's PC restore breaks.

Output focus per step:
  - JSR step (step 1): PC marker, SP marker, STACK0 marker. We want to know
    PC_OUTPUT (should encode 24=0x18), SP_OUTPUT (0xFFF8 b0=0xF8), STACK0
    OUTPUT (should encode return addr 16=0x10).
  - LEV step (step 3): PC marker (should encode 16=0x10), SP marker (should
    restore to 0x10000), BP marker (should restore to 0).
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

    # Simple JSR/LEV roundtrip
    prog = [
        (Opcode.IMM, 7),
        (Opcode.JSR, 3),
        Opcode.EXIT,
        (Opcode.ENT, 0),
        Opcode.LEV,
    ]
    bytecode = _make_bc(prog)
    print(f"Bytecode: {[hex(b) for b in bytecode]}", flush=True)

    # Build context
    context = runner._build_context(bytecode, b"", [], b"")
    prefix_len = len(context)
    print(f"Prefix length (CODE+DATA): {prefix_len}", flush=True)

    def append_step(ctx, pc, ax, sp, bp, stack0, mem_addr=0, mem_val=0):
        """Append 35-token oracle step."""
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

    INSTR_WIDTH = 8
    SP0 = 0x10000

    # Teacher-force step 0 result: PC=0 IMM 7 -> PC=8, AX=7
    # The step we emit is the EXECUTED step (showing post-state of step 0).
    # In _build_context the first state is PC=0; the first "emitted" step
    # shows the post-state PC=8, AX=7.
    # But how _build_context works -- it includes a `PC_OFFSET=2` step. Let's
    # check by running step 0 actually.
    # Easiest: append oracle steps showing register states AFTER each step:

    # PC_OFFSET=2: PC points to immediate byte, opcode is at PC-2.
    # So PC=2 means instruction 0 (IMM 7), PC=10 means instruction 1 (JSR 3).
    PC_OFFSET = 2
    pc_of_idx = lambda i: i * INSTR_WIDTH + PC_OFFSET
    # Oracle step 0 (post IMM 7): PC=10 (instr 1), AX=7, SP=0x10000, BP=0, STACK0=0
    step0_start = len(context)
    append_step(context, pc=pc_of_idx(1), ax=7, sp=SP0, bp=0, stack0=0)
    print(f"Step 0 (post IMM 7): PC={pc_of_idx(1)}, AX=7, SP=0x10000 at pos {step0_start}", flush=True)

    # Oracle step 1 INPUT (just PC marker so model emits step 1 bytes)
    step1_pc_marker_pos = len(context)
    context.append(Token.REG_PC)
    print(f"Step 1 PC marker at pos {step1_pc_marker_pos}", flush=True)

    # Ask model to generate full step 1 (35 tokens: 4 PC bytes + AX marker+4 + SP+4 + BP+4 + STACK0+4 + MEM 9 + STEP_END)
    print(f"\nGenerating step 1 (JSR 3 should produce PC=24, STACK0=16)...", flush=True)
    with torch.no_grad():
        for k in range(34):
            tok = model.generate_next(context, use_incremental=False)
            context.append(tok)
        step1_end = len(context)

    # Extract step 1 emitted values
    def extract_quad(ctx, start_after_marker):
        b0 = ctx[start_after_marker] & 0xFF
        b1 = ctx[start_after_marker + 1] & 0xFF
        b2 = ctx[start_after_marker + 2] & 0xFF
        b3 = ctx[start_after_marker + 3] & 0xFF
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), (b0, b1, b2, b3)

    pc1_val, pc1_b = extract_quad(context, step1_pc_marker_pos + 1)
    # AX marker is 5 positions after PC marker (PC marker + 4 bytes)
    ax1_marker = step1_pc_marker_pos + 5
    # Find AX marker actual position via the layout
    # In autoregressive generation, the model also emits the marker tokens
    print(f"  Step 1 emitted tokens [PC marker .. STEP_END]:", flush=True)
    print(f"    Tokens: {context[step1_pc_marker_pos:step1_end]}", flush=True)

    # Find marker tokens
    markers_pos = {}
    for j, tok in enumerate(context[step1_pc_marker_pos:step1_end]):
        if tok in (Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP,
                   Token.STACK0, Token.MEM, Token.STEP_END):
            name = {
                Token.REG_PC: "PC",
                Token.REG_AX: "AX",
                Token.REG_SP: "SP",
                Token.REG_BP: "BP",
                Token.STACK0: "STACK0",
                Token.MEM: "MEM",
                Token.STEP_END: "STEP_END",
            }[tok]
            markers_pos[name] = step1_pc_marker_pos + j

    print(f"  Markers found: {markers_pos}", flush=True)

    # Decode each register if marker present
    for name in ("PC", "AX", "SP", "BP", "STACK0"):
        if name in markers_pos:
            val, bts = extract_quad(context, markers_pos[name] + 1)
            print(f"  Step 1 {name} = 0x{val:08x} (bytes: {bts})", flush=True)

    # Expected after JSR 3:
    # PC = 3*8 + 2 = 26 = 0x1A   (jump target instr 3 = ENT)
    # SP = 0x10000 - 8 = 0xFFF8
    # STACK0 = return_addr = exec_pc + 8 = 10 + 8 = 18 = 0x12 (back to instr 2 = EXIT)
    print(f"  Expected: PC=0x1A (instr 3), SP=0xFFF8 (low 16), BP=0, STACK0=0x12 (return addr to instr 2)", flush=True)

    # Now do a forward pass over the full step-1 context and inspect residual
    # stream at marker positions.
    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    print(f"\n=== Forward pass over step 1 context (shape {token_ids.shape}) ===", flush=True)

    def dump_at(label, x, pos, name=""):
        v = x[0, pos]
        op_jsr = v[BD.OP_JSR].item()
        op_lev = v[BD.OP_LEV].item()
        op_ent = v[BD.OP_ENT].item()
        op_imm = v[BD.OP_IMM].item()
        op_exit = v[BD.OP_EXIT].item()
        mark_pc = v[BD.MARK_PC].item()
        mark_ax = v[BD.MARK_AX].item()
        mark_sp = v[BD.MARK_SP].item()
        mark_bp = v[BD.MARK_BP].item()
        mark_s0 = v[BD.MARK_STACK0].item()
        out_lo_max = v[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
        out_lo_val = v[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
        out_hi_max = v[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
        out_hi_val = v[BD.OUTPUT_HI:BD.OUTPUT_HI+16].max().item()
        fetch_lo_max = v[BD.FETCH_LO:BD.FETCH_LO+16].argmax().item()
        fetch_lo_val = v[BD.FETCH_LO:BD.FETCH_LO+16].max().item()
        fetch_hi_max = v[BD.FETCH_HI:BD.FETCH_HI+16].argmax().item()
        fetch_hi_val = v[BD.FETCH_HI:BD.FETCH_HI+16].max().item()
        ax_carry_lo_max = v[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax().item()
        ax_carry_lo_val = v[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()
        opc_lo_max = v[BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].argmax().item()
        opc_lo_val = v[BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].max().item()
        opc_hi_max = v[BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].argmax().item()
        opc_hi_val = v[BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].max().item()
        print(f"  [{label}] {name} marks(PC,AX,SP,BP,S0)=({mark_pc:.1f},{mark_ax:.1f},{mark_sp:.1f},{mark_bp:.1f},{mark_s0:.1f}) "
              f"OP(JSR,LEV,ENT,IMM,EXIT)=({op_jsr:+.1f},{op_lev:+.1f},{op_ent:+.1f},{op_imm:+.1f},{op_exit:+.1f})", flush=True)
        print(f"        OUT_LO[{out_lo_max}]={out_lo_val:+.2f} OUT_HI[{out_hi_max}]={out_hi_val:+.2f} "
              f"FETCH_LO[{fetch_lo_max}]={fetch_lo_val:+.2f} FETCH_HI[{fetch_hi_max}]={fetch_hi_val:+.2f} "
              f"AX_CARRY_LO[{ax_carry_lo_max}]={ax_carry_lo_val:+.2f} "
              f"OPCODE(LO[{opc_lo_max}]={opc_lo_val:+.2f},HI[{opc_hi_max}]={opc_hi_val:+.2f})", flush=True)

    inspect_positions = []
    for name in ("PC", "AX", "SP", "BP", "STACK0"):
        if name in markers_pos:
            inspect_positions.append((name, markers_pos[name]))

    # Diagnostic: check L1H1[PC_I=0] at step 0 PC byte 0 position
    # Step 0 starts at prefix_len (44), so PC marker at 44, PC byte 0 at 45
    step0_pc_byte0 = prefix_len + 1
    step0_pc_byte1 = prefix_len + 2
    step0_pc_marker = prefix_len
    print(f"\nDIAGNOSTIC: tokens at step 0 PC region")
    print(f"  pos {step0_pc_marker}: token={context[step0_pc_marker]} (REG_PC=257)")
    print(f"  pos {step0_pc_byte0}: token={context[step0_pc_byte0]} (should be PC byte 0 = 10 = 0x0A)")
    print(f"  pos {step0_pc_byte1}: token={context[step0_pc_byte1]} (should be PC byte 1 = 0)")

    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        print("\n  After embed:", flush=True)
        for name, p in inspect_positions:
            dump_at("embed", x, p, name)

        # Diagnostic at L1 input: check L1H1[PC_I=0] at step 0 PC byte 0/1
        # We'll capture L1H1 at the byte 0/1 positions after L1 (where it's set)
        # Run L0, L1 to set L1H1
        x_diag = x
        for j in range(2):
            x_diag = model.blocks[j](x_diag)
        L1H1_PC = BD.L1H1 + 0
        L1H0_PC = BD.L1H0 + 0
        print(f"\nDIAGNOSTIC: After L1, L1H0[PC]/L1H1[PC] at step 0 PC region:")
        for p, label in [(step0_pc_marker, "step0 PC marker"),
                         (step0_pc_byte0, "step0 PC byte 0"),
                         (step0_pc_byte1, "step0 PC byte 1"),
                         (markers_pos.get("PC", -1), "step1 PC marker"),
                         (markers_pos.get("AX", -1), "step1 AX marker")]:
            if p < 0:
                continue
            l1h0 = x_diag[0, p, L1H0_PC].item()
            l1h1 = x_diag[0, p, L1H1_PC].item()
            is_byte = x_diag[0, p, BD.IS_BYTE].item()
            cembed_lo = x_diag[0, p, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+16]
            cembed_lo_argmax = cembed_lo.argmax().item()
            cembed_lo_val = cembed_lo.max().item()
            print(f"  pos={p} ({label}): L1H0[PC]={l1h0:.2f} L1H1[PC]={l1h1:.2f} IS_BYTE={is_byte:.2f} CLEAN_EMBED_LO[{cembed_lo_argmax}]={cembed_lo_val:.2f}")

        # Also check L3 head 0 outputs by manually computing the attention
        # for head 0 (PC carry-forward) at step 1 PC marker
        x_pre_l3 = x
        for j in range(3):
            x_pre_l3 = model.blocks[j](x_pre_l3)
        attn3 = model.blocks[3].attn
        HD = attn3.W_q.shape[0] // attn3.num_heads
        head_idx = 0
        base_h = head_idx * HD
        W_q = attn3.W_q
        W_k = attn3.W_k
        # Q at step 1 PC marker for head 0
        target_pos = markers_pos["PC"]
        Q_full = torch.nn.functional.linear(x_pre_l3, W_q if not W_q.is_sparse else W_q.to_dense())
        K_full = torch.nn.functional.linear(x_pre_l3, W_k if not W_k.is_sparse else W_k.to_dense())
        Q_h0 = Q_full[0, target_pos, base_h:base_h+HD]
        K_h0 = K_full[0, :, base_h:base_h+HD]
        scores = (Q_h0.unsqueeze(0) * K_h0).sum(-1) / (HD ** 0.5)
        if hasattr(attn3, 'alibi_slopes') and attn3.alibi_slopes is not None:
            slope = attn3.alibi_slopes[head_idx].item()
            distances = torch.arange(scores.shape[0], device=scores.device) - target_pos
            scores = scores - slope * distances.abs().float()
        # Causal mask
        mask = torch.arange(scores.shape[0], device=scores.device) > target_pos
        scores[mask] = float('-inf')
        attn_w = torch.softmax(scores, dim=-1)
        top5 = attn_w.topk(min(5, scores.shape[0]))
        print(f"\nDIAGNOSTIC: L3 attn head 0 (PC carry-fwd) at step 1 PC marker (pos {target_pos}, ALiBi slope={slope}):")
        for k in range(top5.indices.shape[0]):
            p = top5.indices[k].item()
            w = top5.values[k].item()
            tok = context[p] if p < len(context) else "?"
            score_val = scores[p].item()
            l1h1_p = x_pre_l3[0, p, L1H1_PC].item()
            l1h0_p = x_pre_l3[0, p, L1H0_PC].item()
            elo = x_pre_l3[0, p, BD.EMBED_LO:BD.EMBED_LO+16]
            elo_argmax = elo.argmax().item()
            elo_val = elo.max().item()
            print(f"  pos={p} tok={tok} w={w:.4f} score={score_val:+.2f} L1H1[PC]={l1h1_p:.2f} L1H0[PC]={l1h0_p:.2f} EMBED_LO[{elo_argmax}]={elo_val:.2f}")

        for i, block in enumerate(model.blocks):
            x_attn = block.attn(x)
            # Diagnostic: per-PC marker dump at every layer with sub-step (attn/ffn) granularity
            v_attn = x_attn[0, markers_pos["PC"]]
            v_attn_in = x[0, markers_pos["PC"]]
            attn_oplev = v_attn[BD.OP_LEV].item()
            attn_opjsr = v_attn[BD.OP_JSR].item()
            attn_opclo8 = v_attn[BD.OPCODE_BYTE_LO + 8].item()
            attn_opchi0 = v_attn[BD.OPCODE_BYTE_HI + 0].item()
            attn_opclo3 = v_attn[BD.OPCODE_BYTE_LO + 3].item()  # JSR lo nibble
            has_se_in = v_attn_in[BD.HAS_SE].item()
            has_se_out = v_attn[BD.HAS_SE].item()
            x_ffn = block.ffn(x_attn)
            v_ffn = x_ffn[0, markers_pos["PC"]]
            ffn_oplev = v_ffn[BD.OP_LEV].item()
            ffn_opjsr = v_ffn[BD.OP_JSR].item()
            ffn_opclo8 = v_ffn[BD.OPCODE_BYTE_LO + 8].item()
            ffn_opchi0 = v_ffn[BD.OPCODE_BYTE_HI + 0].item()
            # Inspect EMBED at PC marker - L3 should set EMBED to PC value
            emb_lo_argmax = v_attn[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
            emb_lo_val = v_attn[BD.EMBED_LO:BD.EMBED_LO+16].max().item()
            emb_hi_argmax = v_attn[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
            emb_hi_val = v_attn[BD.EMBED_HI:BD.EMBED_HI+16].max().item()
            print(f"  PC[L{i:02d}] HAS_SE(in={has_se_in:+.2f}) EMBED(LO[{emb_lo_argmax}]={emb_lo_val:+.2f},HI[{emb_hi_argmax}]={emb_hi_val:+.2f}) attn: OPLEV={attn_oplev:+.2f} OPC_LO[8]={attn_opclo8:+.2f} OPC_LO[3]={attn_opclo3:+.2f} OPC_HI[0]={attn_opchi0:+.2f} | "
                  f"ffn: OPLEV={ffn_oplev:+.2f} OPJSR={ffn_opjsr:+.2f} OPC_LO[8]={ffn_opclo8:+.2f} OPC_HI[0]={ffn_opchi0:+.2f}", flush=True)
            x = x_ffn
            # Only print interesting layers: L5 (fetch), L6 (function call weights), L9 (LEV relays), L16 (LEV routing)
            if i in (3, 5, 6, 7, 8, 9, 14, 15, 16):
                print(f"\n  After L{i}:", flush=True)
                for name, p in inspect_positions:
                    dump_at(f"L{i}", x, p, name)

    # Decode AX value emitted to confirm whether it was clobbered
    # Continue: now let's run autoregressively for steps 2, 3, 4 and check the
    # final AX or whether we reach EXIT.

    print(f"\n=== Continuing autoregressive execution for steps 2..4 ===", flush=True)
    with torch.no_grad():
        # Generate enough tokens for 4 more steps
        for k in range(35 * 4):
            tok = model.generate_next(context, use_incremental=False)
            context.append(tok)

    print(f"Final context length: {len(context)}", flush=True)
    # Walk through emitted steps
    pos = step1_pc_marker_pos
    step_idx = 1
    while pos < len(context) and step_idx < 6:
        # Scan ahead 35 tokens to find marker boundaries
        end = min(pos + 35, len(context))
        seg = context[pos:end]
        marks = {}
        for j, tok in enumerate(seg):
            if tok in (Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP,
                       Token.STACK0, Token.MEM, Token.STEP_END):
                name = {
                    Token.REG_PC: "PC", Token.REG_AX: "AX", Token.REG_SP: "SP",
                    Token.REG_BP: "BP", Token.STACK0: "STACK0", Token.MEM: "MEM",
                    Token.STEP_END: "STEP_END",
                }[tok]
                marks[name] = pos + j
        print(f"\n  Step {step_idx} markers in [{pos},{end}): {marks}", flush=True)
        for name in ("PC", "AX", "SP", "BP", "STACK0"):
            if name in marks:
                try:
                    val, bts = extract_quad(context, marks[name] + 1)
                    print(f"    {name} = 0x{val:08x} bytes={bts}", flush=True)
                except Exception:
                    pass
        # Advance
        if "STEP_END" in marks:
            pos = marks["STEP_END"] + 1
        else:
            pos = end
        step_idx += 1

    # Expected sequence (PC_OFFSET=2):
    print(f"\n=== Expected state sequence (PC_OFFSET=2) ===")
    print(f"  Step 0: PC=0x0A, AX=7,  SP=0x10000, BP=0,         STACK0=0    (after IMM 7,  exec_pc=0x02)")
    print(f"  Step 1: PC=0x1A, AX=7,  SP=0x0FFF8, BP=0,         STACK0=0x12 (after JSR 3,  exec_pc=0x0A; return addr=0x12)")
    print(f"  Step 2: PC=0x22, AX=7,  SP=0x0FFF0, BP=0x0FFF8,   STACK0=0    (after ENT 0,  exec_pc=0x1A)")
    print(f"  Step 3: PC=0x12, AX=7,  SP=0x10000, BP=0,         STACK0=*    (after LEV,    exec_pc=0x22)")
    print(f"  Step 4: HALT, AX=7                                              (after EXIT,   exec_pc=0x12)")


if __name__ == "__main__":
    main()
