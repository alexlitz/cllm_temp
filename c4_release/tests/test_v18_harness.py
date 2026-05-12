"""V18 per-layer hook harness: trace IO_IS_PRTF / CMP[5] / NEXT_THINKING_END
data flow for ``printf("Hi");`` with conversational_io=True.

Goal: identify which step in the L5 -> L6 -> logits chain breaks before
``Token.THINKING_END`` should be emitted.

The L6 FFN state-machine bake (units 1400-1401) fires
``NEXT_THINKING_END`` when CMP[5] AND NEXT_SE. So we walk the chain:

  step (PRTF), AX marker:        IO_IS_PRTF on residual after L5 FFN?
  step (PRTF), MARK_SE position: CMP[5] on residual after L6 attn?
                                  NEXT_THINKING_END on residual after L6 FFN?
  step (PRTF), MARK_SE position: L15 lm_head logits -> which token wins?

This file lives in tests/ but begins with ``_`` so pytest's default
collector ignores it. Run directly:

  python tests/_v18_harness.py
"""
from __future__ import annotations

import sys
import os

import torch

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from src.compiler import compile_c


def test_v18_per_layer_hook_harness():
    """Pytest entry: runs the harness; use ``-s`` to see prints.

    This test is intentionally non-asserting — it's a debugging harness,
    not a regression gate. It always passes (provided the model builds).
    """
    main()


def test_baseline_no_convo_io():
    """Baseline: same printf("Hi") program with conversational_io=False.
    Used as a control to see if convo-IO enable is what's causing
    PC corruption / divergence early in the run.
    """
    main(convo_io=False)


def test_synthetic_prtf_step():
    """Construct a synthetic context whose last step *is* the PRTF step:
    bytecode = [PRTF, EXIT], and a hand-crafted REG_PC=0 REG_AX=0 REG_SP=0
    REG_BP=0 STACK0=<fmt_ptr> MEM=zero STEP_END placeholder. We then
    inspect the residual at MARK_AX (PRTF position), MARK_SE (last
    position before STEP_END), and the L15 lm_head logits for the token
    that would replace STEP_END.

    This bypasses the broken main run loop and isolates the V18 chain.
    """
    synthetic_prtf_harness()


def main(convo_io=True):
    c_code = '''
int main() {
    printf("Hi");
    return 0;
}
'''
    code, data = compile_c(c_code)

    # IMPORTANT: use the standard (non-pure_neural) runner: it executes
    # steps via Python dispatch up to (but not including) the PRTF step.
    # The neural pipeline still runs on every forward pass — we just rely
    # on the Python dispatcher to keep PC/SP correct so the PRTF step is
    # actually reached.
    print(f"\n===== HARNESS: conversational_io={convo_io} =====")
    runner = AutoregressiveVMRunner(conversational_io=convo_io)
    model = runner.model
    device = next(model.parameters()).device

    # Resolve dim positions (compiler-allocated). Fall back to _SetDim.
    dp = model.embed._dim_positions
    if dp is None:
        from neural_vm.vm_step import _SetDim as BD
        def D(name):
            return getattr(BD, name)
    else:
        def D(name):
            return dp[name]

    DIM_MARK_AX = D("MARK_AX")
    DIM_MARK_SE = D("MARK_SE")
    DIM_IO_IS_PRTF = D("IO_IS_PRTF")
    DIM_CMP = D("CMP")
    DIM_NEXT_SE = D("NEXT_SE")
    DIM_NEXT_THINKING_END = D("NEXT_THINKING_END")
    DIM_ACTIVE_OPCODE_PRTF = D("ACTIVE_OPCODE_PRTF")
    DIM_IO_STATE = D("IO_STATE")

    print(f"[DIMS] MARK_AX={DIM_MARK_AX} MARK_SE={DIM_MARK_SE} "
          f"IO_IS_PRTF={DIM_IO_IS_PRTF} CMP+5={DIM_CMP+5} "
          f"NEXT_SE={DIM_NEXT_SE} NEXT_THINKING_END={DIM_NEXT_THINKING_END} "
          f"ACTIVE_OPCODE_PRTF={DIM_ACTIVE_OPCODE_PRTF} "
          f"IO_STATE={DIM_IO_STATE}")
    print(f"[Token IDs] THINKING_END={Token.THINKING_END} "
          f"STEP_END={Token.STEP_END} REG_PC={Token.REG_PC} "
          f"REG_AX={Token.REG_AX}")
    print("[Context size] (built by runner.run internally)")

    # Compute the PRTF instruction index in the bytecode.
    from neural_vm.embedding import Opcode
    prtf_instr_idx = None
    for i, ins in enumerate(code):
        if (ins & 0xFF) == Opcode.PRTF:
            prtf_instr_idx = i
            break
    print(f"[Bytecode] PRTF at instruction index {prtf_instr_idx}")

    # Capture residual outputs from blocks + embedding.
    captures = []

    def make_block_hook(name):
        def hook(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return hook

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    handles = []
    handles.append(model.embed.register_forward_hook(embed_hook))
    for li in [3, 4, 5, 6, 7, 8, 9, 10, 14, 15]:
        if li < len(model.blocks):
            handles.append(model.blocks[li].register_forward_hook(
                make_block_hook(f"after_L{li}")
            ))

    # Hook model.forward so every forward pass during runner.run() is
    # logged. The runner uses Python dispatch to keep PC/SP correct,
    # so PRTF will actually be reached. We log when the residual hits
    # a step whose REG_PC decodes to an instruction whose opcode is PRTF.
    from neural_vm.constants import INSTR_WIDTH

    diagnostics = {"prtf_seen": False, "thinking_end_logits_high": False,
                   "ax_seen": False, "last_summary": None}

    orig_forward = model.forward

    def hooked_forward(token_ids, *args, **kwargs):
        captures.clear()
        out = orig_forward(token_ids, *args, **kwargs)
        cap = captures[-1] if captures else None
        if cap is None:
            return out

        tids = cap["token_ids"][0].tolist()
        last_pos = cap["token_ids"].shape[1] - 1

        # Find current step's REG_PC and decode the opcode.
        pc_marker_pos = None
        for i in range(last_pos, -1, -1):
            if tids[i] == Token.REG_PC:
                pc_marker_pos = i
                break
        pc_val = None
        if pc_marker_pos is not None and pc_marker_pos + 4 < len(tids):
            pc_val = 0
            for j in range(4):
                pc_val |= (tids[pc_marker_pos + 1 + j] & 0xFF) << (j * 8)
        cur_opcode = None
        if pc_val is not None:
            idx = pc_val // INSTR_WIDTH
            if 0 <= idx < len(code):
                cur_opcode = code[idx] & 0xFF
        is_prtf_step = (cur_opcode == Opcode.PRTF)
        # Log a brief summary every 5 forward passes (or every PRTF step).
        diagnostics["forward_count"] = diagnostics.get("forward_count", 0) + 1
        fc = diagnostics["forward_count"]
        if not is_prtf_step:
            if fc < 30 or fc % 50 == 0:
                # Brief summary
                tail = tids[-1] if tids else -1
                logits_last = out[0, -1, :].detach()
                nxt = int(logits_last.argmax().item())
                topv, topi = torch.topk(logits_last, 4)
                top = [(int(t), round(float(v), 2)) for v, t in zip(topv.tolist(), topi.tolist())]
                print(f"[fwd #{fc:3d} ctx={len(tids):4d} pc=0x{(pc_val or 0):x} op={cur_opcode}] tail={tail} -> next={nxt} top={top}")
            return out

        # We're now generating tokens of the PRTF step. Find the AX
        # marker position in this step and inspect the residual.
        ax_pos = None
        for i in range(pc_marker_pos, last_pos + 1):
            if tids[i] == Token.REG_AX:
                ax_pos = i
                break

        def dim_at(name, pos, dim, slot=0):
            if pos is None or name not in cap:
                return None
            return float(cap[name][0, pos, dim + slot].item())

        ax_active_l4 = dim_at("after_L4", ax_pos, DIM_ACTIVE_OPCODE_PRTF)
        ax_active_l5 = dim_at("after_L5", ax_pos, DIM_ACTIVE_OPCODE_PRTF)
        ax_mark_l5 = dim_at("after_L5", ax_pos, DIM_MARK_AX)
        ax_io_l5 = dim_at("after_L5", ax_pos, DIM_IO_IS_PRTF)
        ax_io_l6 = dim_at("after_L6", ax_pos, DIM_IO_IS_PRTF)
        ax_cmp5_l6 = dim_at("after_L6", ax_pos, DIM_CMP, 5)

        cmp5_l6 = dim_at("after_L6", last_pos, DIM_CMP, 5)
        next_se_l6 = dim_at("after_L6", last_pos, DIM_NEXT_SE)
        next_te_l6 = dim_at("after_L6", last_pos, DIM_NEXT_THINKING_END)
        next_te_l15 = dim_at("after_L15", last_pos, DIM_NEXT_THINKING_END)
        mark_se_l5 = dim_at("after_L5", last_pos, DIM_MARK_SE)
        io_state_l6 = dim_at("after_L6", last_pos, DIM_IO_STATE)
        io_is_prtf_last_l5 = dim_at("after_L5", last_pos, DIM_IO_IS_PRTF)

        # Last position's logits
        logits_last = out[0, -1, :].detach()
        topv, topi = torch.topk(logits_last, 8)
        top = [(int(t), round(float(v), 2)) for v, t in zip(topv.tolist(), topi.tolist())]
        l_step_end = float(logits_last[Token.STEP_END].item())
        l_te = float(logits_last[Token.THINKING_END].item())

        tail_token = tids[-1] if tids else None
        tail_name = {
            Token.STEP_END: "STEP_END",
            Token.THINKING_END: "THINKING_END",
            Token.THINKING_START: "THINKING_START",
            Token.REG_PC: "REG_PC", Token.REG_AX: "REG_AX",
            Token.REG_SP: "REG_SP", Token.REG_BP: "REG_BP",
            Token.STACK0: "STACK0", Token.MEM: "MEM",
        }.get(tail_token, f"tok{tail_token}")

        msg = (
            f"\n[PRTF step ctx={len(tids)} pc=0x{pc_val:x} tail={tail_name}({tail_token}) "
            f"ax_pos={ax_pos} last_pos={last_pos}]\n"
            f"   AX@{ax_pos}: L4.ACTIVE_OPCODE_PRTF={ax_active_l4} "
            f"L5.ACTIVE_OPCODE_PRTF={ax_active_l5} L5.MARK_AX={ax_mark_l5} "
            f"L5.IO_IS_PRTF={ax_io_l5} L6.IO_IS_PRTF={ax_io_l6} "
            f"L6.CMP[5]={ax_cmp5_l6}\n"
            f"   last@{last_pos}: L5.MARK_SE={mark_se_l5} L5.IO_IS_PRTF={io_is_prtf_last_l5} "
            f"L6.CMP[5]={cmp5_l6} L6.NEXT_SE={next_se_l6} "
            f"L6.NEXT_THINKING_END={next_te_l6} L6.IO_STATE={io_state_l6} "
            f"L15.NEXT_THINKING_END={next_te_l15}\n"
            f"   logits[STEP_END]={l_step_end:.2f} logits[THINKING_END]={l_te:.2f}\n"
            f"   logits top: {top}"
        )
        print(msg)
        diagnostics["prtf_seen"] = True
        diagnostics["last_summary"] = msg
        if ax_pos is not None and ax_pos == last_pos:
            diagnostics["ax_seen"] = True
        return out

    model.forward = hooked_forward
    try:
        output, exit_code = runner.run(code, data, [], max_steps=300)
    finally:
        model.forward = orig_forward
        for h in handles:
            h.remove()

    print("\n========== SUMMARY ==========")
    print(f"Output: {output!r}")
    print(f"Exit code: {exit_code}")
    print(f"PRTF step observed: {diagnostics['prtf_seen']}")
    print(f"AX marker step observed: {diagnostics['ax_seen']}")


def synthetic_prtf_harness():
    """Hand-craft a context with PRTF as the current step and dump residuals."""
    from neural_vm.embedding import Opcode
    from neural_vm.constants import INSTR_WIDTH

    print("\n===== SYNTHETIC PRTF STEP HARNESS =====")
    runner = AutoregressiveVMRunner(conversational_io=True)
    model = runner.model
    device = next(model.parameters()).device

    # Resolve dims
    dp = model.embed._dim_positions
    if dp is None:
        from neural_vm.vm_step import _SetDim as BD
        def D(name, default=None):
            return getattr(BD, name, default)
    else:
        def D(name, default=None):
            return dp.get(name, default)
    DIMS_ALL = [
        "MARK_AX", "MARK_SE", "MARK_PC", "MARK_STACK0",
        "IO_IS_PRTF", "IO_IS_READ", "IO_STATE",
        "CMP", "NEXT_SE", "NEXT_THINKING_END",
        "NEXT_THINKING_START", "IO_IN_OUTPUT_MODE",
        "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
        "OP_PRTF", "OP_EXIT",
        "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
        "LAST_WAS_THINKING_END", "LAST_WAS_THINKING_START",
    ]
    DIMS = {}
    for n in DIMS_ALL:
        v = D(n)
        if v is not None:
            DIMS[n] = v
            print(f"  DIM {n}={v}")
        else:
            print(f"  DIM {n}=<missing>")

    # Bytecode: just [PRTF, EXIT]. PRTF=33, EXIT=38. Format string ptr is
    # whatever STACK0 contains.
    bc = [Opcode.PRTF, Opcode.EXIT]
    # Encode as [op0, imm_b0..b3, 0,0,0, op1, imm_b0..b3, 0,0,0]
    code = []
    for op in bc:
        code.append(op & 0xFF)
        for _ in range(4):
            code.append(0)
        for _ in range(3):
            code.append(0)

    # Build context: CODE section + DATA section + a synthetic PRTF step.
    # PC=0 (so we're executing PRTF at instruction 0).
    ctx = [Token.CODE_START]
    ctx.extend(code)
    ctx.append(Token.CODE_END)
    ctx.append(Token.DATA_START)
    ctx.append(Token.DATA_END)

    # One synthetic step: REG_PC=0 (PRTF), REG_AX=0, REG_SP=0xFFC, REG_BP=0,
    # STACK0=0x10 (some heap ptr), MEM=zeros, STEP_END.
    def append_step(pc=0, ax=0, sp=0xFFC, bp=0, stack0=0x10, mem_addr=0, mem_val=0):
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
        # MEM is 9 tokens total (marker + 8 bytes); no extra padding.
        ctx.append(Token.STEP_END)

    # Build a few prior steps to give L2 lookback meaningful values
    # (LAST_WAS_BYTE etc.). Then the PRTF step itself.
    append_step(pc=8, ax=10, sp=0xFFC, bp=0, stack0=0x100)  # IMM 10 (push fmt ptr)
    append_step(pc=16, ax=10, sp=0xFF4, bp=0, stack0=10)    # PSH 10
    # The PRTF step (current). Token sequence will end at STEP_END but we
    # want to inspect what would have been emitted INSTEAD of STEP_END.
    pc_prtf = 0
    append_step(pc=pc_prtf, ax=0, sp=0xFF4, bp=0, stack0=10)

    print(f"Context built: {len(ctx)} tokens")

    # Mark position of last step's tokens
    last_step_end = len(ctx) - 1  # STEP_END just appended
    last_step_start = last_step_end - 34  # 35 tokens per step
    last_step_ax = last_step_start + 5  # REG_PC[0..4], REG_AX at offset 5
    last_step_se = last_step_end  # SE token

    # Now: we want to inspect the residual at MARK_AX (last_step_ax) and
    # at the position right before STEP_END (i.e. last token of MEM section).
    # The "NEXT_SE" position is the token whose forward emits STEP_END;
    # in autoregressive mode that's the last input token. The L6 FFN state
    # machine fires NEXT_THINKING_END at the position whose NEXT_SE=1 — i.e.
    # the position one before STEP_END. So we run forward on
    # ctx[:last_step_end] (everything up to but not including STEP_END)
    # and the last position is the residual we care about.
    truncated = ctx[:last_step_end]
    print(f"Forwarding {len(truncated)} tokens; last position = {len(truncated) - 1} (pre-STEP_END)")
    print(f"PRTF AX marker position: {last_step_ax}")

    captures = {}

    def make_block_hook(name):
        def hook(module, inputs, output):
            captures[name] = output.detach().clone()
        return hook

    handles = []
    handles.append(model.embed.register_forward_hook(
        lambda m, i, o: captures.__setitem__("embed", o.detach().clone())
    ))
    for li in range(len(model.blocks)):
        handles.append(model.blocks[li].register_forward_hook(
            make_block_hook(f"after_L{li}")
        ))

    try:
        with torch.no_grad():
            tok_ids = torch.tensor([truncated], dtype=torch.long, device=device)
            logits = model.forward(tok_ids)
    finally:
        for h in handles:
            h.remove()

    last_pos = len(truncated) - 1

    def vals_at(pos, name=None):
        rows = {}
        for layer in [3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17]:
            key = f"after_L{layer}"
            if key in captures:
                arr = captures[key][0, pos]
                rows[layer] = arr
        return rows

    # Dump the key dims at MARK_AX and NEXT_SE positions
    print(f"\n--- Residual at MARK_AX (pos {last_step_ax}, token={truncated[last_step_ax]}) ---")
    for name, dim in DIMS.items():
        for li in [3, 4, 5, 6, 7, 8, 9, 10]:
            key = f"after_L{li}"
            if key in captures:
                v = float(captures[key][0, last_step_ax, dim].item())
                if abs(v) > 0.01:
                    print(f"  L{li}.{name}[{dim}]={v:.3f}")

    print(f"\n--- Residual at last_pos (pos {last_pos}, token={truncated[last_pos]}) ---")
    for name, dim in DIMS.items():
        for li in [3, 4, 5, 6, 7, 8, 9, 10, 14, 15]:
            key = f"after_L{li}"
            if key in captures:
                v = float(captures[key][0, last_pos, dim].item())
                if abs(v) > 0.01:
                    print(f"  L{li}.{name}[{dim}]={v:.3f}")
                # Also check CMP+5, CMP+6 explicitly
                if name == "CMP":
                    for slot in [5, 6]:
                        vs = float(captures[key][0, last_pos, dim + slot].item())
                        if abs(vs) > 0.01:
                            print(f"  L{li}.CMP[{slot}]={vs:.3f}")

    print(f"\n--- lm_head logits at last_pos (pos {last_pos}) ---")
    logits_last = logits[0, -1, :].detach()
    topv, topi = torch.topk(logits_last, 15)
    top = [(int(t), round(float(v), 2)) for v, t in zip(topv.tolist(), topi.tolist())]
    print(f"  Top-15: {top}")
    print(f"  STEP_END({Token.STEP_END}) logit: {logits_last[Token.STEP_END].item():.2f}")
    print(f"  THINKING_END({Token.THINKING_END}) logit: {logits_last[Token.THINKING_END].item():.2f}")
    print(f"  argmax token: {int(logits_last.argmax().item())}")

    # Also report whether the L5 IO_IS_PRTF signal is even present at AX.
    # If not, the L5 opcode decode is broken.
    if "after_L5" in captures:
        def safe(name, layer, pos):
            if name not in DIMS:
                return None
            key = f"after_L{layer}"
            if key not in captures:
                return None
            return float(captures[key][0, pos, DIMS[name]].item())
        print(f"\n*** L5 PRTF detection summary at AX marker (pos {last_step_ax}) ***")
        print(f"   L4.ACTIVE_OPCODE_PRTF = {safe('ACTIVE_OPCODE_PRTF', 4, last_step_ax)}")
        print(f"   L5.MARK_AX = {safe('MARK_AX', 5, last_step_ax)}")
        print(f"   L5.OP_PRTF = {safe('OP_PRTF', 5, last_step_ax)}")
        print(f"   L5.IO_IS_PRTF = {safe('IO_IS_PRTF', 5, last_step_ax)}")
        print(f"\n*** L6 PRTF relay summary at last_pos (pos {last_pos}) ***")
        print(f"   L5.IO_IS_PRTF at last_pos = {safe('IO_IS_PRTF', 5, last_pos)}")
        if "CMP" in DIMS:
            cmp_base = DIMS["CMP"]
            print(f"   L6.CMP[5] at last_pos = {float(captures['after_L6'][0, last_pos, cmp_base+5].item()):.3f}")
            print(f"   L6.CMP[6] at last_pos = {float(captures['after_L6'][0, last_pos, cmp_base+6].item()):.3f}")
        print(f"   L6.NEXT_SE at last_pos = {safe('NEXT_SE', 6, last_pos)}")
        print(f"   L6.NEXT_THINKING_END at last_pos = {safe('NEXT_THINKING_END', 6, last_pos)}")
        print(f"   L15.NEXT_THINKING_END at last_pos = {safe('NEXT_THINKING_END', 15, last_pos)}")


if __name__ == "__main__":
    main()
