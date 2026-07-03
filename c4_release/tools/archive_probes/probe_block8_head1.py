#!/usr/bin/env python3
"""Block-8 head-1 (BP/SP operand gather) score probe, spec_k=0.

Head 1 should fire ONLY for LEA/ADJ/ENT (gather BP/SP OUTPUT -> ALU). It
leaks for binary/comparison ops. Probe its raw QK score + softmax1 weight
on the BP/SP marker rows for an EQ program (should be ~0) vs a LEA program
(should be ~1). Reports max attention weight + which rows.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGS = {
    # EQ: head 1 should NOT fire (no BP/SP gather).
    "eq_true_42": _mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42),
                       Opcode.EQ, Opcode.EXIT]),
    # LEA: head 1 SHOULD fire (gather SP OUTPUT -> ALU).
    "lea_basic": _mk([(Opcode.ENT, 0), (Opcode.IMM, 7), Opcode.PSH,
                      (Opcode.LEA, 2), Opcode.EXIT]),
    # ADJ: head 1 SHOULD fire.
    "adj_sp": _mk([Opcode.ENT | (0 << 8), (Opcode.IMM, 5), Opcode.PSH,
                   Opcode.ADJ | (8 << 8), Opcode.EXIT]),
}


def main(block=8, head=1):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    alu_lo = dp["ALU_LO"]
    alu_hi = dp["ALU_HI"]

    for pname, bc in PROGS.items():
        try:
            ctx = probe._final_context(bc, max_steps=20)
        except Exception as e:
            print(f"=== {pname}: SKIP ({e}) ===")
            continue
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        q_row = ax_rows[-1]
        with torch.no_grad():
            resid = model.forward(toks, stop_after_block=block - 1)[0]
        attn = model.blocks[block].attn
        x = resid.unsqueeze(0)
        H = attn.num_heads
        HD = attn.head_dim
        with torch.no_grad():
            Q = F.linear(x, attn.W_q).view(1, S, H, HD).transpose(1, 2)
            K = F.linear(x, attn.W_k).view(1, S, H, HD).transpose(1, 2)
            V = F.linear(x, attn.W_v).view(1, S, H, HD).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
            slopes = attn.alibi_slopes
            pos = torch.arange(S, device=dev).float()
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
            scores = scores - slopes.view(1, H, 1, 1) * dist.view(1, 1, S, S)
            causal = torch.triu(torch.full((S, S), float("-inf"), device=dev),
                                diagonal=1)
            scores = scores + causal.view(1, 1, S, S)
            sink = torch.zeros(1, H, S, 1, device=dev)
            ext = torch.cat([scores, sink], dim=-1)
            w = F.softmax(ext, dim=-1)[..., :S]
            out = torch.matmul(w, V)
            head_out = torch.zeros(1, S, H * HD, device=dev)
            head_out[0, q_row, head * HD:(head + 1) * HD] = out[0, head, q_row]
            contrib = F.linear(head_out, attn.W_o)[0, q_row]
        wh = w[0, head, q_row]
        top = torch.topk(wh, k=4)
        lo = [round(float(contrib[alu_lo + k]), 2) for k in range(16)]
        hi = [round(float(contrib[alu_hi + k]), 2) for k in range(16)]
        lo_nz = [f"{v}@{k}" for k, v in enumerate(lo) if abs(v) > 0.3]
        hi_nz = [f"{v}@{k}" for k, v in enumerate(hi) if abs(v) > 0.3]
        print(f"=== {pname} q_row={q_row} sum_w={float(wh.sum()):.3f} "
              f"head={head} ===")
        print(f"  head{head} ALU_LO={lo_nz} ALU_HI={hi_nz}")
        for wv, r in zip(top.values.tolist(), top.indices.tolist()):
            print(f"    w={wv:.3f} row={r} tok={ctx[r]}")


if __name__ == "__main__":
    main()
