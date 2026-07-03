#!/usr/bin/env python3
"""Per-head contribution to ALU_LO/HI at the block-8 AX row, spec_k=0.

For each of the 8 heads of physical block 8, computes the head's softmax1
attention from the binop AX query row, applies V then W_o, and reports the
head's contribution to the ALU_LO and ALU_HI bands. Isolates WHICH head
injects the @0 magnitude artifact.
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
    "eq_true_42": _mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42),
                       Opcode.EQ, Opcode.EXIT]),
    "eq_false_20": _mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                        Opcode.EQ, Opcode.EXIT]),
}


def fmt(vec, thr=0.3):
    return "[" + ", ".join(f"{v:.2f}@{i}" for i, v in enumerate(vec)
                           if abs(v) > thr) + "]"


def main(block=8):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    alu_lo = dp["ALU_LO"]
    alu_hi = dp["ALU_HI"]

    for pname, bc in PROGS.items():
        ctx = probe._final_context(bc, max_steps=20)
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
            w = F.softmax(ext, dim=-1)[..., :S]  # [1,H,S,S]
            out = torch.matmul(w, V)  # [1,H,S,HD]
            # head-local out at q_row, then W_o per head slice
            print(f"=== {pname} q_row={q_row} block={block} ===")
            for h in range(H):
                head_out = torch.zeros(1, S, H * HD, device=dev)
                head_out[0, q_row, h * HD:(h + 1) * HD] = out[0, h, q_row]
                contrib = F.linear(head_out, attn.W_o)[0, q_row]  # [D]
                lo = [float(contrib[alu_lo + k]) for k in range(16)]
                hi = [float(contrib[alu_hi + k]) for k in range(16)]
                lo_s = fmt(lo)
                hi_s = fmt(hi)
                if lo_s != "[]" or hi_s != "[]":
                    print(f"  head {h} slope={float(slopes[h]):.3f} "
                          f"ALU_LO={lo_s} ALU_HI={hi_s}")
        print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
