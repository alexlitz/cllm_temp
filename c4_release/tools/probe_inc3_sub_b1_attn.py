#!/usr/bin/env python3
"""Inc-3 CLAW-BACK — split blk16 (L11) into ATTN vs FFN for the SUB byte-1 row
and dump WHICH source row the byte-1 ALU_LO nibble comes from.

At blk16 the byte-1 predictor row gains ALU_LO nibble 3 (golden, correct 0x03)
vs nibble 0 (campaign, wrong). This probe:
  1. forwards to blk15 (input to blk16), then runs ONLY blk16.attn on it, then
     blk16.attn+ffn, printing ALU_LO at the byte-1 row at each stage -> says
     whether the nibble is written by the ATTN or the FFN.
  2. if ATTN: re-runs blk16.attn with a hook capturing per-head softmax weights
     and prints, for each head, the top source rows the byte-1 Q row attends to,
     plus the ALU_LO nibble those source rows carry (at the blk15 input).

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_sub_b1_attn.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_sub_b1_attn.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { return 827 - 26; }")
RESULT_STEP = int(os.environ.get("PROBE_STEP", "3"))
BLK = int(os.environ.get("PROBE_BLK", "16"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def topslots(row, base, n=16, k=4):
    vals = [(i, float(row[base + i])) for i in range(n)]
    vals = [(i, v) for i, v in vals if abs(v) > 0.3]
    vals.sort(key=lambda x: -abs(x[1]))
    return " ".join(f"[{i}]={v:.1f}" for i, v in vals[:k]) or "(empty)"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=RESULT_STEP + 2)
    padded = torch.tensor([ctx], device=p._device)
    base = pl + RESULT_STEP * STEP
    axmark = None
    for j in range(STEP):
        if ctx[base + j] == int(Token.REG_AX):
            axmark = j
            break
    b1row = base + axmark + 1
    alu_lo = dp["ALU_LO"]
    print(f"=== {cfg} STEP={STEP} axmark_in={axmark} b1row={b1row} BLK={BLK} src={SRC!r} ===")

    # input to blk BLK = forward stopping after BLK-1
    x_in = p.model.forward(padded, stop_after_block=BLK - 1)[0]  # [S, D]
    blk = p.model.blocks[BLK]

    # ATTN only
    x_attn = blk.attn(x_in.unsqueeze(0))[0]
    # ATTN + FFN
    x_ff = blk.ffn(x_attn.unsqueeze(0))[0]
    for op in blk.post_ops:
        x_ff = op(x_ff.unsqueeze(0))[0]
    print(f"  b1row ALU_LO  in(blk{BLK-1}): {topslots(td(x_in)[b1row], alu_lo)}")
    print(f"  b1row ALU_LO  +attn        : {topslots(td(x_attn)[b1row], alu_lo)}")
    print(f"  b1row ALU_LO  +attn+ffn    : {topslots(td(x_ff)[b1row], alu_lo)}")

    # capture per-head attention weights for the byte-1 Q row
    attn = blk.attn
    H = attn.num_heads
    HD = attn.head_dim
    x = x_in.unsqueeze(0)
    B, S, D = x.shape
    if getattr(attn, "_is_compact", False):
        x_in_c = x[:, :, attn._compact_in_idx]
        n_out = len(attn._compact_out_idx)
        Q = F.linear(x_in_c, attn.W_q).view(B, S, H, n_out // H).transpose(1, 2)
        K = F.linear(x_in_c, attn.W_k).view(B, S, H, n_out // H).transpose(1, 2)
        hd = n_out // H
    else:
        lin = (lambda a, w: torch.sparse.mm(w, a.reshape(-1, a.shape[-1]).t()).t().reshape(a.shape[0], a.shape[1], -1)) if attn.W_q.is_sparse else F.linear
        Q = lin(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
        K = lin(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
        hd = HD
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hd ** 0.5)  # [B,H,S,S]
    # alibi
    slopes = getattr(attn, "alibi_slopes", None)
    dev = x.device
    qpos = torch.arange(S, device=dev)
    kpos = torch.arange(S, device=dev)
    dist = (qpos[:, None] - kpos[None, :]).clamp(min=0).float()
    causal = torch.triu(torch.ones(S, S, device=dev), diagonal=1).bool()
    for h in range(H):
        sc = scores[0, h].clone()
        if slopes is not None:
            sl = float(slopes[h]) if hasattr(slopes, '__len__') else float(slopes)
            sc = sc - sl * dist
        sc = sc.masked_fill(causal, float('-inf'))
        # softmax1 (add a 1 to the denom): approximate with plain softmax for top-row id
        w = torch.softmax(sc, dim=-1)
        wrow = w[b1row]
        top = torch.topk(wrow, 4)
        srcs = [(int(i), float(v)) for i, v in zip(top.indices, top.values) if float(v) > 0.02]
        if not srcs:
            continue
        desc = []
        for si, sv in srcs:
            instep = (si - pl) % STEP if si >= pl else -1
            tok = ctx[si] if si < len(ctx) else -1
            alu = topslots(td(x_in)[si], alu_lo, k=2)
            desc.append(f"row{si}(in={instep},tok={tok},w={sv:.2f},ALU_LO {alu})")
        print(f"  head{h}: " + " | ".join(desc))


if __name__ == "__main__":
    main()
