#!/usr/bin/env python3
"""Inc-3 ROOT A — which ROW does the LI-reload head (blk16 head_1) attend to?

GPU-grounded (this session): the byte-1 value 0x03 is delivered into OUTPUT_LO at
blk16 (L11) in GOLDEN (OUTPUT_LO slot[3]=2.0) but NOT in CAMPAIGN (stays slot[0]).
The owning op is ``layer10_byte_passthrough_bake.head_1`` (l10_ops.py:2085), the AX
byte passthrough + LI reload head (ALiBi slope 1.0). Its LI-reload side slots
(Q 39-48) select a MEM value-byte row and copy CLEAN_EMBED_LO/HI -> OUTPUT_LO/HI.

This probe instruments head_1's post-softmax attention at the step5 byte-1
predictor row (off=6): which source ROW wins, and what is that row's
CLEAN_EMBED_LO/HI value (the byte it copies). If golden attends a row with
CLEAN_EMBED decoding to 3 and campaign attends a row with CLEAN_EMBED=0 (or a
different/wrong row), the fix is row-selection. If campaign attends the RIGHT
row but its CLEAN_EMBED is 0, the upstream mem-write truncated byte-1.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_li_head_attn.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_li_head_attn.py
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
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"
HEAD_IDX = 1
BLK = 16


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item())
    hi_ = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_ * 16 + li, float(row[lo + li]), float(row[hi + hi_])


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
    ctx = p._final_context(bytecode, max_steps=9)
    off = pl + 5 * STEP + 6
    padded = torch.tensor([ctx], device=p._device)
    ce_lo, ce_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    mvb1 = dp["MEM_VAL_B1"]
    memstore = dp["MEM_STORE"]
    markax = dp["MARK_AX"]

    # Input to block BLK = output of block BLK-1 (the residual the head reads).
    x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])  # [T, D]
    block = p.model.blocks[BLK]
    attn = block.attn
    Wq = td(attn.W_q)
    Wk = td(attn.W_k)
    HD = Wq.shape[0] // attn.num_heads if hasattr(attn, "num_heads") else None
    nh = attn.num_heads
    hd = Wq.shape[0] // nh
    h = HEAD_IDX
    xin_t = x_in
    q_all = xin_t @ Wq.T  # [T, nh*hd]
    k_all = xin_t @ Wk.T
    qh = q_all[:, h * hd:(h + 1) * hd]
    kh = k_all[:, h * hd:(h + 1) * hd]
    qrow = qh[off]  # [hd]
    scores = kh @ qrow  # [T]
    # ALiBi: slope for this head
    T = x_in.shape[0]
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        slope = float(td(attn.alibi_slopes)[h])
        pos = torch.arange(T, dtype=torch.float)
        bias = -slope * (off - pos).clamp(min=0)
        scores = scores + bias
    # causal mask
    scores[off + 1:] = -1e30
    raw = kh @ qrow  # pre-ALiBi
    probs = torch.softmax(scores, dim=0)
    top = torch.topk(probs, 6)
    print(f"=== {cfg} STEP={STEP} blk{BLK} head{h} off={off} "
          f"(step5 byte1 predictor) ALiBi_slope={slope if 'slope' in dir() else 'NA'} ===")
    # Explicitly report the AX[1] (off7) and PC_marker (off0) candidate rows of
    # step4 with their raw K-score, ALiBi bias, and final score.
    for label, roff in (("AX[1]", 7), ("PC_marker", 0), ("AX_marker", 5),
                        ("AX[0]", 6)):
        rr = pl + 4 * STEP + roff
        rawv = float(raw[rr])
        biasv = float(bias[rr]) if 'bias' in dir() else 0.0
        finv = float(scores[rr])
        prv = float(probs[rr])
        print(f"  [cand {label:9s} row{rr}] raw_K={rawv:+8.1f} "
              f"alibi={biasv:+7.1f} final={finv:+8.1f} p={prv:.4f}")
    for r, pr in zip(top.indices.tolist(), top.values.tolist()):
        if pr < 0.01:
            continue
        # what is at row r?
        step = (r - pl) // STEP if r >= pl else -1
        roff = (r - pl) % STEP if r >= pl else -1
        tag = _step_offset_field(roff) if roff >= 0 else "prompt"
        rrow = x_in[r]
        ceval, celo, cehi = nib(rrow, ce_lo, ce_hi)
        ms = float(rrow[memstore])
        mvb = float(rrow[mvb1])
        ma = float(rrow[markax])
        print(f"  row{r:4d} p={pr:.3f} step{step} off{roff}({tag:8s}) "
              f"CLEAN_EMBED={ceval}(lo{celo:.1f}/hi{cehi:.1f}) "
              f"MEM_STORE={ms:+.1f} MEM_VAL_B1={mvb:+.1f} MARK_AX={ma:+.1f}")


if __name__ == "__main__":
    main()
