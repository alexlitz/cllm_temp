#!/usr/bin/env python3
"""Inc-3 MAP — nested_quad LI-reload head: which ROW does it attend, byte-0?

nested_quad step3 LI loads arg x=10 from the frame; got_ax=0. This probes the
LI-reload head (layer10_byte_passthrough_bake.head_1, blk16/L11, ALiBi slope 1.0)
post-softmax attention at the step3 AX byte-0 predictor row (off=6) and reports
the winning source row + that row's CLEAN_EMBED value (the byte copied). Compares
GOLDEN vs CAMPAIGN: does campaign attend a WRONG row (deeper frame / PC-marker),
or the right row whose CLEAN_EMBED is 0?

Run TWICE (clear cache between):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=0  python tools/probe_inc3_nested_li_attn.py
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_nested_li_attn.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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

SRC = (
    "int double_it(int x) { return x * 2; }\n"
    "            int quad(int x) { return double_it(double_it(x)); }\n"
    "            int main() { return quad(10); }"
)
DIV_STEP = 3
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
    ctx = p._final_context(bytecode, max_steps=DIV_STEP + 2)
    off = pl + DIV_STEP * STEP + 6  # AX byte-0 predictor row
    padded = torch.tensor([ctx], device=p._device)
    ce_lo, ce_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    mvb0 = dp.get("MEM_VAL_B0")
    memstore = dp["MEM_STORE"]
    markax = dp["MARK_AX"]

    x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    block = p.model.blocks[BLK]
    attn = block.attn
    Wq = td(attn.W_q); Wk = td(attn.W_k)
    nh = attn.num_heads
    hd = Wq.shape[0] // nh
    h = HEAD_IDX
    q_all = x_in @ Wq.T
    k_all = x_in @ Wk.T
    qh = q_all[:, h * hd:(h + 1) * hd]
    kh = k_all[:, h * hd:(h + 1) * hd]
    qrow = qh[off]
    scores = kh @ qrow
    raw = kh @ qrow
    T = x_in.shape[0]
    slope = None
    bias = torch.zeros(T)
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        slope = float(td(attn.alibi_slopes)[h])
        pos = torch.arange(T, dtype=torch.float)
        bias = -slope * (off - pos).clamp(min=0)
        scores = scores + bias
    scores[off + 1:] = -1e30
    probs = torch.softmax(scores, dim=0)
    top = torch.topk(probs, 8)
    print(f"=== {cfg} STEP={STEP} blk{BLK} head{h} off={off} "
          f"(step{DIV_STEP} byte0 predictor) ALiBi_slope={slope} ===")
    for r, pr in zip(top.indices.tolist(), top.values.tolist()):
        if pr < 0.01:
            continue
        step = (r - pl) // STEP if r >= pl else -1
        roff = (r - pl) % STEP if r >= pl else -1
        tag = _step_offset_field(roff) if roff >= 0 else "prompt"
        rrow = x_in[r]
        ceval, celo, cehi = nib(rrow, ce_lo, ce_hi)
        ms = float(rrow[memstore])
        mvb = float(rrow[mvb0]) if mvb0 is not None else 0.0
        ma = float(rrow[markax])
        print(f"  row{r:4d} p={pr:.3f} step{step} off{roff}({tag:9s}) "
              f"CLEAN_EMBED={ceval}(lo{celo:.1f}/hi{cehi:.1f}) "
              f"MEM_STORE={ms:+.1f} MEM_VAL_B0={mvb:+.1f} MARK_AX={ma:+.1f}")


if __name__ == "__main__":
    main()
