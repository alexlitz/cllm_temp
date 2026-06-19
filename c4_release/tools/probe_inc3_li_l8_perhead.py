#!/usr/bin/env python3
"""Inc-3 — per-head ALU_LO/HI contribution at the LI byte-0 row, L8 (blk11).

nested_quad step3 LI loads arg x=10 from frame. GOLDEN delivers ALU=10 at
blk11/L8; CAMPAIGN gives ALU=170 garbage. This probe decomposes the L8 attn
output into per-head ALU_LO/HI contributions at the step3 AX byte-0 predictor
row (off = base) so we can see WHICH head delivers 10 in golden and WHICH head
writes the 170 in campaign.

FINDING (2026-06-19, GPU spec_k=0): the ALU corruption is L8 head-5
(layer8_mem_to_alu). On the LI/PSH step head 5 is Q-gated OFF, but its dim-0 K
column carries a LATENT cross-op collision (MARK_AX=+30 / OP_IMM=-30 from a
sibling L8 attn op baked into the same head_idx*HD column). At a prior IMM
AX-marker K[0]=-110 and the suppressed query Q[0]=-1985 multiply to +218k — a
neg*neg spurious attractor — pinning head 5 onto the wrong cross-step row and
copying garbage into ALU (0x0A->0xAA). FIX (l8_ops.py, flag-gated): clear head
5 dim-0 K to CONST=10 only -> ALU stays 10 (campaign now matches golden).

BUT this is VERDICT-NEUTRAL: with operand-A correctly in ALU, the LI/binary
value STILL never reaches OUTPUT. The downstream blocker is the L20 (blk34)
OUTPUT-band materializer-silu EXPLOSION (OUTPUT_LO -> 208 -> inf/nan, the
l16_stack0_*_marker_from_alu_* family — the ROOT-A/B silu-firing class) on the
result/load row. THAT is the real verdict lever for expr/var, and it lives on
the l16 materializer surface, NOT the L8 operand-delivery path. The brief's
"operand-A starvation" was the ALU symptom; the verdict root is the OUTPUT
explosion. (probe_inc3_mul_operand_l8 + a full ALU/OUTPUT block-chain confirm.)

Run TWICE (clear ~/.cache between):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=0  python tools/probe_inc3_li_l8_perhead.py
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_li_l8_perhead.py
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
BLK = 11  # L8 attn block


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(vec, lo, hi):
    li = int(torch.argmax(vec[lo:lo + 16]).item())
    hi_ = int(torch.argmax(vec[hi:hi + 16]).item())
    return hi_ * 16 + li, float(vec[lo + li]), float(vec[hi + hi_])


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
    off = pl + DIV_STEP * STEP + 6  # AX byte-0 predictor row (AX field at +6)
    padded = torch.tensor([ctx], device=p._device)
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    ce_lo, ce_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]

    x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    block = p.model.blocks[BLK]
    attn = block.attn
    Wq = td(attn.W_q); Wk = td(attn.W_k); Wv = td(attn.W_v); Wo = td(attn.W_o)
    nh = attn.num_heads
    hd = Wq.shape[0] // nh
    T = x_in.shape[0]
    print(f"=== {cfg} STEP={STEP} blk{BLK}(L8) off={off} step{DIV_STEP} byte0 "
          f"predictor; nh={nh} hd={hd} ===")
    v_pre, lo_pre, hi_pre = nib(torch.tensor(x_in[off]), alu_lo, alu_hi)
    print(f"  ALU before L8: {v_pre} (lo{lo_pre:.1f}/hi{hi_pre:.1f})")
    q_all = x_in @ Wq.T
    k_all = x_in @ Wk.T
    v_all = x_in @ Wv.T
    slopes = None
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        slopes = td(attn.alibi_slopes)
    for h in range(nh):
        qh = q_all[:, h * hd:(h + 1) * hd]
        kh = k_all[:, h * hd:(h + 1) * hd]
        vh = v_all[:, h * hd:(h + 1) * hd]
        qrow = qh[off]
        scores = kh @ qrow
        if slopes is not None:
            slope = float(slopes[h])
            pos = torch.arange(T, dtype=torch.float)
            scores = scores - slope * (off - pos).clamp(min=0)
        scores[off + 1:] = -1e30
        probs = torch.softmax(torch.cat([scores, torch.zeros(1)]), dim=0)
        ctx_v = probs[:-1] @ vh
        o_cols = Wo[:, h * hd:(h + 1) * hd]
        contrib = o_cols @ ctx_v
        cv, clo, chi = nib(contrib, alu_lo, alu_hi)
        win = int(torch.argmax(probs[:-1]).item())
        wp = float(probs[win])
        woff = (win - pl) % STEP if win >= pl else -1
        tag = _step_offset_field(woff) if woff >= 0 else "prompt"
        ceval, _, _ = nib(torch.tensor(x_in[win]), ce_lo, ce_hi)
        flag = "  <-- ALU contrib" if (abs(clo) > 0.5 or abs(chi) > 0.5) else ""
        print(f"  head{h}: ALU_contrib byte={cv}(lo{clo:.2f}/hi{chi:.2f}) "
              f"win=row{win}(off{woff},{tag},p{wp:.2f},CE={ceval}){flag}")


if __name__ == "__main__":
    main()
