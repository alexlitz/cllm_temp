#!/usr/bin/env python3
"""Per-head ALU_LO/HI contribution at the LEA/LI operand row, L8 (blk11) — the
var-multilocal address-leak decomposition (golden config).

var_mul step6 (LEA &a) expects ALU=0xffe0 (=&a, BP-relative slot 0) but golden
delivers 0xffe8 (=&b, slot 1) — the WRONG frame slot. var_update step9 (LI x)
expects ALU=50 (the loaded value) but golden delivers 0x00 (nothing). This probe
decomposes the L8 attn output into per-head ALU contributions at the operand
predictor row so we can see WHICH head writes the address/garbage and WHICH row
it attends.

Run (golden config — the brief's authority; no campaign flags):
  CUDA_VISIBLE_DEVICES=0 python tools/probe_varml_l8_perhead.py
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

CASES = [
    ("var_mul",    "int main() { int a; int b; a = 23; b = 47; return a * b; }", 6),
    ("var_three",  "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }", 6),
    ("var_update", "int main() { int x; x = 50; x = x + 7; return x; }", 9),
]
BLK = 11  # L8 attn block


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(vec, lo, hi):
    li = int(torch.argmax(vec[lo:lo + 16]).item())
    hi_ = int(torch.argmax(vec[hi:hi + 16]).item())
    return hi_ * 16 + li, float(vec[lo + li]), float(vec[hi + hi_])


def main():
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    ce_lo, ce_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    out_lo, out_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    print(f"=== GOLDEN STEP={STEP} blk{BLK}(L8) per-head operand decomposition ===")
    for name, src, div_step in CASES:
        bytecode, _ = compile_c(src)
        pl = len(p._build_context(bytecode))
        ctx = p._final_context(bytecode, max_steps=div_step + 2)
        # operand predictor rows: AX field at +5/+6
        for offfield in (5, 6):
            off = pl + div_step * STEP + offfield
            padded = torch.tensor([ctx], device=p._device)
            x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
            block = p.model.blocks[BLK]
            attn = block.attn
            Wq = td(attn.W_q); Wk = td(attn.W_k); Wv = td(attn.W_v); Wo = td(attn.W_o)
            nh = attn.num_heads
            hd = Wq.shape[0] // nh
            T = x_in.shape[0]
            v_pre, lo_pre, hi_pre = nib(torch.tensor(x_in[off]), alu_lo, alu_hi)
            print(f"\n--- {name} step{div_step} off+{offfield} (row{off}); "
                  f"ALU before L8={v_pre} ---")
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
                if abs(clo) < 0.5 and abs(chi) < 0.5:
                    continue
                win = int(torch.argmax(probs[:-1]).item())
                wp = float(probs[win])
                woff = (win - pl) % STEP if win >= pl else -1
                tag = _step_offset_field(woff) if woff >= 0 else "prompt"
                wstep = (win - pl) // STEP if win >= pl else -1
                # what's in the winning row's OUTPUT band (the value being copied)
                owin, olo, ohi = nib(torch.tensor(x_in[win]), out_lo, out_hi)
                print(f"  head{h}: ALU+=byte={cv}(lo{clo:.1f}/hi{chi:.1f}) "
                      f"win=row{win}(step{wstep},off{woff},{tag},p{wp:.2f}) "
                      f"win.OUTPUT=0x{owin:02x}")


if __name__ == "__main__":
    main()
