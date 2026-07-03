#!/usr/bin/env python3
"""Inc-3 cont.: decompose L11 block-16 head 1 (the AX byte-1 carry) at step5 off6.

Golden: head 1 attends step4 AX[1] (pos with prev AX byte-1=3) -> +2.0 to
OUTPUT_HI[0] at the step5 AX[0] predictor row (the LI byte-1 deliverer).
This probe reconstructs head 1's full attention row in BOTH configs so we see
WHERE it points in the 30-tok frame (and why it drops the byte-1).

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_l11h1.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_l11h1.py
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
L11_BLOCK_LOGICAL = 11
HEAD = 1


def todense(w):
    if w.layout != torch.strided:
        w = w.to_dense()
    return w.detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    OUT_HI = dp["OUTPUT_HI_THIS_STEP"]

    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off6 = pl + 5 * STEP + 6
    blk_map = p.block_layer_map()
    b = [i for i, m in enumerate(blk_map) if m["logical"] == L11_BLOCK_LOGICAL][0]
    print(f"=== {cfg} STEP={STEP} L11 block={b} off6={off6} head={HEAD} ===")

    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    x_in = todense(p.model.forward(padded, stop_after_block=b - 1)[0]).contiguous()
    attn = p.model.blocks[b].attn
    H = attn.num_heads
    HD = attn.head_dim
    Wq = todense(attn.W_q.data)
    Wk = todense(attn.W_k.data)
    Wv = todense(attn.W_v.data)
    Wo = todense(attn.W_o.data)
    S = x_in.shape[0]
    Q = (x_in @ Wq.T).contiguous().view(S, H, HD)
    K = (x_in @ Wk.T).contiguous().view(S, H, HD)
    V = (x_in @ Wv.T).contiguous().view(S, H, HD)
    h = HEAD
    sc = (K[:, h, :] @ Q[off6, h, :]) * attn.scale  # [S]
    if attn.alibi_slopes is not None:
        sl = float(attn.alibi_slopes[h])
        pos = torch.arange(S).float()
        dist = (pos - off6).clamp(max=0)  # k - q <= 0
        sc = sc + sl * dist
    sc[off6 + 1:] = -1e30
    mx = torch.maximum(sc.max(), torch.zeros(1))
    ex = torch.exp(sc - mx)
    attw = ex / (torch.exp(-mx) + ex.sum())
    ctx_h = attw @ V[:, h, :]
    contrib = float(ctx_h @ Wo[OUT_HI, h * HD:(h + 1) * HD])
    print(f" head{h} OUT_HI[0] contrib = {contrib:+.3f}")
    # top-5 attended positions
    topw, topi = attw.topk(5)
    for w, i in zip(topw.tolist(), topi.tolist()):
        if w < 0.01:
            continue
        rel = i - pl
        tag = _step_offset_field(rel % STEP) if rel >= 0 else "pre"
        st = rel // STEP if rel >= 0 else -1
        print(f"   w={w:.3f} pos{i} (step{st} {tag}) tok={ctx[i]} score={float(sc[i]):+.2f}")
    print(f"   softmax1 sink mass = {float(1 - attw.sum()):.3f}")

    # Decompose the K-score at the winning position into per-input-dim
    # contributions: score = sum_d x[win,d] * (W_k.T @ Q_head)[d] * scale.
    win = int(attw.argmax())
    qh = Q[off6, h, :]                       # [HD]
    wk_head = Wk[h * HD:(h + 1) * HD, :]     # [HD, D]
    kproj = (qh @ wk_head) * attn.scale      # [D] per-input-dim weight
    contribs = x_in[win] * kproj             # [D]
    rev = {v: k for k, v in dp.items()}
    top = contribs.abs().topk(12)
    print(f"   K-score dims at winning pos{win} (step{(win-pl)//STEP} "
          f"{_step_offset_field((win-pl) % STEP)}):")
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        c = float(contribs[idx])
        if abs(c) < 1.0:
            continue
        # name the dim (band base + offset)
        name = rev.get(idx)
        if name is None:
            # find band base
            for off in range(1, 17):
                if (idx - off) in rev:
                    name = f"{rev[idx-off]}+{off}"
                    break
            name = name or f"dim{idx}"
        print(f"     {name:24s} x={float(x_in[win,idx]):+.2f} -> Kscore {c:+.1f}")


if __name__ == "__main__":
    main()
