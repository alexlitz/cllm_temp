#!/usr/bin/env python3
"""Inc-3 ROOT A: the AX byte-1 DUMP carry head row-selection probe.

Recomputes the ``layer13_ax_byte1_dump_carry.head_7`` (logical L13, block 18)
softmax attention over the KV window from the step5 byte-1 predictor (AX[0])
row, GOLDEN vs CAMPAIGN, for x=990. Prints the top attended rows with their
``H1`` one-hot (the value the head V-copies into ``H1_PREV_STEP``) so we can see
WHICH prior-step row the head lands on and how the 30-tok stride shifts it.

The head's V copies dim ``H1`` of the attended row into ``H1_PREV_STEP`` (slots
10..16, the H1 block). So the H1 one-hot of the dominant attended row IS what
lands in ``H1_PREV_STEP``. GOLDEN should attend a prev row whose H1 == byte1
of x (0x03 -> slot 2 via H1+(v+2)? we read the H1 argmax). CAMPAIGN attends a
different prev row -> H1 slot 5.

Run TWICE (clear ~/.cache/c4_release/compiled_vm/ between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_rootA_carryattn.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_rootA_carryattn.py
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
import math
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"
BLK = 18    # logical L13 host of the carry head
HEAD = 7    # _AX_BYTE1_DUMP_CARRY_HEAD_IDX


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def oh(vec, base, n=16, thr=0.4):
    seg = vec[base:base + n]
    m = float(torch.max(seg))
    return (int(torch.argmax(seg)) if m > thr else -1), m


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off5 = pl + 5 * STEP + 6   # step5 AX[0] predictor row
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    # residual entering block BLK (== output of block BLK-1):
    x = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    attn = p.model.blocks[BLK].attn
    nh = attn.num_heads
    HD = attn.W_q.shape[0] // nh
    Wq = td(attn.W_q)[HEAD * HD:(HEAD + 1) * HD]
    Wk = td(attn.W_k)[HEAD * HD:(HEAD + 1) * HD]
    d_model = Wq.shape[1]
    xr = x[:, :d_model]
    q = xr[off5] @ Wq.T
    K = xr @ Wk.T
    scores = K @ q / math.sqrt(HD)
    slope = 0.0
    if getattr(attn, "alibi_slopes", None) is not None:
        slope = float(td(attn.alibi_slopes)[HEAD])
    T = xr.shape[0]
    pos = torch.arange(T, dtype=torch.float32)
    bias = -slope * (off5 - pos).abs()
    bias[pos > off5] = -1e9
    sc = scores + bias
    sc[off5 + 1:] = -1e9
    w = torch.softmax(sc, dim=0)
    h1 = dp["H1"]
    addr5 = dp.get("ADDR_B0_LO")
    print(f"=== {cfg} STEP={STEP} off5(step5 AX[0] pred)={off5} "
          f"head{HEAD} slope={slope} HD={HD} nh={nh} ===")
    # Show per-step the AX-byte predictor rows + their H1 one-hot, and where the
    # head attends. For each VM step, the AX byte rows live at offsets 6..9
    # (AX[0]..AX[3]) and the AX marker at the AX_marker offset.
    print("  --- per-row: attn-weight, score, H1 one-hot, ADDR_B0_LO ---")
    for st in range(0, 6):
        base = pl + st * STEP
        for o in range(0, min(10, STEP)):
            r = base + o
            if r > off5:
                break
            wv = float(w[r])
            if wv < 0.02 and o not in (5, 6, 7):
                continue
            h1v, h1m = oh(xr[r], h1)
            a5 = float(xr[r][addr5 + 5]) if addr5 is not None else 0.0
            mark = ""
            tag = _step_offset_field(o)
            print(f"    s{st}.{o:02d} row={r:4d} {tag:11s} "
                  f"w={wv:.3f} sc={float(scores[r]):+7.1f} "
                  f"H1={h1v}({h1m:.2f}) ADDR_B0_LO+5={a5:+.2f}")
    top = torch.topk(w, 8)
    print("  --- TOP attended rows ---")
    for wv, r in zip(top.values.tolist(), top.indices.tolist()):
        if wv < 0.005:
            continue
        st = (r - pl) // STEP if r >= pl else -1
        o = (r - pl) % STEP if r >= pl else r
        h1v, h1m = oh(xr[r], h1)
        print(f"  attn={wv:.3f} row={r:4d} (s{st}.{o:02d} {_step_offset_field(o):11s}) "
              f"H1={h1v}({h1m:.2f})")
    # The resulting H1_PREV_STEP = sum_r w[r] * xr[r][H1+j] for each cell j:
    h1prev_acc = torch.zeros(7)
    for r in range(T):
        if float(w[r]) < 1e-4:
            continue
        for j in range(7):
            h1prev_acc[j] += float(w[r]) * float(xr[r][h1 + j])
    best = int(torch.argmax(h1prev_acc))
    print(f"  => predicted H1_PREV_STEP one-hot (V-copy of H1): "
          f"slot {best} ({float(h1prev_acc[best]):.2f}) "
          f"full={[round(float(v),2) for v in h1prev_acc]}")


if __name__ == "__main__":
    main()
