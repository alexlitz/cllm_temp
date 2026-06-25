#!/usr/bin/env python3
"""AR-LI: decode L15 head-0 attention distribution at the LI byte-0 predictor row.

Computes the actual softmax1 weights for L15 head 0 at the AX-marker row (off5)
of the LI step, on the TRUE AR tape, to see which K row wins and why. Prints the
top-attended positions with their CLEAN_EMBED byte value + key signals.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 PROBE_SRC='int main(){int x;x=28;return x;}' \
    PROBE_LI_STEP=7 python tools/probe_arli_attn.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { int x; x = 28; return x; }")
LI_STEP = int(os.environ.get("PROBE_LI_STEP", "7"))
OFF = int(os.environ.get("PROBE_OFF", "5"))
HEAD = int(os.environ.get("PROBE_HEAD", "0"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def onehot16(row, base):
    idx = int(torch.argmax(row[base:base + 16]).item())
    return idx, float(row[base + idx])


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
    prompt_len = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=LI_STEP + 1)
    padded = torch.tensor([ctx], device=p._device)
    blk_map = p.block_layer_map()
    import torch as _t
    # The LIVE LI/LC lookup heads are identified by slot-31 OP_LI_RELAY=20000
    # (the suppress signature), not by the logical-15 label (post-op expansion
    # places them on a later physical block, mislabelled L19).
    forced = os.environ.get("PROBE_BLK")
    if forced is not None:
        l15 = int(forced)
    else:
        l15 = None
        for b in range(len(p.model.blocks)):
            attn = getattr(p.model.blocks[b], "attn", None)
            if attn is None:
                continue
            hd = attn.W_q.shape[0] // attn.num_heads
            if hd > 31 and abs(float(td(attn.W_q)[31, dp["OP_LI_RELAY"]]) - 20000.0) < 1.0:
                l15 = b
                break
        if l15 is None:
            l15 = [b for b in range(len(p.model.blocks))
                   if blk_map[b].get("logical") == 15][0]

    s = prompt_len + LI_STEP * STEP
    qpos = s + OFF
    print(f"=== {cfg} L15(blk{l15}) head{HEAD} qpos={qpos} (step{LI_STEP} off{OFF}) ===")

    # Residual entering L15 (after the block before it).
    x = td(p.model.forward(padded, stop_after_block=l15 - 1)[0])  # [S, D]
    attn = p.model.blocks[l15].attn
    Wq = td(attn.W_q); Wk = td(attn.W_k)
    HD = Wq.shape[0] // attn.num_heads
    base = HEAD * HD
    q = x[qpos] @ Wq[base:base + HD].T  # [HD]
    K = x @ Wk[base:base + HD].T        # [S, HD]
    scale = float(getattr(attn, "scale", 1.0 / math.sqrt(HD)))
    scores = (K @ q) * scale            # [S]
    # ALiBi
    slope = None
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        slope = float(td(attn.alibi_slopes)[HEAD])
    if slope is not None:
        dist = torch.arange(len(ctx)).float() - qpos
        scores = scores + slope * dist  # dist<=0 for causal
    # causal mask
    scores[qpos + 1:] = -1e30
    # production uses standard F.softmax (no softmax1 zero anchor)
    m = float(scores.max())
    ex = torch.exp(scores - m)
    denom = float(ex.sum())
    w = ex / denom
    top = torch.topk(w, 8)
    print(f"  blk{l15} slope={slope} scale={scale:.4f} max_score={m:+.2f}")
    CE_LO = dp["CLEAN_EMBED_LO"]; CE_HI = dp["CLEAN_EMBED_HI"]
    for wi, pi in zip(top.values.tolist(), top.indices.tolist()):
        row = x[pi]
        loi, _ = onehot16(row, CE_LO)
        hii, _ = onehot16(row, CE_HI)
        val = hii * 16 + loi
        stepi = (pi - prompt_len) // STEP if pi >= prompt_len else -1
        offi = (pi - prompt_len) % STEP if pi >= prompt_len else pi
        print(f"    pos{pi:3d} (step{stepi} off{offi}) w={wi:.4f} score={float(scores[pi]):+.1f} "
              f"CE_val={val} MEM_STORE={float(row[dp['MEM_STORE']]):.1f} "
              f"L2H0[MEM]={float(row[dp['L2H0']+4]):.1f} MARK_MEM={float(row[dp['MARK_MEM']]):.1f}")


if __name__ == "__main__":
    main()
