#!/usr/bin/env python3
"""AR-LI: why is OP_LI_RELAY=0? Decode L7 head-5 attention + V-contribution to
OP_LI_RELAY at the LI step AX-marker row (off5) on the TRUE AR tape.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys, math
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


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


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
    nblk = len(p.model.blocks)
    l7 = [b for b in range(nblk) if blk_map[b].get("logical") == 7]
    blk = min(l7)  # the attn-bearing L7 block
    HEAD = 5
    s = prompt_len + LI_STEP * STEP
    qpos = s + OFF
    print(f"=== {cfg} L7(blk{blk}) head{HEAD} qpos={qpos} (step{LI_STEP} off{OFF}) ===")

    x = td(p.model.forward(padded, stop_after_block=blk - 1)[0])
    attn = p.model.blocks[blk].attn
    Wq = td(attn.W_q); Wk = td(attn.W_k); Wv = td(attn.W_v)
    HD = Wq.shape[0] // attn.num_heads
    base = HEAD * HD
    q = x[qpos] @ Wq[base:base + HD].T
    K = x @ Wk[base:base + HD].T
    scores = (K @ q) / math.sqrt(HD)
    slope = float(td(attn.alibi_slopes)[HEAD]) if (
        hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None) else None
    if slope is not None:
        dist = torch.arange(len(ctx)).float() - qpos
        scores = scores + slope * dist
    scores[qpos + 1:] = -1e30
    m = float(scores.max())
    ex = torch.exp(scores - m)
    denom = float(ex.sum()) + math.exp(0.0 - m)
    w = ex / denom
    zero_anchor = math.exp(0 - m) / denom
    print(f"  slope={slope}  softmax1 zero-anchor weight={zero_anchor:.4f}  max_score={m:+.2f}")
    top = torch.topk(w, 6)
    OP_LI = dp["OP_LI"]
    for wi, pi in zip(top.values.tolist(), top.indices.tolist()):
        row = x[pi]
        stepi = (pi - prompt_len) // STEP if pi >= prompt_len else -1
        offi = (pi - prompt_len) % STEP if pi >= prompt_len else pi
        print(f"    pos{pi:3d} (step{stepi} off{offi}) w={wi:.4f} score={float(scores[pi]):+.2f} "
              f"MARK_AX={float(row[dp['MARK_AX']]):+.1f} OP_LI={float(row[OP_LI]):+.1f} "
              f"OP_IMM={float(row[dp['OP_IMM']]):+.1f}")
    # V-contribution to OP_LI_RELAY: O writes OP_LI_RELAY from V slot 1 (= OP_LI*0.2).
    # head out = sum_j w_j * V[j]; V slot1 = x[j] @ Wv[base+1].
    Vslot1 = x @ Wv[base + 1]  # [S]
    out_slot1 = float((w * Vslot1).sum())
    print(f"  V-slot1 (OP_LI*0.2) attention-weighted sum = {out_slot1:.4f} "
          f"-> OP_LI_RELAY add (O scale 1.0)")


if __name__ == "__main__":
    main()
