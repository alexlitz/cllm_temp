#!/usr/bin/env python3
"""Single-build probe: which K position does L15 head 0 attend to on the
step-1 (LEA) STACK0[0] predictor row, and what marker does that row carry?

Confirms the BP-row K-alias: head 0's STACK0-pop lookup attends to the
BP/frame row (MARK_BP=1, holding 0xFFF0) instead of the empty stack top.
Builds the final context ONCE, runs ONE forward to the L15-input block, then
manually replays head 0's softmax to dump the attention distribution.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
import torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

reg = build_default_registry_dynamic()

MARKERS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
           "MARK_STACK0", "MARK_SE_ONLY", "IS_BYTE", "MEM_STORE", "HAS_SE",
           "CMP", "OP_LEA", "OP_ENT", "OP_LI_RELAY", "OP_LC_RELAY"]


def D(name):
    base, off = name, 0
    if "+" in name:
        base, o = name.split("+")
        off = int(o)
    if base not in reg.slots:
        return None
    return reg.slots[base].start + off


SRC = "int main() { int x; x = 990; return x; }"


def marker_of(row):
    active = []
    for m in MARKERS:
        d = D(m)
        if d is not None and abs(float(row[d])) > 0.5:
            active.append(f"{m}={round(float(row[d]),1)}")
    return ",".join(active) if active else "(none)"


def main():
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    prompt_len = len(p._build_context(bytecode))
    model = p.model
    bmap = p.block_layer_map()

    l15_blocks = [b["physical"] for b in bmap if b["logical"] == 15]
    print(f"L15 physical blocks: {l15_blocks}")
    for b in bmap:
        if b["logical"] == 15:
            print(f"  phys{b['physical']}: attn={b['attn']} ffn={b['ffn']} "
                  f"post={b['is_post_op_expansion']}")

    l15_attn_blk = None
    for phys in l15_blocks:
        attn = getattr(model.blocks[phys], "attn", None)
        if attn is not None and hasattr(attn, "W_q"):
            nh = getattr(attn, "num_heads", 1)
            if nh >= 4:
                l15_attn_blk = phys
                print(f"  -> using phys{phys} memory_lookup (num_heads={nh})")
                break
    if l15_attn_blk is None:
        for phys in l15_blocks:
            if hasattr(getattr(model.blocks[phys], "attn", None), "W_q"):
                l15_attn_blk = phys
                break
    print(f"L15 memory_lookup attn block = {l15_attn_blk}")

    ctx = p._final_context(bytecode, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    resid_in = model.forward(padded, stop_after_block=l15_attn_blk - 1)[0]

    attn = model.blocks[l15_attn_blk].attn
    H = attn.num_heads
    HD = attn.head_dim
    S = resid_in.shape[0]

    x = resid_in.unsqueeze(0)
    Q = F.linear(x, attn.W_q).view(1, S, H, HD).transpose(1, 2)
    K = F.linear(x, attn.W_k).view(1, S, H, HD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [1,H,S,S]

    # ALiBi bias + causal mask (mirror AutoregressiveAttention.forward).
    pos = torch.arange(S, device=x.device)
    if getattr(attn, "alibi_slopes", None) is not None:
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs().float()  # [S,S]
        bias = -attn.alibi_slopes.view(1, H, 1, 1) * dist
    else:
        bias = torch.zeros(1, H, S, S, device=x.device)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device),
                        diagonal=1)
    scores = scores + bias + causal.view(1, 1, S, S)

    if getattr(attn, "use_softmax1", False):
        # softmax1: append a zero-logit sink column, drop it after softmax.
        sink = torch.zeros(1, H, S, 1, device=x.device)
        ext = torch.cat([scores, sink], dim=-1)
        aw = F.softmax(ext, dim=-1)[..., :S]
    else:
        aw = F.softmax(scores, dim=-1)

    pred = prompt_len + 35 + 20
    print(f"\nprompt_len={prompt_len} pred_pos(step1 off20)={pred} S={S}")
    print(f"\n=== Head 0 attention from STACK0[0] predictor row {pred} ===")
    row_aw = aw[0, 0, pred]
    topk = torch.topk(row_aw, k=12)
    for w, j in zip(topk.values.tolist(), topk.indices.tolist()):
        if w < 1e-4:
            continue
        rel = j - prompt_len
        mk = marker_of(resid_in[j])
        tok = ctx[j] if j < len(ctx) else -1
        rawscore = float(scores[0, 0, pred, j])
        print(f"  pos{j:3d} (rel{rel:+4d}) tok={tok:4d} w={w:.4f} "
              f"score={rawscore:.1f}  {mk}")

    for h in range(1, 4):
        row_aw = aw[0, h, pred]
        topk = torch.topk(row_aw, k=3)
        parts = []
        for w, j in zip(topk.values.tolist(), topk.indices.tolist()):
            if w < 1e-3:
                continue
            mk = marker_of(resid_in[j]).split(",")[0]
            parts.append(f"pos{j}(w{w:.2f},{mk})")
        print(f"  head{h}: {' '.join(parts)}")

    # max attention weight on this row = sharpness indicator.
    print(f"\n  head0 max attn weight = {float(aw[0,0,pred].max()):.4f} "
          f"(sum={float(aw[0,0,pred].sum()):.4f}; "
          f"diffuse if max small / sum<1 => softmax1 sink dominates)")

    # === Compare: SI/LI roundtrip working STACK0-load row (must stay sharp) ===
    from tests.test_smoke import _MEM_TESTS
    li_bc = next(t["bytecode"] for t in _MEM_TESTS
                 if t["name"].endswith("test_si_li_roundtrip"))
    li_plen = len(p._build_context(li_bc))
    li_ctx = p._final_context(li_bc, max_steps=20)
    li_resid = model.forward(
        torch.tensor([li_ctx], dtype=torch.long, device=p._device),
        stop_after_block=l15_attn_blk - 1)[0]
    Sl = li_resid.shape[0]
    xl = li_resid.unsqueeze(0)
    Ql = F.linear(xl, attn.W_q).view(1, Sl, H, HD).transpose(1, 2)
    Kl = F.linear(xl, attn.W_k).view(1, Sl, H, HD).transpose(1, 2)
    sl = torch.matmul(Ql, Kl.transpose(-2, -1)) * attn.scale
    posl = torch.arange(Sl, device=x.device)
    if getattr(attn, "alibi_slopes", None) is not None:
        distl = (posl.unsqueeze(1) - posl.unsqueeze(0)).abs().float()
        biasl = -attn.alibi_slopes.view(1, H, 1, 1) * distl
    else:
        biasl = torch.zeros(1, H, Sl, Sl, device=x.device)
    causl = torch.triu(torch.full((Sl, Sl), float("-inf"), device=x.device),
                       diagonal=1)
    sl = sl + biasl + causl.view(1, 1, Sl, Sl)
    if getattr(attn, "use_softmax1", False):
        sink = torch.zeros(1, H, Sl, 1, device=x.device)
        awl = F.softmax(torch.cat([sl, sink], dim=-1), dim=-1)[..., :Sl]
    else:
        awl = F.softmax(sl, dim=-1)
    # STACK0 predictor rows in SI/LI: find MARK_STACK0=1 byte-predictor rows.
    st0 = D("MARK_STACK0")
    print("\n=== SI/LI head0 attn on STACK0 predictor rows (must stay sharp) ===")
    for j in range(li_plen, Sl):
        if abs(float(li_resid[j][st0])) > 0.5:
            mx = float(awl[0, 0, j].max())
            sm = float(awl[0, 0, j].sum())
            arg = int(awl[0, 0, j].argmax())
            print(f"  pos{j}(rel{j-li_plen:+d}) STACK0row "
                  f"max_w={mx:.4f} sum={sm:.4f} argmax=pos{arg} "
                  f"({marker_of(li_resid[arg]).split(',')[0]})")


if __name__ == "__main__":
    main()
