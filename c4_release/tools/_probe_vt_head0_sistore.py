#!/usr/bin/env python3
"""Probe: var_three L15 head-0 behaviour at SI-store rows + LI-load rows.

Teacher-forces the oracle tape for var_three id300 (a=29,b=6,c=20; return a+b+c)
under the campaign config and reports, per step:
  - the opcode markers at the AX byte-0 emit row (OP_SI/OP_LI/OP_LI_RELAY etc.),
  - whether L15 head-0 (the LI/LC + STACK0 load head, value_scale=40) FIRES
    there (softmax mass on a non-self row) and what CLEAN_EMBED it would dump
    into OUTPUT,
  - the decoded AX byte-0 vs the oracle AX,
so we can see whether an SI-store row carries a stray OP_LI_RELAY that makes
head-0 mis-fire and overwrite the store value with a spurious lookup.

Run: CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 python tools/_probe_vt_head0_sistore.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_CAMPAIGN", "1")
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

_ID = int(os.environ.get("VT_PROBE_ID", "300"))
SRC = generate_test_programs()[_ID][0]


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(60):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = model.embed._dim_positions
    STEP = int(Token.STEP_TOKENS)

    op_li_relay = dp["OP_LI_RELAY"]
    # Find the L15 lookup block: head-0 slot-0 carries huge OP_LI_RELAY Q.
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li_relay].item())) > 1000.0:
            L15 = bi
    nH = model.blocks[L15].attn.num_heads
    print(f"L15 block={L15} num_heads={nH}")

    bc, _ = compile_c(SRC)
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    cap = {}
    h = model.blocks[L15].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
    with torch.no_grad():
        logits = model(padded)
        if logits.is_sparse:
            logits = logits.to_dense()
    h.remove()
    preds = torch.argmax(logits[0], dim=-1)

    pre = cap["pre"][0]
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    hd = attn.W_q.shape[0] // nH
    for w in ("W_q", "W_k"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_dense_" + w, t.to_dense())
    Wq = getattr(attn, "_dense_W_q", attn.W_q)
    Wk = getattr(attn, "_dense_W_k", attn.W_k)
    Q = (pre @ Wq.T).view(seq, nH, hd)
    K = (pre @ Wk.T).view(seq, nH, hd)

    def val(pos, base):
        return float(pre[pos, dp[base]].item())

    def nib(pos, base):
        band = pre[pos, dp[base]:dp[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()), round(mx, 2)) if mx > 0.3 else (None, round(mx, 2))

    def hexval(pos, lo, hi):
        l, _ = nib(pos, lo); hh, _ = nib(pos, hi)
        if l is None and hh is None:
            return None
        return ((hh or 0) << 4) | (l or 0)

    HEAD = 0
    print(f"\noracle steps ({len(steps)}): "
          + " ".join(f"{i}:ax={ax:#x}" for i, (pc, ax) in enumerate(steps)))
    for si in range(len(steps)):
        pc, ax = steps[si]
        base = pl + si * STEP
        b0row = base + 5
        if b0row + 1 > seq:
            break
        # opcode markers at the byte-0 emit row
        ms = {nm: val(b0row, nm) for nm in (
            "OP_SI", "OP_SC", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
            "OP_PSH", "OP_ENT", "OP_ADD", "MARK_AX", "MARK_STACK0", "MEM_STORE",
            "CMP")}
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        # head-0 attention at this row
        qv = Q[b0row, HEAD]
        scores = (K[:, HEAD, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        top = torch.topk(probs, k=3)
        selfp = float(probs[b0row].item())
        outlo = float(pre[b0row, dp["OUTPUT_LO"]:dp["OUTPUT_LO"] + 16].max().item())
        active = {k: round(v, 2) for k, v in ms.items() if abs(v) > 0.3}
        flag = ""
        if ms["OP_SI"] > 0.3 or ms["OP_SC"] > 0.3:
            flag = " <-- SI/SC STORE ROW"
        print(f"\nstep{si} ax={ax:#06x} got=0x{got:04x} "
              f"{'OK' if got == ax else 'XX'} b0row={b0row}{flag}")
        print(f"   markers: {active}")
        print(f"   head0 self_p={selfp:.3f} OUTPUT_LO_max={outlo:.2f}")
        for r in range(3):
            kp = int(top.indices[r].item())
            pr = float(top.values[r].item())
            if pr < 0.02:
                continue
            ce = hexval(kp, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
            tag = " (SELF)" if kp == b0row else ""
            print(f"      h0 K@{kp:3d} p={pr:.3f} CLEAN_EMBED={None if ce is None else hex(ce)}{tag}")


if __name__ == "__main__":
    main()
