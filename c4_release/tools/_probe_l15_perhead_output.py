#!/usr/bin/env python3
"""Per-head OUTPUT_LO/HI contribution decomposition at the LI byte-0 row.

For a given id + step, teacher-forces the oracle tape and, at the AX byte-0
emit row, computes EACH L15 head's post-softmax attention output projected
through W_o into OUTPUT_LO / OUTPUT_HI (all 16 nibbles), so we can see which
head writes the spurious nibble-0 default that out-votes head-0's value.

Run:
  CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 VT_PROBE_ID=675 VT_STEP=12 python tools/_probe_l15_perhead_output.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ.setdefault("C4_JSR_BP_BYTE3_CLEAR", "1")
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

_ID = int(os.environ.get("VT_PROBE_ID", "675"))
_STEP = int(os.environ.get("VT_STEP", "12"))
_TESTS = generate_test_programs()
SRC = _TESTS[_ID][0]
DESC = _TESTS[_ID][2]


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
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li_relay].item())) > 1000.0:
            L15 = bi
    nH = model.blocks[L15].attn.num_heads

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
        model(padded)
    h.remove()

    pre = cap["pre"][0]
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    hd = attn.W_q.shape[0] // nH
    for w in ("W_q", "W_k", "W_v", "W_o"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_d_" + w, t.to_dense())
    Wq = getattr(attn, "_d_W_q", attn.W_q)
    Wk = getattr(attn, "_d_W_k", attn.W_k)
    Wv = getattr(attn, "_d_W_v", attn.W_v)
    Wo = getattr(attn, "_d_W_o", attn.W_o)

    pc, ax = steps[_STEP]
    base = pl + _STEP * STEP
    b0row = base + 5
    print(f"id{_ID} [{DESC}] step{_STEP} pc={pc} ax={ax:#06x} b0row={b0row} nH={nH}")

    Q = (pre @ Wq.T).view(seq, nH, hd)
    K = (pre @ Wk.T).view(seq, nH, hd)
    V = (pre @ Wv.T).view(seq, nH, hd)

    lo0 = dp["OUTPUT_LO"]
    hi0 = dp["OUTPUT_HI"]
    # alibi slopes: try to read from attn
    slopes = getattr(attn, "alibi_slopes", None)

    def head_out(H):
        qv = Q[b0row, H]
        scores = (K[:, H, :] @ qv) / (hd ** 0.5)
        if slopes is not None:
            sl = float(slopes[H]) if hasattr(slopes, "__len__") else float(slopes)
            dist = torch.arange(b0row + 1, device=scores.device) - b0row
            scores[:b0row + 1] = scores[:b0row + 1] + sl * dist.float()
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        ctxv = (probs.unsqueeze(-1) * V[:, H, :]).sum(0)  # hd
        # project this head's slice through W_o
        out = torch.zeros(Wo.shape[0], device=ctxv.device)
        out = Wo[:, H * hd:(H + 1) * hd] @ ctxv
        selfp = float(probs[b0row].item())
        win = int(torch.argmax(probs).item())
        return out, selfp, win

    def nibhex(pos, base):
        b = pre[pos, dp[base]:dp[base] + 16]
        mx = float(b.max().item())
        return int(b.argmax().item()) if mx > 0.3 else None

    def rowinfo(pos):
        acl = nibhex(pos, "AX_CARRY_LO")
        ach = nibhex(pos, "AX_CARRY_HI")
        adl = nibhex(pos, "ADDR_B0_LO")
        adh = nibhex(pos, "ADDR_B0_HI")
        axc = None if acl is None and ach is None else ((ach or 0) << 4) | (acl or 0)
        adb = None if adl is None and adh is None else ((adh or 0) << 4) | (adl or 0)
        ops = {nm: round(float(pre[pos, dp[nm]].item()), 1)
               for nm in ("OP_SI", "OP_SC", "OP_LI", "OP_ENT", "OP_JSR", "OP_PSH", "MARK_AX")
               if abs(float(pre[pos, dp[nm]].item())) > 0.3}
        return (f"AX_CARRY={None if axc is None else hex(axc)} "
                f"ADDR_B0={None if adb is None else hex(adb)} {ops}")

    print(f"\nLI-query row {b0row}: {rowinfo(b0row)}")
    print(f"\n{'head':>4} {'self_p':>7} {'win':>5}  OUTPUT_LO nibble contribs (>2)")
    for H in range(nH):
        out, selfp, win = head_out(H)
        lo = out[lo0:lo0 + 16]
        hi = out[hi0:hi0 + 16]
        lo_big = {i: round(float(lo[i]), 1) for i in range(16) if abs(float(lo[i])) > 2}
        hi_big = {i: round(float(hi[i]), 1) for i in range(16) if abs(float(hi[i])) > 2}
        if lo_big or hi_big:
            print(f"{H:>4} {selfp:>7.3f} {win:>5}  LO={lo_big}  HI={hi_big}")
            if H in (0, 14, 16):
                print(f"          win-row {win}: {rowinfo(win)}")


if __name__ == "__main__":
    main()
