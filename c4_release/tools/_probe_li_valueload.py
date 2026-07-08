#!/usr/bin/env python3
"""Probe: L15 head-0 LI value-load delivery at every LI/LC row for arbitrary ids.

Teacher-forces the oracle tape (campaign config) and reports, per step:
  - opcode markers at the AX byte-0 emit row (OP_LI/OP_LC/OP_LI_RELAY/GT/...),
  - the L15 head-0 (li_lc_stack0_h0) softmax winner rows + their CLEAN_EMBED,
    MEM_VAL_B1, ADDR_B0 one-hots, OP_JSR/OP_ENT/OP_GT residues,
  - decoded AX byte-0 vs oracle AX,
so we can attribute WHY a genuine LI-load (e.g. func_max/min step-12 `b`)
delivers AX=0 instead of the stored value.

Honours whatever C4_* env is set (run twice, flag unset vs =1, to A/B a fix).

Run:
  CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 VT_PROBE_ID=650 python tools/_probe_li_valueload.py
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

_ID = int(os.environ.get("VT_PROBE_ID", "650"))
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
    print(f"id{_ID} [{DESC}]  L15 block={L15} num_heads={nH} "
          f"C4_L15_LI_VALROW_B1={os.environ.get('C4_L15_LI_VALROW_B1')} "
          f"C4_L15_LI_JSR_PHANTOM={os.environ.get('C4_L15_LI_JSR_PHANTOM')}")

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
        l, _ = nib(pos, lo)
        hh, _ = nib(pos, hi)
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
        ms = {nm: val(b0row, nm) for nm in (
            "OP_SI", "OP_SC", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
            "OP_PSH", "OP_ENT", "OP_JSR", "OP_LEA", "OP_GT", "OP_LT",
            "MARK_AX", "MARK_STACK0", "MEM_STORE")}
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        is_li = ms["OP_LI"] > 0.3 or ms["OP_LC"] > 0.3 or ms["OP_LI_RELAY"] > 0.3
        if not is_li and got == ax:
            continue  # only detail LI rows or wrong rows
        qv = Q[b0row, HEAD]
        scores = (K[:, HEAD, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        top = torch.topk(probs, k=4)
        selfp = float(probs[b0row].item())
        active = {k: round(v, 2) for k, v in ms.items() if abs(v) > 0.3}
        flag = " <-- LI/LC LOAD ROW" if is_li else ""
        print(f"\nstep{si} pc={pc} ax={ax:#06x} got=0x{got:04x} "
              f"{'OK' if got == ax else 'XX'} b0row={b0row}{flag}")
        print(f"   markers: {active}")
        print(f"   head0 self_p={selfp:.3f}")
        for r in range(4):
            kp = int(top.indices[r].item())
            pr = float(top.values[r].item())
            if pr < 0.02:
                continue
            ce = hexval(kp, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
            mvb1 = round(val(kp, "MEM_VAL_B1"), 2)
            a_lo, _ = nib(kp, "ADDR_B0_LO")
            a_hi, _ = nib(kp, "ADDR_B0_HI")
            adb0 = None if a_lo is None and a_hi is None else ((a_hi or 0) << 4) | (a_lo or 0)
            jsr = round(val(kp, "OP_JSR"), 2)
            ent = round(val(kp, "OP_ENT"), 2)
            tag = " (SELF)" if kp == b0row else ""
            print(f"      h0 K@{kp:3d} p={pr:.3f} CLEAN={None if ce is None else hex(ce)} "
                  f"ADDR_B0={None if adb0 is None else hex(adb0)} MEM_VAL_B1={mvb1} "
                  f"OP_JSR={jsr} OP_ENT={ent}{tag}")


if __name__ == "__main__":
    main()
