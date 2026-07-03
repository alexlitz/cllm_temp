#!/usr/bin/env python3
"""Flag-ON verification of the SI-store-addr CAM head (id275).

With C4_SI_STORE_ADDR=1 the L15 head-16 CAM should content-address the ``LI a``
target (AX_CARRY=0xE8) to the a-store marker (ADDR_B0=0xE8, AX_CARRY=0x17=23)
and copy AX_CARRY into OUTPUT so ``LI a`` decodes 23 (not b's 47). This probe
teacher-forces the oracle context and reports:
  - head-16 attention at each LI byte-0 lookup row (which row wins, its
    AX_CARRY value),
  - the decoded AX byte-0..3 at that row (should match the target local's value).

Run: CUDA_VISIBLE_DEVICES=0 C4_SI_STORE_ADDR=1 C4_NO_STACK0_EMIT=1 \
     C4_OPERAND_FROM_MEMSP=1 python tools/_probe_si_store_addr_on.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_SI_STORE_ADDR", "1")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
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

SRC = "int main() { int a; int b; a = 23; b = 47; return a * b; }"
LI_STEPS = list(range(8, 20))


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(40):
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

    op_li = dp["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
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
    x = pre
    Q = (x @ Wq.T).view(seq, nH, hd)
    K = (x @ Wk.T).view(seq, nH, hd)
    HEAD = 16

    def nib(pos, base):
        band = pre[pos, dp[base]:dp[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()), round(mx, 2)) if mx > 0.3 else (None, round(mx, 2))

    def hexval(pos, lo, hi):
        l, _ = nib(pos, lo); hh, _ = nib(pos, hi)
        if l is None and hh is None:
            return None
        return ((hh or 0) << 4) | (l or 0)

    npass = 0
    ntot = 0
    for si in LI_STEPS:
        if si >= len(steps):
            continue
        pc, ax = steps[si]
        base = pl + si * STEP
        b0row = base + 5
        opli = pre[b0row, dp['OP_LI']].item()
        if opli < 0.5:
            continue  # only real LI emit rows
        ntot += 1
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        qaxc = hexval(b0row, "AX_CARRY_LO", "AX_CARRY_HI")
        ok = (got == ax)
        npass += int(ok)
        print(f"\n== ostep{si} LI want_ax=0x{ax:04x} got=0x{got:04x} "
              f"{'PASS' if ok else 'FAIL'} b0row={b0row} "
              f"OP_LI={opli:.2f} target_AXCARRY={None if qaxc is None else hex(qaxc)} ==")
        qv = Q[b0row, HEAD]
        scores = (K[:, HEAD, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        topk = torch.topk(probs, k=4)
        for r in range(4):
            kp = int(topk.indices[r].item())
            pr = float(topk.values[r].item())
            ab = hexval(kp, "ADDR_B0_LO", "ADDR_B0_HI")
            ac = hexval(kp, "AX_CARRY_LO", "AX_CARRY_HI")
            opsh = pre[kp, dp["OP_PSH"]].item()
            print(f"   h16 K@{kp:3d} p={pr:.3f} ADDR_B0={None if ab is None else hex(ab)} "
                  f"AX_CARRY={None if ac is None else hex(ac)} OP_PSH={opsh:.2f}")

    print(f"\n=== teacher-forced LI decode: {npass}/{ntot} correct ===")


if __name__ == "__main__":
    main()
