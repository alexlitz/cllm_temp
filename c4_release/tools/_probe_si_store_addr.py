#!/usr/bin/env python3
"""Verify the SI-store address-provenance LEVER on the BUILT layout (id275).

var_mul id275 (`int a;int b;a=23;b=47;return a*b;`) fails: `LI a` at step 11
returns b's 47 not a's 23. The brief's LEVER: the clean address AND clean value
BOTH live on the store AX-MARKER (a-marker: ADDR_B0=0xE8 + AX_CARRY=23;
b-marker: ADDR_B0=0xE0 + AX_CARRY=47) and the LI-query row carries its target
AX_CARRY=0xE8. This probe dumps, for the campaign config:
  - which rows are SI store AX-markers (OP_SI/MARK_AX) and their ADDR_B0 +
    AX_CARRY signatures,
  - the LI-query row's AX_CARRY (the target address key),
  - the L15 head-0 CAM's current attention (which row wins, what CLEAN it
    copies),
to CONFIRM (or refute) the lever before building the head.

Run: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     python tools/_probe_si_store_addr.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
# Campaign config (the 30-token MEM-from-SP path that this family lives in).
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
# LI steps: the a*b sequence does `LI a` then `LI b`. We scan a broad range.
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
    rev = {v: k for k, v in dp.items()}
    STEP = int(Token.STEP_TOKENS)
    print(f"STEP_TOKENS={STEP}")

    op_li = dp["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    print(f"L15 memory_lookup block = {L15}")

    bc, _ = compile_c(SRC)
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    cap = {}

    def pre_hook(module, inputs):
        cap["pre"] = inputs[0].detach().clone()

    h = model.blocks[L15].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        logits = model(padded)
        if logits.is_sparse:
            logits = logits.to_dense()
    h.remove()
    preds = torch.argmax(logits[0], dim=-1)

    pre = cap["pre"][0]
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    nH, hd = attn.num_heads, attn.head_dim
    for w in ("W_q", "W_k", "W_v"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_dense_" + w, t.to_dense())
    Wq = getattr(attn, "_dense_W_q", attn.W_q)
    Wk = getattr(attn, "_dense_W_k", attn.W_k)
    x = pre
    Q = (x @ Wq.T).view(seq, nH, hd)
    K = (x @ Wk.T).view(seq, nH, hd)

    def markers(pos):
        out = []
        for nm in ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                   "MARK_STACK0"):
            if nm in dp and pre[pos, dp[nm]].item() > 0.5:
                out.append(nm.replace("MARK_", ""))
        return out

    def nib(pos, base):
        band = pre[pos, dp[base]:dp[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()), round(mx, 2)) if mx > 0.3 else (None, round(mx, 2))

    def hexval(pos, lo_base, hi_base):
        lo, _ = nib(pos, lo_base)
        hi, _ = nib(pos, hi_base)
        if lo is None and hi is None:
            return None
        return ((hi or 0) << 4) | (lo or 0)

    # ---- Find all SI stores + their AX markers ----
    print("\n===== SI store frames (OP_SI rows) + AX-marker signatures =====")
    op_si = dp.get("OP_SI")
    mark_ax = dp["MARK_AX"]
    for pos in range(seq):
        is_si = op_si is not None and pre[pos, op_si].item() > 0.5
        is_axmk = pre[pos, mark_ax].item() > 0.5
        if not (is_si or (is_axmk and any(
                pre[pos, op_si].item() > 0.5 for _ in [0]) if op_si else False)):
            pass
        if is_si:
            addr_b0 = hexval(pos, "ADDR_B0_LO", "ADDR_B0_HI")
            axc = hexval(pos, "AX_CARRY_LO", "AX_CARRY_HI")
            print(f"  OP_SI@{pos:3d} mk={markers(pos)} "
                  f"ADDR_B0={addr_b0 if addr_b0 is None else hex(addr_b0)} "
                  f"AX_CARRY={axc if axc is None else hex(axc)}")
            # also inspect the neighboring rows (marker often on an adjacent row)
            for d in (-2, -1, 1, 2, 3, 4, 5):
                q = pos + d
                if 0 <= q < seq and pre[q, mark_ax].item() > 0.5:
                    ab = hexval(q, "ADDR_B0_LO", "ADDR_B0_HI")
                    ac = hexval(q, "AX_CARRY_LO", "AX_CARRY_HI")
                    print(f"      AX-mk@{q:3d}(d={d:+d}) "
                          f"ADDR_B0={ab if ab is None else hex(ab)} "
                          f"AX_CARRY={ac if ac is None else hex(ac)} "
                          f"mk={markers(q)}")

    # ---- Also dump all AX markers with an ADDR_B0 signal (the store markers) ----
    print("\n===== ALL rows with MARK_AX AND an ADDR_B0 nibble =====")
    for pos in range(seq):
        if pre[pos, mark_ax].item() > 0.5:
            ab = hexval(pos, "ADDR_B0_LO", "ADDR_B0_HI")
            if ab is not None:
                ac = hexval(pos, "AX_CARRY_LO", "AX_CARRY_HI")
                opsi = pre[pos, op_si].item() if op_si else 0.0
                print(f"  AX@{pos:3d} ADDR_B0={hex(ab)} "
                      f"AX_CARRY={ac if ac is None else hex(ac)} "
                      f"OP_SI={opsi:.2f} mk={markers(pos)}")

    # ---- The LI steps: query row's AX_CARRY + head-0 winner ----
    for si in LI_STEPS:
        if si >= len(steps):
            continue
        pc, ax = steps[si]
        base = pl + si * STEP
        b0row = base + 5
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        opli = pre[b0row, dp['OP_LI_RELAY']].item()
        if opli < 0.5 and got == 0 and ax == 0:
            continue
        qaddr = hexval(b0row, "ADDR_B0_LO", "ADDR_B0_HI")
        qaxc = hexval(b0row, "AX_CARRY_LO", "AX_CARRY_HI")
        print(f"\n===== ostep{si} want_ax=0x{ax:04x} got=0x{got:04x} "
              f"b0row={b0row} OP_LI_RELAY={opli:.2f} =====")
        print(f"  LI-query ADDR_B0={qaddr if qaddr is None else hex(qaddr)} "
              f"AX_CARRY={qaxc if qaxc is None else hex(qaxc)} mk={markers(b0row)}")
        qv = Q[b0row, 0]
        scores = (K[:, 0, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        topk = torch.topk(probs, k=5)
        for r in range(5):
            kp = int(topk.indices[r].item())
            pr = float(topk.values[r].item())
            clo = pre[kp, dp["CLEAN_EMBED_LO"]:dp["CLEAN_EMBED_LO"]+16]
            cli = int(clo.argmax().item())
            ab = hexval(kp, "ADDR_B0_LO", "ADDR_B0_HI")
            ac = hexval(kp, "AX_CARRY_LO", "AX_CARRY_HI")
            print(f"  K@{kp:3d} p={pr:.3f} mk={markers(kp)} CLEAN_LO={cli}/{float(clo[cli]):.2f} "
                  f"ADDR_B0={ab if ab is None else hex(ab)} "
                  f"AX_CARRY={ac if ac is None else hex(ac)}")


if __name__ == "__main__":
    main()
