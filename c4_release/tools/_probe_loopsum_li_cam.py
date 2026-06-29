#!/usr/bin/env python3
"""Trace L15 head 0 (LI byte-0 value lookup) CAM at the loop_sum id450 in-loop LI.

loop_sum id450 (campaign config) reaches step 11 = the in-loop `LI &i` (the
loop-condition variable load, pc 106->114). Oracle ax=1 (i was stored=1 at the
prologue SI step5); model returns ax=0. This probe teacher-forces the clean
oracle context and dumps, at the step-11 AX-byte-0 prediction row, which K row
(MEM store) head 0 attends, the per-row addr/value signature, and the
post-softmax probabilities — revealing WHY the in-loop CAM returns 0.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_li_cam.py [pid] [li_step]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 450
LI_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 11


def oracle_windows(bc, n=20):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(n):
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
    dimpos = model.embed._dim_positions
    rev = {v: k for k, v in dimpos.items()}
    STEP = int(Token.STEP_TOKENS)

    op_li = dimpos["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    print(f"L15 memory_lookup block = {L15} STEP_TOKENS={STEP}")

    src, exp, desc = generate_test_programs()[PID]
    bc = compile_c(src)[0]
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc, LI_STEP + 3)
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
            if nm in dimpos and pre[pos, dimpos[nm]].item() > 0.5:
                out.append(nm.replace("MARK_", ""))
        return out

    mem_pos = [i for i in range(seq) if pre[i, dimpos["MARK_MEM"]].item() > 0.5]
    store_rows = [(mp, mp + 5) for mp in mem_pos
                  if float(pre[mp, dimpos["MEM_STORE"]].item()) > 0.5]
    print(f"seq={seq} pl={pl} MEM rows: {mem_pos}")
    print(f"MEM STORE rows: {[mp for mp, _ in store_rows]}")

    si = LI_STEP
    pc, ax = steps[si]
    base = pl + si * STEP
    b0row = base + 5  # AX byte-0 prediction row
    emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
    got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
    print(f"\n===== ostep{si} (pc={pc}) want_ax=0x{ax:04x} got=0x{got:04x} "
          f"b0row={b0row} markers={markers(b0row)} =====")
    opli = pre[b0row, dimpos['OP_LI_RELAY']].item()
    print(f"  OP_LI_RELAY@b0row={opli:.2f}")
    for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "FETCH_LO", "FETCH_HI"):
        if nm not in dimpos:
            continue
        band = pre[b0row, dimpos[nm]:dimpos[nm]+16]
        top = [(int(i), round(float(band[i]), 2)) for i in
               torch.nonzero(band.abs() > 0.3).flatten().tolist()]
        print(f"    Q.{nm}: {top}")

    h0 = 0
    qv = Q[b0row, h0]
    scores = (K[:, h0, :] @ qv) / (hd ** 0.5)
    scores[b0row + 1:] = float("-inf")
    probs = torch.softmax(scores, dim=0)
    topk = torch.topk(probs, k=8)
    print("  --- head-0 attention (top-8 K rows) ---")
    for r in range(8):
        kp = int(topk.indices[r].item())
        pr = float(topk.values[r].item())
        sc = float(scores[kp].item())
        mst = float(pre[kp, dimpos["MEM_STORE"]].item())
        mvb = [float(pre[kp, dimpos[f"MEM_VAL_B{j}"]].item())
               for j in range(4) if f"MEM_VAL_B{j}" in dimpos]
        clo = pre[kp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
        cli = int(clo.argmax().item())
        print(f"  K@{kp:3d} p={pr:.3f} score={sc:9.1f} mk={markers(kp)} "
              f"MEM_STORE={mst:.1f} CLEAN_LO={cli}/{float(clo[cli]):.2f} "
              f"MEM_VAL_B={['%.1f'%v for v in mvb]}")

    print(f"  --- per-slot Q*K to each store VALUE row ---")
    Wq_h0 = Wq[0:hd]
    Wk_h0 = Wk[0:hd]
    for mp, vr in store_rows:
        addr_lo = pre[mp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
        addr_lo_i = int(addr_lo.argmax().item())
        val_b0 = pre[vr, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
        val_i = int(val_b0.argmax().item()) if float(val_b0.max()) > 0.3 else -1
        qslot = x[b0row] @ Wq_h0.T
        kslot = x[vr] @ Wk_h0.T
        prod = qslot * kslot
        tot = float(prod.sum().item()) / (hd ** 0.5)
        mark = " <<WIN" if vr == int(topk.indices[0].item()) else ""
        print(f"    store@{mp}(addrlo={addr_lo_i:x},val={val_i:x}) "
              f"vrow={vr} score={tot:.0f}{mark}")

    print(f"  --- MEM store frames (bytes hi|lo per offset) + ADDR_KEY ---")
    for kp in mem_pos:
        mst = float(pre[kp, dimpos["MEM_STORE"]].item())
        if mst < 0.5:
            continue
        row = []
        for d in range(0, 9):
            pp = kp + d
            if pp >= seq:
                break
            clo = pre[pp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            chi = pre[pp, dimpos["CLEAN_EMBED_HI"]:dimpos["CLEAN_EMBED_HI"]+16]
            lo = int(clo.argmax().item()) if float(clo.max()) > 0.3 else 0
            hi = int(chi.argmax().item()) if float(chi.max()) > 0.3 else 0
            row.append(f"{hi:x}{lo:x}")
        ak = pre[kp, dimpos["ADDR_KEY"]:dimpos["ADDR_KEY"]+16] if "ADDR_KEY" in dimpos else []
        print(f"    MEM@{kp}: bytes(hi|lo)={row} "
              f"ADDR_KEY=[{','.join('%.1f'%v for v in ak[:8])}]")

    # winner row full nonzero dump
    winner = int(topk.indices[0].item())
    print(f"  --- WINNER K@{winner} nonzero dims (|v|>0.4) ---")
    nz = torch.nonzero(pre[winner].abs() > 0.4).flatten().tolist()
    print("    " + ", ".join(
        f"{rev.get(d, d)}={float(pre[winner, d]):.2f}" for d in nz[:50]))


if __name__ == "__main__":
    main()
