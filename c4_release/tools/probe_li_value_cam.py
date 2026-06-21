#!/usr/bin/env python3
"""Trace L15 head 0 (LI byte-0 value lookup) CAM at a MULTI-LOCAL LI step.

Teacher-forces the clean oracle context for var_three and dumps, at the LI
step's AX-byte-0 prediction row, which K row (MEM store) head 0 attends, the
per-row ADDR_KEY signature, and the post-softmax probabilities. Reveals WHY the
CAM returns the wrong local's value (or 0) instead of mem[addr].

Run: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     python tools/probe_li_value_cam.py
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
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }"
# LI steps to inspect (osteps where an LI value should be loaded):
#   ostep 8 -> LI b? ; ostep15 -> LI a (got b's value=6); ostep18 -> LI b
LI_STEPS = [8, 15, 18, 22]


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
    dimpos = model.embed._dim_positions
    rev = {v: k for k, v in dimpos.items()}
    STEP = int(Token.STEP_TOKENS)

    # Locate L15 memory_lookup block (W_q[0, OP_LI_RELAY] huge).
    op_li = dimpos["OP_LI_RELAY"]
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

    pre = cap["pre"][0]  # (seq, dim)
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    nH, hd = attn.num_heads, attn.head_dim
    W_q, W_k, W_v = attn.W_q, attn.W_k, attn.W_v
    for w in ("W_q", "W_k", "W_v"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_dense_" + w, t.to_dense())
    Wq = getattr(attn, "_dense_W_q", W_q)
    Wk = getattr(attn, "_dense_W_k", W_k)

    x = pre
    Q = (x @ Wq.T).view(seq, nH, hd)
    K = (x @ Wk.T).view(seq, nH, hd)

    def markers(pos):
        out = []
        for nm in ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                   "MARK_STACK0"):
            if pre[pos, dimpos[nm]].item() > 0.5:
                out.append(nm.replace("MARK_", ""))
        return out

    def addr_sig(pos):
        # decode ADDR_B0_LO/HI, ADDR_B1.. one-hots as a hex addr if present
        def nib(base):
            band = pre[pos, dimpos[base]:dimpos[base] + 16]
            mx = float(band.max().item())
            return (int(band.argmax().item()), mx) if mx > 0.3 else (None, mx)
        return {b: nib(b) for b in ("ADDR_B0_LO", "ADDR_B0_HI",
                                    "ADDR_KEY")}

    mem_pos = [i for i in range(seq) if pre[i, dimpos["MARK_MEM"]].item() > 0.5]
    print(f"seq={seq} MEM marker rows: {mem_pos}")

    for si in LI_STEPS:
        if si >= len(steps):
            continue
        pc, ax = steps[si]
        base = pl + si * STEP
        b0row = base + 5  # AX byte-0 prediction row
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        print(f"\n===== ostep{si} want_ax=0x{ax:04x} got=0x{got:04x} "
              f"b0row={b0row} markers={markers(b0row)} =====")
        # operand address at the LI step (what we're looking up)
        print(f"  b0row ADDR sig: {addr_sig(b0row)}")
        opli = pre[b0row, dimpos['OP_LI_RELAY']].item()
        print(f"  OP_LI_RELAY@b0row={opli:.2f}")
        # full address nibble bands at b0row (what the CAM Q keys on)
        for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "FETCH_LO", "FETCH_HI"):
            band = pre[b0row, dimpos[nm]:dimpos[nm]+16]
            top = [(int(i), round(float(band[i]), 2)) for i in
                   torch.nonzero(band.abs() > 0.3).flatten().tolist()]
            print(f"    Q.{nm}: {top}")
        # head 0 attention
        h = 0
        qv = Q[b0row, h]
        scores = (K[:, h, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        topk = torch.topk(probs, k=6)
        for r in range(6):
            kp = int(topk.indices[r].item())
            pr = float(topk.values[r].item())
            sc = float(scores[kp].item())
            mst = float(pre[kp, dimpos["MEM_STORE"]].item())
            mvb = [float(pre[kp, dimpos[f"MEM_VAL_B{j}"]].item())
                   for j in range(4)]
            clo = pre[kp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            cli = int(clo.argmax().item())
            print(f"  K@{kp:3d} p={pr:.3f} score={sc:9.1f} mk={markers(kp)} "
                  f"MEM_STORE={mst:.1f} CLEAN_LO={cli}/{float(clo[cli]):.2f} "
                  f"MEM_VAL_B={['%.1f'%v for v in mvb]}")
        # Per-slot score breakdown: b0row Q vs candidate value rows.
        # value-byte-0 row of each MEM store = mem_pos + 5
        store_rows = [(mp, mp + 5) for mp in mem_pos
                      if float(pre[mp, dimpos["MEM_STORE"]].item()) > 0.5]
        Wq_h0 = Wq[0:hd]   # head 0 Q rows (slot indexing)
        Wk_h0 = Wk[0:hd]
        print(f"  --- per-slot Q*K to each store VALUE row (top slots) ---")
        for mp, vr in store_rows:
            addr_lo = pre[mp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            addr_lo_i = int(addr_lo.argmax().item())
            val_b0 = pre[vr, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            val_i = int(val_b0.argmax().item()) if float(val_b0.max()) > 0.3 else -1
            qslot = x[b0row] @ Wq_h0.T  # (hd,)
            kslot = x[vr] @ Wk_h0.T
            prod = qslot * kslot
            tot = float(prod.sum().item()) / (hd ** 0.5)
            top = torch.topk(prod.abs(), k=5)
            ts = " ".join(f"s{int(top.indices[j])}:{float(prod[int(top.indices[j])]):.0f}"
                          for j in range(5))
            mark = " <<WIN" if vr == int(topk.indices[0].item()) else ""
            print(f"    store@{mp}(addrlo={addr_lo_i:x},val={val_i:x}) "
                  f"vrow={vr} score={tot:.0f} [{ts}]{mark}")
        # winner row full nonzero dump
        winner = int(topk.indices[0].item())
        print(f"  --- WINNER K@{winner} nonzero dims (|v|>0.4) ---")
        nz = torch.nonzero(pre[winner].abs() > 0.4).flatten().tolist()
        print("    " + ", ".join(
            f"{rev.get(d, d)}={float(pre[winner, d]):.2f}" for d in nz[:40]))
        # Dump VALUE rows (d=5) of &a vs &b stores: what addr signal do they carry?
        if si == LI_STEPS[1]:  # ostep15 only
            for tag, mp in (("&a", 390), ("&b", 510)):
                vr = mp + 5
                print(f"  --- {tag} value row @{vr} nonzero (addr-ish dims) ---")
                for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_KEY", "FETCH_LO",
                           "FETCH_HI", "MEM_ADDR_SRC", "MEM_STORE",
                           "MEM_VAL_B0", "MEM_VAL_B1"):
                    base_d = dimpos[nm]
                    w = 16 if nm in ("ADDR_B0_LO", "ADDR_B0_HI", "FETCH_LO",
                                     "FETCH_HI") else 1
                    band = pre[vr, base_d:base_d + (16 if nm == "ADDR_KEY" else w)]
                    top = [(int(i), round(float(band[i]), 2)) for i in
                           torch.nonzero(band.abs() > 0.3).flatten().tolist()]
                    print(f"      {nm}: {top}")
        # dump MEM store rows: addr (from CLEAN at the store frame) + value byte
        print(f"  --- MEM store frames (addr@d0..3, val@d4..7) ---")
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
            ak = pre[kp, dimpos["ADDR_KEY"]:dimpos["ADDR_KEY"]+16]
            print(f"    MEM@{kp}: bytes(hi|lo)={row} "
                  f"ADDR_KEY=[{','.join('%.1f'%v for v in ak[:8])}]")


if __name__ == "__main__":
    main()
