#!/usr/bin/env python3
"""Trace L15 head-0 LI byte-0 value-lookup CAM for var_simple #250 (x=990).

Confirms the zero-address discrimination root (#301/#318): x lives at BP+0 ->
address 0x00 (all-zero nibbles), the #313 ADDR-CAM drops the k=0 match, so an
OLD nonzero-address committed store out-scores the correct latest x-store.

Dumps, at the step-7 LI AX-byte-0 prediction row: the per-candidate-store
score, address byte-0 nibbles, MEM_VAL_B0, MEM_STORE / MEM_STORE_AT_VAL, and
position (for recency). Run in campaign config.

Run: CUDA_VISIBLE_DEVICES= C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     python tools/_probe_zeroaddr_cam.py [id]
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
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    src, exp, desc = generate_test_programs()[pid]
    print(f"id={pid} {desc} -> exp={exp}\n  {src}")

    p = build_groundtruth_probe()
    model = p.model
    dimpos = model.embed._dim_positions
    rev = {v: k for k, v in dimpos.items()}
    STEP = int(Token.STEP_TOKENS)
    print(f"STEP_TOKENS={STEP}")

    op_li = dimpos["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    print(f"L15 memory_lookup block = {L15}")

    bc, _ = compile_c(src)
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
    for w in ("W_q", "W_k"):
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
            if pre[pos, dimpos[nm]].item() > 0.5:
                out.append(nm.replace("MARK_", ""))
        return out

    def nibhex(pos, base):
        band = pre[pos, dimpos[base]:dimpos[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()) if mx > 0.3 else None, round(mx, 2))

    mem_pos = [i for i in range(seq) if pre[i, dimpos["MARK_MEM"]].item() > 0.5]
    store_rows = [mp for mp in mem_pos
                  if float(pre[mp, dimpos["MEM_STORE"]].item()) > 0.5]
    print(f"seq={seq} MEM marker rows={mem_pos}")
    print(f"committed-store marker rows (MEM_STORE>0.5)={store_rows}")

    # Identify the LI step: scan steps for the return-x LI (AX should == exp).
    for si, (pc, ax) in enumerate(steps):
        base = pl + si * STEP
        if base + 5 >= seq:
            continue
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        b0row = base + 5
        opli = pre[b0row, dimpos['OP_LI_RELAY']].item()
        tag = ""
        if opli > 0.3 or ax == exp:
            tag = "  <-- LI-ish"
        print(f"step{si:2d} want_ax=0x{ax:04x} got=0x{got:04x} "
              f"OP_LI_RELAY={opli:.2f}{tag}")

    force_step = None
    for a in sys.argv[2:]:
        if a.startswith("--step="):
            force_step = int(a.split("=", 1)[1])

    # Detailed dump for any step where AX wrong AND want==exp (the return),
    # OR the forced step.
    for si, (pc, ax) in enumerate(steps):
        base = pl + si * STEP
        if base + 5 >= seq:
            continue
        b0row = base + 5
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        if force_step is not None:
            if si != force_step:
                continue
        elif got == ax:
            continue
        opli = pre[b0row, dimpos['OP_LI_RELAY']].item()
        print(f"\n===== FAIL step{si} want=0x{ax:04x} got=0x{got:04x} "
              f"b0row={b0row} markers={markers(b0row)} OP_LI_RELAY={opli:.2f} ====")
        print(f"  Q b0row addr: ADDR_B0_LO={nibhex(b0row,'ADDR_B0_LO')} "
              f"ADDR_B0_HI={nibhex(b0row,'ADDR_B0_HI')}")
        h0 = 0
        qv = Q[b0row, h0]
        scores = (K[:, h0, :] @ qv) / (hd ** 0.5)
        scores[b0row + 1:] = float("-inf")
        probs = torch.softmax(scores, dim=0)
        topk = torch.topk(probs, k=8)
        print(f"  --- head-0 top-8 attended rows ---")
        for r in range(8):
            kp = int(topk.indices[r].item())
            pr = float(topk.values[r].item())
            sc = float(scores[kp].item())
            mvb0 = float(pre[kp, dimpos["MEM_VAL_B0"]].item())
            try:
                msav = float(pre[kp, dimpos["MEM_STORE_AT_VAL"]].item())
            except KeyError:
                msav = float('nan')
            clo = pre[kp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            cli = int(clo.argmax().item())
            print(f"   K@{kp:3d} p={pr:.3f} sc={sc:9.1f} mk={markers(kp)} "
                  f"alo={nibhex(kp,'ADDR_B0_LO')} ahi={nibhex(kp,'ADDR_B0_HI')} "
                  f"MVB0={mvb0:.2f} MSAV={msav:.2f} CLO={cli}/{float(clo[cli]):.2f}")
        # per committed-store-value-row breakdown (value row = mp+5)
        print(f"  --- committed store VALUE rows (vrow=mp+5) score breakdown ---")
        winner = int(topk.indices[0].item())
        Wq_h0 = Wq[0:hd]
        Wk_h0 = Wk[0:hd]
        for mp in store_rows:
            vr = mp + 5
            if vr > b0row:
                continue
            qslot = x[b0row] @ Wq_h0.T
            kslot = x[vr] @ Wk_h0.T
            prod = qslot * kslot
            tot = float(prod.sum().item()) / (hd ** 0.5)
            mvb0 = float(pre[vr, dimpos["MEM_VAL_B0"]].item())
            try:
                msav = float(pre[vr, dimpos["MEM_STORE_AT_VAL"]].item())
            except KeyError:
                msav = float('nan')
            top = torch.topk(prod.abs(), k=6)
            ts = " ".join(
                f"s{int(top.indices[j])}:{float(prod[int(top.indices[j])]):.0f}"
                for j in range(6))
            mark = " <<WINNER-ROW" if vr == winner else ""
            # address at MARKER row (mp) and value byte at vrow
            valclo = pre[vr, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
            valhi = pre[vr, dimpos["CLEAN_EMBED_HI"]:dimpos["CLEAN_EMBED_HI"]+16]
            vlo = int(valclo.argmax().item()) if float(valclo.max()) > 0.3 else 0
            vhi = int(valhi.argmax().item()) if float(valhi.max()) > 0.3 else 0
            print(f"   store@{mp} (pos={mp}) vrow={vr} markerADDR_lo="
                  f"{nibhex(mp,'ADDR_B0_LO')} markerADDR_hi={nibhex(mp,'ADDR_B0_HI')} "
                  f"vrowADDR_lo={nibhex(vr,'ADDR_B0_LO')} vrowADDR_hi={nibhex(vr,'ADDR_B0_HI')} "
                  f"valbyte=0x{vhi:x}{vlo:x} MVB0={mvb0:.2f} MSAV={msav:.2f} "
                  f"score={tot:.0f}{mark}\n      slots[{ts}]")
        # recency dim check: dump what dim varies with position on value rows
        print(f"  --- recency/position signals on store value rows ---")
        for cand in ("STEP_ID", "POS_ID", "POSITION", "NEXT_MEM", "MEM_AGE",
                     "RECENCY", "TIME"):
            if cand in dimpos:
                vals = [round(float(pre[mp + 5, dimpos[cand]].item()), 3)
                        for mp in store_rows if mp + 5 <= b0row]
                print(f"    {cand}: {vals}")
        break


if __name__ == "__main__":
    main()
