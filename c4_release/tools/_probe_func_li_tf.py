#!/usr/bin/env python3
"""TEACHER-FORCED func PARAM LI value-load probe (the upstream blocker).

interp_oracle_gate flags func_max/min/square/add as CROSS-STEP at step 9 AX[0]
= the LI that loads the first PARAM's value. That LI is UPSTREAM of the &b
re-read LEA (step 11) the keystone amplifier targets, so the LEA fix is never
reached. This probe dumps, at each func param-LI step, the L15 head-0 byte-0
lookup attention + every MEM store VALUE row's address/value signature, to see
WHY the param LI returns 0x00 / the wrong value in the campaign config -- i.e.
whether the var LI CAM (#313 ADDR_B0 + #318 MEM_VAL_B0) reaches the func param
store rows, or the param-delivery PSH-materializer produces a value row the CAM
cannot content-address.

Run inside campaign env:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
  C4_VM_CACHE_DIR=/tmp/c4cache_multilea python tools/_probe_func_li_tf.py
"""
from __future__ import annotations
import os, sys, contextlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()
OPNAME = {0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ", 6: "ENT",
          7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI", 12: "SC", 13: "PSH",
          20: "GT", 19: "LT", 27: "MUL", 0x26: "EXIT"}


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    rev = {v: k for k, v in dimpos.items()}
    nblocks = len(model.blocks)

    # Locate L15 memory_lookup block (W_q[0, OP_LI_RELAY] huge).
    op_li = dimpos["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    print(f"nblocks={nblocks} L15_lookup_block={L15} STEP={STEP} d={model.d_model}")

    cases = [("func_square_625", 625), ("func_max_650", 650),
             ("func_min_675", 675), ("func_add_575", 575)]

    for label, pid in cases:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        ot = oracle_tape_and_steps(bc, b"", max_steps=40)
        tape = list(prompt) + list(ot.draft_tokens)
        tok = torch.tensor([tape], dtype=torch.long)
        plen = len(prompt)

        cap = {}
        h = model.blocks[L15].register_forward_pre_hook(
            lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                logits = model.forward(tok)
        h.remove()
        if logits.is_sparse:
            logits = logits.to_dense()
        preds = torch.argmax(logits[0], dim=-1)
        pre = cap["pre"][0]
        seq = pre.shape[0]
        attn = model.blocks[L15].attn
        nH, hd = attn.num_heads, attn.head_dim
        for w in ("W_q", "W_k"):
            t = getattr(attn, w)
            if t.is_sparse_csr or t.is_sparse:
                setattr(attn, "_d_" + w, t.to_dense())
        Wq = getattr(attn, "_d_W_q", attn.W_q)
        Wk = getattr(attn, "_d_W_k", attn.W_k)
        x = pre
        Q = (x @ Wq.T).view(seq, nH, hd)
        K = (x @ Wk.T).view(seq, nH, hd)

        def markers(pos):
            return [nm.replace("MARK_", "") for nm in
                    ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                     "MARK_STACK0") if pre[pos, dimpos[nm]].item() > 0.5]

        mem_pos = [i for i in range(seq)
                   if pre[i, dimpos["MARK_MEM"]].item() > 0.5]
        store_rows = [mp for mp in mem_pos
                      if float(pre[mp, dimpos["MEM_STORE"]].item()) > 0.5]
        print(f"\n===== {label} exp={exp} seq={seq} =====")
        print(f"  opcodes: {[OPNAME.get(o, hex(o)) for o in ot.opcodes]}")
        print(f"  MEM store rows: {store_rows}")

        # Find the LI steps
        for step, op in enumerate(ot.opcodes):
            if op != 9:  # LI
                continue
            base = plen + step * STEP
            b0row = base + 5
            if b0row + 4 > seq:
                break
            pc, ax = ot.steps[step] if step < len(ot.steps) else (-1, -1)
            emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
            got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
            opli = float(pre[b0row, dimpos['OP_LI_RELAY']].item())
            flag = "OK" if got == ax else "**WRONG**"
            print(f"\n  -- LI step{step} want_ax=0x{ax:04x} got=0x{got:04x} "
                  f"{flag} OP_LI_RELAY={opli:.1f} markers={markers(b0row)} --")
            # Q address nibbles at the lookup row
            for nm in ("ADDR_B0_LO", "ADDR_B0_HI"):
                band = pre[b0row, dimpos[nm]:dimpos[nm]+16]
                top = [(int(i), round(float(band[i]), 2)) for i in
                       torch.nonzero(band.abs() > 0.3).flatten().tolist()]
                print(f"     Q.{nm}: {top}")
            # head-0 attention winners
            qv = Q[b0row, 0]
            scores = (K[:, 0, :] @ qv) / (hd ** 0.5)
            scores[b0row + 1:] = float("-inf")
            probs = torch.softmax(scores, dim=0)
            topk = torch.topk(probs, k=5)
            for r in range(5):
                kp = int(topk.indices[r].item())
                pr = float(topk.values[r].item())
                mvb = [round(float(pre[kp, dimpos[f"MEM_VAL_B{j}"]].item()), 1)
                       for j in range(2)]
                clo = pre[kp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
                cli = int(clo.argmax().item())
                alo = pre[kp, dimpos["ADDR_B0_LO"]:dimpos["ADDR_B0_LO"]+16]
                ali = int(alo.argmax().item()) if float(alo.max()) > 0.3 else -1
                print(f"     K@{kp:3d} p={pr:.3f} mk={markers(kp)} "
                      f"CLEAN_LO={cli} ADDR_B0_LO={ali} MEM_VAL_B={mvb}")
            # dump each store value row's signature
            print(f"     -- store VALUE rows (addr signature + value byte) --")
            for mp in store_rows:
                vr = mp + 5
                if vr >= seq:
                    continue
                alo = pre[vr, dimpos["ADDR_B0_LO"]:dimpos["ADDR_B0_LO"]+16]
                ali = int(alo.argmax().item()) if float(alo.max()) > 0.3 else -1
                ahi = pre[vr, dimpos["ADDR_B0_HI"]:dimpos["ADDR_B0_HI"]+16]
                ahi_i = int(ahi.argmax().item()) if float(ahi.max()) > 0.3 else -1
                mvb0 = float(pre[vr, dimpos["MEM_VAL_B0"]].item())
                clo = pre[vr, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
                cli = int(clo.argmax().item()) if float(clo.max()) > 0.3 else -1
                print(f"       store@{mp} vrow={vr} ADDR_B0[hi={ahi_i:x},lo="
                      f"{ali:x}] MEM_VAL_B0={mvb0:.1f} CLEAN_LO={cli:x}")


if __name__ == "__main__":
    main()
