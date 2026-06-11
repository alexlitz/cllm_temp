#!/usr/bin/env python3
"""FULL-CHAIN divergence map for var_simple_12 (id 262, `int x; x=28; return x;`).

Walks the WHOLE program at spec_k=0, hook-free, in a single model build:
  1. Replay neural per-step register trace (PC/AX/SP/BP, 4 LE bytes each).
  2. Build the symbolic oracle per-step trace (ground truth, incl. mem ops).
  3. Enumerate EVERY (step, reg, byte) divergence (not just the first).
  4. For each divergence's prediction row: residual genesis sweep across all
     37 physical blocks (OUTPUT_LO/HI nibble decode + key marker/opcode dims)
     to pin the genesis block / logical layer; plus the LM-head top-token.

READ-ONLY. No weight edits, no hooks. Dims via probe.model.dim_positions.

Usage:  python tools/probe_var_full_chain.py [id]   (default 262)
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.unified_compiler.symbolic_program import (  # noqa: E402
    SymbolicDeclarativeProgramRunner,
)

OPNAMES = {int(getattr(Opcode, n)): n for n in dir(Opcode)
           if not n.startswith("_") and isinstance(getattr(Opcode, n), int)}
MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}


def to_bytes4(v):
    v &= 0xFFFFFFFF
    return tuple((v >> (8 * j)) & 0xFF for j in range(4))


def decode_neural_steps(ctx, prompt_len):
    steps = []
    cur = {}
    cur_pos = {}
    i = prompt_len
    n = len(ctx)
    while i < n:
        t = ctx[i]
        name = MARKERS.get(t)
        if name in ("PC", "AX", "SP", "BP"):
            if i + 4 < n:
                bs = tuple(ctx[i + 1 + j] & 0xFF for j in range(4))
                val = sum(bs[j] << (8 * j) for j in range(4))
                cur[name] = (val, bs)
                cur_pos[name] = i
            i += 5
        elif name == "STEP_END":
            steps.append((cur, cur_pos))
            cur, cur_pos = {}, {}
            i += 1
        elif name == "HALT":
            if cur:
                steps.append((cur, cur_pos))
            break
        else:
            i += 1
    if cur and (not steps or steps[-1][0] is not cur):
        steps.append((cur, cur_pos))
    return steps


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    print(f"=== id={idx} {desc} ===", flush=True)
    print(f"src={src!r} expected={exp}")
    print("bytecode:")
    for j, w in enumerate(bc):
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 23):
            imm -= (1 << 24)
        print(f"  [{j}] pc={j*8:3d} {OPNAMES.get(op,'?'):6s} imm={imm}")

    sym = SymbolicDeclarativeProgramRunner()
    st = sym.run(list(bc), data, max_steps=2000)
    print(f"\n=== SYMBOLIC oracle: exit={st.ax & 0xFFFFFFFF} steps={st.steps} ===")
    print(f"{'st':>3} {'op':6s} {'pc_aft':>6} {'ax_aft':>10} {'sp_aft':>10} "
          f"{'bp_aft':>10} {'mem_addr':>10} {'mem_val':>8}")
    for tr in st.trace:
        print(f"{tr.step:>3} {tr.name:6s} {tr.pc_after:>6} {tr.ax_after:>10} "
              f"{tr.sp_after:>10} {tr.bp_after:>10} {tr.mem_addr:>10} "
              f"{tr.mem_value:>8}")

    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    bl = probe.block_layer_map()
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    out, code = probe.emitted_result(bc)
    print(f"\nprompt_len={prompt_len} total_ctx={len(ctx)} "
          f"neural_exit={code} expected={exp}", flush=True)

    nsteps = decode_neural_steps(ctx, prompt_len)
    print(f"\n=== NEURAL emitted per-step (spec_k=0) ===")
    print(f"{'st':>3} {'PC':>10} {'AX':>10} {'SP':>12} {'BP':>12}")
    for si, (s, _p) in enumerate(nsteps):
        def f(k):
            return s[k][0] if k in s else None
        print(f"{si:>3} {str(f('PC')):>10} {str(f('AX')):>10} "
              f"{str(f('SP')):>12} {str(f('BP')):>12}")

    print(f"\n=== ALL DIVERGENCES (neural vs symbolic after-state) ===", flush=True)
    divs = []
    for si, tr in enumerate(st.trace):
        if si >= len(nsteps):
            break
        ns, npos = nsteps[si]
        expected = {"PC": tr.pc_after, "AX": tr.ax_after,
                    "SP": tr.sp_after, "BP": tr.bp_after}
        for reg in ("PC", "AX", "SP", "BP"):
            if reg not in ns:
                continue
            got_val, got_bytes = ns[reg]
            exp_bytes = to_bytes4(expected[reg])
            for bi in range(4):
                if got_bytes[bi] != exp_bytes[bi]:
                    divs.append((si, tr.name, reg, bi, exp_bytes[bi],
                                 got_bytes[bi], npos[reg],
                                 expected[reg], got_val))
    if not divs:
        print("  NO DIVERGENCES — program is correct!")
    for (si, opn, reg, bi, eb, gb, mpos, ev, gv) in divs:
        print(f"  step={si} op={opn} {reg} byte{bi}: exp=0x{eb:02x} "
              f"neu=0x{gb:02x}  (exp_val={ev:#x} got_val={gv:#x})")

    LO = dp["OUTPUT_LO"]
    HI = dp["OUTPUT_HI_THIS_STEP"]
    ALO = dp["ALU_LO"]
    AHI = dp["ALU_HI"]

    def nib_decode(row, base):
        v = [float(row[base + k]) for k in range(16)]
        a = max(range(16), key=lambda k: v[k])
        return a, v[a]

    def marks(row):
        out = {}
        for nm in ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
                   "MARK_MEM", "IS_BYTE", "HAS_SE", "OP_LEA", "OP_ENT",
                   "OP_JSR", "OP_LEV", "OP_SI", "OP_LI", "MEM_STORE",
                   "MEM_READ", "MEM_WRITE", "CMP+7"):
            if "+" in nm:
                base, off = nm.rsplit("+", 1)
                out[nm] = float(row[dp[base] + int(off)]) if base in dp else float("nan")
            else:
                out[nm] = float(row[dp[nm]]) if nm in dp else float("nan")
        return out

    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    pred_rows = []
    for (si, opn, reg, bi, eb, gb, mpos, ev, gv) in divs:
        pred_rows.append(mpos + 1 + bi - 1)
    pred_rows = sorted(set(pred_rows))

    print(f"\n=== per-block residual cache for {len(pred_rows)} prediction rows ===",
          flush=True)
    cache = {}
    with torch.no_grad():
        for phys in range(len(m.blocks)):
            r = m.forward(padded, stop_after_block=phys)[0].float()
            cache[phys] = {pr: r[pr].clone() for pr in pred_rows}
        # final logits
        final_logits = m.forward(padded)[0].float()

    for (si, opn, reg, bi, eb, gb, mpos, ev, gv) in divs:
        byte_pos = mpos + 1 + bi
        pred_row = byte_pos - 1
        print(f"\n############ DIVERGENCE step={si} op={opn} {reg} byte{bi} "
              f"exp=0x{eb:02x} neu=0x{gb:02x}  pred_row={pred_row} "
              f"byte_pos={byte_pos} ############", flush=True)
        print(f"ctx around byte: {ctx[byte_pos-2:byte_pos+3]}")
        logits = final_logits[pred_row]
        topk = torch.topk(logits, 6)
        tops = [(int(t), round(float(v), 2))
                for v, t in zip(topk.values.tolist(), topk.indices.tolist())]
        print(f"LM-head top6 @pred_row: {tops}")
        print(f"  exp-byte token 0x{eb:02x} logit={float(logits[eb]):.3f}  "
              f"neu-byte token 0x{gb:02x} logit={float(logits[gb]):.3f}")
        print(f"{'phys':>4}{'log':>4} {'OUTdec':>7} {'LOnib':>6}{'LOv':>9} "
              f"{'HInib':>6}{'HIv':>9} {'ALUdec':>7} | mAX mBP mSP mST mMM ISB "
              f"HSE LEA ENT JSR LEV SI LI")
        for phys in range(len(m.blocks)):
            row = cache[phys][pred_row]
            la, lv = nib_decode(row, LO)
            ha, hv = nib_decode(row, HI)
            ala, _ = nib_decode(row, ALO)
            aha, _ = nib_decode(row, AHI)
            byte = (ha << 4) | la
            ab = (aha << 4) | ala
            mk = marks(row)
            mstr = (f"{mk['MARK_AX']:4.1f}{mk['MARK_BP']:4.1f}"
                    f"{mk['MARK_SP']:4.1f}{mk['MARK_STACK0']:4.1f}"
                    f"{mk['MARK_MEM']:4.1f}{mk['IS_BYTE']:4.1f}"
                    f"{mk['HAS_SE']:4.1f}{mk['OP_LEA']:4.1f}{mk['OP_ENT']:4.1f}"
                    f"{mk['OP_JSR']:4.1f}{mk['OP_LEV']:4.1f}{mk['OP_SI']:4.1f}"
                    f"{mk['OP_LI']:4.1f}")
            print(f"{phys:>4}{bl[phys]['logical']:>4} 0x{byte:02x}    "
                  f"{la:>5}{lv:>9.2f} {ha:>5}{hv:>9.2f} 0x{ab:02x}    | {mstr}")


if __name__ == "__main__":
    main()
