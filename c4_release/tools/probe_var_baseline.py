#!/usr/bin/env python3
"""Diagnostic probe: re-attribute the var stack-store baseline failure.

spec_k=0, hook-free. Uses tools/probe_groundtruth.py.

Step 1: replay var_simple_12 (id 262, single-byte x=28) at spec_k=0 and decode
        the neural per-step register bytes from the emitted token stream.
Step 2: build the symbolic oracle per-step trace (ground truth).
Step 3: find the FIRST diverging (step, register, byte) -> pins STORE vs LOAD.
Step 4: residual_at sweep across all 37 physical blocks at the divergent
        prediction row to find where the byte first goes wrong.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import E, Opcode  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.verification.symbolic_program import (  # noqa: E402
    SymbolicDeclarativeProgramRunner,
)

OPNAMES = {int(getattr(Opcode, n)): n for n in dir(Opcode)
           if not n.startswith("_") and isinstance(getattr(Opcode, n), int)}

MARKERS = {
    int(Token.REG_PC): "PC",
    int(Token.REG_AX): "AX",
    int(Token.REG_SP): "SP",
    int(Token.REG_BP): "BP",
    int(Token.STEP_END): "STEP_END",
    int(Token.HALT): "HALT",
}


def decode_neural_steps(ctx, prompt_len):
    """Parse the emitted token stream into per-step {reg: 4-byte-value} dicts.

    Each step emits markers REG_PC/AX/SP/BP each followed by 4 LE byte tokens,
    terminated by STEP_END. Returns list of dicts keyed PC/AX/SP/BP, each a
    4-byte tuple (b0,b1,b2,b3) plus int value.
    """
    steps = []
    cur = {}
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
            i += 5
        elif name == "STEP_END":
            steps.append(cur)
            cur = {}
            i += 1
        elif name == "HALT":
            if cur:
                steps.append(cur)
            break
        else:
            i += 1
    if cur and (not steps or cur is not steps[-1]):
        steps.append(cur)
    return steps


def to_bytes4(v):
    v &= 0xFFFFFFFF
    return tuple((v >> (8 * j)) & 0xFF for j in range(4))


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    print(f"=== id={idx} {desc} ===")
    print(f"src={src!r} expected={exp}")
    print("bytecode:")
    for j, w in enumerate(bc):
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 23):
            imm -= (1 << 24)
        print(f"  [{j}] pc={j*8:3d} {OPNAMES.get(op,'?'):6s} imm={imm}")

    # Symbolic ground-truth per-step trace
    runner = SymbolicDeclarativeProgramRunner()
    st = runner.run(list(bc), data, max_steps=2000)
    print(f"\n=== SYMBOLIC oracle: exit={st.ax & 0xFFFFFFFF} steps={st.steps} ===")
    print(f"{'step':>4} {'op':6s} {'pc_aft':>6} {'ax_aft':>8} {'sp_aft':>10} "
          f"{'bp_aft':>10} {'mem_addr':>10} {'mem_val':>8}")
    for tr in st.trace:
        print(f"{tr.step:>4} {tr.name:6s} {tr.pc_after:>6} {tr.ax_after:>8} "
              f"{tr.sp_after:>10} {tr.bp_after:>10} {tr.mem_addr:>10} "
              f"{tr.mem_value:>8}")

    # Neural replay (spec_k=0)
    probe = build_groundtruth_probe()
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    print(f"\nprompt_len={prompt_len} total_ctx={len(ctx)}")
    nsteps = decode_neural_steps(ctx, prompt_len)
    print(f"\n=== NEURAL emitted per-step (spec_k=0) ===")
    print(f"{'nstep':>5} {'PC':>10} {'AX':>10} {'SP':>12} {'BP':>12}")
    for si, s in enumerate(nsteps):
        def f(k):
            return s[k][0] if k in s else None
        print(f"{si:>5} {str(f('PC')):>10} {str(f('AX')):>10} "
              f"{str(f('SP')):>12} {str(f('BP')):>12}")

    # Align neural steps to symbolic trace by step index and find first divergence
    print(f"\n=== FIRST DIVERGENCE (neural vs symbolic, after-state) ===")
    first = None
    for si, tr in enumerate(st.trace):
        if si >= len(nsteps):
            break
        ns = nsteps[si]
        expected = {
            "PC": tr.pc_after, "AX": tr.ax_after,
            "SP": tr.sp_after, "BP": tr.bp_after,
        }
        for reg in ("PC", "AX", "SP", "BP"):
            if reg not in ns:
                continue
            got_val, got_bytes = ns[reg]
            exp_bytes = to_bytes4(expected[reg])
            for bi in range(4):
                if got_bytes[bi] != exp_bytes[bi]:
                    line = (f"step={si} op={tr.name} reg={reg} byte{bi}: "
                            f"expected=0x{exp_bytes[bi]:02x} "
                            f"neural=0x{got_bytes[bi]:02x}  "
                            f"(exp_val={expected[reg]} got_val={got_val})")
                    print("  " + line)
                    if first is None:
                        first = (si, tr, reg, bi, exp_bytes[bi], got_bytes[bi])
    if first is None:
        print("  no divergence found in aligned steps")
        return
    si, tr, reg, bi, eb, gb = first
    print(f"\n>>> FIRST: step={si} op={tr.name} reg={reg} byte{bi} "
          f"exp=0x{eb:02x} neural=0x{gb:02x}")

    # ---- residual sweep at the prediction row for the diverged register byte ----
    # Find the context position of the diverged byte token, probe each block.
    # Locate the marker for (step si, reg) in ctx, then byte position = marker+1+bi
    # The PREDICTION row for that byte token is the token BEFORE it (the model
    # predicts ctx[p] from row p-1).
    marker_tok = {"PC": int(Token.REG_PC), "AX": int(Token.REG_AX),
                  "SP": int(Token.REG_SP), "BP": int(Token.REG_BP)}[reg]
    # Walk to step si's marker
    step_seen = -1
    i = prompt_len
    byte_pos = None
    cur_has = set()
    while i < len(ctx):
        t = ctx[i]
        nm = MARKERS.get(t)
        if nm == "STEP_END":
            step_seen += 1
            cur_has = set()
            i += 1
            continue
        if nm in ("PC", "AX", "SP", "BP"):
            if (step_seen + 1) == si and t == marker_tok:
                byte_pos = i + 1 + bi
                break
            i += 5
            continue
        i += 1
    if byte_pos is None:
        print("  could not locate byte position for residual sweep")
        return
    pred_row = byte_pos - 1
    print(f"\nbyte token at ctx pos {byte_pos}, prediction row = {pred_row}")
    print(f"ctx around byte: {ctx[byte_pos-2:byte_pos+3]}")

    # Residual dims to read: the per-nibble RESULT/OUTPUT region. The emitted
    # byte is produced from the LM head reading the residual at pred_row. We
    # read register/mem/output dims across all 8 nibble positions is not
    # directly indexable here (probe reads one residual dim index per call on
    # the [D]=872 row). Read the high-signal scalar dims instead.
    dim_names = {
        "OPCODE": E.OPCODE, "AX_BASE": E.AX_BASE,
        "MEM_ADDR_BASE": E.MEM_ADDR_BASE, "MEM_DATA_BASE": E.MEM_DATA_BASE,
        "MEM_WRITE": E.MEM_WRITE, "MEM_READ": E.MEM_READ,
        "MEM_READY": E.MEM_READY, "RESULT": E.RESULT, "TEMP": E.TEMP,
    }
    print(f"\n=== residual_at sweep across 37 blocks @ pred_row={pred_row} ===")
    bl = probe.block_layer_map()
    print(f"{'phys':>4} {'log':>3}  " + "  ".join(f"{k:>12}" for k in dim_names))
    for phys in range(len(probe.model.blocks)):
        res = probe.residual_at(bc, block_idx=phys, position=pred_row,
                                dim_names=dim_names)
        log = bl[phys]["logical"]
        vals = "  ".join(f"{res[k]:>12.4f}" for k in dim_names)
        print(f"{phys:>4} {log:>3}  {vals}")


if __name__ == "__main__":
    main()
