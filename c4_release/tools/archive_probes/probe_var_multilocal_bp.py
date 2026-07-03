#!/usr/bin/env python3
"""Trace the multi-local BP-frame drift for var_mul / var_three / if_var.

Free-run decodes the program (spec_k=0, autoregressive) via the groundtruth
probe, then decodes EACH 35-token step window into its register frame
(PC/AX/SP/BP/STACK0/MEM bytes) so we can SEE:
  - the step-1 ENT establishing BP = 0x0000fff0 and whether the BP HIGH bytes
    are emitted cleanly (byte0=f0 byte1=ff byte2=00 byte3=00) or noisy;
  - cumulative BP-frame-byte noise across the 35-token steps;
  - the step-6 second-variable LEA result (should be BP-16 = 0xffe0, drift gives
    BP-8 = 0xffe8 so the two locals ALIAS).

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/probe_var_multilocal_bp.py 275
  CUDA_VISIBLE_DEVICES=0 python tools/probe_var_multilocal_bp.py 275 300 425
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token

STEP = int(Token.STEP_TOKENS)  # 35
# Per-step offsets (faithful_interpreter constants).
ROLES = {
    0: ("PC", int(Token.REG_PC)),
    5: ("AX", int(Token.REG_AX)),
    10: ("SP", int(Token.REG_SP)),
    15: ("BP", int(Token.REG_BP)),
    20: ("STACK0", None),
    25: ("MEM", None),
    34: ("STEP_END", int(Token.STEP_END)),
}


def reg4(ctx, base):
    """4 little-endian bytes -> (value, [b0,b1,b2,b3])."""
    bs = [ctx[base + 1 + j] & 0xFF for j in range(4)]
    val = sum(b << (8 * j) for j, b in enumerate(bs))
    return val, bs


def decode_frame(ctx, start):
    """Decode one 35-token step window into a dict of register (value, bytes)."""
    out = {}
    for off, (name, marker) in ROLES.items():
        if name == "STEP_END":
            out["END"] = ctx[start + off]
            continue
        mk = ctx[start + off]
        # 4 bytes following the marker
        bs = [ctx[start + off + 1 + j] & 0xFF for j in range(4)]
        val = sum(b << (8 * j) for j, b in enumerate(bs))
        out[name] = (mk, val, bs)
    return out


def oracle_trace(bc, data):
    from neural_vm.unified_compiler.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )
    r = SymbolicDeclarativeProgramRunner()
    st = r.run(list(bc), data, max_steps=40)
    rows = []
    for t in st.trace:
        rows.append(dict(step=t.step, op=t.name, pc_before=t.pc_before,
                         pc_after=t.pc_after, ax=t.ax_after, sp=t.sp_after,
                         bp=t.bp_after, mem_addr=t.mem_addr,
                         mem_val=t.mem_value))
    return rows


def trace(idx):
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, _data = compile_c(src)
    print(f"\n==== ORACLE id {idx}  {desc[:48]} ====")
    for r in oracle_trace(bc, _data):
        print(f"  ostep{r['step']:2d} {r['op']:6s} pc={r['pc_before']:#06x}->"
              f"{r['pc_after']:#06x} AX={r['ax']:#06x} SP={r['sp']:#010x} "
              f"BP={r['bp']:#010x} memaddr={r['mem_addr']:#06x} "
              f"memval={r['mem_val']:#06x}")
    p = build_groundtruth_probe()
    # Free-run decode (autoregressive spec_k=0).
    plen = len(p._build_context(bc))
    # Use a generous step budget so we see step-6+.
    recs = p.probe(bc, max_steps=14, top_k=3)
    # Rebuild the full emitted context from records (positions are absolute).
    # probe() appends each token; reconstruct via the final ctx it built:
    ctx = p._final_context(bc, max_steps=14)
    n_emitted = len(ctx) - plen
    n_steps = n_emitted // STEP
    print(f"\n==== id {idx}  {desc[:48]}  exp={exp} ====")
    print(f"prompt_len={plen} emitted={n_emitted} steps={n_steps}")
    for s in range(n_steps):
        base = plen + s * STEP
        fr = decode_frame(ctx, base)
        pc = fr["PC"][1]
        ax = fr["AX"][1]
        sp = fr["SP"]
        bp = fr["BP"]
        st0 = fr["STACK0"]
        end = fr["END"]
        endname = "END" if end == int(Token.STEP_END) else f"<{end}>"
        bp_bytes = bp[2]
        sp_bytes = sp[2]
        st0_bytes = st0[2]
        # flag BP high-byte noise: expected 0x0000fff0 => bytes [f0,ff,00,00]
        bp_hi_noise = (bp_bytes[2] != 0x00 or bp_bytes[3] != 0x00
                       or bp_bytes[1] not in (0xff, 0x00))
        flag = "  <-- BP-HI NOISE" if bp_hi_noise else ""
        print(f" step{s:2d} base={base:4d} PC={pc:#06x} AX={ax:#06x} "
              f"SP=0x{sp[1]:08x}{sp_bytes} BP=0x{bp[1]:08x}{bp_bytes} "
              f"STACK0=0x{st0[1]:08x}{st0_bytes} {endname}{flag}")
    return ctx, plen, n_steps


def dump_step_tokens(idx, step):
    """Print the raw 35 tokens of a given step window with role labels."""
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, _data = compile_c(src)
    p = build_groundtruth_probe()
    plen = len(p._build_context(bc))
    ctx = p._final_context(bc, max_steps=14)
    base = plen + step * STEP
    role = {0: "PC_MARK", 5: "AX_MARK", 10: "SP_MARK", 15: "BP_MARK",
            20: "STACK0_MARK", 25: "MEM_MARK", 34: "STEP_END"}
    print(f"\n--- raw tokens id{idx} step{step} base={base} ---")
    for off in range(STEP):
        tok = ctx[base + off]
        r = role.get(off, "")
        print(f"  off{off:2d} tok={tok:4d} {r}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--dump":
        dump_step_tokens(int(args[1]), int(args[2]))
    else:
        ids = [int(a) for a in args] or [275]
        for i in ids:
            trace(i)
