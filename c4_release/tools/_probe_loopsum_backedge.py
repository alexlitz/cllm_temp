#!/usr/bin/env python3
"""Loop back-edge control-flow trace for loop_sum id450.

Autoregressive-replays the model in the campaign config and dumps, for each
emitted step, the per-step PC bytes + the FULL 35-token slice (so we can see if
a step emits a short/long token count, mis-fetches, or doubles an instruction).
Compares against the DraftVM oracle per-step (pc_after, ax).

The brief's symptom: step 10 should give pc_after=106 (LEA &i in the loop
condition), but the model gives pc_after=114 (as if it executed the LI too).

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loopdesync \
       python tools/_probe_loopsum_backedge.py [pid] [lo_step] [hi_step]
"""
from __future__ import annotations
import os, sys
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa
from neural_vm.batched_pure_neural import Token, DraftVM  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa
from neural_vm.unified_compiler.symbolic_forward import decode_instr, _OPCODE_NAMES  # noqa

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}
INV_REGS = {v: k for k, v in REGS.items()}


class _St:
    def __init__(self, m): self.model = m


def bic(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_St(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, ms):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(ms * STEP):
        p = torch.tensor([ctx], dtype=torch.long, device=dev)
        l = model.forward(p)[0]
        ctx.append(int(l[len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= ms: break
    return ctx


def split_steps(ctx, pl):
    """Split the emitted stream into per-step token lists (between STEP_ENDs)."""
    out = []; cur = []
    i = pl
    while i < len(ctx):
        t = ctx[i]
        cur.append(t)
        if t == SE:
            out.append(cur); cur = []
        i += 1
    if cur:
        out.append(cur)
    return out


def decode_reg(slice_toks, regname):
    """Find regname marker in a step's token slice; return its 4 value bytes."""
    rid = INV_REGS[regname]
    for j, t in enumerate(slice_toks):
        if t == rid and j + 4 < len(slice_toks):
            return [slice_toks[j + 1 + k] & 0xff for k in range(4)]
    return None


def reg_val(slice_toks, regname):
    b = decode_reg(slice_toks, regname)
    if b is None: return None
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def oracle_trace(bc, n):
    vm = DraftVM(list(bc))
    out = []
    for st in range(n):
        if vm.halted: break
        pc_before = int(vm.pc)
        word = bc[pc_before // 8] if 0 <= pc_before // 8 < len(bc) else None
        op, _ = decode_instr(word & 0xffffffff) if word is not None else (None, None)
        nm = _OPCODE_NAMES.get(op, f"OP{op}") if op is not None else "?"
        if not vm.step(): break
        out.append((pc_before, nm, int(vm.pc) & 0xffffffff, int(vm.ax) & 0xffffffff))
        if vm.halted: break
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 450
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    plc = bic(model, bc); pl = len(plc)
    ctx = replay(model, plc, hi + 4)
    steps = split_steps(ctx, pl)
    orc = oracle_trace(bc, hi + 4)
    print(f"id{pid} {desc} expected={exp} emitted_steps={len(steps)} STEP_TOKENS={STEP}")
    print("step | ntok | model(PC,AX) | oracle(pc_before instr -> pc_after, ax) | VERDICT")
    for st in range(min(len(steps), hi + 1)):
        if st < lo: continue
        sl = steps[st]
        ntok = len(sl)
        pc = reg_val(sl, "PC"); ax = reg_val(sl, "AX")
        sp = reg_val(sl, "SP"); bp = reg_val(sl, "BP")
        if st < len(orc):
            opc_b, onm, opc_a, oax = orc[st]
            verdict = "OK" if (pc == opc_a and ax == oax) else "**DIVERGE**"
            ostr = f"{opc_b:4d}({onm:4s})->{opc_a:4d} ax={oax}"
        else:
            verdict = "??"; ostr = "(no oracle)"
        print(f"  {st:2d} | {ntok:3d}  | PC={pc} AX={ax} SP={sp} BP={bp} | {ostr} | {verdict}")
    # Dump raw token slices for the suspect window
    print("\n--- RAW token slices (suspect window) ---")
    inv_tok = {}
    for nm, v in [("PC", INV_REGS["PC"]), ("AX", INV_REGS["AX"]), ("SP", INV_REGS["SP"]),
                  ("BP", INV_REGS["BP"]), ("STACK0", INV_REGS["STACK0"]),
                  ("MEM", INV_REGS["MEM"]), ("SE", SE)]:
        inv_tok[v] = nm
    for st in range(max(lo, 0), min(len(steps), hi + 1)):
        sl = steps[st]
        annotated = []
        i = 0
        while i < len(sl):
            t = sl[i]
            if t in inv_tok and inv_tok[t] != "SE" and i + 4 < len(sl):
                vals = [sl[i + 1 + k] & 0xff for k in range(4)]
                annotated.append(f"{inv_tok[t]}={[hex(v) for v in vals]}")
                i += 5
            elif t == SE:
                annotated.append("SE"); i += 1
            else:
                annotated.append(str(t)); i += 1
        print(f"  step{st} ({len(sl)} tok): {' '.join(annotated)}")


if __name__ == "__main__":
    main()
