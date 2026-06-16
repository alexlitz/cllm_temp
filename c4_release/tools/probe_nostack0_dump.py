#!/usr/bin/env python3
"""Dump the FULL emitted token stream + per-step register slices for one program
under the no-STACK0 (30-token) layout. Autoregressive spec_k=0, hook-free.

Generates the model's own tokens (no teacher forcing), then prints every step's
raw 30-token (or 35) slice with the marker positions annotated and the decoded
PC/AX/SP/BP vs the DraftVM oracle. Pinpoints the FIRST step whose frame drifts
(wrong marker token / wrong token count) — the desync origin.

Usage:
    CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 python tools/probe_nostack0_dump.py 250
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import (  # noqa: E402
    BatchedPureNeuralRunner, _step_offset_field,
)
from neural_vm.vm_step import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


def _tok(t):
    names = {256: "SEP", 257: "rPC", 258: "rAX", 259: "rSP", 260: "rBP",
             261: "MEM", 262: "SE", 263: "HALT", 268: "STK0"}
    if t in names:
        return names[t]
    if t < 256:
        return f"{t:02x}"
    return f"t{t}"


def _decode(slice_toks, marker):
    for i, tk in enumerate(slice_toks):
        if tk == marker and i + 4 < len(slice_toks):
            v = 0
            for j in range(4):
                v |= (int(slice_toks[i + 1 + j]) & 0xFF) << (j * 8)
            return v
    return None


@torch.no_grad()
def main(idx, max_steps):
    ST = Token.STEP_TOKENS
    print(f"### STEP_TOKENS={ST}")
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)

    # autoregressive generation, ST*max_steps tokens
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        logits = model.forward(padded)[0]
        nxt = int(logits[-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    emitted = context[len(prefix):]
    # DraftVM oracle
    vm = DraftVM(list(bc))
    oracle = []
    for _ in range(max_steps):
        if vm.halted or not vm.step():
            break
        oracle.append((vm.pc & 0xFFFFFFFF, vm.ax & 0xFFFFFFFF,
                       vm.sp & 0xFFFFFFFF, vm.bp & 0xFFFFFFFF))
        if vm.halted:
            break

    print(f"\n===== id={idx} {desc[:50]} exp={exp} =====")
    print(f"emitted {len(emitted)} tokens = {len(emitted)/ST:.2f} steps")
    nsteps = len(emitted) // ST
    for si in range(nsteps + 1):
        sl = emitted[si * ST:(si + 1) * ST]
        if not sl:
            break
        # annotate markers
        ann = " ".join(_tok(t) for t in sl)
        gpc = _decode(sl, Token.REG_PC)
        gax = _decode(sl, Token.REG_AX)
        gsp = _decode(sl, Token.REG_SP)
        gbp = _decode(sl, Token.REG_BP)
        o = oracle[si] if si < len(oracle) else None
        ok = ""
        if o is not None:
            opc, oax, osp, obp = o
            pc_ok = (gpc == opc)
            ok = "PCok" if pc_ok else f"PC!! exp={opc}"
        # marker offset check: are markers where the layout expects?
        mark_off = {}
        for i, t in enumerate(sl):
            if t in (257, 258, 259, 260, 261, 268):
                mark_off[i] = _tok(t)
        print(f"\n step {si} [{ok}] gPC={gpc} gAX={gax} gSP={gsp} gBP={gbp}"
              + (f"  oracle PC={o[0]} AX={o[1]} SP={o[2]} BP={o[3]}" if o else ""))
        print(f"   markers@offsets: {mark_off}")
        print(f"   {ann}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    idx = args[0] if args else 250
    ms = args[1] if len(args) > 1 else 8
    main(idx, ms)
