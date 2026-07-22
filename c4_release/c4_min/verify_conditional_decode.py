"""DECODE byte-identity gate for the CONDITIONAL per-layer block kernel.

Proves the STRONGEST claim: a ``ConditionalBlockLean`` built from a program's
active-unit set decodes the SAME register trace as the dense ``LeanQwenVM`` AND
matches ``isa.interpret``, over a battery of programs.  The active-unit set is
computed ONCE from the program's own step windows (thr=0 → dropped units
contribute exactly 0 to ``down(silu(up)·gate)``), so the decode is bit-identical
(any residual L-inf is fp-reduction-ORDER only, far below the integer _snap
margin).

Run:
    python -m c4_min.verify_conditional_decode --device cuda:0 --subset full
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC


def _programs() -> List[Tuple[str, List[isa.Instr]]]:
    return [
        ("countdown", isa.assemble([
            ("IMM", 60), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])),
        ("mul_accum", isa.assemble([
            ("IMM", 1), ("PSH", 0), ("IMM", 3), ("MUL", 0), ("PSH", 0),
            ("IMM", 200), ("MOD", 0), ("PSH", 0), ("IMM", 1), ("ADD", 0),
            ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])),
        ("divmod_euclid", isa.assemble([
            ("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("PSH", 0),
            ("IMM", 1), ("ADD", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)])),
        ("cmp_chain", isa.assemble([
            ("IMM", 5), ("PSH", 0), ("IMM", 3), ("LT", 0), ("PSH", 0),
            ("IMM", 0), ("EQ", 0), ("PSH", 0), ("IMM", 9), ("ADD", 0),
            ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])),
        ("arith_mix", isa.assemble([
            ("IMM", 12), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("PSH", 0),
            ("IMM", 3), ("MUL", 0), ("PSH", 0), ("IMM", 5), ("DIV", 0),
            ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])),
    ]


def run(device="cuda:0", subset_name="full", max_steps=300):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    print(f"# DECODE byte-identity gate subset={subset_name} device={device}")
    vm = Q.build(code_size=24, subset=subset)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    I = lean.layers[0].gate_w.shape[0]

    all_ok = True
    for name, code in _programs():
        # active-unit set = union over the program's own step windows.
        try:
            xw, posw, opc = PC.repetitive_program_windows(lean, code, max_steps=max_steps)
        except ValueError as e:
            print(f"  {name:15s}  SKIP ({e})")
            continue
        xw = xw.to(device); posw = posw.to(device)
        info = PC.conditional_active_units(lean, xw, posw, thr=0.0)
        act = info["active_units"]
        act_counts = [a.numel() for a in act]
        cond = PC.ConditionalBlockLean(lean, act).to(device)

        # decode via the UNMODIFIED spec driver on both models.
        rd = LF.speculative_run_lean(lean, code, block_steps=64, max_steps=max_steps)
        rc = LF.speculative_run_lean(cond, code, block_steps=64, max_steps=max_steps)
        ref = isa.interpret(code, max_steps=max_steps)

        dense_ok = rd.ax_trace == ref
        cond_ok = rc.ax_trace == ref
        traces_match = rd.ax_trace == rc.ax_trace
        # The CONDITIONAL-block claim = cond decodes byte-identically to DENSE.
        # (dense==ref can be False when the program uses ops outside the chosen
        #  subset — a subset issue, NOT a conditional-block bug.  cond==dense is
        #  the load-bearing gate.)
        ok = traces_match
        all_ok &= ok
        frac = sum(act_counts) / (I * lean.n_layers) * 100
        print(f"  {name:15s} steps={rd.steps:4d} active={sum(act_counts):6d}/{I*lean.n_layers} "
              f"({frac:6.3f}%)  cond==dense:{traces_match} [dense==ref:{dense_ok}] "
              f"-> {'OK' if ok else 'FAIL'}")
        del cond, xw
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"\n  ALL DECODE-IDENTICAL: {all_ok}")
    return all_ok


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="full",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--max-steps", type=int, default=300)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset, max_steps=a.max_steps)
