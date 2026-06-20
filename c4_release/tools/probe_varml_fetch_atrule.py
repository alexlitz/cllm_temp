#!/usr/bin/env python3
"""Dump the FETCH_LO / FETCH_HI band magnitudes at the LEA AX-marker row (the row
``tail_lea_local_ax_marker_byte0_e8`` fires on) for each multi-local LEA, at the
INPUT to the tail bank (~block 41 / L25). This tells us the EXACT discriminator
signature so the over-fire guard can block ALL non-imm=-8 LEAs (not just the two
the current guard hardcodes). Oracle teacher-forced context.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_varml_fetch_atrule.py
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

CASES = [
    ("var_three", "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }",
     [(2, "imm=-8 &a(legit 0xe8)"), (6, "imm=-16 &b(0xe0)"), (10, "imm=-24 &c(0xd8)")]),
    ("var_mul", "int main() { int a; int b; a = 23; b = 47; return a * b; }",
     [(2, "imm=-8 (0xe8)"), (6, "imm=-16 (0xe0)")]),
]
# tail bank is at L25 -> physical block 41. Probe a couple late blocks.
BLKS = [38, 40, 41]


def oracle_windows(bc):
    vm = DraftVM(list(bc)); toks = []
    for _ in range(40):
        if vm.halted: break
        if not vm.step(): break
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted: break
    return toks


def main():
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    flo, fhi = dp["FETCH_LO"], dp["FETCH_HI"]

    def topnibs(vec, base, n=3):
        vals = [(k, float(vec[base + k])) for k in range(16)]
        vals.sort(key=lambda x: -abs(x[1]))
        return vals[:n]

    for name, src, targets in CASES:
        bc, _ = compile_c(src)
        prompt = p._build_context(bc); pl = len(prompt)
        toks = oracle_windows(bc)
        ctx = list(prompt)
        for w in toks: ctx.extend(w)
        padded = torch.tensor([ctx], device=p._device)
        print(f"\n===== {name} =====")
        for blk in BLKS:
            with torch.no_grad():
                full = p.model.forward(padded, stop_after_block=blk)
                if full.is_sparse: full = full.to_dense()
            print(f"  -- block {blk} (FETCH at AX-marker row off+5) --")
            for ost, label in targets:
                row = pl + ost * STEP + 5
                r = full[0, row]
                lon = topnibs(r, flo); hin = topnibs(r, fhi)
                lo_s = " ".join(f"L{k}={v:+.1f}" for k, v in lon)
                hi_s = " ".join(f"H{k}={v:+.1f}" for k, v in hin)
                print(f"     ostep{ost:2d} {label:22s} FETCH_LO[{lo_s}] FETCH_HI[{hi_s}]")


if __name__ == "__main__":
    main()
