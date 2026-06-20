#!/usr/bin/env python3
"""Trace the LEA ADDRESS computation (base + offset) through the L8 pipeline at a
WRONG LEA step, in the CLEAN oracle teacher-forced context. The LEA emits the
4 AX bytes at window offsets 6,7,8,9 (predicted at rows base+5..base+8). We dump
ALU / OUTPUT / FETCH at the byte-0 predictor row (base+5) and byte-1 row
(base+6) across blocks to see whether the per-LEA OFFSET (FETCH) is applied.

var_three ostep6 wants 0xffe0 (=&a, BP-16); model emits 0xffe8 (=BP-8). The
offset that distinguishes &a from &b is +(-16) vs +(-8). This shows whether
FETCH carries the right displacement and whether L8 lea_lo adds it.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_varml_lea_addr_pipeline.py
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
    # (name, src, [(ostep, label, want_ax)])
    ("var_three", "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }",
     [(2, "LEA&a-FIRST(ok)", 0xffe8), (6, "LEA&b-WRONG", 0xffe0), (10, "LEA&c-WRONG", 0xffd8)]),
]


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

    def b(vec, name):
        x = dp[name]; i = int(torch.argmax(vec[x:x + 16])); return i, float(vec[x + i])

    def bv(vec, lon, hin):
        l, _ = b(vec, lon); h, _ = b(vec, hin); return h * 16 + l

    for name, src, targets in CASES:
        bc, _ = compile_c(src)
        prompt = p._build_context(bc); pl = len(prompt)
        toks = oracle_windows(bc)
        ctx = list(prompt)
        for w in toks: ctx.extend(w)
        padded = torch.tensor([ctx], device=p._device)
        print(f"\n===== {name} (oracle TF) =====")
        for ost, label, want in targets:
            print(f"\n--- ostep{ost} {label} want_ax=0x{want:04x} ---")
            for offf in (5, 6):  # byte0 predictor / byte1 predictor
                row = pl + ost * STEP + offf
                bprow = pl + ost * STEP + 15
                line = []
                for blk in (4, 7, 8, 11, 12, 13):
                    with torch.no_grad():
                        full = p.model.forward(padded, stop_after_block=blk)
                        if full.is_sparse: full = full.to_dense()
                    r = full[0, row]
                    alu = bv(r, "ALU_LO", "ALU_HI")
                    out = bv(r, "OUTPUT_LO", "OUTPUT_HI")
                    fetch = bv(r, "FETCH_LO", "FETCH_HI")
                    line.append(f"b{blk}:ALU{alu:02x}/OUT{out:02x}/FE{fetch:02x}")
                bpout = None
                bpout = bv(full[0, bprow], "OUTPUT_LO", "OUTPUT_HI")
                print(f"  off+{offf} (addr byte{offf-5}): " + " ".join(line)
                      + f"  [BProw.OUT={bpout:02x}]")


if __name__ == "__main__":
    main()
