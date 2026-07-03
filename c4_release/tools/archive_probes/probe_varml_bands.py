#!/usr/bin/env python3
"""Dump the frame-base (BP_marker OUTPUT), FETCH offset, ALU and OUTPUT bands at
the LEA/LI operand rows for the var-multilocal clusters, across the L8 address
pipeline (blocks 8..13), golden config.

The LEA address = frame_base(ALU, delivered by L8 head-1) + offset(FETCH). This
probe shows whether the wrong-slot bug (0xffe0 vs 0xffe8) is a wrong BASE
(head-1 attends wrong row) or a wrong OFFSET (FETCH carries the wrong local's
displacement) — and for LI whether the VALUE ever appears.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_varml_bands.py
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
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

CASES = [
    ("var_mul",    "int main() { int a; int b; a = 23; b = 47; return a * b; }", [2, 6]),
    ("var_update", "int main() { int x; x = 50; x = x + 7; return x; }", [2, 6, 8, 9]),
]


def main():
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)

    def band(vec, name):
        b = dp[name]
        lo = int(torch.argmax(vec[b:b + 16]))
        return lo, float(vec[b + lo])

    def byteval(vec, lon, hin):
        l, lv = band(vec, lon)
        h, hv = band(vec, hin)
        return h * 16 + l, lv, hv

    for name, src, steps in CASES:
        bc, _ = compile_c(src)
        pl = len(p._build_context(bc))
        ctx = p._final_context(bc, max_steps=max(steps) + 2)
        padded = torch.tensor([ctx], device=p._device)
        print(f"\n===== {name} =====")
        for st in steps:
            # off+5 = operand predictor; off+6 = AX byte0 row; off15 = BP_marker row
            for offf in (5, 6):
                op_row = pl + st * STEP + offf
                bp_row = pl + st * STEP + 15
                print(f"-- step{st} off+{offf} (op_row={op_row}) --")
                for blk in (7, 8, 11, 12, 13):
                    with torch.no_grad():
                        full = p.model.forward(padded, stop_after_block=blk)
                        if full.is_sparse:
                            full = full.to_dense()
                    r = full[0, op_row]
                    bpr = full[0, bp_row]
                    alu, _, _ = byteval(r, "ALU_LO", "ALU_HI")
                    out, _, _ = byteval(r, "OUTPUT_LO", "OUTPUT_HI")
                    fetch, flo, _ = byteval(r, "FETCH_LO", "FETCH_HI")
                    bp_out, _, _ = byteval(bpr, "OUTPUT_LO", "OUTPUT_HI")
                    print(f"   blk{blk:2d}: ALU=0x{alu:02x} OUT=0x{out:02x} "
                          f"FETCH=0x{fetch:02x}(lo+{flo:.0f}) | BP_row.OUT=0x{bp_out:02x}")


if __name__ == "__main__":
    main()
