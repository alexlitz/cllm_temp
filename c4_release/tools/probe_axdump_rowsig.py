#!/usr/bin/env python3
"""Find a robust discriminator for the AX byte-2/3 dump rows (leak) vs the
working add byte-2/3 rows (emit 0 correctly).

Prints, at the LAST block, for each of the 4 AX-dump rows (marker+1..+4) of
func step-1 (ENT, leaks) and add step-0 (works), the BYTE_INDEX one-hot,
IS_BYTE, OUTPUT_LO/HI argmax, opcode flags, and AX_CARRY_OVERFLOW. The goal:
a signature that is TRUE on the leaking func byte-2/3 rows and FALSE on (a) the
working add byte-2/3 rows and (b) any row with a legitimate nonzero high byte.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
for f in ("C4_BP_SAVE_DUMP", "C4_ENT_SP_BYTE1_FF_H1_HARDEN",
          "C4_PSH_ARG_VAL_AX", "C4_AX_BYTE1_DUMP"):
    os.environ.setdefault(f, "1")
os.environ.setdefault("C4_POST_ENT_SE_SUPPRESS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)

_m, _layout = compile_full_vm_dynamic(disk_cache=True)
_DP = dict(_layout.dim_positions)

SRC_FUNC = ("int identity(int x) { return x; }\n"
            "            int main() { return identity(70); }")
SRC_ADD = "int main() { return 654 + 114; }"
# A program whose AX legitimately has nonzero byte-2 (AX >= 65536). Use a
# 32-bit value via large multiply: 1000*1000 = 1_000_000 = 0x000F4240 ->
# byte-2 = 0x0F nonzero. (mul may not be reliable; also include a direct one.)
SRC_BIG = "int main() { return 70000 + 70000; }"  # 140000 = 0x000222E0, byte2=0x02

_CTX = {}


def _markers(probe, bc, ms):
    trace = probe.probe(bc, max_steps=ms)
    RAX = int(Token.REG_AX)
    return [p for p in sorted(trace) if trace[p]["token"] == RAX], trace


@torch.no_grad()
def _pad(probe, bc, ms):
    k = (id(bc), ms)
    if k not in _CTX:
        _CTX[k] = torch.tensor([probe._final_context(bc, max_steps=ms)],
                               dtype=torch.long, device=probe._device)
    return _CTX[k]


@torch.no_grad()
def _row(probe, bc, pos, ms):
    blk = len(probe.model.blocks) - 1
    x = probe.model.forward(_pad(probe, bc, ms), stop_after_block=blk)
    r = x[0, pos]
    if r.is_sparse:
        r = r.to_dense()
    return r.float().cpu()


def _arg(r, base, n=16, thr=0.3):
    vals = [float(r[base + k]) for k in range(n)]
    m = max(vals)
    return (max(range(n), key=lambda k: vals[k]), round(m, 2)) if m > thr else (None, round(m, 2))


def _report(probe, label, bc, marker, ms):
    print(f"\n### {label} (marker {marker}) ###")
    for off in range(0, 5):
        pos = marker + off
        r = _row(probe, bc, pos, ms)
        bi = [round(float(r[_DP["BYTE_INDEX_0"] + k]), 2) for k in range(4)]
        isb = round(float(r[_DP["IS_BYTE"]]), 2)
        ol = _arg(r, _DP["OUTPUT_LO"])
        oh = _arg(r, _DP["OUTPUT_HI"])
        byte = None
        if ol[0] is not None:
            byte = ol[0] | ((oh[0] or 0) << 4)
        flags = {nm: round(float(r[_DP[nm]]), 1)
                 for nm in ("MARK_AX", "OP_ENT", "OP_JSR", "OP_IMM", "OP_ADD",
                            "AX_CARRY_OVERFLOW")
                 if nm in _DP}
        bs = f"0x{byte:02x}" if byte is not None else "--"
        print(f"  row +{off} (pos {pos}): IS_BYTE={isb} BYTE_INDEX={bi} "
              f"OL={ol} OH={oh} -> {bs}  {flags}")


def main():
    probe = GroundTruthProbe.build()
    bc_func, _ = compile_c(SRC_FUNC)
    bc_add, _ = compile_c(SRC_ADD)
    bc_big, _ = compile_c(SRC_BIG)
    fm, _ = _markers(probe, bc_func, 8)
    am, _ = _markers(probe, bc_add, 4)
    bm, _ = _markers(probe, bc_big, 4)
    _report(probe, "FUNC step-1 ENT (LEAKS 0x0a)", bc_func, fm[1], 8)
    _report(probe, "ADD step-0 (works, AX=768)", bc_add, am[0], 4)
    # Big: find the step whose AX has nonzero byte-2.
    print("\n--- BIG (140000=0x000222E0) all AX steps byte decode ---")
    for s, m in enumerate(bm):
        r1 = _row(probe, bc_big, m + 1, 4)
        print(f"  step {s} marker {m}")
    _report(probe, "BIG last AX step (byte-2 should be 0x02)", bc_big, bm[-1], 4)


if __name__ == "__main__":
    main()
