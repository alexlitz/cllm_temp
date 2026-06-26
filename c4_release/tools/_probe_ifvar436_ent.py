#!/usr/bin/env python3
"""Localize the if_var id436 step-1 ENT AX-dump leak (744 = 0x2E8) in the
GOLDEN config. The oracle AX=0 on the ENT step; the model emits 0xE8 (byte-0)
+ 0x02 (byte-1). Suspect: tail_lea_local_ax_marker_byte0_e8 (0xE8) +
tail_ax_add_byte1_missing_stack_high_02 (0x02). Report the row signature for
each AX-marker byte row on the ENT step, plus a comparison against a clean
var_simple ENT step (which decodes AX=0 correctly) and a genuine LEA step.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"

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

SRC_436 = "int main() { int x; x = 66; if (x > 24) return 1; return 0; }"
# var_simple: single local ENT-8, AX should carry 0 across the ENT step too.
SRC_VAR = "int main() { int x; x = 5; return x; }"

_CTX = {}


@torch.no_grad()
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
    if base not in _DP:
        return (None, 0.0)
    b = _DP[base]
    vals = [float(r[b + k]) for k in range(n)]
    m = max(vals)
    return (max(range(n), key=lambda k: vals[k]), round(m, 2)) if m > thr else (None, round(m, 2))


def _g(r, nm):
    if "+" in nm:
        base, off = nm.split("+")
        return float(r[_DP[base] + int(off)]) if base in _DP else None
    return float(r[_DP[nm]]) if nm in _DP else None


def _report(probe, label, bc, marker, ms):
    print(f"\n### {label} (marker {marker}) ###")
    for off in range(0, 5):
        pos = marker + off
        r = _row(probe, bc, pos, ms)
        bi = [round(float(r[_DP["BYTE_INDEX_0"] + k]), 2) for k in range(4)]
        isb = round(float(r[_DP["IS_BYTE"]]), 2)
        ol = _arg(r, "OUTPUT_LO")
        oh = _arg(r, "OUTPUT_HI")
        byte = None
        if ol[0] is not None:
            byte = ol[0] | ((oh[0] or 0) << 4)
        bs = f"0x{byte:02x}" if byte is not None else "--"
        # gate signatures for the two suspect rules
        e8_gate = {nm: round(v, 2) for nm in (
            "MARK_AX", "OP_LEA", "FETCH_LO+8", "MEM_ADDR_SRC", "OP_ENT",
            "OPCODE_BYTE_LO+6", "OPCODE_BYTE_LO+0", "OPCODE_BYTE_HI+0",
            "AX_CARRY_LO+0", "AX_CARRY_HI+0")
            if (v := _g(r, nm)) is not None}
        b1_02_gate = {nm: round(v, 2) for nm in (
            "IS_BYTE", "HAS_SE", "H1+1", "TEMP+8", "TEMP+9", "CARRY+1",
            "FETCH_HI+1", "BYTE_INDEX_0", "BYTE_INDEX_1")
            if (v := _g(r, nm)) is not None}
        print(f"  row +{off} (pos {pos}): IS_BYTE={isb} BYTE_INDEX={bi} "
              f"OL={ol} OH={oh} -> {bs}")
        print(f"      e8_gate={e8_gate}")
        print(f"      b1_02_gate={b1_02_gate}")


def main():
    probe = GroundTruthProbe.build()
    bc436, _ = compile_c(SRC_436)
    bcvar, _ = compile_c(SRC_VAR)
    m436, _ = _markers(probe, bc436, 14)
    mvar, _ = _markers(probe, bcvar, 8)
    print("id436 AX markers:", m436)
    print("var AX markers:", mvar)
    # step 1 is ENT (the divergence). marker index 1.
    _report(probe, "id436 step-1 ENT (LEAKS 744=0x2E8)", bc436, m436[1], 14)
    # step 2 is a GENUINE LEA -8 (AX should = 0xFFE8). The 0xE8 writer SHOULD
    # fire here. Confirms OP_ENT discriminates ENT (leak) from LEA (genuine).
    _report(probe, "id436 step-2 LEA -8 (GENUINE 0xFFE8)", bc436, m436[2], 14)
    if len(mvar) > 1:
        _report(probe, "var_simple x=5 step-1 ENT", bcvar, mvar[1], 8)


if __name__ == "__main__":
    main()
