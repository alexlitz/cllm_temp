#!/usr/bin/env python3
"""Find a clean per-position discriminator that is TRUE on the ENT-step AX
byte-1 row (where 0x02 leaks) and FALSE on (a) a genuine LEA byte-1 row and
(b) a genuine IMM value byte-1 row that legitimately needs a nonzero byte-1.

Scans ALL named dims at each byte-1 row and prints the ones that differ.
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
# A program whose AX byte-1 is legitimately 0x02 at some step: IMM 0x0205 = 517.
SRC_IMM = "int main() { return 517; }"  # 517 = 0x205, byte-1 = 0x02

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


def _vec(r):
    out = {}
    for nm, base in _DP.items():
        if "." in nm:
            continue
        v = float(r[base])
        if abs(v) > 0.15:
            out[nm] = round(v, 2)
    return out


def main():
    probe = GroundTruthProbe.build()
    bc436, _ = compile_c(SRC_436)
    bcimm, _ = compile_c(SRC_IMM)
    m436, _ = _markers(probe, bc436, 14)
    mimm, _ = _markers(probe, bcimm, 4)
    # ENT byte-1 row (id436 step-1 marker+1) -- LEAKS 0x02
    ent_b1 = _vec(_row(probe, bc436, m436[1] + 1, 14))
    # genuine LEA byte-1 row (id436 step-2 marker+1) -- emits 0xff
    lea_b1 = _vec(_row(probe, bc436, m436[2] + 1, 14))
    # genuine IMM value byte-1 row (517=0x205, the IMM step) -- legit 0x02
    # find the IMM step (the one before EXIT). just use the last AX marker.
    imm_b1 = _vec(_row(probe, bcimm, mimm[-2] + 1, 4)) if len(mimm) >= 2 else {}

    print("=== dims lit (>0.15) on each byte-1 row ===")
    keys = sorted(set(ent_b1) | set(lea_b1) | set(imm_b1))
    print(f"{'dim':28} {'ENT(leak)':>10} {'LEA(0xff)':>10} {'IMM(0x02)':>10}")
    for k in keys:
        e = ent_b1.get(k, 0.0)
        l = lea_b1.get(k, 0.0)
        i = imm_b1.get(k, 0.0)
        # highlight dims TRUE on ENT but FALSE on both genuine rows
        mark = " <== ENT-only" if (abs(e) > 0.5 and abs(l) < 0.3 and abs(i) < 0.3) else ""
        print(f"{k:28} {e:>10} {l:>10} {i:>10}{mark}")


if __name__ == "__main__":
    main()
