#!/usr/bin/env python3
"""Probe the MEM-store rows of test_si_li_roundtrip to see OP_JSR / MARK_MEM /
MEM_STORE there (the e0 rule must NOT fire on a non-JSR SI store). Compare with
the JSR store + the id262 STEP_END mis-fire row. spec_k=0, hook-free.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.embedding import Opcode as Op  # noqa

DIMS = ["OP_JSR", "OP_ENT", "MARK_MEM", "MEM_STORE", "HAS_SE", "IS_BYTE", "PSH_AT_SP"]


def getdim(dp, name):
    base, _, off = name.partition("+")
    return int(dp[base]) + (int(off) if off else 0)


def bc(ops):
    out = []
    for o in ops:
        if isinstance(o, tuple):
            op, imm = o; out.append(op | (imm << 8))
        else:
            out.append(o)
    return out


def main():
    probe = build_groundtruth_probe()
    m = probe.model; dp = m.dim_positions; dev = next(m.parameters()).device
    # test_si_li_roundtrip
    prog = bc([(Op.IMM, 0x200), Op.PSH, (Op.IMM, 42), Op.SI, (Op.IMM, 0x200), Op.LI, Op.EXIT])
    ctx = probe._final_context(prog, max_steps=30)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=28)[0].float()
    pl = len(probe._build_context(prog))
    print(f"si_li_roundtrip prompt_len={pl} total={len(ctx)}")
    # report all MEM-marker rows + discriminator + e0-rule score (my new weights)
    def e0_score(row):
        return (2.0*float(row[getdim(dp,"OP_JSR")]) - 10.0*float(row[getdim(dp,"OP_ENT")])
                + 100.0*float(row[getdim(dp,"MARK_MEM")]) + 100.0*float(row[getdim(dp,"MEM_STORE")])
                + 1.0*float(row[getdim(dp,"HAS_SE")])
                - 1e12*float(row[getdim(dp,"IS_BYTE")])
                - 1e5*float(row[getdim(dp,"PSH_AT_SP")])
                + float(row[getdim(dp,"OUTPUT_LO+0")]) - float(row[getdim(dp,"OUTPUT_LO+8")])
                + float(row[getdim(dp,"OUTPUT_HI_THIS_STEP+14")]) - float(row[getdim(dp,"OUTPUT_HI_THIS_STEP+15")]))
    for p in range(pl, len(ctx)):
        if float(r[p][int(dp["MARK_MEM"])]) > 0.5 or float(r[p][int(dp["MEM_STORE"])]) > 0.5:
            row = r[p]
            scal = " ".join(f"{nm}={float(row[getdim(dp,nm)]):+.2f}" for nm in DIMS)
            print(f"  pos {p}: {scal}  | e0_score(new)={e0_score(row):+.1f} gate=HAS_SE={float(row[getdim(dp,'HAS_SE')]):+.2f}")


if __name__ == "__main__":
    main()
