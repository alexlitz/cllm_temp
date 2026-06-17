#!/usr/bin/env python3
"""Free-run (spec_k=0) LI-step AX and LEV-step PC for func_identity_0 under a
set of flag configs, each built fresh in a SUBPROCESS so the cache key is
honoured. This is the authoritative A/B for the LI/LEV chain.

Run as: python tools/_probe_func_flag_matrix.py
"""
from __future__ import annotations
import os, sys, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)

import json
_CFG_OVERRIDE = os.environ.get("MATRIX_CONFIGS")
if _CFG_OVERRIDE:
    CONFIGS = json.loads(_CFG_OVERRIDE)
else:
    CONFIGS = {
        "DEFAULT": {},
        "framing": {"C4_PSH_STACK0_BYTE3_RELAY_DARKEN": "1"},
        "pc_restore": {"C4_L15_LEV_PC_RESTORE": "1"},
        "pc+widen": {"C4_L15_LEV_PC_RESTORE": "1", "C4_L15_LEV_ADDR_WIDEN": "1"},
        "all3": {"C4_PSH_STACK0_BYTE3_RELAY_DARKEN": "1", "C4_L15_LEV_PC_RESTORE": "1",
                 "C4_L15_LEV_ADDR_WIDEN": "1"},
    }

CHILD = r'''
import os, sys, contextlib, io
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
sys.path.insert(0, %r)
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
pid=550; LISTEP=7; LEVSTEP=8
src,exp,desc=generate_test_programs()[pid]; bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    probe=build_groundtruth_probe()
    recs=probe.probe(bc, max_steps=12)
positions=sorted(recs.keys())
SE=int(Token.STEP_END); HALT=int(Token.HALT)
steps=[]; cur=[]
for p in positions:
    cur.append(p)
    if recs[p]["token"] in (SE,HALT):
        steps.append(cur); cur=[]
def axb0(stp):
    # off6 = AX byte0
    return recs[steps[stp][6]]["token"] if len(steps)>stp and len(steps[stp])>6 else None
def pcb0(stp):
    return recs[steps[stp][1]]["token"] if len(steps)>stp and len(steps[stp])>1 else None
print(f"LI_AX={axb0(LISTEP)} LEV_PC={pcb0(LEVSTEP)} nsteps={len(steps)}")
'''


def main():
    print("config                LI_AX(want70)  LEV_PC(want90)")
    for name, env in CONFIGS.items():
        e = dict(os.environ)
        e["CUDA_VISIBLE_DEVICES"] = "0"
        # set all toggles explicitly off first, then apply config
        for k in ("C4_PSH_STACK0_BYTE3_RELAY_DARKEN", "C4_L15_LEV_PC_RESTORE",
                  "C4_L15_LEV_ADDR_WIDEN", "C4_L15_LEV_PC_ONLY"):
            e[k] = "0"
        e.update(env)
        out = subprocess.run([sys.executable, "-c", CHILD % _PKG], env=e,
                             capture_output=True, text=True, cwd=_PKG, timeout=900)
        line = [l for l in out.stdout.splitlines() if l.startswith("LI_AX=")]
        res = line[-1] if line else f"ERR {out.stderr[-200:]}"
        print(f"{name:20s}  {res}")


if __name__ == "__main__":
    main()
