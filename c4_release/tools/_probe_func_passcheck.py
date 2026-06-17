#!/usr/bin/env python3
"""Fast full-trace pass check (spec_k=0) for a set of program ids.

Builds the model ONCE (honouring the env flags) and for each id compares the
neural emitted exit_code to the declarative oracle exit_code (masked to 8 bits,
the canonical full-trace criterion). Also flags any step that over/under-emits
(ntok != 35) so a frame desync that happens to land the right AX is still
visible.

Usage: python tools/_probe_func_passcheck.py <id_lo-id_hi[,id_lo-id_hi...]>
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); HALT = int(Token.HALT)


def parse_ids(spec):
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


@torch.no_grad()
def main():
    ids = parse_ids(sys.argv[1])
    maxsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    progs = generate_test_programs()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        probe = build_groundtruth_probe()
    npass = nfail = nerr = 0
    fails = []
    for pid in ids:
        if pid >= len(progs):
            continue
        src, exp, desc = progs[pid]
        try:
            bc = compile_c(src)[0]
            with contextlib.redirect_stderr(io.StringIO()):
                _, exit_code = probe.emitted_result(bc, max_steps=maxsteps)
            # also check frame coherence (ntok==35 each step)
            ctx = probe._final_context(bc, max_steps=maxsteps)
            pl = len(probe._build_context(bc))
            ntoks = []; cur = 0
            for p in range(pl, len(ctx)):
                cur += 1
                if ctx[p] == SE: ntoks.append(cur); cur = 0
                if ctx[p] == HALT: break
            desync = [n for n in ntoks if n != 35]
            ok = (exit_code & 0xFF) == (exp & 0xFF)
            if ok and not desync:
                npass += 1
            else:
                nfail += 1
                fails.append((pid, desc[:28], exp & 0xFF, exit_code & 0xFF,
                              "DESYNC" + str(desync[:3]) if desync else "axwrong"))
        except Exception as e:
            nerr += 1
            fails.append((pid, desc[:28] if 'desc' in dir() else '?', exp, None, f"ERR:{str(e)[:40]}"))
    print(f"PASS={npass} FAIL={nfail} ERR={nerr}  (n={len(ids)})")
    for pid, d, e, g, why in fails[:60]:
        print(f"  FAIL id{pid} {d:30s} exp={e} got={g} {why}")


if __name__ == "__main__":
    main()
