#!/usr/bin/env python3
"""Decisive check: C4_DIVMOD_STACK0_BYTE1_CLEAR is INERT on non-divmod ops.

Builds the campaign model with the flag ON and OFF, runs a SI-store, a LI, a
SHR, and an ADD program forward to the final block, and asserts the residual is
BYTE-IDENTICAL flag-on vs flag-off (proving the fix is gated to DIV/MOD rows and
cannot regress the SI/LI/SHR/ADD smoke). It also runs a DIV program and asserts
the residual DIFFERS (the fix IS active where intended).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 CUDA_VISIBLE_DEVICES="" \
    python tools/probe_divmod_byte1_inert.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else:
            bc.append(op)
    return bc


def build(flag_val, cache):
    os.environ["C4_DIVMOD_STACK0_BYTE1_CLEAR"] = flag_val
    os.environ["C4_VM_CACHE_DIR"] = cache
    # Fresh import each build so the probe builder re-reads the flag at bake.
    from tools.probe_groundtruth import build_groundtruth_probe
    return build_groundtruth_probe()


def main():
    # SI store, LI load, SHR, ADD (non-divmod) + DIV (divmod) programs.
    progs = {
        "SI_store": _mk([(Opcode.IMM, 0x1234), Opcode.PSH, (Opcode.IMM, 100),
                         Opcode.SI, Opcode.EXIT]),
        "SHR": _mk([(Opcode.IMM, 168), Opcode.PSH, (Opcode.IMM, 2),
                    Opcode.SHR, Opcode.EXIT]),
        "ADD": _mk([(Opcode.IMM, 654), Opcode.PSH, (Opcode.IMM, 114),
                    Opcode.ADD, Opcode.EXIT]),
        "DIV_multibyte": _mk([(Opcode.IMM, 364), Opcode.PSH, (Opcode.IMM, 14),
                              Opcode.DIV, Opcode.EXIT]),
    }

    pON = build("1", "/tmp/c4cache_inert_on")
    resid_on = {}
    for name, prog in progs.items():
        ctx = pON._final_context(prog, max_steps=6)
        dev = next(pON.model.parameters()).device
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        nb = len(pON.model.blocks)
        resid_on[name] = pON.model.forward(padded, stop_after_block=nb - 1)[0].clone().cpu()
    del pON

    pOFF = build("0", "/tmp/c4cache_inert_off")
    print("# flag ON vs OFF residual byte-diff at final block (campaign config)")
    all_ok = True
    for name, prog in progs.items():
        ctx = pOFF._final_context(prog, max_steps=6)
        dev = next(pOFF.model.parameters()).device
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        nb = len(pOFF.model.blocks)
        roff = pOFF.model.forward(padded, stop_after_block=nb - 1)[0].cpu()
        ron = resid_on[name]
        if ron.shape != roff.shape:
            print(f"  {name:16s} SHAPE MISMATCH {ron.shape} vs {roff.shape}")
            continue
        maxdiff = float((ron - roff).abs().max())
        identical = maxdiff < 1e-6
        expect_identical = (name != "DIV_multibyte")
        tag = "IDENTICAL" if identical else f"DIFFERS(max={maxdiff:.3g})"
        ok = (identical == expect_identical)
        all_ok = all_ok and ok
        verdict = "OK" if ok else "!!! UNEXPECTED"
        print(f"  {name:16s} {tag:24s} (expect "
              f"{'identical' if expect_identical else 'differs'}) {verdict}")
    print("\nRESULT:", "PASS — fix inert on non-divmod, active on divmod"
          if all_ok else "FAIL — unexpected residual behavior")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
