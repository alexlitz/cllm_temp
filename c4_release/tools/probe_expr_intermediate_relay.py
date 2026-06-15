#!/usr/bin/env python3
"""Probe the expr-chain intermediate handoff: a first op produces a >=256
result, it is PSH'd, then a SECOND op (DIV/MOD/ADD) reads it.

Uses the REAL compiled corpus bytecode (compile_c) for the failing ids so the
instruction stream matches the test suite exactly.

Question chain (spec_k=0, BUILT dims, hook-free):
  1. After the first op produces a >=256 result, where does its HIGH byte
     live at the MARK_AX row? (CLEAN_EMBED? AX_FULL? OUTPUT?)
  2. When that result is PSH'd, does ``layer10_psh_ax_broadcast`` deposit the
     high byte into STACK0_BYTE_VAL_1_LO/HI at the picked STACK0_BYTE1 row?
  3. Does the second op's operand relay read it (or read 0x00 / truncate)?

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_expr_intermediate_relay.py 852 800 875
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe


def _corpus_bytecode(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    tests = generate_test_programs()
    source, expected, desc = tests[idx]
    bytecode, data = compile_c(source)
    return bytecode, data, expected, desc


def _disasm(bytecode):
    from neural_vm.embedding import Opcode
    names = {int(getattr(Opcode, n)): n for n in dir(Opcode)
             if not n.startswith("_") and isinstance(getattr(Opcode, n), int)}
    out = []
    for i, instr in enumerate(bytecode):
        op = instr & 0xFF
        imm = (instr >> 8)
        out.append(f"{i}:{names.get(op, op)}" + (f"({imm})" if imm else ""))
    return " ".join(out)


def onehot(row, base, w=16):
    return [(i, round(float(row[base + i].item()), 2))
            for i in range(w) if float(row[base + i].item()) > 0.5]


def decode_nib(row, lo, hi):
    los = onehot(row, lo); his = onehot(row, hi)
    if len(los) == 1 and len(his) == 1:
        return his[0][0] * 16 + los[0][0]
    return None


def main(ids):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device

    def D(name):
        return dp.get(name)

    S1LO = D("STACK0_BYTE_VAL_1_LO"); S1HI = D("STACK0_BYTE_VAL_1_HI")
    CLO = D("CLEAN_EMBED_LO"); CHI = D("CLEAN_EMBED_HI")
    AXFLO = D("AX_FULL_LO"); AXFHI = D("AX_FULL_HI")
    OLO = D("OUTPUT_LO"); OHI = D("OUTPUT_HI")
    ALU_LO = D("ALU_LO"); ALU_HI = D("ALU_HI")
    STACK0_BYTE1 = D("STACK0_BYTE1"); MARK_AX = D("MARK_AX")
    OP = {nm: D("OP_" + nm) for nm in
          ("PSH", "MUL", "DIV", "ADD", "MOD", "IMM", "EXIT")}
    print(f"# dims: S1LO={S1LO} CLEAN_LO={CLO} AXFULL_LO={AXFLO} "
          f"OUTPUT_LO={OLO} ALU_LO={ALU_LO} STACK0_BYTE1={STACK0_BYTE1}")

    for idx in ids:
        bytecode, data, expected, desc = _corpus_bytecode(idx)
        assert not data, f"id={idx} has non-empty data segment {data!r}"
        _, code = probe.emitted_result(bytecode, max_steps=12)
        ctx = probe._final_context(bytecode, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        resid = model.forward(padded, stop_after_block=11)[0]

        print(f"\n==== id={idx} {desc} exp={expected} neural_code={code} ====")
        print(f"  disasm: {_disasm(bytecode)}")
        nrows = resid.shape[0]
        for r in range(nrows):
            if float(resid[r, MARK_AX].item()) <= 0.5:
                continue
            ops = [nm for nm, dd in OP.items()
                   if dd is not None and float(resid[r, dd].item()) > 0.5]
            ax_full = decode_nib(resid[r], AXFLO, AXFHI) if AXFLO else None
            clean = decode_nib(resid[r], CLO, CHI) if CLO else None
            out = decode_nib(resid[r], OLO, OHI) if OLO else None
            alu = decode_nib(resid[r], ALU_LO, ALU_HI) if ALU_LO else None
            print(f"  AX row {r:3d} op={ops}: AX_FULL={ax_full} "
                  f"CLEAN={clean} OUTPUT={out} ALU={alu}")

        s1_rows = [r for r in range(nrows)
                   if float(resid[r, STACK0_BYTE1].item()) > 0.5]
        print(f"  STACK0_BYTE1 rows: {s1_rows}")
        for r in s1_rows:
            s1 = decode_nib(resid[r], S1LO, S1HI) if S1LO else None
            clean = decode_nib(resid[r], CLO, CHI) if CLO else None
            print(f"    @s1 row {r}: STACK0_BYTE_VAL_1={s1} "
                  f"(lo={onehot(resid[r], S1LO)} hi={onehot(resid[r], S1HI)})  "
                  f"CLEAN={clean}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [852, 800, 875]
    main(ids)
