"""STEP-4 verification: signed comparison on the FULL PRODUCTION model.

Builds the SINGLE full-op-set pure-forward model (build_compact_sparse_streaming
-> the same weights the 1096 corpus + demos run) ONCE, then:

  (A) runs the three #673 f3 SIGNED programs (f3_signed_cmp / f3_if_neg /
      f3_count_down_neg) through model.forward and asserts the final AX matches
      the SIGNED ideal (== gcc) byte-exact; and

  (B) runs a stratified sample of the 1096 COMPARISON clusters (if_gt / if_lt /
      if_eq / func_max / func_min / absdiff / loop_countdown) through the SAME
      model and asserts each still matches its gcc ``expected`` (no regression).

Memory-lean: one streaming-sparse build (peak ~one dense block), sequential
runs, small code_size.  Run:

    OMP_NUM_THREADS=4 PYTHONPATH=<repo-root> \
        python -m c4_min._verify_signed_cmp_production          # 8-bit fold
    OMP_NUM_THREADS=4 C4_VM_WIDTH32=1 PYTHONPATH=<repo-root> \
        python -m c4_min._verify_signed_cmp_production          # 32-bit substrate
"""
from __future__ import annotations

import os
import sys

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min import isa
from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min._diag_signed_cmp_baseline import (
    prog_f3_signed_cmp, prog_f3_if_neg, prog_f3_count_down_neg, signed_ideal_trace,
)

WIDTH32 = os.environ.get("C4_VM_WIDTH32", "0") == "1"
_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm):
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode):
    out = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            simm = _sign32(imm)
            assert simm % _WORD == 0
            out.append(isa.Instr(op, simm // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


def main():
    code_size = int(os.environ.get("VERIFY_CODE_SIZE", "48"))
    print(f"C4_VM_WIDTH32={os.environ.get('C4_VM_WIDTH32','0')} "
          f"code_size={code_size}", flush=True)
    print("building FULL production model (streaming sparse) ...", flush=True)
    model, L, _stats = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    print(f"model: dim={L.D} blocks={len(model.blocks)}\n", flush=True)

    mask = 0xFFFFFFFF
    fails = 0

    # (A) the three f3 signed programs ----------------------------------------
    print("=== (A) f3 SIGNED programs (want == gcc signed) ===", flush=True)
    for name, (code, desc) in [
        ("f3_signed_cmp", prog_f3_signed_cmp()),
        ("f3_if_neg", prog_f3_if_neg()),
        ("f3_count_down_neg", prog_f3_count_down_neg()),
    ]:
        trace = run_pure_forward_complete(model, L, code, max_steps=64, mask=mask)
        got = trace[-1] if trace else None
        want = signed_ideal_trace(code)[-1]
        ok = got == want
        fails += not ok
        print(f"  {name:<20} got={got} want={want} "
              f"{'PASS' if ok else 'FAIL'}  ({desc})", flush=True)

    # (A2) DIAGNOSTIC: isolate the count_down failure — does a buried-stack SUB
    # deliver the negative to AX at all (independent of the compare)?
    print("\n=== (A2) diagnostics (buried-stack negative delivery) ===", flush=True)
    diag = {
        # IMM2 PSH IMM0 PSH IMM3 SUB HALT : AX should be -3 with [2] still buried.
        "buried_SUB_neg (AX=-3)":
            isa.assemble([("IMM", 2), ("PSH", 0), ("IMM", 0), ("PSH", 0),
                          ("IMM", 3), ("SUB", 0), ("HALT", 0)]),
        # same but ADD the buried 2 back:  ... SUB(AX=-3) ADD(pop 2 + -3 = -1)
        "buried_then_ADD (AX=-1)":
            isa.assemble([("IMM", 2), ("PSH", 0), ("IMM", 0), ("PSH", 0),
                          ("IMM", 3), ("SUB", 0), ("ADD", 0), ("HALT", 0)]),
    }
    for name, code in diag.items():
        trace = run_pure_forward_complete(model, L, code, max_steps=64, mask=mask)
        got = trace[-1] if trace else None
        want = signed_ideal_trace(code)[-1]
        print(f"  {name:<24} got={got} want={want} "
              f"{'ok' if got == want else 'DIVERGE'}", flush=True)

    # (B) 1096 comparison-cluster regression sample ---------------------------
    print("\n=== (B) 1096 comparison clusters (want == gcc expected) ===",
          flush=True)
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    import re

    def clu(d):
        b = d.split(":", 1)[0].strip()
        b = re.sub(r"_\d+$", "", b); b = re.sub(r"\d+$", "", b)
        return b.rstrip("_") or "misc"

    want_clusters = {"if_gt", "if_lt", "if_eq", "func_max", "func_min",
                     "absdiff", "loop_countdown"}
    per_cluster = int(os.environ.get("VERIFY_PER_CLUSTER", "3"))
    seen = {}
    n_ok = n_run = 0
    for src, expected, d in generate_test_programs():
        k = clu(d)
        if k not in want_clusters:
            continue
        if seen.get(k, 0) >= per_cluster:
            continue
        seen[k] = seen.get(k, 0) + 1
        try:
            bytecode, _data = compile_c(src)
            code = bytecode_to_isa(bytecode)
            if len(code) > code_size:
                print(f"  SKIP {d} (len {len(code)} > code_size)", flush=True)
                continue
            trace = run_pure_forward_complete(model, L, code, max_steps=4000,
                                              mask=mask)
            got = trace[-1] if trace else None
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {d}: {e}", flush=True)
            fails += 1
            continue
        exp = expected & 0xFFFFFFFF
        ok = got == exp
        n_run += 1; n_ok += ok; fails += not ok
        print(f"  [{k:<14}] {d:<28} got={got} exp={exp} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\ncomparison-cluster sample: {n_ok}/{n_run} PASS", flush=True)
    print(f"TOTAL FAILS: {fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
