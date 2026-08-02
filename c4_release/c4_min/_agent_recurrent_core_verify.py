#!/usr/bin/env python3
"""END-TO-END byte-exactness proof for C4_RECURRENT_CORE (the nested divmod
weight-tie).

Builds the pure-forward complete model with ``recurrent_divmod=True`` in BOTH
flag states (C4_RECURRENT_CORE unset vs =1), runs a battery of DIV / MOD
programs through the vanilla ``model.forward`` decode, and asserts the emitted
exit value is (a) IDENTICAL between the two builds and (b) the correct arithmetic
result.  This is the model-level counterpart to the gadget-level
``_test_recurrent_divmod_gadget.py``.

The tie is byte-EXACT by construction (the qbc carry-round block is
position-independent src==dst==QB, so the APPLIED block sequence is unchanged and
only the STORED physical set shrinks 95->90); this script confirms that end to end
through the real transformer, not just at the gadget.

Run in TWO processes (the flag is read once at build):

    C4_RECURRENT_CORE=0 python c4_min/_agent_recurrent_core_verify.py --tag OFF --out /tmp/rc_off.json
    C4_RECURRENT_CORE=1 python c4_min/_agent_recurrent_core_verify.py --tag ON  --out /tmp/rc_on.json
    python c4_min/_agent_recurrent_core_verify.py --compare /tmp/rc_off.json /tmp/rc_on.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete,
)


def _divmod_prog(a: int, b: int, op: str):
    """AX = a op b, then HALT AX.  op in {DIV, MOD}.  ISA semantics: STACK0 op AX,
    so ``IMM a; PSH; IMM b; op`` computes ``a op b`` (a=dividend, b=divisor)."""
    opc = {"DIV": isa.DIV, "MOD": isa.MOD}[op]
    return [
        isa.Instr(isa.IMM, a),      # AX = a  (dividend)
        isa.Instr(isa.PSH, 0),      # STACK0 = a
        isa.Instr(isa.IMM, b),      # AX = b  (divisor)
        isa.Instr(opc, 0),          # AX = STACK0 op AX = a op b
        isa.Instr(isa.HALT, 0),     # halt with AX
    ]


# (a, b, op) battery: small, edge, 32-bit, and value-carrying literals.
_CASES = [
    (100, 7, "DIV"), (100, 7, "MOD"),
    (999, 13, "DIV"), (999, 13, "MOD"),
    (255, 16, "DIV"), (255, 16, "MOD"),
    (1000, 1, "DIV"), (1000, 1, "MOD"),
    (720, 6, "DIV"), (84, 5, "MOD"),
    (65535, 255, "DIV"), (65535, 255, "MOD"),
    (12345, 678, "DIV"), (12345, 678, "MOD"),
    (5, 7, "DIV"), (5, 7, "MOD"),        # a < b
    (0, 3, "DIV"), (0, 3, "MOD"),        # a == 0
    (42, 42, "DIV"), (41, 42, "MOD"),
]


def _expected(a, b, op):
    if b == 0:
        return 0
    return (a // b) if op == "DIV" else (a % b)


def run_battery(tag, n_cases=None):
    model, L = build_pure_forward_complete_model(code_size=64, recurrent_divmod=True)
    results = {"tag": tag,
               "phys_blocks": len(L._block_names),
               "applied_blocks": len(model.blocks),
               "recurrent_core": os.environ.get("C4_RECURRENT_CORE", "0"),
               "cases": []}
    ok = 0
    cases = _CASES if n_cases is None else _CASES[:n_cases]
    for a, b, op in cases:
        code = _divmod_prog(a, b, op)
        trace = run_pure_forward_complete(model, L, code, max_steps=64,
                                          mask=0xFFFFFFFF)
        # trace is the per-step AX trace (list of ints, like ref_interpret); the
        # final AX is the EXIT value.
        got_val = int(trace[-1]) if trace else None
        exp = _expected(a, b, op)
        correct = (got_val == exp)
        ok += correct
        results["cases"].append({"a": a, "b": b, "op": op,
                                 "got": got_val, "exp": exp, "ok": correct})
    results["correct"] = ok
    results["total"] = len(cases)
    return results


def compare(path_off, path_on):
    off = json.load(open(path_off))
    on = json.load(open(path_on))
    print(f"OFF: phys={off['phys_blocks']} applied={off['applied_blocks']} "
          f"correct={off['correct']}/{off['total']}")
    print(f"ON : phys={on['phys_blocks']} applied={on['applied_blocks']} "
          f"correct={on['correct']}/{on['total']}")
    identical = 0
    mism = []
    for co, cn in zip(off["cases"], on["cases"]):
        if co["got"] == cn["got"]:
            identical += 1
        else:
            mism.append((co, cn))
    print(f"BYTE-IDENTICAL outputs OFF==ON: {identical}/{len(off['cases'])}")
    if mism:
        print("MISMATCHES:", mism)
    stored_delta = off["phys_blocks"] - on["phys_blocks"]
    print(f"STORED-BLOCK REDUCTION (OFF - ON): {stored_delta} "
          f"({off['phys_blocks']} -> {on['phys_blocks']})")
    all_correct = off["correct"] == off["total"] and on["correct"] == on["total"]
    exact = identical == len(off["cases"]) and off["applied_blocks"] == on["applied_blocks"]
    print(f"VERDICT: byte-exact={exact}  all-correct={all_correct}")
    return 0 if (exact and all_correct) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="RUN")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", type=int, default=None,
                    help="Limit to the first N cases (CPU decode is slow).")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    res = run_battery(args.tag, n_cases=args.cases)
    print(f"[{args.tag}] phys={res['phys_blocks']} applied={res['applied_blocks']} "
          f"correct={res['correct']}/{res['total']} "
          f"C4_RECURRENT_CORE={res['recurrent_core']}")
    for c in res["cases"]:
        flag = "ok " if c["ok"] else "BAD"
        print(f"  [{flag}] {c['a']} {c['op']} {c['b']} -> {c['got']} (exp {c['exp']})")
    if args.out:
        json.dump(res, open(args.out, "w"))
    sys.exit(0 if res["correct"] == res["total"] else 1)


if __name__ == "__main__":
    main()
