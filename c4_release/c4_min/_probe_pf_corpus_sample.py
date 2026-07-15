"""M5 — CORPUS-SAMPLE pure-forward fraction (LEAN complete model).

One representative program per op-class (the same 30 op-classes ``oracle.py``
enumerates, mirroring its program shapes), run through the COMPLETE pure-forward
driver (multi-slot stack via the KV head + callconv JSR/ENT/LEA/LI/LEV + 32-bit
ALU) on a LEAN model (``include_divmod=False`` -> 40 blocks, NOT the 262-block
long-division), each under the no-python-compute guard.  Reports the fraction that
runs 100%-in-forward byte-exact vs ``ref_interpret`` and names the exact residual.

The lean model keeps the full multi-slot stack + calling convention + ADD/SUB/MUL
+ cmp + bitwise + memory; only DIV/MOD need the (separately-proven) 32-bit
long-division block, so those two op-classes are the expected residual.

Memory addresses are small (8-bit AX): the oracle's ``DATA_BASE`` (0x10000) is
remapped to a byte address and the preloaded-data LI/LC cases become store-then-
load so the pure-forward KV memory is exercised end-to-end (the semantics are
identical: LC/LI read the most recent write to that address)."""
from __future__ import annotations
import sys, time

# small stack base so frame-relative LEA (AX is 8-bit) reaches the frame.
import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

from c4_min import isa
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret)
from c4_min.nibble_pure_forward import assert_no_python_compute

DATA = 0x40          # small data address in place of the oracle's 0x10000


def progs_by_class():
    """One representative isa-format program per op-class (mirrors oracle.py)."""
    P = {}
    # --- binops: IMM a; PSH; IMM b; <op>; HALT ---
    binop_cases = {
        "ADD": (3, 4), "SUB": (9, 4), "MUL": (6, 7), "DIV": (84, 7), "MOD": (84, 5),
        "AND": (0x6C, 0x3A), "OR": (0x6C, 0x3A), "XOR": (0x6C, 0x3A),
        "SHL": (5, 3), "SHR": (200, 2),
        "EQ": (5, 5), "NE": (7, 9), "LT": (7, 9), "GT": (9, 7), "LE": (7, 9), "GE": (9, 7),
    }
    for op, (a, b) in binop_cases.items():
        P[op] = [(("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0))]
    # --- IMM ---
    P["IMM"] = [(("IMM", 42), ("HALT", 0))]
    # --- PSH (round-trip through ADD 0) ---
    P["PSH"] = [(("IMM", 200), ("PSH", 0), ("IMM", 0), ("ADD", 0), ("HALT", 0))]
    # --- JMP ---
    P["JMP"] = [(("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0))]
    # --- BZ / BNZ ---
    P["BZ"] = [(("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0))]
    P["BNZ"] = [(("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0))]
    # --- LI / LC / SI / SC : store-then-load (the pure-forward KV memory) ---
    P["SI"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
               ("IMM", DATA), ("LI", 0), ("HALT", 0))]
    P["SC"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x41), ("SC", 0),
               ("IMM", DATA), ("LC", 0), ("HALT", 0))]
    P["LI"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x5A), ("SI", 0),
               ("IMM", DATA), ("LI", 0), ("HALT", 0))]
    P["LC"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x5A), ("SC", 0),
               ("IMM", DATA), ("LC", 0), ("HALT", 0))]
    # --- LEA / ENT : frame local store+load (ENT n; LEA slot; SI; LEA slot; LI) ---
    #   ENT 1 (imm=SLOTS): reserve 1 local at BP-4 -> LEA slot -1 (BP-4).
    P["ENT"] = [(("ENT", 1), ("LEA", -1), ("PSH", 0), ("IMM", 0x2A), ("SI", 0),
                ("LEA", -1), ("LI", 0), ("HALT", 0))]
    P["LEA"] = [(("ENT", 1), ("LEA", -1), ("PSH", 0), ("IMM", 0x37), ("SI", 0),
                ("LEA", -1), ("LI", 0), ("HALT", 0))]
    # --- ADJ : push then discard ---
    P["ADJ"] = [(("IMM", 0x99), ("PSH", 0), ("ADJ", 1), ("IMM", 0x11), ("HALT", 0))]
    # --- JSR / LEV : a tiny call to a leaf returning 0x2A ---
    #   [0] IMM 0 ; [1] JSR 3 ; [2] HALT ; [3] IMM 0x2A ; [4] LEV
    call = (("IMM", 0), ("JSR", 3), ("HALT", 0), ("IMM", 0x2A), ("LEV", 0))
    P["JSR"] = [call]
    P["LEV"] = [call]
    return P


ALL = ("ADD", "SUB", "MUL", "DIV", "MOD",
       "EQ", "NE", "LT", "GT", "LE", "GE",
       "AND", "OR", "XOR", "SHL", "SHR",
       "LI", "LC", "SI", "SC", "PSH",
       "LEA", "IMM", "JMP", "JSR", "ENT", "ADJ", "LEV", "BZ", "BNZ")


def main():
    t = time.time()
    m, L = build_pure_forward_complete_model(code_size=24, include_bitwise=True,
                                             include_divmod=False)
    print(f"LEAN complete model: dim={L.D} blocks={len(m.blocks)} "
          f"heads={m.blocks[0].attn.n_heads} ({time.time()-t:.1f}s)\n", flush=True)
    P = progs_by_class()
    npass = nfail = 0
    fails = []
    for op in ALL:
        for prog in P[op]:
            code = isa.assemble(list(prog))
            ref = ref_interpret(code)
            try:
                got = assert_no_python_compute(run_pure_forward_complete, m, L, code,
                                               max_steps=64)
            except AssertionError as e:
                print(f"  [GUARD-FAIL] {op:5s} {e}", flush=True)
                got = None
            ok = got == ref
            tag = "PASS" if ok else "FAIL"
            if ok:
                npass += 1
            else:
                nfail += 1
                fails.append((op, got[-1] if got else None, ref[-1] if ref else None))
            gv = (got[-1] if got else None)
            rv = (ref[-1] if ref else None)
            print(f"  [{tag}] {op:5s} got={gv} ref={rv}", flush=True)
    n = npass + nfail
    print(f"\n=== CORPUS-SAMPLE pure-forward fraction: {npass}/{n} "
          f"= {100.0*npass/n:.0f}% byte-exact 100%-in-forward, guard-clean ===",
          flush=True)
    if fails:
        print("RESIDUAL (need python / unsupported in lean model):", flush=True)
        for op, g, r in fails:
            print(f"    {op:5s} got={g} ref={r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
