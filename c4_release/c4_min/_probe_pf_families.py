"""Probe the pure-forward op families independently (lean builds, fast).

Builds only the families needed per case so we avoid the 181k-unit 8-bit muldiv
table when it isn't under test. Verifies each family byte-exact vs isa.interpret,
all under the no-python-compute guard.
"""
from __future__ import annotations
import sys, time
from c4_min import isa
from c4_min.nibble_pure_forward import (
    build_pure_forward_model, run_pure_forward, assert_no_python_compute,
)


def check(model, L, prog, label):
    code = isa.assemble(prog)
    ref = isa.interpret(code)
    got = assert_no_python_compute(run_pure_forward, model, L, code)
    ok = got == ref
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:22s} got={got} ref={ref}")
    return ok


def main():
    results = {}

    # ---- MEMORY (KV), lean: memory only, no cmp/bitwise/muldiv ----------------
    t = time.time()
    print("=== MEMORY (KV over emitted MEM tokens) — lean build ===")
    m, L = build_pure_forward_model(code_size=20, include_memory=True,
                                    include_cmp=False, include_bitwise=False,
                                    include_muldiv=False)
    print(f"  built dim={L.D} blocks={len(m.blocks)} heads={m.blocks[0].attn.n_heads} ({time.time()-t:.1f}s)")
    ok = True
    ok &= check(m, L, [("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                       ("IMM", 0x40), ("LI", 0), ("HALT", 0)], "store->load")
    ok &= check(m, L, [("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                       ("IMM", 0x80), ("LI", 0), ("HALT", 0)], "zfod (unwritten=0)")
    ok &= check(m, L, [("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                       ("IMM", 0x40), ("PSH", 0), ("IMM", 99), ("SI", 0),
                       ("IMM", 0x40), ("LI", 0), ("HALT", 0)], "latest-wins")
    ok &= check(m, L, [("IMM", 0x40), ("PSH", 0), ("IMM", 11), ("SI", 0),
                       ("IMM", 0x44), ("PSH", 0), ("IMM", 22), ("SI", 0),
                       ("IMM", 0x44), ("LI", 0), ("HALT", 0)], "two-addr")
    results["memory"] = ok
    del m

    # ---- CMP, lean: cmp only ---------------------------------------------------
    t = time.time()
    print("=== COMPARISONS (EQ/NE/LT/GT/LE/GE) — lean build ===")
    m, L = build_pure_forward_model(code_size=20, include_memory=False,
                                    include_cmp=True, include_bitwise=False,
                                    include_muldiv=False)
    print(f"  built dim={L.D} blocks={len(m.blocks)} ({time.time()-t:.1f}s)")
    ok = True
    for name, a, b in [("EQ", 5, 5), ("EQ", 5, 6), ("NE", 5, 6), ("LT", 3, 7),
                       ("GT", 7, 3), ("LE", 3, 3), ("GE", 3, 7)]:
        # push a (STK), IMM b (AX), CMP -> AX = a CMP b
        ok &= check(m, L, [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)],
                    f"{name} {a}?{b}")
    results["cmp"] = ok
    del m

    # ---- BITWISE, lean: bitwise only ------------------------------------------
    t = time.time()
    print("=== BITWISE (OR/XOR/AND/SHL/SHR) — lean build ===")
    m, L = build_pure_forward_model(code_size=20, include_memory=False,
                                    include_cmp=False, include_bitwise=True,
                                    include_muldiv=False)
    print(f"  built dim={L.D} blocks={len(m.blocks)} ({time.time()-t:.1f}s)")
    ok = True
    for name, a, b in [("OR", 0x0C, 0x03), ("XOR", 0xFF, 0x0F), ("AND", 0xF0, 0x3C),
                       ("SHL", 0x03, 2), ("SHR", 0xF0, 3)]:
        ok &= check(m, L, [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)],
                    f"{name} {a},{b}")
    results["bitwise"] = ok
    del m

    # ---- MULDIV (8-bit table) — big, do a couple of cases ---------------------
    t = time.time()
    print("=== MUL/DIV/MOD (8-bit table) — build is slow (181k units) ===")
    m, L = build_pure_forward_model(code_size=20, include_memory=False,
                                    include_cmp=False, include_bitwise=False,
                                    include_muldiv=True)
    print(f"  built dim={L.D} blocks={len(m.blocks)} ({time.time()-t:.1f}s)")
    ok = True
    for name, a, b in [("MUL", 6, 7), ("MUL", 20, 20), ("DIV", 84, 7),
                       ("MOD", 85, 7), ("DIV", 5, 0), ("MOD", 5, 0)]:
        ok &= check(m, L, [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)],
                    f"{name} {a},{b}")
    results["muldiv"] = ok
    del m

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
