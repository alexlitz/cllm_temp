"""Probe the pure-forward honest boundary: multi-slot stack (PSH depth>1).

The frame carries a single STACK0 mirror slot, so two pushes before a consume
should collide. Report the exact margin.
"""
from __future__ import annotations
from c4_min import isa
from c4_min.nibble_pure_forward import (
    build_pure_forward_model, run_pure_forward, assert_no_python_compute,
)


def main():
    m, L = build_pure_forward_model(code_size=20, include_memory=False,
                                    include_cmp=False, include_bitwise=False,
                                    include_muldiv=False)
    print(f"built dim={L.D} blocks={len(m.blocks)}")

    # depth-1 (works): IMM 6; PSH; IMM 7; ADD  -> 13
    p1 = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]
    c1 = isa.assemble(p1)
    t1 = assert_no_python_compute(run_pure_forward, m, L, c1)
    print(f"depth-1  IMM6;PSH;IMM7;ADD   got={t1[-1]} ref={isa.interpret(c1)[-1]}  "
          f"{'OK' if t1 == isa.interpret(c1) else 'MISMATCH'}")

    # depth-2 (boundary): IMM 10; PSH; IMM 20; PSH; IMM 5; ADD; ADD
    # ref: pop 20 + 5 = 25 ; pop 10 + 25 = 35. Single STACK0 slot -> 2nd PSH clobbers.
    p2 = [("IMM", 10), ("PSH", 0), ("IMM", 20), ("PSH", 0), ("IMM", 5),
          ("ADD", 0), ("ADD", 0), ("HALT", 0)]
    c2 = isa.assemble(p2)
    t2 = assert_no_python_compute(run_pure_forward, m, L, c2)
    ref2 = isa.interpret(c2)
    print(f"depth-2  ..PSH;..PSH;..ADD;ADD  got={t2[-1]} ref={ref2[-1]}  "
          f"{'OK' if t2 == ref2 else 'MISMATCH (single-STACK0 boundary)'}")
    print(f"   full got={t2}")
    print(f"   full ref={ref2}")


if __name__ == "__main__":
    main()
