"""measure_exact_steps.py — the STEP-COUNT reduction of the memory-operand ALU
family (C4_EXACT_STEPS), on the SAME CPU model.forward.

For a length-N fused-arithmetic chain (here a dot product's per-element
multiply-accumulate expressed with the memory-operand ops), it measures three
counts:

  1. INTERPRETED — the base-ISA codegen for the SAME arithmetic.  An addressed
     binary op ``AX = mem[addr] <op> AX`` is  IMM addr; LI; PSH; <OP>  = 4
     model.forwards; a per-element MAC (a*b + acc) is ~8.

  2. EXACT c4 instruction count — the number of c4 instructions the fused program
     actually EXECUTES (one per element for a MAC/OPM chain: this IS the exact
     c4 trace length).

  3. FOLDED — the memory-operand model's ACTUAL forward count: one forward per
     fused op (the addressed read + the ALU fold into the SAME instruction's
     forward).  Success = folded == exact c4 count.

Both the interpreted and the folded runs are byte-exact through model.forward.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("C4_EXACT_STEPS", "1")
os.environ.setdefault("C4_MEM_OPERAND", "1")

from c4_min import isa
from c4_min import nibble_exact_steps as ES
from c4_min import nibble_pure_forward_complete as PFC


def interpreted_chain_program(ops_addrs, ax0):
    """The base-ISA codegen for a chain of  AX = mem[addr] <op> AX  ops, PRESERVING
    the running accumulator (AX) across steps.  Each fused op lowers to FOUR
    forwards:

        PSH ; IMM addr ; LI ; <OP>

    ``PSH`` stages the running accumulator as the popped operand, ``IMM addr; LI``
    reads ``mem[addr]`` into AX (the load), and ``<OP>`` computes
    ``popped_acc <op> mem[addr]``.  For the COMMUTATIVE ops used here (ADD/MUL)
    this equals ``mem[addr] <op> acc`` — byte-identical to the folded ``<OP>M``, so
    the two step counts are an apples-to-apples comparison of the SAME computation.

    (A non-commutative SUB/DIV/MOD would need an extra accumulator SPILL to a
    scratch cell to keep the operand order — MORE interpreted steps, which only
    widens the fold's advantage; we keep the demo commutative so the values match
    exactly.)"""
    prog = [isa.Instr(isa.IMM, ax0)]
    for op, addr in ops_addrs:
        prog += [
            isa.Instr(isa.PSH, 0),      # push running acc (AX)  -> staged operand
            isa.Instr(isa.IMM, addr),   # AX = addr
            isa.Instr(isa.LI, 0),       # AX = mem[addr]         (the LOAD)
            isa.Instr(op, 0),           # AX = acc <op> mem[addr] (== mem <op> acc, commutative)
        ]
    prog.append(isa.Instr(isa.HALT, 0))
    return prog


def folded_chain_program(ops_addrs, ax0):
    """The memory-operand codegen: one <OP>M per op."""
    _MAP = {isa.ADD: ES.ADDM, isa.SUB: ES.SUBM, isa.MUL: ES.MULM,
            isa.DIV: ES.DIVM, isa.MOD: ES.MODM}
    prog = [isa.Instr(isa.IMM, ax0)]
    for op, addr in ops_addrs:
        prog.append(isa.Instr(_MAP[op], addr))
    prog.append(isa.Instr(isa.HALT, 0))
    return prog


def main():
    # a length-N COMMUTATIVE fused chain: AX = mem[a_i] <op_i> AX, mixing + and *.
    # (commutative so the base-ISA `PSH acc; IMM; LI; OP` lowering is byte-identical
    # to the folded `<OP>M`, an apples-to-apples step comparison of the SAME value.)
    addrs = [0x40, 0x44, 0x48, 0x4C]
    seed = {0x40: 10, 0x44: 3, 0x48: 4, 0x4C: 2}
    ops = [isa.ADD, isa.MUL, isa.ADD, isa.MUL]
    ax0 = 7
    ops_addrs = list(zip(ops, addrs))
    N = len(ops_addrs)

    # ---- interpreted (complete VM) ----
    iprog = interpreted_chain_program(ops_addrs, ax0)
    cmodel, cL = PFC.build_pure_forward_complete_model(code_size=len(iprog))
    itrace = PFC.run_pure_forward_complete(cmodel, cL, iprog, max_steps=256, seed_mem=seed)
    interp_steps = len(itrace)
    interp_final = itrace[-1]

    # ---- folded (memory-operand ALU) ----
    fprog = folded_chain_program(ops_addrs, ax0)
    fmodel, fL = ES.build_exact_steps_model(code_size=len(fprog))
    ftrace = ES.run_exact_steps(fmodel, fL, fprog, max_steps=64, seed_mem=seed)
    fold_steps = len(ftrace)
    fold_final = ftrace[-1]

    # exact c4 instruction count for the FUSED program = one instruction per fused op
    # (+ the leading IMM + the HALT), and it IS the folded trace length by design.
    ref = ES.ref_interpret_exact(fprog, seed_mem=seed, mask=0xFF)
    exact_c4 = len(ref)

    print(f"length-{N} fused chain  AX = mem[a] <op> AX  (ops {[isa.NAMES[o] for o in ops]})")
    print(f"  reference final AX (numpy/ref) = {ref[-1]}")
    print()
    print(f"  INTERPRETED (IMM addr; LI; PSH; <OP> per op):")
    print(f"    {interp_steps} model.forwards  -> final AX {interp_final} "
          f"{'OK' if interp_final == ref[-1] else 'MISMATCH'}")
    print(f"    = {interp_steps / N:.1f} forwards / fused-op (4 base ISA ops each)")
    print()
    print(f"  EXACT c4 instruction count (fused program trace length):")
    print(f"    {exact_c4} instructions executed")
    print()
    print(f"  FOLDED (memory-operand model.forward):")
    print(f"    {fold_steps} model.forwards  -> final AX {fold_final} "
          f"{'OK' if fold_final == ref[-1] else 'MISMATCH'}")
    print(f"    = {fold_steps / N:.1f} forwards / fused-op")
    print()
    ok = (fold_steps == exact_c4 and fold_final == ref[-1] and ftrace == ref)
    print(f"  SUCCESS (folded == exact-c4 AND byte-exact) = {ok}")
    print(f"  reduction: interpreted {interp_steps} -> exact-c4/folded {fold_steps} "
          f"({interp_steps / max(1, fold_steps):.1f}x fewer forwards)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
