"""_analyze_opclass_steps.py — CPU-only per-opcode-class step census of a real
c4-compiled matmul, to ground WHICH op classes cost the inflated forwards.

No neural model is built: it compiles matmul C-source via the real c4 toolchain
and single-steps the draft VM (ref_interpret) while tallying executed opcodes by
class.  This tells us the marginal per-MAC breakdown (loads vs address-arith vs
loop-counter vs call-convention), so the folding targets the right classes.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min.selfhost._matmul_general_src import matmul_general_c


def compile_matmul(M, K, N, seed=7, call_fpmul=True):
    import random
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    rng = random.Random(seed)
    A = [rng.randint(0, 2) for _ in range(M * K)]
    B = [rng.randint(0, 2) for _ in range(K * N)]
    src = matmul_general_c(A, B, M, K, N, call_fpmul=call_fpmul)
    bc, _ = compile_c(src)
    return bytecode_to_isa(bc), A, B


def census(code, max_steps=5_000_000):
    """Single-step the draft VM, tally executed opcodes."""
    import c4_min.nibble_pure_forward_complete as PFC
    exec_ops = Counter()
    mem = {}
    SP_INIT = PFC.SP_INIT
    sp = bp = SP_INIT
    ax = pc = 0
    steps = 0
    mask = 0xFFFFFFFF
    ADJ = isa.ADJ
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        exec_ops[op] += 1
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask; sp += 4
            if op == isa.ADD: ax = (v + ax) & mask
            elif op == isa.SUB: ax = (v - ax) & mask
            elif op == isa.MUL: ax = (v * ax) & mask
            elif op == isa.DIV: ax = ((v // ax) if ax else 0) & mask
            else: ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            v = mem.get(sp, 0) & mask; sp += 4
            if op == isa.OR: ax = (v | ax) & mask
            elif op == isa.XOR: ax = (v ^ ax) & mask
            elif op == isa.AND: ax = (v & ax) & mask
            elif op == isa.SHL: ax = (v << ax) & mask
            else: ax = (v >> ax) & mask
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & mask; sp += 4; av = ax & mask
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: v < av,
                 isa.GT: v > av, isa.LE: v <= av, isa.GE: v >= av}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & 0xFF
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & 0xFF
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if ax == 0 else pc
        elif op == isa.BNZ: pc = imm if ax != 0 else pc
        elif op == isa.JSR: sp -= 4; mem[sp] = (i + 1) & 0xFF; pc = imm
        elif op == isa.ENT: mem[sp - 4] = bp & 0xFFFFFFFF; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == ADJ: sp += 4 * imm
        elif op == isa.LEV: sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.PRTF: pass
        elif op == isa.NOP: pass
        elif op == isa.HALT: break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
    return exec_ops, steps


CLASSES = {
    "load (LI/LC)": [isa.LI, isa.LC],
    "store (SI/SC)": [isa.SI, isa.SC],
    "imm (IMM/LEA)": [isa.IMM, isa.LEA],
    "alu (ADD..MOD)": [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD],
    "push (PSH)": [isa.PSH],
    "cmp (EQ..GE)": [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE],
    "branch (JMP/BZ/BNZ)": [isa.JMP, isa.BZ, isa.BNZ],
    "callconv (JSR/ENT/ADJ/LEV)": [isa.JSR, isa.ENT, isa.ADJ, isa.LEV],
    "io (PRTF)": [isa.PRTF],
    "nop/halt": [isa.NOP, isa.HALT],
}


def class_of(op):
    for cname, ops in CLASSES.items():
        if op in ops:
            return cname
    return f"other({isa.NAMES.get(op, op)})"


def breakdown(M, K, N, call_fpmul=True):
    code, A, B = compile_matmul(M, K, N, call_fpmul=call_fpmul)
    ops, steps = census(code)
    by_class = Counter()
    for op, c in ops.items():
        by_class[class_of(op)] += c
    return steps, by_class, len(code)


def main():
    print("Per-opcode-class census of the c4-compiled general matmul (draft VM)\n")
    for call_fpmul in (True, False):
        tag = "fpmul-CALL" if call_fpmul else "INLINE a*b/s"
        s1, c1, _ = breakdown(1, 1, 1, call_fpmul)
        s2, c2, _ = breakdown(1, 8, 1, call_fpmul)   # +7 inner-loop MACs
        d_macs = 7
        print(f"=== {tag} ===")
        print(f"  total steps: K=1 -> {s1}, K=8 -> {s2}  "
              f"(marginal {s2 - s1} over {d_macs} MACs = {(s2 - s1) / d_macs:.1f}/MAC)")
        print(f"  marginal per-MAC opcode-class breakdown:")
        allc = set(c1) | set(c2)
        for cname in sorted(allc, key=lambda k: -(c2[k] - c1[k])):
            d = c2[cname] - c1[cname]
            if d:
                print(f"    {cname:32s} {d/d_macs:6.2f}/MAC   "
                      f"(K1={c1[cname]:4d} K8={c2[cname]:4d})")
        print()


if __name__ == "__main__":
    main()
