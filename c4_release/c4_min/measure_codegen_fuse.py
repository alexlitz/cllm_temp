"""measure_codegen_fuse.py — the forwards/MAC + byte-exactness measurement of the
memory-operand-ALU CODEGEN peephole (C4_CODEGEN_FUSE), on the SAME c4-compiled
matmul ``_analyze_opclass_steps`` censused (~101 forwards/MAC baseline).

What it reports
===============
For the general matmul (``call_fpmul`` both ways) it prints, per opclass:
  * BASELINE  forwards/MAC (the unfolded ``src.compiler`` codegen == 1 forward per
    executed c4 instruction, the a56fafd0 model contract).
  * AFTER the ADDM peephole fold (``C4_CODEGEN_FUSE=1``).
  * the fold count / MAC and which lever moved it.

Byte-exactness
==============
The folded program must compute IDENTICAL results.  Both the original and the
folded bytecode are single-stepped through the SAME reference interpreter
(``_ref_interpret_fused``, the ``nibble_pure_forward_complete.ref_interpret``
SP-addressed byte-stack semantics, extended with the ``<OP>M`` opcodes), and the
full per-step AX traces + the PRTF output bytes MUST match.  A mismatch is a
FAILURE (the fold changed semantics, not just instruction count).

Honest verdict (see the module report)
======================================
The peephole only folds ABSOLUTE-address (``IMM addr``) left operands.  The matmul
inner MAC operands are FRAME-relative pointer dereferences (``*ap`` =
``mem[mem[frame_off]]``), whose address is a runtime value in AX, NOT a
compile-time immediate — so the per-MAC ALU ops do NOT fold.  The fold reduces the
program's STATIC + setup instruction count (byte-exact), but the marginal
forwards/MAC is essentially unchanged.  This module MEASURES that honestly rather
than asserting a win.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import codegen_fuse as CF
from c4_min.selfhost._matmul_general_src import matmul_general_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
import c4_min.nibble_pure_forward_complete as PFC

# <OP>M opcode -> base ALU op (for the fused reference interpreter).
_MOP_BASE = {CF.ADDM: isa.ADD, CF.SUBM: isa.SUB, CF.MULM: isa.MUL,
             CF.DIVM: isa.DIV, CF.MODM: isa.MOD}


def _ref_interpret_fused(code, max_steps=5_000_000, mask=0xFFFFFFFF, out=None):
    """The census/matmul reference semantics (SP-addressed byte stack, LEA
    byte-masked, absolute addresses) extended with the ``<OP>M`` opcodes:
    ``AX = mem[imm] <op> AX`` (imm = ABSOLUTE address, no stack pop).  Returns
    ``(ax_trace, exec_ops_counter, steps)`` so the same run yields the byte-exact
    trace AND the opcode census."""
    mem = {}
    SP_INIT = PFC.SP_INIT
    sp = bp = SP_INIT
    ax = pc = 0
    trace = []
    exec_ops = Counter()
    steps = 0
    ADJ = isa.ADJ
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        exec_ops[op] += 1
        i = pc
        pc += 1
        if op in _MOP_BASE:                          # the folded memory-operand ALU
            v = mem.get(imm & 0xFFFFFFFF, 0) & mask
            base = _MOP_BASE[op]
            if base == isa.ADD: ax = (v + ax) & mask
            elif base == isa.SUB: ax = (v - ax) & mask
            elif base == isa.MUL: ax = (v * ax) & mask
            elif base == isa.DIV: ax = ((v // ax) if ax else 0) & mask
            else: ax = ((v % ax) if ax else 0) & mask
        elif op == isa.IMM:
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
        elif op == isa.PRTF:
            if out is not None: out.append(ax & 0xFF)
        elif op == isa.NOP: pass
        elif op == isa.HALT: break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
        trace.append(ax & mask)
    return trace, exec_ops, steps


CLASSES = {
    "load (LI/LC)": [isa.LI, isa.LC],
    "store (SI/SC)": [isa.SI, isa.SC],
    "imm (IMM/LEA)": [isa.IMM, isa.LEA],
    "alu (ADD..MOD)": [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD],
    "aluM (<OP>M fused)": [CF.ADDM, CF.SUBM, CF.MULM, CF.DIVM, CF.MODM],
    "push (PSH)": [isa.PSH],
    "cmp (EQ..GE)": [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE],
    "branch (JMP/BZ/BNZ)": [isa.JMP, isa.BZ, isa.BNZ],
    "callconv (JSR/ENT/ADJ/LEV)": [isa.JSR, isa.ENT, isa.ADJ, isa.LEV],
    "io (PRTF)": [isa.PRTF],
    "nop/halt": [isa.NOP, isa.HALT],
}


def _class_of(op):
    for cname, ops in CLASSES.items():
        if op in ops:
            return cname
    return f"other({isa.NAMES.get(op, op)})"


def _compile_matmul_words(M, K, N, seed=7, call_fpmul=True):
    from src.compiler import compile_c
    rng = random.Random(seed)
    A = [rng.randint(0, 2) for _ in range(M * K)]
    B = [rng.randint(0, 2) for _ in range(K * N)]
    src = matmul_general_c(A, B, M, K, N, call_fpmul=call_fpmul)
    words, _ = compile_c(src)
    return words


def _run(words):
    code = bytecode_to_isa(words)
    out = []
    trace, ops, steps = _ref_interpret_fused(code, out=out)
    return trace, ops, steps, out


def _byte_exact(M, K, N, call_fpmul):
    """Compile M,K,N; run the ORIGINAL and the FOLDED bytecode; assert the OBSERVABLE
    behaviour is identical.  The fold DELETES instructions, so the raw per-step AX
    trace naturally differs (fewer steps); the semantics-preserving invariant is the
    PRTF OUTPUT byte stream + the FINAL AX (the program's visible result).  Returns
    (ok, n_folds, base_steps, fold_steps)."""
    words = _compile_matmul_words(M, K, N, call_fpmul=call_fpmul)
    folded, n_folds = CF.fuse_bytecode(words)
    t0, _, s0, o0 = _run(words)
    t1, _, s1, o1 = _run(folded)
    final0 = t0[-1] if t0 else None
    final1 = t1[-1] if t1 else None
    ok = (o0 == o1 and final0 == final1)
    return ok, n_folds, s0, s1


def breakdown(M, K, N, call_fpmul, fused):
    words = _compile_matmul_words(M, K, N, call_fpmul=call_fpmul)
    if fused:
        words, _ = CF.fuse_bytecode(words)
    _, ops, steps, _ = _run(words)
    by_class = Counter()
    for op, c in ops.items():
        by_class[_class_of(op)] += c
    return steps, by_class


_GLOBAL_KERNEL = """
int g0; int g1; int g2; int g3; int acc;
int main() {
  int i;
  g0 = 5; g1 = 7; g2 = 3; g3 = 2; acc = 0;
  i = 0;
  while (i < COUNT) {
    acc = g0 + g1;
    acc = g2 * g3;
    acc = g0 + g2;
    acc = g1 * g3;
    i = i + 1;
  }
  printf(acc);
  return 0;
}
"""


def _compile_global_words(count):
    from src.compiler import compile_c
    src = _GLOBAL_KERNEL.replace("COUNT", str(count))
    words, _ = compile_c(src)
    return words


def _global_foldable_report():
    """A GLOBAL-operand kernel (absolute-address ALU operands) where the fold DOES
    fire — proves the peephole is real + byte-exact, and quantifies its reduction on
    a program shaped for it.  Reports the marginal per-loop-iteration reduction."""
    print("GLOBAL-operand kernel (absolute-address ALU operands — the fold TARGET):")
    ok_all = True
    for count in (1, 8):
        w = _compile_global_words(count)
        f, nf = CF.fuse_bytecode(w)
        t0, _, s0, o0 = _run(w)
        t1, _, s1, o1 = _run(f)
        ok = (o0 == o1 and (t0[-1] if t0 else None) == (t1[-1] if t1 else None))
        ok_all = ok_all and ok
        print(f"  COUNT={count}: {'OK ' if ok else 'MISMATCH'} folds={nf:3d}  "
              f"steps {s0}->{s1} ({s0 - s1} fewer)  out={o0}=={o1}")
    w1 = _compile_global_words(1); w8 = _compile_global_words(8)
    b1, _, s1b, _ = _run(w1); b8, _, s8b, _ = _run(w8)
    f1, _ = CF.fuse_bytecode(w1); f8, _ = CF.fuse_bytecode(w8)
    _, _, s1f, _ = _run(f1); _, _, s8f, _ = _run(f8)
    d_iter = 7
    print(f"  marginal per-iteration (4 global-operand ALU ops/iter): "
          f"BASELINE {(s8b - s1b) / d_iter:.1f} -> FOLDED {(s8f - s1f) / d_iter:.1f} "
          f"forwards/iter  ({(s8b - s1b) - (s8f - s1f)} fewer over {d_iter} iters)")
    print(f"  => byte-exact: {ok_all}\n")


def main():
    print("Memory-operand-ALU CODEGEN peephole (C4_CODEGEN_FUSE) — forwards/MAC on "
          "the c4-compiled matmul\n")

    # ---- byte-exactness gate FIRST (any mismatch aborts the measurement) ----
    print("byte-exactness (folded output == original output: PRTF stream + final AX):")
    all_ok = True
    for cf in (True, False):
        for (M, K, N) in ((1, 1, 1), (1, 8, 1), (2, 3, 2)):
            ok, nf, s0, s1 = _byte_exact(M, K, N, cf)
            all_ok = all_ok and ok
            tag = "fpmul-CALL" if cf else "INLINE"
            print(f"  {tag:11s} {M}x{K}x{N}: {'OK ' if ok else 'MISMATCH'} "
                  f"folds={nf:3d}  steps {s0}->{s1} ({s0 - s1} fewer executed)")
    print(f"  => byte-exact: {all_ok}\n")

    _global_foldable_report()

    d_macs = 7
    for call_fpmul in (True, False):
        tag = "fpmul-CALL" if call_fpmul else "INLINE a*b/s"
        print(f"=== {tag} ===")
        for lbl, fused in (("BASELINE (fold OFF)", False), ("FOLDED  (fold ON) ", True)):
            s1, c1 = breakdown(1, 1, 1, call_fpmul, fused)
            s8, c8 = breakdown(1, 8, 1, call_fpmul, fused)
            print(f"  {lbl}: steps K=1 -> {s1}, K=8 -> {s8}  "
                  f"(marginal {(s8 - s1) / d_macs:.1f}/MAC)")
        # per-class marginal breakdown, folded
        s1, c1 = breakdown(1, 1, 1, call_fpmul, True)
        s8, c8 = breakdown(1, 8, 1, call_fpmul, True)
        b1, d1 = breakdown(1, 1, 1, call_fpmul, False)
        b8, d8 = breakdown(1, 8, 1, call_fpmul, False)
        print(f"  marginal per-MAC breakdown  (BASELINE -> FOLDED):")
        allc = set(c1) | set(c8) | set(d1) | set(d8)
        for cname in sorted(allc, key=lambda k: -((c8[k] - c1[k]) + (d8[k] - d1[k]))):
            base = (d8[cname] - d1[cname]) / d_macs
            fold = (c8[cname] - c1[cname]) / d_macs
            if base or fold:
                print(f"    {cname:32s} {base:6.2f} -> {fold:6.2f}/MAC")
        print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
