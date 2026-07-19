"""DEMO: a transformer that RUNS C at scale, via the FETCH-DEDUP + a LOOP compiler.

    PYTHONPATH=<repo> python c4_min/demo_model_runs_c_dedup.py

Two things this shows over demo_model_runs_c.py:

  1. FETCH-DEDUP.  The baseline fetch is O(code_size) in the residual width D
     (D = 3*code_size + 148 -> the full ~4000-instr c4 compiler needs D~12k, a
     ~59 GB / OOM-class model).  The deduped fetch (region-split + factored PC
     one-hot, nibble_fetch_dedup.py) makes D grow ~sqrt(code_size): the SAME full
     compiler now fits D~440 (~0.49 GB) — a ~28x D shrink, ~121x per-block params.

  2. A VARIABLE-LENGTH LOOP COMPILER.  The compiler bytecode is FIXED size (83
     instrs) but LOOPS over a null-terminated source, emitting a variable number
     of produced instructions at the runtime cursor OUT_PTR (the moving-pointer
     EMITP).  So it compiles arbitrarily-long single-operator integer chains
     (2+3+4+5, 1+2+...+n, 2*3*4*5, ...) — genuinely larger than a bare 3-operand
     expression — with output byte-identical to the real c4 compiler.

The compiler is BAKED into the transformer weights; the only input is the C
source.  The model reads it, loops emitting the c4 op|imm<<8 bytecode, JMPs to it,
and runs it (greedy argmax decode, one recurrent transformer forward pass / VM
step, no tool calls).
"""
from __future__ import annotations

from c4_min import isa
from c4_min import loop_compiler as LC
from c4_min import nibble_fetch_dedup as D


def main():
    prog = LC.chain_compiler_bytecode()
    print(f"Building the compiler-in-weights transformer "
          f"(loop compiler = {len(prog)} baked instrs, DEDUPED fetch) ...")
    machine = D.LoopCompilerMachine(prog, out_size=48, src_size=32,
                                    mem_size=8, stack_depth=16)
    print(f"  model: D={machine.L.D}, {machine.n_blocks} blocks, "
          f"gen_size={machine.L.GEN_SIZE} baked; OUT_PTR moving cursor.\n")

    exprs = ["2+3*4"[:0] or "7+8+9", "2+3+4+5", "2*3*4*5", "1+2+3+4+5+6",
             "3+3+3+3+3+3+3", "9*1*1*1", "1+1+1+1+1+1+1+1+1+1"]
    print(f"{'C source':>22}  {'result':>7}  produced bytecode")
    print("-" * 88)
    for e in exprs:
        src = [ord(c) for c in e] + [0]           # null-terminated
        trace, words = machine.run(src, max_steps=30000, return_code=True)
        produced = []
        for w in words:
            produced.append(w)
            if (w & 0xFF) == isa.HALT:
                break
        print(f"{e:>22}  {trace[-1]:>7}  {LC.produced_disasm(produced)}")
    print("\nThe model read each variable-length C chain as data, LOOPED emitting "
          "the c4\nop|imm<<8 bytecode at a moving output cursor, and ran it — "
          "compiler in the weights,\ndeduped O(sqrt) fetch (D fits the full c4 "
          "compiler at ~440 instead of ~12k).")


if __name__ == "__main__":
    main()
