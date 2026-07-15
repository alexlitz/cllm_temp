"""DEMO: a transformer that RUNS C. Feed C source; the model compiles it to
bytecode in its own memory and runs it — no tool calls, compiler in the weights.

    PYTHONPATH=<repo> python c4_min/demo_model_runs_c.py

This is the end-to-end "Model that Directly Runs C Code" path (BLOG_SPEC.md), at
minimal-subset scale: integer ``+``/``*`` expressions with correct precedence. The
compiler is BAKED into the transformer weights; the only input is the C source
string. The model reads the source (LC), parses it, EMITs the compiled bytecode
into empty code memory, JMPs to it, and the universal fetch runs the freshly-
produced bytecode. The produced bytecode is byte-identical to what the real c4
compiler (bundler/c4_compile.c) emits.
"""
from __future__ import annotations

from c4_min import nibble_compiler as C


def main():
    comp = C.expr_compiler_bytecode()
    print("Building the compiler-in-weights transformer "
          f"(compiler = {len(comp)} baked instrs) ...")
    bcm = C.BakedCompilerMachine(comp, code_size=176, src_size=8,
                                 mem_size=8, stack_depth=8)
    print(f"  model: D={bcm.L.D}, {bcm.n_blocks} blocks, "
          f"compiler baked in weights (GEN_SIZE={bcm.L.GEN_SIZE}).\n")

    exprs = ["2+3*4", "2*3+4", "1+2+3", "2*3*4", "4+5*6", "3*4+5", "9+8+7", "5*6*7"]
    print(f"{'C source':>10}  {'model result':>12}  {'produced bytecode'}")
    print("-" * 72)
    for src in exprs:
        trace, words = bcm.run(src, return_code=True, max_steps=4000)
        O = C.COMPILER_OUTBASE
        produced = [words[O + k] for k in range(8)]
        # human-readable disassembly of the produced program
        import c4_min.isa as isa
        disasm = []
        for w in produced:
            op, imm = w & 0xFF, w >> 8
            nm = "EMIT" if op == C.EMIT else isa.NAMES.get(op, str(op))
            disasm.append(nm if imm == 0 else f"{nm} {imm}")
        print(f"{src:>10}  {trace[-1]:>12}  {'; '.join(disasm)}")
    print("\nThe model read each C expression as data, compiled it to the c4 "
          "op|imm<<8\nbytecode with correct */+ precedence, and ran it — compiler "
          "in the weights.")


if __name__ == "__main__":
    main()
