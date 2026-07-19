# Model-runs-C, FULL path: a COMPILER in the weights — C source in → runs it (2026-07-14)

Branch: `nibble-model-runs-c-full`. Substrate: `c4_min/` (the tiny CPU-only
BLOG_SPEC foundation). Authority: `docs/BLOG_SPEC.md` §"Model that Directly Runs C
Code" + §"Baking Prompts/Programs into the Transformer Weights". Predecessor:
`docs/NIBBLE_MODEL_RUNS_C_2026_07_14.md` (proved the EMIT/handoff *mechanism*).

## What this delivers (the headline)

**C source in → a transformer compiles it to bytecode → the transformer runs that
bytecode → correct result, compiler in the WEIGHTS, no tool calls.**

The predecessor session proved the compile-then-execute HANDOFF (an `EMIT` opcode
writes a produced instruction word into code memory, `JMP` hands off, the universal
fetch runs it). But its "generator" had the produced program's opcodes/immediates
**hardcoded in its own bytecode** — it was a code *emitter*, not a *compiler*: it
never read any source. This session closes that gap and runs the FULL path:

```
$ python c4_min/demo_model_runs_c.py

  C source  model result  produced bytecode
------------------------------------------------------------------------
     2+3*4            14  IMM 2; PSH; IMM 3; PSH; IMM 4; MUL; ADD; HALT
     2*3+4            10  IMM 2; PSH; IMM 3; MUL; PSH; IMM 4; ADD; HALT
     1+2+3             6  IMM 1; PSH; IMM 2; ADD; PSH; IMM 3; ADD; HALT
     2*3*4            24  IMM 2; PSH; IMM 3; MUL; PSH; IMM 4; MUL; HALT
     4+5*6            34  IMM 4; PSH; IMM 5; PSH; IMM 6; MUL; ADD; HALT
     3*4+5            17  IMM 3; PSH; IMM 4; MUL; PSH; IMM 5; ADD; HALT
```

The C expression is loaded **as data** (a source string in a `SRC` band); the
compiler is **baked into the transformer weights**; the input state carries no
bytecode at all (`CODE_WORD` is all-zero at load). The model reads the source
characters, decides `*`-over-`+` precedence *at runtime*, EMITs the compiled
bytecode into empty code memory, jumps to it, and the universal fetch executes the
freshly-produced program. The produced bytecode is the **exact `op | imm<<8` packed
encoding the real c4 compiler emits** (`code[code_pos] = op | (imm<<8)`,
`bundler/c4_compile.c:352`): for `2+3*4`, `IMM 2; PSH; IMM 3; PSH; IMM 4; MUL; ADD`
= `2+(3*4)` = **14**, byte-identical to c4's own output.

## The one missing capability a compiler needs: reading source

The predecessor's handoff machine had a full universal interpreter + `EMIT`, but no
way to READ its input. A real compiler must load the source at a runtime address.
This session adds that as ONE new op, plus the arithmetic/memory the produced
programs and the parser need — every one a bilinear read/write select over a data
band, the same gadget family the fetch already uses:

| op | effect | gadget |
|---|---|---|
| **`LC`** | `AX := SRC[AX]` (read C source char) | product-select of `SRC` by the AX one-hot — the write-mirror of the fetch's `WORD = Σ PC_IS[i]·CODE_WORD[i]`, addressed by AX |
| `LI` | `AX := MEM[AX]` (read compiler variable) | same select over scratch `MEM` |
| `SI` | `MEM[STACK0] := AX` (write variable) | store-select over `MEM` by the STACK0 one-hot |
| `MUL` | `AX := STACK0·AX` (2+3*4 needs multiply) | one masked bilinear product, exact `POW2=256` fold |
| `EMIT` | `CODE_WORD[IMM] := AX + 256·STACK0`; pop | store-select over `CODE_WORD`; opcode **30** (27 = MUL) |

`LC` is the load-bearing addition: it is the compiler's window into its input — the
same `src[pos]` byte load the c4 lexer's `next()` does. Nothing about `2+3*4` is in
the compiler bytecode; the digits and operators are read out of `SRC` at runtime.

### The stack had to become real (SP-indexed)

The universal substrate carried a *single* `STACK0` scalar as the top-of-stack
mirror — correct only at depth ≤ 1. But `2+3*4` produces a **depth-2** program (push
2, push 3, push 4, MUL pops 4&3→12, ADD pops 12&2→14): on a single mirror it gives
`15` (12+3) not `14`. We wire in the proven **SP-indexed arbitrary-depth stack**
(`c4_min/stack.py`, task #525): `STACK[0..K-1]` cells addressed by an `SP` depth
counter via the exact-integer `SP_IS[i]=(SP==i)` one-hot, `STACK0 = STACK[SP-1]`
re-derived every step. PSH writes `STACK[SP]` and `SP+=1`; ADD/SUB/MUL pop (`SP-=1`).
Now the produced `2+3*4` evaluates to **14** on the model.

## How the compiler works (the bytecode)

`expr_compiler_bytecode()` (135 c4_min instructions) compiles the minimal C-subset
grammar `D op D op D` over single ASCII digits and `+`/`*` — a straight-line
instance of the c4 `expr()`/`stmt()` recursive descent, with the operator
**precedence decided by runtime branches** on the operator characters:

- read `src[1]` (op1); `BZ` on `(op1 - '*')` chooses the multiply-first vs
  add-first skeleton (the precedence lookahead the c4 compiler does);
- for each produced instruction: put the immediate on the stack (a digit read from
  `SRC` via `LC` and converted `-'0'`, or a constant `0`), set `AX` to the produced
  opcode constant, `EMIT` it into the next output slot;
- `op1=='+'` with `op2=='*'` defers the ADD to the end (`IMM d0;PSH;IMM d1;PSH;IMM
  d2;MUL;ADD`), so `2+3*4` binds `3*4` first — correct precedence;
- finally `EMIT HALT` and `JMP OUTBASE` — the HANDOFF to the produced program.

All four precedence cases (`+/+`, `+/*`, `*/+`, `*/*`) are covered by runtime
branches; the produced bytecode changes with the source operators (see the demo
table). `EMIT` pops the produced-immediate it consumed, so a compiler that emits
many instructions keeps a balanced stack.

## Two ways to run it (both proven byte-exact on the neural model)

1. **Compiler in DATA** (`CompilerMachine`): the compiler bytecode is loaded into
   `CODE_WORD` memory, the C source into `SRC`. `run(prog, source="2+3*4") → 14`.
   One fixed-weight interpreter; swap the data to compile any expression.

2. **Compiler in WEIGHTS** (`BakedCompilerMachine`): the compiler bytecode is baked
   into the FFN via a **hybrid word-select** — `WORD = Σ_{i<GEN} PC_IS[i]·word_i`
   (baked constants, the read-only code segment) `+ Σ_{i≥GEN} PC_IS[i]·CODE_WORD[i]`
   (the writable memory the produced program lives in). The input state carries
   **only the C source**; `CODE_WORD` is all-zero at load. `run("2+3*4") → 14` with
   no bytecode anywhere in the data. THE COMPILER IS THE WEIGHTS.

Both use the exact `POW2=256` normaliser on the word select (a `silu(60)/60`
reciprocal is not exact and flipped a produced word's low byte `0x201 → 0x200`).

### Tests (`c4_min/test_nibble_compiler.py`, 10/10)

- ISA ops byte-exact vs the reference: `MUL` folds the stack, `LC` reads any source
  address, `LI/SI` roundtrip, the SP-stack evaluates depth-2 `2+3*4 → 14`.
- the compiler compiles+runs all 8 test expressions with correct precedence;
- the produced bytecode for `2+3*4` is byte-identical to c4's `IMM 2;PSH;IMM 3;PSH;
  IMM 4;MUL;ADD;HALT`; the produced slots are 0 at load (program produced at
  runtime); the model matches the reference interpreter exactly;
- the BAKED compiler runs with **no bytecode in the input** (only the source).

Predecessor `test_handoff.py` (6/6) and `test_nibble_bake.py` (12/12) unregressed.

## Scoping the FULL c4 compiler bake

The minimal-subset compiler proves the *mechanism* end-to-end. Baking the full
`bundler/c4_compile.c` (964 LOC) is a size, not a research, problem.

### Full-compiler bytecode size

- **Measured (built `gcc -m32 -w -fpermissive`)**: `int main(){return 2+3*4;}` →
  **11** instrs; a 10-iteration `while`-loop program → **45** instrs (≈ 4
  instrs/line of C). `bundler/c4_compile.c` self-compile hangs natively (a known
  LP64 port bug in the JSR path — the compiler *logic* is fine; loop/expr programs
  compile), so we estimate from density.
- **LOC scaling**: 964 LOC × ~4 instrs/line ≈ **~3,500 instructions**.
- **emit-site density**: 81 static `emit(` sites, mostly inside the recursive
  `expr()`/`stmt()` descent → ~3,000–3,600 dynamic. Agrees with the mainstream-c4
  self-host figure (~2,800–3,500). **Use ~3,500 as the full-compiler size.**

### Dimensional cost on this substrate (measured)

| code_size | D (residual width) |
|---|---|
| 64 | 364 |
| 176 (our compiler) | 700 |
| 512 | 1,716 |
| 1,024 | 3,284 |

`D ≈ 3·code_size + const` (each slot carries `CODE_WORD` + `PC_IS` + `ADDR_IS`; the
compiler-in-weights baked word rides an FFN bias, not the residual). A ~3,500-instr
compiler + its produced-program slots needs `code_size ≈ 4,000` → **D ≈ 12k**, and
the fetch/decode FFN is `O(code_size)` wide (one product/detector unit per slot).
That is a mid-size build, not tiny — but every block is the SAME gadget family
already proven here; nothing new is required except scale.

### ISA gap (minimal → full)

The minimal compiler and its produced programs use `IMM/LEA/PSH/ADD/SUB/MUL/LC/LI/
SI/BZ/BNZ/JMP/HALT/EMIT`. The full c4 compiler additionally emits and itself uses:
`JSR/ENT/LEV/ADJ` (calls — for its recursive descent), `DIV/MOD`, the bitwise/
compare ops `OR/XOR/AND/EQ/NE/GT/GE/LE/SHL/SHR`, and the syscalls `OPEN/READ/CLOS/
PRTF/MALC/MSET/MCMP/GETCHAR/PUTCHAR`. **All of these already exist as c4_min nibble
gadgets** (tasks #529–534: cmp, bitwise, muldivmod, callconv, I/O) — wiring them
into this universal dispatch table is **1 `FFNRule`/opcode**, the same M-effort the
predecessor scoped. The compiler-in-weights fetch (baking) is done; the calling
convention needs the SP-stack we already added to carry the return-address frame.

### The remaining lever: expert de-duplication

The `O(code_size)` fetch is the size driver. The spec's stated optimisation
(experts for slots with the same `(op,imm)` share; nibble-EQ results are shared
across addresses) collapses it toward `O(distinct words)` ≈ a few hundred. Not yet
applied here (we keep 1 unit/slot for a clean 1:1 read); it is the single biggest
shrink for a practical full bake. Prototyped in `docs/NIBBLE_EXPERT_DEDUP_*` and
task #544.

## Verdict

The **model-runs-C FULL path is proven end-to-end at minimal-subset scale**: a
transformer, with a **compiler baked into its weights** and **only C source as
input**, reads the source, compiles it to the exact c4 `op|imm<<8` bytecode with
correct operator precedence, writes it into its own code memory, and runs it —
`2+3*4 → 14`, no tool calls. The full `c4_compile.c` bake is dimensionally scoped
(≈3,500 instrs → D≈12k on this substrate, or a few hundred with expert de-dup) and
gated only on wiring the remaining already-built ISA gadgets into the dispatch — a
mid-size build, not a research risk.

### Artifacts (all on `nibble-model-runs-c-full`)

- `c4_min/nibble_compiler.py` — the compiler VM substrate (LC/LI/SI/MUL/EMIT +
  SP-indexed stack), the minimal-subset compiler bytecode (`expr_compiler_bytecode`),
  `CompilerMachine` (compiler in data), and `BakedCompilerMachine` (compiler in
  weights, hybrid baked/memory fetch).
- `c4_min/test_nibble_compiler.py` — 10/10 (ISA ops, SP-stack, compiler precedence,
  produced-slots-empty, reference-exact, baked no-bytecode-in-input, baked runs).
- `c4_min/demo_model_runs_c.py` — the end-to-end demo (C source → produced bytecode
  → result, compiler in the weights).
- `c4_min/stack.py` — the SP-indexed arbitrary-depth stack (brought from #525).
- `c4_min/nibble_handoff.py`, `nibble_bake.py`, `universal.py` — the proven
  handoff/baking/universal-fetch halves this builds on.
- `bundler/c4_compile.c` — the 964-LOC c4 compiler (the full-scale bake target).
