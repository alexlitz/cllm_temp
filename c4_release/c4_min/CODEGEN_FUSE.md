# CODEGEN_FUSE — the memory-operand-ALU codegen peephole (C4_CODEGEN_FUSE)

**Files:** `codegen_fuse.py` (the pass), `measure_codegen_fuse.py` (the
forwards/MAC + byte-exactness measurement), `test_codegen_fuse.py` (the gate).
**Flag:** `C4_CODEGEN_FUSE` (default **OFF** — `compile_c` output, and therefore the
neural golden, is byte-identical; this is a compiler-only change, no model weights).

## The goal (from the a56fafd0 census)

`a56fafd0` proved the emulation runs **1 forward per c4 instruction**, and a
per-opclass census of a c4-compiled matmul (`_analyze_opclass_steps`) showed
**~101 forwards/MAC**, dominated by operand-load/push/apply and loop-index address
arithmetic. `nibble_exact_steps` built the fused memory-operand ALU opcode

    <OP>M [addr]        #  AX = mem[addr] <op> AX   (ADDM/SUBM/MULM/DIVM/MODM, 41-45)

which does the addressed load + the ALU fold in ONE forward — replacing the base
ISA's four-instruction `<load a>; PSH; <compute b>; <OP>`. But the compiler never
emitted `<OP>M`, so no real program benefited. **This is the missing codegen.**

## What was built — the ADDM peephole (LEVER a)

`fuse_bytecode(code)` is a post-emission peephole over the flat c4 word list (the
`src.compiler` compiler is single-pass with no IR, so this is the only place to
fold). It rewrites

    IMM addr ; LI/LC ; PSH ; <stack-balanced b> ; <OP>
    ->        <stack-balanced b> ; <OP>M (addr & 0xFF)

when the LEFT ALU operand is a PURE **absolute-address** load. This is byte-exact
of result: c4 lowers `a <op> b` as `compile a -> AX ; PSH ; compile b -> AX ; <OP>`
where `<OP>` computes `pop()(=a) <op> AX(=b)`; `<OP>M [addr]` computes `mem[addr]
<op> AX` with the SAME operand order, so folding away the left-operand load leaves
AX = b at the `<OP>M` and yields the identical `mem[addr] <op> b = a <op> b`.

Correctness details that matter:
- **address masking** — `IMM addr` masks the constant to the value width (0xFF)
  before `LI` reads `mem[AX]`, so `IMM addr; LI` loads `mem[addr & 0xFF]`. `<OP>M
  imm` reads `mem[imm]` at full width. The fold therefore emits `<OP>M (addr &
  0xFF)` (the SAME effective cell). Missing this mask was a real bug found + fixed.
- **relocation** — each fold deletes 3 instructions, shifting every downstream PC
  target. The pass rebuilds with an old→new index map and re-patches every
  JMP/JSR/BZ/BNZ immediate, so control flow is preserved.
- **span safety** — the `b` code between the PSH and its matching `<OP>` must be
  stack-balanced and straight-line; a branch/call/label crossing the span aborts
  the fold (the pushed value could be consumed on another path).
- **store-aliasing guard** — `PSH` snapshots the VALUE `mem[addr]`; `<OP>M`
  RE-READS `mem[addr]`. A store (`SI`/`SC`) in the span could alias `addr`
  (`r = g + (g = 2)`), so any store in the span aborts the fold. (Found + fixed;
  regression-tested.)
- **fixpoint** — nested `a <op> (c <op> d)` folds the inner op on a later pass.

Validated: the folded bytecode runs **byte-exact on the actual neural exact-steps
model** (`test_codegen_fuse.py::test_folded_runs_on_neural_model`), not just on the
reference interpreter — the `<OP>M` it emits is executable.

## Measured result — HONEST

### On the censused matmul: 0 folds (the fold is INERT here)

    === fpmul-CALL ===    BASELINE 101.0/MAC  ->  FOLDED 101.0/MAC   (0 folds)
    === INLINE a*b/s ===  BASELINE  88.0/MAC  ->  FOLDED  88.0/MAC   (0 folds)

**Why:** the matmul's inner-MAC operands are `*ap` / `*bp2` — POINTER
DEREFERENCES: `mem[mem[frame_off]]`, a double indirection whose address is a
runtime value in AX (`LEA off; LI; LI`), NOT a compile-time immediate. And the
few directly-loaded operands are FRAME-relative (`LEA off`) locals, whose absolute
address `bp + off` is not known at compile time. `<OP>M`'s `imm` is an ABSOLUTE
address, so neither shape folds. **The ADDM peephole removes 0 of the ~101/MAC on
this kernel.**

### On a global-operand kernel: the fold fires, byte-exact

A kernel whose ALU operands are GLOBAL variables (absolute `IMM addr` loads):

    GLOBAL-operand kernel (4 global-operand ALU ops/iteration):
      per-iteration: BASELINE 51.0 -> FOLDED 39.0 forwards/iter   (byte-exact)

Each fold removes exactly 3 instructions (`IMM; LI; PSH`), so 4 folds/iter = 12
fewer forwards/iter. This proves the peephole is real and correct on programs
shaped for it — the win is `3 * (number of absolute-address ALU operands)`.

### Split of the ~101/MAC

| lever | removes on the matmul | why |
|---|---|---|
| (a) ADDM peephole | **0** | operands are `*ptr` / frame-local, not absolute-`IMM` loads |
| (b) address hoisting / strength reduction | **0 as implemented** (assessed) | see below |
| (c) irreducible | the rest | genuine loop-carried index arithmetic + the MAC itself |

## Loop-invariant hoisting / strength reduction (LEVER b) — ASSESSED, NOT built

The census attributes the bulk of the per-MAC cost to `imm (31/MAC) + push
(24/MAC) + load (19/MAC)` — the pointer-address recompute
`ap = ab + (p*K + r)*4` / `bp2 = bb + (r*N + q)*4` emitted fresh every inner
iteration (~38 of the ~48 inner-loop instructions per MAC). These addresses are
NOT loop-invariant (they change with `r`); they are **induction variables** —
`ap` advances by a constant `+4` and `bp2` by `+N*4` each `r++`. So the applicable
transform is **strength reduction** (init `ap` before the loop, `ap += 4` in the
body), not hoisting, and it could remove most of the address-recompute.

**Why it is not implemented here (tractability verdict):**
1. The `src.compiler` compiler is a strict single-pass recursive-descent emitter
   with **no IR, no basic-block graph, and no loop representation** — it never
   builds a structure to hoist from. Strength reduction would require, on the flat
   bytecode: reconstructing loops from back-edges, building a byte-stack def-use
   graph, proving each address subexpression is an affine induction variable, and
   proving the increment is loop-invariant and side-effect-free. That is a full
   optimizing-compiler pass on reverse-engineered structure — far beyond a
   byte-neutral peephole, and high-risk for byte-exactness.
2. Even if the addresses were strength-reduced, **the reduced operands still can't
   use `<OP>M`**: `*ap` is a pointer dereference (`mem[ap]` where `ap` is a runtime
   value), not an absolute-immediate address. Strength reduction would cut the
   `imm/push/load` address-arithmetic but leaves the MAC's own loads on the LI-of-AX
   path, orthogonal to this module's fold.

The honest conclusion: **on a pointer-walking matmul the memory-operand fold is
inert; the reducible cost is the induction-variable address arithmetic, which needs
a real loop optimizer (an IR the current compiler lacks), not this peephole.** The
peephole delivers a genuine, byte-exact `3 * n_absolute_operands` reduction on
global-operand code, and is gated OFF by default so nothing else changes.

## Reproduce

    python -m c4_min.measure_codegen_fuse          # forwards/MAC + byte-exactness
    python -m c4_min._analyze_opclass_steps        # the a56fafd0 ~101/MAC baseline
    python -m pytest c4_min/test_codegen_fuse.py -q
