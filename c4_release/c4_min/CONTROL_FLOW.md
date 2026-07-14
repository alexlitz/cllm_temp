# c4_min — Control Flow (depth=time, PC-driven dispatch)

**Status:** IMPLEMENTED + VALIDATED. This document extends `DESIGN.md` §(c)/(e)
with the control-flow mechanism the straight-line slice deferred ("Branches
require restoring PC-driven fetch … scoped as a documented extension"). Where the
two differ, `DESIGN.md` is authoritative for the substrate/architecture; this
file owns the **control-flow mechanism** and the `JMP`/`BZ`/`BNZ` implementation.

Code: `c4_min/control.py` (gadgets) + `c4_min/compiler.py`
(`compile_program_pc` / PC-truncated `run`). Tests: `c4_min/test_slice.py`
(`test_jmp_forward_skips`, `test_bz_*`, `test_bnz_*`,
`test_if_then_else_computed_predicate`). Oracle: `JMP`/`BZ`/`BNZ` PASS.

---

## 1. The question — fixed unroll vs. PC-driven fetch

The straight-line slice bakes instruction `code[k]` into transformer block `k`
and runs it **unconditionally** (block `k` = "execute the k-th instruction").
That works only when the executed instruction sequence is *known at compile
time*. With a branch, the instruction executed at time-step `k` depends on a
**computed** PC, so a fixed per-position bake cannot know which instruction to
run.

Two mechanisms were on the table:

| | Mechanism | Fetch | Cost |
|---|-----------|-------|------|
| **A. Fixed unroll** | block `k` = `code[k]`, always run | none (baked) | cannot branch |
| **B. PC-driven dispatch** | block `k` = *universal step*: run the instruction at the runtime PC | PC → instruction, in-block | control flow works |

**Decision: (B) PC-driven dispatch, on the existing single-position depth=time
substrate.** Each unrolled block is a *universal VM step* that dispatches on the
runtime PC. This is the cleanest fit: it keeps the whole VM state at one position
flowing through depth (DESIGN.md §(c)), reuses the exact-integer FFN rule
compiler, and needs **no attention and no core-model change** — only *which
guard* selects each op changes (from an always-on `ONE` window to a PC one-hot).

**Why not a content-match attention over a code table (the DESIGN §(e) `fetch`
row)?** That is the general fetch (needed if the code table were materialised at
many positions, or for unbounded loops). But the pass-ceiling corpus (332
pure-minimal programs) has **bounded depth** (max 36 steps, deep loops >40
excluded), so a *bounded unroll* of universal step-blocks suffices — PC-driven
selection inside the FFN, no cross-position attention. Attention is left as
identity, exactly as in the slice.

### Depth = **executed-step** time

The unroll length is `max_steps` (default `2*n_code + 4`, a safe bound since every
op advances ≥1 executed step). Block-group `k` is "the k-th *executed* VM step"
— **not** the k-th instruction. A forward branch that skips instructions simply
means step `k+1` dispatches from the post-branch PC; the skipped instructions are
never a step. `run` truncates the emitted trace at the first step that executed
`HALT`, so the returned trace length == the reference's executed-step count.

---

## 2. The mechanism, block by block

Each executed step is **four pure-FFN blocks** (single residual position; no
attention). All state (`AX/SP/BP/PC/STACK0`) lives in scalar residual bands and
persists across blocks via the additive residual (DESIGN.md §(b)/(c)).

```
per step k:
  1. FETCH    : PC (scalar) -> PC_IS[i] one-hot (i in 0..n_code-1)  +  AX_ZERO
  2. DISPATCH : rules gated on PC_IS[i] apply code[i]'s AX/STACK0 effect
                AND the PC update (sequential +1, or branch target)
  3. FOLD     : AX mod-256 (unconditional; no-op unless AX overflowed a byte)
  4. EMIT     : OUT_k <- AX ; HALT_SEEN_k <- (cumulative HALTED)
```

### 2.1 FETCH — PC one-hot + branch predicate (`control.compile_pc_fetch`)

`PC_IS[i] == (PC == i)` is an **exact integer one-hot** built from a unit
triangular pulse:

```
PC_IS[i] = tri_i(PC) = relu(PC-(i-1)) - 2·relu(PC-i) + relu(PC-(i+1))
```

which is `1` at `PC==i` and `0` at every other integer. Each `relu` is the exact
`silu`-based identity `relu(z) = silu(RELU_S·z)/RELU_S` (RELU_S=200); one shared
relu unit is baked per distinct threshold `t ∈ [-1 .. n_code]` and routed with
`+1/-2/+1` coefficients (normalised by `1/RELU_S`).

`AX_ZERO == (AX == 0)` is the **BZ/BNZ predicate**, `AX_ZERO = relu(1 - AX)`
(AX is a non-negative 8-bit int, so this is `1` at AX==0 and `0` for AX≥1). One
relu unit.

Because the residual is additive, these bands are **SET, not incremented**: each
target band gets a `silu`-identity self-clear unit (`up=S, gate=old_band,
down=-1/silu(S)` → subtracts the previous step's value) so re-running FETCH every
step is idempotent.

### 2.2 DISPATCH — PC-gated op + PC update (`control.dispatch_rules`)

One `FFNRule` set per code index `i`, all guarded on the window
`(PC_IS[i], 0.5, 1.5)` — i.e. the rule fires **iff PC == i**. The data effect
(AX / STACK0 writes) is byte-identical to the straight-line slice's
`_step_rules`; only the guard changed from the always-on `ONE` to `PC_IS[i]`. So
control flow adds **zero new arithmetic** — it is entirely a question of *which*
guard selects each op.

The **PC update** is an extra write into the `PC` band, per op (recall PC==i when
the rule fires):

| op | PC write (LinearExpr, gated on PC_IS[i]) | effect |
|----|------------------------------------------|--------|
| non-branch | `+1` | `PC := i+1` (sequential) |
| `JMP t` | `+(t - i)` | `PC := t` |
| `BZ t` | `1 + AX_ZERO·(t - i - 1)` | AX==0 → `PC:=t` ; else `PC:=i+1` |
| `BNZ t` | `(t - i) + AX_ZERO·(1 - (t - i))` | AX!=0 → `PC:=t` ; else `PC:=i+1` |
| `HALT` | `+0` | freeze PC (halted step re-dispatches HALT) |

The branch algebra is the whole trick: `BZ`'s delta evaluates to `t-i` when
`AX_ZERO=1` (taken) and `1` when `AX_ZERO=0` (fall-through) — a single linear
write in the two materialised bands, no second-order gate needed. `AX_ZERO` is
read from the residual (materialised by FETCH from the *pre-step* AX), so the
branch tests AX as it stood at the start of the step, exactly per ISA_SPEC §3.12.

Target semantics match the reference oracle: `imm` is an **instruction index**
(`SymbolicProgramState.resolve_static_target_idx` resolves `0 <= target <
len(code)` as an index first; `isa.interpret` uses the same index-PC), so `JMP 2`
lands on `code[2]`.

### 2.3 FOLD + EMIT

`FOLD` is the slice's exact mod-256 fold (`compile_ffn.compile_fold`) run every
step. It is a **no-op unless `AX ≥ 256`**, so it is safe to run unconditionally
even when the dispatched op was not an ALU op (ADD/SUB are the only overflow
sources, and both leave AX in `[0, 511]`). `EMIT` copies post-step `AX` into the
step's private `OUT_k` slot (write-once, so additive is fine) and snapshots the
cumulative `HALTED` band into `HALT_SEEN_k` (the trace-truncation signal).

---

## 3. Validation

- **Oracle (`c4_min/run_oracle.py`):** `JMP`, `BZ` (taken+not-taken), `BNZ`
  (taken+not-taken) all flip `0 → PASS`. `ADD/SUB/PSH/IMM` unchanged. Op-class
  PASS 4/30 → 7/30. (The remaining fails are un-built op-classes — CMP, bitwise,
  shift, memory, calling convention — out of this lane's scope.)
- **If-then-else, computed predicate (`test_if_then_else_computed_predicate`):**
  `if a==b then X else Y` compiled as `IMM a; PSH; IMM b; SUB; BZ then; IMM Y;
  JMP end; IMM X; HALT`. The branch depends on the **runtime** `a-b`; both the
  THEN path (`a==b`) and the ELSE path (`a!=b`, one extra JMP step) decode
  **byte-exact** vs the reference interpreter, and the not-taken branch's
  instructions are correctly skipped. Verified for `(5,5)`, `(5,7)`, `(9,9)`,
  `(3,8)`.
- **Full suite:** `test_slice.py` 12/12 + `test_oracle_harness.py` 6/6 pass.

## 4. Core-model impact — NONE (flag for consolidation)

**No changes to `model.py`** (the runtime `Transformer`). Control flow is
implemented purely as **additive FFN gadgets** (`control.py`) on the existing
single-position depth=time substrate + a routing check in `compiler.py`
(`has_control_flow` → `compile_program_pc`). The FFN rule/attention DSL, the
model forward, and the emission/decode contract are all unchanged. This keeps the
change union-merge-friendly and confirms the substrate architecture (DESIGN.md
§(e)) already supports control flow without a core rework.

## 5. Reachability of the 332 ceiling

Forward branches (the if-family: `if_gt/lt/eq/var`) are the only control-flow the
332 pure-minimal corpus needs — it has **no unbounded loops** (max 36 steps).
This mechanism handles arbitrary forward `JMP`/`BZ`/`BNZ` (and, by the same PC
write, backward targets too, within the bounded unroll). So the control-flow
substrate is **not** a blocker to the 332 ceiling; the remaining gap is the
per-op-class compute (CMP result → AX, bitwise, shift, memory, calling
convention), which is independent op-rule work.
