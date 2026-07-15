# The Universal NIBBLE VM base — one fixed-weight interpreter, any program

Date: 2026-07-14 · Branch: `nibble-skeleton` (base `greenfield-blogspec-foundation`)

## What this is

The **coherent universal-nibble-VM base** the opcode fan-out builds on. It ports
three proven scalar-track mechanisms onto the BLOG_SPEC nibble foundation, so
**ONE fixed-weight transformer runs ANY program supplied as data**, on the spec's
16-nibble register representation, emitting the 30-token frame each step,
re-quantised by the vanilla autoregressive token round-trip (no `torch.round`).

| ported from | mechanism | where it lands here |
|-------------|-----------|---------------------|
| `gf-assembled:c4_min/control.py` | **runtime-PC dispatch** — live PC selects the instruction | `compile_pc_fetch` + `base_dispatch_rules` |
| `greenfield-universal:c4_min/universal.py` | **universal fetch-from-data** — program in DATA bands, one model many programs | `load_program` + `compile_code_select` + `compile_opcode_decode` |
| `greenfield-vanilla-requant:c4_min/recurrent_vanilla.py` | **vanilla requant** — argmax→token→re-embed, no `round()` | `_emit_and_reembed` / `_snap_lane` |

Deliverable files (added alongside the foundation `blogspec_*`, nothing removed):

| file | role |
|------|------|
| `c4_min/nibble_vm_layout.py` | residual layout: nibble register bands + scalar value lanes + CODE data bands + fetch/decode/dispatch scratch |
| `c4_min/nibble_vm.py` | the universal step-block builder + the vanilla requant driver + the **dispatch interface** |
| `c4_min/test_nibble_vm.py` | 9 tests: pillars + universality + 402-step deep loop |

## The coherent nibble VM in one picture

The persistent VM state is the **nibble bands** (`PC AX SP BP STACK0`, 16 dims
each — the spec representation). One transformer **step-block** = one VM step,
applied recurrently (depth = time) to a single residual position. Its seven FFN
sub-blocks (attention zeroed → identity; softmax1+ALiBi still runs every block):

```
 nibble bands ──1─▶ scalar value lanes ──2─▶ PC one-hot + AX_ZERO
   (canonical)      (per-step image)          PC_IS[i]=(PC==i)
                                                    │
                          ┌─────────3── fetch@PC from DATA ──────────┐
                          │  OP_VAL = Σ PC_IS[i]·CODE_OP[i]          │
                          │  IMM    = Σ PC_IS[i]·CODE_IMM[i]         │
                          └──────────────────┬──────────────────────┘
                                    4─▶ OP_IS[op] = (OP_VAL==op)   (decode)
                                             │
                       5─▶ DISPATCH: one FFNRule per opcode VALUE, gated on
                           OP_IS[op] → writes value lanes + PC delta
                                             │
                       6─▶ BRANCH: bilinear BZ/BNZ PC update (AX_ZERO·IMM·PC)
                                             │
                       7─▶ FOLD: AX_VAL mod 256
                                             │
   nibble bands ◀── vanilla requant ─────────┘
   (next step)     emit 30-token frame: argmax value token per register →
                   re-embed its bytes' nibbles (the LM-head snap, no round)
```

Block 1 (`compile_nibble_to_scalar`) is **the bridge**: it recomposes each
register `VAL = Σ_j 16^j·nibble_j` (silu-identity reads, exact for nibbles 0..15,
`16^4=65536 < 2^24` so fp32-exact) so the *proven exact scalar dispatch algebra*
from `control.py`/`universal.py` runs unchanged. The nibble band stays canonical;
the scalar lane is its per-step image; the driver's frame round-trip writes the
scalar next-state back into the nibble bands.

### Residual layout (`code_size=12` → `D=192`)

```
PC[0:16] AX[16:32] SP[32:48] BP[48:64] STACK0[64:80]      ← 16-nibble registers
CUR_NIB[80:96] CTX[96:102]                                 ← frame ingest scratch
PC_VAL AX_VAL SP_VAL BP_VAL STK_VAL BP_LOW [103:109]       ← scalar value lanes
CODE_OP[109:121] CODE_IMM[121:133]                         ← PROGRAM as DATA
PC_IS[133:145] AX_ZERO OP_VAL IMM [145:148]                ← fetch/decode scratch
OP_IS[148:188]                                             ← decoded opcode one-hot
HALTED[188] ONE[189]                                       ← flag + constant lane
```

`BP_LOW` = only nibbles 0,1 of BP (its low byte); LEA is the 8-bit op
`AX=(BP+imm)&0xFF`, so it adds the frame-pointer's low byte and one mod-256 fold
keeps AX a byte.

## The four proofs (the mission checklist)

**Runtime-PC dispatch.** `compile_pc_fetch` builds the exact-integer PC one-hot
via the triangular pulse `tri_i(x)=relu(x−(i−1))−2relu(x−i)+relu(x−(i+1))` on the
recomposed `PC_VAL`. The executed instruction is chosen by the *runtime* one-hot,
so a forward branch that skips instructions Just Works
(`test_branch_skips_instruction_via_runtime_pc`: `JMP` over `IMM 99` never emits 99).

**Universal fetch-from-data.** `load_program` writes each `[op,imm]` cell into the
`CODE_OP[i]`/`CODE_IMM[i]` DATA bands of the *initial state* — as INPUT, never
baked into a weight. `compile_code_select` does the bilinear read
`OP_VAL=Σ PC_IS[i]·CODE_OP[i]` (SwiGLU `silu(S·PC_IS[i])·CODE_op[i]/silu(S)`),
`compile_opcode_decode` turns `OP_VAL` into `OP_IS[op]`, and dispatch has ONE rule
per opcode VALUE. So the weights are the interpreter and the program is data:
`test_one_model_runs_many_programs` runs **11 programs** on ONE model with an
**invariant weight hash**; two independently-built models of the same `code_size`
are **bit-identical** (`weight_hash` `7379cd20…`).

**Vanilla requant.** Between steps `_emit_and_reembed` snaps each register lane to
an exact integer via the LM-head value argmax `argmax_v(2·v·x − v²)` (the emitted
value token) and re-embeds its little-endian byte nibbles into the register band
— **this is the 30-token frame**. There is NO `torch.round` on the exec path
(`test_no_torch_round_in_exec_path` AST-guards the `nibble_vm` + `blogspec_model`
modules). Frames are exactly `30·steps + BOS + HALT` tokens.

**Nibble + softmax1 + ALiBi.** The step-block IS the `blogspec_model.Transformer`
(softmax1 ZFOD + geometric ALiBi slopes + SwiGLU). Registers are 16-nibble bands
(`test_registers_are_nibble_bands`).

**Deep loop / unbounded.** `test_deep_loop_countdown_exact`: a countdown from 100
runs **402 VM steps** on ONE recurrently-applied step-block, byte-exact vs the
reference, SP oscillating `0x10000 ↔ 0xFFFC`. A countdown from 250 runs **1002
steps** exact. The step count is the program's real length — decoupled from the
bake; the requant annihilates the O(1e-6) SwiGLU residue every step so error never
compounds.

---

# ★ THE NIBBLE DISPATCH INTERFACE (the fan-out contract)

**This is what the comparison / bitwise / muldiv / memory / callconv agents plug
into.** An opcode's effect is expressed as a **dispatch rule on the value lanes**,
gated on the decoded opcode one-hot. To add an opcode you append rules — nothing
else in the interpreter changes.

### The bands a dispatch rule may read / write

Read (the pre-op machine state, recomposed from the nibble bands by block 1):

| band | meaning |
|------|---------|
| `L.AX_VAL L.SP_VAL L.BP_VAL L.STK_VAL L.PC_VAL` | current register scalars |
| `L.BP_LOW` | BP's low byte (for 8-bit frame-relative address ops) |
| `L.IMM` | the fetched immediate (from data memory, at the live PC) |
| `L.AX_ZERO` | `(AX==0)` predicate (already materialised by fetch) |
| `L.OP_IS + op` | the decoded opcode one-hot — **the gate for every rule** |
| `L.ONE` | constant 1.0 lane |

Write (the next-state, SET semantics — include `−old` where you replace):

| band | meaning |
|------|---------|
| `L.AX_VAL` | next AX (folded mod 256 downstream by block 7) |
| `L.SP_VAL L.BP_VAL L.STK_VAL` | next SP / BP / stack-top |
| `L.PC_VAL` | PC **delta** (`+1` sequential is the default an op overrides) |
| `L.HALTED` | latch to 1.0 to stop the run |

The driver's frame round-trip writes whatever ends up in the value lanes back
into the canonical nibble bands.

### How to express an opcode

One or more `FFNRule`s, each `FFNRule(when, write)`:

```python
from c4_min.dsl import FFNRule, LinearExpr

def G(L, op):                       # the gate: fires iff decoded opcode == op
    return [(L.OP_IS + op, 0.5, 1.5)]

# IMM: AX = imm ; PC += 1      (SET: the -AX cancels the old image)
FFNRule(G(L, isa.IMM), {
    L.AX_VAL: LinearExpr.of(L.IMM, 1.0) + LinearExpr.of(L.AX_VAL, -1.0),
    L.PC_VAL: LinearExpr.c(1.0),
})

# ADD: AX = pop + AX ; SP += 4 ; PC += 1   (fold mod 256 is automatic)
FFNRule(G(L, isa.ADD), {
    L.AX_VAL: LinearExpr.of(L.STK_VAL, 1.0),   # STACK0 + AX  (the -AX seed cancels)
    L.SP_VAL: LinearExpr.c(4.0),
    L.PC_VAL: LinearExpr.c(1.0),
})
```

`LinearExpr` is `dst += Σ coeff·src_band + const`. `compile_ffn` turns each
`(rule, dst)` into one SwiGLU hidden unit: `gate = write-expr`, `up = S·(guard
indicator)`, `down` routes `guard·expr` into `dst`. Guards on a one-hot band are
exact 0/1.

### The base op table (`base_dispatch_rules`, the reference gadgets)

| op | effect | notes |
|----|--------|-------|
| `IMM` | `AX = imm` | |
| `LEA` | `AX = (BP + imm) & 0xFF` | reads `BP_LOW` (8-bit frame-relative addr) |
| `PSH` | `STACK0 = AX ; SP −= 4` | 4-byte-aligned stack |
| `ADD` | `AX = STACK0 + AX ; SP += 4` | mod-256 fold downstream |
| `SUB` | `AX = STACK0 − AX + 256 ; SP += 4` | two's-complement then fold |
| `JMP` | `PC = imm` | delta `IMM − PC` |
| `BZ`/`BNZ` | conditional `PC = imm` | **bilinear** — see below |
| `HALT` | latch `HALTED`, freeze PC | ends the run |

### Two rules for the fan-out

1. **Linear vs bilinear.** A single `FFNRule` write is *linear* in the read bands
   (`gate` is one linear combination). If your effect needs the **product of two
   live bands** (e.g. `AX_ZERO · IMM` for a data-dependent branch target), it can
   NOT be a plain `FFNRule` — bake a dedicated SwiGLU product block (the
   `up = BIG·(op_hot + predicate − 1.5)` AND-gate pattern in `compile_branch_delta`,
   which fires iff *both* 0/1 inputs hold and carries the value through `gate`).
   BZ/BNZ use exactly this; comparison/memory gadgets that select on a runtime
   value will too.

2. **Data-flow across blocks.** An additive-residual FFN cannot read a band it
   writes in the *same* block. If your gadget needs `A` then `B(A)`, split it into
   two sub-blocks (this is why fetch = `compile_pc_fetch` **then**
   `compile_code_select`, and dispatch **then** `compile_branch_delta`). Add your
   sub-blocks to the `ffn_specs` list in `build_step_model` in dependency order.

### Where multi-byte / nibble-native gadgets slot in

The value lanes are 8-bit-folded (AX) or small (PC) — fine for the byte ops. A
**nibble-native** gadget (16-bit add carry, bitwise per-nibble, shift) operates
directly on the register **nibble bands** (`L.AX + j`, `L.STACK0 + j`) *before*
block 1's recompose, or writes extra nibbles the recompose then reads. The
`blogspec_compiler.nibble_add_gadget` / `nibble_sub_gadget` are the reference
per-nibble SwiGLU ALU primitives to build these from. To widen a register past
8 bits, raise `hi_nibbles` in `compile_nibble_to_scalar` (staying under the
`16^{hi} < 2^24` fp32-exact bound, i.e. `hi ≤ 5`) or carry the wide result on the
nibble bands directly and skip the scalar lane for that op.

## How to run

```bash
PYTHONPATH=$(pwd) python -m pytest c4_min/test_nibble_vm.py -q      # 9 tests
PYTHONPATH=$(pwd) python c4_min/test_nibble_vm.py                    # verbose
```

```python
from c4_min import isa, nibble_vm as N
vm = N.NibbleVM(code_size=12)                       # ONE fixed-weight interpreter
for prog in (P1, P2, P3):                           # ANY programs, no rebuild
    tokens, frames = vm.run(prog)
    assert N.decode_trace(frames) == isa.interpret(isa.assemble(prog))
    assert vm.weight_hash() == vm.weight_hash()      # invariant
```

## Status / limits

- Base ops proven: `IMM LEA PSH ADD SUB JMP BZ BNZ HALT` (the 8-bit subset).
  `POP`/`ADJ`/`NOP` are trivial SP/PC deltas — add as one-line `FFNRule`s.
- SP/BP tracked as full 32-bit (via the 5-nibble recompose + wide value vocab);
  the byte ALU folds AX/STACK0 mod 256.
- `code_size` sizes the PC/data bands (the only program-dependent shape); the
  weights are otherwise a pure function of `code_size` — the interpreter.
- Memory (`LI/SI`), the calling convention (`JSR/ENT/LEV`), comparisons, bitwise,
  shifts, mul/div are the fan-out layers this base is designed to carry — each is
  a new `OP_IS[op]`-gated rule set (+ a product/sub-block where non-linear), per
  the interface above.
