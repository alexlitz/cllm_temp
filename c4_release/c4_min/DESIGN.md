# c4_min — Green-field Minimal-ISA Compiler Substrate

**Goal:** compile the 8-bit C4 ISA to a transformer (`torch.state_dict`) in **< 10K total
code-LOC** (substrate + all ops). Clean-room; does **not** import from `neural_vm/`. The
existing codebase's substrate *alone* is 14,504 LOC because it is general (wide/multi-byte
ALU, multipass IR, campaign tooling). By constraining to 8-bit-only and choosing a
scalar-per-register residual layout, we target a small fraction of that.

This document is the **design contract**. Sections (a)–(f) below are the deliverables. A
working IMM/ADD/HALT vertical slice validating the contract lives in `slice_demo.py` +
`test_slice.py`.

---

## (a) Minimal ISA subset

C4 is a **stack machine with a single accumulator `AX`**. Binary ops consume the top of
stack and `AX`: `AX = pop() OP AX`. The C4 compiler emits a `PSH` before the second operand
of every binary op, so a source expression `a OP b` compiles to `IMM a; PSH; IMM b; OP`.

We target **22 core opcodes, 8-bit only** (values match the reference `neural_vm` `Opcode`
enum so semantics are shared, but only the 8-bit subset is implemented — no MUL/DIV/MOD,
no syscalls, no 32-bit multi-byte):

| Op   | Val | Semantics (8-bit, mask 0xFF)          | Class        |
|------|-----|----------------------------------------|--------------|
| IMM  | 1   | `AX = imm`                             | load         |
| LEA  | 0   | `AX = BP + imm`                        | load         |
| PSH  | 13  | `SP -= 1; stack[SP] = AX`              | stack        |
| ADD  | 25  | `AX = pop() + AX`                      | alu          |
| SUB  | 26  | `AX = pop() - AX`                      | alu          |
| AND  | 16  | `AX = pop() & AX`                      | bitwise      |
| OR   | 14  | `AX = pop() \| AX`                     | bitwise      |
| XOR  | 15  | `AX = pop() ^ AX`                      | bitwise      |
| SHL  | 23  | `AX = pop() << AX`                     | shift        |
| SHR  | 24  | `AX = pop() >> AX`                     | shift        |
| EQ   | 17  | `AX = (pop() == AX)`                   | cmp          |
| NE   | 18  | `AX = (pop() != AX)`                   | cmp          |
| LT   | 19  | `AX = (pop() <  AX)`                   | cmp          |
| GT   | 20  | `AX = (pop() >  AX)`                   | cmp          |
| LE   | 21  | `AX = (pop() <= AX)`                   | cmp          |
| GE   | 22  | `AX = (pop() >= AX)`                   | cmp          |
| LI   | 9   | `AX = mem[AX]`   (1 byte)              | memory       |
| SI   | 11  | `mem[pop()] = AX`(1 byte)              | memory       |
| JMP  | 2   | `PC = imm`                             | control      |
| BZ   | 4   | `if AX==0: PC = imm  else PC += width` | control      |
| BNZ  | 5   | `if AX!=0: PC = imm  else PC += width` | control      |
| JSR  | 3   | `SP -= 1; stack[SP] = PC+width; PC=imm`| control      |
| ENT  | 6   | `SP-=1; stack[SP]=BP; BP=SP; SP-=imm`  | control      |
| LEV  | 8   | `SP=BP; BP=pop(); PC=pop()`            | control      |
| HALT | 38  | stop (alias EXIT)                      | control      |

*(LC/SC collapse into LI/SI at 8-bit — one byte each — so they add 0 LOC. `HALT` is the
canonical name for `EXIT`.)*

**Instruction encoding.** Each instruction is a fixed slot of `WIDTH = 2` cells:
`[opcode, imm]`. `PC` counts in cells; `PC += WIDTH` per non-branch step. (The reference
uses an 8-byte slot with a 4-byte immediate; we compress to 2 cells because immediates are
8-bit.) The program lives in a read-only **code table** baked into the embedding.

---

## (b) Residual-band layout (named dims)

Values are **scalar** (not nibble one-hot): one residual dim holds an 8-bit value directly
as a float in `[0,255]`. This is the single biggest LOC lever vs. the reference — the C4
model spends most of its 14.5K substrate LOC on per-nibble binary one-hots and carry
cascades for 32-bit width. At 8-bit we do exact integer arithmetic in scalar dims using
SwiGLU/attention gadgets and re-quantize with a modular fold when needed.

Residual width `D` is the number of named bands (padded to a multiple of `N_HEADS`):

```
Registers (scalar value dims):
  AX        accumulator
  SP        stack pointer (cell index into stack region)
  BP        base pointer
  PC        program counter (cell index into code)

Stack top mirror:
  STACK0    == stack[SP]   (materialised each step for pop())

ALU scratch:
  ALU_A     first operand  (= popped value)
  ALU_B     second operand (= AX)
  ALU_LO    ALU result

Instruction (fetched from code table by PC):
  OP_ONEHOT[0..NUM_OPS)   one-hot of current opcode
  IMM                     current immediate (scalar 8-bit)

Emission / decode:
  OUTPUT    the value the step exposes to the LM head (= AX after the step)
  HALTED    1.0 once HALT executed (sticky)

Control scratch:
  ONE       constant 1.0 bias lane (baked into embedding, never written)
```

Only `OP_ONEHOT` is a one-hot (size `NUM_OPS`, needed for opcode dispatch). Everything else
is a single scalar dim. This keeps `D` at roughly `NUM_OPS + ~14 ≈ 55` rather than the
reference's 872.

---

## (c) Token / emission + decode contract

**Reference fragility avoided.** The reference emits a **fixed 35-token stride per step**
(PC/AX/SP/BP/STACK0 markers + MEM + terminator) and the verdict decodes by fixed offset —
memory notes repeatedly cite "fixed-35-slice framing drift" as a top failure root. We drop
this entirely.

**c4_min contract: one token per step, value = AX.** The model runs autoregressively over a
sequence whose position `t` corresponds to VM step `t`. After processing the prefix of `t+1`
tokens, the LM head at position `t` emits **one token** = the low byte of `OUTPUT` (= the
`AX` value produced by step `t`). Token vocabulary is `0..255` (byte values) plus a single
`HALT` token id `256`.

Decode is trivial and stride-free: `result[t] = argmax(logits[t])`. The program's answer is
the `AX` value emitted on the step *before* `HALT` (or at any queried step). Because there
is no multi-token framing, there is no stride to drift.

**How state advances without re-reading tokens.** The VM state (AX/SP/BP/PC/stack/mem) lives
in the residual stream and is carried forward by attention: at step `t` the residual copies
the previous step's registers from position `t-1` (a "carry-forward" attention head reading
`STACK`/register bands at `t-1`), then applies the opcode fetched at the new `PC`. Thus the
emitted token stream is purely an *observation* of AX; it does not need to be fed back to
reconstruct state (the model is run on a fixed-length "step ruler" input; see (e)).

---

## (d) Compact DSL

Two primitives express the entire ISA.

**`FFNRule`** — a conditioned write, compiled into SwiGLU hidden units:
```
FFNRule(
  when   = [(band, lo, hi), ...]   # AND of scalar-band windows (e.g. OP_ONEHOT[ADD] ~ 1)
  write  = {band: LinearExpr}      # dst_band += Σ coeff*src_band + const, gated by `when`
)
```
Compiles to `k` SwiGLU hidden units per rule: `gate` reads the guard (product of windows
via a sharp silu AND), `up` reads the linear write-expression, `W_down` routes the product
to the destination band. The exact-integer SwiGLU identity is `silu(S)·(v/1)` with `S`
large (`silu(100)=100` to fp32), so `up = S`, `gate = v`, `down = 1/S` yields `+v`.

**`AttentionSpec`** — a Q/K/V band route (for pop / carry-forward / memory load):
```
AttentionSpec(
  q = band,           # query key baked so it matches...
  k = band,           # ...this key band at the source position
  v = band,           # value band copied
  dst = band,         # written to this band at the query position
  alibi_slope = s,    # positional focus (0 = content-only)
)
```
Compiles to one head: `W_q`/`W_k` project the match bands into a head subspace with high
gain (sharp softmax → hard selection), `W_v`/`W_o` copy `v`→`dst`. The additive `mask`
buffer carries the ALiBi slope. `pop()` is `q=SP, k=cell_index, v=stack_value`.

That's it — no wide_alu_dsl, no multipass IR, no computation-graph OpType zoo.

---

## (e) Compiler stages (DSL → torch state_dict)

Architecture matches the reference exactly (verified against `neural_vm/base_layers.py`):
**no RMSNorm, bare additive residual, ALiBi via additive attention mask, softmax attention,
SwiGLU FFN.**

```
compile(program) →
  1. assemble(program)         # source ops → code table [(op, imm)], resolve labels
  2. layout()                  # assign band offsets → D, NUM_OPS
  3. embed()                   # build W_embed: token/pos → residual.
                               #   - "step ruler" input: position t embeds the constant
                               #     ONE lane + a positional slot; the code table is baked
                               #     into the FFN that fetches by PC.
  4. for each layer spec:      # a small fixed pipeline, ~6 logical layers:
       - carry_forward attn    #   copy registers from step t-1
       - fetch FFN             #   PC → OP_ONEHOT, IMM  (code table baked here)
       - pop attn              #   SP → STACK0 (top of stack)
       - alu/cmp/bitwise FFN   #   per-op-class rules write ALU_LO
       - writeback FFN         #   AX/SP/BP/PC ← results per opcode
       - emit FFN              #   OUTPUT ← AX ; HALTED latch
     each spec → PureAttention or PureFFN weights
  5. lm_head()                 # W_unembed: OUTPUT scalar → 257 logits (byte + HALT)
  6. pack state_dict           # {embed, blocks.i.attn.*, blocks.i.ffn.*, lm_head}
```

The runtime model (`Transformer`) is a ~40-line `nn.Module`: embed → N×(attn, ffn) → head,
with the exact forward from the reference:
`ffn: x + W_down(silu(W_up x + b_up) * (W_gate x + b_gate) + b_down)`;
`attn: x + W_o( softmax(QKᵀ·scale + mask) V )`.

---

## (f) LOC budget

Projected LOC per component (measured for skeleton, projected for full ISA):

| Component                       | LOC (proj) | Notes                                     |
|---------------------------------|-----------:|-------------------------------------------|
| `isa.py` (opcodes, encoding)    |        120 | enum + assemble + reference interpreter   |
| `layout.py` (band registry)     |        110 | named-dim allocator                        |
| `model.py` (Transformer nn)     |        130 | embed + N×(attn,ffn) + head, exact fwd     |
| `dsl.py` (FFNRule/AttentionSpec)|        180 | the two primitives + LinearExpr            |
| `compile_ffn.py` (rule→SwiGLU)  |        320 | AND-gate + write bake                       |
| `compile_attn.py` (spec→head)   |        220 | Q/K/V route bake + ALiBi                    |
| `compiler.py` (pipeline stages) |        260 | assemble→layout→embed→layers→head→pack     |
| **substrate subtotal**          | **~1,340** |                                            |
| per-op-class rule sets:         |            | (ops share compilers; only rules differ)   |
|   load (IMM/LEA)                |         40 |                                            |
|   alu (ADD/SUB)                 |         90 | scalar add + mod-256 fold                  |
|   bitwise (AND/OR/XOR)          |        140 | bit-decompose gadget (8 bits)              |
|   shift (SHL/SHR)               |         90 |                                            |
|   cmp (EQ/NE/LT/LE/GT/GE)       |        130 | step/threshold gadgets                     |
|   stack (PSH/pop)               |         60 |                                            |
|   memory (LI/SI)                |        120 | addr→cell attn                             |
|   control (JMP/BZ/BNZ/JSR/ENT/LEV)|      220 | PC writeback + push/pop PC                  |
| **ops subtotal**                |   **~890** |                                            |
| tests / demo                    |        400 | slice + per-op-class oracle                 |
| **TOTAL (proj)**                | **~2,630** | **well under 10K**                         |

**Verdict on <10K:** the scalar-per-register layout removes the reference's dominant cost
(per-nibble one-hots + 32-bit carry cascades). Even tripling every projection for
unforeseen gadget complexity (bitwise/cmp exactness at 8-bit) lands at ~7.9K — still under
budget. **Preliminary GO**, contingent on the vertical slice proving the substrate
compiles + runs + decodes correctly (see `test_slice.py`). See REPORT section at bottom for
the measured skeleton LOC and final GO/NO-GO.
