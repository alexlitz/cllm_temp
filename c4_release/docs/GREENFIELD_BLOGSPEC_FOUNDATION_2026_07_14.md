# Green-field BLOG_SPEC foundation — re-founding `c4_min` on the spec's representation

Date: 2026-07-14 · Branch: `greenfield-blogspec-foundation` · Commit: `7e9b9844`

## What this is

The previous green-field `c4_min/` proved the op-gadget *mechanisms* but
**deviated from `docs/BLOG_SPEC.md`** in three ways that the mission called out:
it kept each register as a **single scalar float** (not nibbles), it used
**`torch.round`** for per-step re-quantization (`recurrent.py::_requantize`), and
its runtime used **plain `F.softmax`** (not `softmax1`).

This foundation re-houses the proven math on the spec's representation. It is a
new, coherent skeleton alongside the old modules (which other tasks still
depend on), added as `c4_min/blogspec_*`:

| file | role | spec section |
|------|------|--------------|
| `blogspec_layout.py`   | 16×4-bit **nibble** bands per register | §Internal Representation (569–571) |
| `blogspec_vocab.py`    | byte+marker vocab, **30-token** register frame | §Registers (442–461), §Tokenization (405–407) |
| `blogspec_model.py`    | vanilla transformer: **softmax1 + ALiBi** + SwiGLU | §Vanillaness (233–350), §Attention (488–502) |
| `blogspec_compiler.py` | nibble ALU (SwiGLU add/sub), register-ingest attention, emit head | §Building Blocks (504–545), §Basic Arithmetic (591–615) |
| `blogspec_run.py`      | standard autoregressive loop; LM-head nibble decode | §generation loop, §Tokenization (re-quant) |
| `test_blogspec_foundation.py` | 10 tests proving the four pillars | — |

## 1. Nibble representation (§Internal Representation)

> "We represent each [32-bit value] as 16 4-bit nibbles ... Registers are also
> all loaded in the same manner in different dims."

Every register is a band of **16 dims**, dim `REG+j` holding the integer value
(0–15) of little-endian nibble `j`. The residual width is `D = 104`:

```
PC[0:16]  AX[16:32]  SP[32:48]  BP[48:64]  STACK0[64:80]
CUR_NIB[80:96]   CTX[96:102]   BYTE_OFS[102]   ONE[103]
```

This is *not* scalar-per-register. `test_registers_are_16_nibbles` asserts each
register band is 16 dims wide; `test_32bit_value_spans_high_nibbles` decodes
`SP = 0x10000` (the spec's init) — which needs **nibble index 4** — proving the
representation carries true 32-bit values, not just bytes. An 8-bit value uses
nibbles 0–1; the identical layout scales to the full 32-bit ISA with no shape
change.

## 2. The 30-token register emission per VM step (§Registers)

Every VM step emits **exactly 30 tokens** (little-endian bytes):

```
1 REG_PC  + 4 PC bytes
1 REG_AX  + 4 AX bytes
1 REG_SP  + 4 SP bytes
1 REG_BP  + 4 BP bytes
1 MEM     + 8 (addr+value)
1 STEP_END
= 30
```

`build_step_frame(pc,ax,sp,bp,...)` produces it; `parse_step_frame` inverts it.
For the proof program the first emitted frame is

```
[REG_PC 1 0 0 0  REG_AX 6 0 0 0  REG_SP 0 0 1 0  REG_BP 0 0 1 0  MEM 0 0 0 0 0 0 0 0  STEP_END]
```

i.e. after `IMM 6`: PC=1, AX=6, SP=BP=0x10000 (the `1` in byte-2). The full run
stream is `152 = 5 steps × 30 + BOS + HALT` tokens.

### This emit → re-embed IS the re-quantization (replaces `torch.round`)

> §Tokenization: "we write the up to date values each step ... many of the
> passes of the transformer are just outputting the registers."

A freshly emitted byte token is an **exact integer**; it re-enters the residual
through the **integer-exact embedding table** at the next position, which
annihilates the O(1e-6) SwiGLU fp residue — the identical effect the old
`_requantize`/`torch.round` had, but achieved by the **vanilla autoregressive
loop**. `test_no_torch_round_in_exec_path` walks the AST of
`blogspec_run` / `blogspec_compiler` / `blogspec_model` and asserts there is
**no `round` call node** anywhere in the exec path. Register bytes are read out
of the nibble bands *only* by the LM byte-head's argmax — the model's own head
is the sole quantizer.

## 3. softmax1 + ALiBi (§Vanillaness, §The Attention Layer)

`blogspec_model.Attn.forward` is a copy of the spec's `PureAttention` reference
(lines 315–345): QKV projections, `scores·scale`, **ALiBi** additive bias
(`-slope·|i−j|`, geometric slopes `2^(-8/n·(i+1))` computed *inside* forward),
causal mask, then **`softmax1`**:

```python
def softmax1(x, dim=-1):        # exp(x) / (1 + sum exp(x))  — ZFOD
```

Implemented so the implicit **0-logit sink** ("the +1") is correctly scaled in
every regime (a query that matches nothing attends to nothing → ZFOD, giving
the VM its zero-fill-on-demand for unwritten registers/memory). `FFN` is SwiGLU
(`SiLU(up)·gate`, §467–478). Input is a **token-id stream** through a `vocab×dim`
embedding — the ordinary decode-only interface, no one-hot-position hack.

`test_softmax1_is_zfod` checks `[0,0,0]→0.75` (matches the naive spec form when
max≥0) and `[-50,-50,-50]→~0` (ZFOD). `test_attention_uses_alibi_slopes` checks
the geometric slopes.

### The model does real work (not trace-replay)

`ingest_ax_lowbyte` runs the **actual `model.forward`** (softmax1 + ALiBi
attention + SwiGLU FFN) over a 3-token frame and shows the attention gathers the
AX byte's nibbles out of the frame into the AX band, whence the LM head decodes
it. `test_model_forward_ingests_register_from_frame` verifies this for
{0,6,13,42,128,255}. This is the spec's core register mechanism — "write the
registers each step, retrieve by attending" — running on the vanilla transformer.

## 4. The proof program — `IMM 6; PSH; IMM 7; ADD; EXIT` → 13

`test_proof_program_add_13` runs it end to end:

```
step IMM  -> pc=1 ax=6  sp=65536 bp=65536
step PSH  -> pc=2 ax=6  sp=65532 bp=65536      (SP -= 4, 4-byte aligned)
step IMM  -> pc=3 ax=7  sp=65532 bp=65536
step ADD  -> pc=4 ax=13 sp=65536 bp=65536      (AX = pop 6 + AX 7, nibble adder)
step HALT -> pc=5 ax=13 sp=65536 bp=65536
AX trace: [6, 6, 7, 13, 13]   == isa.interpret(code)
```

The `ADD` runs through the **SwiGLU nibble add gadget** (`nibble_add_gadget`),
which computes `a+b` per nibble as `silu(S·(a+b))/S` with the carry/fold built
from the exact clamped-ReLU (`silu(RELU_S·z)/RELU_S`) — no python `+` on the
values, no rounding. `test_nibble_alu_gadgets_exact` confirms add/sub are exact
over random bytes; a 3000-sample sweep is byte-exact. `SUB` is the same adder on
the two's-complement `(a + (256−b)) mod 256`, matching the spec's "subtraction
naturally can work similarly" (§593).

## Mapping to the reference building blocks (`building_blocks_dsl.py` = §504–568)

The nibble ALU reuses the spec's constructor *patterns* — `step_function_rule`
(the `silu(S(x±ε))` step), the clamped-ReLU **range-check** fold
(`band_range_check_rules`), and the add/multiply primitives from §Basic
Arithmetic — but realised directly on this substrate's nibble bands (the
reference DSL is coupled to the `neural_vm` IR; c4_min keeps its own light
`compile_ffn` lowering). The one-hot **point indicator** (§510) and MoE
opcode-gating (§566) are the levers the opcode fan-out will use on top of this
skeleton.

## What is foundation vs. follow-on

Delivered (proven): nibble reps · 30-token emit / vanilla re-quant · softmax1 +
ALiBi + SwiGLU · the model's own attention doing register ingest · exact SwiGLU
nibble ALU · the proof program.

Designed to carry (next layers, out of scope here, per §566 MoE-per-opcode):
the full opcode fan-out as PC-indexed MoE experts, the memory KV
(store→key-by-address / load-by-attention, §Memory), deep control flow, and I/O.
The skeleton's shapes (16-nibble registers, 30-token frame, softmax1 sink) are
exactly what those layers need — nothing here has to change to add them.

## Reproduce

```
PYTHONPATH=<repo> python -m pytest c4_min/test_blogspec_foundation.py -q   # 10 passed
PYTHONPATH=<repo> python c4_min/test_blogspec_foundation.py                # human-readable
```
