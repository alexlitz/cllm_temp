# The Pure-Forward C4 VM — the whole VM runs through the vanilla `model.forward`

Date: 2026-07-14 · Branch: `nibble-pure-forward` (base `nibble-unified-model` `04193dcc`)

## The mission

Assemble the **definitive pure vanilla-forward VM**: a standard decoder-only
transformer over a **token stream**, where the ONLY Python is the standard
`argmax`-generate-and-append loop. **No** Python VM state, **no** `DictMemStack`,
**no** functional gadgets, **no** Python integer ALU. State lives in the token
stream + KV cache; the ALU / dispatch / control live in FFN weights; memory is
softmax1-KV attention over the emitted MEM tokens; PC/registers live in the
emitted 30-token frames; HALT is the loop's stop condition.

This is the blogspec's **true form**. It replaces the *hybrid* driver
(`blogspec_run._apply_op`, which computes the transition in Python — an if/elif
dispatch + a Python `DictMemStack` + per-call SwiGLU gadgets, with the transformer
only emitting/ingesting) with a driver where **the transition happens IN the
forward**.

## What got built

`c4_min/nibble_pure_forward.py` — `build_pure_forward_model` + `run_pure_forward`
+ the `assert_no_python_compute` trace guard. The step loop is:

```python
stream = [BOS] + init_frame
for _ in range(max_steps):
    x = model.embed[stream]          # token embedding
    overlay(x)                       # program-in-DATA + frame-slot ROLE tags (structural)
    for blk in model.blocks: x = blk(x)   # == model.forward minus the LM-head linear
    frame = emit_frame(x[-1])        # LM value-argmax snap (the spec re-quantiser, no round)
    stream += frame                  # APPEND — state round-trips through the token stream
```

Every VM step is **one `model.forward` over the whole growing stream**. There is
no persistent Python residual, no Python dict, no gadget — the register state for
step *n* is **read back out of the token stream** by attention, the op is computed
by FFN weights, and the next frame is emitted and appended.

### The pure-forward step mechanism

The stream is `BOS` then one 30-token register frame per step
(`blogspec_vocab.build_step_frame`: `PC AX SP BP` + a `MEM(addr,val)` slot). One
step = one forward over the whole stream so far:

| block | role | how |
|-------|------|-----|
| **block 0 ATTN** | **FRAME-INGEST CAM** | 20 heads (one per (register, byte)) — a softmax1 + ALiBi content-addressable read keyed on a rigid per-slot ROLE tag, a query-exclusion penalty, and recency ALiBi so the **latest** frame's bytes are gathered into this position's register nibble bands. **State comes from the SEQUENCE, not a Python variable.** |
| block 0 FFN … k | the baked VM STEP | recompose nibbles→scalar lanes, fetch@PC over the program (code-as-data), opcode decode, MoE-style opcode-gated dispatch, branch delta, mod-256 fold — the **same persistent weights** as `nibble_vm.build_step_model`. The op RESULT is computed by these FFN weights inside `model.forward`. |
| one ATTN | **§Memory KV head** | softmax1-KV CAM over the emitted MEM tokens: a LOAD query content-addresses the store rows in the stream and writes the loaded value nibbles into AX. **Memory IS the token stream + attention**, not a dict. |

Contrast with the recurrent driver (`nibble_vm.run_program`): that carries VM
state in a **persistent residual vector** and snaps lanes in Python between steps.
Here the state instead round-trips through the **token stream** — exactly a real
decoder-only transformer. The green-field's one-token-per-nibble emit makes the
old 48k model's framing-desync **impossible by construction** (each register byte
is a single argmax'd token, re-embedded to its two nibbles), which is why this
runs clean where the old model stalled at ~40%.

## The zero-python-compute PROOF (the acceptance criterion)

`assert_no_python_compute(fn, ...)` installs a `sys.settrace` guard that raises if
the run enters **any** forbidden compute-path function. The forbidden set exactly
covers every gadget the hybrid `_apply_op` uses:

```
_apply_op, DictMemStack.{__init__,store_int,load_int},
nibble_add_gadget, nibble_sub_gadget,          # ALU
mul32, div32, mod32,                           # muldiv
compare, to_bit,                               # cmp
or_gadget, xor_gadget, and_gadget, shl_gadget, shr_gadget   # bitwise
```

* **Positive control** (`test_trace_guard_is_live_catches_old_path`): running the
  old hybrid path (`DictMemStack()` + `_apply_op(...)`) under the guard **is
  caught** — the proof is not vacuous.
* **The proof program** `IMM 6; PSH; IMM 7; ADD; HALT`
  (`test_proof_program_is_pure_no_python_compute`): runs entirely through
  `model.forward` + argmax-append → `[6,6,7,13,13]` (byte-exact vs
  `isa.interpret`), guard **clean**. **ADD ran in the FFN weights.**
* Every extended family below is verified under the same guard.

## What runs 100%-in-forward — verified byte-exact vs `isa.interpret`

All under `assert_no_python_compute`, all through `run_pure_forward`:

| family | ops | status | evidence |
|--------|-----|:------:|----------|
| **immediate / stack(depth-1) / arith** | IMM PSH ADD SUB LEA | **100%-in-forward** | proof program `[6,6,7,13,13]`; SUB-borrow `10-20=246` |
| **control flow / PC / loops** | JMP BZ BNZ HALT | **100%-in-forward** | forward BZ-skip → `[0,0,7,7]`; **backward BNZ countdown loop 14 steps** → `[3,3,1,2,2,2,1,1,1,1,1,0,0,0]` byte-identical (ONE fixed step-block, growing stream, argmax loop) |
| **comparisons** | EQ NE LT GT LE GE | **100%-in-forward** | all 6 (`5==5→1`, `3<7→1`, `7>3→1`, …) — the §Comparisons zero-detector + sign-of-diff computed ungated each step, boolean written by the opcode-gated expert |
| **bitwise / shift** | OR XOR AND SHL SHR | **100%-in-forward** | `0x0C|0x03=0x0F`, `0xFF^0x0F=0xF0`, `0xF0&0x3C=0x30`, `3<<2=12`, `0xF0>>3=30` — folded per-nibble table FFN |
| **8-bit muldiv** | MUL DIV MOD | **100%-in-forward** | `6*7=42`, `20*20=144(mod256)`, `84/7=12`, `85%7=1`, `div/0→0`, `mod/0→0` — the byte×byte lookup table in the FFN |
| **memory (KV)** | LI SI (LC SC) | **100%-in-forward** | store→load(42), **ZFOD**(unwritten→0), **latest-write-wins**(99), **cross-address**(11/22) — the store DATA rides in the emitted MEM token, the load VALUE is retrieved by the model's own softmax1-KV attention over the stream. **The Python dict is gone.** |

Tests: `c4_min/test_pure_forward.py` (11 tests, all green). Standalone probe:
`c4_min/_probe_pf_families.py` (memory/cmp/bitwise/muldiv, `ALL PASS`).

**This closes the unified-model doc's boundary #1 for the LOAD/STORE half:** there,
the §Memory KV head and the dispatch ran in *different position regimes* (multi-
position frame log vs single-position residual) and "were not yet composed into
one recurrent driver." Here every VM step **is** a position in one growing stream,
so the KV head and the dispatch run in the **same** regime — and LI/SI are proven
byte-exact end-to-end in that composed driver.

## The honest boundary — what does NOT run fully-in-forward, and the margin

### 1. Multi-slot stack: PSH depth > 1

The frame carries a **single `STACK0` mirror slot**, so two pushes before a
consume collide. Measured (`c4_min/_probe_pf_boundary.py`):

```
IMM 10; PSH; IMM 20; PSH; IMM 5; ADD; ADD
   got = [10,10,20,20,5,25,45,45]      # 2nd PSH clobbered STACK0; outer ADD read 20 not 10
   ref = [10,10,20,20,5,25,35,35]      # 20+5=25, then 10+25=35
```

Depth-1 (the common case — one operand popped per binary op) is exact for every
family above; depth>1 is off by the clobbered slot (**45 vs 35**, margin = the
clobbered push, here 10).

**Plan to close it (the pieces already exist and are proven):** route PSH/consume
through the **same softmax1-KV memory head that already works**. `PSH` = a store
to address `SP` (emit a MEM token, `SP-=4`); a consuming op's pop = a load at `SP`
(`SP+=4`). This is the *identical* store/load contract already verified byte-exact
for SI/LI above, applied to the stack addresses instead of program addresses. The
head is in the model; the wiring is: on PSH lay a store MEM token keyed on `SP`
(the driver already lays store tokens for SI), and make the pop-consuming ops
(ADD/SUB/CMP/bitwise/muldiv) read their `STK` operand from a `QRY_BIN=SP` KV load
instead of the STACK0 mirror. No new mechanism — it reuses the verified KV path.
JSR/ENT/LEV/ADJ (call convention) reduce to this same store/load contract (push
return addr, save/restore BP, reserve locals), so they are blocked by the same
single-slot boundary, not by any missing arithmetic.

### 2. 32-bit MUL/DIV/MOD

The folded muldiv is an **8-bit lookup table** (256×256→byte) — byte-exact on the
8-bit substrate and spec-sanctioned ("a table in the FFN"). The **full 32-bit**
MUL/DIV/MOD are data-dependent iteration (carry-round partial products, base-16
long division) that do **not** collapse into a fixed FFN stack; they need O(width)
unrolled blocks or an iterative gadget. Recorded as `UNFOLDABLE` in
`nibble_unified.py`. This is an arithmetic-width limit, unrelated to the
pure-forward mechanism.

## Coverage summary

| family | runs 100%-in-forward? | byte-exact |
|--------|:---------------------:|:----------:|
| IMM LEA PSH(depth-1) ADD SUB | YES | YES |
| JMP BZ BNZ HALT + loops | YES | YES |
| EQ NE LT GT LE GE | YES | YES |
| OR XOR AND SHL SHR | YES | YES |
| MUL DIV MOD (8-bit) | YES | YES |
| LI SI LC SC (memory, KV over the stream) | YES | YES |
| PSH/pop depth>1, multi-slot stack | NO — single STACK0 slot | mismatch 45 vs 35; **plan: route through the verified KV head** |
| JSR ENT LEV ADJ | NO — reduce to multi-slot stack | same boundary |
| MUL DIV MOD (32-bit) | NO — iterative, `UNFOLDABLE` | n/a |

## Parameter accounting

Exact dense / nonzero (`code_size=20`; the nonzero count is the true size of the
wired logic — the dense figure is inflated by full-`dim×dim` zero attention and
uniform FFN hidden-dim padding, per the unified-model doc):

| config | dim | blocks | heads | dense | nonzero |
|--------|----:|-------:|------:|------:|--------:|
| base (fetch+dispatch+base ops, no families) | 840 | 7 | 20 | 21,285,039 | **1,702** |
| + KV memory | 1071 | 9 | 21 | 50,117,485 | **3,628** |
| + comparisons | 840 | 8 | 20 | 24,262,121 | **1,824** |

(The 8-bit muldiv table adds ~181k hidden units → ~1.1M nonzero, same as the
unified model; kept out of the memory/base builds so those stay fast.)

The pure-forward layout adds **20 ingest heads** (one per (register, byte)) + **1
memory head** = 21 attention heads on block 0; the block-0 ATTN carries the ingest
CAM, the mem-cam block's ATTN carries the §Memory KV head, all other block ATTN is
zeroed (identity). The FFN weights are identical to `nibble_vm.build_step_model`
plus the opcode-gated cmp/bitwise/muldiv/memory experts.

## Files

| file | role |
|------|------|
| `c4_min/nibble_pure_forward.py` | `build_pure_forward_model` (the ONE pure-forward Transformer), `run_pure_forward` (the argmax-generate driver over the token stream), the frame-ingest CAM, the §Memory KV head bake, the cmp/bitwise/muldiv opcode-gated experts, and `assert_no_python_compute` (the trace guard) |
| `c4_min/test_pure_forward.py` | 11 tests: frame-ingest, the proof program + purity guard + positive control, token-stream-is-state, head count, memory (KV) store/load/zfod/latest/two-addr, cmp/bitwise/muldiv families |
| `c4_min/_probe_pf_families.py` | lean per-family byte-exact probe (memory/cmp/bitwise/muldiv), all under the guard |
| `c4_min/_probe_pf_boundary.py` | the multi-slot-stack boundary probe (depth-1 OK, depth-2 mismatch 45 vs 35) |

## Bottom line

The proof program `IMM 6; PSH; IMM 7; ADD; EXIT` — and comparisons, bitwise,
8-bit muldiv, memory (KV), and unbounded loops — run **entirely through
`model.forward` + argmax-generate-and-append**, with a live `sys.settrace` guard
proving **zero** Python compute (no `_apply_op`, no `DictMemStack`, no gadget, no
Python ALU). The single remaining structural limit is the multi-slot stack
(PSH depth>1), whose fix is to route push/pop through the **same softmax1-KV
memory head that is already proven byte-exact** — no new mechanism, just wiring.
