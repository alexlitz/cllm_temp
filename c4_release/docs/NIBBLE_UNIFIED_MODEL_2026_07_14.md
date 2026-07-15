# The Unified C4 VM — ONE persistent Transformer that IS the whole VM step

Date: 2026-07-14 · Branch: `nibble-unified-model` (base `nibble-integrated` `d7ce2ef6`)

## The mission, and what actually got built

**Mission:** assemble ONE persistent `blogspec_model.Transformer` (a coherent
`nn.Module`) that IS the whole C4 VM step, with everything IN the model weights —
no functional gadget building weights on-the-fly, no Python `if/elif` dispatch, no
Python memory dict on the compute path; only the standard argmax generation loop
stays Python. Then count the ONE model's params and verify per-family end-to-end.

**What got built** (`c4_min/nibble_unified.py`, `build_unified_model`): a single
`Transformer` whose block stack performs one complete VM step. Everything the
step needs is baked as persistent weights:

| piece | mission item | where in the ONE model |
|-------|--------------|------------------------|
| universal fetch (code-as-data) | (4) | `recompose → pc-fetch → code-select` FFN blocks (bilinear select over DATA bands) |
| in-model MoE dispatch | (2) | `dispatch` block = spec `StandardMoEFFN`, opcode one-hot routing, **0.0 diff vs dense** |
| op transitions as REAL FFN weights | (1) | per-opcode `PureFFN` experts (base/cmp) + bitwise-table FFN blocks + 8-bit muldiv-table FFN blocks — all materialised, nothing on-the-fly |
| softmax1-KV memory | (3) | `bake_memory_head` on **block-0 attention** (real §Memory CAM, replaces the dict) |

The old scored VM (`blogspec_run._apply_op`) was a **hybrid**: Python `if/elif`
dispatch, per-call functional SwiGLU gadgets (`nibble_add_gadget`, `_cmp.compare`,
`_bit.*`, `_md.mul32`) that store nothing, and a `DictMemStack`. The unified model
removes all three from the compute path: dispatch is the model's own MoE, the op
math is persistent FFN weights, and memory is the model's own attention head.

## The block stack (ONE Transformer, `n_blocks` FFN+attn blocks)

Each block is `softmax1+ALiBi attention` then `SwiGLU FFN` (`blogspec_model.Block`).
Attention is zeroed (identity) in every block **except block 0**, which carries
the KV-memory CAM head. The full stack (`include_mdm_table` + `include_bitwise`):

```
 block 0  recompose      nibble bands → scalar value lanes       [FFN]  +  KV-MEMORY head [ATTN]
       1  pc-fetch       PC_VAL → PC one-hot + AX_ZERO           [FFN]
       2  code-select    fetch OP_VAL/IMM @PC from DATA (universal fetch)   [FFN]
       3  opcode-decode  OP_VAL scalar → OP_IS[op] one-hot (all 23 ops)    [FFN]
       4  mdm-expand     STK/AX bytes → 256-cell one-hots        [FFN]   (8-bit muldiv)
       5  mdm-select     table op(a,b)→MDM_RES for MUL/DIV/MOD   [FFN]   (8-bit muldiv)
       6  bw-expand      AX/STACK0 nibbles → 16-cell one-hots    [FFN]   (bitwise)
       7  bw-bit4        shift bit-4 fold (SHL/SHR)              [FFN]   (bitwise)
       8  bw-select      per-nibble 256-entry OR/XOR/AND + shift [FFN]   (bitwise)
       9  bw-recompose   AX nibbles → AX_VAL (bitwise result)    [FFN]   (bitwise)
      10  dispatch       StandardMoEFFN — per-opcode experts     [MoE FFN]
      11  branch-delta   bilinear BZ/BNZ PC update               [FFN]
      12  fold           AX_VAL mod-256                          [FFN]
```

The MoE `dispatch` block holds **18 experts** keyed by opcode one-hot:
`IMM LEA PSH JMP BZ BNZ HALT` (base), `EQ NE LT GT LE GE` (cmp),
`MUL DIV MOD` (8-bit muldiv housekeep), plus the 5 bitwise ops route their
PC/SP housekeeping through experts while `bw-select`/`bw-recompose` compute the
result. The step loop is the standard generation loop: apply the blocks to the
single-position residual, snap each register lane by the LM value-argmax (no
`torch.round`), re-embed the byte tokens → nibble bands, repeat.

## Parameter accounting — the definitive "size of the fully wired VM"

Exact per-tensor dense/nonzero counts (`param_report`). Three build configs:

| config | dim | blocks | heads | vocab | TOTAL (dense) | NONZERO | sparsity |
|--------|----:|-------:|------:|------:|--------------:|--------:|---------:|
| base (no bitwise, no muldiv) | 800 | 7 | 4 | 265 | **19,341,079** | **1,721** | 99.9911% |
| + bitwise | 1348 | 11 | 4 | 265 | **952,928,215** | **145,366** | 99.9847% |
| + 8-bit muldiv + bitwise (FULL, code_size 16) | 1348 | 13 | 4 | 265 | **7,886,694,183** | **1,111,235** | 99.9859% |
| + 8-bit muldiv + bitwise (FULL, code_size 32) | 1396 | 13 | 4 | 265 | **8,170,873,287** | **1,111,475** | 99.9864% |

Where the dense params live (FULL, code_size 16), aggregated:

| component | params | nonzero |
|-----------|-------:|--------:|
| attention (all blocks; only block-0 CAM is live) | 94,489,408 | 165 |
| FFN (all blocks incl. MoE experts + tables) | 7,791,490,070 | 1,110,325 |
| embed | 357,220 | 745 |
| lm_head + lm_bias | 357,485 | 0 |

### Reading the numbers honestly

* **The dense total is dominated by two kinds of near-empty tensors, not by
  content.** (a) Every block has full `dim×dim` attention matrices (`W_q/k/v/o`),
  ~all zero (identity) except block-0's CAM head (165 nonzeros). (b) `_load_ffn_padded`
  pads **every** FFN block's hidden dim to the widest block's hidden. The 8-bit
  MUL/DIV/MOD table (`mdm-select`) has ~181k hidden units, so the padding blows
  every block's `W_up/W_gate (hidden×dim)` and `W_down (dim×hidden)` up to that
  width — quadratic waste. That single design choice takes the total from ~19M
  (base) to ~7.9B (full).
* **The real content is the NONZERO count: ~1.1M weights for the full VM**
  (~146k without the 8-bit table, ~1.7k for the base ops+cmp+fetch+dispatch). The
  base interpreter — universal fetch + MoE dispatch of IMM/LEA/PSH/ADD/SUB/JMP/
  BZ/BNZ/HALT + all 6 comparisons + the block-0 memory CAM — is **1,721 nonzero
  weights**. That is the honest "size" of the wired logic.
* **Sparsity ≈ 99.99%** in every config: the VM is a hand-baked exact circuit, so
  almost all of the dense tensor is structural zero.
* `lm_head` is all-zero: the driver requantises via the value-argmax snap
  (`_snap_lane`), the spec's vanilla emit-token round-trip, so the model's own
  `F.linear(x, lm_head)` head is unused on this path (the step loop reads the
  scalar value lanes directly). This is the one place where a literal
  `model.forward(tokens) → logits` call is not the step interface; the step loop
  is `for blk in model.blocks: x = blk(x)` (i.e. `model.forward` minus the final
  LM-head linear) plus the value-argmax snap.

## Verification — per-family, end-to-end, byte-exact vs the reference

`c4_min/verify_unified.py` drives the FULL model (code_size 32, muldiv+bitwise)
through the step loop and checks byte-exactness vs `isa.interpret` (the 8-bit
reference). **All families PASS** (30 step-loop programs + 5 KV cases):

```
dim=1396 n_blocks=13 n_heads=4 vocab=265   (build 12.9s)
TOTAL(dense)=8,170,873,287  NONZERO=1,111,475  SPARSITY=99.9864%

[step-loop dispatch : IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ + loops + CMP]   all OK
   IMM LEA ADD SUB ADD_wrap(300&255=44) SUB_borrow(10-20=246) JMP
   BZ_taken BZ_skip BNZ_taken loop_count(BNZ backward, →0)
   EQ NE LT GT LE GE                                             (18 progs)
[step-loop dispatch : OR/XOR/AND/SHL/SHR  (folded bitwise FFN)]    all OK  (5)
[step-loop dispatch : MUL/DIV/MOD  (folded 8-bit table FFN)]       all OK  (6)
   incl. DIV/0→0, MOD/0→0
[folded KV-memory head (block-0 attn of SAME model) : LI/SI]       all OK  (5)
   store/load, cross-address, ZFOD(unwritten→0), latest-write-wins, free(→0)
ALL: PASS
```

Extra checks:
* **MoE dispatch is byte-EXACT vs the dense FFN**: max abs diff `0.0` over all
  routed opcodes (the per-op experts partition the dense hidden units and the
  one-hot blend is an algebraic identity).
* **Deep control flow falls out free**: the `loop_count` BNZ backward branch runs
  14 steps, byte-identical trace `[3,3,1,2,2,2,1,1,1,1,1,0,0,0]` vs the reference —
  ONE fixed step-block, unbounded loop, standard generation loop.
* Existing suite `test_nibble_vm.py` (9 tests) still green.

**Wall cost:** build 12.9s; the 30 step-loop programs take ~171s total (~5.7s/prog)
because each block application multiplies the 8B-dense (mostly-zero) padded tensors
on CPU. The base config (no 8-bit table) runs each program in <1s. The dense-tensor
padding is a materialisation artifact, not intrinsic cost — a sparse/ragged-hidden
build would run at base speed.

## The honest boundary — what is ONE model vs what still needs a separate path

Two things do **not** fold into the single-position step-loop dispatch, and I am
reporting them precisely rather than faking a monolith:

1. **The multi-slot stack, and with it LI/SI/LC/SC and PSH/POP depth>1.**
   The step residual carries a **single `STACK0` slot**, so a program that pushes
   two values before consuming them is *not* exact — e.g.
   `IMM 10 · PSH · IMM 20 · PSH · IMM 5 · ADD · ADD` gives the model **45** vs the
   reference **35** (the second PSH overwrote `STACK0`). A true multi-slot stack
   (and the load/store opcodes) is exactly what the §Memory KV head is for: push =
   store, pop = load, LI/SI = load/store. That head **is** in the one model's
   weights (block-0 attention) and is **proven byte-exact standalone** (the 5 KV
   cases above: ZFOD, latest-write-wins, free). **But** it operates over the
   **multi-position** append-only frame log (`model.blocks[0].attn` applied to the
   whole token stream), not over the single-position recurrent step. So LI/SI are
   *in the model* but on a **different position-regime** than the step-loop
   dispatch — the two are not yet composed into one recurrent driver. Wiring them
   together (every VM step becomes a position in one growing stream, the KV head
   servicing PSH/POP/LI/SI as store/load) is the remaining integration; the
   pieces exist and each is verified, they are not fused.

2. **32-bit MUL/DIV/MOD.** The folded muldiv is an **8-bit lookup table**
   (256×256→byte, the same "table in the FFN" the spec sanctions for bitwise) and
   is byte-exact on the 8-bit substrate. The **full 32-bit** MUL/DIV/MOD in
   `_apply_op` (`_md.mul32/div32/mod32`) are data-dependent Python iteration
   (carry-round partial products, base-16 long division) that do **not** collapse
   into a fixed FFN stack — they need O(width) unrolled blocks or the iterative
   gadget. `UNFOLDABLE = {"MUL/DIV/MOD@32bit"}` in `nibble_unified.py` records this.

3. **JSR/ENT/LEV/ADJ/NOP** are not built as experts at all. They are call-convention
   ops whose effect is entirely stack/frame memory manipulation (push return addr,
   save/restore BP, reserve locals) — i.e. they reduce to the same store/load
   contract as (1) and are blocked by the same single-slot-stack boundary, not by
   any missing arithmetic. `NOP` is trivial (PC+=1) and could be added as a
   one-line expert; it simply was not in the corpus slice tested.

### Summary of coverage

| family | folded into ONE model? | verified byte-exact |
|--------|------------------------|---------------------|
| IMM LEA PSH(depth-1) ADD SUB JMP BZ BNZ HALT | YES (MoE experts, step loop) | YES |
| EQ NE LT GT LE GE | YES (SwiGLU experts, step loop) | YES |
| OR XOR AND SHL SHR | YES (bitwise-table FFN, step loop) | YES |
| MUL DIV MOD (8-bit) | YES (8-bit table FFN, step loop) | YES |
| MUL DIV MOD (32-bit) | NO — iterative, `UNFOLDABLE` | n/a |
| LI SI LC SC (memory) | head IS in the model (block-0 attn), proven standalone | YES (standalone); NOT wired to step loop |
| PSH/POP depth>1, multi-slot stack | NO — single `STACK0` slot in step residual | mismatch shown (45 vs 35) |
| JSR ENT LEV ADJ | NO — reduce to stack/frame memory (same boundary) | n/a |
| NOP | trivial, not built | n/a |

**Honest param total of the in-model part:** the fully-wired step interpreter
(fetch + MoE dispatch + base/cmp/bitwise/8-bit-muldiv experts + block-0 memory
CAM) is **~1.1M nonzero weights** (7.9B dense, 99.99% sparse); dropping the 8-bit
table leaves **~146k nonzero**; the base ops+cmp+memory core is **1,721 nonzero**.
The dense figure is inflated by full-`dim×dim` zero attention and uniform FFN
hidden-dim padding — the nonzero count is the true size of the wired logic.

## Files

| file | role |
|------|------|
| `c4_min/nibble_unified.py` | `build_unified_model` (the ONE Transformer) + `param_report` + the folded cmp / 8-bit muldiv / bitwise / memory bakes + `UNFOLDABLE` |
| `c4_min/nibble_moe.py` | `NibbleStandardMoEFFN` — the spec's in-model MoE dispatch (byte-exact vs dense) |
| `c4_min/verify_unified.py` | per-family end-to-end verification harness (step loop + KV head) |
| `c4_min/_probe_unified.py` | quick CMP / bitwise / 8-bit muldiv step-loop probe |
| `c4_min/_probe_unified_mem.py` | KV-memory head standalone byte-exact probe |
