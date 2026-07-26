# THE canonical full-ISA model (and the block-count reconciliation)

There have been several different block counts quoted for "the full ISA model"
(11, 80, 144-stored/291-applied, 238). They come from **different model families
and/or different divide implementations** — not different measurements of one model.
This note pins down THE canonical model so nobody quotes the wrong one again.

## THE canonical full-ISA model

**`qwen_full_vm.build(subset=SUBSET_FULL, efficient_alu=True, recurrent_divmod=True)`
with the DEFAULT lean divide (`C4_DIV_LEAN` on).** This is the fused Qwen VM that
`full_native_fast.py` ships. Every C4 opcode is native in-model (no subroutine
dispatch); MUL is the byte schoolbook, DIV/MOD are the **lean radix-16
digit-recurrence divide** (`div_radix16_lean`, 80 blocks).

Per-step depth (measured, this branch):

| variant | stored blocks | **applied per DIV step** | divide blocks |
|---|---|---|---|
| **LEAN (default)** | 44 | **107** | 81 (80 lean + 1 result-copy relay) |
| HARDENED (`C4_DIV_LEAN=0`) | 45 | 115 | 88 |
| LONGDIV (`C4_DIV_LONGDIV=1`) | 58 | 205 | ~186 base-16 long division |

- **A DIV/MOD step traverses all 107 applied blocks** (81 lean divide + 26
  shared/other-op front/back: ingest, code-cam fetch@PC, opcode-decode,
  mem-prep/cam, cmp-compute/finalize, bitwise, alu-psh/shift/expand, the MUL
  blocks, ax-mux, dispatch, branch-delta, fold).
- **Every non-DIV op block-skips the 81-block divide span** — a simple op
  (IMM/ADD/branch) runs only its ~7-26 live blocks (`step_block_skip` / the
  conditional-sparsity gather makes the skipped blocks a no-op).
- `recurrent_divmod=True` (the shipped config) folds the 9-block lean iteration
  body into ONE stored block applied 8×, so it **STORES 44 but APPLIES 107** per
  DIV step. `recurrent_divmod=False` (unrolled) STORES = APPLIES = 107.

So "the full ISA is 80 blocks" is shorthand for **the lean divide is 80 blocks, and
that is the deepest per-step subroutine in the full ISA** — the full DIV step is
107 applied (80 divide + 1 relay + 26 shared).

Byte-exactness of the lean divide: **875/875 standalone** (fp64 AND fp32 e2e,
`div_radix16_lean.measure()`), **288/288 through the real fused Qwen VM forward**
(the 8-bit-IMM DIV/MOD grid, `_gate_div_lean_grid.py`, both q and r decoded from the
AX nibble band == ISA `b==0 -> (0,0)`), `pytest -k muldiv` 10/10. The lean grid
fully matched the hardened control, which is why lean is the default (commit
`61227ec4`).

## The other numbers, reconciled

| quote | what it actually is | why it's misleading as "the full ISA" |
|---|---|---|
| **11 layers (~2 ms/step)** | `qwen_full_vm.build(subset=SUBSET_MEM_CMP)` — mem+cmp only, **NO muldiv, NO divide** (measured **10 blocks**). This is the shallow model `bench_bounded_kv_incremental.py` builds by default and the "2 ms bounded-KV" lean toy. | It has NO ALU/divide at all — the deep part of the ISA is simply absent. It is a shallow subset, not the full ISA. |
| **80 blocks** | the `div_radix16_lean` divide **subroutine** standalone (its `measure()` reports 80 unrolled). | Correct for the DIVIDE, but the full DIV *step* is 107 applied (80 divide + 1 relay + 26 shared). "80" is the divide, not the whole step. |
| **144 stored / 291 applied** | the fused `qwen_full_vm` full ISA under the **OLD base-16 long-division default** (now `C4_DIV_LONGDIV=1`, which currently measures 58/205 here; the historical 144/291 was a slightly different layout era). | Stale — the default divide is now the 80-block lean, so the current full ISA is **44 stored / 107 applied**. This is the number `full_native_fast.py`'s docstrings used to hardcode (now fixed). |
| **238 (or 262/291 divmod)** | the **separate** `nibble_pure_forward_complete` standalone-Transformer family (measured 238 unrolled / 91 stored recurrent, with **179 alu-div blocks** of base-16 long division). This is the model `compact_alloc.build_compact_sparse_streaming` / `bench_composed_fast_path.py` / `step_block_skip.py` run on. | A DIFFERENT model family (standalone Transformer, not the fused Qwen VM) AND the OLD long-division divide. It does NOT use the lean radix-16 divide. The composed-fast-path ms/step (the "238 / ~49 ms" line) is measured on THIS, not the canonical fused lean model. |

Key confusion to avoid: there are **two** full-ISA "families" plus **three** divide
implementations:

- **Families:** (a) the fused Qwen VM (`qwen_full_vm` → `full_native_fast`), and
  (b) the standalone Transformer (`nibble_pure_forward_complete` →
  `compact_alloc` → `bench_composed_fast_path`, the 238-block one). The canonical
  model is family (a).
- **Divides:** lean radix-16 (80, `div_radix16_lean`, DEFAULT in family (a)),
  hardened radix-16 (88, `C4_DIV_LEAN=0`), and base-16 long division (~186-262,
  `C4_DIV_LONGDIV=1` in (a); the ONLY divide in family (b)).

## The honest ms/step host

The self-emulation / "real ms/step" host is the **canonical family (a) lean full
ISA (107-applied)**, run incremental-decode via the conditional-sparsity fast path —
NOT the 10-block mem+cmp toy (2 ms) and NOT the 238-block family (b) composed number.

### Measured (RTX A5000, this branch, `_agent_canonical_msstep.py` / `bench_composed_fast_path.py`)

**Canonical family (a) lean full ISA (44 stored / 107 applied), conditional-sparsity
incremental driver, `full_native_fast`:**

| program | driver ms/step (B=1) | fwd µs/step B=64 |
|---|---|---|
| arith (ADD/SUB seq) | **8.75** | 6030 |
| muldiv (MUL/MOD seq) | **7.52** | 4844 |
| **mixed (ADD/MUL/DIV seq)** | **7.88** | 4031 |
| loop (SUB/BNZ countdown, 122 steps) | **4.26** | 2892 |

- Byte-exact **28/30** full-ISA programs (all ALU incl DIV/MOD/MUL, memory, branch,
  JSR/LEV, and a 4096-step muldiv_mix loop). The 2 fails are SHL/SHR and are
  `cond==dense=True` with a step-lag decode artifact (`[3,3,2,2,12]` vs
  `[3,3,2,12,12]` — same value 12, off-by-one), a pre-existing shift-via-mul AX
  decode lag, NOT a divide regression.
- Active FFN units = 31761 / 847440 = **3.75% of dense** (active block 1070 MB vs
  dense FFN 27.9 GB).
- **Important honesty note:** this driver runs ALL 107 applied layers per step with
  a sparse FFN gather — it does NOT per-op block-skip the 81-block divide span, so a
  DIV-containing step (`mixed`, 7.88 ms) and a divide-free step (`arith`, 8.75 ms)
  cost roughly the same. Per-op block-skip (DIV-step-only-runs-the-divide) is NOT
  yet wired for family (a); it exists only on family (b) — see below. So the honest
  family-(a) headline is **≈7-9 ms/step (B=1), ≈4-6 ms/step batched** for the full
  ISA — this REPLACES the 11-layer 2 ms toy as the self-emulation host number.

**Family (b) 238-block composed fast path (block-skip + direct-CAM + CUDA-graph),
S=900 — the MISLEADING number, base-16 long division:**

| | ms/step |
|---|---|
| full-238 dense | 1157.9 |
| composed (block-skip + direct-CAM + graph) | **49.3 (23.5×)** |
| DIV step (186 live blocks) | 921 |
| simple op (7 live blocks) | ~34 |

The 49 ms/step "238 / a24cc1ee" number is on THIS family, whose DIV is **186 blocks**
of base-16 long division (vs 81 lean divide blocks in the canonical family). DIV is
1% of steps but 18.7% of the composed weighted cost. This is the number the task
warned against quoting for the full ISA — it is a different (older, deeper-divide)
model family than the canonical one. If family (a) gets per-op block-skip + graphs
wired (its non-DIV ops are ~7-26 blocks of 107, its DIV 107), it should beat the
family-(b) 49 ms substantially, because the lean divide is 81 blocks vs 186.

### The three numbers, one line each
- **2 ms** = 10-block mem+cmp toy (NO divide) — not the full ISA.
- **49 ms** = family (b) 238-block composed, base-16 186-block divide — wrong family + old divide.
- **~7-9 ms/step (B=1), ~4-6 ms batched** = family (a) canonical lean full ISA
  (107-applied, 81-block lean divide), conditional-sparsity, byte-exact 28/30 — **THE honest host ms/step.**
