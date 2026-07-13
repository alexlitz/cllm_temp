# L11 OUTPUT byte-0 leak ROOT + `ShiftOutputClearFFN` deletion

**Date:** 2026-07-13
**Branch:** `output-byte0-leak-root` (off `main`)
**Flag:** `C4_OUTPUT_B0_NOLEAK` (DEFAULT-ON; opt out `=0`)
**Golden flag-OFF (`C4_OUTPUT_B0_NOLEAK=0`):** `e50521f3` — verified UNCHANGED.
**Golden default (flag-ON):** `8135b989` — intended change (+1 FFN unit; mirrors
the `C4_CLEAN_EMITTER` default-ON golden change).
**Corrector removed:** `ShiftOutputClearFFN` (roadmap item #14, Class B).
**Mission:** eliminate the `ShiftOutputClearFFN` corrector by deriving the L11
OUTPUT-byte-0 emission so it does NOT leak a stale `0x00` default onto the shift
compute row — the L11 / block-16 OUTPUT-band leak (the shared var/if/bool framing
mega-root).

## The derived root (spec_k=0, campaign `C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`)

Localized with `tools/probe_shr_output_b0_leak.py`, `probe_shr_blk17_emitter.py`,
`probe_shr_verify_and_isolate.py`, `probe_blk17_attn_heads.py`,
`probe_shr_vs_shl_discriminator.py`, and `probe_output_b0_noleak_verify.py`.

- `test_shr` (`84 >> 1 == 42`) decodes **`0x00`** with the corrector OFF; the
  residual delta on the SHR MARK_AX compute row at the L11 block changes **exactly
  two dims**: `OUTPUT_LO+0: 0 -> +2.0` and `OUTPUT_HI+0: 0 -> +2.0`.
- The leak is planted at **physical block 17 = logical L11** (the
  `layer11_mul_partial` block). Sub-block trace: the `+2.0` is written by L11's
  **attention** (the register-byte emission head `V=CLEAN_EMBED → O=OUTPUT_LO/HI`),
  not the L11 FFN — the shift ALU-compute MARK_AX row ALSO carries the frame's
  `IS_BYTE` + `BYTE_INDEX_3` tags, so the emission head spuriously fires and plants
  the `0x00` zero-default one-hot (magnitude `+2.0`).
- **SHR-vs-SHL discriminator:** the block-16 residual on the shift MARK_AX row is
  **byte-identical between the leaking SHR row and the clean SHL row EXCEPT the
  `OP_SHR` flag**. So the emission leaks the default on `shr` but NOT on `shl`
  (whose OUTPUT band stays empty all the way to the composite). This is why `shl`
  already passes and only `shr` needed the corrector.
- Downstream, the L17 `ALUShiftComposite` (`GEToBDConverter`) **ADDS** its result
  one-hot (`OUTPUT_LO+10 / OUTPUT_HI+2 = 0x2A`, magnitude `+2.0`), so the stale
  `+2.0` at `+0` **TIES** the true `0x2A` → the LM-head argmax breaks the tie
  toward the lower nibble index (cell-0) → both bytes decode `0x00` → result `0`.

The old `ShiftOutputClearFFN` (was `efficient_alu_neural.py:2936`) patched this
consumer-side: it multiplicatively zeroed the whole OUTPUT band on the
OP_SHL/OP_SHR + MARK_AX row at the L17 composite block, so the composite's result
stood unopposed.

## The fix (`C4_OUTPUT_B0_NOLEAK`, DEFAULT-ON)

A **1-unit FFN** (`make_output_b0_noleak_op`,
`neural_vm/unified_compiler/ops/l11_ops.py`) fires on the `MARK_AX + OP_SHR`
compute row (SHR ONLY) and writes `-0.8/S` to `OUTPUT_LO+0` / `OUTPUT_HI+0`. Under
the saturated-AND-fire gain this lands a residual delta that cancels the leaked
`+2.0` emission-head one-hot to **exactly 0.0** at the shift-composite block input
(measured spec_k=0: band = `-0.0000`), i.e. the SHL-clean empty-band state. The op
is a post-op on the L11 `layer11_mul_partial` block (co-located at L11 via
`target_op_name="_layer11_ffn_dep_anchor"`), so the OUTPUT band is empty on the
SHR row by the time the L17 composite runs — exactly like `shl`.

Two design points, both load-bearing:

1. **Cancel to exactly ZERO** (not a large negative). A large-negative kill would
   leave the band `!= 0` at the composite input, so the deleted corrector's
   multiplicative-zero (`band * 0`) would NOT equal `inner` (`band`). Cancelling
   to `0` makes `forward == inner` (the inertness that unblocks the delete).
2. **SHR ONLY, not SHL.** SHL's band is already clean (never leaks). An OP_SHL
   cancel corrupts the multi-step **8-bit** SHL decomposition
   (`test_shl_8bit` 256 → 273, a `+0x11` residue). Scoping to OP_SHR fixes SHR
   and leaves every SHL path untouched.

The `0.8` numerator is the leak-cancel amplitude at the AND unit's operating point
(the balanced-silu gain), verified by `probe_output_b0_noleak_verify.py` (band
lands at 0, clear inert) — it is the measured cancel of the measured `+2.0` leak,
not an arbitrary per-op constant.

### Scheduling bug fixed (why the earlier WIP baked nothing)

The prior WIP declared `requires={"after": "layer11_mul_partial"}`. In the layer
compiler a `requires["after"]` forces a **strictly LATER layer** (`ref_layer+1`),
which evicted the op off L11 → its bake never ran (the leak survived unchanged).
The fix co-locates at L11 purely via `target_op_name="_layer11_ffn_dep_anchor"`
(the exact pattern `layer11_step_end_operand_relay` uses); the post_op append runs
after `block.ffn` (the mul_partial leak) automatically, so no `requires` is needed.

### Flag plumbing

- `ops/shared.output_b0_noleak_enabled()` — reads `C4_OUTPUT_B0_NOLEAK`
  (DEFAULT-ON); the op only bakes weights under
  `no_stack0_emit_enabled() and output_b0_noleak_enabled()` (so a bare
  non-campaign build with `C4_NO_STACK0_EMIT=0` is byte-identical regardless).
- `full_vm_compiler_dynamic.py` — `C4_OUTPUT_B0_NOLEAK` registered in BOTH
  cache-key snapshots (in-proc memo + disk-cache) so ON/OFF never share an entry.

## Inertness proof + `ShiftOutputClearFFN` DELETED

`tools/probe_output_b0_noleak_verify.py` (spec_k=0, campaign, corrector STILL
present) feeds the ShiftOutputClear wrapper the REAL block-input residual and
compares `wrap.forward(x)` (clear-then-composite) vs `wrap.inner(x)`
(composite-only):

| build | `ShiftOutputClear.forward` vs `.inner` max-abs-diff | verdict |
|---|---|---|
| root OFF (`C4_OUTPUT_B0_NOLEAK=0`) | **2.0** | clear is ACTIVE (removes the leak) |
| root ON (`C4_OUTPUT_B0_NOLEAK=1`)  | **2.38e-07** (≈ 0, fp noise) | **INERT → deletable** |

With the root ON the OUTPUT band is already empty on the SHR row at the composite
input, so the multiplicative-zero clear has nothing to remove → `forward == inner`
→ the wrap is a TRUE no-op and is **DELETED**:

- `neural_vm/efficient_alu_neural.py` — the `ShiftOutputClearFFN` class removed
  (−85 LOC, replaced by a short provenance comment).
- `neural_vm/unified_compiler/ops/alu_ops.py` — the `l13_alu_shift_install` bake
  no longer wraps the composite in `ShiftOutputClearFFN` (now `block.ffn =
  builder.composite`); the install glue + the `no_stack0_emit / shift_output_..`
  imports removed (−38 net).
- `neural_vm/verification/faithful_interpreter.py` — dropped the corrector from
  `COMPOSITE_ALU_FFN`.
- `neural_vm/unified_compiler/ops/shared.py` — `shift_output_byte0_clear_enabled()`
  marked DEAD (nothing on the build path reads it; retained only so its cache-key
  entry resolves — follow-up dead-flag sweep can drop it + its two cache-key
  entries in `full_vm_compiler_dynamic.py`).

## Other OUTPUT-band correctors the root touches

None became inert as a *side effect* of this specific SHR cancel — the fix is a
single-cell (`OUTPUT_{LO,HI}+0`), single-opcode (OP_SHR) cancel on the L11
shift-compute row, so it only subsumes the `ShiftOutputClearFFN`'s SHR job. It
does **confirm** the shared framing mega-root mechanism (a spurious
`IS_BYTE+BYTE_INDEX` emission-head fire planting an OUTPUT-band `0x00` default on a
compute row) that also gates the var/if/bool `!=STEP_TOKENS` drift — a follow-up
could generalise the same L11-source cancel to those rows, but that is a separate
(verdict-moving) mission, out of scope here.

## Gates

| gate | result |
|---|---|
| golden flag-OFF (`C4_OUTPUT_B0_NOLEAK=0`, `_isa_golden_hash.py`) | `e50521f3` — **== golden** (escape hatch reverts) |
| golden default (flag-ON) | `8135b989` — intended (+1 FFN unit; corrector gone) |
| inertness (`probe_output_b0_noleak_verify.py`, spec_k=0) | root ON → clear diff **2.38e-07 → INERT** (root OFF → 2.0) |
| smoke shift (`test_shl/shr/shl_8bit/shr_8bit`, spec_k=0, default = fix ON + corrector deleted) | **4/4 PASS** |
| smoke FULL (`pytest tests/test_smoke.py`, spec_k=0, GPU) | **51 passed, 1 deselected** (== golden 51/51) |
| corrector-deleted + fix-OFF control (regression check) | `test_shr` FAILS (0≠42) — confirms the fix, not the corrector, now carries SHR |
| build (CPU, `disk_cache=False`) | clean; `ShiftOutputClearFFN` absent from model + module |
| fast-gate `--base main` (L11-interacting clusters) | see below |

### fast-gate

`tools/fast_gate.py --base main --clusters
edge_pow,mul,div,mod,var_simple,var_mul,expr_add_mul,func_mul` (main =
corrector-present/no-root; HEAD = corrector-deleted/root-ON), 65 ids × 8 clusters
on GPU 0:

```
OFF (base main, corrector present): pass=52/65
ON  (HEAD, corrector deleted+root): pass=52/65
REGRESSIONS 0   GAINS 0   NET = +0   (0 per-id verdict differences)
MEM-SMOKE clusters (var_*): clean
```

**NET = +0 (no regression).** The `func_mul` cluster (LEA-heavy function calls) is
byte-identical base↔HEAD, confirming the added L11 passthrough block does NOT
shift any LEA-/framing-relevant runtime position (the model-op hardcoded
`blocks[15]`/`blocks[16]` bakes run on the PRE-expansion logical block list, so
the mid-stack L11 post_op is invisible to them).

### note on the added physical block

Unlike the deleted `ShiftOutputClearFFN` (a same-block `block.ffn` wrapper), the
root op is a `block.post_ops` entry on L11 → after `_expand_wrapper_blocks` it
becomes one extra passthrough block (n_blocks 59 → 60 flag-ON). The fast-gate +
full smoke (incl. `lea_basic`, func, control-flow) prove this is LEA-/framing-safe
(the absolute-position LEA contract is about pre-expansion model-op indices +
sequence positions, both unchanged). A future tidy could fold the 1 unit into the
L11 mul_partial FFN (via `C4_DISABLE_WRAPPER_EXPANSION`'s Sequential merge or the
allocator) to keep the block count, but it is not required for correctness.

## LOC delta

Code only (`git diff --shortstat main`, excl. docs): the corrector class
(−85) + install glue (−38 net) − the 1-unit L11 root op (+~90 incl. docstrings) +
provenance comments. Net corrector-debt removed ≈ **the full `ShiftOutputClearFFN`
77-LOC class + its install glue**; the L11 root op is a load-bearing computed
cancel (not corrector debt). The consumer-side patch is gone; the leak is fixed at
its source.
