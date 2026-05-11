# `set_active_opcode` Purity Analysis

**Author:** scoping agent
**Date:** 2026-05-11
**Branch:** `b1-active-opcode-purity`
**Status:** scoping only, no code modified

---

## TL;DR — Verdict

**The pure_neural path has already been migrated.** `set_active_opcode` is
skipped in pure_neural mode (FIX 2026-05-09 in `run_vm.py`), so the residual
violation only applies to the legacy / conversational-I/O modes. For
pure_neural the model already derives the opcode flags it needs from
`OPCODE_BYTE_LO`/`HI` via L5 FFN and L6 attention relays, without any
Python-side injection. The two open hand-offs that remain depend on
*future* phase work (Phase 4 BZ/BNZ and Phase 5 LEV), not on this migration.

A blanket deletion of `set_active_opcode` and `_inject_active_opcode` would
break:

1. **Legacy / handler mode** (`pure_neural=False`, `conversational_io=False`):
   MoE weight-swap for FFN sub-matrices (`compact_moe` path used by
   `fast_runner`, `batch_runner`, `transformer_first_runner`, benchmarks).
2. **Conversational-I/O mode** (`conversational_io=True`): L5 FFN +
   L6 attention head 4-5 depend on `ACTIVE_OPCODE_PRTF` /
   `ACTIVE_OPCODE_READ` being globally set at every position to override
   the BZ/BNZ `CONST=-65` Q-penalty.
3. **Compiled L6 attention head 6** (`unified_compiler/compiler.py:791,
   833`): explicitly disables OP_LEV relay because "injected globally
   via `_inject_active_opcode`". Removing the injection without also
   re-enabling the relay would leave OP_LEV permanently zero at the PC
   marker, breaking L9 LEV-BP-to-PC and L14 LEV TEMP-clearing.

So the recommendation is: **leave the call sites in place**, document the
purity status as "already neutralised for pure_neural", and treat full
deletion as a multi-phase project gated on Phase 4 (BZ/BNZ) + Phase 5 (LEV)
completing their neural-only pathways. This document records the design
and the open work.

---

## 1. Exact Injection State

### Source method
`c4_release/neural_vm/vm_step.py:1217-1237` — `AutoregressiveVM.set_active_opcode`.

```python
def set_active_opcode(self, opcode_value):
    self._active_opcode = opcode_value
    if opcode_value is None:
        dim = None
    else:
        dim = _opcode_dim_from_positions(self.dim_positions, opcode_value)
    for block in self.blocks:
        ffn = block.ffn
        if getattr(ffn, "_moe_combined", None) is not None:
            ffn._activate_moe(dim)
```

`set_active_opcode` does **two** things:

| # | Effect | Consumer |
|---|---|---|
| **A** | Stores `opcode_value` on `self._active_opcode`. The next `forward()` passes that into `NeuralVMEmbedding.forward(active_opcode=…)`, which calls `_inject_active_opcode` (`neural_embedding.py:349-371`) to set five residual dims at **all** positions: `ACTIVE_OPCODE_PRTF=5.0` (PRTF=33), `ACTIVE_OPCODE_READ=5.0` (READ=31), `OP_LEV=5.0` (LEV=8), `OP_BZ=5.0` (BZ=4), `OP_BNZ=5.0` (BNZ=5). | L5 routing FFN, L6 attention heads 4-5, L9 LEV BP→PC heads, L14 LEV TEMP clear. |
| **B** | If FFN has been MoE-partitioned (`compact_moe()` was called), swaps `W_up`/`W_gate`/`W_down`/`b_up`/`b_gate` to the active opcode's sub-matrices (`base_layers.py:266-…`, `_activate_moe(dim)`). Otherwise no-op. | The MoE-compacted FFN of every transformer block. |

Effect **A** is the purity violation (5 residual dims written from Python
state). Effect **B** is a runtime optimisation (no semantic change to
output, just sparser compute) but still relies on knowing the next opcode
externally.

### Five injected dims (per `_OPCODE_INJECTION_MAP` in `neural_embedding.py:362-368`)

| Opcode value | Name | BD dim | Used by |
|---|---|---|---|
| 33 (PRTF) | `ACTIVE_OPCODE_PRTF` | 504 | L5 FFN `_set_conversational_io_opcode_decode` (PRTF→IO_IS_PRTF), L6 head 4 Q-gate (`setup_helpers.py:1697`) |
| 31 (READ) | `ACTIVE_OPCODE_READ` | 505 | L5 FFN (`setup_helpers.py:1651`) — produces IO_IS_READ |
| 8 (LEV) | `OP_LEV` | (BD-allocated) | L6 head 6 (relay **disabled**, see `compiler.py:791,833`), L9 `_set_layer9_lev_addr_relay` (`vm_step.py:803`), L9 `_set_layer9_lev_bp_to_pc` (line 746), L14 TEMP clear (`vm_step.py:1335`) |
| 4 (BZ) | `OP_BZ` | (BD-allocated) | L6 head 4 Q-gate (`setup_helpers.py:1501`), L6 head 4 V (`compiler.py:721,726`) |
| 5 (BNZ) | `OP_BNZ` | (BD-allocated) | L6 head 4 Q-gate (`setup_helpers.py:1502`), L6 head 4 V (`compiler.py:722,727`) |

---

## 2. Runner-Side Trigger Sites

### Two call sites in `run_vm.py`

**Initial call** (`run_vm.py:327-334`, before the per-step loop):
```python
# FIX 2026-05-09: In pure_neural mode, do not inject the active opcode from
# Python — the network must determine the opcode from its own bytecode attention.
# Other modes still need the MoE/embedding-injection hint to function.
if not self.pure_neural:
    init_exec = self._exec_pc() // INSTR_WIDTH
    if 0 <= init_exec < len(bytecode):
        opcode = bytecode[init_exec] & 0xFF
        self.model.set_active_opcode(opcode)
```

**Per-step call** (`run_vm.py:854-860`, inside `_dispatch_step`):
```python
# FIX 2026-05-09: Skip set_active_opcode in pure_neural mode (Python peek at bytecode).
if not self.pure_neural:
    next_exec = self._exec_pc() // INSTR_WIDTH
    if 0 <= next_exec < len(bytecode):
        self.model.set_active_opcode(bytecode[next_exec] & 0xFF)
    else:
        self.model.set_active_opcode(None)
```

Both branches are **already gated by `if not self.pure_neural`** since the
2026-05-09 agent-C1.A fix. The pure_neural runner leaves `self._active_opcode`
at its initial value `None`. `NeuralVMEmbedding.forward` then short-circuits
`_inject_active_opcode` at line 219 of `neural_embedding.py`:

```python
if active_opcode is not None:
    self._inject_active_opcode(token_ids, x, active_opcode)
```

So in pure_neural mode the five residual dims (`ACTIVE_OPCODE_PRTF`,
`ACTIVE_OPCODE_READ`, `OP_LEV`, `OP_BZ`, `OP_BNZ`) are **never written from
Python**. The MoE weight-swap (effect B) is also skipped because pure_neural
runners do not call `compact_moe()`, so `_moe_combined is None` and the inner
loop is a no-op even if `set_active_opcode` were called.

---

## 3. How the Network Compensates in pure_neural

### Option B confirmation: the active opcode is already implicit in the bytecode-decoded `OP_*` flags

The unified compiler's L5 FFN (`unified_compiler/compiler.py:2024-2069`) bakes
opcode decoding at **MARK_AX**: for every opcode in the 33-entry table it
fires a unit when `OPCODE_BYTE_LO[lo] + OPCODE_BYTE_HI[hi]` are both ~1 and
`MARK_AX=1`, then writes `OP_xxx ≈ 5.0` via `W_down`. Additionally, lines
2071-2106 bake a **first-step** opcode decode at MARK_PC (gated by `NOT
HAS_SE`) for 18 of those opcodes (excluding BZ, BNZ, LEV, EXIT).

Then the L6 attention head 4 (`unified_compiler/compiler.py:715-740`) relays
`OP_BZ` and `OP_BNZ` from AX marker → PC marker on all steps. So for BZ and
BNZ, the network does not need `_inject_active_opcode` once the AX-marker
decode has fired.

There is **one gap** in the unified compiler relative to the hand-set
baseline (`vm_step.py:3122-3139`, the "All-step opcode decode at PC marker"
block, gated by MARK_PC alone, no HAS_SE). That block in the hand-set path
wrote `OP_BZ`, `OP_BNZ`, `OP_LEV`, `OP_EXIT`, `OP_JMP` at the PC marker on
**all** steps. The unified compiler omits it — its first-step decode covers
JMP and EXIT (PC marker, NOT HAS_SE) but **not** BZ, BNZ, LEV on non-first
steps. The L6 head-4 relay covers BZ/BNZ but the L6 head-6 OP_LEV relay is
explicitly **disabled** (`compiler.py:791,833`) with the comment "injected
globally via `_inject_active_opcode`".

So for pure_neural, OP_LEV at the PC marker is currently **not set on any
non-first-step LEV invocation**. This is why Phase 5 LEV is partially
blocked (`test_pure_neural_jsr_ent_lev.py::test_jsr_callee_writes_ax`
xfails). The L9 LEV-BP-to-PC head Q-gate (`vm_step.py:803`,
`W_q[base, BD.OP_LEV] = L / 5`) reads OP_LEV at PC marker — which is zero
in pure_neural.

The fix is not "remove `set_active_opcode`" — it is "complete the L5 / L6
bakes for OP_LEV". Specifically:

1. **Add an all-step PC-marker OP_LEV decode** to the unified compiler's L5
   FFN, mirroring `vm_step.py:3122-3139` for at least `OP_LEV`:
   ```python
   for op_val, lo, hi in [(Opcode.LEV, 8, 0)]:
       op_dim = BD.opcode_dim(op_val)
       ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
       ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
       ffn.W_up.data[unit, BD.MARK_PC] = S
       ffn.b_up.data[unit] = -S * 2.5
       ffn.b_gate.data[unit] = 1.0
       ffn.W_down.data[op_dim, unit] = 10.0 / S
       unit += 1
   ```
   This costs **1 FFN unit** and makes the OP_LEV flag autoregressive.
2. **Re-enable the L6 head-6 OP_LEV V[0]** in the unified compiler
   (`compiler.py:791` currently commented "disabled"). With the L5 decode
   in place, the L6 amplification (×2) becomes safe again because the
   pre-L5 OP_LEV at PC is zero (no double-counting from
   `_inject_active_opcode`).
3. Same pattern for `OP_BZ`/`OP_BNZ` at the PC marker on non-first steps
   (currently the L6 head-4 relay covers them, but only via attention
   averaging — making the L5 decode all-step would harden the signal).

This is Phase 4 / Phase 5 scope, not a pure-deletion refactor.

---

## 4. Why the Other Three Injected Dims Cannot Be Bake-Migrated Without More Work

### `ACTIVE_OPCODE_PRTF` and `ACTIVE_OPCODE_READ`
Used only in `enable_conversational_io=True` runs. They appear in L5 FFN
units 410-411 (`_set_conversational_io_opcode_decode`, `setup_helpers.py:
1643-1654`) where they gate the IO_IS_PRTF / IO_IS_READ down-projection,
and in L6 head 4 Q-gate (`setup_helpers.py:1697`,
`W_q[base, BD.ACTIVE_OPCODE_PRTF] = L * 1.5`) where they cancel a -65
BZ/BNZ-CONST penalty so the PRTF relay can fire at the SE position.

To remove the injection, the L5 FFN would need to decode PRTF/READ from
`OPCODE_BYTE_LO`/`HI` **at every position the embedding currently broadcasts
to**. The injection is global (every position); a baked replacement needs
either (a) the L5 main opcode decoder (which fires at MARK_AX) plus an
attention head that broadcasts that flag to every position, or (b) the
network can route PRTF/READ via the existing OP_PUTCHAR/OP_GETCHAR
mechanism.

Since pure_neural does **not** enable conversational_io
(`test_pure_neural_io.py` uses tool-call mode, not conversational),
these two dims are dead in the pure_neural path. They only matter for the
mixed handler+conversational mode, which is a separate migration scope.

### `OP_LEV` / `OP_BZ` / `OP_BNZ`
Covered in §3 above. The L6 attention relay heads already do most of this
work — the missing piece is one L5 FFN unit per opcode at PC marker
(all-step). Adding those three units (or 5, including EXIT and JMP for
parity) closes the gap.

---

## 5. Why Outright Deletion Is Unsafe Today

If we delete `set_active_opcode` and `_inject_active_opcode` now, the
following breaks **without** pure_neural mode involvement:

1. **Legacy `pure_neural=False` runs that use `compact_moe()`** for speed:
   `fast_runner.py:50`, `batch_runner.py:159`, `batch_runner_v2.py:64,261`,
   `transformer_first_runner.py:60`, `src/transformer_vm.py:301`,
   `tools/benchmark_vm.py:37,101,128`, `tools/demo_speedup.py:107`,
   `tools/rebuild_and_test.py:23`. The MoE path requires `set_active_opcode`
   to swap the active expert sub-matrices each step. Deletion means the
   compacted FFN runs against an arbitrary (last-active) opcode's expert,
   producing wrong outputs.
2. **Conversational I/O mode** (`enable_conversational_io=True`): L5
   `_set_conversational_io_opcode_decode` + L6 head 4/5 (
   `_set_conversational_io_relay_heads`) require
   `ACTIVE_OPCODE_PRTF`/`ACTIVE_OPCODE_READ` ≥ 1 at the AX / SE positions.
   Without injection, IO_IS_PRTF stays zero and conversational PRTF never
   triggers THINKING_END.
3. **Unified compiler L6 head 6 OP_LEV path**: explicitly relies on
   `_inject_active_opcode` to globally set OP_LEV (`compiler.py:791,833`
   comments). Removing the injection without the §3 L5 PC-marker decode
   makes OP_LEV zero everywhere outside of MARK_AX, breaking the L9
   LEV-BP-to-PC head and L14 TEMP clear.

---

## 6. Test Confirmation

Baseline on `b1-active-opcode-purity` branch (no code changes):

| Suite | Result |
|---|---|
| `tests/test_pure_neural_pc.py` | 12 passed / 1 failed (`test_imm_byte_values[255]` — pre-existing high-bit IMM issue, unrelated to active-opcode) |
| `tests/test_smoke_pure_neural.py::TestSmokePureNeuralBasic::test_imm_exit` | 1 passed |

This confirms pure_neural is functional at the current Phase 1 (IMM/EXIT)
scope **without** any `set_active_opcode` calls firing (they are gated off
at both `run_vm.py:330` and `run_vm.py:855`).

---

## 7. Recommendation

**Defer full deletion.** Track three follow-ups:

1. **Phase 5 LEV completion**: add the all-step PC-marker `OP_LEV` decode
   to the unified compiler's L5 FFN (1 unit) and re-enable the L6 head-6
   V[0] relay. Verify `test_pure_neural_jsr_ent_lev.py::test_jsr_callee_
   writes_ax` flips from xfail to pass.
2. **Phase 4 BZ/BNZ hardening (optional)**: add all-step PC-marker decode
   for BZ/BNZ (2 units). The L6 head-4 relay already covers them; this
   would just harden the signal and remove the OP_BZ/OP_BNZ entries from
   `_OPCODE_INJECTION_MAP`.
3. **Conversational-I/O migration** (separate scope, currently
   handler-mode only): replace `ACTIVE_OPCODE_PRTF`/`READ` injection with
   a baked broadcast head — out of scope for pure_neural purity.

After (1)+(2) land, `_OPCODE_INJECTION_MAP` collapses to just the two
conversational-I/O entries, and `set_active_opcode` can be reduced to
*only* the MoE weight-swap (effect B), which is no longer a purity
violation (it doesn't write residual stream, it swaps a Python attribute
that points at pre-computed expert sub-matrices). Pure_neural runs
already don't call it.

**No code changes recommended at this scoping pass.**

---

## Appendix A — Files touched in analysis (read-only)

- `c4_release/neural_vm/vm_step.py` (set_active_opcode, all-step L5 PC decode)
- `c4_release/neural_vm/run_vm.py` (call sites, pure_neural gating)
- `c4_release/neural_vm/neural_embedding.py` (_inject_active_opcode, forward)
- `c4_release/neural_vm/setup_helpers.py` (PRTF/READ L5 FFN + L6 heads)
- `c4_release/neural_vm/unified_compiler/compiler.py` (L5 FFN opcode decode,
  L6 head 4/6 relay disable comments)
- `c4_release/neural_vm/base_layers.py` (_activate_moe)
- `c4_release/tests/test_pure_neural_pc.py` (Phase 1 12/13 baseline)
- `c4_release/tests/test_smoke_pure_neural.py` (smoke baseline)
