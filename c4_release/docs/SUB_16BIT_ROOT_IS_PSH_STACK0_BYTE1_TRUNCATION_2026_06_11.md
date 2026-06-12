# sub_16bit / 16-bit MUL / var multi-byte root: PSH truncates STACK0 high bytes

**Date:** 2026-06-11 (spec_k=0, hook-free, GPU 1). **Status:** ROOT IDENTIFIED,
fix is a multi-session upstream arch-block. NO code change landed (correct
zero-regression outcome). Supersedes the L19/block-30/dim-98 localization in
the brief and in `project_sub_16bit_discriminator_destroyed_at_l19`.

## TL;DR

`test_sub_16bit` (`IMM 0x100; PSH; IMM 1; SUB`, wants `0x000000FF`) emits
`0xFFFFFFFF`. The brief said the byte-1 discriminator "peaks at block 30 / L19
dim 98 and is erased by block 31," and asked to preserve it through to the L25
tail. **That premise is false.** Ground-truth diffing of `sub_16bit` vs
`sub_borrow_cascade` (`IMM 0; PSH; IMM 1; SUB`, wants `0xFFFFFFFF`, PASSES)
shows the two programs are **byte-identical across the ENTIRE SUB step (rows
150-159) at ALL 37 blocks**, except a ~0.6 noise residue on `OUTPUT_HI+13/14`
at block 31 that is swamped by an **identical** `OUTPUT_LO+15 = +1005`
sign-extension write. There is literally NO separable signal anywhere in the
SUB-step residual. A preservation dim has nothing to preserve.

The real root is **6 logical layers UPSTREAM of where the brief looked**: the
PSH of `0x100` stores only byte 0 into the STACK0 byte band. `STACK0_BYTE1`
(dim 710) is **0 at every SUB-step row** even though the pushed value is
`0x100` (byte 1 = `0x01`). The minuend's high byte is gone before the SUB even
runs, so the model computes the SAME wrong answer for `0x100-1` and `0-1`.

## The evidence chain (all spec_k=0, `tools/probe_sub_byte1.py` + `residual_at`)

1. **Context diff:** the two free-running greedy decodes differ at only 2 token
   positions (3 and 51 — the IMM operand high byte). Both emit AX bytes
   `255,255,255,255`. So `sub_16bit` genuinely produces `0xFFFFFFFF` at decode.

2. **Whole-SUB-step diff:** over rows 150-159 × all 37 blocks × 872 dims, the
   only divergence > 0.1 is `OUTPUT_HI+13/14 ≈ 0.6` at block 31. The
   load-bearing byte-1 signal `OUTPUT_LO+15 = +1005` / `OUTPUT_LO+0 = -1002`
   is **byte-identical** between the two cases. The "dim 98 discriminator" is
   this 0.6 noise, NOT a usable signal.

3. **The byte-1 = 0xFF write is at block 18 = logical L14**, not L19. Tracing
   `OUTPUT_LO+15` at the predict-byte-1 row (156): flat ≈ 0 through block 17,
   then **block 18 writes +1005.6** (delta), stable thereafter. The op is
   `CarryPropagationPostOp(byte_idx=0, cascade=False)` (`vm_step.py:855`,
   hidden=512), SUB-borrow loop (lines 975-1022), units ~256-270 (15 units
   each contributing +53.6 to `OUTPUT_LO+15`, summing to +1010).

4. **The borrow loop math is CORRECT on its input.** It reads the relayed
   minuend byte (`OUTPUT_LO+lo` / `OUTPUT_HI+hi`) and writes
   `new_val = (lo + hi*16 - 1) & 0xFF`. Going INTO block 18 at the byte-1 row,
   BOTH programs have `OUTPUT_LO+0 = 3.44, OUTPUT_HI+0 = 3.44` → relayed
   minuend byte 1 = **0x00 for both** → `(0x00 - borrow) = 0xFF` for both.
   `CARRY+2 = 2.0` (SUB borrow) for both. The op does the right thing on
   truncated input.

5. **The truncation is at PSH store.** After L10 (block 11), `STACK0_BYTE1`
   (dim 710) is `None`/≈0 at every SUB-step row, while `STACK0_BYTE0` (dim 181)
   carries a one-hot. At the IMM/PSH step (rows 50-53) the byte-1 = 0x01 DOES
   exist (`OUTPUT_LO+1 = 12`, `CLEAN_EMBED_LO+1 = 1.0`, `ADDR_B1_LO+1 = 1.0`),
   but it is never written into `STACK0_BYTE1`. So PSH stores byte 0 only; the
   `layer10_nonbitwise_stack0_byte_relay_head` (head 5, `l10_ops.py:2160`,
   slots 31/32/34 keyed on `STACK0_BYTE1/2/3`) correctly relays an empty band.

## Why the brief's fix shape cannot work

- **No tail rule can separate the cases** (confirmed in
  `project_sub_16bit_discriminator_destroyed_at_l19`): the tail residual is
  byte-identical. The L25 tail bank is width-sensitive (must stay 2059) and
  was NOT touched.
- **No preservation dim at L14/L19/L25** helps: there is no signal in that
  whole region to copy into it. A "no-op preservation dim gated until wired"
  would be dead weight with no validated wiring target, and any new dim risks
  the `d_model 920→952` expansion that regresses `test_bnz_branch`
  (`project_mul_div_mod_arch_blocked`, agent a4cf5208). So no partial dim was
  landed — that would be unsafe scaffolding, per
  `feedback_single_rule_fixes_are_zero_sum`.

## The real fix (multi-session, upstream)

Make PSH store the full multi-byte value into the STACK0 byte band so
`STACK0_BYTE1/2/3` carry bytes 1-3 of the pushed value, OR equivalently make
the L10 stack0 relay recover them from the stored MEM row. Concretely:

1. **PSH store path** — find where PSH writes `STACK0_BYTE0` (the store that
   sets dim 181 to the pushed byte). Grep: `STACK0_BYTE0` writers in
   `l3_ops.py` / `l15_ops.py` / `vm_step.py` (`layer15_store_stack0_sp_byte0_addr`
   is a candidate — it already names byte0). Extend it to also write
   `STACK0_BYTE1/2/3` (dims 710/711/712) from the pushed value's bytes 1-3
   (available in `CLEAN_EMBED`/`OUTPUT` of the PSH operand row, or from the
   memory row the value is stored to).
2. **Verify the relay reads it** — `_layer10_nonbitwise_stack0_byte_relay_head`
   (head 5) slots 31/32/34 already key on `STACK0_BYTE1/2/3`; confirm its
   attention selects the PSH'd value's stored row once those dims are
   populated, so ALU_LO/HI byte 1-3 carry the minuend high bytes.
3. **Then the existing L14 `CarryPropagationPostOp` borrow loop is correct**:
   with `OUTPUT_LO+1 = 0x01` relayed, `(0x01 - borrow) = 0x00` → byte 1 = 0x00
   for sub_16bit, while sub_borrow's `STACK0_BYTE1 = 0x00` still gives
   `(0x00 - borrow) = 0xFF`. The two cases become genuinely separable at the
   borrow loop's INPUT — no tail rule needed.
4. **Byte-identity gate + smoke**: PSH store is load-bearing for every stack
   op (sub_basic, add_*, cmp, all 32-bit). Gate any change with
   `compare_symbolic_to_lowered_*`, run `tools/run_full_smoke.py` (spec_k=0,
   GPU 1), and watch sub_basic / sub_borrow_cascade / add_* / si_li_* / cmp.

## Cross-test convergence

This same STACK0 high-byte truncation is the documented root behind:
- **16-bit MUL** (`project_mul_div_mod_arch_blocked`): width=2 product is
  byte-identity correct at install but the operand high bytes are truncated
  upstream too; the L15/L20 "corruptor" they saw is the same multi-byte-result
  emit on truncated operands.
- **var/nested 1096 cluster** (`project_ax_bytes_1_3_ff_leak_root`): step-3 PSH
  high-byte leak — same PSH store band.

So PSH multi-byte store is the single highest-leverage upstream fix. It is a
Wall-2 load-bearing surface (`project_operand_gather_hybrid_encoding_is_cmp_alu_root`);
expect to gate heavily and iterate.

## State left

- NO code change. Tree clean. Tail rule count = 2059 (verified, untouched).
  dim-alias unchanged (no new dims). Smoke baseline reconfirmed: **35 pass /
  6 fail / 8-of-8 guardrails**; sub_16bit FAIL (got 4294967295), sub_basic +
  sub_borrow_cascade PASS.
- Probe: `tools/probe_sub_byte1.py` (already on disk). The decisive new
  measurement is the STACK0_BYTE1=0 finding (step 5 above), reproduce with
  `residual_at(..., stop_after_block=11)` reading dim 710 at the SUB-step rows.
