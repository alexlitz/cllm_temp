# JSR-target PC byte-1 delivery — root, staging, ON-ROW DELIVERY LANDED (2026-06-26)

> **UPDATE 2026-06-26 (flag `C4_JSR_PC_BYTE1`, default OFF — on-row delivery LANDED).**
> The full stage->relay->emit pipeline now flips the PC byte-1 of high-JSR-target
> step-0 programs at the byte-0 row. Verified (`spec_k=0` CPU autoregressive,
> `tools/_jsr_byte1_resid_probe.py`-style trace): **gcd id900 0x101,0xa,0x1**
> (PC 0x010a, idx 33) and **rec_fib id725 0x101,0x2,0x1** (PC 0x0102, idx 32) now
> emit the correct byte-1, while the HOLD gates **func_identity id550 (idx 7)** and
> **rec_factorial id700 (idx 25)** still emit byte-1=0. Flag-OFF (default) is
> WHOLE-state_dict SHA256 IDENTICAL to the base `8fc9a3d3`
> (`2d227d482689186d…`), zero production regression.
>
> Three things changed from the blueprint below (all in `ops/model_ops.py`):
> 1. **Staging threshold** lowered (`T_jsr_pc_stage=1.5`) — the override's 4.0 was
>    too high for gcd's anomalous step-0 marker `IS_JSR`(TEMP[0])~1.2 (most JSRs are
>    ~10.0), so the byte-1 stage never fired for gcd. 1.5 fires at MARK_PC+TEMP[0]
>    >=2.2 while still blocking non-JSR markers (TEMP[0]=0 -> 1.0<1.5).
> 2. **The relay head is baked INSIDE `make_function_call_weights_op`** on
>    `model.blocks[7]` — the physical block ONE PAST the override FFN block — NOT on
>    the L6 block itself. The marker stage lives in the L6 FFN, which runs AFTER the
>    L6 attention, so a same-block (L6) relay head reads `JSR_PC_B1=0`. (The L6
>    free-head zeroing in the old blueprint was a red herring; the head SURVIVED but
>    read the not-yet-staged band.)
> 3. **The emit is a post-tail-corruptor FFN** (`make_jsr_pc_byte1_emit_op`, bound
>    to `l10_post_ops_combined` `requires after tail_bit32_result_correction`,
>    mirroring `l11_ops.bp_save_dump_repopulate`). The L25 tail corruptor re-zeros
>    the PC byte-0 OUTPUT (and on rec_fib leaks a ~2e10 SENTINEL at OUTPUT_LO+2 —
>    the byte0 value bleeding into the byte1 prediction), so the emit HARD-SETS
>    OUTPUT_LO at the byte-0 row (+2*WS at the byte1 nibble, -WS elsewhere, WS=5e7)
>    to dominate it. Running the emit on the L6 block (before the corruptor) flipped
>    byte-1 transiently but the corruptor clobbered it back to 0.

The high-JSR-target deep-loop wall (gcd 900-949, rec_fib 725-749 step-0). A JSR
whose target instruction index is **>= 32** jumps to a function at
`PC = idx*8 + 2 >= 256`, so PC **byte-1** must be NONZERO. The model emits it as
`0`, so the very first JSR jumps to the wrong PC and the program diverges at
step 0.

This is the H1-onehot / AX-byte1-dump wall family (#311) applied to the PC: no
dedicated PC byte-1 delivery exists for the JSR-target path.

## Root (verified, spec_k=0 CPU autoregressive decode)

* The JSR PC override (`model_ops._function_call_jsr_pc_override_rules`, the L6
  FFN, IR path — the LIVE production path; `vm_step.py` ~8911 is the dead legacy
  twin) materialises PC byte-0 **and its high nibble** at the PC MARKER row, and
  explicitly **blocks `IS_BYTE` (-10)** so it NEVER fires at a PC byte position.
  The `jsr_pc_fetch_hi_reserved` block (16 units gating `FETCH_HI+k` with no
  down-write) is a documented dead no-op.
* So PC byte-1 is driven solely by the L3 register-default (= 0). gcd id900
  (idx=33 -> PC 266 = 0x010A, byte1=1) and rec_fib id725 (idx=32 -> 258 =
  0x0102, byte1=1) emit PC `0x0A` / `0x02` instead of `0x010A` / `0x0102`.
* Programs whose first JSR target idx < 32 are UNAFFECTED (byte1=0): func_identity
  550-574 (idx=7), rec_factorial 700-724 (idx=25), rec_power 775+ (idx=28). These
  are the HOLD gates.

### The byte-1 value and its source (verified at the PC MARKER, post-attention)

`PC byte1 = (idx*8+2) >> 8 = FETCH_HI_nibble >> 1` (the +2 and FETCH_LO*8 < 128
never carry into bit 8). At the PC marker the L6 FFN-input residual carries a
**clean** FETCH: for gcd (idx=33) `FETCH_HI` dominant index = 2 (mag ~40), for
func (idx=7) `FETCH_HI` dominant index = 0. So `byte1 = FETCH_HI_nibble >> 1`
(2>>1=1 for gcd, 0 for func). The byte-0 override already reads this same relayed
FETCH (L6 head-5 first_step_fetch_relay copies FETCH AX-marker -> PC-marker).

Probe tool: `tools/_jsr_byte1_resid_probe2.py <id>` (hooks block-6 FFN input,
dumps FETCH/marker dims at the marker + PC byte rows on CPU).

## Landed: the marker STAGE (flag `C4_JSR_PC_BYTE1`, DEFAULT OFF)

The dead `jsr_pc_fetch_hi_reserved` block is REPURPOSED, when the flag is on, to
STAGE `byte1 = (k>>1)` into a dedicated `JSR_PC_B1` residual band from
`FETCH_HI+k`, sharing the override's JSR-exclusive `TEMP[0] (IS_JSR) + opcode
blocker` gate. The stage is therefore nonzero ONLY on a real JSR-to-idx>=32 step.
This is verified to produce the right band one-hot.

* Flag OFF (default + the escape hatch) is **byte-identical to golden f2b040aa**
  (whole-`state_dict` SHA256 confirmed). The band is flag-gated
  (`register_residual_band(..., flag=_jsr_pc_byte1_enabled)`) so OFF -> no extra
  dim, smaller d_model, byte-identical model.

## Remaining blocker: the marker -> byte0 RELAY + emit (delivery)

The byte-1 value is staged at the MARKER but must be DELIVERED to the PC byte-0
token row (whose OUTPUT predicts byte-1 under the autoregressive shift). FETCH is
NOT present at the byte-0 row (verified), so an attention relay (Q at byte-0, K at
the same step's MARK_PC, V copies `JSR_PC_B1`) + an emit FFN are required —
exactly the AX-byte1-dump pattern (carry head -> dedicated band -> dump FFN).

The relay head needs a FREE attention head on the L6 attention block. **Blocker:**
a free padding head pinned on L6 (block 6) is **zeroed before the final model**,
even when declared via the `compiler_ir_factory` IR (allocator-pinned, mirroring
the working `l11_ops.bp_save_prev_carry` head 8 on L13). The same IR pattern's
free heads SURVIVE on other blocks (verified: blocks 7/10/17/19 keep heads 7/8),
so the zeroing is **L6-block-specific** — most likely the interaction with
`layer6_attn_bake` (the `kind="model"` op at phase 998.5 that owns the L6 attn
weights) and/or the L6 head allocator (`_allocate_layer6_heads`, `layer_max_heads
=8`). The WIP relay/emit ops (`make_jsr_pc_byte1_relay_head_op` /
`make_jsr_pc_byte1_emit_op`) are committed but inert at the default (flag off).

### Next-session delivery plan (pick one)

1. **L6 free-head fix** — get a free L6 head to survive: raise
   `_allocate_layer6_heads` `layer_max_heads` and have `layer6_attn_bake` declare
   the padding head, OR add the relay head INSIDE `layer6_attn_bake`'s own spec
   list (so it is part of the op the L6 layout is built from), OR host the relay
   on a NON-L6 block whose free heads survive (e.g. L7/block-7) and stage there.
2. **Piggyback head 5** — extend the existing `first_step_fetch_relay` (head 5)
   `branch_pc_byte0_relay` slot-52 path (already Q@byte0 / K@MARK_PC) to also
   V/O-copy `JSR_PC_B1` -> `JSR_PC_B1_AT_B0` (free slots 67+). Risk: shared-head
   softmax — gate via `lint_cross_op_attention`. First verify slot-52 actually
   lands the head's attention on the marker at the byte-0 row (the probe showed
   FETCH absent at byte0, i.e. it may NOT).
3. **Emit** — once the relayed band reaches the byte-0 row, a standalone PureFFN
   L6 post_op (mirror `bp_save_dump_repopulate`) cancels the L3 default-zero and
   writes byte1's low nibble. Verify no later layer (L7..L25) overwrites the PC
   byte-1 OUTPUT before the LM head (PC byte rows should be untouched by the ALU
   band; confirm via `cpu_full_trace` on gcd id900).

## Gates / verification

* Flag-OFF byte-identity: `CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py`
  -> `f2b040aa…` (default AND `C4_JSR_PC_BYTE1=0`). CONFIRMED.
* Once delivery lands: `tools/cpu_full_trace.py --ids 900,725 --spec-k 0
  --max-steps-cap 600` (gcd/rec_fib step-0 before->after), HOLD gates
  `--ids 550-574,700-724` (0 ok->fail), and the FINAL
  `run_1096_canonical --ids "900-949,700-724,550-574" --spec-k 0
  --criterion full_trace --max-steps-cap 600`.
* Unblocks gcd (~50) + rec_fib (~25) at step 0.
