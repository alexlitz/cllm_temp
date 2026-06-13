# AX byte-1 register-dump cross-step carry — LANDED (gated) (2026-06-13)

The dedicated-band re-architecture of the AX byte-1 register-dump carry. The
full machinery is landed and byte-identical by default; the LM-head EMISSION is
behind `C4_AX_BYTE1_DUMP` (default OFF) pending one last gate-precision fix.

## Results (spec_k=0, GPU 0, canonical runner = efficient ALU)

| build | add 0-49 full_trace | sub 50-99 full_trace | smoke |
|-------|---------------------|----------------------|-------|
| baseline (HEAD c3e73ed1) | 12/50 | 5/50 | 49/2 |
| `C4_AX_BYTE1_DUMP=0` (default) | 12/50 (byte-identical) | 5/50 | **49/2** |
| `C4_AX_BYTE1_DUMP=1` | **40/50** (+28) | **9/50** (+4) | 47/4 |

id-0 (`654+114`) passes at `C4_AX_BYTE1_DUMP=1`: the PSH-step byte-1 dump
re-emits `0x02` (654) instead of dropping it (142).

## Architecture (all landed)

1. **Dedicated bands** `H1_PREV_STEP` (7) + `H1_DUMP_OUT` (7), in the
   production-default `extra_residual_dims` of `compile_full_vm_dynamic`. The
   14-dim widen rounds head-dim-preservingly `d_model 872 -> 981` (`n_heads
   8 -> 9`). Threaded by NAME via `layout.dim_positions`; forward-declared as
   `compiler.pending_extra_dims` before the `add_op` loop (so the carry ops
   validate) and bump-allocated at the tail AFTER it (so the base head_dim 109
   is captured clean). Both bands are in the dim-liveness `_LIVENESS_NEVER_SHARE`
   set — a multi-cell same-width liveness merge left donor residue in
   `H1_DUMP_OUT` (read a stale 1.0 instead of the carried one-hot).

2. **Carry head** (`layer13_ax_byte1_dump_carry`, L13 block-16 head 6,
   `enable=True`): reads the PREVIOUS step's `H1` one-hot via `H1.*.-1` (SSA
   cross-step, allowlisted in `CROSS_STEP_DOCUMENTED_SAFE`) and writes the
   distinct `H1_PREV_STEP` band UNCONDITIONALLY -> breaks the H1-write 2-cycle.
   K-selection (the key reliability fix): a SHARP signature on slot 0
   (`ADDR_B0_LO+5`, weight 60) confines attention to byte-1 predictor rows
   among the prev step; `ADDR_B1_HI+8` on slot 1 (AX-register, ~4.0 AX / ~0
   PC/SP/BP, program-stable) rejects other registers; the AX_CARRY differential
   on slot 2 (its OWN slot — NOT mixed into the signature slot, which spread
   attention across prev byte 0..3) picks the FRESH prev row over the current
   carried one. Verified: H1_PREV_STEP holds the correct `H1+(v+2)` one-hot for
   v in 0..4 across 654/754/913/432/228.

3. **Dump FFN** (`ax_byte1_dump_repopulate`, L25 tail post_op after
   `tail_bit32_result_correction`): gate-copies `H1_PREV_STEP -> H1_DUMP_OUT`
   on a carried byte-1 row. MIXED dim_map — the legacy gate dims
   (`ADDR_B0_LO`, `ADDR_B1_HI`, `AX_CARRY`, `MARK_*`) resolve from the dynamic
   *registry* (`build_default_registry_dynamic()`) because the model residual
   carries them at registry positions (the declarative `dim_positions` layout
   differs: `ADDR_B0_LO=506` vs registry `12`); the NEW bands resolve from
   `dim_positions`. Gate (balanced AND): `ADDR_B1_HI+8*0.25 + ADDR_B0_LO+5*2 +
   AX_CARRY` over threshold 4.5 — fires on a genuine CARRY (AX_CARRY ~+2.7),
   dark on fresh IMM/ADD (~-988), on PC/SP/BP (no AX-register signal), on AX
   byte-2/3 (no signature), and on memory LOADs (AX_CARRY ~+0.8).

4. **Head bake** (`ax_byte1_dump_head_bake`, phase 1002): mirrors the H1
   high-byte emission columns onto `H1_DUMP_OUT`
   (`head.weight[v, H1_DUMP_OUT+(v+2)]=5.0`, v in 0..4). Additive; gated by
   `C4_AX_BYTE1_DUMP` (in both disk- and in-proc cache keys). OFF -> no columns
   -> byte-identical.

## The remaining last mile (why default OFF)

The dump-read gate over-fires on TWO step classes whose AX_CARRY sits OUTSIDE
the `+2.7` carry band: SHL-result (AX_CARRY ~ +12.8) and JMP (~ +47.9). A
linear AND threshold cannot band-pass them out. The CLEAN discriminator is the
CURRENT row's own `H1` high-byte one-hot (~12 on a fresh compute/write, ~0 on a
carry) — a blocker on `H1+4/5/6`. It fixes smoke to 49/2 in the lookup path BUT
breaks the efficient-ALU canonical decode (pc=None across the corpus) for a
reason not yet isolated (the read-side `H1+4/5/6` blocker should be inert on
non-AX-byte rows). That single incompatibility is the only thing between this
and a default-ON 49/2 + (+28 add / +4 sub) landing.

Tool: `tools/probe_h1prev_carry.py` (carry-band + AX_CARRY gate at the final
block; must build with `alu_mode='efficient'` to match the canonical runner).
